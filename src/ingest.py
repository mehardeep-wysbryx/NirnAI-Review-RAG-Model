"""
Ingestion pipeline for NirnAI RAG Review.
Loads precedent JSONs, chunks them, and stores in ChromaDB.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from .embeddings import EmbeddingsProvider, get_embeddings_provider


@dataclass
class PrecedentChunk:
    """A single chunk from a precedent case."""
    case_id: str
    chunk_type: str
    text: str
    metadata: Dict[str, Any]


class PrecedentStore:
    """
    Vector store for precedent cases using ChromaDB.
    """
    
    CHUNK_TYPES = [
        "fingerprint",
        "key_fields", 
        "ec_summary",
        "review_notes",
        "exceptions",
        "flow_summary",
    ]
    
    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "precedents",
        embeddings_provider: Optional[EmbeddingsProvider] = None,
    ):
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError(
                "chromadb is required. Install with: pip install chromadb"
            )
        
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Initialize embeddings provider
        self.embeddings_provider = embeddings_provider or get_embeddings_provider("sentence_transformer")
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "NirnAI precedent case chunks"}
        )
    
    def chunk_precedent(self, precedent: Dict) -> List[PrecedentChunk]:
        """
        Break a precedent into semantic chunks for embedding.
        
        Expected precedent schema:
        {
            "case_id": "...",
            "meta": {"state": "...", "district": "...", "sro": "..."},
            "fingerprint": "...",
            "key_fields": {...},
            "ec_summary": "...",
            "review_notes": "...",
            "exceptions": "...",
            "flow_summary": "..." (optional)
        }
        """
        case_id = precedent.get("case_id", "unknown")
        meta = precedent.get("meta", {})
        key_fields = precedent.get("key_fields", {})
        
        # Base metadata for all chunks
        base_metadata = {
            "case_id": case_id,
            "state": meta.get("state", ""),
            "district": meta.get("district", ""),
            "sro": meta.get("sro", ""),
            "survey_no": key_fields.get("survey_no", ""),
            "extent": str(key_fields.get("extent", "")),
            "deed_types": ",".join(key_fields.get("deed_types", [])),
        }
        
        chunks = []
        
        # 1. Fingerprint chunk
        fingerprint = precedent.get("fingerprint", "")
        if fingerprint:
            chunks.append(PrecedentChunk(
                case_id=case_id,
                chunk_type="fingerprint",
                text=fingerprint,
                metadata={**base_metadata, "chunk_type": "fingerprint"}
            ))
        
        # 2. Key fields chunk
        if key_fields:
            key_fields_text = self._format_key_fields(key_fields)
            chunks.append(PrecedentChunk(
                case_id=case_id,
                chunk_type="key_fields",
                text=key_fields_text,
                metadata={**base_metadata, "chunk_type": "key_fields"}
            ))
        
        # 3. EC summary chunk
        ec_summary = precedent.get("ec_summary", "")
        if ec_summary:
            chunks.append(PrecedentChunk(
                case_id=case_id,
                chunk_type="ec_summary",
                text=f"EC Summary: {ec_summary}",
                metadata={**base_metadata, "chunk_type": "ec_summary"}
            ))
        
        # 4. Review notes chunk
        review_notes = precedent.get("review_notes", "")
        if review_notes:
            chunks.append(PrecedentChunk(
                case_id=case_id,
                chunk_type="review_notes",
                text=f"L2 Review Notes: {review_notes}",
                metadata={**base_metadata, "chunk_type": "review_notes"}
            ))
        
        # 5. Exceptions chunk (false positives to avoid)
        exceptions = precedent.get("exceptions", "")
        if exceptions:
            chunks.append(PrecedentChunk(
                case_id=case_id,
                chunk_type="exceptions",
                text=f"Exceptions (do not flag): {exceptions}",
                metadata={**base_metadata, "chunk_type": "exceptions"}
            ))
        
        # 6. Flow summary chunk (optional)
        flow_summary = precedent.get("flow_summary", "")
        if flow_summary:
            chunks.append(PrecedentChunk(
                case_id=case_id,
                chunk_type="flow_summary",
                text=f"Title Flow Summary: {flow_summary}",
                metadata={**base_metadata, "chunk_type": "flow_summary"}
            ))
        
        return chunks
    
    def _format_key_fields(self, key_fields: Dict) -> str:
        """Format key fields as readable text for embedding."""
        parts = []
        
        if key_fields.get("survey_no"):
            parts.append(f"Survey No: {key_fields['survey_no']}")
        if key_fields.get("house_no"):
            parts.append(f"House No: {key_fields['house_no']}")
        if key_fields.get("extent"):
            unit = key_fields.get("extent_unit", "")
            parts.append(f"Extent: {key_fields['extent']} {unit}")
        if key_fields.get("doc_nos"):
            parts.append(f"Documents: {', '.join(key_fields['doc_nos'])}")
        if key_fields.get("deed_types"):
            parts.append(f"Deed Types: {', '.join(key_fields['deed_types'])}")
        if key_fields.get("mortgage_flag"):
            parts.append("Mortgage: Active")
        
        return " | ".join(parts)
    
    def ingest_precedent(self, precedent: Dict) -> int:
        """
        Ingest a single precedent into the vector store.
        Returns the number of chunks added.
        """
        chunks = self.chunk_precedent(precedent)
        
        if not chunks:
            return 0
        
        # Generate embeddings
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embeddings_provider.embed_batch(texts)
        
        # Prepare for ChromaDB
        ids = [f"{chunk.case_id}_{chunk.chunk_type}" for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        
        # Upsert to collection
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        
        return len(chunks)
    
    def ingest_directory(self, directory: str) -> Dict[str, int]:
        """
        Ingest all precedent JSON files from a directory.
        Returns a summary of ingested files.
        """
        directory = Path(directory)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        results = {
            "files_processed": 0,
            "total_chunks": 0,
            "errors": [],
        }
        
        json_files = list(directory.glob("*.json"))
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    precedent = json.load(f)
                
                chunks_added = self.ingest_precedent(precedent)
                results["files_processed"] += 1
                results["total_chunks"] += chunks_added
                
            except Exception as e:
                results["errors"].append(f"{json_file.name}: {str(e)}")
        
        return results
    
    def retrieve(
        self,
        query: str,
        k: int = 8,
        filter_state: Optional[str] = None,
    ) -> List[Dict]:
        """
        Retrieve similar chunks from the vector store.
        
        Args:
            query: The query text (typically case fingerprint)
            k: Number of chunks to retrieve
            filter_state: Optional state filter (e.g., "KA", "TN")
        
        Returns:
            List of retrieved chunks with metadata
        """
        # Generate query embedding
        query_embedding = self.embeddings_provider.embed_text(query)
        
        # Build where filter
        where_filter = None
        if filter_state:
            where_filter = {"state": filter_state}
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        retrieved = []
        
        if results and results['ids'] and results['ids'][0]:
            for i, doc_id in enumerate(results['ids'][0]):
                retrieved.append({
                    "id": doc_id,
                    "text": results['documents'][0][i] if results['documents'] else "",
                    "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                    "distance": results['distances'][0][i] if results['distances'] else 0,
                })
        
        return retrieved
    
    def retrieve_precedents(
        self,
        query: str,
        k: int = 8,
        n: int = 5,
        filter_state: Optional[str] = None,
    ) -> List[Dict]:
        """
        Retrieve top N precedents (consolidated from top K chunks).
        
        Args:
            query: The query text (typically case fingerprint)
            k: Number of chunks to retrieve initially
            n: Number of unique precedents to return
        
        Returns:
            List of precedent summaries grouped by case_id
        """
        chunks = self.retrieve(query, k=k, filter_state=filter_state)
        
        # Group by case_id
        precedents = {}
        
        for chunk in chunks:
            case_id = chunk['metadata'].get('case_id', 'unknown')
            
            if case_id not in precedents:
                precedents[case_id] = {
                    "case_id": case_id,
                    "state": chunk['metadata'].get('state', ''),
                    "district": chunk['metadata'].get('district', ''),
                    "sro": chunk['metadata'].get('sro', ''),
                    "survey_no": chunk['metadata'].get('survey_no', ''),
                    "deed_types": chunk['metadata'].get('deed_types', ''),
                    "chunks": [],
                    "min_distance": chunk['distance'],
                }
            
            precedents[case_id]["chunks"].append({
                "type": chunk['metadata'].get('chunk_type', ''),
                "text": chunk['text'],
            })
            
            # Track minimum distance for sorting
            if chunk['distance'] < precedents[case_id]["min_distance"]:
                precedents[case_id]["min_distance"] = chunk['distance']
        
        # Sort by minimum distance and return top N
        sorted_precedents = sorted(
            precedents.values(),
            key=lambda x: x['min_distance']
        )
        
        return sorted_precedents[:n]
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store."""
        return {
            "collection_name": self.collection_name,
            "total_chunks": self.collection.count(),
            "persist_directory": self.persist_directory,
        }
    
    def clear(self):
        """Clear all data from the collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "NirnAI precedent case chunks"}
        )


def format_precedents_for_prompt(precedents: List[Dict]) -> str:
    """
    Format retrieved precedents into a string for LLM prompts.
    """
    if not precedents:
        return "No similar precedents found."
    
    parts = []
    
    for i, prec in enumerate(precedents, 1):
        prec_parts = [
            f"--- Precedent {i}: {prec['case_id']} ---",
            f"Location: {prec['state']}, {prec['district']}, SRO: {prec['sro']}",
            f"Survey: {prec['survey_no']} | Deed Types: {prec['deed_types']}",
        ]
        
        # Add chunk texts
        for chunk in prec['chunks']:
            chunk_type = chunk['type'].replace('_', ' ').title()
            prec_parts.append(f"[{chunk_type}] {chunk['text']}")
        
        parts.append("\n".join(prec_parts))
    
    return "\n\n".join(parts)


# CLI for ingestion
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest precedents into vector store")
    parser.add_argument(
        "--directory",
        "-d",
        default="./data/precedents",
        help="Directory containing precedent JSON files"
    )
    parser.add_argument(
        "--persist",
        "-p",
        default="./chroma_db",
        help="Directory to persist ChromaDB"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing data before ingestion"
    )
    
    args = parser.parse_args()
    
    print(f"Initializing PrecedentStore...")
    store = PrecedentStore(persist_directory=args.persist)
    
    if args.clear:
        print("Clearing existing data...")
        store.clear()
    
    print(f"Ingesting precedents from {args.directory}...")
    results = store.ingest_directory(args.directory)
    
    print(f"\nIngestion complete:")
    print(f"  Files processed: {results['files_processed']}")
    print(f"  Total chunks: {results['total_chunks']}")
    
    if results['errors']:
        print(f"  Errors: {len(results['errors'])}")
        for error in results['errors']:
            print(f"    - {error}")
    
    stats = store.get_stats()
    print(f"\nStore stats: {stats}")
