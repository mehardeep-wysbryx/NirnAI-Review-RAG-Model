"""
Pinecone-based vector store for NirnAI RAG Review.
Cloud-hosted alternative to ChromaDB for production deployment.
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


class PineconeStore:
    """
    Vector store for precedent cases using Pinecone (cloud).
    Drop-in replacement for PrecedentStore (ChromaDB).
    """
    
    CHUNK_TYPES = [
        "fingerprint",
        "key_fields", 
        "ec_summary",
        "review_notes",
        "exceptions",
        "flow_summary",
    ]
    
    INDEX_NAME = "nirnai-precedents"
    DIMENSION = 1536  # OpenAI text-embedding-3-small dimension
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        index_name: Optional[str] = None,
        embeddings_provider: Optional[EmbeddingsProvider] = None,
    ):
        try:
            from pinecone import Pinecone, ServerlessSpec
        except ImportError:
            raise ImportError(
                "pinecone is required. Install with: pip install pinecone"
            )
        
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Pinecone API key required. Set PINECONE_API_KEY environment variable."
            )
        
        self.index_name = index_name or self.INDEX_NAME
        
        # Initialize embeddings provider - use OpenAI for production reliability
        self.embeddings_provider = embeddings_provider or get_embeddings_provider("openai")
        
        # Initialize Pinecone client
        self.pc = Pinecone(api_key=self.api_key)
        
        # Create index if it doesn't exist
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        
        if self.index_name not in existing_indexes:
            print(f"Creating Pinecone index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
        
        # Connect to index
        self.index = self.pc.Index(self.index_name)
    
    def chunk_precedent(self, precedent: Dict, filename: str = "unknown") -> List[PrecedentChunk]:
        """
        Break a precedent into semantic chunks for embedding.
        Handles both processed precedent format AND raw case format.
        """
        # Check if this is a raw case format (has attachments, encumbranceDetails, reportJson)
        if "attachments" in precedent or "encumbranceDetails" in precedent or "reportJson" in precedent:
            return self._chunk_raw_case(precedent, filename)
        else:
            return self._chunk_processed_precedent(precedent)
    
    def _chunk_raw_case(self, case: Dict, filename: str) -> List[PrecedentChunk]:
        """
        Chunk a raw case JSON (with attachments, encumbranceDetails, reportJson).
        This is the actual format of NirnAI case files.
        """
        from .utils import build_fingerprint, extract_from_encumbrance_details, extract_from_report_json
        
        # Generate case_id from filename or report code
        report = case.get("reportJson", {})
        case_id = report.get("code") or filename.replace(".json", "")
        
        # Extract data from all sources
        ec_details = case.get("encumbranceDetails", [])
        ec_extracted = extract_from_encumbrance_details(ec_details)
        report_extracted = extract_from_report_json(report)
        
        # Get key fields from report
        prop_details = report_extracted.get("property_details", {})
        
        # Base metadata
        base_metadata = {
            "case_id": case_id,
            "state": prop_details.get("state", "") or ec_extracted.get("detected_state", ""),
            "district": prop_details.get("district", ""),
            "sro": prop_details.get("sro", "") or ec_extracted.get("sro", ""),
            "survey_no": prop_details.get("survey_no", "") or ec_extracted.get("survey_no", ""),
            "extent": str(prop_details.get("extent", "")),
            "deed_types": prop_details.get("deed_type", ""),
        }
        
        chunks = []
        
        # 1. Fingerprint chunk - build from case data
        try:
            fingerprint = build_fingerprint(case)
            if fingerprint:
                chunks.append(PrecedentChunk(
                    case_id=case_id,
                    chunk_type="fingerprint",
                    text=fingerprint,
                    metadata={**base_metadata, "chunk_type": "fingerprint"}
                ))
        except Exception:
            pass  # Skip if fingerprint fails
        
        # 2. Key fields chunk from report
        key_fields_parts = []
        if prop_details.get("survey_no"):
            key_fields_parts.append(f"Survey: {prop_details['survey_no']}")
        if prop_details.get("house_no"):
            key_fields_parts.append(f"House: {prop_details['house_no']}")
        if prop_details.get("extent"):
            key_fields_parts.append(f"Extent: {prop_details['extent']}")
        if prop_details.get("deed_type"):
            key_fields_parts.append(f"Deed: {prop_details['deed_type']}")
        if prop_details.get("owner"):
            key_fields_parts.append(f"Owner: {prop_details['owner']}")
        if prop_details.get("village"):
            key_fields_parts.append(f"Village: {prop_details['village']}")
        
        if key_fields_parts:
            chunks.append(PrecedentChunk(
                case_id=case_id,
                chunk_type="key_fields",
                text=" | ".join(key_fields_parts),
                metadata={**base_metadata, "chunk_type": "key_fields"}
            ))
        
        # 3. EC summary chunk
        ec_transactions = ec_extracted.get("transactions", [])
        if ec_transactions:
            ec_summary_parts = []
            for txn in ec_transactions[:3]:  # First 3 transactions
                if txn.get("doc_no") and txn.get("deed_type"):
                    ec_summary_parts.append(f"{txn['deed_type']} ({txn['doc_no']})")
            if ec_summary_parts:
                ec_text = f"EC Transactions: {', '.join(ec_summary_parts)}"
                if ec_extracted.get("mortgage_flag"):
                    ec_text += " | Mortgage: Active"
                chunks.append(PrecedentChunk(
                    case_id=case_id,
                    chunk_type="ec_summary",
                    text=ec_text,
                    metadata={**base_metadata, "chunk_type": "ec_summary"}
                ))
        
        # 4. Boundaries chunk
        boundaries = report_extracted.get("boundaries", {})
        if boundaries:
            boundary_parts = []
            for direction in ["north", "south", "east", "west"]:
                if boundaries.get(direction):
                    boundary_parts.append(f"{direction.upper()}: {boundaries[direction]}")
            if boundary_parts:
                chunks.append(PrecedentChunk(
                    case_id=case_id,
                    chunk_type="boundaries",
                    text=f"Boundaries: {' | '.join(boundary_parts)}",
                    metadata={**base_metadata, "chunk_type": "boundaries"}
                ))
        
        # 5. Location chunk
        location_parts = []
        if prop_details.get("village"):
            location_parts.append(f"Village: {prop_details['village']}")
        if prop_details.get("taluk"):
            location_parts.append(f"Taluk: {prop_details['taluk']}")
        if prop_details.get("district"):
            location_parts.append(f"District: {prop_details['district']}")
        if prop_details.get("state"):
            location_parts.append(f"State: {prop_details['state']}")
        
        if location_parts:
            chunks.append(PrecedentChunk(
                case_id=case_id,
                chunk_type="location",
                text=" | ".join(location_parts),
                metadata={**base_metadata, "chunk_type": "location"}
            ))
        
        return chunks
    
    def _chunk_processed_precedent(self, precedent: Dict) -> List[PrecedentChunk]:
        """
        Chunk a processed precedent format (legacy format with case_id, meta, fingerprint, etc.).
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
            "deed_types": ",".join(key_fields.get("deed_types", [])) if isinstance(key_fields.get("deed_types"), list) else key_fields.get("deed_types", ""),
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
            if key_fields_text:
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
        
        # 5. Exceptions chunk
        exceptions = precedent.get("exceptions", "")
        if exceptions:
            chunks.append(PrecedentChunk(
                case_id=case_id,
                chunk_type="exceptions",
                text=f"Exceptions (do not flag): {exceptions}",
                metadata={**base_metadata, "chunk_type": "exceptions"}
            ))
        
        # 6. Flow summary chunk
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
    
    def ingest_precedent(self, precedent: Dict, filename: str = "unknown") -> int:
        """
        Ingest a single precedent into Pinecone.
        Returns the number of chunks added.
        """
        chunks = self.chunk_precedent(precedent, filename)
        
        if not chunks:
            return 0
        
        # Generate embeddings
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embeddings_provider.embed_batch(texts)
        
        # Prepare vectors for Pinecone
        vectors = []
        for i, chunk in enumerate(chunks):
            # Pinecone requires metadata values to be strings, numbers, or lists
            metadata = {
                "case_id": chunk.metadata.get("case_id", ""),
                "state": chunk.metadata.get("state", ""),
                "district": chunk.metadata.get("district", ""),
                "sro": chunk.metadata.get("sro", ""),
                "survey_no": chunk.metadata.get("survey_no", ""),
                "extent": chunk.metadata.get("extent", ""),
                "deed_types": chunk.metadata.get("deed_types", ""),
                "chunk_type": chunk.metadata.get("chunk_type", ""),
                "text": chunk.text[:1000],  # Store truncated text in metadata
            }
            
            vectors.append({
                "id": f"{chunk.case_id}_{chunk.chunk_type}",
                "values": embeddings[i],
                "metadata": metadata
            })
        
        # Upsert to Pinecone in batches of 100
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)
        
        return len(chunks)
    
    def ingest_directory(self, directory: str, batch_size: int = 50) -> Dict[str, int]:
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
        total_files = len(json_files)
        
        print(f"Found {total_files} JSON files to process")
        
        # Process in batches for progress reporting
        for i, json_file in enumerate(json_files):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    precedent = json.load(f)
                
                chunks_added = self.ingest_precedent(precedent, json_file.name)
                results["files_processed"] += 1
                results["total_chunks"] += chunks_added
                
                # Progress reporting
                if (i + 1) % 100 == 0:
                    print(f"  Processed {i + 1}/{total_files} files...")
                    
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
        Retrieve similar chunks from Pinecone.
        """
        # Generate query embedding
        query_embedding = self.embeddings_provider.embed_text(query)
        
        # Build filter
        filter_dict = None
        if filter_state:
            filter_dict = {"state": {"$eq": filter_state}}
        
        # Query Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=k,
            include_metadata=True,
            filter=filter_dict
        )
        
        # Format results
        retrieved = []
        
        for match in results.matches:
            retrieved.append({
                "id": match.id,
                "text": match.metadata.get("text", ""),
                "metadata": match.metadata,
                "distance": 1 - match.score,  # Convert similarity to distance
            })
        
        return retrieved
    
    def retrieve_precedents(
        self,
        query: str,
        k: int = 15,
        n: int = 8,
        filter_state: Optional[str] = None,
    ) -> List[Dict]:
        """
        Retrieve top N precedents (consolidated from top K chunks).
        Same interface as ChromaDB version.
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
            
            if chunk['distance'] < precedents[case_id]["min_distance"]:
                precedents[case_id]["min_distance"] = chunk['distance']
        
        # Sort by minimum distance and return top N
        sorted_precedents = sorted(
            precedents.values(),
            key=lambda x: x['min_distance']
        )
        
        return sorted_precedents[:n]
    
    def get_stats(self) -> Dict:
        """Get statistics about the Pinecone index."""
        stats = self.index.describe_index_stats()
        return {
            "index_name": self.index_name,
            "total_vectors": stats.total_vector_count,
            "dimension": stats.dimension,
            "namespaces": dict(stats.namespaces) if stats.namespaces else {},
        }
    
    def clear(self):
        """Clear all data from the index."""
        self.index.delete(delete_all=True)


def get_vector_store(use_pinecone: bool = None):
    """
    Factory function to get the appropriate vector store.
    
    Args:
        use_pinecone: If True, use Pinecone. If False, use ChromaDB.
                     If None, auto-detect based on environment.
    
    Returns:
        Either PineconeStore or PrecedentStore
    """
    if use_pinecone is None:
        # Auto-detect: use Pinecone if API key is available
        use_pinecone = bool(os.getenv("PINECONE_API_KEY"))
    
    if use_pinecone:
        return PineconeStore()
    else:
        from .ingest import PrecedentStore
        return PrecedentStore()
