#!/usr/bin/env python3
"""
Migration script to push all local precedents to Pinecone.
Run this once to populate your Pinecone index with precedent data.

Usage:
    python migrate_to_pinecone.py --directory ./data/precedents
    
This will:
1. Connect to Pinecone using PINECONE_API_KEY from .env
2. Create the index if it doesn't exist
3. Process all JSON files in the directory
4. Upload embeddings to Pinecone

Progress is shown during migration. The script is idempotent - 
running it again will upsert (update) existing records.
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()


def migrate_to_pinecone(directory: str, batch_size: int = 50, clear_first: bool = False):
    """
    Migrate all precedent JSON files to Pinecone.
    """
    from src.pinecone_store import PineconeStore
    
    print("=" * 60)
    print("NirnAI Pinecone Migration")
    print("=" * 60)
    
    # Check API key
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("❌ Error: PINECONE_API_KEY not found in environment")
        print("   Add it to your .env file: PINECONE_API_KEY=your-key-here")
        sys.exit(1)
    
    print(f"\n✅ Pinecone API key found: {api_key[:15]}...")
    
    # Initialize store
    print("\n📡 Connecting to Pinecone...")
    store = PineconeStore()
    
    # Get initial stats
    initial_stats = store.get_stats()
    print(f"   Index: {initial_stats['index_name']}")
    print(f"   Current vectors: {initial_stats['total_vectors']}")
    
    # Clear if requested
    if clear_first:
        print("\n🗑️  Clearing existing data...")
        store.clear()
        print("   Done.")
    
    # Check directory
    directory = Path(directory)
    if not directory.exists():
        print(f"\n❌ Error: Directory not found: {directory}")
        sys.exit(1)
    
    json_files = list(directory.glob("*.json"))
    total_files = len(json_files)
    
    print(f"\n📁 Found {total_files} JSON files in {directory}")
    
    if total_files == 0:
        print("   No files to process.")
        return
    
    # Estimate time
    estimated_minutes = (total_files / 100) * 1.5  # ~1.5 min per 100 files
    print(f"   Estimated time: ~{estimated_minutes:.1f} minutes")
    
    # Start migration
    print(f"\n🚀 Starting migration...")
    print("-" * 40)
    
    start_time = datetime.now()
    results = store.ingest_directory(str(directory))
    end_time = datetime.now()
    
    # Results
    duration = (end_time - start_time).total_seconds()
    
    print("-" * 40)
    print(f"\n✅ Migration complete!")
    print(f"   Files processed: {results['files_processed']}")
    print(f"   Total chunks: {results['total_chunks']}")
    print(f"   Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    print(f"   Rate: {results['files_processed'] / duration * 60:.0f} files/minute")
    
    if results['errors']:
        print(f"\n⚠️  Errors ({len(results['errors'])}):")
        for error in results['errors'][:10]:  # Show first 10
            print(f"   - {error}")
        if len(results['errors']) > 10:
            print(f"   ... and {len(results['errors']) - 10} more")
    
    # Final stats
    final_stats = store.get_stats()
    print(f"\n📊 Final index stats:")
    print(f"   Total vectors: {final_stats['total_vectors']}")
    print(f"   Index: {final_stats['index_name']}")


def main():
    parser = argparse.ArgumentParser(
        description="Migrate precedent JSONs to Pinecone vector store"
    )
    parser.add_argument(
        "--directory", "-d",
        default="./data/precedents",
        help="Directory containing precedent JSON files (default: ./data/precedents)"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing Pinecone data before migration"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=50,
        help="Batch size for processing (default: 50)"
    )
    
    args = parser.parse_args()
    
    migrate_to_pinecone(
        directory=args.directory,
        batch_size=args.batch_size,
        clear_first=args.clear
    )


if __name__ == "__main__":
    main()
