#!/usr/bin/env python3
"""
Quick test script for NirnAI RAG Review.
Run: python test_run.py
"""

import os
import sys
import json

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

def main():
    print("=" * 60)
    print("NirnAI RAG Review - Test Run")
    print("=" * 60)
    
    # Step 1: Check for API key
    print("\n[1/5] Checking for LLM API key...")
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    
    if openai_key:
        print(f"  ✓ Found OPENAI_API_KEY: {openai_key[:10]}...")
    elif anthropic_key:
        print(f"  ✓ Found ANTHROPIC_API_KEY: {anthropic_key[:10]}...")
    else:
        print("  ✗ ERROR: No API key found!")
        print("    Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable")
        print("    Example: export OPENAI_API_KEY='sk-your-key-here'")
        sys.exit(1)
    
    # Step 2: Test imports
    print("\n[2/5] Testing imports...")
    try:
        from src.embeddings import get_embeddings_provider
        from src.ingest import PrecedentStore
        from src.utils import build_fingerprint, build_current_case_extract
        from src.review import ReviewPipeline
        print("  ✓ All imports successful")
    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        print("    Run: pip install -r requirements.txt")
        sys.exit(1)
    
    # Step 3: Initialize vector store and ingest
    print("\n[3/5] Initializing vector store...")
    try:
        store = PrecedentStore(persist_directory="./chroma_db")
        
        # Check if we need to ingest
        current_count = store.collection.count()
        print(f"  Current chunks in store: {current_count}")
        
        if current_count == 0:
            print("  Ingesting precedents from ./data/precedents/...")
            results = store.ingest_directory("./data/precedents")
            print(f"  ✓ Ingested {results['files_processed']} files, {results['total_chunks']} chunks")
        else:
            print("  ✓ Precedents already ingested")
            
    except Exception as e:
        print(f"  ✗ Vector store error: {e}")
        sys.exit(1)
    
    # Step 4: Load test case
    print("\n[4/5] Loading test case...")
    test_file = "./examples/example_merged_case2.json"
    
    if not os.path.exists(test_file):
        print(f"  ✗ Test file not found: {test_file}")
        sys.exit(1)
    
    with open(test_file, 'r', encoding='utf-8') as f:
        merged_case = json.load(f)
    
    # Show fingerprint
    fingerprint = build_fingerprint(merged_case)
    print(f"  ✓ Loaded: {test_file}")
    print(f"  Fingerprint: {fingerprint[:80]}...")
    
    # Test retrieval
    print("\n  Testing RAG retrieval...")
    precedents = store.retrieve_precedents(fingerprint, k=8, n=3)
    print(f"  ✓ Retrieved {len(precedents)} similar precedents:")
    for p in precedents:
        print(f"    - {p['case_id']} (distance: {p['min_distance']:.3f})")
    
    # Step 5: Run full review
    print("\n[5/5] Running two-stage review...")
    print("  This will call the LLM API (may take 30-60 seconds)...")
    
    try:
        pipeline = ReviewPipeline(
            precedent_store=store,
            output_dir="./outputs"
        )
        
        review = pipeline.review(
            merged_case=merged_case,
            save_output=True,
            case_id="test_case",
            verbose=True
        )
        
        print("\n" + "=" * 60)
        print("TEST COMPLETE - REVIEW RESULTS")
        print("=" * 60)
        
        print(f"\nRisk Level: {review.get('overall_risk_level', 'N/A')}")
        print(f"Summary: {review.get('overall_summary', 'N/A')[:200]}...")
        
        # Count issues
        total_issues = 0
        print("\nIssues by Section:")
        for section, issues in review.get('sections', {}).items():
            if isinstance(issues, list) and len(issues) > 0:
                total_issues += len(issues)
                print(f"  {section}: {len(issues)} issues")
                for issue in issues[:2]:  # Show first 2 issues
                    print(f"    - [{issue.get('severity', '?')}] {issue.get('id', '?')}: {issue.get('message_for_maker', '')[:60]}...")
        
        print(f"\nTotal Issues Found: {total_issues}")
        print(f"\nOutput saved to: ./outputs/")
        
    except NotImplementedError as e:
        print(f"\n  ✗ LLM not configured: {e}")
        print("    Make sure your API key is set correctly")
        sys.exit(1)
    except Exception as e:
        print(f"\n  ✗ Review failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
