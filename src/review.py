"""
Main review pipeline for NirnAI RAG Review.
Two-stage LLM review: Generator → Critic.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .utils import (
    build_fingerprint,
    build_current_case_extract,
    deduplicate_issues,
    renumber_issues,
)
from .ingest import PrecedentStore, format_precedents_for_prompt
from .prompts import format_generator_prompt, format_critic_prompt
from .llm import call_llm_json, validate_review_object, validate_issue


class ReviewPipeline:
    """
    Two-stage LLM review pipeline with RAG retrieval.
    
    Stage 1 (Generator): High-recall issue detection
    Stage 2 (Critic): Prune, deduplicate, calibrate severity
    """
    
    def __init__(
        self,
        precedent_store: Optional[PrecedentStore] = None,
        chroma_persist_dir: str = "./chroma_db",
        output_dir: str = "./outputs",
    ):
        """
        Initialize the review pipeline.
        
        Args:
            precedent_store: Optional pre-initialized PrecedentStore
            chroma_persist_dir: Directory for ChromaDB persistence
            output_dir: Directory to save review outputs
        """
        self.precedent_store = precedent_store or PrecedentStore(
            persist_directory=chroma_persist_dir
        )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def review(
        self,
        merged_case: Dict,
        save_output: bool = True,
        case_id: Optional[str] = None,
        verbose: bool = False,
    ) -> Dict:
        """
        Run the full two-stage review pipeline on a merged case.
        
        Args:
            merged_case: The merged case JSON containing translated_docs, ec, report
            save_output: Whether to save the review output to file
            case_id: Optional case identifier for naming output file
            verbose: Print progress information
        
        Returns:
            The final REVIEW_OBJECT
        """
        if verbose:
            print("=" * 60)
            print("NirnAI Two-Stage Review Pipeline")
            print("=" * 60)
        
        # Step 1: Build fingerprint and extract
        if verbose:
            print("\n[1/5] Building case fingerprint and extract...")
        
        fingerprint = build_fingerprint(merged_case)
        case_extract = build_current_case_extract(merged_case)
        # Use ensure_ascii=False to preserve Tamil/other Unicode characters
        case_extract_str = json.dumps(case_extract, indent=2, default=str, ensure_ascii=False)
        
        if verbose:
            print(f"  Fingerprint: {fingerprint[:100]}...")
        
        # Step 2: Retrieve precedents
        if verbose:
            print("\n[2/5] Retrieving similar precedents...")
        
        state_filter = merged_case.get('meta', {}).get('state')
        precedents = self.precedent_store.retrieve_precedents(
            query=fingerprint,
            k=15,
            n=8,
            filter_state=state_filter,
        )
        precedent_snippets = format_precedents_for_prompt(precedents)
        
        if verbose:
            print(f"  Retrieved {len(precedents)} precedents")
        
        # Step 3: Stage 1 - Generator
        if verbose:
            print("\n[3/5] Running Stage 1 (Generator)...")
        
        generator_prompt = format_generator_prompt(
            case_extract=case_extract_str,
            precedent_snippets=precedent_snippets,
        )
        
        candidate_review = call_llm_json(generator_prompt)
        
        if verbose:
            total_candidates = self._count_issues(candidate_review)
            print(f"  Generated {total_candidates} candidate issues")
        
        # Step 4: Stage 2 - Critic
        if verbose:
            print("\n[4/5] Running Stage 2 (Critic)...")
        
        critic_prompt = format_critic_prompt(
            case_extract=case_extract_str,
            precedent_snippets=precedent_snippets,
            # Use ensure_ascii=False to preserve Tamil/other Unicode characters
            candidate_review=json.dumps(candidate_review, indent=2, ensure_ascii=False),
        )
        
        final_review = call_llm_json(critic_prompt)
        
        # Step 5: Post-process and validate
        if verbose:
            print("\n[5/5] Validating and post-processing...")
        
        final_review = self._post_process(final_review)
        
        try:
            validate_review_object(final_review)
            if verbose:
                print("  ✓ Review object validated")
        except ValueError as e:
            if verbose:
                print(f"  ⚠ Validation warning: {e}")
        
        # Save output
        if save_output:
            output_path = self._save_output(final_review, case_id)
            if verbose:
                print(f"\n  Output saved to: {output_path}")
        
        if verbose:
            self._print_summary(final_review)
        
        return final_review
    
    def review_from_file(
        self,
        file_path: str,
        save_output: bool = True,
        verbose: bool = False,
    ) -> Dict:
        """
        Run review on a merged case JSON file.
        
        Args:
            file_path: Path to the merged case JSON file
            save_output: Whether to save the review output
            verbose: Print progress information
        
        Returns:
            The final REVIEW_OBJECT
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            merged_case = json.load(f)
        
        # Extract case_id from filename
        case_id = Path(file_path).stem
        
        return self.review(
            merged_case=merged_case,
            save_output=save_output,
            case_id=case_id,
            verbose=verbose,
        )
    
    def _post_process(self, review: Dict) -> Dict:
        """Post-process the review object."""
        sections = review.get('sections', {})
        
        # Deduplicate issues within each section
        for section_name, issues in sections.items():
            if isinstance(issues, list):
                sections[section_name] = deduplicate_issues(issues)
        
        # Renumber issues with consistent IDs
        sections = renumber_issues(sections)
        review['sections'] = sections
        
        # Filter out issues without proper evidence
        for section_name, issues in sections.items():
            valid_issues = []
            for issue in issues:
                evidence = issue.get('evidence', {})
                if evidence.get('from_report') and evidence.get('from_source_docs'):
                    valid_issues.append(issue)
            sections[section_name] = valid_issues
        
        # Re-renumber after filtering
        sections = renumber_issues(sections)
        review['sections'] = sections
        
        return review
    
    def _count_issues(self, review: Dict) -> int:
        """Count total issues in a review object."""
        total = 0
        for issues in review.get('sections', {}).values():
            if isinstance(issues, list):
                total += len(issues)
        return total
    
    def _save_output(self, review: Dict, case_id: Optional[str] = None) -> Path:
        """Save review output to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if case_id:
            filename = f"review_{case_id}_{timestamp}.json"
        else:
            filename = f"review_{timestamp}.json"
        
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(review, f, indent=2, ensure_ascii=False)
        
        return output_path
    
    def _print_summary(self, review: Dict):
        """Print a summary of the review."""
        print("\n" + "=" * 60)
        print("REVIEW SUMMARY")
        print("=" * 60)
        print(f"Risk Level: {review.get('overall_risk_level', 'N/A')}")
        print(f"Summary: {review.get('overall_summary', 'N/A')[:200]}...")
        print("\nIssues by Section:")
        
        for section, issues in review.get('sections', {}).items():
            if isinstance(issues, list) and len(issues) > 0:
                critical = sum(1 for i in issues if i.get('severity') == 'critical')
                major = sum(1 for i in issues if i.get('severity') == 'major')
                minor = sum(1 for i in issues if i.get('severity') == 'minor')
                print(f"  {section}: {len(issues)} issues (C:{critical} M:{major} m:{minor})")
            else:
                print(f"  {section}: 0 issues")
        
        total = self._count_issues(review)
        print(f"\nTotal Issues: {total}")


def run_review(
    merged_case_path: str,
    precedent_dir: str = "./data/precedents",
    chroma_dir: str = "./chroma_db",
    output_dir: str = "./outputs",
    verbose: bool = True,
) -> Dict:
    """
    Convenience function to run a review on a merged case file.
    
    Args:
        merged_case_path: Path to the merged case JSON
        precedent_dir: Directory containing precedent JSONs
        chroma_dir: ChromaDB persistence directory
        output_dir: Output directory for reviews
        verbose: Print progress
    
    Returns:
        The final REVIEW_OBJECT
    """
    # Initialize store and ingest precedents if needed
    store = PrecedentStore(persist_directory=chroma_dir)
    
    # Check if we need to ingest
    if store.collection.count() == 0:
        if Path(precedent_dir).exists():
            if verbose:
                print(f"Ingesting precedents from {precedent_dir}...")
            results = store.ingest_directory(precedent_dir)
            if verbose:
                print(f"  Ingested {results['files_processed']} files, {results['total_chunks']} chunks")
    
    # Run pipeline
    pipeline = ReviewPipeline(
        precedent_store=store,
        output_dir=output_dir,
    )
    
    return pipeline.review_from_file(
        file_path=merged_case_path,
        save_output=True,
        verbose=verbose,
    )


# =============================================================================
# CLI INTERFACE
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run NirnAI Two-Stage Review on a merged case"
    )
    parser.add_argument(
        "case_file",
        help="Path to merged case JSON file"
    )
    parser.add_argument(
        "--precedents",
        "-p",
        default="./data/precedents",
        help="Directory containing precedent JSONs"
    )
    parser.add_argument(
        "--chroma",
        "-c",
        default="./chroma_db",
        help="ChromaDB persistence directory"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="./outputs",
        help="Output directory for reviews"
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress output"
    )
    
    args = parser.parse_args()
    
    review = run_review(
        merged_case_path=args.case_file,
        precedent_dir=args.precedents,
        chroma_dir=args.chroma,
        output_dir=args.output,
        verbose=not args.quiet,
    )
    
    print("\nFinal Review Output:")
    print(json.dumps(review, indent=2, ensure_ascii=False))
