#!/usr/bin/env python3
"""
Evaluation harness for NirnAI RAG Review.
Runs reviews on sample cases and outputs metrics + scoring template.
"""

import json
import os
import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class ReviewMetrics:
    """Metrics for a single review."""
    case_id: str
    total_issues: int
    critical_count: int
    major_count: int
    minor_count: int
    property_details_count: int
    schedule_of_property_count: int
    documents_scrutinized_count: int
    encumbrance_certificate_count: int
    flow_of_title_count: int
    mutation_and_tax_count: int
    conclusion_and_remarks_count: int
    layout_and_flowchart_count: int
    risk_level: str
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ScoringEntry:
    """Entry for manual precision/recall scoring."""
    case_id: str
    issue_id: str
    section: str
    severity: str
    message: str
    true_positive: Optional[bool]  # None = not scored yet
    notes: str


def calculate_metrics(review: Dict, case_id: str = "unknown") -> ReviewMetrics:
    """Calculate metrics from a review object."""
    sections = review.get("sections", {})
    
    total = 0
    critical = 0
    major = 0
    minor = 0
    section_counts = {}
    
    for section_name, issues in sections.items():
        if isinstance(issues, list):
            section_counts[section_name] = len(issues)
            total += len(issues)
            
            for issue in issues:
                severity = issue.get("severity", "minor")
                if severity == "critical":
                    critical += 1
                elif severity == "major":
                    major += 1
                else:
                    minor += 1
    
    return ReviewMetrics(
        case_id=case_id,
        total_issues=total,
        critical_count=critical,
        major_count=major,
        minor_count=minor,
        property_details_count=section_counts.get("property_details", 0),
        schedule_of_property_count=section_counts.get("schedule_of_property", 0),
        documents_scrutinized_count=section_counts.get("documents_scrutinized", 0),
        encumbrance_certificate_count=section_counts.get("encumbrance_certificate", 0),
        flow_of_title_count=section_counts.get("flow_of_title", 0),
        mutation_and_tax_count=section_counts.get("mutation_and_tax", 0),
        conclusion_and_remarks_count=section_counts.get("conclusion_and_remarks", 0),
        layout_and_flowchart_count=section_counts.get("layout_and_flowchart", 0),
        risk_level=review.get("overall_risk_level", "N/A"),
    )


def generate_scoring_template(
    review: Dict,
    case_id: str = "unknown"
) -> List[ScoringEntry]:
    """Generate a scoring template for manual precision/recall annotation."""
    entries = []
    sections = review.get("sections", {})
    
    for section_name, issues in sections.items():
        if isinstance(issues, list):
            for issue in issues:
                entries.append(ScoringEntry(
                    case_id=case_id,
                    issue_id=issue.get("id", ""),
                    section=section_name,
                    severity=issue.get("severity", ""),
                    message=issue.get("message_for_maker", "")[:100],  # Truncate
                    true_positive=None,
                    notes="",
                ))
    
    return entries


def save_scoring_template_csv(
    entries: List[ScoringEntry],
    output_path: str
):
    """Save scoring template to CSV for manual annotation."""
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "case_id", "issue_id", "section", "severity", 
            "message", "true_positive", "notes"
        ])
        
        for entry in entries:
            writer.writerow([
                entry.case_id,
                entry.issue_id,
                entry.section,
                entry.severity,
                entry.message,
                "",  # true_positive - to be filled manually
                "",  # notes - to be filled manually
            ])


def save_metrics_json(
    metrics: List[ReviewMetrics],
    output_path: str
):
    """Save metrics summary to JSON."""
    data = {
        "generated_at": datetime.now().isoformat(),
        "total_cases": len(metrics),
        "aggregate": {
            "total_issues": sum(m.total_issues for m in metrics),
            "critical_count": sum(m.critical_count for m in metrics),
            "major_count": sum(m.major_count for m in metrics),
            "minor_count": sum(m.minor_count for m in metrics),
            "avg_issues_per_case": sum(m.total_issues for m in metrics) / len(metrics) if metrics else 0,
        },
        "per_case": [m.to_dict() for m in metrics]
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def run_evaluation(
    case_files: List[str],
    output_dir: str = "./eval_results",
    precedent_dir: str = "./data/precedents",
    chroma_dir: str = "./chroma_db",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run evaluation on multiple case files.
    
    Args:
        case_files: List of paths to merged case JSON files
        output_dir: Directory to save evaluation results
        precedent_dir: Directory containing precedent JSONs
        chroma_dir: ChromaDB persistence directory
        verbose: Print progress
    
    Returns:
        Evaluation summary
    """
    from src.review import ReviewPipeline
    from src.ingest import PrecedentStore
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize store
    store = PrecedentStore(persist_directory=chroma_dir)
    
    # Ingest precedents if needed
    if store.collection.count() == 0:
        if Path(precedent_dir).exists():
            if verbose:
                print(f"Ingesting precedents from {precedent_dir}...")
            results = store.ingest_directory(precedent_dir)
            if verbose:
                print(f"  Ingested {results['files_processed']} files")
    
    # Initialize pipeline
    pipeline = ReviewPipeline(
        precedent_store=store,
        output_dir=str(output_dir / "reviews"),
    )
    
    # Run reviews
    all_metrics = []
    all_scoring_entries = []
    
    for i, case_file in enumerate(case_files):
        case_id = Path(case_file).stem
        
        if verbose:
            print(f"\n[{i+1}/{len(case_files)}] Processing {case_id}...")
        
        try:
            review = pipeline.review_from_file(
                file_path=case_file,
                save_output=True,
                verbose=False,
            )
            
            # Calculate metrics
            metrics = calculate_metrics(review, case_id)
            all_metrics.append(metrics)
            
            # Generate scoring template
            scoring = generate_scoring_template(review, case_id)
            all_scoring_entries.extend(scoring)
            
            if verbose:
                print(f"  Total issues: {metrics.total_issues} "
                      f"(C:{metrics.critical_count} M:{metrics.major_count} m:{metrics.minor_count})")
                print(f"  Risk level: {metrics.risk_level}")
            
        except Exception as e:
            if verbose:
                print(f"  ERROR: {e}")
            continue
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    metrics_path = output_dir / f"metrics_{timestamp}.json"
    save_metrics_json(all_metrics, str(metrics_path))
    if verbose:
        print(f"\nMetrics saved to: {metrics_path}")
    
    scoring_path = output_dir / f"scoring_template_{timestamp}.csv"
    save_scoring_template_csv(all_scoring_entries, str(scoring_path))
    if verbose:
        print(f"Scoring template saved to: {scoring_path}")
    
    # Print summary
    if verbose and all_metrics:
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Cases processed: {len(all_metrics)}")
        print(f"Total issues: {sum(m.total_issues for m in all_metrics)}")
        print(f"  Critical: {sum(m.critical_count for m in all_metrics)}")
        print(f"  Major: {sum(m.major_count for m in all_metrics)}")
        print(f"  Minor: {sum(m.minor_count for m in all_metrics)}")
        print(f"Average issues per case: {sum(m.total_issues for m in all_metrics) / len(all_metrics):.1f}")
        
        risk_dist = {}
        for m in all_metrics:
            risk_dist[m.risk_level] = risk_dist.get(m.risk_level, 0) + 1
        print(f"\nRisk level distribution:")
        for level, count in risk_dist.items():
            print(f"  {level}: {count}")
    
    return {
        "cases_processed": len(all_metrics),
        "total_issues": sum(m.total_issues for m in all_metrics),
        "metrics_file": str(metrics_path),
        "scoring_file": str(scoring_path),
    }


def evaluate_directory(
    cases_dir: str,
    output_dir: str = "./eval_results",
    precedent_dir: str = "./data/precedents",
    chroma_dir: str = "./chroma_db",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run evaluation on all case files in a directory.
    """
    cases_dir = Path(cases_dir)
    
    if not cases_dir.exists():
        raise FileNotFoundError(f"Cases directory not found: {cases_dir}")
    
    case_files = list(cases_dir.glob("*.json"))
    
    if not case_files:
        raise ValueError(f"No JSON files found in {cases_dir}")
    
    if verbose:
        print(f"Found {len(case_files)} case files in {cases_dir}")
    
    return run_evaluation(
        case_files=[str(f) for f in case_files],
        output_dir=output_dir,
        precedent_dir=precedent_dir,
        chroma_dir=chroma_dir,
        verbose=verbose,
    )


# =============================================================================
# CLI INTERFACE
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate NirnAI Review on sample cases"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Single file command
    single_parser = subparsers.add_parser("single", help="Evaluate a single case file")
    single_parser.add_argument("case_file", help="Path to merged case JSON")
    single_parser.add_argument("--output", "-o", default="./eval_results", help="Output directory")
    
    # Directory command
    dir_parser = subparsers.add_parser("directory", help="Evaluate all cases in a directory")
    dir_parser.add_argument("cases_dir", help="Directory containing case JSON files")
    dir_parser.add_argument("--output", "-o", default="./eval_results", help="Output directory")
    
    # Examples command
    examples_parser = subparsers.add_parser("examples", help="Run evaluation on example files")
    examples_parser.add_argument("--output", "-o", default="./eval_results", help="Output directory")
    
    # Common arguments
    for p in [single_parser, dir_parser, examples_parser]:
        p.add_argument("--precedents", "-p", default="./data/precedents", help="Precedents directory")
        p.add_argument("--chroma", "-c", default="./chroma_db", help="ChromaDB directory")
        p.add_argument("--quiet", "-q", action="store_true", help="Suppress output")
    
    args = parser.parse_args()
    
    if args.command == "single":
        run_evaluation(
            case_files=[args.case_file],
            output_dir=args.output,
            precedent_dir=args.precedents,
            chroma_dir=args.chroma,
            verbose=not args.quiet,
        )
    
    elif args.command == "directory":
        evaluate_directory(
            cases_dir=args.cases_dir,
            output_dir=args.output,
            precedent_dir=args.precedents,
            chroma_dir=args.chroma,
            verbose=not args.quiet,
        )
    
    elif args.command == "examples":
        # Run on example file
        example_file = "./examples/example_merged_case.json"
        if Path(example_file).exists():
            run_evaluation(
                case_files=[example_file],
                output_dir=args.output,
                precedent_dir=args.precedents,
                chroma_dir=args.chroma,
                verbose=not args.quiet,
            )
        else:
            print(f"Example file not found: {example_file}")
    
    else:
        parser.print_help()
