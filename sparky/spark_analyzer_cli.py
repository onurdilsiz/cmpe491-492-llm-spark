#!/usr/bin/env python3
# spark_analyzer_cli.py
import os
import sys
import argparse
import asyncio
import datetime
import subprocess
from pathlib import Path

def create_sample_pdfs(base_dir, run_id):
    """Create empty sample PDFs for testing"""
    run_dir = Path(base_dir) / "runs" / f"spark_run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Create empty PDFs for each tab
    for tab in ["jobs", "stages", "storage", "environment", "executors", "sql"]:
        pdf_path = run_dir / f"{tab}.pdf"
        if not pdf_path.exists():
            # Create an empty PDF file
            with open(pdf_path, "wb") as f:
                # Minimal PDF content
                f.write(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<</Type/Catalog/Pages 2 0 R>>\nendobj\n2 0 obj\n<</Type/Pages/Kids[3 0 R]/Count 1>>\nendobj\n3 0 obj\n<</Type/Page/MediaBox[0 0 595 842]/Parent 2 0 R/Resources<<>>/Contents 4 0 R>>\nendobj\n4 0 obj\n<</Length 10>>stream\nBT\n/F1 12 Tf\nET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f \n0000000015 00000 n \n0000000060 00000 n \n0000000111 00000 n \n0000000198 00000 n \ntrailer\n<</Size 5/Root 1 0 R>>\nstartxref\n256\n%%EOF")
    
    print(f"Created sample PDFs in {run_dir}")

def setup_directories(base_dir):
    """Set up the directory structure for the analyzer"""
    # Create main directories
    for dir_path in ["runs", "outputs", "logs"]:
        path = Path(base_dir) / dir_path
        path.mkdir(parents=True, exist_ok=True)
    
    print(f"Directory structure created in {base_dir}")

async def run_analysis(run_id, base_dir, model, temperature):
    """Run the Spark UI analysis"""
    from spark_analyzer import SparkAnalyzerOrchestrator
    
    # Create and run the orchestrator
    orchestrator = SparkAnalyzerOrchestrator(
        run_id=run_id,
        base_directory=base_dir,
        model_name=model,
        temperature=temperature
    )
    
    status = await orchestrator.run_full_analysis()
    
    if "error" in status:
        print(f"Error in analysis: {status['error']}")
        return 1
    
    # Print the location of the final report
    print(f"\nAnalysis complete!")
    print(f"Final report available at: {status['final_report_path']}")
    print(f"Analysis took {status['execution_time_seconds']:.2f} seconds")
    
    return 0

def main():
    parser = argparse.ArgumentParser(description="Spark UI Performance Analyzer CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Set up directory structure")
    setup_parser.add_argument("--base-dir", required=True, help="Base directory for the analyzer system")
    
    # Create-sample command
    sample_parser = subparsers.add_parser("create-sample", help="Create sample PDFs for testing")
    sample_parser.add_argument("--run-id", required=True, help="Identifier for the sample run")
    sample_parser.add_argument("--base-dir", required=True, help="Base directory for the analyzer system")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze Spark UI data")
    analyze_parser.add_argument("--run-id", required=True, help="Identifier for the Spark run to analyze")
    analyze_parser.add_argument("--base-dir", required=True, help="Base directory for the analyzer system")
    analyze_parser.add_argument("--model", default="gpt-4o", help="LLM model to use")
    analyze_parser.add_argument("--temperature", type=float, default=0.0, help="Temperature setting for the LLM")
    
    # List runs command
    list_parser = subparsers.add_parser("list-runs", help="List available Spark runs")
    list_parser.add_argument("--base-dir", required=True, help="Base directory for the analyzer system")
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command == "setup":
        setup_directories(args.base_dir)
        return 0
    
    elif args.command == "create-sample":
        create_sample_pdfs(args.base_dir, args.run_id)
        return 0
    
    elif args.command == "analyze":
        return asyncio.run(run_analysis(args.run_id, args.base_dir, args.model, args.temperature))
    
    elif args.command == "list-runs":
        runs_dir = Path(args.base_dir) / "runs"
        if not runs_dir.exists():
            print(f"Runs directory does not exist: {runs_dir}")
            return 1
        
        runs = [d.name for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("spark_run_")]
        
        if not runs:
            print("No Spark runs found.")
            return 0
        
        print("Available Spark runs:")
        for run in sorted(runs):
            run_id = run.replace("spark_run_", "")
            output_dir = Path(args.base_dir) / "outputs" / run
            status_file = output_dir / "status.json"
            
            status = "Not analyzed"
            if output_dir.exists():
                if (output_dir / "final_report.md").exists():
                    status = "Analysis complete"
                elif (output_dir / "error_report.md").exists():
                    status = "Analysis failed"
            
            print(f"  - {run_id} ({status})")
        
        return 0
    
    else:
        parser.print_help()
        return 1

if __name__ == "__main__":
    sys.exit(main())
