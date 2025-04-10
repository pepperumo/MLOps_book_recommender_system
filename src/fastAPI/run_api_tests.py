#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Wrapper script to run API tests and generate HTML reports.

This script runs the pytest-based tests in test_api_pytest.py
and automatically generates an HTML report in the reports/fastAPI folder.

Usage:
    python run_api_tests.py [options]

Options:
    --skip-stress      Skip stress tests (faster execution)
    --include-benchmarks   Include benchmark tests
    --api-url URL      Specify custom API URL (default: http://localhost:9998)
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime

# Set up project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
reports_dir = os.path.join(project_root, 'reports', 'fastAPI')
os.makedirs(reports_dir, exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description="Run API tests and generate HTML report")
    parser.add_argument("--skip-stress", action="store_true", help="Skip stress tests")
    parser.add_argument("--include-benchmarks", action="store_true", help="Include benchmark tests")
    parser.add_argument("--api-url", help="Specify custom API URL", default=None)
    
    args = parser.parse_args()
    
    # Create unique report filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(reports_dir, f"test_api_report_{timestamp}.html")
    
    # Build pytest command
    pytest_cmd = ["pytest", "test_api_pytest.py", "-v"]
    
    # Add HTML report generation
    pytest_cmd.extend(["--html", report_path, "--self-contained-html"])
    
    # Skip stress tests if requested
    if args.skip_stress:
        pytest_cmd.extend(["-k", "not stress"])
        os.environ["SKIP_STRESS_TEST"] = "true"
    
    # Handle benchmark tests
    if args.include_benchmarks:
        # No special flag needed, benchmarks will run if pytest-benchmark is installed
        pass
    else:
        # Skip benchmark tests
        pytest_cmd.extend(["-k", "not benchmark"])
    
    # Set custom API URL if provided
    if args.api_url:
        os.environ["API_URL"] = args.api_url
    
    print(f"Running API tests with command: {' '.join(pytest_cmd)}")
    print(f"HTML report will be generated at: {report_path}")
    
    # Run pytest with the specified options
    result = subprocess.run(pytest_cmd)
    
    # Print report location after completion
    if os.path.exists(report_path):
        print(f"\nTest execution completed. HTML report available at:")
        print(f"  {report_path}")
        
        # On Windows, provide a more accessible local path
        local_path = os.path.relpath(report_path)
        print(f"  (Local path: {local_path})")
    else:
        print("\nTest execution completed, but HTML report was not generated successfully.")
    
    # Return the exit code from pytest
    return result.returncode

if __name__ == "__main__":
    sys.exit(main())