"""
Validity & Reliability Analysis Suite
=====================================

Online experiment psychometric validation for academic publication.

Modules:
- reliability_suite: Internal consistency (Cronbach's alpha) and split-half reliability
- validity_suite: Factor analysis, convergent/discriminant validity
- data_quality_suite: Response time validation, careless responding detection

Usage:
    python -m publication.validity_reliability.reliability_suite
    python -m publication.validity_reliability.validity_suite
    python -m publication.validity_reliability.data_quality_suite
"""

from pathlib import Path

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent.parent / "results" / "analysis_outputs" / "validity_reliability"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
