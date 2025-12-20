"""
Validity & Reliability Analysis
================================

Online experiment psychometric validation for academic publication.

Dataset-specific runners:
- complete_overall/reliability.py, validity.py, data_quality.py
- complete_stroop/reliability.py, validity.py, data_quality.py
- complete_prp/reliability.py, validity.py, data_quality.py
- complete_wcst/reliability.py, validity.py, data_quality.py

Usage:
    python -m publication.validity_reliability.complete_overall.reliability
    python -m publication.validity_reliability.complete_overall.validity
    python -m publication.validity_reliability.complete_overall.data_quality
"""

from pathlib import Path

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent.parent / "results" / "analysis_outputs" / "validity_reliability"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
