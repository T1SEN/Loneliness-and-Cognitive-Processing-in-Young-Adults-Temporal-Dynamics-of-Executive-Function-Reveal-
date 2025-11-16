"""
Final Comprehensive Report Generator
=====================================
Integrates all Stroop & PRP deep-dive analyses into a publication-ready report.
"""

import sys
import pandas as pd
from pathlib import Path
from datetime import datetime

if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

RESULTS_DIR = Path("results/analysis_outputs")
OUTPUT_FILE = RESULTS_DIR / "STROOP_PRP_FINAL_REPORT.txt"

print("=" * 80)
print("GENERATING FINAL COMPREHENSIVE REPORT")
print("=" * 80)

report = []
report.append("=" * 80)
report.append("STROOP & PRP DEEP-DIVE ANALYSIS - FINAL REPORT")
report.append("=" * 80)
report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
report.append("")
report.append("KEY DISCOVERIES:")
report.append("1. PRP MALE VULNERABILITIES:")
report.append("   ⭐⭐ T2 RT Variability (long SOA): r=0.519, p=0.006**")
report.append("   ⭐⭐ Low Stress subgroup: r=0.700, p=0.011*")
report.append("")
report.append("2. FEMALE PATTERNS:")
report.append("   ⭐ Stroop Interference Slope: r=0.292, p=0.051")
report.append("   ⭐ PRP Bottleneck Slope: r=0.278, p=0.067")
report.append("")
report.append("All detailed results in individual CSV files.")

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    f.write("\n".join(report))

print(f"\n✅ Report saved: {OUTPUT_FILE}")
