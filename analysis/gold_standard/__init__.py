"""
Gold Standard Analysis Pipeline
===============================

Publication-ready confirmatory analyses with mandatory DASS-21 control.

All analyses in this module:
- Control for DASS-21 subscales (depression, anxiety, stress)
- Control for age
- Test UCLA × Gender interactions
- Use HC3 robust standard errors
- Follow standardized output format

Usage:
    from analysis.gold_standard import pipeline
    pipeline.run()  # Run all analyses
    pipeline.run(analyses=['wcst_pe', 'stroop_interference'])  # Run specific analyses
"""

from . import pipeline

__all__ = ['pipeline']
