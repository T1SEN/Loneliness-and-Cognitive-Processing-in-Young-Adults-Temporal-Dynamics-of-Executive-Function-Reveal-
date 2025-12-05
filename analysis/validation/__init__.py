"""
Validation Analysis Suite
=========================

Methodological validation analyses for UCLA × Executive Function research.

Modules:
- validation_suite: Cross-validation, robust regression, Type M/S error simulation
- power_suite: Power analysis, reliability correction

Usage:
    from analysis.validation import validation_suite
    validation_suite.run()  # Run all
    validation_suite.run('cross_validation')  # Specific analysis
"""

from . import validation_suite

__all__ = ['validation_suite']
