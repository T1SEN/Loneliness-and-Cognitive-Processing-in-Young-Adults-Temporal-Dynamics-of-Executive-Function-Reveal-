"""
Synthesis Analysis Suite
========================

Integration and summary analyses for UCLA × Executive Function research.

Modules:
- synthesis_suite: Group comparisons, forest plots, correlations, variance tests

Usage:
    from analysis.synthesis import synthesis_suite
    synthesis_suite.run()  # Run all
    synthesis_suite.run('forest_plot')  # Specific analysis
"""

from . import synthesis_suite

__all__ = ['synthesis_suite']
