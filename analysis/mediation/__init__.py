"""
Mediation Analysis Suite
========================

Bootstrap mediation analyses where DASS is the MEDIATOR (not covariate).

NOTE: In mediation analysis, DASS is tested as a mechanism through which
UCLA affects executive function. This is methodologically different from
confirmatory analyses where DASS is controlled as a covariate.

Pathway: UCLA → DASS → Executive Function

Available analyses:
- dass_mediation: Bootstrap mediation with DASS subscales as mediators
- gender_stratified: Separate male/female mediation models
- moderated_mediation: Conditional indirect effects by gender
"""

__all__ = ['mediation_suite']
