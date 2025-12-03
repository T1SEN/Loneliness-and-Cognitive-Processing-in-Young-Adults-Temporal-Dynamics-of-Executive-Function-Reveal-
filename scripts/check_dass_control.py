"""
DASS Control Verification Script
=================================

Verifies that Gold Standard analysis scripts include proper DASS-21 covariate control.

Checks for:
- z_dass_dep (depression)
- z_dass_anx (anxiety)
- z_dass_str (stress)
- z_age (age covariate)
- Gender interaction (z_ucla * C(gender_male))

Usage:
    python scripts/check_dass_control.py
    python scripts/check_dass_control.py --path analysis/my_script.py
    python scripts/check_dass_control.py --gold-standard-only

Exit codes:
    0: All checks passed
    1: Violations found
"""

from __future__ import annotations

import sys
import re
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
from dataclasses import dataclass


# Required DASS control terms
REQUIRED_TERMS = [
    'z_dass_dep',
    'z_dass_anx',
    'z_dass_str',
    'z_age',
]

# Pattern for gender interaction
GENDER_INTERACTION_PATTERNS = [
    r'z_ucla\s*\*\s*C\(gender_male\)',
    r'C\(gender_male\)\s*\*\s*z_ucla',
    r'z_ucla\s*:\s*C\(gender_male\)',
]

# Gold Standard scripts that MUST have DASS control
GOLD_STANDARD_SCRIPTS = [
    'master_dass_controlled_analysis.py',
    'prp_comprehensive_dass_controlled.py',
    'loneliness_exec_models.py',
    'trial_level_mixed_effects.py',
    'prp_exgaussian_dass_controlled.py',
    'gold_standard/pipeline.py',
]


@dataclass
class ValidationResult:
    """Result of validating a single file."""
    filepath: Path
    formulas_found: List[str]
    missing_terms: Dict[str, List[str]]  # formula -> missing terms
    has_gender_interaction: Dict[str, bool]  # formula -> has interaction
    is_valid: bool
    error: str = None


def extract_formulas(content: str) -> List[Tuple[str, int]]:
    """
    Extract regression formulas from Python code.

    Returns list of (formula, line_number) tuples.
    """
    formulas = []

    # Pattern for formula strings in smf.ols() calls
    patterns = [
        # smf.ols("formula", ...)
        r'smf\.ols\s*\(\s*["\']([^"\']+)["\']',
        # formula = "..."
        r'formula\s*=\s*["\']([^"\']+)["\']',
        # f"..." format strings with formula
        r'f["\']([^"\']*~[^"\']*)["\']',
    ]

    lines = content.split('\n')
    for i, line in enumerate(lines, 1):
        for pattern in patterns:
            matches = re.findall(pattern, line)
            for match in matches:
                if '~' in match:  # Must be a formula
                    formulas.append((match, i))

    return formulas


def check_formula(formula: str) -> Tuple[List[str], bool]:
    """
    Check if formula has required DASS controls.

    Returns (missing_terms, has_gender_interaction).
    """
    missing = []

    for term in REQUIRED_TERMS:
        if term not in formula:
            missing.append(term)

    # Check for gender interaction
    has_interaction = any(
        re.search(pattern, formula)
        for pattern in GENDER_INTERACTION_PATTERNS
    )

    # If formula has z_ucla but no gender interaction, flag it
    if 'z_ucla' in formula and not has_interaction:
        missing.append('UCLA×Gender interaction')

    return missing, has_interaction


def validate_file(filepath: Path) -> ValidationResult:
    """Validate a single Python file for DASS control."""
    try:
        content = filepath.read_text(encoding='utf-8')
    except Exception as e:
        return ValidationResult(
            filepath=filepath,
            formulas_found=[],
            missing_terms={},
            has_gender_interaction={},
            is_valid=False,
            error=str(e)
        )

    formulas = extract_formulas(content)

    if not formulas:
        # No formulas found - could be a utility script
        return ValidationResult(
            filepath=filepath,
            formulas_found=[],
            missing_terms={},
            has_gender_interaction={},
            is_valid=True
        )

    missing_terms = {}
    has_interaction = {}
    all_valid = True

    for formula, line_num in formulas:
        missing, interaction = check_formula(formula)
        formula_key = f"L{line_num}: {formula[:50]}..."

        if missing:
            missing_terms[formula_key] = missing
            all_valid = False

        has_interaction[formula_key] = interaction

    return ValidationResult(
        filepath=filepath,
        formulas_found=[f[0] for f in formulas],
        missing_terms=missing_terms,
        has_gender_interaction=has_interaction,
        is_valid=all_valid
    )


def validate_directory(
    directory: Path,
    gold_standard_only: bool = False
) -> List[ValidationResult]:
    """Validate all Python files in directory."""
    results = []

    if gold_standard_only:
        files = [directory / s for s in GOLD_STANDARD_SCRIPTS if (directory / s).exists()]
    else:
        files = list(directory.glob('**/*.py'))

    for filepath in sorted(files):
        # Skip __pycache__ and test files
        if '__pycache__' in str(filepath) or 'test_' in filepath.name:
            continue

        result = validate_file(filepath)
        results.append(result)

    return results


def print_results(results: List[ValidationResult], verbose: bool = True) -> int:
    """Print validation results and return exit code."""
    violations = [r for r in results if not r.is_valid]
    valid = [r for r in results if r.is_valid and r.formulas_found]

    print("=" * 70)
    print("DASS CONTROL VERIFICATION REPORT")
    print("=" * 70)

    print(f"\nFiles scanned: {len(results)}")
    print(f"Files with formulas: {len([r for r in results if r.formulas_found])}")
    print(f"Valid: {len(valid)}")
    print(f"Violations: {len(violations)}")

    if violations:
        print("\n" + "=" * 70)
        print("VIOLATIONS FOUND")
        print("=" * 70)

        for result in violations:
            print(f"\n{result.filepath}")

            if result.error:
                print(f"  ERROR: {result.error}")
                continue

            for formula_key, missing in result.missing_terms.items():
                print(f"  {formula_key}")
                print(f"    Missing: {', '.join(missing)}")

        print("\n" + "-" * 70)
        print("REQUIRED FORMULA TEMPLATE:")
        print("-" * 70)
        print('  "{outcome} ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"')
        print()

        return 1

    print("\n✓ All Gold Standard scripts have proper DASS control.")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Verify DASS control in analysis scripts")
    parser.add_argument('--path', '-p', type=Path, default=None,
                       help='Specific file to check')
    parser.add_argument('--directory', '-d', type=Path, default=Path('analysis'),
                       help='Directory to scan (default: analysis/)')
    parser.add_argument('--gold-standard-only', '-g', action='store_true',
                       help='Only check Gold Standard scripts')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Minimal output')

    args = parser.parse_args()

    if args.path:
        results = [validate_file(args.path)]
    else:
        results = validate_directory(args.directory, args.gold_standard_only)

    exit_code = print_results(results, verbose=not args.quiet)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
