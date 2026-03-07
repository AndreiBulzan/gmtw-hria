"""
Shared constraint checking (language-agnostic)

Re-exports constraint functions so base_metrics doesn't
import directly from gmtw_ro.
"""

from ..gmtw_ro.eval.constraints import check_constraint, CONSTRAINT_FUNCTIONS

__all__ = [
    "check_constraint",
    "CONSTRAINT_FUNCTIONS",
]