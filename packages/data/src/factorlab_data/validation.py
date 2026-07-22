"""Validation of factor panels.

Data quality is where quantitative research most often goes wrong: a silent gap,
a stale vintage, a percentage left un-normalized, or a duplicated date will
quietly corrupt every downstream regression.  :class:`FactorValidator` turns
those failure modes into explicit, severity-tagged findings.

Validation is intentionally *separate* from parsing so it can be applied to any
panel regardless of provenance, and so callers choose the strictness
(``assert_valid`` raises; ``validate`` returns a report to inspect).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from factorlab_data.errors import FactorValidationError
from factorlab_data.panel import FactorPanel

__all__ = ["FactorValidator", "Severity", "ValidationIssue", "ValidationReport"]


class Severity:
    """Issue severities.  ``ERROR`` makes a report invalid; ``WARNING`` does not."""

    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    severity: str
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Outcome of validating a panel: an ordered list of issues."""

    issues: tuple[ValidationIssue, ...] = ()

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(i for i in self.issues if i.severity == Severity.ERROR)

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        return tuple(i for i in self.issues if i.severity == Severity.WARNING)

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        if not self.issues:
            return "OK (no issues)"
        return "\n".join(f"[{i.severity.upper()}] {i.code}: {i.message}" for i in self.issues)


# Expected spacing (in days) between consecutive observations, per frequency.
_EXPECTED_SPACING = {
    "daily": (1, 5),      # 1-5 calendar days (weekends/holidays)
    "weekly": (5, 10),
    "monthly": (28, 31),
    "annual": (365, 366),
}


@dataclass(slots=True)
class FactorValidator:
    """Configurable validator for :class:`FactorPanel`.

    Parameters
    ----------
    allow_missing:
        If False, any NaN in the values is an error.  If True (default), missing
        values are permitted (models perform listwise deletion downstream).
    max_abs_return:
        Absolute per-period return above which a value is flagged (in decimal
        units; e.g. ``1.5`` = 150%).  Catches un-normalized percentages.
    check_frequency_spacing:
        If True, verify observed date spacing matches the declared frequency.
    flag_constant:
        If True, warn on any factor with (near-)zero variance.
    """

    allow_missing: bool = True
    max_abs_return: float = 1.5
    check_frequency_spacing: bool = True
    flag_constant: bool = True

    def validate(self, panel: FactorPanel) -> ValidationReport:
        issues: list[ValidationIssue] = []
        issues.extend(self._check_shape(panel))
        issues.extend(self._check_dates(panel))
        issues.extend(self._check_values(panel))
        if self.check_frequency_spacing:
            issues.extend(self._check_frequency(panel))
        return ValidationReport(tuple(issues))

    def assert_valid(self, panel: FactorPanel) -> None:
        """Raise :class:`FactorValidationError` if the panel has any error."""
        report = self.validate(panel)
        if not report.is_valid:
            raise FactorValidationError(
                "Factor panel failed validation:\n" + report.summary()
            )

    # ------------------------------------------------------------------ #
    # Individual checks                                                   #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _check_shape(panel: FactorPanel) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        if panel.n_observations == 0:
            issues.append(
                ValidationIssue(Severity.ERROR, "empty", "Panel has no observations.")
            )
        if panel.n_factors == 0:
            issues.append(
                ValidationIssue(Severity.ERROR, "no_factors", "Panel has no factors.")
            )
        if len(set(panel.factor_names)) != len(panel.factor_names):
            issues.append(
                ValidationIssue(
                    Severity.ERROR, "duplicate_factor", "Duplicate factor names present."
                )
            )
        return issues

    @staticmethod
    def _check_dates(panel: FactorPanel) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        dates = panel.dates
        if dates.shape[0] < 2:
            return issues
        diffs = np.diff(dates)
        if np.any(diffs <= np.timedelta64(0, "D")):
            issues.append(
                ValidationIssue(
                    Severity.ERROR,
                    "date_order",
                    "Dates are not strictly increasing (duplicate or unsorted).",
                )
            )
        return issues

    def _check_values(self, panel: FactorPanel) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        values = panel.values
        finite = np.isfinite(values)

        if not self.allow_missing and not finite.all():
            n_missing = int((~finite).sum())
            issues.append(
                ValidationIssue(
                    Severity.ERROR, "missing", f"{n_missing} missing value(s) present."
                )
            )

        for j, name in enumerate(panel.factor_names):
            col = values[:, j]
            col_finite = col[np.isfinite(col)]
            if col_finite.size == 0:
                issues.append(
                    ValidationIssue(
                        Severity.ERROR, "all_missing", f"Factor {name!r} is entirely missing."
                    )
                )
                continue
            if self.flag_constant and np.var(col_finite) <= 1e-14:
                issues.append(
                    ValidationIssue(
                        Severity.WARNING, "constant", f"Factor {name!r} is constant."
                    )
                )
            extreme = np.abs(col_finite) > self.max_abs_return
            if np.any(extreme):
                worst = float(np.max(np.abs(col_finite)))
                issues.append(
                    ValidationIssue(
                        Severity.WARNING,
                        "extreme_value",
                        f"Factor {name!r} has |value| up to {worst:.3f} "
                        f"(> {self.max_abs_return}); check percent-vs-decimal units.",
                    )
                )
        return issues

    @staticmethod
    def _check_frequency(panel: FactorPanel) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        bounds = _EXPECTED_SPACING.get(panel.frequency)
        if bounds is None or panel.n_observations < 3:
            return issues
        diffs = np.diff(panel.dates).astype("timedelta64[D]").astype(int)
        lo, hi = bounds
        median = float(np.median(diffs))
        if not (lo <= median <= hi):
            issues.append(
                ValidationIssue(
                    Severity.WARNING,
                    "frequency_mismatch",
                    f"Median date spacing {median:.0f}d is outside the expected "
                    f"[{lo}, {hi}]d for declared frequency {panel.frequency!r}.",
                )
            )
        return issues
