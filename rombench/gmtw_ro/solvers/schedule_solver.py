"""
Complete constraint solver for Schedule world instances.

This solver verifies that a schedule instance has at least one valid solution
by checking ALL constraints, including complex interactions like:
- Priority-day restrictions (medium priority can't be on certain days)
- Drop order (must drop low priority before medium, medium before high)
- Slot restrictions (medical appointments in morning only)
- Back-to-back restrictions
"""

from itertools import combinations, product
from typing import Any


class ScheduleSolver:
    """
    Comprehensive constraint solver for Schedule instances.

    Uses backtracking to find at least one valid assignment that satisfies
    all constraints simultaneously.
    """

    def __init__(self, world: Any):
        self.world = world
        self.appointments = world.payload.get("appointments", [])
        self.days_ro = world.payload.get("days_ro", [])
        self.days_en = world.payload.get("days_en", [])
        self.slots_ro = world.payload.get("slots_ro", [])
        self.slots_en = world.payload.get("slots_en", [])

        # Parse all constraints
        self._parse_constraints()

    def _parse_constraints(self):
        """Parse all constraints from the world."""
        self.max_per_day = 2
        self.max_total = len(self.appointments)
        self.keep_high = False
        self.min_spread = 1
        self.no_back_to_back = False
        self.drop_order = False
        self.forbidden_days_for_medium = set()
        self.medical_morning_only = False
        self.medical_keywords = []
        self.allowed_slots_for_medical = []
        self.high_priority_required_slots = []

        for c in self.world.constraints:
            cid = c.id
            params = c.params if hasattr(c, 'params') else {}

            if cid == "C_MAX_PER_DAY":
                self.max_per_day = params.get("max_per_day", 2)

            elif cid == "C_MAX_TOTAL":
                self.max_total = params.get("max_total", len(self.appointments))

            elif cid in ("C_KEEP_HIGH_PRIORITY", "C_KEEP_HIGH"):
                self.keep_high = True

            elif cid == "C_SPREAD":
                self.min_spread = params.get("min_days_with_appointments", 1)

            elif cid == "C_NO_BACK_TO_BACK":
                self.no_back_to_back = True

            elif cid == "C_DROP_ORDER":
                self.drop_order = True

            elif cid in ("C_MED_RESTRICT", "C_PRIORITY_DAY_RESTRICTION"):
                priority = params.get("priority", "medium")
                if priority == "medium":
                    forbidden = params.get("forbidden_days", [])
                    for day in forbidden:
                        self.forbidden_days_for_medium.add(day.lower())

            elif cid == "C_MEDICAL_MORNING":
                self.medical_morning_only = True
                self.medical_keywords = ["control", "medical"]
                self.allowed_slots_for_medical = ["dimineață", "morning"]

            elif cid == "C_SLOT_TYPE_RESTRICTION":
                # Generic slot restriction
                kw = params.get("type_keyword", "").lower()
                if kw:
                    self.medical_keywords.append(kw)
                    self.allowed_slots_for_medical = params.get("allowed_slots", [])

            elif cid in ("C_HIGH_MORNING", "C_PRIORITY_SLOT_RESTRICTION"):
                # High priority appointments must be in specific slot
                priority = params.get("priority", "high")
                required_slots = params.get("required_slots", [])
                if priority == "high":
                    self.high_priority_required_slots = required_slots

    def _get_priority_order(self) -> list[str]:
        """Return priorities in order from lowest to highest."""
        return ["low", "medium", "high"]

    def _appointments_by_priority(self) -> dict[str, list[dict]]:
        """Group appointments by priority."""
        by_priority = {"high": [], "medium": [], "low": []}
        for apt in self.appointments:
            p = apt.get("priority", "low")
            if p in by_priority:
                by_priority[p].append(apt)
        return by_priority

    def _is_valid_subset(self, selected_ids: set[str]) -> bool:
        """
        Check if a subset of appointments satisfies priority drop ordering.

        Rule: If a low priority is kept, all medium and high must be kept.
              If a medium priority is kept, all high must be kept.
        """
        if not self.drop_order:
            return True

        by_priority = self._appointments_by_priority()

        # Get IDs for each priority level
        high_ids = {a.get("id") for a in by_priority["high"]}
        medium_ids = {a.get("id") for a in by_priority["medium"]}
        low_ids = {a.get("id") for a in by_priority["low"]}

        # Check: if any low is kept, all high and medium must be kept
        kept_low = selected_ids & low_ids
        if kept_low:
            if not high_ids.issubset(selected_ids):
                return False
            if not medium_ids.issubset(selected_ids):
                return False

        # Check: if any medium is kept, all high must be kept
        kept_medium = selected_ids & medium_ids
        if kept_medium:
            if not high_ids.issubset(selected_ids):
                return False

        return True

    def _can_place_appointment(self, apt: dict, day: str, slot: str) -> bool:
        """Check if an appointment can be placed in a specific slot."""
        day_lower = day.lower()
        slot_lower = slot.lower()

        # Check priority-day restriction (medium can't be on forbidden days)
        if apt.get("priority") == "medium":
            if day_lower in self.forbidden_days_for_medium:
                return False

        # Check high priority slot restriction (high must be in morning)
        if apt.get("priority") == "high" and self.high_priority_required_slots:
            slot_ok = any(
                allowed.lower() in slot_lower or slot_lower in allowed.lower()
                for allowed in self.high_priority_required_slots
            )
            if not slot_ok:
                return False

        # Check medical/control appointments must be in morning
        apt_name = apt.get("name_ro", "").lower()
        is_medical = any(kw in apt_name for kw in self.medical_keywords)

        if is_medical and self.allowed_slots_for_medical:
            slot_ok = any(
                allowed.lower() in slot_lower or slot_lower in allowed.lower()
                for allowed in self.allowed_slots_for_medical
            )
            if not slot_ok:
                return False

        return True

    def _try_assignment(self, selected: list[dict]) -> bool:
        """
        Try to find a valid slot assignment for the selected appointments.

        Uses backtracking to assign each appointment to a (day, slot) pair.
        """
        if not selected:
            return True

        num_days = len(self.days_ro)
        num_slots = len(self.slots_ro)

        # Check basic feasibility
        if self.no_back_to_back:
            # With no back-to-back, max 1 appointment per day
            if len(selected) > num_days:
                return False
        else:
            # Max max_per_day appointments per day
            if len(selected) > num_days * self.max_per_day:
                return False

        # Check spread requirement
        if self.min_spread > num_days:
            return False

        # Generate all possible slot assignments
        all_slots = []
        for d_idx, day in enumerate(self.days_ro):
            for s_idx, slot in enumerate(self.slots_ro):
                all_slots.append((d_idx, day, s_idx, slot))

        def backtrack(apt_idx: int, assignment: dict, day_counts: dict, days_used: set) -> bool:
            if apt_idx == len(selected):
                # All appointments assigned - check constraints
                if len(days_used) < self.min_spread:
                    return False
                return True

            apt = selected[apt_idx]
            apt_id = apt.get("id")

            for d_idx, day, s_idx, slot in all_slots:
                slot_key = (d_idx, s_idx)

                # Skip if slot already taken
                if slot_key in assignment:
                    continue

                # Check max per day
                if day_counts.get(d_idx, 0) >= self.max_per_day:
                    continue

                # Check no back-to-back
                if self.no_back_to_back and day_counts.get(d_idx, 0) >= 1:
                    continue

                # Check if this appointment can go in this slot
                if not self._can_place_appointment(apt, day, slot):
                    continue

                # Try this assignment
                assignment[slot_key] = apt_id
                new_day_counts = day_counts.copy()
                new_day_counts[d_idx] = new_day_counts.get(d_idx, 0) + 1
                new_days_used = days_used | {d_idx}

                if backtrack(apt_idx + 1, assignment, new_day_counts, new_days_used):
                    return True

                # Backtrack
                del assignment[slot_key]

            return False

        return backtrack(0, {}, {}, set())

    def solve(self) -> bool:
        """
        Find at least one valid solution.

        Returns True if a valid solution exists, False otherwise.
        """
        if not self.appointments:
            return True

        by_priority = self._appointments_by_priority()
        high_apts = by_priority["high"]
        medium_apts = by_priority["medium"]
        low_apts = by_priority["low"]

        # If we must keep high priority, check if they fit
        if self.keep_high and len(high_apts) > self.max_total:
            return False

        # Determine which appointments must be kept
        must_keep_ids = set()
        if self.keep_high:
            must_keep_ids = {a.get("id") for a in high_apts}

        # Try different subsets, respecting priority ordering
        # Start with all appointments and progressively drop
        all_ids = [a.get("id") for a in self.appointments]

        # Generate valid subsets (respecting drop order if enabled)
        def generate_valid_subsets():
            """Generate subsets that respect priority drop ordering."""
            # If drop_order is enabled, we can only drop in order: low first, then medium
            # High must always be kept if keep_high is enabled

            if self.drop_order:
                # With drop order: try dropping low first, then medium
                # Never drop high if keep_high

                for num_low_to_keep in range(len(low_apts), -1, -1):
                    for num_med_to_keep in range(len(medium_apts), -1, -1):
                        # Can only drop medium if all low are dropped
                        if num_med_to_keep < len(medium_apts) and num_low_to_keep > 0:
                            continue

                        for low_combo in combinations(range(len(low_apts)), num_low_to_keep):
                            for med_combo in combinations(range(len(medium_apts)), num_med_to_keep):
                                selected = list(high_apts)  # Always keep high
                                selected.extend(medium_apts[i] for i in med_combo)
                                selected.extend(low_apts[i] for i in low_combo)

                                # Check max_total
                                if len(selected) <= self.max_total:
                                    yield selected
            else:
                # Without drop order: try all combinations
                for size in range(self.max_total, 0, -1):
                    for combo in combinations(self.appointments, size):
                        selected = list(combo)
                        selected_ids = {a.get("id") for a in selected}

                        # Check must_keep
                        if not must_keep_ids.issubset(selected_ids):
                            continue

                        yield selected

        for selected in generate_valid_subsets():
            selected_ids = {a.get("id") for a in selected}

            # Check must_keep constraint
            if not must_keep_ids.issubset(selected_ids):
                continue

            # Check if valid subset (priority ordering)
            if not self._is_valid_subset(selected_ids):
                continue

            # Try to find valid slot assignment
            if self._try_assignment(selected):
                return True

        return False


def solve_schedule(world: Any) -> bool:
    """
    Convenience function to check if a schedule world is solvable.

    Args:
        world: The schedule World instance

    Returns:
        True if at least one valid solution exists
    """
    solver = ScheduleSolver(world)
    return solver.solve()
