"""
Canonical entity matcher for GMTW-Ro

Matches entity mentions in natural language text to world entities using fuzzy matching.
"""

import re
from dataclasses import dataclass
from typing import Optional, Any
from rapidfuzz import fuzz, process


# Romanian negation patterns
NEGATION_PATTERNS = [
    r'\bnu\b',
    r'\bfără\b',
    r'\bevit',  # evităm, evitat, evitate
    r'\bnu\s+includ',  # nu includem, nu include
    r'\bnu\s+merg',  # nu mergem, nu merge
    r'\bnu\s+vizit',  # nu vizităm, nu vizitează
    r'\bnu\s+vom\s+vizita\b',
    r'\bexclus',  # exclud, exclusă, excluse, exclude
    r'\ba\s+fost\s+exclus',  # a fost exclusă/exclus
    r'\bnu\s+este\s+inclu',  # nu este inclus/inclusă
    r'\bnu\s+o\s+includ',  # nu o includem, nu o include
    r'\bam\s+ales\s+să\s+nu',  # am ales să nu
    r'\bam\s+decis\s+să\s+nu',  # am decis să nu
]


@dataclass
class EntityMention:
    """A mention of an entity in text"""
    entity_id: str
    entity_name: str
    positive: bool  # True if mentioned positively, False if negated
    span: tuple[int, int]  # Character span in text
    match_score: float  # Fuzzy match score


class CanonicalMatcher:
    """Matcher for canonical entities in text"""

    def __init__(self, threshold: float = 85.0, context_window: int = 50):
        """
        Initialize matcher

        Args:
            threshold: Minimum fuzzy match score (0-100)
            context_window: Number of characters to check for negation around match
        """
        self.threshold = threshold
        self.context_window = context_window

    def extract_mentions(
        self,
        text: str,
        canonical_entities: dict[str, Any]
    ) -> list[EntityMention]:
        """
        Extract entity mentions from text

        Args:
            text: Input text to search
            canonical_entities: Dictionary of entity_id -> Entity objects

        Returns:
            List of EntityMention objects
        """
        mentions = []

        # Normalize text for matching
        normalized_text = self._normalize_text(text)

        for entity_id, entity in canonical_entities.items():
            # Build search terms (name + aliases)
            search_terms = [entity.name] + entity.aliases

            # Try to find each search term
            for term in search_terms:
                normalized_term = self._normalize_text(term)

                # Find all occurrences using fuzzy matching
                entity_mentions = self._find_fuzzy_matches(
                    normalized_text,
                    normalized_term,
                    text,  # Original text for span calculation
                    entity_id,
                    entity.name,
                )

                mentions.extend(entity_mentions)

        # De-duplicate mentions (keep highest score per entity)
        mentions = self._deduplicate_mentions(mentions)

        return mentions

    def _normalize_text(self, text: str) -> str:
        """Normalize text for matching"""
        # Lowercase
        text = text.lower()

        # Strip Romanian diacritics for more robust matching
        diacritic_map = {
            'ă': 'a',
            'â': 'a',
            'î': 'i',
            'ș': 's',
            'ţ': 't',
            'ț': 't',
        }

        for original, replacement in diacritic_map.items():
            text = text.replace(original, replacement)

        return text

    def _find_fuzzy_matches(
        self,
        haystack: str,
        needle: str,
        original_text: str,
        entity_id: str,
        entity_name: str,
    ) -> list[EntityMention]:
        """Find all fuzzy matches of needle in haystack"""
        mentions = []

        # Use sliding window approach
        needle_len = len(needle)
        window_size = needle_len + 5  # Allow some flexibility

        for i in range(len(haystack) - needle_len + 1):
            window = haystack[i:i + window_size]

            # Compute fuzzy match score
            score = fuzz.partial_ratio(needle, window)

            if score >= self.threshold:
                # Found a match
                span_start = i
                span_end = i + needle_len

                # Check for negation in context
                is_positive = self._check_positivity(
                    original_text,
                    span_start,
                    span_end,
                )

                mentions.append(EntityMention(
                    entity_id=entity_id,
                    entity_name=entity_name,
                    positive=is_positive,
                    span=(span_start, span_end),
                    match_score=score,
                ))

        return mentions

    def _check_positivity(self, text: str, start: int, end: int) -> bool:
        """
        Check if mention is positive or negated

        Args:
            text: Original text
            start: Start position of mention
            end: End position of mention

        Returns:
            True if positive mention, False if negated
        """
        # Extract context window
        context_start = max(0, start - self.context_window)
        context_end = min(len(text), end + self.context_window)
        context = text[context_start:context_end]

        # Check for negation patterns
        for pattern in NEGATION_PATTERNS:
            if re.search(pattern, context, re.IGNORECASE):
                return False

        return True

    def _deduplicate_mentions(self, mentions: list[EntityMention]) -> list[EntityMention]:
        """Remove duplicate mentions, keeping the best match per entity"""
        # Group by entity_id
        by_entity = {}
        for mention in mentions:
            if mention.entity_id not in by_entity:
                by_entity[mention.entity_id] = []
            by_entity[mention.entity_id].append(mention)

        # Keep only the best match per entity
        deduped = []
        for entity_id, entity_mentions in by_entity.items():
            # Sort by score descending
            entity_mentions.sort(key=lambda m: m.match_score, reverse=True)
            deduped.append(entity_mentions[0])

        return deduped


def extract_entity_mentions(
    text: str,
    canonical_entities: dict[str, Any],
    threshold: float = 85.0,
) -> list[EntityMention]:
    """
    Convenience function to extract entity mentions

    Args:
        text: Input text
        canonical_entities: Entity dictionary
        threshold: Fuzzy match threshold

    Returns:
        List of entity mentions
    """
    matcher = CanonicalMatcher(threshold=threshold)
    return matcher.extract_mentions(text, canonical_entities)
