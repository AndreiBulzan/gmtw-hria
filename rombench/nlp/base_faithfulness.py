"""
Base faithfulness computation (language-agnostic foundation)

Provides common logic for checking if explanation mentions plan entities.
Language-specific implementations add morphology, inflection, etc.
"""

from abc import ABC, abstractmethod
from typing import Any, Set


class BaseFaithfulness(ABC):
    """
    Base class for faithfulness checking.
    
    Subclasses implement language-specific:
    - Text normalization
    - Morphological forms generation
    - Entity matching strategies
    """
    
    @abstractmethod
    def normalize_text(self, text: str) -> str:
        """Normalize text for matching"""
        pass
    
    @abstractmethod
    def generate_forms(self, token: str) -> Set[str]:
        """
        Generate morphological forms of a token.
        
        For English: might just return {token, token+"s", token+"ed", ...}
        For Romanian: generates all inflected forms
        """
        pass
    
    def extract_entities(self, world: Any, plan: dict) -> Set[str]:
        """
        Extract unique entities from the plan.
        
        This is language-agnostic - just extracts strings from the plan.
        """
        entities = set()
        world_type = world.world_type

        # plan may be a dict (expected) or a list/other (malformed). Create
        # a safe iterator over its entries and a flag for mapping access.
        is_mapping = isinstance(plan, dict)
        entries_iter = plan.values() if is_mapping else (plan if isinstance(plan, (list, tuple)) else [])

        if world_type == "travel":
            # Extract locations/activities from plan
            for day_activities in entries_iter:
                if isinstance(day_activities, list):
                    # day_activities may be a list of strings, dicts or nested lists.
                    for item in day_activities:
                        if isinstance(item, str) and item:
                            entities.add(item)
                        elif isinstance(item, (list, tuple)):
                            for sub in item:
                                if isinstance(sub, str) and sub:
                                    entities.add(sub)
                                else:
                                    # fallback: stringify small atomic items
                                    if not isinstance(sub, (list, dict, tuple)):
                                        entities.add(str(sub))
                        elif isinstance(item, dict):
                            # try to extract string values from dict
                            for v in item.values():
                                if isinstance(v, str) and v:
                                    entities.add(v)
                                else:
                                    if not isinstance(v, (list, dict, tuple)):
                                        entities.add(str(v))
                        else:
                            # fallback: stringify small atomic items
                            if not isinstance(item, (list, dict, tuple)) and item is not None:
                                entities.add(str(item))
                    
        elif world_type == "recipe":
            # Extract recipes from plan (expecting mapping of keys->strings)
            for recipe in entries_iter:
                if isinstance(recipe, str) and recipe:
                    entities.add(recipe)
                    
        elif world_type == "schedule":
            # Extract activities from plan (mapping of slot->string) or
            # fallback to list of activities
            for activity in entries_iter:
                if isinstance(activity, str) and activity:
                    entities.add(activity)
                    
        elif world_type == "fact":
            # Safe access to 'answer' when plan may not be a mapping
            if is_mapping:
                answer = plan.get("answer", "")
            elif isinstance(plan, str):
                answer = plan
            else:
                answer = ""
            if isinstance(answer, str) and answer.strip():
                entities.add(answer.strip())
                parts = [p.strip() for p in answer.split(",") if p.strip()]
                for part in parts:
                    if part and len(part) > 3:
                        entities.add(part)
        
        return entities
    
    def check_entity_mentioned(
        self, 
        entity: str, 
        normalized_text: str
    ) -> bool:
        """
        Check if entity is mentioned in text.
        
        Uses morphological forms to allow for inflections.
        """
        # Normalize the entity before generating forms, so that morphological
        # variants are compared against the already-normalized text (diacritics
        # stripped, punctuation removed, etc.)
        normalized_entity = self.normalize_text(entity)
        forms = self.generate_forms(normalized_entity)

        # Also handle multi-word entities
        if ' ' in entity:
            for phrase_form in self._generate_multiword_forms(entity):
                forms.add(self.normalize_text(phrase_form))

        # Check if any form appears in text
        for form in forms:
            if form in normalized_text:
                return True

        return False
    
    def _generate_multiword_forms(self, phrase: str) -> Set[str]:
        """
        Generate forms for multi-word phrases.
        
        Base implementation just returns the phrase itself.
        Subclasses can add language-specific logic.
        """
        return {phrase.lower()}
    
    def compute_faithfulness(
        self,
        world: Any,
        plan: dict,
        explanation: str,
        **kwargs
    ) -> dict[str, Any]:
        """
        Compute faithfulness score.
        
        Checks what fraction of plan entities are mentioned in explanation.
        """
        if not plan or not explanation:
            return {
                "F": 0.0,
                "entities_total": 0,
                "entities_mentioned": 0,
                "missing_entities": [],
            }
        
        # Extract entities from plan
        entities = self.extract_entities(world, plan)
        
        if not entities:
            return {
                "F": 1.0,
                "entities_total": 0,
                "entities_mentioned": 0,
                "missing_entities": [],
                "note": "No entities to check",
            }
        
        # Normalize explanation text
        normalized_text = self.normalize_text(explanation)
        
        # Check each entity
        mentioned = []
        missing = []
        
        for entity in entities:
            if self.check_entity_mentioned(entity, normalized_text):
                mentioned.append(entity)
            else:
                missing.append(entity)
        
        # Compute score
        total = len(entities)
        mentioned_count = len(mentioned)
        F_raw = mentioned_count / total if total > 0 else 1.0
        
        # Apply severity exponent (like U score)
        severity_exponent = kwargs.get('severity_exponent', 3.0)
        F = F_raw ** severity_exponent
        
        return {
            "F": F,
            "F_linear": F_raw,
            "entities_total": total,
            "entities_mentioned": mentioned_count,
            "mentioned_entities": mentioned,
            "missing_entities": missing,
        }