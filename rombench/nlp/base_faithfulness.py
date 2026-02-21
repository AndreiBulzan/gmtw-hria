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
        
        if world_type == "travel":
            # Extract locations from plan
            for day_activities in plan.values():
                if isinstance(day_activities, list):
                    entities.update(day_activities)
                    
        elif world_type == "recipe":
            # Extract recipes from plan
            for recipe in plan.values():
                if isinstance(recipe, str) and recipe:
                    entities.add(recipe)
                    
        elif world_type == "schedule":
            # Extract activities from plan
            for activity in plan.values():
                if isinstance(activity, str) and activity:
                    entities.add(activity)
                    
        elif world_type == "fact":
            # Extract answer entities
            answer = plan.get("answer", "")
            if isinstance(answer, str):
                # Split answer into potential entities
                words = answer.split()
                for word in words:
                    if len(word) > 3:  # Only meaningful words
                        entities.add(word)
        
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
        # Generate all forms of the entity
        forms = self.generate_forms(entity.lower())
        
        # Also handle multi-word entities
        if ' ' in entity:
            forms.update(self._generate_multiword_forms(entity))
        
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