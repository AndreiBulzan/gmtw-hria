"""
Base NLP toolkit interfaces (language-agnostic)

Defines the interface that language-specific toolkits should implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Any


@dataclass
class TextQualityReport:
    """
    Text quality analysis result (language-agnostic structure).
    
    Language-specific toolkits may add additional fields.
    """
    # Core scores (0.0-1.0, higher is better)
    overall_score: float        # Combined quality score
    grammar_score: Optional[float] = None
    style_score: Optional[float] = None
    length_score: Optional[float] = None
    
    # Language-specific scores
    language_specific_scores: dict[str, float] = None
    
    # Basic stats
    total_tokens: int = 0
    total_words: int = 0
    
    # Flags
    is_too_short: bool = False
    
    # Detailed analysis  
    details: Optional[dict] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        result = {
            "overall_score": self.overall_score,
            "total_tokens": self.total_tokens,
            "total_words": self.total_words,
            "is_too_short": self.is_too_short,
        }
        
        if self.grammar_score is not None:
            result["grammar_score"] = self.grammar_score
        if self.style_score is not None:
            result["style_score"] = self.style_score
        if self.length_score is not None:
            result["length_score"] = self.length_score
        if self.language_specific_scores:
            result["language_specific_scores"] = self.language_specific_scores
        if self.details:
            result["details"] = self.details
            
        return result


class BaseNLPToolkit(ABC):
    """
    Base class for language-specific NLP toolkits.
    
    Provides interface for text quality analysis specific to each language.
    """
    
    @abstractmethod
    def analyze(self, text: str) -> TextQualityReport:
        """
        Analyze text quality for the target language.
        
        Args:
            text: Input text to analyze
            
        Returns:
            TextQualityReport with quality metrics
        """
        pass
    
    @abstractmethod
    def compute_g_score(self, text: str) -> dict[str, Any]:
        """
        Compute G (Generation Quality) score for evaluation metrics.
        
        Returns dict with 'G' score and component details.
        """
        pass
    
    @abstractmethod
    def tokenize(self, text: str) -> list[str]:
        """Tokenize text into words"""
        pass
    
    @abstractmethod
    def normalize(self, text: str) -> str:
        """Normalize text for matching"""
        pass


class BaseGrammarChecker(ABC):
    """Base interface for grammar checkers (e.g., LanguageTool)"""
    
    @abstractmethod
    def check(self, text: str) -> dict[str, Any]:
        """
        Check grammar and return analysis.
        
        Returns dict with:
        - score: float (0.0-1.0)
        - errors: list of error details
        - error_count: int
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if grammar checker is available"""
        pass