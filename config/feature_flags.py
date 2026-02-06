"""
Feature Flag Configuration

Controls gradual rollout of new El-Harezmi pipeline.
Enables A/B testing and safe production deployment.
"""

import os
import random
from enum import Enum
from typing import Optional
from dataclasses import dataclass


class PipelineVersion(Enum):
    """Available pipeline versions"""
    LEGACY = "legacy"        # Old RAGEngine (ChromaDB)
    EL_HAREZMI = "el_harezmi"  # New 5-stage pipeline (Qdrant)


@dataclass
class FeatureFlags:
    """Feature flag configuration"""
    
    # El-Harezmi pipeline rollout percentage (0.0 - 1.0)
    # 0.0 = all traffic to legacy
    # 0.5 = 50% to each
    # 1.0 = all traffic to El-Harezmi
    el_harezmi_rollout: float = 1.0  # FULL ROLLOUT - El-Harezmi is now default
    
    # Force specific pipeline (for testing)
    # Set to "legacy" or "el_harezmi" to override rollout
    force_pipeline: Optional[str] = None
    
    # Enable/disable specific El-Harezmi stages
    enable_kg_validation: bool = True  # Stage 4
    enable_llm_extraction: bool = True  # Stage 3
    
    # A/B testing user groups
    ab_test_enabled: bool = False
    ab_test_group_a: str = "legacy"
    ab_test_group_b: str = "el_harezmi"
    
    # Fallback behavior
    fallback_to_legacy_on_error: bool = True
    
    # Logging
    log_pipeline_selection: bool = True
    
    def get_pipeline_for_request(
        self,
        user_id: Optional[str] = None,
        force_version: Optional[str] = None
    ) -> PipelineVersion:
        """
        Determine which pipeline to use for a request.
        
        Args:
            user_id: Optional user identifier for consistent routing
            force_version: Optional override from request
            
        Returns:
            PipelineVersion to use
        """
        # Check force override from request
        if force_version:
            if force_version.lower() == "legacy":
                return PipelineVersion.LEGACY
            elif force_version.lower() in ["el_harezmi", "elharezmi", "v2"]:
                return PipelineVersion.EL_HAREZMI
        
        # Check global force override
        if self.force_pipeline:
            if self.force_pipeline.lower() == "legacy":
                return PipelineVersion.LEGACY
            elif self.force_pipeline.lower() in ["el_harezmi", "elharezmi", "v2"]:
                return PipelineVersion.EL_HAREZMI
        
        # A/B testing with consistent user routing
        if self.ab_test_enabled and user_id:
            # Use user_id hash for consistent routing
            user_hash = hash(user_id) % 100
            if user_hash < 50:
                return PipelineVersion.LEGACY if self.ab_test_group_a == "legacy" else PipelineVersion.EL_HAREZMI
            else:
                return PipelineVersion.LEGACY if self.ab_test_group_b == "legacy" else PipelineVersion.EL_HAREZMI
        
        # Percentage-based rollout
        if self.el_harezmi_rollout >= 1.0:
            return PipelineVersion.EL_HAREZMI
        elif self.el_harezmi_rollout <= 0.0:
            return PipelineVersion.LEGACY
        else:
            # Random selection based on rollout percentage
            if random.random() < self.el_harezmi_rollout:
                return PipelineVersion.EL_HAREZMI
            else:
                return PipelineVersion.LEGACY


# Global feature flags instance
_feature_flags: Optional[FeatureFlags] = None


def get_feature_flags() -> FeatureFlags:
    """Get or create feature flags singleton"""
    global _feature_flags
    
    if _feature_flags is None:
        _feature_flags = FeatureFlags(
            # Load from environment
            el_harezmi_rollout=float(os.environ.get("EL_HAREZMI_ROLLOUT", "1.0")),  # Default: El-Harezmi enabled
            force_pipeline=os.environ.get("FORCE_PIPELINE"),
            enable_kg_validation=os.environ.get("ENABLE_KG_VALIDATION", "true").lower() == "true",
            enable_llm_extraction=os.environ.get("ENABLE_LLM_EXTRACTION", "true").lower() == "true",
            ab_test_enabled=os.environ.get("AB_TEST_ENABLED", "false").lower() == "true",
            fallback_to_legacy_on_error=os.environ.get("FALLBACK_ON_ERROR", "true").lower() == "true",
            log_pipeline_selection=os.environ.get("LOG_PIPELINE_SELECTION", "true").lower() == "true",
        )
    
    return _feature_flags


def set_feature_flags(flags: FeatureFlags):
    """Set feature flags (for testing)"""
    global _feature_flags
    _feature_flags = flags


def set_rollout_percentage(percentage: float):
    """Quick method to update rollout percentage"""
    flags = get_feature_flags()
    flags.el_harezmi_rollout = max(0.0, min(1.0, percentage))


# Convenience functions
def is_el_harezmi_enabled() -> bool:
    """Check if El-Harezmi is enabled at all"""
    flags = get_feature_flags()
    return flags.el_harezmi_rollout > 0.0 or flags.force_pipeline in ["el_harezmi", "elharezmi", "v2"]


def get_pipeline_version(user_id: Optional[str] = None, force: Optional[str] = None) -> PipelineVersion:
    """Get pipeline version for a request"""
    return get_feature_flags().get_pipeline_for_request(user_id, force)
