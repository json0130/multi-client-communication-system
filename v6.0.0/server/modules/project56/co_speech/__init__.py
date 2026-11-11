"""Alternative co-speech segmentation implementations and shared models."""

from modules.project56.co_speech.atomic_speech_unitizer import AtomicSpeechUnitizer
from modules.project56.co_speech.duration_estimator import SpeechDurationEstimator
from modules.project56.co_speech.llm_speech_segmenter import LLMSpeechSegmenter
from modules.project56.co_speech.lookahead import (
    LookaheadPlanningTiming,
    LookaheadRunResult,
    OneSegmentLookaheadCoordinator,
    PreparedPlanSegment,
    SegmentExecutor,
)
from modules.project56.co_speech.models import (
    AtomicSpeechUnit,
    LinearSpeechDurationProfile,
    SpeechSegment,
    SpeechTimingProfile,
)
from modules.project56.co_speech.navel_executor import (
    DEFAULT_VARIANTS,
    DefaultVariantSelector,
    DirectNavelWebSocketTransport,
    NavelSegmentExecutionRecord,
    NavelSegmentExecutor,
    NavelSegmentTransport,
)
from modules.project56.co_speech.plan import (
    CoSpeechPlan,
    CoSpeechPlanBuilder,
    PlannedSegment,
    SelectionStatus,
)
from modules.project56.co_speech.plan_orchestrator import CoSpeechPlanOrchestrator
from modules.project56.co_speech.profiles import (
    NAVEL_LINEAR_SPEECH_DURATION_PROFILE,
    NAVEL_SPEECH_TIMING_PROFILE,
)
from modules.project56.co_speech.speech_segmenter import (
    SegmentationTimingPolicy,
    SpeechSegmenter,
)

__all__ = [
    "AtomicSpeechUnit",
    "AtomicSpeechUnitizer",
    "CoSpeechPlan",
    "CoSpeechPlanBuilder",
    "CoSpeechPlanOrchestrator",
    "DEFAULT_VARIANTS",
    "DefaultVariantSelector",
    "LinearSpeechDurationProfile",
    "NAVEL_LINEAR_SPEECH_DURATION_PROFILE",
    "NAVEL_SPEECH_TIMING_PROFILE",
    "NavelSegmentExecutionRecord",
    "NavelSegmentExecutor",
    "NavelSegmentTransport",
    "PlannedSegment",
    "SelectionStatus",
    "LLMSpeechSegmenter",
    "LookaheadPlanningTiming",
    "LookaheadRunResult",
    "OneSegmentLookaheadCoordinator",
    "PreparedPlanSegment",
    "SegmentExecutor",
    "SpeechDurationEstimator",
    "SpeechSegment",
    "SpeechSegmenter",
    "SegmentationTimingPolicy",
    "SpeechTimingProfile",
    "DirectNavelWebSocketTransport",
]
