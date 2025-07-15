# Podcast Transcription Pipeline - Source Package
from .utils.logger import setup_logging, get_logger
from .config.settings import Settings
from .core.pipeline_orchestrator import PipelineOrchestrator

__version__ = "1.0.0"
__author__ = "Podcast Pipeline Team"

# Initialize logging by default
setup_logging()
