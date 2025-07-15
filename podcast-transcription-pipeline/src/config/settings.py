"""
Settings and configuration management for the podcast transcription pipeline.
"""
import os
from pathlib import Path
from typing import Dict, Any

class Settings:
    """Configuration settings for the pipeline."""
    
    # Default model configurations
    DEFAULT_WHISPER_MODEL = "large-v3"
    DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1" 
    DEFAULT_EMOTION_TEXT_MODEL = "j-hartmann/emotion-english-distilroberta-base"
    DEFAULT_EMOTION_AUDIO_MODEL = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
    DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Processing defaults
    DEFAULT_CONFIDENCE_THRESHOLD = 0.6
    DEFAULT_MAX_KEYWORDS = 20
    
    # File paths
    DEFAULT_OUTPUT_DIR = "./output"
    DEFAULT_CONFIG_FILE = "config.yaml"
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """Get the default configuration dictionary."""
        return {
            "audio": {
                "sample_rate": 16000,
                "chunk_size": 30,
                "max_file_size": 1000,
                "supported_formats": ['.wav', '.mp3', '.m4a', '.flac', '.ogg']
            },
            "transcription": {
                "model": cls.DEFAULT_WHISPER_MODEL,
                "language": "auto",
                "temperature": 0.0,
                "beam_size": 5,
                "best_of": 5,
                "patience": 1.0
            },
            "diarization": {
                "model": cls.DEFAULT_DIARIZATION_MODEL,
                "min_speakers": 1,
                "max_speakers": 10,
                "clustering_threshold": 0.7
            },
            "emotion": {
                "text_model": cls.DEFAULT_EMOTION_TEXT_MODEL,
                "audio_model": cls.DEFAULT_EMOTION_AUDIO_MODEL,
                "confidence_threshold": 0.5
            },
            "topics": {
                "model": cls.DEFAULT_EMBEDDING_MODEL,
                "clustering_method": "kmeans",
                "min_topic_confidence": 0.3,
                "max_topics_per_block": 3
            },
            "keywords": {
                "method": "tfidf",
                "max_keywords_global": cls.DEFAULT_MAX_KEYWORDS,
                "max_keywords_per_block": 10,
                "min_keyword_score": 0.1
            },
            "processing": {
                "device": "auto",
                "batch_size": 8,
                "num_workers": 4,
                "memory_limit": "8GB"
            },
            "validation": {
                "min_transcript_length": 10,
                "max_empty_blocks": 0.1,
                "min_confidence_threshold": cls.DEFAULT_CONFIDENCE_THRESHOLD
            },
            "output": {
                "base_dir": cls.DEFAULT_OUTPUT_DIR,
                "session_prefix": "session",
                "compression": False,
                "backup_intermediates": True
            }
        }
    
    @classmethod
    def get_model_requirements(cls) -> Dict[str, Dict[str, Any]]:
        """Get model requirements and resource estimates."""
        return {
            "whisper": {
                "small": {"vram_gb": 1, "ram_gb": 2, "disk_gb": 0.5},
                "medium": {"vram_gb": 2, "ram_gb": 4, "disk_gb": 1},
                "large": {"vram_gb": 4, "ram_gb": 8, "disk_gb": 2},
                "large-v2": {"vram_gb": 4, "ram_gb": 8, "disk_gb": 2},
                "large-v3": {"vram_gb": 4, "ram_gb": 8, "disk_gb": 2}
            },
            "diarization": {
                "default": {"vram_gb": 2, "ram_gb": 4, "disk_gb": 1}
            },
            "emotion": {
                "text": {"vram_gb": 1, "ram_gb": 2, "disk_gb": 0.5},
                "audio": {"vram_gb": 2, "ram_gb": 4, "disk_gb": 1}
            }
        }
    
    @classmethod
    def validate_environment(cls) -> Dict[str, Any]:
        """Validate the current environment for pipeline requirements."""
        validation = {
            "status": "ok",
            "warnings": [],
            "errors": [],
            "system_info": {}
        }
        
        # Check Python version
        import sys
        python_version = sys.version_info
        if python_version < (3, 8):
            validation["errors"].append("Python 3.8+ required")
        
        # Check available disk space
        output_dir = Path(cls.DEFAULT_OUTPUT_DIR)
        output_dir.mkdir(exist_ok=True)
        
        try:
            import shutil
            free_space_gb = shutil.disk_usage(output_dir).free / (1024**3)
            validation["system_info"]["free_disk_gb"] = free_space_gb
            
            if free_space_gb < 5:
                validation["warnings"].append("Low disk space (< 5GB)")
        except Exception:
            validation["warnings"].append("Could not check disk space")
        
        # Check memory
        try:
            import psutil
            memory = psutil.virtual_memory()
            total_ram_gb = memory.total / (1024**3)
            available_ram_gb = memory.available / (1024**3)
            
            validation["system_info"]["total_ram_gb"] = total_ram_gb
            validation["system_info"]["available_ram_gb"] = available_ram_gb
            
            if available_ram_gb < 4:
                validation["warnings"].append("Low available RAM (< 4GB)")
                
        except ImportError:
            validation["warnings"].append("Could not check system memory")
        
        # Check GPU availability
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                validation["system_info"]["gpu_available"] = True
                validation["system_info"]["gpu_count"] = gpu_count
                validation["system_info"]["gpu_memory_gb"] = gpu_memory
            else:
                validation["system_info"]["gpu_available"] = False
                validation["warnings"].append("No GPU available - processing will be slower")
        except ImportError:
            validation["warnings"].append("PyTorch not available")
        
        # Set overall status
        if validation["errors"]:
            validation["status"] = "error"
        elif validation["warnings"]:
            validation["status"] = "warning"
        
        return validation