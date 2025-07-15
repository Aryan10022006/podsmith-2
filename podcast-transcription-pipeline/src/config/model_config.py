# Model configuration settings for the podcast transcription pipeline

MODEL_CONFIG = {
    "transcription": {
        "asr_model": "whisper-large",
        "language": "en",
        "sample_rate": 16000,
        "max_length": 300,
        "min_length": 50
    },
    "diarization": {
        "model": "pyAudioAnalysis",
        "num_speakers": 2,
        "min_speaker_duration": 1.0
    },
    "emotion_detection": {
        "text_model": "emotion-nlp-model",
        "audio_model": "audio-emotion-model",
        "confidence_threshold": 0.7
    },
    "semantic_analysis": {
        "topic_model": "topic-classifier-model",
        "keyword_model": "keyword-extractor-model"
    },
    "output": {
        "format": "json",
        "directory": "./output/sessions/"
    }
}

class ModelConfig:
    """Model configuration management class."""
    
    def __init__(self, config_dict: dict = None):
        self.config = config_dict or MODEL_CONFIG
    
    def get_transcription_config(self) -> dict:
        """Get transcription model configuration."""
        return self.config.get("transcription", {})
    
    def get_diarization_config(self) -> dict:
        """Get diarization model configuration."""
        return self.config.get("diarization", {})
    
    def get_emotion_config(self) -> dict:
        """Get emotion detection model configuration."""
        return self.config.get("emotion_detection", {})
    
    def get_semantic_config(self) -> dict:
        """Get semantic analysis model configuration."""
        return self.config.get("semantic_analysis", {})
    
    def get_output_config(self) -> dict:
        """Get output configuration."""
        return self.config.get("output", {})
    
    def update_config(self, new_config: dict) -> None:
        """Update configuration with new values."""
        self.config.update(new_config)