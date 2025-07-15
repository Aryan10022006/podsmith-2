import unittest
from unittest.mock import Mock, patch, MagicMock
import json
import tempfile
import os
from pathlib import Path

from src.core.pipeline_orchestrator import PipelineOrchestrator

class TestPipelineIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a mock config
        self.mock_config = {
            "transcription": {"model": "base"},
            "diarization": {"model": "test-model"},
            "emotion": {"text_model": "test-emotion", "confidence_threshold": 0.5},
            "topics": {"model": "test-topics", "max_topics_per_block": 3},
            "summarization": {"model": "test-summary", "max_length": 150, "min_length": 30},
            "keywords": {"method": "tfidf", "max_keywords_global": 20, "max_keywords_per_block": 10, "min_keyword_score": 0.1},
            "validation": {"min_confidence_threshold": 0.6},
            "output": {"base_dir": self.temp_dir}
        }
        
        # Sample audio file path (will be mocked)
        self.sample_audio_file = os.path.join(self.temp_dir, "test_audio.mp3")
        with open(self.sample_audio_file, 'w') as f:
            f.write("fake audio content")
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('src.core.pipeline_orchestrator.yaml.safe_load')
    @patch('builtins.open', create=True)
    def test_pipeline_initialization(self, mock_open, mock_yaml):
        """Test pipeline initialization with config."""
        mock_yaml.return_value = self.mock_config
        
        orchestrator = PipelineOrchestrator("dummy_config.yaml")
        
        self.assertIsNotNone(orchestrator.config)
        self.assertIsNotNone(orchestrator.session_manager)
        self.assertIsNotNone(orchestrator.device_manager)
        self.assertIsNotNone(orchestrator.validator)
    
    @patch('src.core.pipeline_orchestrator.yaml.safe_load')
    @patch('builtins.open', create=True)
    @patch('src.transcription.whisper_transcriber.whisper')
    @patch('src.transcription.diarization.Pipeline')
    @patch('src.analysis.emotion_detector.pipeline')
    @patch('src.analysis.summarizer.pipeline')
    def test_complete_pipeline_process(self, mock_sum_pipeline, mock_emotion_pipeline, 
                                     mock_diarization_pipeline, mock_whisper, mock_open, mock_yaml):
        """Test complete pipeline processing with mocked components."""
        mock_yaml.return_value = self.mock_config
        
        # Mock Whisper
        mock_whisper_model = MagicMock()
        mock_whisper_model.transcribe.return_value = {
            "language": "en",
            "text": "Hello world, this is a test podcast.",
            "segments": [
                {
                    "start": 0.0,
                    "end": 5.0,
                    "text": "Hello world, this is a test podcast.",
                    "avg_logprob": -0.2,
                    "words": []
                }
            ]
        }
        mock_whisper.load_model.return_value = mock_whisper_model
        
        # Mock diarization
        mock_diar_pipeline = MagicMock()
        mock_diarization_pipeline.from_pretrained.return_value = mock_diar_pipeline
        mock_diar_pipeline.return_value = MagicMock()  # Mock annotation object
        
        # Mock emotion detection
        mock_emotion_pipe = MagicMock()
        mock_emotion_pipe.return_value = [[{"label": "neutral", "score": 0.8}]]
        mock_emotion_pipeline.return_value = mock_emotion_pipe
        
        # Mock summarization
        mock_sum_pipe = MagicMock()
        mock_sum_pipe.return_value = [{"summary_text": "This is a test summary."}]
        mock_sum_pipeline.return_value = mock_sum_pipe
        
        # Mock file operations for various components
        with patch('src.analysis.semantic_segmenter.SentenceTransformer', return_value=None):
            with patch('librosa.get_duration', return_value=10.0):
                orchestrator = PipelineOrchestrator("dummy_config.yaml")
                
                # This should complete without errors
                result = orchestrator.process_audio(self.sample_audio_file)
        
        # Verify result structure
        self.assertIn("session_info", result)
        self.assertIn("session_files", result)
        self.assertIn("validation_report", result)
        self.assertEqual(result["status"], "completed")
        
        # Verify session was created
        session_info = result["session_info"]
        self.assertIn("session_id", session_info)
        self.assertIn("session_dir", session_info)
        
        # Verify session directory exists
        session_dir = Path(session_info["session_dir"])
        self.assertTrue(session_dir.exists())
    
    @patch('src.core.pipeline_orchestrator.yaml.safe_load')
    @patch('builtins.open', create=True)
    def test_pipeline_error_handling(self, mock_open, mock_yaml):
        """Test pipeline error handling."""
        mock_yaml.return_value = self.mock_config
        
        orchestrator = PipelineOrchestrator("dummy_config.yaml")
        
        # Test with non-existent audio file
        with self.assertRaises(FileNotFoundError):
            orchestrator.process_audio("non_existent_file.mp3")
    
    @patch('src.core.pipeline_orchestrator.yaml.safe_load')
    @patch('builtins.open', create=True)
    def test_session_listing(self, mock_open, mock_yaml):
        """Test session listing functionality."""
        mock_yaml.return_value = self.mock_config
        
        orchestrator = PipelineOrchestrator("dummy_config.yaml")
        
        # Should return empty list initially
        sessions = orchestrator.list_sessions()
        self.assertEqual(len(sessions), 0)
        
        # Create a mock session directory
        session_dir = Path(orchestrator.session_manager.sessions_dir) / "test_session"
        session_dir.mkdir()
        
        session_info = {
            "session_id": "test123",
            "status": "completed",
            "completed_steps": ["transcription"],
            "failed_steps": [],
            "created_at": "2025-01-01T00:00:00"
        }
        
        with open(session_dir / "session_info.json", "w") as f:
            json.dump(session_info, f)
        
        # Should now return one session
        sessions = orchestrator.list_sessions()
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0]["session_name"], "test_session")
    
    @patch('src.core.pipeline_orchestrator.yaml.safe_load')
    @patch('builtins.open', create=True)
    def test_session_status_retrieval(self, mock_open, mock_yaml):
        """Test retrieving session status."""
        mock_yaml.return_value = self.mock_config
        
        orchestrator = PipelineOrchestrator("dummy_config.yaml")
        
        # Test with non-existent session
        status = orchestrator.get_pipeline_status("non_existent_session")
        self.assertIn("error", status)
        
        # Create a mock session
        session_dir = Path(orchestrator.session_manager.sessions_dir) / "test_session"
        session_dir.mkdir()
        
        session_info = {
            "session_id": "test123",
            "status": "processing",
            "completed_steps": ["transcription", "diarization"],
            "failed_steps": [],
            "created_at": "2025-01-01T00:00:00"
        }
        
        with open(session_dir / "session_info.json", "w") as f:
            json.dump(session_info, f)
        
        # Should return session status
        status = orchestrator.get_pipeline_status("test_session")
        self.assertEqual(status["status"], "processing")
        self.assertEqual(len(status["completed_steps"]), 2)
    
    def test_config_fallback(self):
        """Test configuration fallback when config file is missing."""
        # This should use default config
        orchestrator = PipelineOrchestrator("non_existent_config.yaml")
        
        self.assertIsNotNone(orchestrator.config)
        self.assertIn("transcription", orchestrator.config)
        self.assertIn("output", orchestrator.config)

if __name__ == "__main__":
    unittest.main()