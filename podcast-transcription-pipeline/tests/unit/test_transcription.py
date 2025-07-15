import unittest
from unittest.mock import Mock, patch, MagicMock
import json
import tempfile
from pathlib import Path

from src.transcription.whisper_transcriber import WhisperTranscriber
from src.transcription.diarization import SpeakerDiarizer

class TestTranscriptionComponents(unittest.TestCase):
    """Test cases for transcription components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Sample Whisper result
        self.sample_whisper_result = {
            "language": "en",
            "text": "Hello everyone, welcome to our podcast.",
            "segments": [
                {
                    "start": 0.0,
                    "end": 5.0,
                    "text": "Hello everyone, welcome to our podcast.",
                    "avg_logprob": -0.2,
                    "words": [
                        {"word": "Hello", "start": 0.0, "end": 0.5, "probability": 0.9},
                        {"word": "everyone", "start": 0.5, "end": 1.2, "probability": 0.85}
                    ]
                }
            ]
        }
        
        # Sample transcript segments
        self.sample_transcript = {
            "language": "en",
            "duration": 10.0,
            "text": "Hello everyone, welcome to our podcast.",
            "segments": [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": 5.0,
                    "text": "Hello everyone, welcome to our podcast.",
                    "confidence": 0.8,
                    "words": []
                },
                {
                    "id": 1,
                    "start": 5.0,
                    "end": 10.0,
                    "text": "Today we're discussing AI ethics.",
                    "confidence": 0.9,
                    "words": []
                }
            ]
        }
        
        # Sample speaker segments
        self.sample_speaker_segments = [
            {
                "speaker": "Speaker 1",
                "start": 0.0,
                "end": 7.0,
                "duration": 7.0
            },
            {
                "speaker": "Speaker 2",
                "start": 7.0,
                "end": 10.0,
                "duration": 3.0
            }
        ]
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('src.transcription.whisper_transcriber.whisper')
    def test_whisper_transcriber_load_model(self, mock_whisper):
        """Test loading Whisper model."""
        mock_model = MagicMock()
        mock_whisper.load_model.return_value = mock_model
        
        from src.utils.device_manager import DeviceManager
        device_manager = DeviceManager()
        
        transcriber = WhisperTranscriber("base", device_manager)
        transcriber.load_model()
        
        mock_whisper.load_model.assert_called_once_with("base", device=device_manager.device)
        self.assertEqual(transcriber.model, mock_model)
    
    @patch('src.transcription.whisper_transcriber.whisper')
    def test_whisper_transcriber_transcribe(self, mock_whisper):
        """Test transcription process."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = self.sample_whisper_result
        mock_whisper.load_model.return_value = mock_model
        
        from src.utils.device_manager import DeviceManager
        device_manager = DeviceManager()
        
        transcriber = WhisperTranscriber("base", device_manager)
        transcriber.model = mock_model
        
        result = transcriber.transcribe("dummy_audio.wav")
        
        self.assertIn("language", result)
        self.assertIn("segments", result)
        self.assertIn("duration", result)
        self.assertEqual(result["language"], "en")
        self.assertEqual(len(result["segments"]), 1)
    
    def test_whisper_transcriber_format_transcription(self):
        """Test transcription formatting."""
        from src.utils.device_manager import DeviceManager
        device_manager = DeviceManager()
        
        transcriber = WhisperTranscriber("base", device_manager)
        result = transcriber._format_transcription(self.sample_whisper_result)
        
        self.assertIn("language", result)
        self.assertIn("segments", result)
        self.assertIn("duration", result)
        self.assertIn("model_used", result)
        
        # Check segment formatting
        segment = result["segments"][0]
        self.assertIn("id", segment)
        self.assertIn("start", segment)
        self.assertIn("end", segment)
        self.assertIn("text", segment)
        self.assertIn("confidence", segment)
        self.assertIn("words", segment)
    
    @patch('src.transcription.diarization.Pipeline')
    def test_speaker_diarizer_load_model(self, mock_pipeline_class):
        """Test loading diarization model."""
        mock_pipeline = MagicMock()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline
        
        from src.utils.device_manager import DeviceManager
        device_manager = DeviceManager()
        
        diarizer = SpeakerDiarizer("test-model", device_manager)
        diarizer.load_model()
        
        mock_pipeline_class.from_pretrained.assert_called_once()
        self.assertEqual(diarizer.pipeline, mock_pipeline)
    
    def test_speaker_diarizer_fallback_single_speaker(self):
        """Test fallback to single speaker when diarization fails."""
        from src.utils.device_manager import DeviceManager
        device_manager = DeviceManager()
        
        diarizer = SpeakerDiarizer("test-model", device_manager)
        
        with patch('librosa.get_duration', return_value=30.0):
            result = diarizer._fallback_single_speaker("dummy_audio.wav")
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["speaker"], "Speaker 1")
        self.assertEqual(result[0]["duration"], 30.0)
    
    def test_speaker_diarizer_merge_with_transcript(self):
        """Test merging speaker information with transcript."""
        from src.utils.device_manager import DeviceManager
        device_manager = DeviceManager()
        
        diarizer = SpeakerDiarizer("test-model", device_manager)
        
        merged = diarizer.merge_with_transcript(self.sample_transcript, self.sample_speaker_segments)
        
        self.assertEqual(len(merged), len(self.sample_transcript["segments"]))
        
        # Check that speaker information was added
        for segment in merged:
            self.assertIn("speaker", segment)
            self.assertIn("id", segment)
            self.assertIn("start", segment)
            self.assertIn("end", segment)
            self.assertIn("text", segment)
    
    def test_speaker_diarizer_find_dominant_speaker(self):
        """Test finding dominant speaker for time range."""
        from src.utils.device_manager import DeviceManager
        device_manager = DeviceManager()
        
        diarizer = SpeakerDiarizer("test-model", device_manager)
        
        # Test overlap with first speaker (0-7s)
        speaker = diarizer._find_dominant_speaker(1.0, 4.0, self.sample_speaker_segments)
        self.assertEqual(speaker, "Speaker 1")
        
        # Test overlap with second speaker (7-10s)
        speaker = diarizer._find_dominant_speaker(8.0, 9.0, self.sample_speaker_segments)
        self.assertEqual(speaker, "Speaker 2")
        
        # Test overlap with both speakers - should pick dominant one
        speaker = diarizer._find_dominant_speaker(6.0, 8.0, self.sample_speaker_segments)
        # This should be Speaker 1 as they have more overlap (6-7 = 1s vs 7-8 = 1s, but Speaker 1 starts first)
        self.assertEqual(speaker, "Speaker 1")

if __name__ == "__main__":
    unittest.main()