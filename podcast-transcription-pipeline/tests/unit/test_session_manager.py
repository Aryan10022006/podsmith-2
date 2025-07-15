import unittest
from unittest.mock import Mock, patch, MagicMock
import json
from pathlib import Path
import tempfile
import os

from src.core.session_manager import SessionManager

class TestSessionManager(unittest.TestCase):
    """Test cases for SessionManager functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.session_manager = SessionManager(self.temp_dir)
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_new_session(self):
        """Test creating a new session."""
        audio_file = "test_audio.mp3"
        session_info = self.session_manager.create_session(audio_file)
        
        # Check session info structure
        self.assertIn("session_id", session_info)
        self.assertIn("session_name", session_info)
        self.assertIn("session_dir", session_info)
        self.assertEqual(session_info["audio_file"], audio_file)
        self.assertEqual(session_info["status"], "initialized")
        
        # Check directory was created
        session_dir = Path(session_info["session_dir"])
        self.assertTrue(session_dir.exists())
        
        # Check session info file exists
        self.assertTrue((session_dir / "session_info.json").exists())
        
        # Check processing log exists
        self.assertTrue((session_dir / "processing_log.txt").exists())
    
    def test_session_recovery(self):
        """Test recovering an existing session."""
        audio_file = "test_audio.mp3"
        
        # Create initial session
        session_info = self.session_manager.create_session(audio_file)
        session_dir = Path(session_info["session_dir"])
        
        # Create some fake completed files
        (session_dir / "transcript.json").write_text('{"test": "data"}')
        (session_dir / "emotions_text.json").write_text('[]')
        
        # Try to create session again (should recover)
        recovered_info = self.session_manager.create_session(audio_file, session_info["session_id"])
        
        self.assertEqual(recovered_info["status"], "recovered")
        self.assertIn("transcription", recovered_info["completed_steps"])
        self.assertIn("text_emotions", recovered_info["completed_steps"])
    
    def test_update_session_status(self):
        """Test updating session status."""
        audio_file = "test_audio.mp3"
        session_info = self.session_manager.create_session(audio_file)
        
        # Update status to completed
        self.session_manager.update_session_status(session_info, "transcription", "completed")
        
        self.assertIn("transcription", session_info["completed_steps"])
        
        # Update status to failed
        self.session_manager.update_session_status(session_info, "diarization", "failed")
        
        self.assertIn("diarization", session_info["failed_steps"])
    
    def test_should_skip_step(self):
        """Test step skipping logic."""
        audio_file = "test_audio.mp3"
        session_info = self.session_manager.create_session(audio_file)
        
        # Should not skip initially
        self.assertFalse(self.session_manager.should_skip_step(session_info, "transcription"))
        
        # Mark as completed
        session_info["completed_steps"] = ["transcription"]
        
        # Should skip now
        self.assertTrue(self.session_manager.should_skip_step(session_info, "transcription"))
    
    def test_get_session_files(self):
        """Test getting session file paths."""
        audio_file = "test_audio.mp3"
        session_info = self.session_manager.create_session(audio_file)
        
        files = self.session_manager.get_session_files(session_info)
        
        # Check all expected files are present
        expected_files = [
            "transcript", "emotions_text", "emotions_audio", 
            "semantic_blocks", "summaries", "keywords_topics",
            "validation_report", "processing_log"
        ]
        
        for file_key in expected_files:
            self.assertIn(file_key, files)
            self.assertIsInstance(files[file_key], Path)

if __name__ == "__main__":
    unittest.main()