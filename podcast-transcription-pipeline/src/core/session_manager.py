import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import uuid
import logging

class SessionManager:
    """Manages podcast analysis sessions with crash recovery and state persistence."""
    
    def __init__(self, base_output_dir: str = "./output"):
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(exist_ok=True)
        self.sessions_dir = self.base_output_dir / "sessions"
        self.sessions_dir.mkdir(exist_ok=True)
        
    def create_session(self, audio_file_path: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Create a new session or recover existing one."""
        if session_id is None:
            session_id = str(uuid.uuid4())[:8]
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_name = f"session_{timestamp}_{session_id}"
        session_dir = self.sessions_dir / session_name
        
        # Check for existing session
        if session_dir.exists():
            return self._recover_session(session_dir, audio_file_path)
        
        # Create new session
        session_dir.mkdir(exist_ok=True)
        
        session_info = {
            "session_id": session_id,
            "session_name": session_name,
            "session_dir": str(session_dir),
            "audio_file": audio_file_path,
            "created_at": datetime.now().isoformat(),
            "status": "initialized",
            "completed_steps": [],
            "failed_steps": [],
            "processing_metadata": {}
        }
        
        # Save session info
        with open(session_dir / "session_info.json", "w") as f:
            json.dump(session_info, f, indent=2)
        
        # Initialize processing log
        self._init_processing_log(session_dir)
        
        return session_info
    
    def _recover_session(self, session_dir: Path, audio_file_path: str) -> Dict[str, Any]:
        """Recover existing session and determine resumption point."""
        try:
            with open(session_dir / "session_info.json", "r") as f:
                session_info = json.load(f)
            
            session_info["status"] = "recovered"
            session_info["recovered_at"] = datetime.now().isoformat()
            
            # Determine completed steps
            completed_steps = self._detect_completed_steps(session_dir)
            session_info["completed_steps"] = completed_steps
            
            self._log_processing_step(session_dir, "session_recovery", 
                                    f"Recovered session with {len(completed_steps)} completed steps")
            
            return session_info
            
        except Exception as e:
            raise RuntimeError(f"Failed to recover session: {e}")
    
    def _detect_completed_steps(self, session_dir: Path) -> List[str]:
        """Detect which processing steps have been completed."""
        completed_steps = []
        
        # Check for required output files
        step_files = {
            "transcription": "transcript.json",
            "diarization": "diarization.json",
            "text_emotions": "emotions_text.json", 
            "audio_emotions": "emotions_audio.json",
            "semantic_segmentation": "semantic_blocks.json",
            "keyword_extraction": "keywords_topics.json",
            "validation": "validation_report.json"
        }
        
        for step, filename in step_files.items():
            if (session_dir / filename).exists():
                completed_steps.append(step)
        
        return completed_steps
    
    def _init_processing_log(self, session_dir: Path):
        """Initialize processing log file."""
        log_file = session_dir / "processing_log.txt"
        with open(log_file, "w") as f:
            f.write(f"Processing Log - Session: {session_dir.name}\n")
            f.write(f"Created: {datetime.now().isoformat()}\n")
            f.write("=" * 50 + "\n\n")
    
    def _log_processing_step(self, session_dir: Path, step: str, message: str, 
                           status: str = "info"):
        """Log a processing step with timestamp."""
        log_file = session_dir / "processing_log.txt"
        timestamp = datetime.now().isoformat()
        
        with open(log_file, "a") as f:
            f.write(f"[{timestamp}] [{status.upper()}] {step}: {message}\n")
    
    def update_session_status(self, session_info: Dict[str, Any], step: str, 
                            status: str, metadata: Optional[Dict] = None):
        """Update session status and log progress."""
        session_dir = Path(session_info["session_dir"])
        
        if status == "completed":
            if step not in session_info["completed_steps"]:
                session_info["completed_steps"].append(step)
        elif status == "failed":
            if step not in session_info["failed_steps"]:
                session_info["failed_steps"].append(step)
        
        if metadata:
            session_info["processing_metadata"][step] = metadata
        
        # Update session info file
        with open(session_dir / "session_info.json", "w") as f:
            json.dump(session_info, f, indent=2)
        
        # Log the step
        self._log_processing_step(session_dir, step, f"Status: {status}")
    
    def should_skip_step(self, session_info: Dict[str, Any], step: str) -> bool:
        """Check if a step should be skipped (already completed)."""
        return step in session_info.get("completed_steps", [])
    
    def get_session_files(self, session_info: Dict[str, Any]) -> Dict[str, Path]:
        """Get paths to all session output files."""
        session_dir = Path(session_info["session_dir"])
        
        return {
            "transcript": session_dir / "transcript.json",
            "transcript_txt": session_dir / "transcript.txt",
            "diarization": session_dir / "diarization.json",  # Speaker diarization results
            "emotions_text": session_dir / "emotions_text.json",
            "emotions_audio": session_dir / "emotions_audio.json",
            "emotions": session_dir / "emotions_combined.json",  # Combined emotions
            "semantic_blocks": session_dir / "semantic_blocks.json",
            "keywords_topics": session_dir / "keywords_topics.json",
            "validation_report": session_dir / "validation_report.json",
            "processing_log": session_dir / "processing_log.txt"
        }