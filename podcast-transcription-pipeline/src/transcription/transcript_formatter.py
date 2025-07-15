import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

class TranscriptFormatter:
    """Formats transcription results into various output formats."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def format_transcript(self, transcript_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format transcript data with consistent structure."""
        formatted_transcript = []
        
        for entry in transcript_data:
            formatted_entry = {
                "speaker": entry.get("speaker", "UNKNOWN"),
                "start": float(entry.get("start", 0.0)),
                "end": float(entry.get("end", 0.0)),
                "text": entry.get("text", "").strip(),
                "confidence": entry.get("confidence", 1.0)
            }
            formatted_transcript.append(formatted_entry)
        
        return formatted_transcript
    
    def format_for_display(self, transcript_data: List[Dict[str, Any]]) -> str:
        """Format transcript for human-readable display."""
        formatted_lines = []
        
        for entry in transcript_data:
            timestamp = self._format_timestamp(entry.get("start", 0.0))
            speaker = entry.get("speaker", "UNKNOWN")
            text = entry.get("text", "").strip()
            
            formatted_lines.append(f"[{timestamp}] {speaker}: {text}")
        
        return "\n".join(formatted_lines)
    
    def _format_timestamp(self, seconds: float) -> str:
        """Convert seconds to MM:SS format."""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    def save_transcript_to_json(self, transcript: List[Dict[str, Any]], output_path: Path) -> None:
        """Save transcript to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as json_file:
            json.dump(transcript, json_file, indent=4, ensure_ascii=False)
        
        self.logger.info(f"Transcript saved to {output_path}")
    
    def save_formatted_transcript(self, transcript: List[Dict[str, Any]], output_path: Path) -> None:
        """Save human-readable transcript to text file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        formatted_text = self.format_for_display(transcript)
        
        with open(output_path, 'w', encoding='utf-8') as text_file:
            text_file.write(formatted_text)
        
        self.logger.info(f"Formatted transcript saved to {output_path}")
    
    def save_transcript_txt(self, transcript: List[Dict[str, Any]], output_path: Path) -> None:
        """Save transcript line-by-line in simple txt format."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as txt_file:
            for entry in transcript:
                timestamp = self._format_timestamp(entry.get("start", 0.0))
                speaker = entry.get("speaker", "Unknown")
                text = entry.get("text", "").strip()
                
                # Write each line with timestamp and speaker
                txt_file.write(f"[{timestamp}] {speaker}: {text}\n")
        
        self.logger.info(f"Line-by-line transcript saved to {output_path}")

# Backward compatibility functions
def format_transcript(transcript_data):
    formatted_transcript = []
    
    for entry in transcript_data:
        formatted_entry = {
            "speaker": entry["speaker"],
            "start": entry["start"],
            "end": entry["end"],
            "text": entry["text"]
        }
        formatted_transcript.append(formatted_entry)
    
    return formatted_transcript

def save_transcript_to_json(transcript, output_path):
    import json
    
    with open(output_path, 'w') as json_file:
        json.dump(transcript, json_file, indent=4)

def load_transcript_from_json(input_path):
    import json
    
    with open(input_path, 'r') as json_file:
        return json.load(json_file)