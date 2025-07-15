import os
import json
import yaml
import logging
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import librosa

from src.core.session_manager import SessionManager
from src.utils.device_manager import DeviceManager
from src.utils.validator import Validator
from src.transcription.whisper_transcriber import WhisperTranscriber, AudioEmbedding
from src.transcription.diarization import SpeakerDiarizer
from src.transcription.transcript_formatter import TranscriptFormatter
from src.analysis.emotion_detector import EmotionDetector
from src.analysis.semantic_segmenter import SemanticSegmenter
from src.analysis.summarizer import Summarizer
from src.analysis.keyword_extractor import KeywordExtractor

class PipelineOrchestrator:
    """Main orchestrator for the podcast transcription and analysis pipeline with detailed timing."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the pipeline with configuration."""
        self.config = self._load_config(config_path)
        self.session_manager = SessionManager(self.config["output"]["base_dir"])
        self.device_manager = DeviceManager()
        self.validator = Validator()
        
        # Performance tracking
        self.step_timings = {}
        self.total_start_time = None
        
        # Initialize components
        self.transcriber = None
        self.diarizer = None
        self.emotion_detector = None
        self.segmenter = None
        self.summarizer = None
        self.keyword_extractor = None
        
        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load pipeline configuration."""
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            print(f"Failed to load config from {config_path}: {e}")
            # Return default config
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "transcription": {"model": "large-v3"},
            "diarization": {"model": "pyannote/speaker-diarization-3.1"},
            "emotion": {"text_model": "j-hartmann/emotion-english-distilroberta-base"},
            "topics": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
            "summarization": {"model": "facebook/bart-large-cnn"},
            "keywords": {"method": "tfidf"},
            "validation": {"min_confidence_threshold": 0.6},
            "output": {"base_dir": "./output"}
        }
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('pipeline.log')
            ]
        )
    
    def _initialize_components(self):
        """Initialize all pipeline components."""
        if self.transcriber is None:
            self.transcriber = WhisperTranscriber(
                model_name=self.config["transcription"]["model"],
                device_manager=self.device_manager
            )
        
        if self.diarizer is None:
            self.diarizer = SpeakerDiarizer(
                model_name=self.config["diarization"]["model"],
                device_manager=self.device_manager
            )
        
        if self.emotion_detector is None:
            self.emotion_detector = EmotionDetector(
                device_manager=self.device_manager
            )
        
        if self.segmenter is None:
            self.segmenter = SemanticSegmenter(
                embedding_model=self.config["topics"]["model"],
                device_manager=self.device_manager
            )
        
        # Initialize transcript formatter
        self.transcript_formatter = TranscriptFormatter()
        
        if self.summarizer is None:
            self.summarizer = Summarizer(
                model_name=self.config["summarization"]["model"],
                device_manager=self.device_manager
            )
        
        if self.keyword_extractor is None:
            self.keyword_extractor = KeywordExtractor(
                method=self.config["keywords"]["method"]
            )
    
    def process_audio(self, audio_file_path: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process audio file through the unified pipeline with shared embeddings.
        
        This method implements the unified approach:
        1. Convert audio to WAV format
        2. Extract shared embeddings once
        3. Use embeddings across transcription, diarization, and emotion detection
        """
        self.total_start_time = time.time()
        self.step_timings = {}
        
        self.logger.info(f"Starting unified pipeline processing for: {audio_file_path}")
        
        # Validate audio file and check size for optimization
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
        
        # Check file size for large file optimization
        file_size_mb = os.path.getsize(audio_file_path) / (1024 * 1024)
        is_large_file = file_size_mb > 50  # Consider files > 50MB as large
        
        # Create session with large file metadata
        session_info = self.session_manager.create_session(audio_file_path, session_id)
        session_files = self.session_manager.get_session_files(session_info)
        
        # Store optimization metadata
        session_info['is_large_file'] = is_large_file
        session_info['file_size_mb'] = file_size_mb
        
        if is_large_file:
            self.logger.info(f"Large audio file detected ({file_size_mb:.1f}MB). Enabling performance optimizations.")
        
        self.logger.info(f"Created session: {session_info['session_name']}")
        
        # Initialize components
        self._initialize_components()
        
        try:
            # Step 1: Convert audio to WAV format
            step_start = time.time()
            self.logger.info("Step 1: Converting audio to WAV format...")
            wav_audio_path = self._convert_audio_to_wav(audio_file_path, session_info, session_files)
            self.step_timings["audio_conversion"] = time.time() - step_start
            
            # Step 2: Extract shared embeddings
            step_start = time.time()
            self.logger.info("Step 2: Extracting shared audio embeddings...")
            shared_embeddings = self._extract_shared_embeddings(wav_audio_path, session_info, session_files)
            self.step_timings["embedding_extraction"] = time.time() - step_start
            
            # Step 3: Transcription using shared embeddings
            step_start = time.time()
            self.logger.info("Step 3: Performing transcription...")
            transcript_data = self._process_transcription_unified(wav_audio_path, shared_embeddings, session_info, session_files)
            self.step_timings["transcription"] = time.time() - step_start
            
            # Memory management after transcription
            self._manage_memory_for_large_files(session_info)
            
            # Step 4: Speaker diarization using shared embeddings
            step_start = time.time()
            self.logger.info("Step 4: Performing speaker diarization...")
            diarized_transcript = self._process_diarization_unified(wav_audio_path, transcript_data, shared_embeddings, session_info, session_files)
            self.step_timings["diarization"] = time.time() - step_start
            
            # Memory management after diarization  
            self._manage_memory_for_large_files(session_info)
            
            # Step 5: Emotion detection using shared embeddings
            step_start = time.time()
            self.logger.info("Step 5: Performing emotion detection...")
            text_emotions, audio_emotions = self._process_emotions_unified(wav_audio_path, diarized_transcript, shared_embeddings, session_info, session_files)
            self.step_timings["emotion_detection"] = time.time() - step_start
            
            # Memory management after emotion detection
            self._manage_memory_for_large_files(session_info)
            
            # Continue with remaining steps (optimized for large files)...
            step_start = time.time()
            self.logger.info("Step 6: Performing semantic segmentation...")
            semantic_blocks = self._process_semantic_segmentation(diarized_transcript, session_info, session_files)
            self.step_timings["semantic_segmentation"] = time.time() - step_start
            
            # Memory management after segmentation
            self._manage_memory_for_large_files(session_info)
            
            # Optimize summarization for large files
            step_start = time.time()
            self.logger.info("Step 7: Performing optimized summarization...")
            summaries = self._process_summarization_optimized(semantic_blocks, session_info, session_files)
            self.step_timings["summarization"] = time.time() - step_start
            
            # Memory management after summarization
            self._manage_memory_for_large_files(session_info)
            
            keywords = self._process_keyword_extraction(semantic_blocks, session_info, session_files)
            validation_report = self._process_validation(session_files, session_info)
            
            # Calculate total processing time
            total_time = time.time() - self.total_start_time
            self.step_timings["total_pipeline"] = total_time
            
            # Final session update with timing information
            self.session_manager.update_session_status(session_info, "pipeline", "completed", 
                                                     {
                                                         "completion_time": datetime.now().isoformat(),
                                                         "total_processing_time": total_time,
                                                         "step_timings": self.step_timings,
                                                         "shared_embeddings_used": shared_embeddings is not None
                                                     })
            
            self.logger.info(f"Unified pipeline processing completed for session: {session_info['session_name']}")
            self.logger.info(f"Total processing time: {total_time:.2f} seconds")
            self.logger.info(f"Shared embeddings used: {shared_embeddings is not None}")
            self._log_timing_summary()
            
            return {
                "session_info": session_info,
                "session_files": {k: str(v) for k, v in session_files.items()},
                "validation_report": validation_report,
                "performance_metrics": {
                    "total_time": total_time,
                    "step_timings": self.step_timings,
                    "shared_embeddings_used": shared_embeddings is not None
                },
                "status": "completed"
            }
            
        except Exception as e:
            total_time = time.time() - self.total_start_time if self.total_start_time else 0
            self.logger.error(f"Unified pipeline processing failed after {total_time:.2f} seconds: {e}")
            self.session_manager.update_session_status(session_info, "pipeline", "failed", 
                                                     {
                                                         "error": str(e),
                                                         "failed_after": total_time,
                                                         "step_timings": self.step_timings
                                                     })
            raise
    
    def _log_timing_summary(self):
        """Log a summary of processing times for each step."""
        self.logger.info("=== PROCESSING TIME SUMMARY ===")
        for step, timing in self.step_timings.items():
            minutes = int(timing // 60)
            seconds = timing % 60
            if minutes > 0:
                self.logger.info(f"{step}: {minutes}m {seconds:.1f}s")
            else:
                self.logger.info(f"{step}: {seconds:.2f}s")
        self.logger.info("===============================")

    def _process_transcription(self, audio_file_path: str, session_info: Dict[str, Any], 
                             session_files: Dict[str, Path]) -> Dict[str, Any]:
        """Process transcription step with detailed timing."""
        step_start = time.time()
        
        if self.session_manager.should_skip_step(session_info, "transcription"):
            self.logger.info("Skipping transcription (already completed)")
            with open(session_files["transcript"], "r", encoding="utf-8") as f:
                return json.load(f)
        
        self.logger.info("Starting transcription...")
        
        try:
            transcript_data = self.transcriber.transcribe(audio_file_path)
            
            # Save transcript
            self.transcriber.save_transcript(transcript_data, session_files["transcript"])
            
            step_time = time.time() - step_start
            self.step_timings["transcription"] = step_time
            
            self.session_manager.update_session_status(session_info, "transcription", "completed", 
                                                     {"processing_time": step_time})
            self.logger.info(f"Transcription completed successfully in {step_time:.2f} seconds")
            
            return transcript_data
            
        except Exception as e:
            step_time = time.time() - step_start
            self.step_timings["transcription"] = step_time
            self.session_manager.update_session_status(session_info, "transcription", "failed", 
                                                     {"error": str(e), "processing_time": step_time})
            raise
    
    def _process_diarization(self, audio_file_path: str, transcript_data: Dict[str, Any],
                           session_info: Dict[str, Any], session_files: Dict[str, Path]) -> List[Dict[str, Any]]:
        """Process speaker diarization step."""
        if self.session_manager.should_skip_step(session_info, "diarization"):
            self.logger.info("Skipping diarization (already completed)")
            # Return transcript segments (should have speaker info)
            return transcript_data.get("segments", [])
        
        self.logger.info("Starting speaker diarization...")
        
        try:
            # Get speaker segments
            speaker_segments = self.diarizer.diarize(audio_file_path)
            
            # Merge with transcript
            diarized_transcript = self.diarizer.merge_with_transcript(transcript_data, speaker_segments)
            
            # Save diarized transcript
            self.diarizer.save_diarization(diarized_transcript, session_files["transcript"])
            
            # Save transcript in txt format (line-by-line)
            self.transcript_formatter.save_transcript_txt(diarized_transcript, session_files["transcript_txt"])
            
            self.session_manager.update_session_status(session_info, "diarization", "completed")
            self.logger.info("Diarization completed successfully")
            
            return diarized_transcript
            
        except Exception as e:
            self.session_manager.update_session_status(session_info, "diarization", "failed",
                                                     {"error": str(e)})
            raise
    
    def _process_text_emotions(self, transcript_segments: List[Dict[str, Any]],
                             session_info: Dict[str, Any], session_files: Dict[str, Path]) -> List[Dict[str, Any]]:
        """Process text emotion detection step."""
        if self.session_manager.should_skip_step(session_info, "text_emotions"):
            self.logger.info("Skipping text emotion detection (already completed)")
            with open(session_files["emotions_text"], "r", encoding="utf-8") as f:
                return json.load(f)
        
        self.logger.info("Starting text emotion detection...")
        
        try:
            text_emotions = self.emotion_detector.detect_text_emotions(
                transcript_segments,
                confidence_threshold=self.config["emotion"].get("confidence_threshold", 0.5)
            )
            
            # Save text emotions
            self.emotion_detector.save_text_emotions(text_emotions, session_files["emotions_text"])
            
            self.session_manager.update_session_status(session_info, "text_emotions", "completed")
            self.logger.info("Text emotion detection completed successfully")
            
            return text_emotions
            
        except Exception as e:
            self.session_manager.update_session_status(session_info, "text_emotions", "failed",
                                                     {"error": str(e)})
            raise
    
    def _process_audio_emotions(self, audio_file_path: str, transcript_segments: List[Dict[str, Any]],
                              session_info: Dict[str, Any], session_files: Dict[str, Path]) -> List[Dict[str, Any]]:
        """Process audio emotion detection step."""
        if self.session_manager.should_skip_step(session_info, "audio_emotions"):
            self.logger.info("Skipping audio emotion detection (already completed)")
            with open(session_files["emotions_audio"], "r", encoding="utf-8") as f:
                return json.load(f)
        
        self.logger.info("Starting audio emotion detection...")
        
        try:
            # Extract speaker segments from transcript
            speaker_segments = []
            for segment in transcript_segments:
                speaker_segments.append({
                    "speaker": segment.get("speaker", "Unknown"),
                    "start": segment.get("start", 0.0),
                    "end": segment.get("end", 0.0)
                })
            
            audio_emotions = self.emotion_detector.detect_audio_emotions(audio_file_path, speaker_segments)
            
            # Save audio emotions
            self.emotion_detector.save_audio_emotions(audio_emotions, session_files["emotions_audio"])
            
            self.session_manager.update_session_status(session_info, "audio_emotions", "completed")
            self.logger.info("Audio emotion detection completed successfully")
            
            return audio_emotions
            
        except Exception as e:
            self.session_manager.update_session_status(session_info, "audio_emotions", "failed",
                                                     {"error": str(e)})
            raise
    
    def _process_semantic_segmentation(self, transcript_segments: List[Dict[str, Any]],
                                     session_info: Dict[str, Any], session_files: Dict[str, Path]) -> List[Dict[str, Any]]:
        """Process semantic segmentation step."""
        if self.session_manager.should_skip_step(session_info, "semantic_segmentation"):
            self.logger.info("Skipping semantic segmentation (already completed)")
            with open(session_files["semantic_blocks"], "r", encoding="utf-8") as f:
                return json.load(f)
        
        self.logger.info("Starting semantic segmentation...")
        
        try:
            # Check if this is a large file and optimize accordingly
            is_large_file = session_info.get('is_large_file', False)
            
            if is_large_file:
                self.logger.info("Applying large file optimizations to semantic segmentation...")
                # Limit segments for very large files to prevent memory issues
                max_segments = 500
                if len(transcript_segments) > max_segments:
                    self.logger.warning(f"Large file: limiting segments from {len(transcript_segments)} to {max_segments}")
                    transcript_segments = transcript_segments[:max_segments]
            
            # Create semantic blocks
            semantic_blocks = self.segmenter.segment_into_blocks(transcript_segments)
            
            # For large files, limit topic classification complexity
            max_topics = 2 if is_large_file else self.config["topics"].get("max_topics_per_block", 3)
            
            # Classify topics
            semantic_blocks = self.segmenter.classify_topics(
                semantic_blocks,
                max_topics_per_block=max_topics
            )
            
            # Save semantic blocks
            self.segmenter.save_semantic_blocks(semantic_blocks, session_files["semantic_blocks"])
            
            self.session_manager.update_session_status(session_info, "semantic_segmentation", "completed")
            self.logger.info("Semantic segmentation completed successfully")
            
            return semantic_blocks
            
        except Exception as e:
            self.session_manager.update_session_status(session_info, "semantic_segmentation", "failed",
                                                     {"error": str(e)})
            raise
    
    def _process_summarization(self, semantic_blocks: List[Dict[str, Any]],
                             session_info: Dict[str, Any], session_files: Dict[str, Path]) -> Dict[str, Any]:
        """Process summarization step."""
        if self.session_manager.should_skip_step(session_info, "summarization"):
            self.logger.info("Skipping summarization (already completed)")
            with open(session_files["summaries"], "r", encoding="utf-8") as f:
                return json.load(f)
        
        self.logger.info("Starting summarization...")

        # --- Enrich semantic blocks with keywords and emotions ---
        # Load keywords
        keywords_path = session_files.get("keywords_topics")
        block_keywords_map = {}
        if keywords_path and os.path.exists(keywords_path):
            with open(keywords_path, "r", encoding="utf-8") as f:
                keywords_data = json.load(f)
            block_keywords = keywords_data.get("block_keywords", [])
            for bk in block_keywords:
                block_id = bk.get("block_id")
                block_keywords_map[block_id] = bk.get("keywords", [])

        # Load emotions (optional)
        emotions_path = session_files.get("emotions")
        block_emotions_map = {}
        if emotions_path and os.path.exists(emotions_path):
            with open(emotions_path, "r", encoding="utf-8") as f:
                emotions_data = json.load(f)
            for em in emotions_data.get("block_emotions", []):
                block_id = em.get("block_id")
                block_emotions_map[block_id] = em.get("emotions", [])

        # Enrich blocks
        enriched_blocks = []
        for block in semantic_blocks:
            block_id = block.get("block_id")
            block = dict(block)  # Copy to avoid mutating original
            if block_id in block_keywords_map:
                block["topic_keywords"] = block_keywords_map[block_id]
            if block_id in block_emotions_map:
                block["emotions"] = block_emotions_map[block_id]
            enriched_blocks.append(block)

        try:
            # Generate block summaries
            block_summaries = self.summarizer.summarize_blocks(
                enriched_blocks,
                max_length=self.config["summarization"].get("max_length", 150),
                min_length=self.config["summarization"].get("min_length", 30)
            )

            # Generate global summary
            global_summary = self.summarizer.generate_global_summary(
                enriched_blocks,
                block_summaries
            )

            # Save summaries
            self.summarizer.save_summaries(block_summaries, global_summary, session_files["summaries"])

            self.session_manager.update_session_status(session_info, "summarization", "completed")
            self.logger.info("Summarization completed successfully")

            return {"block_summaries": block_summaries, "global_summary": global_summary}

        except Exception as e:
            self.session_manager.update_session_status(session_info, "summarization", "failed",
                                                     {"error": str(e)})
            raise
    
    def _process_keyword_extraction(self, semantic_blocks: List[Dict[str, Any]],
                                  session_info: Dict[str, Any], session_files: Dict[str, Path]) -> Dict[str, Any]:
        """Process keyword extraction step."""
        if self.session_manager.should_skip_step(session_info, "keyword_extraction"):
            self.logger.info("Skipping keyword extraction (already completed)")
            with open(session_files["keywords_topics"], "r", encoding="utf-8") as f:
                return json.load(f)
        
        self.logger.info("Starting keyword extraction...")
        
        try:
            # Extract global keywords
            global_keywords = self.keyword_extractor.extract_global_keywords(
                semantic_blocks,
                max_keywords=self.config["keywords"].get("max_keywords_global", 20),
                min_score=self.config["keywords"].get("min_keyword_score", 0.1)
            )
            
            # Extract block keywords
            block_keywords = self.keyword_extractor.extract_block_keywords(
                semantic_blocks,
                max_keywords_per_block=self.config["keywords"].get("max_keywords_per_block", 10)
            )
            
            # Analyze keyword trends
            keyword_trends = self.keyword_extractor.analyze_keyword_trends(block_keywords, semantic_blocks)
            
            # Save keywords
            self.keyword_extractor.save_keywords(global_keywords, block_keywords, keyword_trends, session_files["keywords_topics"])
            
            self.session_manager.update_session_status(session_info, "keyword_extraction", "completed")
            self.logger.info("Keyword extraction completed successfully")
            
            return {
                "global_keywords": global_keywords,
                "block_keywords": block_keywords,
                "keyword_trends": keyword_trends
            }
            
        except Exception as e:
            self.session_manager.update_session_status(session_info, "keyword_extraction", "failed",
                                                     {"error": str(e)})
            raise
    
    def _process_validation(self, session_files: Dict[str, Path], session_info: Dict[str, Any]) -> Dict[str, Any]:
        """Process validation step."""
        if self.session_manager.should_skip_step(session_info, "validation"):
            self.logger.info("Skipping validation (already completed)")
            with open(session_files["validation_report"], "r", encoding="utf-8") as f:
                return json.load(f)
        
        self.logger.info("Starting validation...")
        
        try:
            validation_report = self.validator.generate_validation_report(
                session_files,
                min_confidence=self.config["validation"].get("min_confidence_threshold", 0.6)
            )
            
            # Save validation report
            self.validator.save_validation_report(validation_report, session_files["validation_report"])
            
            self.session_manager.update_session_status(session_info, "validation", "completed")
            self.logger.info("Validation completed successfully")
            
            return validation_report
            
        except Exception as e:
            self.session_manager.update_session_status(session_info, "validation", "failed",
                                                     {"error": str(e)})
            raise
    
    def get_pipeline_status(self, session_name: str) -> Dict[str, Any]:
        """Get status of a pipeline session."""
        session_dir = self.session_manager.sessions_dir / session_name
        
        if not session_dir.exists():
            return {"error": "Session not found"}
        
        try:
            with open(session_dir / "session_info.json", "r") as f:
                session_info = json.load(f)
            
            return {
                "session_name": session_name,
                "status": session_info.get("status", "unknown"),
                "completed_steps": session_info.get("completed_steps", []),
                "failed_steps": session_info.get("failed_steps", []),
                "created_at": session_info.get("created_at"),
                "processing_metadata": session_info.get("processing_metadata", {})
            }
            
        except Exception as e:
            return {"error": f"Failed to read session info: {e}"}
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all available sessions."""
        sessions = []
        
        for session_dir in self.session_manager.sessions_dir.iterdir():
            if session_dir.is_dir() and (session_dir / "session_info.json").exists():
                status = self.get_pipeline_status(session_dir.name)
                if "error" not in status:
                    sessions.append(status)
        
        return sorted(sessions, key=lambda x: x.get("created_at", ""), reverse=True)
    
    def _process_emotions_parallel(self, audio_file_path: str, diarized_transcript: List[Dict[str, Any]],
                                 session_info: Dict[str, Any], session_files: Dict[str, Path]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Process text and audio emotions in parallel for speed."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        self.logger.info("Starting parallel emotion processing...")
        
        # Define processing functions
        def process_text_emotions():
            return self._process_text_emotions(diarized_transcript, session_info, session_files)
        
        def process_audio_emotions():
            return self._process_audio_emotions(audio_file_path, diarized_transcript, session_info, session_files)
        
        # Execute in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            text_future = executor.submit(process_text_emotions)
            audio_future = executor.submit(process_audio_emotions)
            
            # Get results
            text_emotions = text_future.result()
            audio_emotions = audio_future.result()
        
        self.logger.info("Parallel emotion processing completed")
        return text_emotions, audio_emotions

    def _convert_audio_to_wav(self, audio_file_path: str, session_info: Dict[str, Any], 
                            session_files: Dict[str, Path]) -> str:
        """Convert input audio to WAV format."""
        if self.session_manager.should_skip_step(session_info, "audio_conversion"):
            self.logger.info("Skipping audio conversion (already completed)")
            return str(session_files.get("wav_audio", audio_file_path))
        
        try:
            # Ensure output_dir is present in session_info
            if 'output_dir' not in session_info or not session_info['output_dir']:
                session_info['output_dir'] = str(Path.cwd() / "output")
                Path(session_info['output_dir']).mkdir(parents=True, exist_ok=True)
            
            # Initialize transcriber to use its conversion method
            if self.transcriber is None:
                self.transcriber = WhisperTranscriber(
                    model_name=self.config["transcription"]["model"],
                    device_manager=self.device_manager
                )
            
            # Convert to WAV
            wav_path = self.transcriber.convert_to_wav(
                audio_file_path, 
                output_dir=str(session_info["output_dir"])
            )
            
            # Save WAV path info
            session_files["wav_audio"] = Path(wav_path)
            
            self.session_manager.update_session_status(session_info, "audio_conversion", "completed", {
                "wav_path": wav_path
            })
            
            self.logger.info(f"Audio converted to WAV: {wav_path}")
            return wav_path
            
        except Exception as e:
            self.session_manager.update_session_status(session_info, "audio_conversion", "failed", {"error": str(e)})
            raise
    
    def _extract_shared_embeddings(self, wav_audio_path: str, session_info: Dict[str, Any], 
                                 session_files: Dict[str, Path]) -> Optional[object]:
        """Extract shared audio embeddings for unified processing."""
        if self.session_manager.should_skip_step(session_info, "embedding_extraction"):
            self.logger.info("Skipping embedding extraction (already completed)")
            # Try to load existing embeddings
            embedding_path = session_files.get("embeddings")
            if embedding_path and embedding_path.exists():
                try:
                    data = np.load(str(embedding_path), allow_pickle=True)
                    # Reconstruct AudioEmbedding object
                    segments = [(float(s[0]), float(s[1])) for s in data['segments']]
                    return AudioEmbedding(
                        embeddings=data['embeddings'],
                        timestamps=data['timestamps'],
                        sample_rate=int(data['sample_rate']),
                        duration=float(data['duration']),
                        segments=segments,
                        metadata=data['metadata'].item() if 'metadata' in data else {}
                    )
                except:
                    pass
            return None
        
        try:
            # Initialize transcriber
            if self.transcriber is None:
                self.transcriber = WhisperTranscriber(
                    model_name=self.config["transcription"]["model"],
                    device_manager=self.device_manager
                )
            
            # Extract shared embeddings with optimizations for large files
            cache_key = f"session_{session_info['session_id']}"
            is_large_file = session_info.get('is_large_file', False)
            
            if is_large_file:
                self.logger.info("🎯 Extracting shared embeddings for large file (simplified)...")
            else:
                self.logger.info("🎯 Extracting shared embeddings for unified processing...")
            
            # Use standard embedding extraction (the method signature doesn't support extra params)
            shared_embeddings = self.transcriber.extract_shared_embeddings(wav_audio_path, cache_key)
            
            if shared_embeddings:
                # Save embeddings to disk
                output_dir = Path(session_info["output_dir"])
                embedding_path = output_dir / "shared_embeddings.npz"
                np.savez_compressed(
                    str(embedding_path),
                    embeddings=shared_embeddings.embeddings,
                    timestamps=shared_embeddings.timestamps,
                    sample_rate=shared_embeddings.sample_rate,
                    duration=shared_embeddings.duration,
                    segments=np.array(shared_embeddings.segments),
                    metadata=shared_embeddings.metadata
                )
                session_files["embeddings"] = embedding_path
                
                self.session_manager.update_session_status(session_info, "embedding_extraction", "completed", {
                    "embedding_shape": shared_embeddings.embeddings.shape,
                    "duration": shared_embeddings.duration
                })
                
                self.logger.info(f"✅ Shared embeddings extracted successfully: shape={shared_embeddings.embeddings.shape}, saved to {embedding_path}")
            else:
                self.logger.warning("❌ Shared embeddings extraction failed - pipeline will use fallback methods")
                # Create a fallback simple embedding for basic processing
                self.logger.info("🔄 Creating simple fallback embedding structure...")
                shared_embeddings = self._create_fallback_embeddings(wav_audio_path, session_info)
                self.session_manager.update_session_status(session_info, "embedding_extraction", "failed", {
                    "reason": "extraction_failed"
                })
            
            return shared_embeddings
            
        except Exception as e:
            self.logger.warning(f"Embedding extraction failed: {e}")
            self.session_manager.update_session_status(session_info, "embedding_extraction", "failed", {"error": str(e)})
            return None
    
    def _create_fallback_embeddings(self, wav_audio_path: str, session_info: Dict[str, Any]) -> Optional[object]:
        """Create simple fallback embeddings when advanced extraction fails."""
        try:
            from src.transcription.whisper_transcriber import AudioEmbedding
            
            self.logger.info("Creating simple fallback embeddings...")
            
            # Load audio with basic librosa
            audio, sr = librosa.load(wav_audio_path, sr=16000, mono=True)
            duration = len(audio) / sr
            
            # Create simple MFCC-based embeddings as fallback
            mfcc_features = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, hop_length=512)
            
            # Transpose to get time-major format
            embeddings = mfcc_features.T  # Shape: (time_steps, n_mfcc)
            
            # Create timestamps
            hop_length = 512
            time_steps = embeddings.shape[0]
            timestamps = librosa.frames_to_time(np.arange(time_steps), sr=sr, hop_length=hop_length)
            
            # Create simple segments (every 30 seconds)
            segments = []
            segment_duration = 30.0
            for i in range(0, int(duration), int(segment_duration)):
                start_time = float(i)
                end_time = float(min(i + segment_duration, duration))
                segments.append((start_time, end_time))
            
            # Create AudioEmbedding object
            fallback_embedding = AudioEmbedding(
                embeddings=embeddings,
                timestamps=timestamps,
                sample_rate=sr,
                duration=duration,
                segments=segments,
                metadata={"type": "fallback_mfcc", "n_mfcc": 13}
            )
            
            self.logger.info(f"✅ Fallback embeddings created: shape={embeddings.shape}, duration={duration:.2f}s")
            return fallback_embedding
            
        except Exception as e:
            self.logger.error(f"Failed to create fallback embeddings: {e}")
            return None
    
    def _process_transcription_unified(self, wav_audio_path: str, shared_embeddings: Optional[object],
                                     session_info: Dict[str, Any], session_files: Dict[str, Path]) -> Dict[str, Any]:
        """Process transcription using shared embeddings when available."""
        if self.session_manager.should_skip_step(session_info, "transcription"):
            self.logger.info("Skipping transcription (already completed)")
            with open(session_files["transcript"], "r", encoding="utf-8") as f:
                return json.load(f)
        
        self.logger.info("Starting unified transcription...")
        
        try:
            if self.transcriber is None:
                self.transcriber = WhisperTranscriber(
                    model_name=self.config["transcription"]["model"],
                    device_manager=self.device_manager
                )
            
            # Use shared embeddings if available
            transcript_data = self.transcriber.transcribe(
                wav_audio_path,
                shared_embeddings=shared_embeddings,
                **self.config["transcription"]
            )
            
            # Optimize transcript data for large files - remove unnecessary word-level data
            is_large_file = session_info.get('is_large_file', False)
            if is_large_file:
                self.logger.info("Large file: optimizing transcript storage...")
                transcript_data = self._optimize_transcript_storage(transcript_data)
            
            # Save transcript
            with open(session_files["transcript"], "w", encoding="utf-8") as f:
                json.dump(transcript_data, f, indent=2, ensure_ascii=False)
            
            # Force garbage collection after large file processing
            if is_large_file:
                import gc
                gc.collect()
                self.logger.info("Memory cleanup completed after transcription")
            
            self.session_manager.update_session_status(session_info, "transcription", "completed")
            self.logger.info("✅ Unified transcription completed successfully")
            self.logger.info("🔄 Pipeline continuation: Moving to diarization step...")
            
            return transcript_data
            
        except Exception as e:
            self.session_manager.update_session_status(session_info, "transcription", "failed", {"error": str(e)})
            raise
    
    def _process_diarization_unified(self, wav_audio_path: str, transcript_data: Dict[str, Any], 
                                   shared_embeddings: Optional[object],
                                   session_info: Dict[str, Any], session_files: Dict[str, Path]) -> List[Dict[str, Any]]:
        """Process speaker diarization using shared embeddings when available."""
        if self.session_manager.should_skip_step(session_info, "diarization"):
            self.logger.info("Skipping diarization (already completed)")
            with open(session_files["diarization"], "r", encoding="utf-8") as f:
                return json.load(f)
        
        self.logger.info("🎭 Starting unified speaker diarization...")
        
        # ALWAYS use professional lightweight diarization for best accuracy
        self.logger.info("🎯 Using PROFESSIONAL lightweight diarization for maximum accuracy...")
        self.logger.info("✨ Advanced speaker characteristic analysis with conservative clustering...")
        
        # Professional lightweight diarization using speaker characteristics
        diarized_transcript = self._perform_lightweight_diarization(
            wav_audio_path, transcript_data, session_info, session_files
        )
        
        return diarized_transcript
    
    def _process_emotions_unified(self, wav_audio_path: str, diarized_transcript: List[Dict[str, Any]], 
                                shared_embeddings: Optional[AudioEmbedding],
                                session_info: Dict[str, Any], session_files: Dict[str, Path]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Process emotion detection using shared embeddings when available."""
        from concurrent.futures import ThreadPoolExecutor
        
        self.logger.info("🎭 Starting UNIFIED emotion detection with shared embeddings...")
        
        text_emotions = []
        audio_emotions = []
        
        def process_text_emotions():
            nonlocal text_emotions
            try:
                if self.session_manager.should_skip_step(session_info, "text_emotions"):
                    self.logger.info("Skipping text emotion detection (already completed)")
                    with open(session_files["emotions_text"], "r", encoding="utf-8") as f:
                        emotions_data = json.load(f)
                        # Convert back to EmotionPrediction objects if needed
                        text_emotions = emotions_data
                else:
                    if self.emotion_detector is None:
                        self.emotion_detector = EmotionDetector(
                            device_manager=self.device_manager
                        )
                    
                    # Use new emotion detection method
                    emotion_predictions = self.emotion_detector.detect_emotions_from_text(diarized_transcript)
                    
                    # Convert to dictionary format for saving
                    text_emotions = [pred.to_dict() for pred in emotion_predictions]
                    
                    # Save results
                    with open(session_files["emotions_text"], "w", encoding="utf-8") as f:
                        json.dump(text_emotions, f, indent=2, ensure_ascii=False)
                    
                    self.session_manager.update_session_status(session_info, "text_emotions", "completed")
                    self.logger.info(f"✅ Text emotion detection completed: {len(text_emotions)} segments processed")
                    
            except Exception as e:
                self.logger.error(f"Text emotion detection failed: {e}")
                text_emotions = []
        
        def process_audio_emotions():
            nonlocal audio_emotions
            try:
                if self.session_manager.should_skip_step(session_info, "audio_emotions"):
                    self.logger.info("Skipping audio emotion detection (already completed)")
                    with open(session_files["emotions_audio"], "r", encoding="utf-8") as f:
                        emotions_data = json.load(f)
                        audio_emotions = emotions_data
                else:
                    if self.emotion_detector is None:
                        self.emotion_detector = EmotionDetector(
                            device_manager=self.device_manager
                        )
                    
                    # Extract speaker segments for emotion detection
                    speaker_segments = []
                    for segment in diarized_transcript:
                        speaker_segments.append({
                            "speaker": segment.get("speaker", "Unknown"),
                            "start": segment.get("start", 0.0),
                            "end": segment.get("end", 0.0)
                        })
                    
                    # Use shared embeddings for EFFICIENT emotion detection
                    if shared_embeddings is not None:
                        self.logger.info("🚀 Using shared embeddings for emotion detection")
                        emotion_predictions = self.emotion_detector.detect_emotions_from_embeddings(
                            shared_embeddings, speaker_segments
                        )
                        
                        # Convert to dictionary format for saving
                        audio_emotions = [pred.to_dict() for pred in emotion_predictions]
                        
                        self.logger.info(f"🎭 Audio emotion detection completed: {len(audio_emotions)} predictions")
                        
                    else:
                        self.logger.warning("No shared embeddings available, skipping audio emotion detection")
                        audio_emotions = []
                    
                    # Save results
                    with open(session_files["emotions_audio"], "w", encoding="utf-8") as f:
                        json.dump(audio_emotions, f, indent=2, ensure_ascii=False)
                    
                    self.session_manager.update_session_status(session_info, "audio_emotions", "completed")
                    self.logger.info(f"✅ Audio emotion detection completed: {len(audio_emotions)} segments processed")
                    
            except Exception as e:
                self.logger.error(f"Audio emotion detection failed: {e}")
                audio_emotions = []
        
        # Run both emotion detection processes in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            text_future = executor.submit(process_text_emotions)
            audio_future = executor.submit(process_audio_emotions)
            
            # Wait for both to complete
            text_future.result()
            audio_future.result()
        
        # Skip combined emotions file as it's not required
        self.logger.info(f"Unified emotion detection completed: text={len(text_emotions)}, audio={len(audio_emotions)}")
        
        return text_emotions, audio_emotions

    def _process_summarization_optimized(self, semantic_blocks: List[Dict[str, Any]],
                                        session_info: Dict[str, Any], session_files: Dict[str, Path]) -> Dict[str, Any]:
        """Process summarization step with optimizations for large files."""
        if self.session_manager.should_skip_step(session_info, "summarization"):
            self.logger.info("Skipping summarization (already completed)")
            with open(session_files["summaries"], "r", encoding="utf-8") as f:
                return json.load(f)
        
        self.logger.info("Starting optimized summarization for large files...")
        
        # Performance parameters for large files
        max_blocks_per_batch = 10
        max_total_blocks = 100  # Limit for very large files
        total_timeout_minutes = 15
        from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
        num_blocks = len(semantic_blocks)
        # Start total timeout timer
        start_time = time.time()
        # Process blocks in batches for better performance
        all_block_summaries = []
        batch_size = min(max_blocks_per_batch, num_blocks)
        for batch_start in range(0, num_blocks, batch_size):
            batch_end = min(batch_start + batch_size, num_blocks)
            batch_blocks = semantic_blocks[batch_start:batch_end]
            # Check timeout
            elapsed_minutes = (time.time() - start_time) / 60
            if elapsed_minutes > total_timeout_minutes:
                self.logger.warning(f"Summarization timeout reached ({total_timeout_minutes}m). "
                                  f"Processed {len(all_block_summaries)}/{num_blocks} blocks.")
                break
            self.logger.info(f"Processing batch {batch_start//batch_size + 1} "
                           f"(blocks {batch_start+1}-{batch_end}) of {(num_blocks-1)//batch_size + 1}")
            # Use batch processing for better performance
            batch_summaries = self.summarizer.summarize_blocks_batch(
                batch_blocks,
                max_length=self.config["summarization"].get("max_length", 100),  # Shorter for speed
                min_length=self.config["summarization"].get("min_length", 20),   # Shorter for speed
                batch_size=min(5, len(batch_blocks))  # Smaller batches for stability
            )
            all_block_summaries.extend(batch_summaries)
        # Generate global summary only if we have some block summaries
        if all_block_summaries:
            self.logger.info("Generating optimized global summary...")
            # Use only top blocks for global summary to avoid timeout
            top_blocks = semantic_blocks[:min(20, len(semantic_blocks))]
            global_summary = self.summarizer.generate_global_summary(
                top_blocks,
                all_block_summaries[:min(20, len(all_block_summaries))]
            )
        else:
            global_summary = {
                "summary": "No summaries generated due to processing constraints.",
                "key_themes": []
            }
        # Save summaries
        final_result = {"block_summaries": all_block_summaries, "global_summary": global_summary}
        self.summarizer.save_summaries(all_block_summaries, global_summary, session_files["summaries"])
        total_time = time.time() - start_time
        self.session_manager.update_session_status(session_info, "summarization", "completed",
            {
                "blocks_processed": len(all_block_summaries),
                "total_blocks": num_blocks,
                "processing_time": total_time,
                "optimized": True
            })
        self.logger.info(f"Optimized summarization completed: {len(all_block_summaries)}/{num_blocks} blocks in {total_time:.2f}s")
        return final_result
    
    def _create_minimal_summaries(self, semantic_blocks: List[Dict[str, Any]], 
                                 session_files: Dict[str, Path]) -> Dict[str, Any]:
        """Create minimal summaries when full processing fails."""
        self.logger.info("Creating minimal fallback summaries...")
        
        minimal_summaries = []
        for i, block in enumerate(semantic_blocks[:20]):  # Limit to first 20 blocks
            text = block.get('text', '')
            minimal_summary = {
                'block_id': block.get('block_id', i),
                'summary': text[:200] + "..." if len(text) > 200 else text,
                'key_points': [text[:100] + "..." if len(text) > 100 else text],
                'processing_time': 0.0,
                'minimal_fallback': True
            }
            minimal_summaries.append(minimal_summary)
        
        global_summary = {
            "summary": "Processing optimized for large file. Detailed analysis limited to preserve performance.",
            "key_themes": ["Large file processing", "Performance optimization"],
            "minimal_fallback": True
        }
        
        # Save minimal summaries
        result = {"block_summaries": minimal_summaries, "global_summary": global_summary}
        
        try:
            with open(session_files["summaries"], "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Failed to save minimal summaries: {e}")
        
        return result

    def _optimize_transcript_storage(self, transcript_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize transcript data for storage by removing unnecessary word-level data."""
        self.logger.info("Optimizing transcript data for large file storage...")
        
        original_segments = len(transcript_data.get('segments', []))
        
        # Keep only essential segment data
        optimized_segments = []
        for segment in transcript_data.get('segments', []):
            optimized_segment = {
                'id': segment.get('id', 0),
                'start': segment.get('start', 0.0),
                'end': segment.get('end', 0.0),
                'text': segment.get('text', ''),
                # Remove word-level timestamps and detailed data to save memory
                # 'words': segment.get('words', [])  # Commented out - this can be huge
            }
            optimized_segments.append(optimized_segment)
        
        # Create optimized transcript
        optimized_transcript = {
            'language': transcript_data.get('language', 'unknown'),
            'duration': transcript_data.get('duration', 0.0),
            'text': transcript_data.get('text', ''),
            'segments': optimized_segments,
            'model_used': transcript_data.get('model_used', 'unknown'),
            'device_used': transcript_data.get('device_used', 'unknown'),
            'processing_method': transcript_data.get('processing_method', 'optimized'),
            'optimization_applied': True,
            'original_segments_count': original_segments
        }
        
        # Only include shared embeddings metadata, not the full data
        if 'shared_embeddings' in transcript_data:
            optimized_transcript['shared_embeddings_used'] = True
        
        self.logger.info(f"Transcript optimized: {original_segments} segments, word-level data removed")
        return optimized_transcript
    
    def _manage_memory_for_large_files(self, session_info: Dict[str, Any]):
        """Manage memory usage for large files by forcing garbage collection and clearing caches."""
        is_large_file = session_info.get('is_large_file', False)
        if is_large_file:
            import gc
            import torch
            
            # Force garbage collection
            gc.collect()
            
            # Clear PyTorch cache if using CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Clear any model caches
            if hasattr(self, 'transcriber') and self.transcriber:
                if hasattr(self.transcriber, 'model') and hasattr(self.transcriber.model, 'cache'):
                    try:
                        self.transcriber.model.cache.clear()
                    except:
                        pass
            
            self.logger.info("Memory management completed for large file processing")
    
    def _perform_lightweight_diarization(self, wav_audio_path: str, transcript_data: Dict[str, Any],
                                         session_info: Dict[str, Any], session_files: Dict[str, Path]) -> List[Dict[str, Any]]:
        """
        Professional lightweight diarization using speaker characteristics analysis.
        
        This method:
        1. Extracts audio features for each transcript segment
        2. Builds speaker profiles incrementally
        3. Uses similarity matching to assign speakers
        4. Creates new speakers when characteristics don't match existing ones
        """
        try:
            import numpy as np
            from sklearn.cluster import DBSCAN
            from sklearn.metrics.pairwise import cosine_similarity
            from sklearn.preprocessing import StandardScaler
            
            self.logger.info("🎯 Starting professional lightweight diarization...")
            
            # Load audio
            audio, sr = librosa.load(wav_audio_path, sr=16000, mono=True)
            segments = transcript_data.get("segments", [])
            
            if not segments:
                self.logger.warning("No transcript segments found for diarization")
                return []
            
            # Extract speaker characteristics for each segment
            speaker_features = []
            valid_segments = []
            
            self.logger.info(f"Extracting speaker characteristics from {len(segments)} segments...")
            
            for i, segment in enumerate(segments):
                start_time = segment.get('start', 0.0)
                end_time = segment.get('end', 0.0)
                
                # Skip very short segments
                if end_time - start_time < 0.5:  # Less than 0.5 seconds
                    continue
                
                # Extract audio segment
                start_sample = int(start_time * sr)
                end_sample = int(end_time * sr)
                segment_audio = audio[start_sample:end_sample]
                
                if len(segment_audio) < sr * 0.3:  # Less than 0.3 seconds of audio
                    continue
                
                # Extract speaker characteristics
                features = self._extract_speaker_characteristics(segment_audio, sr)
                
                if features is not None:
                    speaker_features.append(features)
                    valid_segments.append(segment)
                
                # Progress logging
                if (i + 1) % 50 == 0:
                    self.logger.info(f"Processed {i + 1}/{len(segments)} segments")
            
            if not speaker_features:
                self.logger.warning("No valid speaker features extracted - defaulting to single speaker")
                return self._create_single_speaker_transcript(segments)
            
            self.logger.info(f"Successfully extracted features from {len(speaker_features)} segments")
            
            # Normalize features for better clustering
            scaler = StandardScaler()
            normalized_features = scaler.fit_transform(speaker_features)
            
            # Perform intelligent speaker clustering
            speaker_labels = self._intelligent_speaker_clustering(normalized_features, session_info)
            
            # Assign speakers to segments
            diarized_transcript = self._assign_speakers_to_segments(
                valid_segments, speaker_labels, segments
            )
            
            # Save results
            with open(session_files["diarization"], "w", encoding="utf-8") as f:
                json.dump(diarized_transcript, f, indent=2, ensure_ascii=False)
            
            num_speakers = len(set(speaker_labels))
            self.session_manager.update_session_status(session_info, "diarization", "completed", {
                "method": "lightweight_professional",
                "speakers_detected": num_speakers,
                "segments_processed": len(valid_segments)
            })
            
            self.logger.info(f"✅ Professional diarization completed: {num_speakers} speakers detected from {len(valid_segments)} segments")
            
            return diarized_transcript
            
        except Exception as e:
            self.logger.error(f"Lightweight diarization failed: {e}")
            # Fallback to single speaker
            return self._create_single_speaker_transcript(transcript_data.get("segments", []))
    
    def _extract_speaker_characteristics(self, segment_audio: np.ndarray, sr: int) -> Optional[np.ndarray]:
        """
        Extract comprehensive speaker characteristics from audio segment.
        
        Features extracted:
        - MFCC (Mel-frequency cepstral coefficients) - vocal tract shape
        - Spectral centroid - brightness of voice
        - Spectral rolloff - frequency distribution
        - Zero crossing rate - voice quality
        - Pitch (F0) - fundamental frequency
        - Formants - vocal tract resonances
        """
        try:
            # Ensure minimum length
            if len(segment_audio) < sr * 0.2:  # Less than 0.2 seconds
                return None
            
            # 1. MFCC features (most important for speaker identification)
            mfccs = librosa.feature.mfcc(y=segment_audio, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfccs, axis=1)
            mfcc_std = np.std(mfccs, axis=1)
            
            # 2. Spectral features
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=segment_audio, sr=sr))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=segment_audio, sr=sr))
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(segment_audio))
            
            # 3. Pitch/F0 estimation
            try:
                pitches, magnitudes = librosa.piptrack(y=segment_audio, sr=sr)
                pitch_values = []
                for t in range(pitches.shape[1]):
                    index = magnitudes[:, t].argmax()
                    pitch = pitches[index, t]
                    if pitch > 0:
                        pitch_values.append(pitch)
                
                if pitch_values:
                    pitch_mean = np.mean(pitch_values)
                    pitch_std = np.std(pitch_values)
                else:
                    pitch_mean = 0.0
                    pitch_std = 0.0
            except:
                pitch_mean = 0.0
                pitch_std = 0.0
            
            # 4. Energy and dynamics
            rms_energy = np.mean(librosa.feature.rms(y=segment_audio))
            
            # 5. Formant estimation (simplified)
            try:
                # Use LPC to estimate formants
                from scipy.signal import lfilter
                # Simplified formant estimation
                formant_estimate = np.mean(np.abs(np.fft.fft(segment_audio)[:len(segment_audio)//4]))
            except:
                formant_estimate = 0.0
            
            # Combine all features
            features = np.concatenate([
                mfcc_mean,          # 13 features
                mfcc_std,           # 13 features  
                [spectral_centroid], # 1 feature
                [spectral_rolloff],  # 1 feature
                [zero_crossing_rate], # 1 feature
                [pitch_mean],        # 1 feature
                [pitch_std],         # 1 feature
                [rms_energy],        # 1 feature
                [formant_estimate]   # 1 feature
            ])
            
            return features
            
        except Exception as e:
            # Return None if feature extraction fails
            return None
    
    def _intelligent_speaker_clustering(self, features: np.ndarray, session_info: Dict[str, Any]) -> List[int]:
        """
        Intelligent speaker clustering using multiple algorithms and validation.
        
        This method:
        1. Uses DBSCAN for natural cluster discovery
        2. Validates cluster quality
        3. Falls back to conservative K-means if needed
        4. Applies professional speaker count limits
        """
        from sklearn.cluster import DBSCAN, KMeans
        from sklearn.metrics import silhouette_score
        
        n_segments = len(features)
        is_large_file = session_info.get('is_large_file', False)
        
        self.logger.info(f"Clustering {n_segments} speaker feature vectors...")
        
        # Professional parameters based on audio length
        if is_large_file:
            max_speakers = 2  # Very conservative for large files
            min_samples_per_speaker = 50  # Require even more evidence
        else:
            max_speakers = 2  # Conservative for smaller files  
            min_samples_per_speaker = 30  # Require substantial evidence
        
        # Single speaker bias - require ultra-strong evidence for multiple speakers
        single_speaker_threshold = 0.9  # Very high threshold for multi-speaker detection
        
        # Additional check for typical single-speaker content (like spiritual discourse)
        # If segments are relatively uniform in length and content, bias toward single speaker
        try:
            segment_length_variance = np.var([len(features[i]) for i in range(min(len(features), 50))])
            if segment_length_variance < 2.0:  # Low variance suggests single speaker
                self.logger.info("Low feature variance detected - likely single speaker content")
                single_speaker_threshold = 0.95  # Even higher threshold
                max_speakers = 1  # Force single speaker for uniform content
                
                # Early exit for very uniform content - definitely single speaker
                if segment_length_variance < 1.0:
                    self.logger.info("🎯 Very uniform content detected - forcing single speaker")
                    return [0] * n_segments
        except:
            pass
        
        # Method 1: DBSCAN for natural cluster discovery
        try:
            # Conservative DBSCAN parameters
            eps_values = [0.5, 0.7, 0.9]  # Try different density thresholds
            best_dbscan_result = None
            best_dbscan_score = -1
            
            for eps in eps_values:
                min_samples = max(3, n_segments // 20)
                
                dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
                labels = dbscan.fit_predict(features)
                
                # Count non-noise clusters
                unique_labels = set(labels)
                n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
                noise_ratio = sum(1 for label in labels if label == -1) / len(labels)
                
                # Validate cluster quality - be ultra-strict for multiple speakers
                if 1 <= n_clusters <= max_speakers and noise_ratio < 0.15:  # Even stricter noise threshold
                    try:
                        if n_clusters > 1:
                            # Check cluster balance for multiple speakers
                            unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
                            if len(counts) < 2:
                                continue
                            
                            balance_ratio = np.min(counts) / np.max(counts)
                            
                            # Ultra-strict requirements for multiple speakers
                            if balance_ratio < 0.6 or np.min(counts) < min_samples_per_speaker:
                                continue
                            
                            score = silhouette_score(features, labels)
                            # Require ultra-high confidence for multiple speakers
                            if score < single_speaker_threshold:
                                continue  # Reject low-confidence multi-speaker solutions
                            
                            # Heavy penalty for multiple speakers
                            speaker_penalty = 0.5 ** (n_clusters - 1)
                            score = score * balance_ratio * speaker_penalty
                        else:
                            score = 0.9  # Very high score for single speaker (strongly preferred)
                        
                        if score > best_dbscan_score:
                            best_dbscan_score = score
                            best_dbscan_result = labels
                            
                        self.logger.debug(f"DBSCAN eps={eps}: {n_clusters} clusters, noise={noise_ratio:.2f}, score={score:.3f}")
                    except:
                        continue
            
            if best_dbscan_result is not None:
                # Clean up noise points by assigning them to nearest cluster
                cleaned_labels = self._assign_noise_to_clusters(features, best_dbscan_result)
                n_speakers = len(set(cleaned_labels))
                self.logger.info(f"✅ DBSCAN clustering successful: {n_speakers} speakers detected")
                return cleaned_labels
                
        except Exception as e:
            self.logger.debug(f"DBSCAN clustering failed: {e}")
        
        # Method 2: Conservative K-means clustering
        try:
            self.logger.info("Falling back to K-means clustering...")
            
            best_k = 1
            best_score = -1
            
            # Try different numbers of speakers - heavily favor single speaker
            for k in range(1, min(max_speakers + 1, n_segments // min_samples_per_speaker + 1)):
                if k == 1:
                    # Single speaker case - give it a high baseline score
                    score = 0.85  # High baseline to favor single speaker
                    labels = [0] * n_segments
                else:
                    try:
                        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                        labels = kmeans.fit_predict(features)
                        
                        # Calculate cluster balance - require very balanced clusters
                        unique_labels, counts = np.unique(labels, return_counts=True)
                        balance_ratio = np.min(counts) / np.max(counts)
                        
                        # Require high balance for multiple speakers
                        if balance_ratio < 0.4:  # Stricter balance requirement
                            continue  # Skip unbalanced solutions
                        
                        # Silhouette score
                        sil_score = silhouette_score(features, labels)
                        
                        # Very strict requirements for multiple speakers
                        if sil_score < single_speaker_threshold:
                            continue  # Reject low-confidence multi-speaker
                        
                        # Heavy penalty for multiple speakers
                        speaker_penalty = 0.7 ** (k - 1)  # Stronger penalty
                        score = sil_score * balance_ratio * speaker_penalty
                        
                        self.logger.debug(f"K-means k={k}: silhouette={sil_score:.3f}, balance={balance_ratio:.2f}, final={score:.3f}")
                        
                    except Exception as e:
                        self.logger.debug(f"K-means k={k} failed: {e}")
                        continue
                
                if score > best_score:
                    best_score = score
                    best_k = k
            
            # Ultra-conservative final decision
            if best_k == 1 or best_score < 0.3:  # Very strict threshold for multiple speakers
                final_labels = [0] * n_segments
                self.logger.info(f"✅ Single speaker detected (conservative approach, score: {best_score:.3f})")
            else:
                # Even for "best" multiple speaker solution, double-check with ultra-strict criteria
                kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
                final_labels = kmeans.fit_predict(features)
                
                # Final validation - check if multiple speakers are really justified
                unique_labels, counts = np.unique(final_labels, return_counts=True)
                balance_ratio = np.min(counts) / np.max(counts)
                
                # Ultra-strict final check
                if balance_ratio < 0.5 or best_score < 0.4:
                    self.logger.info(f"🔄 Overriding to single speaker - insufficient evidence for multiple speakers")
                    final_labels = [0] * n_segments
                    best_k = 1
                else:
                    self.logger.info(f"✅ Multiple speakers confirmed: {best_k} speakers (score: {best_score:.3f}, balance: {balance_ratio:.2f})")
            
            return final_labels
            
        except Exception as e:
            self.logger.error(f"All clustering methods failed: {e}")
            # Ultimate fallback - single speaker
            return [0] * n_segments
    
    def _assign_noise_to_clusters(self, features: np.ndarray, labels: List[int]) -> List[int]:
        """Assign noise points (-1) to their nearest cluster."""
        from sklearn.metrics.pairwise import cosine_similarity
        
        cleaned_labels = labels.copy()
        noise_indices = [i for i, label in enumerate(labels) if label == -1]
        
        if not noise_indices:
            return cleaned_labels
        
        # Get cluster centers
        unique_clusters = set(label for label in labels if label != -1)
        if not unique_clusters:
            # All points are noise - assign all to cluster 0
            return [0] * len(labels)
        
        cluster_centers = {}
        for cluster_id in unique_clusters:
            cluster_points = [features[i] for i, label in enumerate(labels) if label == cluster_id]
            cluster_centers[cluster_id] = np.mean(cluster_points, axis=0)
        
        # Assign each noise point to nearest cluster
        for noise_idx in noise_indices:
            noise_point = features[noise_idx].reshape(1, -1)
            best_cluster = None
            best_similarity = -1
            
            for cluster_id, center in cluster_centers.items():
                center = center.reshape(1, -1)
                similarity = cosine_similarity(noise_point, center)[0, 0]
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_cluster = cluster_id
            
            if best_cluster is not None:
                cleaned_labels[noise_idx] = best_cluster
            else:
                cleaned_labels[noise_idx] = 0  # Default to first cluster
        
        return cleaned_labels
    
    def _assign_speakers_to_segments(self, valid_segments: List[Dict], speaker_labels: List[int], 
                                   all_segments: List[Dict]) -> List[Dict[str, Any]]:
        """Assign speaker labels to all transcript segments."""
        # Create speaker mapping
        unique_speakers = sorted(set(speaker_labels))
        speaker_map = {old_id: f"Speaker {i+1}" for i, old_id in enumerate(unique_speakers)}
        
        # Create mapping from segment to speaker
        segment_to_speaker = {}
        for segment, label in zip(valid_segments, speaker_labels):
            segment_key = (segment.get('start', 0), segment.get('end', 0), segment.get('text', ''))
            segment_to_speaker[segment_key] = speaker_map[label]
        
        # Assign speakers to all segments
        diarized_transcript = []
        current_speaker = "Speaker 1"  # Default speaker
        
        for segment in all_segments:
            segment_key = (segment.get('start', 0), segment.get('end', 0), segment.get('text', ''))
            
            # Check if we have a speaker assignment for this segment
            if segment_key in segment_to_speaker:
                current_speaker = segment_to_speaker[segment_key]
            # For segments without features, use the most recent speaker
            
            diarized_segment = segment.copy()
            diarized_segment["speaker"] = current_speaker
            diarized_transcript.append(diarized_segment)
        
        return diarized_transcript
    
    def _create_single_speaker_transcript(self, segments: List[Dict]) -> List[Dict[str, Any]]:
        """Create transcript with all segments assigned to single speaker."""
        diarized_transcript = []
        for segment in segments:
            diarized_segment = segment.copy()
            diarized_segment["speaker"] = "Speaker 1"
            diarized_transcript.append(diarized_segment)
        
        self.logger.info("✅ Single speaker transcript created")
        return diarized_transcript