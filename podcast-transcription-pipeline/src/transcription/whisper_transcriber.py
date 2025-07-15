from dataclasses import dataclass
import whisper
import torch
import json
import numpy as np
import librosa
import soundfile as sf
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging
from pydub import AudioSegment
from ..utils.device_manager import DeviceManager

@dataclass
class AudioEmbedding:
    """Container for audio embeddings and metadata."""
    embeddings: np.ndarray  # The actual embeddings
    timestamps: np.ndarray  # Corresponding timestamps
    sample_rate: int
    duration: float
    segments: List[Tuple[float, float]]  # List of (start, end) segments
    metadata: Dict[str, Any]

    def to_dict(self):
        """Convert AudioEmbedding to a JSON-serializable dict."""
        return {
            "embeddings": self.embeddings.tolist(),
            "timestamps": self.timestamps.tolist(),
            "sample_rate": self.sample_rate,
            "duration": self.duration,
            "segments": self.segments,
            "metadata": self.metadata
        }

class WhisperTranscriber:
    """High-speed audio transcription using optimized Whisper with unified audio processing."""
    
    def __init__(self, model_name: str = "base", device_manager: DeviceManager = None):
        # Use faster models for speed optimization
        self.speed_models = {
            "fastest": "tiny",      # ~32x faster, good quality
            "fast": "base",         # ~16x faster, better quality
            "balanced": "small",    # ~6x faster, high quality
            "quality": "medium",    # ~2x faster, very high quality
            "best": "large-v3"      # Original speed, best quality
        }
        
        # Default to fast model
        self.model_name = self.speed_models.get("quality", model_name)
        self.device_manager = device_manager or DeviceManager()
        self.model = None
        self.logger = logging.getLogger(__name__)
        
        # Enable optimizations with fallback
        self.use_chunking = True
        self.chunk_duration = 30  # Process in 30-second chunks
        self.parallel_chunks = False  # Disable parallel for stability
        
        # Unified audio processing
        self.target_sample_rate = 16000
        self.wav2vec2_model = None
        self.wav2vec2_processor = None
        self._embedding_cache = {}
        
    def convert_to_wav(self, audio_path: str, output_dir: Optional[str] = None) -> str:
        """
        Convert any audio/video format to WAV format before processing.
        If output_dir is None or invalid, fallback to current working directory.
        
        Args:
            audio_path: Path to input audio/video file
            output_dir: Output directory for converted file
            
        Returns:
            Path to converted WAV file
        """
        try:
            input_path = Path(audio_path)
            # Fallback if output_dir is None or not a valid path
            if not output_dir:
                output_dir = Path.cwd() / "converted_audio"
                output_dir.mkdir(parents=True, exist_ok=True)
            else:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{input_path.stem}_converted.wav"
            
            # Check if already WAV and properly formatted
            if input_path.suffix.lower() == '.wav':
                try:
                    # Verify it's properly formatted
                    audio, sr = librosa.load(str(input_path), sr=self.target_sample_rate)
                    if sr == self.target_sample_rate:
                        self.logger.info(f"Audio already in correct WAV format: {input_path}")
                        return str(input_path)
                except:
                    # If loading fails, need to convert
                    pass
            
            self.logger.info(f"Converting {input_path.suffix} to WAV: {input_path} -> {output_path}")
            
            # Use pydub for robust format conversion
            try:
                audio_segment = AudioSegment.from_file(str(input_path))
                
                # Convert to WAV with target sample rate
                audio_segment = audio_segment.set_frame_rate(self.target_sample_rate)
                audio_segment = audio_segment.set_channels(1)  # Mono
                audio_segment = audio_segment.set_sample_width(2)  # 16-bit
                
                audio_segment.export(str(output_path), format="wav")
                
                self.logger.info(f"Successfully converted to WAV: {output_path}")
                return str(output_path)
                
            except Exception as pydub_error:
                self.logger.warning(f"Pydub conversion failed: {pydub_error}")
                
                # Fallback to librosa + soundfile
                try:
                    audio, sr = librosa.load(str(input_path), sr=self.target_sample_rate, mono=True)
                    sf.write(str(output_path), audio, self.target_sample_rate)
                    
                    self.logger.info(f"Successfully converted using librosa: {output_path}")
                    return str(output_path)
                    
                except Exception as librosa_error:
                    self.logger.error(f"All conversion methods failed. Pydub: {pydub_error}, Librosa: {librosa_error}")
                    raise
                    
        except Exception as e:
            self.logger.error(f"Audio conversion failed: {e}")
            raise
    
    def load_embedding_models(self):
        """Load the shared embedding models for unified processing with advanced fallback logic."""
        if self.wav2vec2_model is None:
            # Advanced model loading with multiple fallback strategies
            model_candidates = [
                "facebook/wav2vec2-large-xlsr-53",
                "facebook/wav2vec2-base-960h", 
                "facebook/wav2vec2-large-960h",
                "microsoft/wavlm-base-plus"
            ]
            
            # Advanced device resolution
            target_device = self._resolve_embedding_device()
            self.logger.info(f"Target device for embeddings: {target_device}")
            
            for model_name in model_candidates:
                try:
                    self.logger.info(f"Attempting to load embedding model: {model_name}")
                    
                    # Import with error handling
                    try:
                        from transformers import Wav2Vec2Model, Wav2Vec2Processor
                    except ImportError as import_err:
                        self.logger.error(f"Transformers library not available: {import_err}")
                        self.logger.info("Try: pip install transformers")
                        self.wav2vec2_model = None
                        return
                    
                    # Load processor first (lighter operation)
                    self.logger.info(f"Loading processor for {model_name}...")
                    self.wav2vec2_processor = Wav2Vec2Processor.from_pretrained(
                        model_name, 
                        cache_dir=None,
                        use_fast=True,
                        trust_remote_code=False
                    )
                    
                    # Load model with memory optimization
                    self.logger.info(f"Loading model weights for {model_name}...")
                    self.wav2vec2_model = Wav2Vec2Model.from_pretrained(
                        model_name,
                        cache_dir=None,
                        torch_dtype=torch.float32,
                        low_cpu_mem_usage=True,
                        trust_remote_code=False
                    )
                    
                    # Move to target device with error handling
                    try:
                        self.wav2vec2_model = self.wav2vec2_model.to(target_device)
                        self.wav2vec2_model.eval()
                        
                        # Test the model with a small dummy input
                        self._test_embedding_model(target_device)
                        
                        self.logger.info(f"✅ Successfully loaded {model_name} on {target_device}")
                        return  # Success - exit the loop
                        
                    except Exception as device_err:
                        self.logger.warning(f"Device placement failed for {model_name}: {device_err}")
                        # Try CPU fallback
                        if target_device != "cpu":
                            try:
                                self.wav2vec2_model = self.wav2vec2_model.to("cpu")
                                self.wav2vec2_model.eval()
                                self._test_embedding_model("cpu")
                                self.logger.info(f"✅ Successfully loaded {model_name} on CPU (fallback)")
                                return
                            except Exception as cpu_err:
                                self.logger.warning(f"CPU fallback also failed: {cpu_err}")
                        
                        # Clean up failed attempt
                        self.wav2vec2_model = None
                        self.wav2vec2_processor = None
                        continue
                        
                except Exception as model_err:
                    self.logger.warning(f"Failed to load {model_name}: {model_err}")
                    self.wav2vec2_model = None
                    self.wav2vec2_processor = None
                    continue
            
            # If all models failed
            self.logger.error("❌ All embedding models failed to load. Embedding extraction disabled.")
            self.wav2vec2_model = None
            self.wav2vec2_processor = None
    
    def _resolve_embedding_device(self) -> str:
        """Advanced device resolution with comprehensive fallback logic."""
        try:
            # Get device from device manager
            raw_device = getattr(self.device_manager, 'device', None)
            self.logger.info(f"DeviceManager.device: {raw_device} (type: {type(raw_device)})")
            
            # Handle different device types
            if isinstance(raw_device, torch.device):
                device_str = str(raw_device).lower()
            elif isinstance(raw_device, str):
                device_str = raw_device.lower()
            elif hasattr(raw_device, 'type'):
                device_str = str(raw_device.type).lower()
            else:
                device_str = "cpu"
            
            # Clean device string
            if "cuda" in device_str:
                # Check CUDA availability
                if torch.cuda.is_available():
                    return "cuda"
                else:
                    self.logger.warning("CUDA requested but not available, falling back to CPU")
                    return "cpu"
            elif device_str in ["cpu"]:
                return "cpu"
            else:
                self.logger.warning(f"Unknown device '{device_str}', defaulting to CPU")
                return "cpu"
                
        except Exception as e:
            self.logger.error(f"Device resolution failed: {e}, defaulting to CPU")
            return "cpu"
    
    def _test_embedding_model(self, device: str):
        """Test the embedding model with a small dummy input to ensure it works."""
        try:
            # Create dummy audio (1 second of silence)
            dummy_audio = np.zeros(16000, dtype=np.float32)
            
            # Process with the model
            inputs = self.wav2vec2_processor(
                dummy_audio,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
            
            # Move inputs to device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                outputs = self.wav2vec2_model(**inputs)
                embeddings = outputs.last_hidden_state
            
            # Verify output shape
            if embeddings.shape[0] != 1 or embeddings.shape[2] < 100:
                raise ValueError(f"Unexpected embedding shape: {embeddings.shape}")
            
            self.logger.info(f"Model test passed - embedding shape: {embeddings.shape}")
            
        except Exception as e:
            raise RuntimeError(f"Model test failed: {e}")
    
    def extract_shared_embeddings(self, audio_path: str, cache_key: Optional[str] = None) -> Optional[AudioEmbedding]:
        """
        Extract shared audio embeddings for use across all processing stages.
        
        Args:
            audio_path: Path to WAV audio file
            cache_key: Optional key for caching embeddings
            
        Returns:
            AudioEmbedding object or None if extraction fails
        """
        # Check cache first
        if cache_key and cache_key in self._embedding_cache:
            self.logger.info(f"Using cached embeddings for {cache_key}")
            return self._embedding_cache[cache_key]
        
        try:
            self.load_embedding_models()
            
            if self.wav2vec2_model is None:
                self.logger.warning("Wav2Vec2 model not available, skipping embedding extraction")
                return None
            
            self.logger.info(f"Extracting shared audio embeddings from: {audio_path}")
            
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.target_sample_rate, mono=True)
            
            # Debug logging for audio loading
            self.logger.info(f"Audio loading result: audio.shape={audio.shape if hasattr(audio, 'shape') else 'no shape'}, sr={sr} (type: {type(sr)})")
            
            # Ensure sr is an integer to prevent division errors
            if not isinstance(sr, int):
                try:
                    sr = int(float(sr))
                    self.logger.info(f"Converted sr to int: {sr}")
                except Exception as conv_error:
                    self.logger.error(f"Sample rate (sr) conversion failed: {conv_error}, sr={sr}")
                    raise ValueError(f"Sample rate (sr) must be an integer, got {type(sr)}: {sr}")
            
            duration = len(audio) / sr
            
            self.logger.info(f"Audio loaded: duration={duration:.2f}s, sample_rate={sr}")
            
            # Process in chunks to manage memory
            chunk_size = int(self.chunk_duration * sr)
            overlap_size = int(2.0 * sr)  # 2-second overlap
            
            all_embeddings = []
            all_timestamps = []
            segments = []
            
            for i in range(0, len(audio), chunk_size - overlap_size):
                chunk_start = i
                chunk_end = min(i + chunk_size, len(audio))
                chunk_audio = audio[chunk_start:chunk_end]
                
                # Skip very short chunks
                if len(chunk_audio) < sr * 1.0:  # Less than 1 second
                    continue
                
                chunk_start_time = float(chunk_start / sr)
                chunk_end_time = float(chunk_end / sr)
                
                # Extract embeddings for this chunk
                chunk_embeddings = self._extract_chunk_embeddings(chunk_audio, sr)
                
                if chunk_embeddings is not None:
                    # Create timestamps for this chunk
                    chunk_timestamps = np.linspace(chunk_start_time, chunk_end_time, len(chunk_embeddings))
                    
                    all_embeddings.append(chunk_embeddings)
                    all_timestamps.append(chunk_timestamps)
                    segments.append((chunk_start_time, chunk_end_time))
            
            if not all_embeddings:
                self.logger.warning("No embeddings could be extracted from audio")
                return None
            
            # Concatenate all embeddings
            final_embeddings = np.concatenate(all_embeddings, axis=0)
            final_timestamps = np.concatenate(all_timestamps)
            
            self.logger.info(f"Shared embeddings extracted: shape={final_embeddings.shape}, duration={duration:.2f}s")
            
            # Create AudioEmbedding object
            audio_embedding = AudioEmbedding(
                embeddings=final_embeddings,
                timestamps=final_timestamps,
                sample_rate=int(sr),
                duration=float(duration),
                segments=segments,
                metadata={
                    "model": "facebook/wav2vec2-large-xlsr-53",
                    "chunk_duration": float(self.chunk_duration),
                    "device": str(self.device_manager.device),
                    "embedding_dim": int(final_embeddings.shape[-1])
                }
            )
            
            # Cache the result
            if cache_key:
                self._embedding_cache[cache_key] = audio_embedding
            
            return audio_embedding
            
        except Exception as e:
            self.logger.error(f"Shared embedding extraction failed: {e}")
            return None
    
    def _extract_chunk_embeddings(self, audio_chunk: np.ndarray, sr: int) -> Optional[np.ndarray]:
        """Extract embeddings from a single audio chunk with robust error handling."""
        try:
            # Validate input
            if len(audio_chunk) == 0:
                self.logger.warning("Empty audio chunk, skipping")
                return None
            
            # Preprocess audio for wav2vec2
            inputs = self.wav2vec2_processor(
                audio_chunk, 
                sampling_rate=sr, 
                return_tensors="pt", 
                padding=True
            )
            
            # Get device from model
            model_device = next(self.wav2vec2_model.parameters()).device
            
            # Move inputs to same device as model
            inputs = {k: v.to(model_device) for k, v in inputs.items()}
            
            # Extract embeddings with memory management
            with torch.no_grad():
                outputs = self.wav2vec2_model(**inputs)
                embeddings = outputs.last_hidden_state
            
            # Convert to numpy and return
            embeddings = embeddings.cpu().numpy()
            
            # Remove batch dimension
            if embeddings.shape[0] == 1:
                embeddings = embeddings[0]
            
            return embeddings
            
        except Exception as e:
            self.logger.warning(f"Failed to extract embeddings from chunk: {e}")
            return None
        
    def load_model(self):
        """Load Whisper model with device optimization."""
        if self.model is None:
            self.logger.info(f"Loading Whisper model: {self.model_name}")
            
            # Optimize for available resources
            optimization = self.device_manager.optimize_for_model(self.model_name)
            
            self.model = whisper.load_model(
                self.model_name, 
                device=self.device_manager.device
            )
            
            self.logger.info(f"Model loaded on {self.device_manager.device}")
            
    def transcribe(self, audio_path: str, shared_embeddings: Optional[AudioEmbedding] = None, **kwargs) -> Dict[str, Any]:
        """Fast transcription with unified audio processing and optional shared embeddings."""
        import os
        # Convert to WAV first if needed
        wav_created = False
        if not audio_path.lower().endswith('.wav'):
            self.logger.info("Converting audio to WAV format before transcription...")
            audio_path = self.convert_to_wav(audio_path)
            wav_created = True
        self.load_model()
        # Extract shared embeddings if not provided
        if shared_embeddings is None:
            cache_key = f"transcribe_{Path(audio_path).stem}"
            shared_embeddings = self.extract_shared_embeddings(audio_path, cache_key)
        # Optimized parameters for speed
        transcription_params = {
            "temperature": 0.0,
            "beam_size": 1,  # Reduced for speed
            "best_of": 1,    # Reduced for speed
            "word_timestamps": True,
            "condition_on_previous_text": True,  # Better for chunking
            **kwargs
        }
        self.logger.info(f"Starting fast transcription of {audio_path}")
        self.logger.info("Transcription progress: [0%] Initializing...")
        try:
            if self.use_chunking and Path(audio_path).stat().st_size > 10 * 1024 * 1024:  # > 10MB
                result = self._transcribe_with_chunking(audio_path, transcription_params, shared_embeddings)
            else:
                self.logger.info("Using full transcription for smaller file")
                result = self._transcribe_full(audio_path, transcription_params, shared_embeddings)
            return result
        except Exception as e:
            self.logger.error(f"Transcription failed: {e}")
            raise
        finally:
            # Remove converted WAV if it was created in this run
            if wav_created:
                try:
                    os.remove(audio_path)
                    self.logger.info(f"Removed temporary converted WAV file: {audio_path}")
                except Exception as remove_err:
                    self.logger.warning(f"Failed to remove temporary WAV file: {remove_err}")
            # Clear memory aggressively
            self.device_manager.clear_memory()
    
    def _transcribe_with_chunking(self, audio_path: str, params: dict, shared_embeddings: Optional[AudioEmbedding] = None) -> Dict[str, Any]:
        """Robust chunked transcription with proper error handling."""
        import librosa
        import numpy as np
        
        self.logger.info("Transcription progress: [10%] Loading audio for chunking...")
        
        # Load audio efficiently
        audio, sr = librosa.load(audio_path, sr=16000)
        total_duration = len(audio) / sr
        
        # Use sequential processing for better stability
        self.logger.info("Transcription progress: [20%] Using sequential chunked processing...")
        
        # Create overlapping chunks for better continuity
        chunk_size = int(self.chunk_duration * sr)
        overlap_size = int(2.0 * sr)  # 2-second overlap
        
        all_segments = []
        segment_id = 0
        
        # Process first chunk from the very beginning
        chunk_start_time = 0.0
        
        for i in range(0, len(audio), chunk_size - overlap_size):
            chunk_end_idx = min(i + chunk_size, len(audio))
            chunk_audio = audio[i:chunk_end_idx]
            
            # Skip very short chunks (less than 2 seconds)
            if len(chunk_audio) < sr * 2.0:
                self.logger.info(f"Skipping short chunk at {i/sr:.1f}s")
                continue
            
            chunk_start = i / sr
            chunk_end = chunk_end_idx / sr
            chunk_num = (i // (chunk_size - overlap_size)) + 1
            total_chunks = (len(audio) // (chunk_size - overlap_size)) + 1
            
            progress = int(20 + (chunk_num / total_chunks) * 60)
            self.logger.info(f"Transcription progress: [{progress}%] Processing chunk {chunk_num}/{total_chunks} ({chunk_start:.1f}s-{chunk_end:.1f}s)")
            
            try:
                # Process chunk directly with Whisper
                import tempfile
                import soundfile as sf
                
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    sf.write(tmp_file.name, chunk_audio, sr)
                    
                    # Simple transcription parameters
                    chunk_params = {
                        "temperature": 0.0,
                        "word_timestamps": True,
                        "condition_on_previous_text": False,  # Prevent cross-chunk contamination
                        "no_speech_threshold": 0.6,
                        "logprob_threshold": -1.0
                    }
                    
                    result = self.model.transcribe(tmp_file.name, **chunk_params)
                    
                    # Process segments with proper time adjustment
                    for segment in result.get("segments", []):
                        # Adjust timestamps to global time
                        adjusted_start = segment["start"] + chunk_start
                        adjusted_end = segment["end"] + chunk_start
                        
                        # Skip if overlapping with previous segments (from overlap region)
                        if all_segments and adjusted_start < all_segments[-1]["end"] - 1.0:
                            continue
                        
                        adjusted_segment = {
                            "id": segment_id,
                            "start": adjusted_start,
                            "end": adjusted_end,
                            "text": segment["text"].strip(),
                            "confidence": segment.get("avg_logprob", 0.0),
                            "words": []
                        }
                        
                        # Adjust word timestamps
                        if "words" in segment:
                            for word in segment["words"]:
                                if word.get("word", "").strip():  # Only add non-empty words
                                    adjusted_segment["words"].append({
                                        "word": word.get("word", "").strip(),
                                        "start": word.get("start", 0.0) + chunk_start,
                                        "end": word.get("end", 0.0) + chunk_start,
                                        "confidence": word.get("probability", 0.0)
                                    })
                        
                        all_segments.append(adjusted_segment)
                        segment_id += 1
                    
                    # Cleanup temporary file
                    import os
                    try:
                        os.unlink(tmp_file.name)
                    except:
                        pass
                        
            except Exception as e:
                self.logger.warning(f"Chunk {chunk_num} processing failed: {e}")
                continue
        
        self.logger.info("Transcription progress: [85%] Merging segments...")
        
        # Sort segments by start time and remove duplicates
        all_segments.sort(key=lambda x: x['start'])
        
        # Combine text
        full_text = " ".join([seg['text'] for seg in all_segments if seg['text'].strip()])
        
        self.logger.info("Transcription progress: [100%] Chunked transcription complete!")
        
        return {
            "language": "auto-detected",
            "duration": total_duration,
            "text": full_text,
            "segments": all_segments,
            "model_used": self.model_name,
            "device_used": self.device_manager.device,
            "processing_method": "chunked",
            "shared_embeddings": shared_embeddings.to_dict() if shared_embeddings else None  # Serialize for downstream use
        }
    
    def _process_chunk(self, chunk_data, params):
        """Process a single audio chunk."""
        chunk_audio, chunk_start, chunk_end, chunk_id = chunk_data
        
        # Create temporary file for chunk
        import tempfile
        import soundfile as sf
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            sf.write(tmp_file.name, chunk_audio, 16000)
            
            # Transcribe chunk with optimized parameters for speed
            chunk_params = {
                "temperature": 0.0,
                "word_timestamps": True,
                "condition_on_previous_text": True
            }
            
            try:
                result = self.model.transcribe(tmp_file.name, **chunk_params)
                
                # Adjust timestamps
                segments = []
                for segment in result.get("segments", []):
                    adjusted_segment = {
                        "id": len(segments),
                        "start": segment["start"] + chunk_start,
                        "end": segment["end"] + chunk_start,
                        "text": segment["text"].strip(),
                        "confidence": segment.get("avg_logprob", 0.0),
                        "words": []
                    }
                    
                    # Adjust word timestamps if available
                    if "words" in segment:
                        for word in segment["words"]:
                            adjusted_segment["words"].append({
                                "word": word.get("word", "").strip(),
                                "start": word.get("start", 0.0) + chunk_start,
                                "end": word.get("end", 0.0) + chunk_start,
                                "confidence": word.get("probability", 0.0)
                            })
                    
                    segments.append(adjusted_segment)
                
                return segments
                
            finally:
                # Cleanup
                import os
                try:
                    os.unlink(tmp_file.name)
                except:
                    pass
    
    def _transcribe_full(self, audio_path: str, params: dict, shared_embeddings: Optional[AudioEmbedding] = None) -> Dict[str, Any]:
        """Standard transcription for smaller files."""
        self.logger.info("Transcription progress: [30%] Processing full audio...")
        
        result = self.model.transcribe(audio_path, **params)
        
        self.logger.info("Transcription progress: [90%] Formatting results...")
        formatted_result = self._format_transcription(result, shared_embeddings)
        
        self.logger.info("Transcription progress: [100%] Full transcription complete!")
        
        return formatted_result
    
    def _format_transcription(self, result: Dict, shared_embeddings: Optional[AudioEmbedding] = None) -> Dict[str, Any]:
        """Format Whisper output for downstream processing."""
        
        # Extract segments with word-level timestamps
        segments = []
        for i, segment in enumerate(result.get("segments", [])):
            formatted_segment = {
                "id": i,
                "start": segment["start"],
                "end": segment["end"], 
                "text": segment["text"].strip(),
                "confidence": segment.get("avg_logprob", 0.0),
                "words": []
            }
            
            # Add word-level timestamps if available
            if "words" in segment:
                for word in segment["words"]:
                    formatted_segment["words"].append({
                        "word": word.get("word", "").strip(),
                        "start": word.get("start", 0.0),
                        "end": word.get("end", 0.0),
                        "confidence": word.get("probability", 0.0)
                    })
            
            segments.append(formatted_segment)
        
        return {
            "language": result.get("language", "unknown"),
            "duration": segments[-1]["end"] if segments else 0.0,
            "text": result.get("text", ""),
            "segments": segments,
            "model_used": self.model_name,
            "device_used": self.device_manager.device,
            "shared_embeddings": shared_embeddings.to_dict() if shared_embeddings else None  # Serialize for downstream use
        }
    
    def save_transcript(self, transcription: Dict[str, Any], output_path: Path):
        """Save transcription to JSON file."""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(transcription, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Transcript saved to {output_path}")
    
    def get_embeddings_for_segments(self, audio_embedding: AudioEmbedding, 
                                  segments: List[Dict[str, Any]]) -> List[np.ndarray]:
        """
        Extract embeddings for specific segments from the full audio embeddings.
        
        Args:
            audio_embedding: Full audio embeddings
            segments: List of segments with 'start' and 'end' times
            
        Returns:
            List of embedding arrays for each segment
        """
        segment_embeddings = []
        
        for segment in segments:
            start_time = segment.get('start', 0.0)
            end_time = segment.get('end', start_time + 1.0)
            
            # Find corresponding embedding indices
            start_idx = np.searchsorted(audio_embedding.timestamps, start_time)
            end_idx = np.searchsorted(audio_embedding.timestamps, end_time)
            
            # Ensure valid indices
            start_idx = max(0, start_idx)
            end_idx = min(len(audio_embedding.embeddings), end_idx)
            
            if start_idx < end_idx:
                segment_emb = audio_embedding.embeddings[start_idx:end_idx]
                # Average embeddings over time for this segment
                avg_embedding = np.mean(segment_emb, axis=0)
                segment_embeddings.append(avg_embedding)
            else:
                # Fallback for very short segments
                closest_idx = min(start_idx, len(audio_embedding.embeddings) - 1)
                segment_embeddings.append(audio_embedding.embeddings[closest_idx])
        
        return segment_embeddings
    
    def clear_cache(self):
        """Clear the embedding cache to free memory."""
        self._embedding_cache.clear()
        self.device_manager.clear_memory()
        self.logger.info("Transcriber embedding cache cleared")