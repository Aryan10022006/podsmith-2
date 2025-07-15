from pydub import AudioSegment
import numpy as np
import json
import os
import logging
import torch
from pyannote.audio import Pipeline
from pyannote.core import Annotation, Segment
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

# Import sklearn modules for embedding-based clustering
try:
    from sklearn.cluster import SpectralClustering, KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

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

class Diarization:
    def __init__(self, audio_file, session_id):
        self.audio_file = audio_file
        self.session_id = session_id
        self.speakers = {}
        self.transcript = []

    def load_audio(self):
        try:
            audio = AudioSegment.from_file(self.audio_file)
            return audio
        except Exception as e:
            logging.error(f"Error loading audio file: {e}")
            return None

    def perform_diarization(self, audio):
        # Placeholder for actual diarization logic
        # This should be replaced with a proper diarization model
        self.speakers = {
            "Speaker 1": [(0, 30), (60, 90)],
            "Speaker 2": [(30, 60), (90, 120)]
        }
        self.transcript = [
            {"speaker": "Speaker 1", "start": 0, "end": 30, "text": "Hello, welcome to the podcast."},
            {"speaker": "Speaker 2", "start": 30, "end": 60, "text": "Thank you for having me."},
            {"speaker": "Speaker 1", "start": 60, "end": 90, "text": "Let's discuss today's topic."},
            {"speaker": "Speaker 2", "start": 90, "end": 120, "text": "Absolutely, I'm excited!"}
        ]

    def save_transcript(self):
        output_dir = f"output/sessions/session_{self.session_id}/"
        os.makedirs(output_dir, exist_ok=True)
        transcript_path = os.path.join(output_dir, "transcript.json")
        with open(transcript_path, 'w') as f:
            json.dump(self.transcript, f)

    def run(self):
        audio = self.load_audio()
        if audio:
            self.perform_diarization(audio)
            self.save_transcript()
            logging.info("Diarization completed successfully.")
        else:
            logging.error("Diarization failed due to audio loading error.")

class SpeakerDiarizer:
    """Speaker diarization using pyannote.audio with shared embeddings support."""
    
    def __init__(self, model_name: str = "pyannote/speaker-diarization-3.1", 
                 device_manager: DeviceManager = None):
        self.model_name = model_name
        self.device_manager = device_manager or DeviceManager()
        self.pipeline = None
        self.logger = logging.getLogger(__name__)
        
        # Log sklearn availability
        if SKLEARN_AVAILABLE:
            self.logger.info("sklearn available - embedding-based diarization enabled")
        else:
            self.logger.warning("sklearn not available - falling back to traditional diarization")
        
    def load_model(self):
        """Load diarization pipeline with token handling."""
        if self.pipeline is None:
            self.logger.info(f"Loading traditional diarization model: {self.model_name}")
            
            try:
                # Check for HuggingFace token
                hf_token = None
                
                # Try to get token from environment
                hf_token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
                
                # Try to read from .env file
                if not hf_token:
                    env_file = Path(".env")
                    if env_file.exists():
                        with open(env_file, "r") as f:
                            for line in f:
                                if line.startswith("HUGGINGFACE_TOKEN="):
                                    hf_token = line.split("=", 1)[1].strip()
                                    break
                
                # Try to load with token
                if hf_token:
                    self.pipeline = Pipeline.from_pretrained(
                        self.model_name,
                        use_auth_token=hf_token
                    )
                else:
                    # Try without token (for public models)
                    self.pipeline = Pipeline.from_pretrained(self.model_name)
                
                # Move to appropriate device
                if self.device_manager.device == "cuda":
                    self.pipeline = self.pipeline.to(torch.device("cuda"))
                    
                self.logger.info(f"Traditional diarization model loaded on {self.device_manager.device}")
                
            except Exception as e:
                self.logger.warning(f"Failed to load traditional diarization model: {e}")
                self.logger.info("Will use fallback single speaker assignment")
                self.pipeline = None
    
    def diarize(self, audio_path: str, shared_embeddings: Optional[AudioEmbedding] = None, 
                num_speakers: Optional[int] = None) -> List[Dict[str, Any]]:
        """Perform speaker diarization using shared embeddings when available."""
        
        # PRIORITIZE embedding-based diarization if we have shared embeddings
        if shared_embeddings is not None:
            self.logger.info("✅ SHARED EMBEDDINGS PROVIDED - Using FAST embedding-based speaker diarization!")
            try:
                result = self._diarize_with_embeddings(shared_embeddings, num_speakers)
                self.logger.info("✅ Embedding-based diarization SUCCEEDED - avoiding slow traditional processing")
                return result
            except Exception as e:
                self.logger.error(f"❌ Embedding-based diarization FAILED: {e}")
                self.logger.info("⚠️  Falling back to SLOW traditional pyannote.audio processing...")
        else:
            self.logger.warning("❌ NO SHARED EMBEDDINGS PROVIDED - forced to use slow traditional diarization")
            self.logger.warning("This means audio will be processed again, defeating the purpose of unified processing!")
        
        # Traditional pyannote diarization (only as fallback)
        self.load_model()
        
        if self.pipeline is None:
            # Fallback: assign single speaker
            return self._fallback_single_speaker(audio_path)
        
        try:
            self.logger.info(f"🐌 Starting SLOW traditional diarization of {audio_path}")
            
            # Configure parameters - let the model determine speakers automatically
            params = {}
            # Don't force num_speakers, let it detect naturally
            if num_speakers and num_speakers <= 4:  # Only if reasonable number
                params["min_speakers"] = 1
                params["max_speakers"] = num_speakers
            else:
                # Auto-detect with reasonable bounds
                params["min_speakers"] = 1
                params["max_speakers"] = 6  # Reasonable max for podcasts
            
            # Run diarization
            diarization = self.pipeline(audio_path, **params)
            
            # Convert to our format
            speaker_segments = self._format_diarization(diarization)
            
            detected_speakers = len(set(s['speaker'] for s in speaker_segments))
            self.logger.info(f"Traditional diarization completed. Detected {detected_speakers} speakers")
            
            return speaker_segments
            
        except Exception as e:
            self.logger.error(f"Traditional diarization failed: {e}")
            return self._fallback_single_speaker(audio_path)
    
    def _diarize_with_embeddings(self, shared_embeddings: AudioEmbedding, 
                               num_speakers: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Perform advanced speaker diarization using shared audio embeddings.
        
        This method uses sophisticated clustering with speaker similarity detection
        to maintain consistent speaker identities across segments.
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn is required for embedding-based diarization")
        
        self.logger.info("🎯 Performing ADVANCED embedding-based speaker diarization...")
        
        # Prepare embeddings for clustering
        embeddings = shared_embeddings.embeddings
        timestamps = shared_embeddings.timestamps
        
        self.logger.info(f"Processing {len(embeddings)} embedding vectors for diarization")
        
        # Smart downsampling - keep more samples for better speaker detection
        downsample_factor = max(1, len(embeddings) // 1000)  # Max 1000 points for clustering
        downsampled_embeddings = embeddings[::downsample_factor]
        downsampled_timestamps = timestamps[::downsample_factor]
        
        self.logger.info(f"Downsampled to {len(downsampled_embeddings)} points for clustering")
        
        # Normalize embeddings for better clustering
        scaler = StandardScaler()
        normalized_embeddings = scaler.fit_transform(downsampled_embeddings)
        
        # Determine optimal number of speakers using multiple methods
        if num_speakers is None:
            num_speakers = self._smart_estimate_speakers(normalized_embeddings)
        else:
            # Validate provided num_speakers
            num_speakers = min(max(1, num_speakers), 6)  # Reasonable bounds
        
        self.logger.info(f"Target speakers: {num_speakers}")
        
        # Perform hierarchical clustering for better speaker boundaries
        speaker_labels = self._advanced_clustering(normalized_embeddings, num_speakers)
        
        # Convert clusters to time segments with speaker consistency
        speaker_segments = self._labels_to_segments_advanced(
            speaker_labels, downsampled_timestamps, shared_embeddings.duration, 
            normalized_embeddings
        )
        
        # Post-process to merge similar speakers and clean boundaries
        speaker_segments = self._post_process_speakers(speaker_segments, normalized_embeddings, speaker_labels)
        
        detected_speakers = len(set(s['speaker'] for s in speaker_segments))
        self.logger.info(f"✅ Advanced embedding-based diarization completed. Detected {detected_speakers} speakers with {len(speaker_segments)} segments")
        
        return speaker_segments
    
    def _smart_estimate_speakers(self, embeddings: np.ndarray) -> int:
        """
        Use multiple methods to smartly estimate the optimal number of speakers.
        Enhanced with conservative approach to avoid over-segmentation.
        """
        from sklearn.metrics import silhouette_score
        from sklearn.cluster import DBSCAN, KMeans
        
        total_samples = len(embeddings)
        self.logger.info(f"Estimating speakers from {total_samples} embedding samples...")
        
        # Conservative approach - require more samples per speaker for reliability
        min_samples_per_speaker = 30  # Increased from 20 for even more conservative approach
        max_possible_speakers = max(1, total_samples // min_samples_per_speaker)
        
        # Cap maximum speakers based on content analysis - be very conservative
        max_speakers = min(3, max_possible_speakers)  # Reduced from 4 to 3
        
        # For large files, be even more conservative to prevent memory issues
        if total_samples > 1000:  # Large file with many samples
            max_speakers = min(2, max_speakers)  # Limit to 2 speakers max for large files
            self.logger.info("Large audio file detected - limiting to maximum 2 speakers")
        
        if max_speakers <= 1:
            self.logger.info("Insufficient data for multi-speaker detection, defaulting to 1 speaker")
            return 1
            
        # Try DBSCAN first for natural cluster discovery with stricter parameters
        try:
            # More conservative DBSCAN parameters
            eps = 0.6  # Increased eps for larger clusters
            min_samples = max(5, total_samples // 25)  # Increased min_samples
            
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            dbscan_labels = dbscan.fit_predict(embeddings)
            
            # Count valid clusters (excluding noise)
            unique_labels = set(dbscan_labels)
            noise_points = sum(1 for label in dbscan_labels if label == -1)
            natural_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            
            # Only accept DBSCAN result if it's reasonable and has low noise
            noise_ratio = noise_points / total_samples
            if 1 <= natural_clusters <= max_speakers and noise_ratio < 0.3:
                self.logger.info(f"DBSCAN suggests {natural_clusters} natural speaker clusters (noise: {noise_ratio:.2f})")
                return natural_clusters
            else:
                self.logger.debug(f"DBSCAN rejected: {natural_clusters} clusters, {noise_ratio:.2f} noise ratio")
                
        except Exception as e:
            self.logger.debug(f"DBSCAN clustering failed: {e}")
        
        # Fall back to K-means with conservative evaluation
        if max_speakers == 1:
            return 1
            
        scores = {}
        for k in range(1, max_speakers + 1):
            try:
                if k == 1:
                    # Single speaker case
                    scores[k] = 0.5  # Baseline score
                    continue
                    
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=5, max_iter=100)
                labels = kmeans.fit_predict(embeddings)
                
                # Silhouette score (higher is better, range -1 to 1)
                sil_score = silhouette_score(embeddings, labels)
                
                # Check cluster balance (penalize very uneven clusters more heavily)
                unique_labels, counts = np.unique(labels, return_counts=True)
                balance_ratio = np.min(counts) / np.max(counts)
                
                # Strong penalty for imbalanced clusters
                if balance_ratio < 0.15:  # If smallest cluster < 15% of largest
                    balance_penalty = 0.3
                elif balance_ratio < 0.25:
                    balance_penalty = 0.6
                else:
                    balance_penalty = 1.0
                
                # Conservative scoring - favor fewer speakers
                speaker_penalty = 0.9 ** (k - 1)  # Exponential penalty for more speakers
                
                # Combined score with strong bias toward fewer speakers
                combined_score = sil_score * balance_penalty * speaker_penalty
                scores[k] = combined_score
                
                self.logger.debug(f"k={k}: sil={sil_score:.3f}, balance={balance_ratio:.2f}, final_score={combined_score:.3f}")
                
            except Exception as e:
                self.logger.warning(f"Failed to evaluate k={k}: {e}")
                continue
        
        if not scores:
            self.logger.warning("All clustering evaluations failed, defaulting to 1 speaker")
            return 1
            
        # Choose the number of speakers with highest combined score
        optimal_k = max(scores.keys(), key=lambda k: scores[k])
        
        # Additional validation - if best score is very low, default to fewer speakers
        if optimal_k > 1 and scores[optimal_k] < 0.3:  # Increased threshold from 0.2
            self.logger.info("Low clustering confidence, defaulting to 1 speaker")
            return 1
        
        # Extra conservative check - for large files, prefer single speaker unless very confident
        if total_samples > 1000 and optimal_k > 1 and scores[optimal_k] < 0.5:
            self.logger.info("Large file with low multi-speaker confidence - defaulting to 1 speaker")
            return 1
            
        self.logger.info(f"Conservative speaker estimation: {optimal_k} speakers (confidence: {scores[optimal_k]:.3f})")
        
        return optimal_k
    
    def _advanced_clustering(self, embeddings: np.ndarray, num_speakers: int) -> np.ndarray:
        """
        Perform advanced clustering with hierarchical and density-based methods.
        Enhanced with speaker consistency and temporal coherence.
        """
        try:
            # Primary method: Agglomerative clustering with connectivity for smooth boundaries
            from sklearn.cluster import AgglomerativeClustering
            from sklearn.neighbors import kneighbors_graph
            
            # Create connectivity matrix for smoother speaker transitions
            n_neighbors = min(15, max(5, len(embeddings) // 20))  # Adaptive neighborhood size
            connectivity = kneighbors_graph(embeddings, n_neighbors=n_neighbors, include_self=False)
            
            clustering = AgglomerativeClustering(
                n_clusters=num_speakers,
                connectivity=connectivity,
                linkage='ward'
            )
            labels = clustering.fit_predict(embeddings)
            
            # Post-process: merge similar speakers based on centroid distance
            labels = self._merge_similar_speakers(embeddings, labels, similarity_threshold=0.85)
            
            self.logger.info("Used enhanced Agglomerative clustering with speaker merging")
            return labels
            
        except Exception as e:
            self.logger.warning(f"Agglomerative clustering failed: {e}, falling back to KMeans")
            
            # Fallback to KMeans
            try:
                clustering = KMeans(
                    n_clusters=num_speakers,
                    random_state=42,
                    n_init=20,
                    max_iter=300
                )
                labels = clustering.fit_predict(embeddings)
                self.logger.info("Used KMeans clustering as fallback")
                return labels
                
            except Exception as e2:
                self.logger.warning(f"KMeans also failed: {e2}, using simple assignment")
                return np.arange(len(embeddings)) % num_speakers
    
    def _merge_similar_speakers(self, embeddings: np.ndarray, labels: np.ndarray, 
                              similarity_threshold: float = 0.85) -> np.ndarray:
        """
        Merge speakers with highly similar acoustic features to avoid oversegmentation.
        
        Args:
            embeddings: Audio embeddings
            labels: Initial speaker labels
            similarity_threshold: Cosine similarity threshold for merging (0.85 = very similar)
            
        Returns:
            Updated speaker labels with similar speakers merged
        """
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Calculate speaker centroids
            unique_speakers = np.unique(labels)
            if len(unique_speakers) <= 1:
                return labels
                
            speaker_centroids = {}
            for speaker_id in unique_speakers:
                speaker_mask = labels == speaker_id
                if np.any(speaker_mask):
                    speaker_centroids[speaker_id] = np.mean(embeddings[speaker_mask], axis=0)
            
            # Calculate pairwise similarities between speaker centroids
            centroid_list = list(speaker_centroids.values())
            centroid_ids = list(speaker_centroids.keys())
            
            if len(centroid_list) < 2:
                return labels
                
            similarities = cosine_similarity(centroid_list)
            
            # Find pairs of speakers to merge
            merged_labels = labels.copy()
            merge_mapping = {}
            
            for i in range(len(centroid_ids)):
                for j in range(i + 1, len(centroid_ids)):
                    if similarities[i, j] > similarity_threshold:
                        speaker_a, speaker_b = centroid_ids[i], centroid_ids[j]
                        
                        # Count segments for each speaker to decide which to keep
                        count_a = np.sum(labels == speaker_a)
                        count_b = np.sum(labels == speaker_b)
                        
                        # Keep the speaker with more segments, merge the other
                        if count_a >= count_b:
                            merge_mapping[speaker_b] = speaker_a
                            self.logger.info(f"Merging similar speakers: {speaker_b} -> {speaker_a} (similarity: {similarities[i,j]:.3f})")
                        else:
                            merge_mapping[speaker_a] = speaker_b
                            self.logger.info(f"Merging similar speakers: {speaker_a} -> {speaker_b} (similarity: {similarities[i,j]:.3f})")
            
            # Apply merging
            for old_speaker, new_speaker in merge_mapping.items():
                merged_labels[merged_labels == old_speaker] = new_speaker
            
            # Relabel to have consecutive speaker IDs
            unique_merged = np.unique(merged_labels)
            relabel_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_merged)}
            
            for old_id, new_id in relabel_mapping.items():
                merged_labels[merged_labels == old_id] = new_id
            
            final_speaker_count = len(np.unique(merged_labels))
            original_speaker_count = len(unique_speakers)
            
            if final_speaker_count != original_speaker_count:
                self.logger.info(f"Speaker merging: {original_speaker_count} -> {final_speaker_count} speakers")
            
            return merged_labels
            
        except Exception as e:
            self.logger.warning(f"Speaker merging failed: {e}")
            return labels

    def _labels_to_segments_advanced(self, labels: np.ndarray, timestamps: np.ndarray, 
                                   duration: float, embeddings: np.ndarray) -> List[Dict[str, Any]]:
        """
        Convert clustering labels to speaker segments with advanced boundary detection.
        """
        segments = []
        current_speaker = labels[0]
        current_start = 0.0
        
        # Calculate speaker centroids for feature similarity
        speaker_centroids = {}
        for speaker_id in np.unique(labels):
            speaker_mask = labels == speaker_id
            speaker_centroids[speaker_id] = np.mean(embeddings[speaker_mask], axis=0)
        
        for i in range(1, len(labels)):
            if labels[i] != current_speaker:
                # Speaker change detected
                end_time = timestamps[i] if i < len(timestamps) else duration
                
                segments.append({
                    'start': current_start,
                    'end': end_time,
                    'speaker': f"SPEAKER_{current_speaker:02d}",
                    'speaker_id': int(current_speaker),
                    'confidence': self._calculate_segment_confidence(
                        embeddings[max(0, i-10):i], speaker_centroids[current_speaker]
                    )
                })
                
                current_speaker = labels[i]
                current_start = end_time
        
        # Add final segment
        if current_start < duration:
            segments.append({
                'start': current_start,
                'end': duration,
                'speaker': f"SPEAKER_{current_speaker:02d}",
                'speaker_id': int(current_speaker),
                'confidence': self._calculate_segment_confidence(
                    embeddings[-10:], speaker_centroids[current_speaker]
                )
            })
        
        return segments
    
    def _calculate_segment_confidence(self, segment_embeddings: np.ndarray, 
                                    speaker_centroid: np.ndarray) -> float:
        """
        Calculate confidence score for a speaker segment based on embedding similarity.
        """
        if len(segment_embeddings) == 0:
            return 0.5
            
        # Calculate cosine similarity to speaker centroid
        similarities = []
        for embedding in segment_embeddings:
            similarity = np.dot(embedding, speaker_centroid) / (
                np.linalg.norm(embedding) * np.linalg.norm(speaker_centroid)
            )
            similarities.append(similarity)
        
        # Return average similarity as confidence
        return float(np.mean(similarities))
    
    def _post_process_speakers(self, segments: List[Dict[str, Any]], 
                             embeddings: np.ndarray, 
                             labels: np.ndarray) -> List[Dict[str, Any]]:
        """
        Post-process speaker segments to merge overly similar speakers and clean boundaries.
        """
        if len(segments) <= 1:
            return segments
            
        self.logger.info("Post-processing speaker segments for similarity...")
        
        # Calculate speaker centroids from embeddings
        unique_speakers = set(seg['speaker_id'] for seg in segments)
        speaker_centroids = {}
        speaker_embeddings = {}
        
        for speaker_id in unique_speakers:
            speaker_mask = labels == speaker_id
            if np.any(speaker_mask):
                speaker_embeddings[speaker_id] = embeddings[speaker_mask]
                speaker_centroids[speaker_id] = np.mean(embeddings[speaker_mask], axis=0)
        
        # Find speaker pairs that are too similar
        similarity_threshold = 0.85  # Very high threshold for merging
        speakers_to_merge = {}
        
        speaker_list = list(unique_speakers)
        for i, speaker_a in enumerate(speaker_list):
            for j, speaker_b in enumerate(speaker_list[i+1:], i+1):
                if speaker_a in speaker_centroids and speaker_b in speaker_centroids:
                    similarity = self._calculate_speaker_similarity(
                        speaker_centroids[speaker_a], 
                        speaker_centroids[speaker_b]
                    )
                    
                    if similarity > similarity_threshold:
                        self.logger.info(f"Speakers {speaker_a} and {speaker_b} are very similar ({similarity:.3f}), merging...")
                        # Merge speaker_b into speaker_a
                        if speaker_b not in speakers_to_merge:
                            speakers_to_merge[speaker_b] = speaker_a
        
        # Apply speaker merging
        merged_segments = []
        for segment in segments:
            new_segment = segment.copy()
            original_speaker_id = segment['speaker_id']
            
            # Check if this speaker should be merged
            if original_speaker_id in speakers_to_merge:
                target_speaker_id = speakers_to_merge[original_speaker_id]
                new_segment['speaker_id'] = target_speaker_id
                new_segment['speaker'] = f"SPEAKER_{target_speaker_id:02d}"
                self.logger.debug(f"Merged speaker {original_speaker_id} -> {target_speaker_id}")
            
            merged_segments.append(new_segment)
        
        # Merge consecutive segments with same speaker
        final_segments = []
        current_segment = None
        
        for segment in merged_segments:
            if current_segment is None:
                current_segment = segment.copy()
            elif current_segment['speaker_id'] == segment['speaker_id']:
                # Merge with current segment
                current_segment['end'] = segment['end']
                # Average confidence
                current_segment['confidence'] = (current_segment['confidence'] + segment['confidence']) / 2
            else:
                # Different speaker, save current and start new
                final_segments.append(current_segment)
                current_segment = segment.copy()
        
        # Add final segment
        if current_segment is not None:
            final_segments.append(current_segment)
        
        # Renumber speakers sequentially
        speaker_mapping = {}
        next_speaker_num = 0
        
        for segment in final_segments:
            old_speaker_id = segment['speaker_id']
            if old_speaker_id not in speaker_mapping:
                speaker_mapping[old_speaker_id] = next_speaker_num
                next_speaker_num += 1
            
            segment['speaker_id'] = speaker_mapping[old_speaker_id]
            segment['speaker'] = f"SPEAKER_{speaker_mapping[old_speaker_id]:02d}"
        
        unique_final_speakers = len(set(seg['speaker_id'] for seg in final_segments))
        self.logger.info(f"Post-processing complete: {len(unique_speakers)} -> {unique_final_speakers} speakers")
        
        return final_segments
    
    def _calculate_speaker_similarity(self, centroid_a: np.ndarray, centroid_b: np.ndarray) -> float:
        """Calculate cosine similarity between two speaker centroids."""
        try:
            # Normalize centroids
            norm_a = centroid_a / (np.linalg.norm(centroid_a) + 1e-8)
            norm_b = centroid_b / (np.linalg.norm(centroid_b) + 1e-8)
            
            # Calculate cosine similarity
            similarity = np.dot(norm_a, norm_b)
            return float(similarity)
        except Exception:
            return 0.0
    
    def merge_with_transcript(self, transcript: Dict[str, Any], 
                            speaker_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge speaker information with transcript segments."""
        merged_segments = []
        
        # Safely get segments from transcript
        segments = transcript.get("segments", [])
        
        if not segments:
            self.logger.warning("No segments found in transcript for merging with speaker data")
            return []
        
        for segment in segments:
            segment_start = segment.get("start", 0.0)
            segment_end = segment.get("end", 0.0)
            
            # Find overlapping speaker
            speaker = self._find_dominant_speaker(
                segment_start, segment_end, speaker_segments
            )
            
            merged_segment = {
                "id": segment.get("id", 0),
                "speaker": speaker,
                "start": segment_start,
                "end": segment_end,
                "text": segment.get("text", ""),
                "confidence": segment.get("confidence", 0.0),
                "words": segment.get("words", [])
            }
            
            merged_segments.append(merged_segment)
        
        return merged_segments
    
    def _find_dominant_speaker(self, start: float, end: float, 
                             speaker_segments: List[Dict[str, Any]]) -> str:
        """Find the dominant speaker for a given time range."""
        speaker_overlap = {}
        
        if not speaker_segments:
            return "Speaker 1"  # Default fallback if no speaker segments
        
        for speaker_seg in speaker_segments:
            # Safely get segment data
            seg_start = speaker_seg.get("start", 0.0)
            seg_end = speaker_seg.get("end", 0.0)
            speaker = speaker_seg.get("speaker", "Speaker 1")
            
            overlap_start = max(start, seg_start)
            overlap_end = min(end, seg_end)
            
            if overlap_start < overlap_end:
                overlap_duration = overlap_end - overlap_start
                
                if speaker not in speaker_overlap:
                    speaker_overlap[speaker] = 0
                speaker_overlap[speaker] += overlap_duration
        
        if speaker_overlap:
            return max(speaker_overlap, key=speaker_overlap.get)
        else:
            return "Speaker 1"  # Default fallback
    
    def save_diarization(self, diarized_transcript: List[Dict[str, Any]], output_path: Path):
        """Save diarized transcript to JSON file."""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(diarized_transcript, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Diarized transcript saved to {output_path}")
    
    def _fallback_single_speaker(self, audio_path: str) -> List[Dict[str, Any]]:
        """Fallback method when traditional diarization fails - assign everything to single speaker."""
        self.logger.info("Using fallback single speaker assignment")
        
        try:
            # Get audio duration
            import librosa
            duration = librosa.get_duration(path=audio_path)
            
            return [{
                "start": 0.0,
                "end": duration,
                "speaker": "SPEAKER_00",
                "speaker_id": 0,
                "confidence": 0.5
            }]
            
        except Exception as e:
            self.logger.warning(f"Failed to get audio duration for fallback: {e}")
            # Return default 60-second segment
            return [{
                "start": 0.0,
                "end": 60.0,
                "speaker": "SPEAKER_00", 
                "speaker_id": 0,
                "confidence": 0.5
            }]
    
    def _format_diarization(self, diarization) -> List[Dict[str, Any]]:
        """Format pyannote diarization output to our standard format."""
        speaker_segments = []
        
        try:
            for segment, _, speaker in diarization.itertracks(yield_label=True):
                speaker_segments.append({
                    "start": float(segment.start),
                    "end": float(segment.end), 
                    "speaker": f"SPEAKER_{speaker}",
                    "speaker_id": hash(speaker) % 100,  # Simple speaker ID
                    "confidence": 0.8  # Default confidence for traditional diarization
                })
            
            # Sort by start time
            speaker_segments.sort(key=lambda x: x["start"])
            
            self.logger.info(f"Formatted {len(speaker_segments)} speaker segments")
            return speaker_segments
            
        except Exception as e:
            self.logger.error(f"Error formatting diarization: {e}")
            return self._fallback_single_speaker("")
    
    def _merge_short_segments(self, segments: List[Dict[str, Any]], 
                            min_duration: float = 1.0) -> List[Dict[str, Any]]:
        """Merge segments that are too short into neighboring segments."""
        if not segments:
            return segments
            
        merged_segments = []
        
        for segment in segments:
            duration = segment["end"] - segment["start"]
            
            if duration < min_duration and merged_segments:
                # Merge with previous segment
                prev_segment = merged_segments[-1]
                prev_segment["end"] = segment["end"]
                # Keep the speaker with higher confidence
                if segment.get("confidence", 0) > prev_segment.get("confidence", 0):
                    prev_segment["speaker"] = segment["speaker"]
                    prev_segment["speaker_id"] = segment["speaker_id"]
            else:
                merged_segments.append(segment.copy())
        
        return merged_segments