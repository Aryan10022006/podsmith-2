"""
Optimized Emotion Detection Module for Podcast Analysis

This module provides fast and accurate emotion detection:
1. Audio emotion detection using shared embeddings with optimized audio models
2. Text emotion detection using lightweight but accurate language models
3. Separate processing pipelines for optimal performance
"""

import json
import time
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
import torch
from pathlib import Path

# Try importing emotion recognition dependencies
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Import shared audio embedding type
from ..transcription.whisper_transcriber import AudioEmbedding


@dataclass
class EmotionPrediction:
    """Container for emotion prediction results."""
    emotion: str
    confidence: float
    probabilities: Dict[str, float]
    segment_start: float
    segment_end: float
    text_id: Optional[int] = None  # ID of the transcript segment
    speaker: Optional[str] = None  # Speaker information
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        # Convert numpy types to regular Python types for JSON serialization
        import numpy as np
        
        def convert_numpy(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        result = {
            'emotion': self.emotion,
            'confidence': convert_numpy(self.confidence),
            'probabilities': convert_numpy(self.probabilities),
            'segment_start': convert_numpy(self.segment_start),
            'segment_end': convert_numpy(self.segment_end)
        }
        if self.text_id is not None:
            result['text_id'] = convert_numpy(self.text_id)
        if self.speaker is not None:
            result['speaker'] = self.speaker
        return result

class EmotionDetector:
    """
    Optimized Emotion Detection for Audio and Text
    
    Two specialized pipelines:
    1. Audio emotion detection from shared embeddings (fast, reuses existing data)
    2. Text emotion detection from transcript segments (accurate, optimized model)
    """
    
    # Standard emotion categories
    EMOTION_CATEGORIES = [
        'neutral', 'happy', 'sad', 'angry', 
        'fear', 'surprise', 'disgust', 'excited'
    ]
    
    def __init__(self, device_manager=None, cache_dir: Optional[str] = None):
        """
        Initialize the optimized emotion detector.
        
        Args:
            device_manager: Device manager for GPU/CPU handling
            cache_dir: Directory for caching models
        """
        self.device_manager = device_manager
        self.cache_dir = Path(cache_dir) if cache_dir else Path.cwd() / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Optimized model configurations
        self.text_emotion_models = {
            "fast": "j-hartmann/emotion-english-distilroberta-base",     # Fast & accurate
            "balanced": "cardiffnlp/twitter-roberta-base-emotion-latest", # Good balance
            "accurate": "facebook/bart-large-mnli"  # Most accurate but slower
        }
        
        # Model components
        self.text_emotion_pipeline = None
        self.audio_emotion_classifier = None
        self.audio_scaler = None
        
        # Model state
        self.text_model_loaded = False
        self.audio_model_loaded = False
        
        # Choose fast model by default
        self.selected_text_model = self.text_emotion_models["fast"]
    
    def load_text_emotion_model(self, model_type: str = "fast") -> bool:
        """
        Load optimized text emotion detection model.
        
        Args:
            model_type: "fast", "balanced", or "accurate"
        """
        if self.text_model_loaded:
            return True
            
        if not TRANSFORMERS_AVAILABLE:
            self.logger.error("Transformers not available for text emotion detection")
            return False
        
        try:
            model_name = self.text_emotion_models.get(model_type, self.text_emotion_models["fast"])
            self.logger.info(f"Loading {model_type} text emotion model: {model_name}")
            
            # Load optimized pipeline
            self.text_emotion_pipeline = pipeline(
                "text-classification",
                model=model_name,
                tokenizer=model_name,
                device=0 if self.device_manager and self.device_manager.device == "cuda" else -1,
                framework="pt",
                return_all_scores=True,
                truncation=True,
                max_length=512
            )
            
            self.text_model_loaded = True
            self.logger.info(f"✅ Text emotion model loaded successfully: {model_type}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load text emotion model: {e}")
            return False
    
    def load_audio_emotion_model(self) -> bool:
        """
        Load lightweight audio emotion classifier for embeddings.
        """
        if self.audio_model_loaded:
            return True
            
        if not SKLEARN_AVAILABLE:
            self.logger.error("Scikit-learn not available for audio emotion detection")
            return False
        
        try:
            # Use lightweight but effective RandomForest classifier
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import StandardScaler
            
            self.audio_emotion_classifier = RandomForestClassifier(
                n_estimators=100,     # Good balance of speed vs accuracy
                max_depth=10,         # Prevent overfitting
                random_state=42,      # Reproducible results
                n_jobs=-1            # Use all CPU cores
            )
            
            self.audio_scaler = StandardScaler()
            
            # Pre-train with basic emotion patterns (simplified for demo)
            self._initialize_audio_emotion_patterns()
            
            self.audio_model_loaded = True
            self.logger.info("✅ Audio emotion model loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load audio emotion model: {e}")
            return False
    
    def _initialize_audio_emotion_patterns(self):
        """
        Initialize audio emotion classifier with basic patterns.
        
        In production, this would be trained on labeled audio emotion data.
        For now, we use pattern-based classification.
        """
        # This is a simplified initialization
        # In production, you'd load pre-trained weights or train on labeled data
        self.emotion_patterns = {
            'neutral': {'energy': 0.3, 'pitch_var': 0.2, 'tempo': 0.5},
            'happy': {'energy': 0.7, 'pitch_var': 0.6, 'tempo': 0.7},
            'sad': {'energy': 0.2, 'pitch_var': 0.1, 'tempo': 0.3},
            'angry': {'energy': 0.9, 'pitch_var': 0.8, 'tempo': 0.8},
            'fear': {'energy': 0.6, 'pitch_var': 0.9, 'tempo': 0.6},
            'surprise': {'energy': 0.8, 'pitch_var': 0.7, 'tempo': 0.6},
            'disgust': {'energy': 0.4, 'pitch_var': 0.3, 'tempo': 0.4},
            'excited': {'energy': 0.9, 'pitch_var': 0.8, 'tempo': 0.9}
        }
    
    def detect_emotions_from_text(self, transcript_segments: List[Dict[str, Any]]) -> List[EmotionPrediction]:
        """
        Detect emotions from transcript text segments using optimized language model.
        
        Args:
            transcript_segments: List of transcript segments with text
            
        Returns:
            List of EmotionPrediction objects
        """
        if not self.load_text_emotion_model():
            self.logger.error("Failed to load text emotion model")
            return []
        
        self.logger.info(f"Detecting text emotions from {len(transcript_segments)} segments...")
        
        emotions = []
        
        for i, segment in enumerate(transcript_segments):
            try:
                text = segment.get('text', '').strip()
                
                if len(text) < 3:  # Skip very short text
                    continue
                
                # Get emotion prediction
                predictions = self.text_emotion_pipeline(text)
                
                if predictions and len(predictions) > 0:
                    # Get the highest confidence emotion
                    if isinstance(predictions[0], list):
                        # Multiple emotions returned
                        best_prediction = max(predictions[0], key=lambda x: x['score'])
                    else:
                        # Single emotion returned
                        best_prediction = predictions[0]
                    
                    # Map model output to our categories
                    emotion = self._map_text_emotion(best_prediction['label'])
                    confidence = best_prediction['score']
                    
                    # Create probability distribution
                    probabilities = {}
                    if isinstance(predictions[0], list):
                        for pred in predictions[0]:
                            mapped_emotion = self._map_text_emotion(pred['label'])
                            probabilities[mapped_emotion] = pred['score']
                    else:
                        probabilities[emotion] = confidence
                    
                    # Create emotion prediction
                    emotion_pred = EmotionPrediction(
                        emotion=emotion,
                        confidence=confidence,
                        probabilities=probabilities,
                        segment_start=segment.get('start', 0.0),
                        segment_end=segment.get('end', 0.0),
                        text_id=segment.get('id', i),
                        speaker=segment.get('speaker', 'Unknown')
                    )
                    
                    emotions.append(emotion_pred)
                
                # Progress logging
                if (i + 1) % 100 == 0:
                    self.logger.info(f"Processed {i + 1}/{len(transcript_segments)} text segments")
                    
            except Exception as e:
                self.logger.debug(f"Failed to process text segment {i}: {e}")
                continue
        
        self.logger.info(f"✅ Text emotion detection completed: {len(emotions)} emotions detected")
        return emotions
    
    def detect_emotions_from_embeddings(self, shared_embeddings: AudioEmbedding, 
                                      speaker_segments: List[Dict[str, Any]]) -> List[EmotionPrediction]:
        """
        Detect emotions from shared audio embeddings (fast, reuses existing data).
        
        Args:
            shared_embeddings: AudioEmbedding object with pre-extracted features
            speaker_segments: List of speaker segments for temporal mapping
            
        Returns:
            List of EmotionPrediction objects
        """
        if not self.load_audio_emotion_model():
            self.logger.error("Failed to load audio emotion model")
            return []
        
        self.logger.info(f"Detecting audio emotions from embeddings for {len(speaker_segments)} segments...")
        
        emotions = []
        
        try:
            # Extract audio features from embeddings
            embeddings = shared_embeddings.embeddings
            timestamps = shared_embeddings.timestamps
            
            for i, segment in enumerate(speaker_segments):
                try:
                    start_time = segment.get('start', 0.0)
                    end_time = segment.get('end', 0.0)
                    speaker = segment.get('speaker', 'Unknown')
                    
                    # Find embeddings within time range
                    mask = (timestamps >= start_time) & (timestamps <= end_time)
                    segment_embeddings = embeddings[mask]
                    
                    if len(segment_embeddings) == 0:
                        continue
                    
                    # Extract emotion features from embeddings
                    emotion_features = self._extract_emotion_features_from_embeddings(segment_embeddings)
                    
                    if emotion_features is not None:
                        # Predict emotion using pattern matching (simplified)
                        emotion, confidence, probabilities = self._predict_audio_emotion(emotion_features)
                        
                        emotion_pred = EmotionPrediction(
                            emotion=emotion,
                            confidence=confidence,
                            probabilities=probabilities,
                            segment_start=start_time,
                            segment_end=end_time,
                            text_id=None,  # No text ID for audio-only detection
                            speaker=speaker
                        )
                        
                        emotions.append(emotion_pred)
                
                except Exception as e:
                    self.logger.debug(f"Failed to process audio segment {i}: {e}")
                    continue
            
            self.logger.info(f"✅ Audio emotion detection completed: {len(emotions)} emotions detected")
            return emotions
            
        except Exception as e:
            self.logger.error(f"Audio emotion detection failed: {e}")
            return []
    
    def _extract_emotion_features_from_embeddings(self, embeddings: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract emotion-relevant features from audio embeddings.
        
        Args:
            embeddings: Audio embeddings for the segment
            
        Returns:
            Feature vector for emotion classification
        """
        try:
            # Statistical features from embeddings
            features = []
            
            # Basic statistics
            features.extend([
                np.mean(embeddings, axis=0).mean(),      # Overall energy
                np.std(embeddings, axis=0).mean(),       # Variability
                np.max(embeddings, axis=0).mean(),       # Peak intensity
                np.min(embeddings, axis=0).mean(),       # Minimum intensity
            ])
            
            # Temporal dynamics
            if len(embeddings) > 1:
                diff = np.diff(embeddings, axis=0)
                features.extend([
                    np.mean(np.abs(diff)),                # Rate of change
                    np.std(diff),                         # Change variability
                ])
            else:
                features.extend([0.0, 0.0])
            
            # Spectral characteristics (approximated from embeddings)
            features.extend([
                np.percentile(embeddings.flatten(), 25),  # Low energy percentile
                np.percentile(embeddings.flatten(), 75),  # High energy percentile
                np.median(embeddings.flatten()),          # Median energy
            ])
            
            return np.array(features)
            
        except Exception as e:
            self.logger.debug(f"Feature extraction failed: {e}")
            return None
    
    def _predict_audio_emotion(self, features: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """
        Predict emotion from audio features using pattern matching.
        
        Args:
            features: Extracted audio features
            
        Returns:
            Tuple of (emotion, confidence, probabilities)
        """
        try:
            # Simplified pattern-based classification
            # In production, this would use a trained classifier
            
            energy = features[0] if len(features) > 0 else 0.5
            variability = features[1] if len(features) > 1 else 0.5
            
            # Normalize to 0-1 range
            energy = max(0, min(1, energy))
            variability = max(0, min(1, variability))
            
            # Pattern matching
            scores = {}
            for emotion, pattern in self.emotion_patterns.items():
                # Simple distance-based scoring
                energy_diff = abs(energy - pattern['energy'])
                var_diff = abs(variability - pattern['pitch_var'])
                
                # Calculate similarity score
                score = 1.0 - (energy_diff + var_diff) / 2.0
                scores[emotion] = max(0.1, score)  # Minimum confidence
            
            # Get best match
            best_emotion = max(scores.keys(), key=lambda x: scores[x])
            best_confidence = scores[best_emotion]
            
            # Normalize probabilities
            total_score = sum(scores.values())
            probabilities = {emotion: score / total_score for emotion, score in scores.items()}
            
            return best_emotion, best_confidence, probabilities
            
        except Exception as e:
            self.logger.debug(f"Audio emotion prediction failed: {e}")
            return 'neutral', 0.5, {'neutral': 1.0}
    
    def _map_text_emotion(self, model_emotion: str) -> str:
        """
        Map model-specific emotion labels to our standard categories.
        
        Args:
            model_emotion: Emotion label from the model
            
        Returns:
            Mapped emotion category
        """
        model_emotion = model_emotion.lower()
        
        # Common mappings
        emotion_mapping = {
            'joy': 'happy',
            'happiness': 'happy',
            'positive': 'happy',
            'sadness': 'sad',
            'negative': 'sad',
            'anger': 'angry',
            'rage': 'angry',
            'annoyance': 'angry',
            'fear': 'fear',
            'anxiety': 'fear',
            'worry': 'fear',
            'surprise': 'surprise',
            'amazement': 'surprise',
            'disgust': 'disgust',
            'hate': 'disgust',
            'love': 'happy',
            'excitement': 'excited',
            'enthusiasm': 'excited',
            'calm': 'neutral',
            'neutral': 'neutral',
            'normal': 'neutral'
        }
        
        return emotion_mapping.get(model_emotion, model_emotion if model_emotion in self.EMOTION_CATEGORIES else 'neutral')
    
    def detect(self, transcript_segments: List[Dict[str, Any]], 
               shared_embeddings: Optional[AudioEmbedding] = None,
               speaker_segments: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Unified emotion detection interface supporting both text and audio analysis.
        
        Args:
            transcript_segments: List of transcript segments with text
            shared_embeddings: Optional audio embeddings for audio emotion detection
            speaker_segments: Optional speaker segments for temporal mapping
            
        Returns:
            Dictionary containing emotion analysis results
        """
        self.logger.info("🎭 Starting unified emotion detection...")
        
        results = {
            'emotions': [],
            'text_emotions': [],
            'audio_emotions': [],
            'summary': {},
            'metadata': {
                'total_segments': len(transcript_segments),
                'has_audio_analysis': shared_embeddings is not None,
                'has_speaker_segments': speaker_segments is not None,
                'processing_time': 0.0
            }
        }
        
        start_time = time.time()
        
        try:
            # Text emotion detection (primary)
            if transcript_segments:
                text_emotions = self.detect_emotions_from_text(transcript_segments)
                results['text_emotions'] = text_emotions
                results['emotions'].extend(text_emotions)
                
                self.logger.info(f"✅ Text emotions: {len(text_emotions)} detected")
            
            # Audio emotion detection (if embeddings available)
            if shared_embeddings and speaker_segments:
                audio_emotions = self.detect_emotions_from_embeddings(shared_embeddings, speaker_segments)
                results['audio_emotions'] = audio_emotions
                
                # Merge audio emotions with text emotions where possible
                merged_emotions = self._merge_audio_text_emotions(text_emotions, audio_emotions)
                results['emotions'] = merged_emotions
                
                self.logger.info(f"✅ Audio emotions: {len(audio_emotions)} detected")
            
            # Generate emotion summary
            results['summary'] = self._generate_emotion_summary(results['emotions'])
            
            # Update metadata
            processing_time = time.time() - start_time
            results['metadata']['processing_time'] = processing_time
            results['metadata']['total_emotions'] = len(results['emotions'])
            
            self.logger.info(f"🎭 Emotion detection completed in {processing_time:.2f}s")
            return results
            
        except Exception as e:
            self.logger.error(f"Emotion detection failed: {e}")
            return results
    
    def _merge_audio_text_emotions(self, text_emotions: List[EmotionPrediction], 
                                  audio_emotions: List[EmotionPrediction]) -> List[EmotionPrediction]:
        """
        Merge audio and text emotion predictions for enhanced accuracy.
        
        Args:
            text_emotions: Emotions detected from text
            audio_emotions: Emotions detected from audio
            
        Returns:
            Merged emotion predictions
        """
        merged = []
        
        # Start with text emotions (more accurate for content)
        merged.extend(text_emotions)
        
        # Add audio emotions that don't overlap significantly
        for audio_emotion in audio_emotions:
            # Find if there's a close text emotion
            overlap_found = False
            for text_emotion in text_emotions:
                # Check for temporal overlap
                if (abs(audio_emotion.segment_start - text_emotion.segment_start) < 5.0 and
                    abs(audio_emotion.segment_end - text_emotion.segment_end) < 5.0):
                    overlap_found = True
                    break
            
            if not overlap_found:
                merged.append(audio_emotion)
        
        # Sort by start time
        merged.sort(key=lambda x: x.segment_start)
        
        return merged
    
    def _generate_emotion_summary(self, emotions: List[EmotionPrediction]) -> Dict[str, Any]:
        """
        Generate summary statistics from detected emotions.
        
        Args:
            emotions: List of emotion predictions
            
        Returns:
            Summary dictionary
        """
        if not emotions:
            return {
                'dominant_emotion': 'neutral',
                'emotion_distribution': {'neutral': 1.0},
                'average_confidence': 0.0,
                'total_duration': 0.0
            }
        
        # Calculate emotion distribution
        emotion_counts = {}
        total_duration = 0.0
        total_confidence = 0.0
        
        for emotion in emotions:
            emotion_name = emotion.emotion
            duration = emotion.segment_end - emotion.segment_start
            
            emotion_counts[emotion_name] = emotion_counts.get(emotion_name, 0) + duration
            total_duration += duration
            total_confidence += emotion.confidence
        
        # Normalize distribution
        emotion_distribution = {}
        for emotion, duration in emotion_counts.items():
            emotion_distribution[emotion] = duration / total_duration if total_duration > 0 else 0
        
        # Find dominant emotion
        dominant_emotion = max(emotion_distribution.keys(), 
                             key=lambda x: emotion_distribution[x]) if emotion_distribution else 'neutral'
        
        return {
            'dominant_emotion': dominant_emotion,
            'emotion_distribution': emotion_distribution,
            'average_confidence': total_confidence / len(emotions) if emotions else 0.0,
            'total_duration': total_duration
        }
    
    def _extract_segment_embeddings(self, shared_embeddings: AudioEmbedding, 
                                  start_time: float, end_time: float) -> Optional[np.ndarray]:
        """
        Extract embeddings for a specific time segment.
        
        Args:
            shared_embeddings: Full audio embeddings
            start_time: Segment start time in seconds
            end_time: Segment end time in seconds
            
        Returns:
            Embeddings for the specified segment
        """
        try:
            timestamps = shared_embeddings.timestamps
            embeddings = shared_embeddings.embeddings
            
            # Find indices corresponding to the time segment
            start_idx = None
            end_idx = None
            
            for i, timestamp in enumerate(timestamps):
                if start_idx is None and timestamp >= start_time:
                    start_idx = i
                if timestamp <= end_time:
                    end_idx = i + 1
                elif end_idx is not None:
                    break
            
            if start_idx is None:
                start_idx = 0
            if end_idx is None:
                end_idx = len(embeddings)
                
            if start_idx >= end_idx:
                return None
                
            # Extract segment embeddings
            segment_embeddings = embeddings[start_idx:end_idx]
            
            # Average the embeddings for this segment
            if len(segment_embeddings) > 0:
                return np.mean(segment_embeddings, axis=0)
            else:
                return None
                
        except Exception as e:
            self.logger.warning(f"Error extracting segment embeddings: {e}")
            return None
    
    def _predict_emotion_from_embedding(self, embedding: np.ndarray, 
                                      start_time: float, end_time: float,
                                      text_id: Optional[int] = None,
                                      speaker: Optional[str] = None) -> EmotionPrediction:
        """
        Predict emotion from a single embedding vector.
        
        Args:
            embedding: Audio embedding vector
            start_time: Segment start time
            end_time: Segment end time
            text_id: Optional text segment ID
            speaker: Optional speaker information
            
        Returns:
            Emotion prediction for the segment
        """
        try:
            # Normalize the embedding
            normalized_embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            
            # Simple emotion detection based on embedding characteristics
            emotion_scores = self._calculate_emotion_scores(normalized_embedding)
            
            # Find the emotion with highest score
            best_emotion = max(emotion_scores.keys(), key=lambda k: emotion_scores[k])
            confidence = emotion_scores[best_emotion]
            
            return EmotionPrediction(
                emotion=best_emotion,
                confidence=confidence,
                probabilities=emotion_scores,
                segment_start=start_time,
                segment_end=end_time,
                text_id=text_id,
                speaker=speaker
            )
            
        except Exception as e:
            self.logger.warning(f"Error predicting emotion: {e}")
            return EmotionPrediction(
                emotion='neutral',
                confidence=0.5,
                probabilities={'neutral': 0.5},
                segment_start=start_time,
                segment_end=end_time,
                text_id=None,
                speaker=None
            )
    
    def _calculate_emotion_scores(self, embedding: np.ndarray) -> Dict[str, float]:
        """
        Calculate emotion scores from embedding using enhanced heuristic approach.
        
        This uses spectral and statistical features of audio embeddings to predict emotions.
        In production, this would use a trained machine learning model.
        
        Args:
            embedding: Normalized audio embedding
            
        Returns:
            Dictionary of emotion scores
        """
        # Enhanced statistical analysis of embeddings
        mean_val = np.mean(embedding)
        std_val = np.std(embedding)
        max_val = np.max(embedding)
        min_val = np.min(embedding)
        range_val = max_val - min_val
        skewness = np.mean(((embedding - mean_val) / (std_val + 1e-8)) ** 3)
        kurtosis = np.mean(((embedding - mean_val) / (std_val + 1e-8)) ** 4) - 3
        
        # Calculate spectral-like features
        fft_magnitudes = np.abs(np.fft.fft(embedding))
        spectral_centroid = np.sum(np.arange(len(fft_magnitudes)) * fft_magnitudes) / (np.sum(fft_magnitudes) + 1e-8)
        spectral_rolloff = np.percentile(fft_magnitudes, 85)
        
        # Calculate emotion probabilities based on enhanced features
        scores = {}
        
        # High energy, high variance - excitement or strong emotions
        if std_val > 0.15 and range_val > 0.6:
            if mean_val > 0.1:
                scores['excited'] = min(0.85, std_val * 4.5 + range_val * 0.3)
                scores['happy'] = min(0.75, std_val * 3.5 + mean_val * 2)
                scores['joy'] = min(0.8, (std_val + mean_val) * 3)
            else:
                scores['angry'] = min(0.85, std_val * 4 + abs(mean_val) * 2)
                scores['fear'] = min(0.7, std_val * 3 + range_val * 0.2)
        
        # Low variance often indicates calm or neutral states
        elif std_val < 0.08:
            scores['neutral'] = min(0.9, (0.1 - std_val) * 12)
            if abs(mean_val) < 0.05:
                scores['neutral'] += 0.1
        
        # Negative skewness with moderate variance - sadness
        elif skewness < -0.5 and std_val > 0.05:
            scores['sad'] = min(0.8, abs(skewness) * 0.8 + std_val * 2)
            scores['sadness'] = scores['sad']  # Alternative label
        
        # High kurtosis - surprise or sudden changes
        elif abs(kurtosis) > 1.5:
            scores['surprise'] = min(0.75, abs(kurtosis) * 0.4 + std_val * 2)
        
        # Spectral features for emotion detection
        if spectral_centroid > len(embedding) * 0.6:  # High frequency content
            scores['fear'] = scores.get('fear', 0) + 0.3
            scores['surprise'] = scores.get('surprise', 0) + 0.2
        elif spectral_centroid < len(embedding) * 0.3:  # Low frequency content
            scores['sad'] = scores.get('sad', 0) + 0.3
            scores['neutral'] = scores.get('neutral', 0) + 0.2
        
        # Positive mean with moderate-high variance - positive emotions
        if mean_val > 0.1 and 0.1 < std_val < 0.2:
            scores['happy'] = scores.get('happy', 0) + min(0.7, mean_val * 5)
            scores['joy'] = scores.get('joy', 0) + min(0.6, mean_val * 4)
        
        # Disgust detection based on irregular patterns
        if abs(skewness) > 1.0 and kurtosis > 2.0:
            scores['disgust'] = min(0.7, (abs(skewness) + kurtosis) * 0.25)
        
        # Ensure all emotions have some probability
        for emotion in self.EMOTION_CATEGORIES:
            if emotion not in scores:
                scores[emotion] = 0.05 / len(self.EMOTION_CATEGORIES)
        
        # Normalize probabilities
        total_score = sum(scores.values())
        if total_score > 0:
            scores = {k: v / total_score for k, v in scores.items()}
        else:
            # Fallback to uniform distribution
            uniform_prob = 1.0 / len(self.EMOTION_CATEGORIES)
            scores = {emotion: uniform_prob for emotion in self.EMOTION_CATEGORIES}
        
        return scores
    
    def generate_emotion_summary(self, predictions: List[EmotionPrediction]) -> Dict[str, Any]:
        """
        Generate a summary of emotions detected across all segments.
        
        Args:
            predictions: List of emotion predictions
            
        Returns:
            Dictionary containing emotion analysis summary
        """
        if not predictions:
            return {
                'total_segments': 0,
                'dominant_emotion': 'neutral',
                'emotion_distribution': {},
                'average_confidence': 0.0,
                'timeline': []
            }
        
        # Calculate emotion distribution
        emotion_counts = {}
        total_confidence = 0.0
        
        for pred in predictions:
            emotion = pred.emotion
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            total_confidence += pred.confidence
        
        # Find dominant emotion
        dominant_emotion = max(emotion_counts.keys(), key=lambda k: emotion_counts[k])
        
        # Calculate percentages
        total_segments = len(predictions)
        emotion_distribution = {
            emotion: (count / total_segments) * 100
            for emotion, count in emotion_counts.items()
        }
        
        # Create timeline
        timeline = [
            {
                'start': pred.segment_start,
                'end': pred.segment_end,
                'emotion': pred.emotion,
                'confidence': pred.confidence
            }
            for pred in predictions
        ]
        
        return {
            'total_segments': total_segments,
            'dominant_emotion': dominant_emotion,
            'emotion_distribution': emotion_distribution,
            'average_confidence': total_confidence / total_segments if total_segments > 0 else 0.0,
            'timeline': timeline,
            'detected_emotions': list(emotion_counts.keys())
        }
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save emotion analysis results to JSON file."""
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
                
            self.logger.info(f"Emotion analysis saved to: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving emotion analysis: {e}")
    
    def load_results(self, input_path: str) -> Optional[Dict[str, Any]]:
        """Load emotion analysis results from JSON file."""
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                return json.load(f)
                
        except Exception as e:
            self.logger.error(f"Error loading emotion analysis: {e}")
            return None
