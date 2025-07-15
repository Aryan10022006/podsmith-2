from typing import List, Dict, Any, Tuple
import json
import re
from pathlib import Path
import logging
import time
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import nltk
from collections import Counter
from ..utils.device_manager import DeviceManager

class SemanticSegmenter:
    """Fast semantic segmentation with optimized processing."""
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device_manager: DeviceManager = None):
        # Use faster, smaller embedding model
        self.fast_models = {
            "fastest": "sentence-transformers/all-MiniLM-L6-v2",     # Fastest
            "fast": "sentence-transformers/paraphrase-MiniLM-L6-v2",  # Fast + good
            "balanced": "sentence-transformers/all-mpnet-base-v2"     # Balanced
        }
        
        self.embedding_model_name = self.fast_models.get("fastest", embedding_model)
        self.device_manager = device_manager or DeviceManager()
        self.embedding_model = None
        self.logger = logging.getLogger(__name__)
        
        # Speed optimizations
        self.topic_cache = {}
        self.use_fast_clustering = True
        self.max_segments_per_block = 50  # Limit block size for speed
        self._ensure_nltk_data()
        
    def _ensure_nltk_data(self):
        """Ensure required NLTK data is available."""
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
        
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
    def load_embedding_model(self):
        """Load advanced sentence embedding model with timing."""
        if self.embedding_model is None:
            try:
                start_time = time.time()
                self.logger.info(f"Loading embedding model: {self.embedding_model_name}")
                
                self.embedding_model = SentenceTransformer(
                    self.embedding_model_name,
                    device=self.device_manager.device
                )
                
                load_time = time.time() - start_time
                self.logger.info(f"Embedding model loaded successfully in {load_time:.2f} seconds")
                
            except Exception as e:
                self.logger.error(f"Failed to load embedding model: {e}")
                self.embedding_model = None
    
    def _generate_intelligent_topic_description(self, text: str, keywords: List[str]) -> str:
        """Generate intelligent topic descriptions from content."""
        # Clean and process text
        sentences = nltk.sent_tokenize(text)
        
        # Find sentences with highest keyword density
        keyword_sentences = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            keyword_count = sum(1 for kw in keywords if kw.lower() in sentence_lower)
            if keyword_count > 0:
                keyword_sentences.append((sentence, keyword_count))
        
        # Sort by keyword density
        keyword_sentences.sort(key=lambda x: x[1], reverse=True)
        
        # Generate description from top keywords
        if len(keywords) >= 2:
            primary_topic = keywords[0]
            secondary_topic = keywords[1] if len(keywords) > 1 else ""
            
            # Create meaningful topic description
            if secondary_topic and primary_topic != secondary_topic:
                return f"{primary_topic} & {secondary_topic}"
            else:
                return primary_topic
        elif keywords:
            return keywords[0]
        else:
            return "general discussion"

    def segment_into_blocks(self, transcript_segments: List[Dict[str, Any]], 
                          min_block_duration: float = 20.0,  # Reduced for speed
                          semantic_threshold: float = 0.4) -> List[Dict[str, Any]]:  # Higher threshold for speed
        """Fast semantic segmentation with optimized processing."""
        start_time = time.time()
        
        if not transcript_segments:
            return []
        
        self.logger.info(f"Starting fast segmentation of {len(transcript_segments)} segments")
        
        # Quick duration-based blocks first
        duration_blocks = self._create_fast_blocks(transcript_segments, min_block_duration)
        
        # Skip expensive semantic processing for very short content
        if len(duration_blocks) <= 3:
            self.logger.info("Short content detected, skipping advanced semantic processing")
            formatted_blocks = self._format_simple_blocks(duration_blocks)
        else:
            # Load model only if needed
            self.load_embedding_model()
            
            if self.embedding_model and self.use_fast_clustering:
                semantic_blocks = self._refine_with_fast_semantics(duration_blocks, semantic_threshold)
                formatted_blocks = self._format_enhanced_semantic_blocks(semantic_blocks)
            else:
                formatted_blocks = self._format_simple_blocks(duration_blocks)
        
        segmentation_time = time.time() - start_time
        self.logger.info(f"Fast segmentation created {len(formatted_blocks)} blocks in {segmentation_time:.2f} seconds")
        
        return formatted_blocks
    
    def _create_fast_blocks(self, segments: List[Dict[str, Any]], 
                           min_duration: float) -> List[List[Dict[str, Any]]]:
        """Create blocks quickly based on duration and speaker changes."""
        blocks = []
        current_block = []
        block_start_time = 0.0
        current_speaker = None
        
        for i, segment in enumerate(segments):
            segment_speaker = segment.get("speaker", "SPEAKER_00")
            
            # Simple splitting logic for speed
            should_split = False
            
            if current_block:
                current_duration = segment.get("end", 0.0) - block_start_time
                
                # Split on duration or speaker change
                if current_duration >= min_duration and current_speaker != segment_speaker:
                    should_split = True
                elif current_duration >= min_duration * 2:  # Max block size
                    should_split = True
                elif len(current_block) >= self.max_segments_per_block:  # Limit segments per block
                    should_split = True
            
            if should_split and current_block:
                blocks.append(current_block)
                current_block = [segment]
                block_start_time = segment.get("start", 0.0)
                current_speaker = segment_speaker
            else:
                if not current_block:
                    block_start_time = segment.get("start", 0.0)
                current_block.append(segment)
                current_speaker = segment_speaker
        
        # Add the last block
        if current_block:
            blocks.append(current_block)
        
        return blocks
    
    def _refine_with_fast_semantics(self, blocks: List[List[Dict[str, Any]]], 
                                   threshold: float) -> List[List[Dict[str, Any]]]:
        """Fast semantic refinement using simplified processing."""
        if not self.embedding_model or len(blocks) < 2:
            return blocks
        
        try:
            # Use smaller text samples for speed
            block_samples = []
            for block in blocks:
                # Take only first few segments for speed
                sample_segments = block[:5]  # Limit to first 5 segments
                sample_text = " ".join([seg.get("text", "")[:100] for seg in sample_segments])  # Truncate text
                block_samples.append(sample_text)
            
            # Fast embedding generation
            embeddings = self.embedding_model.encode(block_samples, show_progress_bar=False)
            
            # Simple merging based on similarity
            refined_blocks = []
            current_block = blocks[0]
            
            for i in range(1, len(blocks)):
                # Calculate similarity quickly
                similarity = cosine_similarity([embeddings[i-1]], [embeddings[i]])[0][0]
                
                if similarity > threshold and len(current_block) < self.max_segments_per_block:
                    # Merge blocks
                    current_block.extend(blocks[i])
                else:
                    refined_blocks.append(current_block)
                    current_block = blocks[i]
            
            refined_blocks.append(current_block)
            return refined_blocks
            
        except Exception as e:
            self.logger.warning(f"Fast semantic refinement failed: {e}")
            return blocks
    
    def _format_simple_blocks(self, blocks: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Fast block formatting without expensive processing."""
        formatted_blocks = []
        
        for block_id, block_segments in enumerate(blocks):
            if not block_segments:
                continue
            
            # Basic metadata calculation
            start_time = block_segments[0].get("start", 0.0)
            end_time = block_segments[-1].get("end", 0.0)
            duration = end_time - start_time
            full_text = " ".join([segment.get("text", "") for segment in block_segments])
            
            # Simple speaker analysis
            speakers = [seg.get("speaker", "Unknown") for seg in block_segments]
            dominant_speaker = max(set(speakers), key=speakers.count) if speakers else "Unknown"
            
            # Fast keyword extraction
            keywords = self._extract_fast_keywords(full_text)
            topic_description = keywords[0] if keywords else "general discussion"
            
            # Get actual segment IDs from source transcript
            original_segment_ids = [seg.get("id", i) for i, seg in enumerate(block_segments)]
            
            formatted_block = {
                "block_id": block_id,  # Use 0-based indexing for consistency with source data
                "text": full_text,
                "start": start_time,
                "end": end_time,
                "duration": duration,
                "speaker": dominant_speaker,
                "speaker_distribution": {speaker: speakers.count(speaker) for speaker in set(speakers)},
                "segment_count": len(block_segments),
                "word_count": len(full_text.split()),
                "original_segments": original_segment_ids,  # Use actual segment IDs
                "topics": [topic_description],
                "topic_probabilities": {topic_description: 1.0},
                "primary_cluster": 0,
                "topic_keywords": keywords[:3],
                "semantic_coherence": 0.8,  # Default value for speed
                "content_density": len(full_text.split()) / max(duration, 1),
                "processing_method": "fast"
            }
            
            formatted_blocks.append(formatted_block)
        
        return formatted_blocks
    
    def _extract_fast_keywords(self, text: str, max_keywords: int = 5) -> List[str]:
        """Fast keyword extraction without heavy processing."""
        # Basic stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
                     'um', 'uh', 'ah', 'like', 'you', 'know', 'i', 'mean', 'just', 'really', 'very', 'so'}
        
        # Fast word extraction
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        
        # Count frequency quickly
        word_freq = Counter([word for word in words if word not in stop_words])
        
        # Return top keywords
        return [word for word, count in word_freq.most_common(max_keywords)]
    
    def _create_intelligent_blocks(self, segments: List[Dict[str, Any]], 
                                 min_duration: float) -> List[List[Dict[str, Any]]]:
        """Create intelligent blocks considering duration, speaker changes, and content shifts."""
        blocks = []
        current_block = []
        block_start_time = 0.0
        current_speaker = None
        
        for i, segment in enumerate(segments):
            segment_speaker = segment.get("speaker", "SPEAKER_00")
            
            # Start new block if speaker changes significantly or we hit duration limit
            should_split = False
            
            if current_block:
                current_duration = segment.get("end", 0.0) - block_start_time
                
                # Split on speaker change after minimum duration
                if (current_duration >= min_duration and 
                    current_speaker != segment_speaker and 
                    len(current_block) > 3):
                    should_split = True
                
                # Split if block gets too long
                elif current_duration >= min_duration * 2:
                    should_split = True
                
                # Split on content shift indicators
                text = segment.get("text", "").lower()
                if (current_duration >= min_duration * 0.5 and
                    any(indicator in text for indicator in [
                        "let's talk about", "moving on", "next topic", "another thing",
                        "by the way", "speaking of", "that reminds me"
                    ])):
                    should_split = True
            
            if should_split and current_block:
                blocks.append(current_block)
                current_block = [segment]
                block_start_time = segment.get("start", 0.0)
                current_speaker = segment_speaker
            else:
                if not current_block:
                    block_start_time = segment.get("start", 0.0)
                current_block.append(segment)
                current_speaker = segment_speaker
        
        # Add the last block
        if current_block:
            blocks.append(current_block)
        
        return blocks
              
    
    def _is_natural_break_point(self, current_segment: Dict[str, Any], 
                              all_segments: List[Dict[str, Any]]) -> bool:
        """Determine if this is a natural place to break between blocks."""
        current_idx = next((i for i, seg in enumerate(all_segments) if seg == current_segment), -1)
        
        if current_idx == -1 or current_idx >= len(all_segments) - 1:
            return True
        
        next_segment = all_segments[current_idx + 1]
        
        # Check for speaker change
        if current_segment.get("speaker") != next_segment.get("speaker"):
            return True
        
        # Check for significant pause (> 2 seconds)
        pause_duration = next_segment.get("start", 0.0) - current_segment.get("end", 0.0)
        if pause_duration > 2.0:
            return True
        
        # Check for sentence-ending punctuation
        text = current_segment.get("text", "").strip()
        if text.endswith(('.', '!', '?')):
            return True
        
        return False
    
    def _refine_with_semantics(self, duration_blocks: List[List[Dict[str, Any]]], 
                             threshold: float) -> List[List[Dict[str, Any]]]:
        """Refine blocks using semantic similarity."""
        if len(duration_blocks) <= 1:
            return duration_blocks
        
        # Get embeddings for each block
        block_texts = []
        for block in duration_blocks:
            block_text = " ".join([seg.get("text", "") for seg in block])
            block_texts.append(block_text)
        
        try:
            embeddings = self.embedding_model.encode(block_texts)
            
            # Calculate similarities between adjacent blocks
            refined_blocks = []
            current_merged_block = duration_blocks[0]
            
            for i in range(1, len(duration_blocks)):
                # Calculate cosine similarity
                similarity = np.dot(embeddings[i-1], embeddings[i]) / (
                    np.linalg.norm(embeddings[i-1]) * np.linalg.norm(embeddings[i])
                )
                
                if similarity > threshold:
                    # Merge with current block
                    current_merged_block.extend(duration_blocks[i])
                else:
                    # Start new block
                    refined_blocks.append(current_merged_block)
                    current_merged_block = duration_blocks[i]
            
            # Add final block
            refined_blocks.append(current_merged_block)
            
            return refined_blocks
            
        except Exception as e:
            self.logger.warning(f"Semantic refinement failed: {e}")
            return duration_blocks
    
    def _format_semantic_blocks(self, blocks: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Format segmented blocks with metadata."""
        formatted_blocks = []
        
        for i, block in enumerate(blocks):
            if not block:
                continue
            
            # Combine text from all segments in block
            combined_text = " ".join([seg.get("text", "") for seg in block])
            
            # Get time boundaries
            start_time = block[0].get("start", 0.0)
            end_time = block[-1].get("end", 0.0)
            
            # Determine dominant speaker
            speakers = [seg.get("speaker", "Unknown") for seg in block]
            dominant_speaker = max(set(speakers), key=speakers.count)
            
            formatted_block = {
                "block_id": i,  # Use 0-based indexing to align with transcript segments
                "text": combined_text.strip(),
                "start": start_time,
                "end": end_time,
                "duration": end_time - start_time,
                "speaker": dominant_speaker,
                "speaker_distribution": {speaker: speakers.count(speaker) for speaker in set(speakers)},
                "segment_count": len(block),
                "word_count": len(combined_text.split()),
                "original_segments": [seg.get("id", i) for i, seg in enumerate(block)]
            }
            
            formatted_blocks.append(formatted_block)
        
        return formatted_blocks
    
    def classify_topics(self, semantic_blocks: List[Dict[str, Any]], 
                       num_topics: int = 5,
                       max_topics_per_block: int = 3) -> List[Dict[str, Any]]:
        """Classify topics for semantic blocks."""
        if not semantic_blocks:
            return semantic_blocks
        
        self.logger.info(f"Classifying topics for {len(semantic_blocks)} blocks")
        
        # Extract texts
        block_texts = [block["text"] for block in semantic_blocks]
        
        try:
            # Use TF-IDF for topic modeling
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1
            )
            
            tfidf_matrix = vectorizer.fit_transform(block_texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Perform clustering
            n_clusters = min(num_topics, len(semantic_blocks))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(tfidf_matrix)
            
            # Get topic keywords for each cluster
            cluster_centers = kmeans.cluster_centers_
            topic_keywords = {}
            
            for i in range(n_clusters):
                # Get top keywords for this cluster
                top_indices = cluster_centers[i].argsort()[-10:][::-1]
                keywords = [feature_names[idx] for idx in top_indices]
                topic_keywords[i] = keywords
            
            # Assign topics to blocks
            enhanced_blocks = []
            for i, block in enumerate(semantic_blocks):
                cluster_id = cluster_labels[i]
                
                # Calculate topic probabilities (simplified)
                block_vector = tfidf_matrix[i].toarray()[0]
                topic_scores = {}
                
                for topic_id, center in enumerate(cluster_centers):
                    similarity = np.dot(block_vector, center) / (
                        np.linalg.norm(block_vector) * np.linalg.norm(center)
                    )
                    topic_scores[topic_id] = float(similarity)
                
                # Get top topics for this block
                sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
                top_topics = sorted_topics[:max_topics_per_block]
                
                # Format topic information
                topics = []
                topic_probabilities = {}
                
                for topic_id, score in top_topics:
                    if score > 0.1:  # Minimum relevance threshold
                        topic_name = " & ".join(topic_keywords[topic_id][:2])
                        topics.append(topic_name)
                        topic_probabilities[topic_name] = score
                
                enhanced_block = block.copy()
                enhanced_block.update({
                    "topics": topics,
                    "topic_probabilities": topic_probabilities,
                    "primary_cluster": int(cluster_id),
                    "topic_keywords": topic_keywords[cluster_id][:5]
                })
                
                enhanced_blocks.append(enhanced_block)
            
            self.logger.info(f"Topic classification completed with {n_clusters} topics")
            
            return enhanced_blocks
            
        except Exception as e:
            self.logger.warning(f"Topic classification failed: {e}")
            # Return original blocks without topic information
            return semantic_blocks
    
    def save_semantic_blocks(self, blocks: List[Dict[str, Any]], output_path: Path):
        """Save semantic blocks to JSON file."""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(blocks, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Semantic blocks saved to {output_path}")
    
    def get_segmentation_summary(self, blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics for segmentation."""
        if not blocks:
            return {}
        
        total_duration = sum(block.get("duration", 0) for block in blocks)
        total_words = sum(block.get("word_count", 0) for block in blocks)
        
        # Topic distribution
        all_topics = []
        for block in blocks:
            all_topics.extend(block.get("topics", []))
        
        topic_counts = {}
        for topic in all_topics:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        return {
            "total_blocks": len(blocks),
            "total_duration": total_duration,
            "total_words": total_words,
            "average_block_duration": total_duration / len(blocks) if blocks else 0,
            "average_words_per_block": total_words / len(blocks) if blocks else 0,
            "topic_distribution": dict(sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)),
            "unique_topics": len(topic_counts)
        }
    
    def _apply_intelligent_clustering(self, blocks: List[List[Dict[str, Any]]]) -> List[List[Dict[str, Any]]]:
        """Apply intelligent clustering to group semantically similar blocks."""
        if not blocks or len(blocks) < 2:
            return blocks
        
        try:
            # Extract text from each block
            block_texts = []
            for block in blocks:
                block_text = " ".join([segment.get("text", "") for segment in block])
                block_texts.append(block_text)
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(block_texts)
            
            # Calculate similarity matrix
            similarity_matrix = cosine_similarity(embeddings)
            
            # Use DBSCAN for intelligent clustering
            clustering = DBSCAN(eps=0.3, min_samples=1, metric='precomputed')
            distances = 1 - similarity_matrix  # Convert similarity to distance
            cluster_labels = clustering.fit_predict(distances)
            
            # Group blocks by cluster but maintain original order
            clustered_blocks = blocks.copy()
            
            # Add cluster information to blocks
            for i, block in enumerate(clustered_blocks):
                for segment in block:
                    segment['cluster_id'] = int(cluster_labels[i])
            
            return clustered_blocks
            
        except Exception as e:
            self.logger.warning(f"Clustering failed: {e}")
            return blocks

    def _format_enhanced_semantic_blocks(self, blocks: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Format blocks with enhanced metadata and intelligent topic descriptions."""
        formatted_blocks = []
        
        for block_id, block_segments in enumerate(blocks):
            if not block_segments:
                continue
            
            # Calculate block metadata
            start_time = block_segments[0].get("start", 0.0)
            end_time = block_segments[-1].get("end", 0.0)
            duration = end_time - start_time
            
            # Combine text
            full_text = " ".join([segment.get("text", "") for segment in block_segments])
            
            # Analyze speakers
            speaker_distribution = {}
            for segment in block_segments:
                speaker = segment.get("speaker", "Unknown")
                speaker_distribution[speaker] = speaker_distribution.get(speaker, 0) + 1
            
            # Extract keywords for topic generation
            keywords = self._extract_block_keywords(full_text)
            
            # Generate intelligent topic description
            topic_description = self._generate_intelligent_topic_description(full_text, keywords)
            
            # Get cluster information
            cluster_id = block_segments[0].get('cluster_id', 0) if block_segments else 0
            
            # Get actual segment IDs from source transcript
            original_segment_ids = [seg.get("id", i) for i, seg in enumerate(block_segments)]
            
            formatted_block = {
                "block_id": block_id,  # Use 0-based indexing for consistency
                "text": full_text,
                "start": start_time,
                "end": end_time,
                "duration": duration,
                "speaker": max(speaker_distribution.items(), key=lambda x: x[1])[0] if speaker_distribution else "Unknown",
                "speaker_distribution": speaker_distribution,
                "segment_count": len(block_segments),
                "word_count": len(full_text.split()),
                "original_segments": original_segment_ids,  # Use actual segment IDs
                "topics": [topic_description],
                "topic_probabilities": {topic_description: 1.0},
                "primary_cluster": cluster_id,
                "topic_keywords": keywords[:5],  # Top 5 keywords
                "semantic_coherence": self._calculate_coherence_score(block_segments),
                "content_density": len(full_text.split()) / max(duration, 1),  # Words per second
            }
            
            formatted_blocks.append(formatted_block)
        
        return formatted_blocks
    
    def _extract_block_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract meaningful keywords from block text."""
        try:
            from nltk.corpus import stopwords
            stop_words = set(stopwords.words('english'))
        except:
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        # Add podcast-specific stop words
        stop_words.update(['um', 'uh', 'ah', 'like', 'you', 'know', 'i', 'mean', 'just', 'really', 'very'])
        
        # Clean and tokenize
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter out stop words and count frequency
        word_freq = Counter([word for word in words if word not in stop_words])
        
        # Get most common words
        keywords = [word for word, count in word_freq.most_common(max_keywords)]
        
        return keywords
    
    def _calculate_coherence_score(self, segments: List[Dict[str, Any]]) -> float:
        """Calculate semantic coherence score for a block."""
        if len(segments) < 2:
            return 1.0
        
        try:
            texts = [segment.get("text", "") for segment in segments]
            if not any(texts):
                return 0.0
            
            # Calculate average similarity between consecutive segments
            embeddings = self.embedding_model.encode(texts)
            similarities = []
            
            for i in range(len(embeddings) - 1):
                sim = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
                similarities.append(sim)
            
            return float(np.mean(similarities)) if similarities else 0.0
            
        except Exception:
            return 0.5  # Default moderate coherence
    
    def _refine_with_advanced_semantics(self, blocks: List[List[Dict[str, Any]]], 
                                       threshold: float) -> List[List[Dict[str, Any]]]:
        """Refine blocks using advanced semantic analysis."""
        if not self.embedding_model or len(blocks) < 2:
            return blocks
        
        try:
            refined_blocks = []
            current_block = blocks[0]
            
            for i in range(1, len(blocks)):
                # Get text from current and next block
                current_text = " ".join([seg.get("text", "") for seg in current_block])
                next_text = " ".join([seg.get("text", "") for seg in blocks[i]])
                
                # Calculate semantic similarity
                embeddings = self.embedding_model.encode([current_text, next_text])
                similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
                
                # Merge blocks if similarity is high
                if similarity > threshold:
                    current_block.extend(blocks[i])
                else:
                    refined_blocks.append(current_block)
                    current_block = blocks[i]
            
            # Add the last block
            refined_blocks.append(current_block)
            
            return refined_blocks
            
        except Exception as e:
            self.logger.warning(f"Semantic refinement failed: {e}")
            return blocks