import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
import logging
import re
import math
import time
from collections import Counter
import yake
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import nltk

class KeywordExtractor:
    """Advanced keyword and keyphrase extraction with stop word filtering."""
    
    def __init__(self, method: str = "tfidf"):
        self.method = method  # "tfidf", "yake", "textrank"
        self.logger = logging.getLogger(__name__)
        
        # Enhanced stop words list
        self.stop_words = self._get_enhanced_stopwords()
        
        # Download NLTK data if needed
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
            
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger', quiet=True)
    
    def _get_enhanced_stopwords(self) -> Set[str]:
        """Get comprehensive stop words including domain-specific ones."""
        try:
            from nltk.corpus import stopwords
            english_stops = set(stopwords.words('english'))
        except:
            # Fallback stop words if NLTK not available
            english_stops = {
                'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
                'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
                'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
                'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
                'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after',
                'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                'further', 'then', 'once'
            }
        
        # Add podcast-specific stop words
        podcast_stops = {
            'um', 'uh', 'ah', 'er', 'like', 'you know', 'i mean', 'sort of', 'kind of',
            'basically', 'actually', 'really', 'very', 'quite', 'just', 'now', 'well',
            'so', 'yeah', 'yes', 'no', 'okay', 'ok', 'right', 'sure', 'exactly',
            'absolutely', 'definitely', 'probably', 'maybe', 'perhaps', 'obviously',
            'clearly', 'certainly', 'course', 'thing', 'things', 'stuff', 'way', 'ways',
            'time', 'times', 'people', 'person', 'guy', 'guys', 'get', 'got', 'getting',
            'go', 'going', 'went', 'come', 'coming', 'came', 'say', 'says', 'said',
            'saying', 'tell', 'telling', 'told', 'talk', 'talking', 'talked', 'think',
            'thinking', 'thought', 'know', 'knowing', 'knew', 'see', 'seeing', 'saw',
            'look', 'looking', 'looked', 'want', 'wanting', 'wanted', 'need', 'needing',
            'needed', 'make', 'making', 'made', 'take', 'taking', 'took', 'give',
            'giving', 'gave', 'put', 'putting', 'something', 'someone', 'somewhere',
            'somehow', 'somewhat', 'anything', 'anyone', 'anywhere', 'anyhow'
        }
        
        return english_stops.union(podcast_stops)
    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text for keyword extraction."""
        # Remove URLs, emails, and special characters
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _filter_keywords(self, keywords: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter out stop words and low-quality keywords."""
        filtered = []
        
        for kw in keywords:
            keyword = kw.get('keyword', '').lower().strip()
            
            # Skip if empty or too short
            if len(keyword) < 2:
                continue
                
            # Skip if it's just a stop word
            if keyword in self.stop_words:
                continue
                
            # Skip if all words are stop words
            words = keyword.split()
            if all(word.lower() in self.stop_words for word in words):
                continue
                
            # Skip if mostly numbers
            if sum(c.isdigit() for c in keyword) > len(keyword) / 2:
                continue
                
            # Skip if too generic
            generic_terms = {
                'thing', 'stuff', 'something', 'anything', 'everything', 'nothing',
                'someone', 'anyone', 'everyone', 'one', 'two', 'three'
            }
            if keyword in generic_terms:
                continue
            
            filtered.append(kw)
        
        return filtered

    def extract_global_keywords(self, semantic_blocks: List[Dict[str, Any]], 
                              max_keywords: int = 20,
                              min_score: float = 0.1) -> List[Dict[str, Any]]:
        """Extract high-quality keywords from the entire podcast content."""
        start_time = time.time()
        
        # Combine all text
        all_text = " ".join([block.get("text", "") for block in semantic_blocks])
        
        if not all_text.strip():
            return []
        
        # Clean the text
        cleaned_text = self._clean_text(all_text)
        
        self.logger.info(f"Extracting global keywords using {self.method}")
        
        if self.method == "yake":
            keywords = self._extract_yake_keywords(cleaned_text, max_keywords, min_score)
        elif self.method == "textrank":
            keywords = self._extract_textrank_keywords(cleaned_text, max_keywords, min_score)
        else:  # tfidf
            keywords = self._extract_tfidf_keywords([cleaned_text], max_keywords, min_score)
        
        # Filter out stop words and low-quality keywords
        filtered_keywords = self._filter_keywords(keywords)
        
        # Add semantic relevance scoring
        enhanced_keywords = self._enhance_with_context(filtered_keywords, semantic_blocks)
        
        extraction_time = time.time() - start_time
        self.logger.info(f"Extracted {len(enhanced_keywords)} global keywords in {extraction_time:.2f} seconds")
        
        return enhanced_keywords[:max_keywords]
    
    def _enhance_with_context(self, keywords: List[Dict[str, Any]], 
                             semantic_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhance keywords with contextual relevance scoring."""
        enhanced = []
        
        for kw in keywords:
            keyword = kw.get('keyword', '')
            base_score = kw.get('score', 0.0)
            
            # Count occurrences across different semantic blocks
            block_appearances = 0
            total_occurrences = 0
            
            for block in semantic_blocks:
                text = block.get('text', '').lower()
                if keyword.lower() in text:
                    block_appearances += 1
                    total_occurrences += text.count(keyword.lower())
            
            # Calculate context relevance
            block_distribution_score = block_appearances / len(semantic_blocks) if semantic_blocks else 0
            frequency_score = min(total_occurrences / 10, 1.0)  # Normalize frequency
            
            # Boost score for keywords that appear in multiple blocks
            context_boost = 1.0 + (block_distribution_score * 0.5) + (frequency_score * 0.3)
            enhanced_score = base_score * context_boost
            
            enhanced.append({
                'keyword': keyword,
                'score': round(enhanced_score, 4),
                'frequency': total_occurrences,
                'block_appearances': block_appearances,
                'context_relevance': round(context_boost - 1.0, 3)
            })
        
        # Sort by enhanced score
        enhanced.sort(key=lambda x: x['score'], reverse=True)
        return enhanced

    def extract_block_keywords(self, semantic_blocks: List[Dict[str, Any]], 
                             max_keywords_per_block: int = 10) -> Dict[str, List[str]]:
        """Extract high-quality keywords for each semantic block with timing."""
        start_time = time.time()
        block_keywords = {}
        
        self.logger.info(f"Extracting keywords for {len(semantic_blocks)} blocks")
        
        block_texts = []
        block_ids = []
        
        for block in semantic_blocks:
            text = block.get("text", "").strip()
            if text:
                cleaned_text = self._clean_text(text)
                block_texts.append(cleaned_text)
                block_ids.append(block.get("block_id", len(block_ids)))
        
        if not block_texts:
            return {}
        
        # Extract keywords using TF-IDF for better block-specific results
        try:
            # Use custom stop words
            vectorizer = TfidfVectorizer(
                max_features=max_keywords_per_block * len(block_texts),
                stop_words=list(self.stop_words),
                ngram_range=(1, 3),  # Include phrases up to 3 words
                min_df=1,
                max_df=0.8,
                lowercase=True
            )
            
            tfidf_matrix = vectorizer.fit_transform(block_texts)
            feature_names = vectorizer.get_feature_names_out()
            
            for i, block_id in enumerate(block_ids):
                # Get TF-IDF scores for this block
                scores = tfidf_matrix[i].toarray().flatten()
                
                # Create keyword-score pairs
                keyword_scores = [
                    (feature_names[j], scores[j]) 
                    for j in range(len(feature_names)) 
                    if scores[j] > 0
                ]
                
                # Sort by score and filter
                keyword_scores.sort(key=lambda x: x[1], reverse=True)
                
                # Filter and format keywords
                filtered_keywords = []
                for keyword, score in keyword_scores[:max_keywords_per_block * 2]:
                    # Additional filtering
                    if (len(keyword) > 2 and 
                        keyword not in self.stop_words and
                        not keyword.isdigit() and
                        score > 0.01):
                        filtered_keywords.append({
                            'keyword': keyword,
                            'score': round(score, 4),
                            'type': 'phrase' if ' ' in keyword else 'word'
                        })
                
                block_keywords[str(block_id)] = filtered_keywords[:max_keywords_per_block]
        
        except Exception as e:
            self.logger.warning(f"TF-IDF extraction failed: {e}")
            # Fallback to YAKE for individual blocks
            for i, (block_id, text) in enumerate(zip(block_ids, block_texts)):
                try:
                    yake_keywords = self._extract_yake_keywords(text, max_keywords_per_block, 0.1)
                    filtered = self._filter_keywords(yake_keywords)
                    block_keywords[str(block_id)] = filtered[:max_keywords_per_block]
                except Exception as fallback_error:
                    self.logger.warning(f"YAKE fallback failed for block {block_id}: {fallback_error}")
                    block_keywords[str(block_id)] = []
        
        extraction_time = time.time() - start_time
        self.logger.info(f"Extracted block keywords in {extraction_time:.2f} seconds")
        
        return block_keywords
    
    def _extract_tfidf_keywords(self, texts: List[str], max_keywords: int, 
                              min_score: float) -> List[Dict[str, Any]]:
        """Extract keywords using TF-IDF."""
        try:
            vectorizer = TfidfVectorizer(
                max_features=max_keywords * 3,
                stop_words='english',
                ngram_range=(1, 3),
                min_df=1,
                max_df=0.8
            )
            
            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Sum TF-IDF scores across all documents
            scores = np.sum(tfidf_matrix.toarray(), axis=0)
            
            # Get top keywords
            top_indices = scores.argsort()[-max_keywords:][::-1]
            
            keywords = []
            for idx in top_indices:
                score = float(scores[idx])
                if score >= min_score:
                    keywords.append({
                        "keyword": feature_names[idx],
                        "confidence": score,
                        "method": "tfidf"
                    })
            
            return keywords
            
        except Exception as e:
            self.logger.warning(f"TF-IDF extraction failed: {e}")
            return self._extract_frequency_keywords(texts[0] if texts else "", max_keywords)
    
    def _extract_yake_keywords(self, text: str, max_keywords: int, 
                             min_score: float) -> List[Dict[str, Any]]:
        """Extract keywords using YAKE algorithm."""
        try:
            # Configure YAKE
            kw_extractor = yake.KeywordExtractor(
                lan="en",
                n=3,  # Maximum number of words in keyphrase
                dedupLim=0.7,  # Deduplication threshold
                top=max_keywords * 2,  # Extract more, then filter
                features=None
            )
            
            keywords_raw = kw_extractor.extract_keywords(text)
            
            keywords = []
            for score, keyword in keywords_raw:
                # YAKE returns lower scores for better keywords
                confidence = 1.0 / (1.0 + score)  # Convert to 0-1 scale
                
                if confidence >= min_score:
                    keywords.append({
                        "keyword": keyword,
                        "confidence": confidence,
                        "method": "yake"
                    })
            
            return keywords[:max_keywords]
            
        except Exception as e:
            self.logger.warning(f"YAKE extraction failed: {e}")
            return self._extract_frequency_keywords(text, max_keywords)
    
    def _extract_textrank_keywords(self, text: str, max_keywords: int, 
                                 min_score: float) -> List[Dict[str, Any]]:
        """Extract keywords using TextRank-like algorithm."""
        # Simple implementation of TextRank
        try:
            # Tokenize and clean
            words = self._tokenize_and_clean(text)
            
            if len(words) < 5:
                return []
            
            # Create word co-occurrence graph
            window_size = 4
            graph = {}
            
            for i, word in enumerate(words):
                if word not in graph:
                    graph[word] = {}
                
                # Look at words in window
                start = max(0, i - window_size)
                end = min(len(words), i + window_size + 1)
                
                for j in range(start, end):
                    if i != j:
                        other_word = words[j]
                        if other_word not in graph[word]:
                            graph[word][other_word] = 0
                        graph[word][other_word] += 1
            
            # Simple PageRank-like scoring
            scores = {}
            for word in graph:
                score = sum(graph[word].values()) / len(graph[word]) if graph[word] else 0
                scores[word] = score
            
            # Sort and format
            sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            
            keywords = []
            for word, score in sorted_words[:max_keywords]:
                normalized_score = min(score / 10.0, 1.0)  # Normalize score
                if normalized_score >= min_score:
                    keywords.append({
                        "keyword": word,
                        "confidence": normalized_score,
                        "method": "textrank"
                    })
            
            return keywords
            
        except Exception as e:
            self.logger.warning(f"TextRank extraction failed: {e}")
            return self._extract_frequency_keywords(text, max_keywords)
    
    def _extract_frequency_keywords(self, text: str, max_keywords: int) -> List[Dict[str, Any]]:
        """Fallback frequency-based keyword extraction."""
        words = self._tokenize_and_clean(text)
        
        if not words:
            return []
        
        # Count frequencies
        word_freq = Counter(words)
        
        # Remove very common words (additional stopwords)
        common_words = {'say', 'said', 'says', 'think', 'thought', 'know', 'thing', 'things', 
                       'way', 'time', 'people', 'person', 'really', 'actually', 'basically'}
        
        filtered_freq = {word: freq for word, freq in word_freq.items() 
                        if word not in common_words and freq > 1}
        
        # Get top words
        top_words = sorted(filtered_freq.items(), key=lambda x: x[1], reverse=True)
        
        keywords = []
        max_freq = top_words[0][1] if top_words else 1
        
        for word, freq in top_words[:max_keywords]:
            confidence = freq / max_freq
            keywords.append({
                "keyword": word,
                "confidence": confidence,
                "method": "frequency"
            })
        
        return keywords
    
    def _tokenize_and_clean(self, text: str) -> List[str]:
        """Tokenize and clean text for keyword extraction."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation and split
        text = re.sub(r'[^\w\s]', ' ', text)
        words = text.split()
        
        # Filter words
        filtered_words = []
        stopwords = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
            'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
            'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
            'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
            'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after',
            'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
            'further', 'then', 'once'
        }
        
        for word in words:
            if (len(word) > 2 and 
                word not in stopwords and 
                word.isalpha() and
                not word.isdigit()):
                filtered_words.append(word)
        
        return filtered_words
    
    def analyze_keyword_trends(self, block_keywords: Dict[str, List[str]], 
                             semantic_blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze how keywords change throughout the podcast."""
        # Collect all keywords with their block positions
        keyword_positions = {}
        
        for block_key, keywords in block_keywords.items():
            try:
                block_id = int(block_key.split('_')[1])
                
                for keyword in keywords:
                    if isinstance(keyword, dict):
                        kw = keyword.get("keyword", keyword)
                    else:
                        kw = keyword
                    
                    if kw not in keyword_positions:
                        keyword_positions[kw] = []
                    keyword_positions[kw].append(block_id)
            except:
                continue
        
        # Analyze trends
        trending_keywords = {}
        consistent_keywords = {}
        
        total_blocks = len(semantic_blocks)
        
        for keyword, positions in keyword_positions.items():
            frequency = len(positions)
            spread = max(positions) - min(positions) if len(positions) > 1 else 0
            
            # Consistent keywords appear throughout
            if frequency >= total_blocks * 0.3:  # Appears in 30%+ of blocks
                consistent_keywords[keyword] = {
                    "frequency": frequency,
                    "consistency_score": frequency / total_blocks,
                    "spread": spread
                }
            
            # Trending keywords appear in clusters
            if len(positions) >= 2:
                # Calculate if keyword appears in clusters (trending)
                position_gaps = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
                avg_gap = sum(position_gaps) / len(position_gaps) if position_gaps else 0
                
                if avg_gap <= 3:  # Keywords appear in close blocks
                    trending_keywords[keyword] = {
                        "frequency": frequency,
                        "average_gap": avg_gap,
                        "positions": positions
                    }
        
        return {
            "consistent_keywords": dict(sorted(consistent_keywords.items(), 
                                             key=lambda x: x[1]["consistency_score"], 
                                             reverse=True)[:10]),
            "trending_keywords": dict(sorted(trending_keywords.items(), 
                                           key=lambda x: x[1]["frequency"], 
                                           reverse=True)[:10]),
            "total_unique_keywords": len(keyword_positions),
            "keyword_density": len(keyword_positions) / total_blocks if total_blocks > 0 else 0
        }
    
    def save_keywords(self, global_keywords: List[Dict[str, Any]], 
                     block_keywords: Dict[str, List[str]], 
                     keyword_trends: Dict[str, Any],
                     output_path: Path):
        """Save keyword analysis results to JSON file."""
        output_data = {
            "global_keywords": global_keywords,
            "per_block": block_keywords,
            "keyword_trends": keyword_trends,
            "extraction_metadata": {
                "method": self.method,
                "total_global_keywords": len(global_keywords),
                "blocks_analyzed": len(block_keywords),
                "extraction_timestamp": str(Path(output_path).stat().st_mtime) if output_path.exists() else None
            }
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Keywords saved to {output_path}")