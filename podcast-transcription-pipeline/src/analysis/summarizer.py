import torch
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, 
    BartForConditionalGeneration, BartTokenizer
)
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
import re
import time
from datetime import datetime
from ..utils.device_manager import DeviceManager
import openai
import os

# Basic NLP libraries
try:
    import nltk
    # Download required NLTK data
    for resource in ['punkt', 'stopwords']:
        try:
            nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}')
        except LookupError:
            nltk.download(resource, quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("NLTK not available")


class AdvancedDomainAnalyzer:
    """Advanced domain detection using open-source models with semantic understanding."""
    
    def __init__(self, device_manager: DeviceManager = None):
        self.logger = logging.getLogger(__name__)
        self.device_manager = device_manager or DeviceManager()
        self.domain_classifier = None
        self.sentiment_analyzer = None
        self.topic_model = None
        
        # Domain mapping for better understanding
        self.domain_keywords = {
            'technology': ['software', 'programming', 'ai', 'machine learning', 'computer', 'tech', 'digital', 'algorithm', 'code', 'data'],
            'business': ['company', 'market', 'revenue', 'profit', 'strategy', 'management', 'investment', 'startup', 'entrepreneur', 'finance'],
            'health': ['medical', 'health', 'doctor', 'patient', 'treatment', 'medicine', 'therapy', 'disease', 'wellness', 'fitness'],
            'education': ['learning', 'student', 'teacher', 'school', 'university', 'education', 'knowledge', 'study', 'academic', 'research'],
            'entertainment': ['movie', 'music', 'game', 'show', 'entertainment', 'fun', 'celebrity', 'comedy', 'drama', 'sport'],
            'spiritual': ['god', 'prayer', 'faith', 'spiritual', 'religion', 'divine', 'meditation', 'soul', 'worship', 'blessed'],
            'lifestyle': ['life', 'lifestyle', 'personal', 'relationship', 'family', 'home', 'travel', 'food', 'fashion', 'hobby']
        }

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for better summarization."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove filler words and sounds
        filler_patterns = [
            r'\b(um|uh|er|ah|like|you know|sort of|kind of)\b',
            r'\[.*?\]',  # Remove speaker annotations
            r'\(.*?\)'   # Remove parenthetical content
        ]
        for pattern in filler_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        # Clean up punctuation
        text = re.sub(r'[.]{2,}', '.', text)
        text = re.sub(r'[,]{2,}', ',', text)
        return text.strip()
    
    def load_models(self):
        """Load open-source models for domain analysis."""
        try:
            # Try to load lightweight models first, fallback to keyword-based analysis if failed
            from transformers import pipeline
            
            # Use a lighter model for domain classification
            self.domain_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if torch.cuda.is_available() else -1
            )
            
            self.logger.info("✅ Domain classification model loaded successfully")
            return True
            
        except Exception as e:
            self.logger.warning(f"Failed to load domain analysis models, using keyword-based fallback: {e}")
            # Set to None to trigger fallback
            self.domain_classifier = None
            return False
    
    def analyze_content_domain(self, text: str) -> Dict[str, Any]:
        """
        Analyze content domain using advanced semantic understanding with automatic fallback.
        """
        try:
            # Try ML-based domain detection first
            if self.domain_classifier:
                return self._ml_based_domain_analysis(text)
            else:
                # Fallback to enhanced keyword-based analysis
                return self._enhanced_keyword_domain_analysis(text)
                
        except Exception as e:
            self.logger.warning(f"Domain analysis failed, using basic fallback: {e}")
            return self._get_fallback_domain_analysis(text)
    
    def _ml_based_domain_analysis(self, text: str) -> Dict[str, Any]:
        """ML-based domain analysis using zero-shot classification."""
        try:
            # Prepare candidate domains
            domains = list(self.domain_keywords.keys())
            
            # Get domain classification
            domain_result = self.domain_classifier(text[:512], domains)
            primary_domain = domain_result['labels'][0]
            confidence = domain_result['scores'][0]
            
            # Simple sentiment analysis (positive/negative/neutral)
            sentiment = self._analyze_sentiment_simple(text)
            
            # Extract key themes using keyword analysis
            key_themes = self._extract_themes_from_keywords(text, primary_domain)
            
            # Determine content complexity
            complexity = self._analyze_complexity(text)
            
            # Determine content type
            content_type = self._determine_content_type(text)
            
            return {
                "primary_domain": primary_domain,
                "secondary_domains": domain_result['labels'][1:3] if len(domain_result['labels']) > 1 else [],
                "content_type": content_type,
                "key_themes": key_themes,
                "complexity_level": complexity,
                "sentiment": sentiment,
                "sentiment_score": confidence,
                "target_audience": self._determine_audience(primary_domain, complexity),
                "summarization_approach": self._get_summarization_approach(primary_domain, content_type),
                "focus_areas": self._get_focus_areas(primary_domain, key_themes),
                "confidence": confidence
            }
            
        except Exception as e:
            self.logger.warning(f"ML domain analysis failed: {e}")
            return self._enhanced_keyword_domain_analysis(text)
    
    def _enhanced_keyword_domain_analysis(self, text: str) -> Dict[str, Any]:
        """Enhanced keyword-based domain analysis as reliable fallback."""
        text_lower = text.lower()
        
        # Score each domain based on keyword matches
        domain_scores = {}
        for domain, keywords in self.domain_keywords.items():
            score = 0
            for keyword in keywords:
                # Count occurrences and weight by importance
                occurrences = text_lower.count(keyword)
                if occurrences > 0:
                    # Weight longer keywords more heavily
                    weight = len(keyword.split()) * 2
                    score += occurrences * weight
            domain_scores[domain] = score
        
        # Find primary domain
        if domain_scores and max(domain_scores.values()) > 0:
            primary_domain = max(domain_scores, key=domain_scores.get)
            confidence = min(domain_scores[primary_domain] / 20.0, 0.9)  # Normalize to 0-0.9
            
            # Get secondary domains
            sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
            secondary_domains = [domain for domain, score in sorted_domains[1:3] if score > 0]
        else:
            primary_domain = "general"
            confidence = 0.3
            secondary_domains = []
        
        # Extract themes and analyze content
        key_themes = self._extract_themes_from_keywords(text, primary_domain)
        complexity = self._analyze_complexity(text)
        content_type = self._determine_content_type(text)
        sentiment = self._analyze_sentiment_simple(text)
        
        return {
            "primary_domain": primary_domain,
            "secondary_domains": secondary_domains,
            "content_type": content_type,
            "key_themes": key_themes,
            "complexity_level": complexity,
            "sentiment": sentiment,
            "sentiment_score": confidence,
            "target_audience": self._determine_audience(primary_domain, complexity),
            "summarization_approach": self._get_summarization_approach(primary_domain, content_type),
            "focus_areas": self._get_focus_areas(primary_domain, key_themes),
            "confidence": confidence
        }
    
    def _analyze_sentiment_simple(self, text: str) -> str:
        """Simple rule-based sentiment analysis."""
        text_lower = text.lower()
        
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'awesome', 
                         'love', 'like', 'enjoy', 'happy', 'pleased', 'satisfied', 'success', 'achievement']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike', 'angry', 'sad', 
                         'disappointed', 'frustrated', 'problem', 'issue', 'difficult', 'challenge', 'failure']
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"
    
    def _extract_themes_from_keywords(self, text: str, domain: str) -> List[str]:
        """Extract themes based on domain-specific keywords."""
        text_lower = text.lower()
        domain_keywords = self.domain_keywords.get(domain, [])
        
        found_themes = []
        for keyword in domain_keywords:
            if keyword in text_lower:
                found_themes.append(keyword)
        
        # Add general themes
        general_themes = []
        if any(word in text_lower for word in ['problem', 'solution', 'challenge']):
            general_themes.append('problem-solving')
        if any(word in text_lower for word in ['example', 'case', 'story']):
            general_themes.append('examples')
        if any(word in text_lower for word in ['future', 'predict', 'forecast']):
            general_themes.append('future-outlook')
        
        return (found_themes + general_themes)[:5]
    
    def _analyze_complexity(self, text: str) -> str:
        """Analyze text complexity."""
        words = text.split()
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        
        # Count complex indicators
        complex_indicators = 0
        if avg_word_length > 6:
            complex_indicators += 1
        if any(word in text.lower() for word in ['therefore', 'however', 'consequently', 'furthermore']):
            complex_indicators += 1
        if len(text.split('.')) > 10:  # Many sentences
            complex_indicators += 1
        
        if complex_indicators >= 2:
            return "high"
        elif complex_indicators == 1:
            return "medium"
        else:
            return "low"
    
    def _determine_content_type(self, text: str) -> str:
        """Determine the type of content."""
        text_lower = text.lower()
        
        if any(phrase in text_lower for phrase in ['question:', 'answer:', 'q:', 'a:']):
            return "interview"
        elif any(phrase in text_lower for phrase in ['welcome', 'today we', 'hello everyone']):
            return "presentation"
        elif any(phrase in text_lower for phrase in ['let me explain', 'first', 'second', 'finally']):
            return "lecture"
        elif any(phrase in text_lower for phrase in ['i think', 'in my opinion', 'personally']):
            return "discussion"
        else:
            return "general_content"
    
    def _determine_audience(self, domain: str, complexity: str) -> str:
        """Determine target audience based on domain and complexity."""
        if complexity == "high":
            return f"{domain}_experts"
        elif complexity == "medium":
            return f"{domain}_professionals"
        else:
            return "general_public"
    
    def _get_summarization_approach(self, domain: str, content_type: str) -> str:
        """Get the best summarization approach for the content."""
        if domain == "technology":
            return "technical_summary"
        elif domain == "business":
            return "analytical_summary"
        elif domain == "education":
            return "instructional_summary"
        elif content_type == "interview":
            return "conversational_summary"
        else:
            return "balanced_summary"
    
    def _get_focus_areas(self, domain: str, themes: List[str]) -> List[str]:
        """Get focus areas for summarization."""
        focus_areas = ["main_points", "key_insights"]
        
        if domain == "business":
            focus_areas.extend(["strategies", "outcomes"])
        elif domain == "technology":
            focus_areas.extend(["implementations", "benefits"])
        elif domain == "education":
            focus_areas.extend(["concepts", "examples"])
        
        if themes:
            focus_areas.append("theme_connections")
        
        return focus_areas[:4]
    
    def _get_fallback_domain_analysis(self, text: str) -> Dict[str, Any]:
        """Fallback analysis using keyword matching."""
        text_lower = text.lower()
        
        # Simple keyword-based domain detection
        domain_scores = {}
        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            domain_scores[domain] = score
        
        primary_domain = max(domain_scores, key=domain_scores.get) if domain_scores else "general"
        
        return {
            "primary_domain": primary_domain,
            "secondary_domains": [],
            "content_type": "general_content",
            "key_themes": self._extract_themes_from_keywords(text, primary_domain),
            "complexity_level": self._analyze_complexity(text),
            "sentiment": "neutral",
            "sentiment_score": 0.5,
            "target_audience": "general",
            "summarization_approach": "balanced_summary",
            "focus_areas": ["main_points", "key_insights"],
            "confidence": 0.3
        }
    
    def generate_domain_specific_prompt(self, text: str, domain_analysis: Dict[str, Any], 
                                      summary_length: str = "medium") -> str:
        """Generate domain-specific summarization prompt - ONLY return the original text."""
        # For transformer models, we don't need complex prompts - just return clean text
        # The domain analysis will guide post-processing instead
        return self._preprocess_text(text)


class Summarizer:
    """
    Advanced summarization system with domain detection and semantic understanding.
    Uses open-source models for high-quality summarization.
    """
    
    def __init__(self, model_name: str = None, device_manager: DeviceManager = None, config_path: str = None):
        self.logger = logging.getLogger(__name__)
        self.device_manager = device_manager or DeviceManager()
        self.models = {}
        self.tokenizers = {}
        self.summarization_pipeline = None
        self.domain_analyzer = AdvancedDomainAnalyzer(self.device_manager)
        
        self.performance_metrics = {
            'domain_analysis_calls': 0,
            'transformer_calls': 0,
            'fallback_calls': 0,
            'total_processing_time': 0
        }
        
        # Enhanced model configuration
        self.model_configs = {
            'short': {
                'model': 'google/pegasus-xsum',
                'max_length': 64,
                'min_length': 8,
                'use_case': 'concise summaries'
            },
            'medium': {
                'model': 'facebook/bart-large-cnn',
                'max_length': 142,
                'min_length': 25,
                'use_case': 'balanced summaries'
            },
            'long': {
                'model': 'google/pegasus-newsroom',
                'max_length': 256,
                'min_length': 50,
                'use_case': 'detailed summaries'
            }
        }

        # Load config from YAML if available
        self.config = {}
        self.model_name = model_name or "facebook/bart-large-cnn"
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), '../../config.yaml')
        try:
            import yaml
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.config = yaml.safe_load(f)
                self.logger.info(f"Loaded summarizer config from {config_path}")
                if 'model_name' in self.config:
                    self.model_name = self.config['model_name']
        except Exception as e:
            self.logger.warning(f"Could not load summarizer config: {e}")
        
        # Initialize domain analyzer
        self.logger.info("🧠 Initializing advanced domain analyzer...")
        # Don't load heavy models immediately, load on demand
        try:
            self.domain_analyzer.load_models()
        except Exception as e:
            self.logger.warning(f"Domain analyzer models not loaded immediately: {e}")
            # Will fallback to keyword-based analysis
    
    def load_model(self, model_name: str = None):
        """Load transformer model for summarization."""
        target_model = model_name or self.model_name
        
        if target_model in self.models:
            return True
        
        try:
            self.logger.info(f"Loading summarization model: {target_model}")
            self.tokenizers[target_model] = AutoTokenizer.from_pretrained(target_model)
            self.models[target_model] = AutoModelForSeq2SeqLM.from_pretrained(
                target_model,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            if torch.cuda.is_available():
                self.models[target_model] = self.models[target_model].to('cuda')
            
            self.logger.info(f"✅ Model {target_model} loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model {target_model}: {e}")
            return False
    
    def summarize_blocks(self, semantic_blocks: List[Dict[str, Any]], 
                        max_length: int = 150, min_length: int = 30) -> List[Dict[str, Any]]:
        """
        Generate semantic summaries for each block using only the transformer model (LLM).
        No extractive or fallback logic. No domain-based model selection.
        """
        start_time = time.time()
        summaries = []
        total_blocks = len(semantic_blocks)
        self.logger.info(f"🚀 Starting LLM-powered summarization for {total_blocks} blocks")
        if not self.load_model(self.model_name):
            raise RuntimeError(f"Failed to load model {self.model_name}")
        for i, block in enumerate(semantic_blocks):
            block_start_time = time.time()
            text = block.get('text', '').strip()
            if len(text) < 20:
                summaries.append(self._create_minimal_summary(block, i))
                self.logger.info(f"Block {i+1}/{total_blocks} skipped (too short)")
                continue
            try:
                summary_result = self._generate_transformer_summary(text, max_length, min_length)
                method = f"llm:{self.model_name}"
                self.performance_metrics['transformer_calls'] += 1
            except Exception as e:
                self.logger.error(f"LLM summary failed for block {i}: {e}")
                summary_result = self._create_error_summary(block, i, str(e))
                method = "error"
            block_time = time.time() - block_start_time
            summary_obj = {
                'block_id': block.get('block_id', i),
                'summary': summary_result['summary'],
                'method': method,
                'original_length': len(text.split()),
                'summary_length': len(summary_result['summary'].split()),
                'compression_ratio': len(summary_result['summary'].split()) / max(len(text.split()), 1),
                'processing_time': round(block_time, 2),
                'key_points': summary_result.get('key_points', []),
                'topic': summary_result.get('domain', 'general'),
                'duration': block.get('duration', 0),
                'word_count': len(text.split()),
                'quality_score': summary_result.get('quality_score', 0.7),
                'confidence': summary_result.get('confidence', 0.7)
            }
            summaries.append(summary_obj)
            self.logger.info(f"Block {i+1}/{total_blocks} completed ({method}) in {block_time:.2f}s")
        total_time = time.time() - start_time
        self.performance_metrics['total_processing_time'] += total_time
        self.logger.info(f"✅ LLM summarization completed: {len(summaries)} summaries in {total_time:.2f}s")
        return summaries
    
    # Removed domain-based model selection. Always use self.model_name.
    
    # Removed advanced summary generation with domain-aware parameters. Only use _generate_transformer_summary.
    
    # Removed domain-specific generation parameters. Use fixed parameters in _generate_transformer_summary.
    
    # Removed domain-based post-processing. Only use model output.
    
    # Removed extractive fallback logic.
    
    # Removed domain-term enhancement logic.
    
    # Use sentences from summary as key points (no domain logic).
    
    def summarize_blocks_batch(self, semantic_blocks: List[Dict[str, Any]], max_length: int = 150, min_length: int = 30, batch_size: int = None) -> List[Dict[str, Any]]:
        """
        Pipeline compatibility: batch summarization method expected by orchestrator.
        Always uses transformer model for summaries.
        """
        if not semantic_blocks:
            return []
        self.logger.info(f"🚀 Processing {len(semantic_blocks)} blocks in batch mode (batch_size={batch_size})")
        return self.summarize_blocks(semantic_blocks, max_length=max_length, min_length=min_length)
    
    # Removed LLM summary generation, always use transformer
    
    # Removed LLM key point extraction
    
    def _generate_transformer_summary(self, text: str, max_length: int, min_length: int) -> Dict[str, Any]:
        """Fallback transformer-based summary generation."""
        try:
            if not self.models or self.model_name not in self.models:
                self.logger.error(f"Model {self.model_name} not loaded, using extractive fallback.")
                raise RuntimeError(f"Model {self.model_name} not loaded.")

            model = self.models[self.model_name]
            tokenizer = self.tokenizers[self.model_name]

            # Preprocess text
            processed_text = self._preprocess_text(text)
            self.logger.info(f"Model input (preprocessed): {processed_text[:120]}...")

            # Tokenize
            inputs = tokenizer(
                processed_text,
                max_length=1024,
                truncation=True,
                return_tensors='pt'
            )

            # Move to device
            if torch.cuda.is_available() and next(model.parameters()).is_cuda:
                self.logger.info(f"Moving model input to CUDA device.")
                inputs = {k: v.cuda() for k, v in inputs.items()}

            # Generate summary
            with torch.no_grad():
                summary_ids = model.generate(
                    inputs['input_ids'],
                    max_length=max_length,
                    min_length=min_length,
                    length_penalty=2.0,
                    num_beams=4,
                    early_stopping=True
                )

            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

            self.logger.info(f"Generated summary with {self.model_name}: {summary[:80]}...")
            return {
                'summary': summary,
                'domain': 'general',
                'quality_score': 0.6,
                'key_points': [],
                'confidence': 0.5
            }

        except Exception as e:
            self.logger.error(f"Transformer summary failed ({self.model_name}): {e}")
            raise
    
    def _generate_simple_extractive_summary(self, text: str, max_length: int) -> Dict[str, Any]:
        """Simple extractive summary as ultimate fallback (backward compatibility)."""
        # Use enhanced extractive with basic domain analysis
        basic_domain_analysis = {'primary_domain': 'general', 'key_themes': []}
        return self._generate_enhanced_extractive_summary(text, max_length, basic_domain_analysis)
    
    def _generate_enhanced_extractive_summary(self, text: str, max_length: int, domain_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced extractive summary with domain awareness."""
        try:
            sentences = nltk.sent_tokenize(text) if NLTK_AVAILABLE else text.split('.')
            sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

            if not sentences:
                return {
                    'summary': "No meaningful content to summarize.",
                    'domain': domain_analysis.get('primary_domain', 'general'),
                    'quality_score': 0.2,
                    'key_points': [],
                    'confidence': 0.1
                }

            # Domain-aware sentence scoring
            domain = domain_analysis.get('primary_domain', 'general')
            themes = domain_analysis.get('key_themes', [])
            
            scored_sentences = []
            for i, sentence in enumerate(sentences):
                score = 0
                sentence_lower = sentence.lower()
                
                # Position score
                if i == 0:
                    score += 3  # First sentence often contains key info
                elif i == len(sentences) - 1:
                    score += 1  # Last sentence for conclusion
                
                # Length score (prefer medium-length sentences)
                words = len(sentence.split())
                if 10 <= words <= 25:
                    score += 2
                elif 8 <= words <= 30:
                    score += 1
                
                # Domain-specific keyword scoring
                if domain in self.domain_analyzer.domain_keywords:
                    domain_words = self.domain_analyzer.domain_keywords[domain]
                    for keyword in domain_words:
                        if keyword in sentence_lower:
                            score += 2
                
                # Theme relevance scoring
                for theme in themes:
                    if theme.lower() in sentence_lower:
                        score += 1
                
                # Avoid sentences with common filler patterns
                if any(filler in sentence_lower for filler in ['um', 'uh', 'you know', 'sort of']):
                    score -= 1
                
                scored_sentences.append((score, sentence, i))
            
            # Sort by score and select best sentences
            scored_sentences.sort(reverse=True, key=lambda x: x[0])
            
            max_sentences = min(len(sentences), max_length // 15)  # Rough estimate
            selected_sentences = []
            total_words = 0
            
            for score, sentence, idx in scored_sentences:
                sentence_words = len(sentence.split())
                if total_words + sentence_words <= max_length and len(selected_sentences) < max_sentences:
                    selected_sentences.append((idx, sentence))
                    total_words += sentence_words
            
            # Sort selected sentences by original order
            selected_sentences.sort(key=lambda x: x[0])
            final_sentences = [sentence for _, sentence in selected_sentences]
            
            summary = '. '.join(final_sentences)
            if summary and not summary.endswith('.'):
                summary += '.'

            return {
                'summary': summary,
                'domain': domain,
                'quality_score': 0.6,
                'key_points': final_sentences[:3],  # Top sentences as key points
                'confidence': 0.5
            }

        except Exception as e:
            self.logger.error(f"Enhanced extractive summary failed: {e}")
            return {
                'summary': text[:max_length] + '...' if len(text) > max_length else text,
                'domain': 'general',
                'quality_score': 0.3,
                'key_points': [],
                'confidence': 0.2
            }
    
    def generate_global_summary(self, semantic_blocks: List[Dict[str, Any]], 
                               block_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate global summary using LLM analysis.
        Maintains backward compatibility.
        """
        start_time = time.time()
        self.logger.info("🌟 Generating LLM-powered global summary")
        
        # Collect all summary texts
        summary_texts = [s.get('summary', '') for s in block_summaries if s.get('summary')]
        if not summary_texts or all(not s.strip() for s in summary_texts):
            # Always return a non-empty fallback summary
            fallback_text = "No meaningful block summaries were generated. Please review the input content."
            return {
                'executive_summary': fallback_text,
                'main_themes': [],
                'key_insights': [],
                'statistics': {
                    'total_duration_minutes': 0,
                    'total_words': 0,
                    'number_of_topics': 0,
                    'number_of_segments': 0
                },
                'processing_time': round(time.time() - start_time, 2),
                'generated_at': datetime.now().isoformat(),
                'method': 'empty_fallback'
            }
        
        combined_text = " ".join(summary_texts)
        try:
            # Always use extractive global summary
            executive_summary = self._extract_key_sentences(combined_text, max_length=200)
            method = "extractive_global"

            # Extract main themes from block domains
            all_domains = [s.get('domain', 'general') for s in block_summaries]
            domain_counts = {}
            for domain in all_domains:
                domain_counts[domain] = domain_counts.get(domain, 0) + 1

            main_themes = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)
            main_themes = [theme[0] for theme in main_themes[:5]]

            # Collect key insights
            all_key_points = []
            for summary in block_summaries:
                all_key_points.extend(summary.get('key_points', []))

            # Remove duplicates while preserving order
            unique_insights = list(dict.fromkeys(all_key_points[:10]))

            # Calculate statistics
            total_duration = sum(block.get('duration', 0) for block in semantic_blocks)
            total_words = sum(s.get('word_count', 0) for s in block_summaries)

            total_time = time.time() - start_time

            return {
                'executive_summary': executive_summary,
                'main_themes': main_themes,
                'key_insights': unique_insights[:7],
                'statistics': {
                    'total_duration_minutes': round(total_duration / 60, 1),
                    'total_words': total_words,
                    'number_of_topics': len(main_themes),
                    'number_of_segments': len(semantic_blocks)
                },
                'processing_time': round(total_time, 2),
                'generated_at': datetime.now().isoformat(),
                'method': method
            }
        except Exception as e:
            self.logger.error(f"Global summary generation failed: {e}")
            return self._create_fallback_global_summary(summary_texts, start_time)
    
    # Removed LLM global summary generation
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for better summarization."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove filler words and sounds
        filler_patterns = [
            r'\b(um|uh|er|ah|like|you know|sort of|kind of)\b',
            r'\[.*?\]',  # Remove speaker annotations
            r'\(.*?\)'   # Remove parenthetical content
        ]
        
        for pattern in filler_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Clean up punctuation
        text = re.sub(r'[.]{2,}', '.', text)
        text = re.sub(r'[,]{2,}', ',', text)
        
        return text.strip()
    
    def _extract_key_sentences(self, text: str, max_length: int = 150) -> str:
        """Extract key sentences using basic scoring."""
        try:
            sentences = nltk.sent_tokenize(text) if NLTK_AVAILABLE else text.split('.')
            sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
            
            if not sentences:
                return text[:max_length]
            
            if len(sentences) <= 2:
                result = '. '.join(sentences)
                return result[:max_length] + '...' if len(result) > max_length else result
            
            # Simple scoring: prefer first and last sentences, and longer sentences
            scored_sentences = []
            for i, sentence in enumerate(sentences):
                score = 0
                
                # Position score
                if i == 0:
                    score += 3  # First sentence
                elif i == len(sentences) - 1:
                    score += 2  # Last sentence
                
                # Length score (prefer medium length)
                words = len(sentence.split())
                if 10 <= words <= 25:
                    score += 2
                elif 8 <= words <= 30:
                    score += 1
                
                scored_sentences.append((score, sentence))
            
            # Sort by score and select
            scored_sentences.sort(reverse=True, key=lambda x: x[0])
            
            selected = []
            total_length = 0
            for score, sentence in scored_sentences:
                sentence_length = len(sentence.split())
                if total_length + sentence_length <= max_length:
                    selected.append(sentence)
                    total_length += sentence_length
                    
                    if len(selected) >= 3:  # Max 3 sentences
                        break
            
            return '. '.join(selected) + '.' if selected else sentences[0]
            
        except Exception as e:
            self.logger.warning(f"Key sentence extraction failed: {e}")
            return text[:max_length] + '...' if len(text) > max_length else text
    
    def _create_minimal_summary(self, block: Dict[str, Any], index: int) -> Dict[str, Any]:
        """Create minimal summary for very short blocks."""
        text = block.get('text', '').strip()
        return {
            'block_id': block.get('block_id', index),
            'summary': text if len(text) < 100 else text[:100] + '...',
            'method': 'minimal',
            'original_length': len(text.split()),
            'summary_length': len(text.split()),
            'compression_ratio': 1.0,
            'processing_time': 0.0,
            'key_points': [],
            'topic': 'general',
            'duration': block.get('duration', 0),
            'word_count': len(text.split()),
            'domain': 'general',
            'quality_score': 0.3,
            'confidence': 0.3
        }
    
    def _create_error_summary(self, block: Dict[str, Any], index: int, error: str) -> Dict[str, Any]:
        """Create error summary when processing fails."""
        return {
            'block_id': block.get('block_id', index),
            'summary': f"Error processing this block: {error[:100]}",
            'method': 'error',
            'original_length': len(block.get('text', '').split()),
            'summary_length': 0,
            'compression_ratio': 0.0,
            'processing_time': 0.0,
            'key_points': [],
            'topic': 'error',
            'duration': block.get('duration', 0),
            'word_count': len(block.get('text', '').split()),
            'domain': 'error',
            'quality_score': 0.0,
            'confidence': 0.0,
            'error': error
        }
    
    def _create_empty_global_summary(self) -> Dict[str, Any]:
        """Create empty global summary when no content is available."""
        return {
            'executive_summary': 'No content available for summarization.',
            'main_themes': [],
            'key_insights': [],
            'statistics': {
                'total_duration_minutes': 0,
                'total_words': 0,
                'number_of_topics': 0,
                'number_of_segments': 0
            },
            'processing_time': 0.0,
            'generated_at': datetime.now().isoformat(),
            'method': 'empty'
        }
    
    def _create_fallback_global_summary(self, summary_texts: List[str], start_time: float) -> Dict[str, Any]:
        """Create fallback global summary when advanced methods fail."""
        combined_text = " ".join(summary_texts)
        fallback_summary = self._extract_key_sentences(combined_text, max_length=200)
        
        return {
            'executive_summary': fallback_summary,
            'main_themes': [],
            'key_insights': [],
            'statistics': {
                'total_duration_minutes': 0,
                'total_words': len(combined_text.split()),
                'number_of_topics': 0,
                'number_of_segments': len(summary_texts)
            },
            'processing_time': time.time() - start_time,
            'generated_at': datetime.now().isoformat(),
            'method': 'fallback'
        }
    
    def save_summaries(self, summaries: List[Dict[str, Any]], 
                      global_summary: Dict[str, Any], output_path: Path):
        """Save summaries to JSON file."""
        output_data = {
            "global_summary": global_summary,
            "block_summaries": summaries,
            "summary_metadata": {
                "total_blocks": len(summaries),
                "summarization_methods": list(set(s.get("method", "unknown") for s in summaries)),
                "average_compression_ratio": sum(s.get("compression_ratio", 0) for s in summaries if "compression_ratio" in s) / len(summaries) if summaries else 0,
                "llm_calls": self.performance_metrics.get('llm_calls', 0),
                "fallback_calls": self.performance_metrics.get('fallback_calls', 0)
            }
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Summaries saved to {output_path}")
    
    # Additional methods for backward compatibility
    def validate_and_heal_summaries(self, summaries: List[Dict[str, Any]], 
                                   original_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and heal summaries (for backward compatibility)."""
        # The new LLM system produces high-quality summaries by default
        # Just ensure all required fields are present
        healed_summaries = []
        
        for summary in summaries:
            # Ensure backward compatibility fields exist
            if 'key_points' not in summary:
                summary['key_points'] = []
            
            if 'topic' not in summary and 'domain' in summary:
                summary['topic'] = summary['domain']
                
            if 'duration' not in summary:
                summary['duration'] = 0
                
            if 'word_count' not in summary:
                summary['word_count'] = summary.get('original_length', 0)
            
            healed_summaries.append(summary)
        
        return healed_summaries
