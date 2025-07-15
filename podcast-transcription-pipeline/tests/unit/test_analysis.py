import unittest
from unittest.mock import Mock, patch, MagicMock
import json
from pathlib import Path
import tempfile

from src.analysis.emotion_detector import EmotionDetector
from src.analysis.semantic_segmenter import SemanticSegmenter
# Summarizer import removed
from src.analysis.keyword_extractor import KeywordExtractor

class TestAnalysisComponents(unittest.TestCase):
    """Test cases for analysis components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Sample transcript segments for testing
        self.sample_segments = [
            {
                "id": 0,
                "speaker": "Speaker 1",
                "start": 0.0,
                "end": 5.0,
                "text": "Hello everyone, welcome to our podcast about artificial intelligence.",
                "confidence": 0.9
            },
            {
                "id": 1,
                "speaker": "Speaker 2", 
                "start": 5.0,
                "end": 12.0,
                "text": "Thanks for having me. I'm excited to discuss the future of AI ethics.",
                "confidence": 0.85
            },
            {
                "id": 2,
                "speaker": "Speaker 1",
                "start": 12.0,
                "end": 20.0,
                "text": "Let's start with bias in machine learning models. This is a critical issue.",
                "confidence": 0.88
            }
        ]
        
        # Sample semantic blocks
        self.sample_blocks = [
            {
                "block_id": 1,
                "text": "Hello everyone, welcome to our podcast about artificial intelligence. Thanks for having me. I'm excited to discuss the future of AI ethics.",
                "start": 0.0,
                "end": 12.0,
                "speaker": "Speaker 1",
                "word_count": 20,
                "topics": ["AI", "Ethics"],
                "topic_probabilities": {"AI": 0.8, "Ethics": 0.7}
            },
            {
                "block_id": 2,
                "text": "Let's start with bias in machine learning models. This is a critical issue that affects fairness in AI systems.",
                "start": 12.0,
                "end": 25.0,
                "speaker": "Speaker 1",
                "word_count": 18,
                "topics": ["Bias", "Machine Learning"],
                "topic_probabilities": {"Bias": 0.9, "Machine Learning": 0.8}
            }
        ]
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('src.analysis.emotion_detector.pipeline')
    def test_emotion_detector_text_emotions(self, mock_pipeline):
        """Test text emotion detection."""
        # Mock the pipeline
        mock_emotion_pipeline = MagicMock()
        mock_emotion_pipeline.return_value = [[
            {"label": "joy", "score": 0.8},
            {"label": "neutral", "score": 0.2}
        ]]
        mock_pipeline.return_value = mock_emotion_pipeline
        
        detector = EmotionDetector()
        detector.text_pipeline = mock_emotion_pipeline
        
        emotions = detector.detect_text_emotions(self.sample_segments)
        
        self.assertGreater(len(emotions), 0)
        self.assertIn("emotion", emotions[0])
        self.assertIn("confidence", emotions[0])
        self.assertEqual(emotions[0]["emotion"], "joy")
    
    def test_semantic_segmenter_duration_blocks(self):
        """Test semantic segmentation into duration-based blocks."""
        segmenter = SemanticSegmenter()
        
        # Test with mock embedding model
        with patch.object(segmenter, 'embedding_model', None):
            blocks = segmenter.segment_into_blocks(self.sample_segments, min_block_duration=10.0)
        
        self.assertGreater(len(blocks), 0)
        self.assertIn("block_id", blocks[0])
        self.assertIn("text", blocks[0])
        self.assertIn("start", blocks[0])
        self.assertIn("end", blocks[0])
    
    # Summarizer test removed
    
    def test_keyword_extractor_frequency_method(self):
        """Test keyword extraction using frequency method."""
        extractor = KeywordExtractor(method="frequency")
        
        # Test global keywords
        global_keywords = extractor.extract_global_keywords(self.sample_blocks, max_keywords=5)
        
        self.assertGreater(len(global_keywords), 0)
        self.assertIn("keyword", global_keywords[0])
        self.assertIn("confidence", global_keywords[0])
        
        # Test block keywords
        block_keywords = extractor.extract_block_keywords(self.sample_blocks, max_keywords_per_block=5)
        
        self.assertIn("block_1", block_keywords)
        self.assertIsInstance(block_keywords["block_1"], list)
    
    @patch('src.analysis.keyword_extractor.TfidfVectorizer')
    def test_keyword_extractor_tfidf_method(self, mock_vectorizer):
        """Test keyword extraction using TF-IDF method."""
        # Mock TF-IDF vectorizer
        mock_vec_instance = MagicMock()
        mock_vec_instance.fit_transform.return_value.toarray.return_value = [[0.8, 0.6, 0.4]]
        mock_vec_instance.get_feature_names_out.return_value = ["artificial", "intelligence", "ethics"]
        mock_vectorizer.return_value = mock_vec_instance
        
        extractor = KeywordExtractor(method="tfidf")
        
        global_keywords = extractor.extract_global_keywords(self.sample_blocks, max_keywords=3)
        
        self.assertGreater(len(global_keywords), 0)
        self.assertEqual(global_keywords[0]["method"], "tfidf")
    
    def test_keyword_trend_analysis(self):
        """Test keyword trend analysis."""
        extractor = KeywordExtractor()
        
        # Mock block keywords
        block_keywords = {
            "block_1": ["AI", "ethics", "future"],
            "block_2": ["AI", "bias", "machine learning"],
            "block_3": ["AI", "fairness", "algorithms"]
        }
        
        trends = extractor.analyze_keyword_trends(block_keywords, self.sample_blocks)
        
        self.assertIn("consistent_keywords", trends)
        self.assertIn("trending_keywords", trends)
        self.assertIn("total_unique_keywords", trends)
        
        # AI should be consistent (appears in all blocks)
        if "AI" in trends["consistent_keywords"]:
            self.assertGreater(trends["consistent_keywords"]["AI"]["frequency"], 1)

if __name__ == "__main__":
    unittest.main()