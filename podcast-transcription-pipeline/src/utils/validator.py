import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

class Validator:
    """Validates pipeline outputs and generates validation reports."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def validate_transcript(self, transcript_data: Dict[str, Any], 
                          min_length: int = 10) -> Dict[str, Any]:
        """Validate transcription output."""
        validation_result = {
            "step": "transcription",
            "status": "passed",
            "errors": [],
            "warnings": [],
            "metrics": {}
        }
        
        try:
            # Check required fields
            required_fields = ["language", "duration", "text", "segments"]
            for field in required_fields:
                if field not in transcript_data:
                    validation_result["errors"].append(f"Missing required field: {field}")
            
            # Check transcript length
            text_length = len(transcript_data.get("text", "").strip())
            if text_length < min_length:
                validation_result["errors"].append(f"Transcript too short: {text_length} characters")
            
            # Check segments
            segments = transcript_data.get("segments", [])
            if not segments:
                validation_result["errors"].append("No segments found in transcript")
            else:
                # Validate segment structure
                for i, segment in enumerate(segments):
                    required_seg_fields = ["start", "end", "text"]
                    for field in required_seg_fields:
                        if field not in segment:
                            validation_result["errors"].append(f"Segment {i} missing field: {field}")
                    
                    # Check timing consistency
                    if "start" in segment and "end" in segment:
                        if segment["start"] >= segment["end"]:
                            validation_result["errors"].append(f"Segment {i} has invalid timing")
            
            # Calculate metrics
            validation_result["metrics"] = {
                "total_segments": len(segments),
                "total_duration": transcript_data.get("duration", 0),
                "text_length": text_length,
                "average_segment_length": text_length / len(segments) if segments else 0,
                "segments_with_confidence": sum(1 for s in segments if "confidence" in s)
            }
            
            # Set status
            if validation_result["errors"]:
                validation_result["status"] = "failed"
            elif validation_result["warnings"]:
                validation_result["status"] = "passed_with_warnings"
                
        except Exception as e:
            validation_result["status"] = "error"
            validation_result["errors"].append(f"Validation error: {str(e)}")
        
        return validation_result
    
    def validate_emotions(self, text_emotions: List[Dict[str, Any]], 
                         audio_emotions: List[Dict[str, Any]],
                         min_confidence: float = 0.6) -> Dict[str, Any]:
        """Validate emotion detection output."""
        validation_result = {
            "step": "emotion_detection",
            "status": "passed",
            "errors": [],
            "warnings": [],
            "metrics": {}
        }
        
        try:
            # Validate text emotions
            valid_text_emotions = 0
            low_confidence_text = 0
            
            for emotion in text_emotions:
                if "emotion" in emotion and "confidence" in emotion:
                    valid_text_emotions += 1
                    if emotion["confidence"] < min_confidence:
                        low_confidence_text += 1
                else:
                    validation_result["warnings"].append("Text emotion missing required fields")
            
            # Validate audio emotions
            valid_audio_emotions = 0
            null_audio_emotions = 0
            
            for emotion in audio_emotions:
                if emotion.get("emotion") is not None:
                    valid_audio_emotions += 1
                else:
                    null_audio_emotions += 1
            
            # Check if we have any valid emotions
            if valid_text_emotions == 0:
                validation_result["errors"].append("No valid text emotions detected")
            
            if null_audio_emotions == len(audio_emotions):
                validation_result["warnings"].append("No audio emotions detected (model may be unavailable)")
            
            # Calculate metrics
            validation_result["metrics"] = {
                "total_text_emotions": len(text_emotions),
                "valid_text_emotions": valid_text_emotions,
                "low_confidence_text_emotions": low_confidence_text,
                "text_emotion_success_rate": valid_text_emotions / len(text_emotions) if text_emotions else 0,
                "total_audio_emotions": len(audio_emotions),
                "valid_audio_emotions": valid_audio_emotions,
                "null_audio_emotions": null_audio_emotions,
                "audio_emotion_success_rate": valid_audio_emotions / len(audio_emotions) if audio_emotions else 0
            }
            
            # Set status
            if validation_result["errors"]:
                validation_result["status"] = "failed"
            elif validation_result["warnings"]:
                validation_result["status"] = "passed_with_warnings"
                
        except Exception as e:
            validation_result["status"] = "error"
            validation_result["errors"].append(f"Validation error: {str(e)}")
        
        return validation_result
    
    def validate_semantic_blocks(self, semantic_blocks: List[Dict[str, Any]], 
                               max_empty_ratio: float = 0.1) -> Dict[str, Any]:
        """Validate semantic segmentation output."""
        validation_result = {
            "step": "semantic_segmentation", 
            "status": "passed",
            "errors": [],
            "warnings": [],
            "metrics": {}
        }
        
        try:
            if not semantic_blocks:
                validation_result["errors"].append("No semantic blocks found")
                validation_result["status"] = "failed"
                return validation_result
            
            # Check block structure
            empty_blocks = 0
            invalid_timing = 0
            blocks_with_topics = 0
            
            for i, block in enumerate(semantic_blocks):
                required_fields = ["block_id", "text", "start", "end"]
                for field in required_fields:
                    if field not in block:
                        validation_result["errors"].append(f"Block {i} missing field: {field}")
                
                # Check for empty blocks
                if not block.get("text", "").strip():
                    empty_blocks += 1
                
                # Check timing
                if "start" in block and "end" in block:
                    if block["start"] >= block["end"]:
                        invalid_timing += 1
                
                # Check topic information
                if block.get("topics"):
                    blocks_with_topics += 1
            
            # Check empty block ratio
            empty_ratio = empty_blocks / len(semantic_blocks)
            if empty_ratio > max_empty_ratio:
                validation_result["errors"].append(f"Too many empty blocks: {empty_ratio:.2%}")
            
            # Warnings
            if blocks_with_topics == 0:
                validation_result["warnings"].append("No topic information found in blocks")
            
            if invalid_timing > 0:
                validation_result["warnings"].append(f"{invalid_timing} blocks have invalid timing")
            
            # Calculate metrics
            total_duration = sum(block.get("duration", 0) for block in semantic_blocks)
            total_words = sum(block.get("word_count", 0) for block in semantic_blocks)
            
            validation_result["metrics"] = {
                "total_blocks": len(semantic_blocks),
                "empty_blocks": empty_blocks,
                "empty_block_ratio": empty_ratio,
                "blocks_with_topics": blocks_with_topics,
                "topic_coverage": blocks_with_topics / len(semantic_blocks),
                "total_duration": total_duration,
                "total_words": total_words,
                "average_block_duration": total_duration / len(semantic_blocks),
                "average_words_per_block": total_words / len(semantic_blocks) if semantic_blocks else 0
            }
            
            # Set status
            if validation_result["errors"]:
                validation_result["status"] = "failed"
            elif validation_result["warnings"]:
                validation_result["status"] = "passed_with_warnings"
                
        except Exception as e:
            validation_result["status"] = "error"
            validation_result["errors"].append(f"Validation error: {str(e)}")
        
        return validation_result
    
    def validate_summaries(self, summaries_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate summarization output."""
        validation_result = {
            "step": "summarization",
            "status": "passed", 
            "errors": [],
            "warnings": [],
            "metrics": {}
        }
        
        try:
            # Check global summary
            global_summary = summaries_data.get("global_summary", {})
            if not global_summary:
                validation_result["errors"].append("No global summary found")
            else:
                if not global_summary.get("global_summary", "").strip():
                    validation_result["errors"].append("Empty global summary")
            
            # Check block summaries
            block_summaries = summaries_data.get("block_summaries", [])
            if not block_summaries:
                validation_result["errors"].append("No block summaries found")
            else:
                empty_summaries = 0
                low_compression = 0
                
                for summary in block_summaries:
                    if not summary.get("summary", "").strip():
                        empty_summaries += 1
                    
                    # Check compression ratio
                    compression_ratio = summary.get("compression_ratio", 0)
                    if compression_ratio > 0.8:  # Very little compression
                        low_compression += 1
                
                if empty_summaries > 0:
                    validation_result["warnings"].append(f"{empty_summaries} empty block summaries")
                
                if low_compression > len(block_summaries) * 0.5:
                    validation_result["warnings"].append("Many summaries have low compression ratio")
            
            # Calculate metrics
            validation_result["metrics"] = {
                "has_global_summary": bool(global_summary.get("global_summary")),
                "total_block_summaries": len(block_summaries),
                "empty_block_summaries": sum(1 for s in block_summaries if not s.get("summary", "").strip()),
                "average_compression_ratio": sum(s.get("compression_ratio", 0) for s in block_summaries) / len(block_summaries) if block_summaries else 0,
                "summarization_methods": list(set(s.get("method", "unknown") for s in block_summaries))
            }
            
            # Set status
            if validation_result["errors"]:
                validation_result["status"] = "failed"
            elif validation_result["warnings"]:
                validation_result["status"] = "passed_with_warnings"
                
        except Exception as e:
            validation_result["status"] = "error"
            validation_result["errors"].append(f"Validation error: {str(e)}")
        
        return validation_result
    
    def validate_keywords(self, keywords_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate keyword extraction output."""
        validation_result = {
            "step": "keyword_extraction",
            "status": "passed",
            "errors": [],
            "warnings": [],
            "metrics": {}
        }
        
        try:
            # Check global keywords
            global_keywords = keywords_data.get("global_keywords", [])
            if not global_keywords:
                validation_result["warnings"].append("No global keywords found")
            
            # Check block keywords
            block_keywords = keywords_data.get("per_block", {})
            if not block_keywords:
                validation_result["warnings"].append("No block keywords found")
            
            # Validate keyword structure
            invalid_global_keywords = 0
            for keyword in global_keywords:
                if not isinstance(keyword, dict) or "keyword" not in keyword:
                    invalid_global_keywords += 1
            
            if invalid_global_keywords > 0:
                validation_result["warnings"].append(f"{invalid_global_keywords} invalid global keywords")
            
            # Check keyword trends
            keyword_trends = keywords_data.get("keyword_trends", {})
            
            # Calculate metrics
            total_block_keywords = sum(len(keywords) for keywords in block_keywords.values())
            
            validation_result["metrics"] = {
                "total_global_keywords": len(global_keywords),
                "valid_global_keywords": len(global_keywords) - invalid_global_keywords,
                "total_blocks_with_keywords": len(block_keywords),
                "total_block_keywords": total_block_keywords,
                "average_keywords_per_block": total_block_keywords / len(block_keywords) if block_keywords else 0,
                "has_keyword_trends": bool(keyword_trends),
                "unique_topics_detected": len(keyword_trends.get("consistent_keywords", {}))
            }
            
            # Set status based on warnings/errors
            if validation_result["errors"]:
                validation_result["status"] = "failed"
            elif validation_result["warnings"]:
                validation_result["status"] = "passed_with_warnings"
                
        except Exception as e:
            validation_result["status"] = "error"
            validation_result["errors"].append(f"Validation error: {str(e)}")
        
        return validation_result
    
    def generate_validation_report(self, session_files: Dict[str, Path],
                                 min_confidence: float = 0.6) -> Dict[str, Any]:
        """Generate comprehensive validation report for all pipeline outputs."""
        validation_report = {
            "validation_timestamp": datetime.now().isoformat(),
            "overall_status": "passed",
            "step_validations": {},
            "summary": {
                "total_steps": 0,
                "passed_steps": 0,
                "failed_steps": 0,
                "steps_with_warnings": 0
            },
            "recommendations": []
        }
        
        try:
            # Validate transcript
            if session_files["transcript"].exists():
                with open(session_files["transcript"], "r", encoding="utf-8") as f:
                    transcript_data = json.load(f)
                validation_report["step_validations"]["transcription"] = self.validate_transcript(transcript_data)
            
            # Validate emotions
            text_emotions = []
            audio_emotions = []
            
            if session_files["emotions_text"].exists():
                with open(session_files["emotions_text"], "r", encoding="utf-8") as f:
                    text_emotions = json.load(f)
            
            if session_files["emotions_audio"].exists():
                with open(session_files["emotions_audio"], "r", encoding="utf-8") as f:
                    audio_emotions = json.load(f)
            
            if text_emotions or audio_emotions:
                validation_report["step_validations"]["emotion_detection"] = self.validate_emotions(
                    text_emotions, audio_emotions, min_confidence
                )
            
            # Validate semantic blocks
            if session_files["semantic_blocks"].exists():
                with open(session_files["semantic_blocks"], "r", encoding="utf-8") as f:
                    semantic_blocks = json.load(f)
                validation_report["step_validations"]["semantic_segmentation"] = self.validate_semantic_blocks(semantic_blocks)
            
            # Validate summaries
            if session_files["summaries"].exists():
                with open(session_files["summaries"], "r", encoding="utf-8") as f:
                    summaries_data = json.load(f)
                validation_report["step_validations"]["summarization"] = self.validate_summaries(summaries_data)
            
            # Validate keywords
            if session_files["keywords_topics"].exists():
                with open(session_files["keywords_topics"], "r", encoding="utf-8") as f:
                    keywords_data = json.load(f)
                validation_report["step_validations"]["keyword_extraction"] = self.validate_keywords(keywords_data)
            
            # Calculate summary statistics
            total_steps = len(validation_report["step_validations"])
            passed_steps = sum(1 for v in validation_report["step_validations"].values() if v["status"] == "passed")
            failed_steps = sum(1 for v in validation_report["step_validations"].values() if v["status"] == "failed")
            steps_with_warnings = sum(1 for v in validation_report["step_validations"].values() if v["status"] == "passed_with_warnings")
            
            validation_report["summary"].update({
                "total_steps": total_steps,
                "passed_steps": passed_steps,
                "failed_steps": failed_steps,
                "steps_with_warnings": steps_with_warnings
            })
            
            # Determine overall status
            if failed_steps > 0:
                validation_report["overall_status"] = "failed"
            elif steps_with_warnings > 0:
                validation_report["overall_status"] = "passed_with_warnings"
            
            # Generate recommendations
            validation_report["recommendations"] = self._generate_recommendations(validation_report)
            
        except Exception as e:
            validation_report["overall_status"] = "error"
            validation_report["error"] = str(e)
            self.logger.error(f"Validation report generation failed: {e}")
        
        return validation_report
    
    def _generate_recommendations(self, validation_report: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        for step, validation in validation_report["step_validations"].items():
            if validation["status"] == "failed":
                recommendations.append(f"Critical: {step} failed validation - review and reprocess")
            elif validation["status"] == "passed_with_warnings":
                recommendations.append(f"Review {step} warnings and consider improvements")
            
            # Specific recommendations based on metrics
            metrics = validation.get("metrics", {})
            
            if step == "transcription":
                if metrics.get("total_duration", 0) < 60:
                    recommendations.append("Short audio duration - consider longer content for better analysis")
            
            elif step == "emotion_detection":
                if metrics.get("text_emotion_success_rate", 0) < 0.5:
                    recommendations.append("Low text emotion detection rate - check content quality")
                if metrics.get("audio_emotion_success_rate", 0) == 0:
                    recommendations.append("Audio emotion detection unavailable - consider installing audio models")
            
            elif step == "semantic_segmentation":
                if metrics.get("topic_coverage", 0) < 0.5:
                    recommendations.append("Low topic coverage - consider tuning segmentation parameters")
            
            elif step == "keyword_extraction":
                if metrics.get("total_global_keywords", 0) < 5:
                    recommendations.append("Few keywords extracted - content may be too general or short")
        
        return recommendations
    
    def save_validation_report(self, validation_report: Dict[str, Any], output_path: Path):
        """Save validation report to JSON file."""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(validation_report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Validation report saved to {output_path}")
        
        # Log summary
        summary = validation_report["summary"]
        self.logger.info(f"Validation Summary: {summary['passed_steps']}/{summary['total_steps']} steps passed, "
                        f"{summary['failed_steps']} failed, {summary['steps_with_warnings']} with warnings")