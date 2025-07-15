#!/usr/bin/env python3
"""
Fast Pipeline Processing Script
Optimized for maximum speed with good quality retention
Includes full argument parsing and session management
"""

import argparse
import sys
import logging
import time
from pathlib import Path
from src.core.pipeline_orchestrator import PipelineOrchestrator

# Configure logging for speed monitoring
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('speed_processing.log')
    ]
)

def main():
    """Main entry point for the fast pipeline."""
    parser = argparse.ArgumentParser(
        description="Process podcast audio through FAST transcription and analysis pipeline"
    )
    
    parser.add_argument(
        "--audio_file",
        nargs='?',  # Make it optional for backward compatibility
        help="Path to the audio file to process"
    )
    
    parser.add_argument(
        "--config",
        default="config_speed.yaml",  # Default to speed config
        help="Path to configuration file (default: config_speed.yaml)"
    )
    
    parser.add_argument(
        "--session-id",
        help="Optional session ID for resuming or custom naming"
    )
    
    parser.add_argument(
        "--list-sessions",
        action="store_true",
        help="List all available sessions"
    )
    
    parser.add_argument(
        "--status",
        help="Get status of a specific session"
    )
    
    args = parser.parse_args()
    
    # If no arguments provided, default to audio.wav for backward compatibility
    if not any(vars(args).values()) or (args.audio_file is None and not args.list_sessions and not args.status):
        args.audio_file = "audio.wav"
        args.config = "config_speed.yaml"
    
    logger = logging.getLogger(__name__)
    
    # Initialize orchestrator
    try:
        orchestrator = PipelineOrchestrator(config_path=args.config)
        logger.info("Fast pipeline orchestrator initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        print(f"Failed to initialize pipeline: {e}")
        sys.exit(1)
    
    # Handle different operations
    if args.list_sessions:
        sessions = orchestrator.list_sessions()
        print("\nAvailable Sessions:")
        print("-" * 50)
        for session in sessions:
            status = session.get("status", "unknown")
            completed = len(session.get("completed_steps", []))
            failed = len(session.get("failed_steps", []))
            print(f"{session['session_name']}: {status} ({completed} completed, {failed} failed)")
        return
    
    if args.status:
        status = orchestrator.get_pipeline_status(args.status)
        if "error" in status:
            print(f"Error: {status['error']}")
        else:
            print(f"\nSession: {status['session_name']}")
            print(f"Status: {status['status']}")
            print(f"Completed steps: {', '.join(status['completed_steps'])}")
            if status['failed_steps']:
                print(f"Failed steps: {', '.join(status['failed_steps'])}")
        return
    
    # Process audio file
    if not args.audio_file:
        logger.error("Audio file is required")
        print("Error: Audio file is required")
        parser.print_help()
        sys.exit(1)
    
    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        logger.error(f"Audio file not found: {audio_path}")
        print(f"Error: Audio file not found: {audio_path}")
        sys.exit(1)
    
    run_fast_pipeline(orchestrator, audio_path, args.session_id, logger)

def run_fast_pipeline(orchestrator, audio_path, session_id=None, logger=None):
    """Run optimized pipeline for maximum speed."""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("=" * 50)
    logger.info("STARTING FAST PIPELINE PROCESSING")
    logger.info("=" * 50)
    logger.info(f"Processing audio file: {audio_path}")
    print(f"Processing audio file: {audio_path}")
    print("Fast processing mode - optimized for speed with good quality retention...")
    
    start_time = time.time()
    
    try:
        # Process with timing
        result = orchestrator.process_audio(str(audio_path), session_id)
        
        total_time = time.time() - start_time
        
        # Performance summary
        logger.info("=" * 50)
        logger.info("FAST PROCESSING COMPLETED!")
        logger.info("=" * 50)
        logger.info(f"Total processing time: {total_time:.2f} seconds")
        
        print(f"\nFast processing completed successfully!")
        print(f"Session: {result['session_info']['session_name']}")
        print(f"Output directory: {result['session_info']['session_dir']}")
        
        if 'performance_metrics' in result:
            metrics = result['performance_metrics']
            print(f"\nPerformance Summary:")
            for step, timing in metrics.get('step_timings', {}).items():
                minutes = int(timing // 60)
                seconds = timing % 60
                if minutes > 0:
                    logger.info(f"{step}: {minutes}m {seconds:.1f}s")
                    print(f"  {step}: {minutes}m {seconds:.1f}s")
                else:
                    logger.info(f"{step}: {seconds:.2f}s")
                    print(f"  {step}: {seconds:.2f}s")
        
        # Speed analysis
        audio_duration = get_audio_duration(audio_path)
        if audio_duration:
            speed_ratio = audio_duration / total_time
            logger.info(f"Processing speed: {speed_ratio:.1f}x real-time")
            print(f"Processing speed: {speed_ratio:.1f}x real-time")
            
            if speed_ratio > 1.0:
                logger.info("✅ Processing faster than real-time!")
                print("✅ Processing faster than real-time!")
            else:
                logger.info("⚠️ Processing slower than real-time")
                print("⚠️ Processing slower than real-time")
        
        # Show validation summary
        validation = result.get("validation_report", {})
        if validation:
            summary = validation.get("summary", {})
            print(f"Validation: {summary.get('passed_steps', 0)}/{summary.get('total_steps', 0)} steps passed")
            
            if validation.get("recommendations"):
                print("\nRecommendations:")
                for rec in validation["recommendations"][:3]:  # Show first 3
                    print(f"  - {rec}")
        
    except KeyboardInterrupt:
        logger.warning("Processing interrupted by user")
        print("\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fast processing failed: {e}")
        print(f"Error during processing: {e}")
        sys.exit(1)

def get_audio_duration(audio_path: Path) -> float:
    """Get audio duration for speed analysis."""
    try:
        import librosa
        duration = librosa.get_duration(filename=str(audio_path))
        return duration
    except:
        return None

if __name__ == "__main__":
    main()
