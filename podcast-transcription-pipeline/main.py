#!/usr/bin/env python3
"""
Podcast Transcription and Analysis Pipeline
Main entry point for running the complete pipeline
"""

import argparse
import sys
import logging
from pathlib import Path

# Setup logging first
from src.utils.logger import setup_logging, get_logger

# Initialize logging
setup_logging()
logger = get_logger(__name__)

from src.core.pipeline_orchestrator import PipelineOrchestrator

def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(
        description="Process podcast audio through transcription and analysis pipeline"
    )
    
    parser.add_argument(
        "audio_file",
        help="Path to the audio file to process"
    )
    
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
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
    
    # Initialize orchestrator
    try:
        orchestrator = PipelineOrchestrator(config_path=args.config)
        logger.info("Pipeline orchestrator initialized successfully")
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
    
    logger.info(f"Starting to process audio file: {audio_path}")
    print(f"Processing audio file: {audio_path}")
    print("This may take several minutes depending on file length and system capabilities...")
    
    try:
        result = orchestrator.process_audio(str(audio_path), args.session_id)
        
        logger.info("Processing completed successfully")
        print(f"\nProcessing completed successfully!")
        print(f"Session: {result['session_info']['session_name']}")
        print(f"Output directory: {result['session_info']['session_dir']}")
        
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
        logger.error(f"Processing failed: {e}")
        print(f"Error during processing: {e}")
        sys.exit(1)
        print("\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Processing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
