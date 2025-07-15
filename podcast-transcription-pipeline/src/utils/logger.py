import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

class PipelineLogger:
    """Centralized logging for the podcast pipeline."""
    
    def __init__(self, log_dir: str = "logs", log_level: str = "INFO"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.log_level = getattr(logging, log_level.upper())
        
        # Configure logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Set up logging configuration."""
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)
        
        # Clear existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.log_level)
        console_handler.setFormatter(simple_formatter)
        root_logger.addHandler(console_handler)
        
        # File handler for detailed logs
        log_file = self.log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(file_handler)
        
        # Error file handler
        error_log_file = self.log_dir / f"errors_{datetime.now().strftime('%Y%m%d')}.log"
        error_handler = logging.FileHandler(error_log_file, encoding='utf-8')
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(error_handler)
    
    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """Get a logger instance."""
        return logging.getLogger(name)
    
    def set_session_context(self, session_id: str, session_dir: Path):
        """Add session-specific logging context."""
        session_log_file = session_dir / "session.log"
        
        # Create session-specific handler
        session_handler = logging.FileHandler(session_log_file, encoding='utf-8')
        session_handler.setLevel(logging.DEBUG)
        
        session_formatter = logging.Formatter(
            f'%(asctime)s - [{session_id}] - %(name)s - %(levelname)s - %(message)s'
        )
        session_handler.setFormatter(session_formatter)
        
        # Add to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(session_handler)
        
        return session_handler
    
    def cleanup_old_logs(self, days_to_keep: int = 30):
        """Clean up old log files."""
        cutoff_date = datetime.now().timestamp() - (days_to_keep * 24 * 60 * 60)
        
        for log_file in self.log_dir.glob("*.log"):
            if log_file.stat().st_mtime < cutoff_date:
                try:
                    log_file.unlink()
                    logging.info(f"Deleted old log file: {log_file}")
                except Exception as e:
                    logging.warning(f"Failed to delete log file {log_file}: {e}")

# Global logger instance
_pipeline_logger = None

def setup_logging(log_dir: str = "logs", log_level: str = "INFO") -> PipelineLogger:
    """Set up global logging configuration."""
    global _pipeline_logger
    
    # Override with environment variables if set
    log_level = os.getenv("PODCAST_PIPELINE_LOG_LEVEL", log_level)
    log_dir = os.getenv("PODCAST_PIPELINE_LOG_DIR", log_dir)
    
    _pipeline_logger = PipelineLogger(log_dir, log_level)
    return _pipeline_logger

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    if _pipeline_logger is None:
        setup_logging()
    return PipelineLogger.get_logger(name)

# Performance logging decorator
def log_performance(func):
    """Decorator to log function performance."""
    import functools
    import time
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            duration = end_time - start_time
            
            logger.info(f"{func.__name__} completed in {duration:.2f} seconds")
            return result
            
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            
            logger.error(f"{func.__name__} failed after {duration:.2f} seconds: {e}")
            raise
    
    return wrapper

# Context manager for step logging
class LogStep:
    """Context manager for logging pipeline steps."""
    
    def __init__(self, step_name: str, logger: Optional[logging.Logger] = None):
        self.step_name = step_name
        self.logger = logger or get_logger(__name__)
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(f"Starting step: {self.step_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        if exc_type is None:
            self.logger.info(f"Completed step: {self.step_name} in {duration:.2f} seconds")
        else:
            self.logger.error(f"Failed step: {self.step_name} after {duration:.2f} seconds - {exc_val}")
        
        return False  # Don't suppress exceptions