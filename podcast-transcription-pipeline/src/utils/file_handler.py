import os
import shutil
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
import json
import yaml
import pickle
from datetime import datetime
import mimetypes
import hashlib
import logging

logger = logging.getLogger(__name__)

class FileHandler:
    """Utility class for file operations in the podcast pipeline."""
    
    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.supported_audio_formats = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac'}
        self.supported_video_formats = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    
    def ensure_directory(self, directory: Union[str, Path]) -> Path:
        """Ensure a directory exists, create if it doesn't."""
        directory = Path(directory)
        
        if not directory.is_absolute():
            directory = self.base_dir / directory
        
        directory.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {directory}")
        return directory
    
    def is_audio_file(self, file_path: Union[str, Path]) -> bool:
        """Check if a file is a supported audio format."""
        file_path = Path(file_path)
        return file_path.suffix.lower() in self.supported_audio_formats
    
    def is_video_file(self, file_path: Union[str, Path]) -> bool:
        """Check if a file is a supported video format."""
        file_path = Path(file_path)
        return file_path.suffix.lower() in self.supported_video_formats
    
    def is_media_file(self, file_path: Union[str, Path]) -> bool:
        """Check if a file is a supported media format."""
        return self.is_audio_file(file_path) or self.is_video_file(file_path)
    
    def get_file_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Get detailed information about a file."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        stat = file_path.stat()
        
        return {
            'path': str(file_path),
            'name': file_path.name,
            'stem': file_path.stem,
            'suffix': file_path.suffix,
            'size_bytes': stat.st_size,
            'size_mb': round(stat.st_size / (1024 * 1024), 2),
            'created': datetime.fromtimestamp(stat.st_ctime),
            'modified': datetime.fromtimestamp(stat.st_mtime),
            'is_audio': self.is_audio_file(file_path),
            'is_video': self.is_video_file(file_path),
            'mime_type': mimetypes.guess_type(str(file_path))[0],
            'md5_hash': self.get_file_hash(file_path)
        }
    
    def get_file_hash(self, file_path: Union[str, Path], chunk_size: int = 8192) -> str:
        """Calculate MD5 hash of a file."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        hash_md5 = hashlib.md5()
        
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                hash_md5.update(chunk)
        
        return hash_md5.hexdigest()
    
    def find_media_files(self, directory: Union[str, Path], recursive: bool = True) -> List[Path]:
        """Find all media files in a directory."""
        directory = Path(directory)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        pattern = "**/*" if recursive else "*"
        media_files = []
        
        for file_path in directory.glob(pattern):
            if file_path.is_file() and self.is_media_file(file_path):
                media_files.append(file_path)
        
        logger.info(f"Found {len(media_files)} media files in {directory}")
        return sorted(media_files)
    
    def safe_filename(self, filename: str, max_length: int = 200) -> str:
        """Create a safe filename by removing/replacing problematic characters."""
        # Replace problematic characters
        safe_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_. "
        safe_name = ''.join(c if c in safe_chars else '_' for c in filename)
        
        # Remove multiple consecutive underscores/spaces
        import re
        safe_name = re.sub(r'[_\s]+', '_', safe_name)
        
        # Trim and limit length
        safe_name = safe_name.strip('_. ')[:max_length]
        
        return safe_name
    
    def backup_file(self, file_path: Union[str, Path], backup_dir: Optional[Union[str, Path]] = None) -> Path:
        """Create a backup of a file."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if backup_dir is None:
            backup_dir = file_path.parent / "backups"
        else:
            backup_dir = Path(backup_dir)
        
        self.ensure_directory(backup_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
        backup_path = backup_dir / backup_name
        
        shutil.copy2(file_path, backup_path)
        logger.info(f"Created backup: {backup_path}")
        
        return backup_path
    
    def save_json(self, data: Any, file_path: Union[str, Path], indent: int = 2) -> Path:
        """Save data as JSON file."""
        file_path = Path(file_path)
        self.ensure_directory(file_path.parent)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False, default=str)
        
        logger.debug(f"Saved JSON data to: {file_path}")
        return file_path
    
    def load_json(self, file_path: Union[str, Path]) -> Any:
        """Load data from JSON file."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"JSON file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.debug(f"Loaded JSON data from: {file_path}")
        return data
    
    def save_yaml(self, data: Any, file_path: Union[str, Path]) -> Path:
        """Save data as YAML file."""
        file_path = Path(file_path)
        self.ensure_directory(file_path.parent)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        
        logger.debug(f"Saved YAML data to: {file_path}")
        return file_path
    
    def load_yaml(self, file_path: Union[str, Path]) -> Any:
        """Load data from YAML file."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"YAML file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        logger.debug(f"Loaded YAML data from: {file_path}")
        return data
    
    def save_pickle(self, data: Any, file_path: Union[str, Path]) -> Path:
        """Save data as pickle file."""
        file_path = Path(file_path)
        self.ensure_directory(file_path.parent)
        
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.debug(f"Saved pickle data to: {file_path}")
        return file_path
    
    def load_pickle(self, file_path: Union[str, Path]) -> Any:
        """Load data from pickle file."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Pickle file not found: {file_path}")
        
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        logger.debug(f"Loaded pickle data from: {file_path}")
        return data
    
    def cleanup_empty_directories(self, directory: Union[str, Path]) -> int:
        """Remove empty directories recursively."""
        directory = Path(directory)
        removed_count = 0
        
        for path in sorted(directory.rglob("*"), reverse=True):
            if path.is_dir() and not any(path.iterdir()):
                try:
                    path.rmdir()
                    removed_count += 1
                    logger.debug(f"Removed empty directory: {path}")
                except OSError:
                    pass  # Directory not empty or permission denied
        
        logger.info(f"Removed {removed_count} empty directories")
        return removed_count
    
    def get_directory_size(self, directory: Union[str, Path]) -> Dict[str, Any]:
        """Calculate total size of a directory."""
        directory = Path(directory)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        total_size = 0
        file_count = 0
        dir_count = 0
        
        for path in directory.rglob("*"):
            if path.is_file():
                total_size += path.stat().st_size
                file_count += 1
            elif path.is_dir():
                dir_count += 1
        
        return {
            'total_size_bytes': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'total_size_gb': round(total_size / (1024 * 1024 * 1024), 2),
            'file_count': file_count,
            'directory_count': dir_count
        }
    
    def move_file(self, source: Union[str, Path], destination: Union[str, Path]) -> Path:
        """Move a file to a new location."""
        source = Path(source)
        destination = Path(destination)
        
        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {source}")
        
        self.ensure_directory(destination.parent)
        
        # Handle case where destination is a directory
        if destination.is_dir():
            destination = destination / source.name
        
        shutil.move(str(source), str(destination))
        logger.info(f"Moved file from {source} to {destination}")
        
        return destination
    
    def copy_file(self, source: Union[str, Path], destination: Union[str, Path]) -> Path:
        """Copy a file to a new location."""
        source = Path(source)
        destination = Path(destination)
        
        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {source}")
        
        self.ensure_directory(destination.parent)
        
        # Handle case where destination is a directory
        if destination.is_dir():
            destination = destination / source.name
        
        shutil.copy2(str(source), str(destination))
        logger.info(f"Copied file from {source} to {destination}")
        
        return destination