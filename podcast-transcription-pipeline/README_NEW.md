# 🎙️ Podcast Transcription and Analysis Pipeline

A sophisticated, real-time-capable podcast transcription and analysis pipeline that extracts rich semantic meaning from audio, diarizes speakers, detects emotions, and stores structured data in a modular, RAG-ready format.

## 🎯 Features

### Core Capabilities
- **High-Accuracy Transcription**: Whisper-based ASR with optimized parameters
- **Speaker Diarization**: Automatic speaker identification and segmentation
- **Emotion Detection**: Text and audio-based emotion analysis
- **Semantic Segmentation**: Intelligent paragraph/topic-based content blocks
- **Multi-Level Summarization**: Block-level and global content summaries
- **Keyword Extraction**: Advanced keyword and keyphrase extraction with trend analysis
- **Comprehensive Validation**: Data quality checks and processing reports

### Technical Features
- **Crash Recovery**: Resumable processing with state persistence
- **Device Optimization**: Automatic GPU/CPU selection with memory management
- **Modular Architecture**: Easy to extend and customize
- **RAG-Ready Output**: Structured data optimized for retrieval-augmented generation
- **Comprehensive Logging**: Detailed processing logs and error tracking

## 🏗️ Architecture

```
podcast-transcription-pipeline/
├── src/
│   ├── core/                    # Core pipeline components
│   │   ├── session_manager.py   # Session management & recovery
│   │   ├── pipeline_orchestrator.py  # Main orchestrator
│   │   └── audio_processor.py   # Audio preprocessing
│   ├── transcription/           # Transcription components
│   │   ├── whisper_transcriber.py  # Whisper integration
│   │   ├── diarization.py       # Speaker diarization
│   │   └── transcript_formatter.py  # Output formatting
│   ├── analysis/                # Analysis components
│   │   ├── emotion_detector.py  # Emotion detection
│   │   ├── semantic_segmenter.py  # Content segmentation
│   │   ├── topic_classifier.py  # Topic classification
│   │   ├── summarizer.py        # Content summarization
│   │   └── keyword_extractor.py # Keyword extraction
│   ├── utils/                   # Utilities
│   │   ├── device_manager.py    # Device optimization
│   │   ├── validator.py         # Data validation
│   │   ├── logger.py           # Logging utilities
│   │   └── file_handler.py     # File operations
│   └── config/                  # Configuration
│       ├── settings.py         # Settings management
│       └── model_config.py     # Model configurations
├── output/                     # Output directory
│   └── sessions/              # Session-based outputs
├── tests/                     # Test suite
├── config.yaml               # Main configuration
├── requirements.txt          # Dependencies
└── main.py                  # Entry point
```

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Process your first podcast
python main.py sample_podcast.mp3

# 3. Check results
ls output/sessions/session_*/
```

## 📊 Output Structure

For each processed audio file, the pipeline creates a session directory:

```
output/sessions/session_20250712_143022_abc12345/
├── transcript.json           # Transcription with speaker info
├── emotions_text.json        # Text-based emotion analysis
├── emotions_audio.json       # Audio-based emotion analysis
├── semantic_blocks.json      # Semantic segmentation & topics
├── summaries.json           # Block & global summaries
├── keywords_topics.json     # Keywords & trend analysis
├── validation_report.json   # Processing validation
├── processing_log.txt       # Detailed processing log
└── session_info.json       # Session metadata
```

### Sample Output Files

#### transcript.json
```json
[
  {
    "id": 0,
    "speaker": "Speaker 1",
    "start": 12.4,
    "end": 16.7,
    "text": "Welcome to the show...",
    "confidence": 0.89,
    "words": [...]
  }
]
```

#### semantic_blocks.json
```json
[
  {
    "block_id": 1,
    "text": "...",
    "start": 45.3,
    "end": 91.6,
    "topics": ["AI Ethics", "Bias"],
    "topic_probabilities": {
      "AI Ethics": 0.72,
      "Bias": 0.58
    },
    "speaker": "Speaker 2"
  }
]
```

## ⚙️ Configuration

### config.yaml Structure
```yaml
transcription:
  model: "large-v3"
  language: "auto"
  temperature: 0.0

diarization:
  model: "pyannote/speaker-diarization-3.1"
  min_speakers: 1
  max_speakers: 10

emotion:
  text_model: "j-hartmann/emotion-english-distilroberta-base"
  confidence_threshold: 0.5

# ... additional configuration
```

## 📋 Requirements

### System Requirements
- Python 3.8+
- 8GB+ RAM (16GB+ recommended)
- 10GB+ free disk space
- GPU with 4GB+ VRAM (optional but recommended)

### Key Dependencies
- PyTorch 2.0+
- OpenAI Whisper
- Transformers (HuggingFace)
- pyannote.audio
- scikit-learn
- sentence-transformers
- librosa

## 🎵 Usage

### Basic Usage
```bash
# Process a single audio file
python main.py path/to/your/podcast.mp3

# With custom configuration
python main.py path/to/your/podcast.mp3 --config custom_config.yaml

# Resume a specific session
python main.py path/to/your/podcast.mp3 --session-id abc12345
```

### Session Management
```bash
# List all sessions
python main.py --list-sessions

# Check session status
python main.py --status session_20250712_143022_abc12345
```

### Programmatic Usage
```python
from src.core.pipeline_orchestrator import PipelineOrchestrator

# Initialize pipeline
orchestrator = PipelineOrchestrator("config.yaml")

# Process audio
result = orchestrator.process_audio("path/to/audio.mp3")

# Access results
session_dir = result["session_info"]["session_dir"]
validation = result["validation_report"]
```

## 🔧 Advanced Features

### Custom Model Integration
```python
# Use custom emotion detection model
config = {
    "emotion": {
        "text_model": "your-custom/emotion-model",
        "confidence_threshold": 0.7
    }
}
orchestrator = PipelineOrchestrator(config)
```

### Integration with RAG Systems
```python
# Load processed data for RAG
import json
from pathlib import Path

session_dir = Path("output/sessions/session_20250712_143022_abc12345")

# Load semantic blocks for vector database
with open(session_dir / "semantic_blocks.json") as f:
    blocks = json.load(f)
    
# Each block is ready for embedding and storage
for block in blocks:
    text = block["text"]
    metadata = {
        "start": block["start"],
        "end": block["end"],
        "topics": block["topics"],
        "speaker": block["speaker"]
    }
    # Add to your vector database
```

## 📈 Performance Optimization

### GPU Acceleration
- Automatic CUDA detection and utilization
- Memory-optimized batch processing
- Mixed precision support for compatible models

### Processing Tips
1. **Audio Quality**: Higher quality audio = better results
2. **File Length**: 5-60 minute files work best
3. **Speaker Count**: 1-5 speakers optimal for diarization
4. **Hardware**: GPU with 6GB+ VRAM recommended

## 🔍 Troubleshooting

### Common Issues

**Out of Memory**
```yaml
# Reduce batch size in config
processing:
  batch_size: 2  # Reduce from default 8
```

**Model Download Failures**
```bash
# Clear cache and retry
rm -rf ~/.cache/huggingface/
python main.py your_audio.mp3
```

**Poor Diarization Results**
```yaml
# Try adjusting speaker count
diarization:
  min_speakers: 2
  max_speakers: 5
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test category
pytest tests/unit/
pytest tests/integration/
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for Whisper ASR
- Pyannote team for speaker diarization
- HuggingFace for transformer models
- The open-source AI community

---

**Ready to transform your podcasts into structured, searchable, and analyzable data!** 🎉
