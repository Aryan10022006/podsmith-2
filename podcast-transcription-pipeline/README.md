# 🎙️ Podcast Transcription and Analysis Pipeline

A comprehensive, real-time-capable pipeline for podcast transcription and semantic analysis that extracts rich meaning from audio content. This modular system processes podcast audio through transcription, speaker diarization, emotion detection, and semantic analysis to produce structured, RAG-ready outputs.

## 🚀 Key Features

### Core Processing
- **High-Accuracy Transcription**: OpenAI Whisper (large-v3) with optimized settings
- **Speaker Diarization**: Automated speaker identification and separation using pyannote.audio
- **Real-time Processing**: Streaming capabilities for live audio analysis
- **Crash Recovery**: Robust session management with automatic resume functionality
- **GPU Optimization**: Automatic GPU detection and memory optimization

### Advanced Analysis
- **Emotion Detection**: Text and audio-based emotion analysis
- **Semantic Segmentation**: Topic-based content segmentation
- **Smart Summarization**: Hierarchical summarization with key insights
- **Keyword Extraction**: Multi-method keyword and phrase extraction using TF-IDF and YAKE
- **Topic Classification**: Automated content categorization

### Output & Integration
- **RAG-Ready Format**: Structured JSON output optimized for retrieval systems
- **Multiple Formats**: JSON, YAML, and structured text outputs
- **Comprehensive Metadata**: Timestamps, confidence scores, and quality metrics
- **Validation Reports**: Quality assessment and improvement recommendations

## 📁 Project Architecture

## Project Structure

```
podcast-transcription-pipeline
├── src
│   ├── core
│   │   ├── session_manager.py
│   │   ├── audio_processor.py
│   │   └── pipeline_orchestrator.py
│   ├── transcription
│   │   ├── whisper_transcriber.py
│   │   ├── diarization.py
│   │   └── transcript_formatter.py
│   ├── analysis
│   │   ├── emotion_detector.py
│   │   ├── semantic_segmenter.py
│   │   ├── topic_classifier.py
│   │   ├── summarizer.py
│   │   └── keyword_extractor.py
│   ├── utils
│   │   ├── logger.py
│   │   ├── validator.py
│   │   ├── device_manager.py
│   │   └── file_handler.py
│   ├── models
│   │   ├── emotion_models.py
│   │   ├── topic_models.py
│   │   └── summarization_models.py
│   └── config
│       ├── settings.py
│       └── model_config.py
├── tests
│   ├── unit
│   │   ├── test_transcription.py
│   │   ├── test_analysis.py
│   │   └── test_session_manager.py
│   ├── integration
│   │   └── test_pipeline.py
│   └── fixtures
│       └── sample_audio.wav
├── output
│   └── sessions
├── requirements.txt
├── setup.py
├── config.yaml
└── README.md
```

## Features

- **Session Management**: Handles the lifecycle of audio processing sessions, including folder setup and resuming from partial states.
- **Audio Processing**: Supports various audio formats and optimizes performance for transcription and analysis.
- **Transcription**: Utilizes advanced ASR models for high-accuracy audio-to-text conversion.
- **Diarization**: Identifies and labels different speakers in the audio.
- **Emotion Detection**: Analyzes emotional tones in the transcribed text and audio.
- **Semantic Analysis**: Segments transcripts into meaningful paragraphs and classifies them into topics.
- **Summarization**: Generates summaries for each segment and a global summary.
- **Keyword Extraction**: Identifies high-signal keywords and keyphrases from the transcript.

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd podcast-transcription-pipeline
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Configure the settings in `config.yaml` as needed.

## Usage

To run the pipeline, execute the main orchestrator script located in `src/core/pipeline_orchestrator.py`. Ensure that the audio files are placed in the appropriate input directory and that the output directory is correctly set in the configuration.

## Testing

Unit and integration tests are provided in the `tests` directory. To run the tests, use:
```
pytest tests/
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.