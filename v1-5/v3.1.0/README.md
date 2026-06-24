# Version 3.1.0 Release Notes

## Major New Features
### Speech-to-Text Integration
- New Speech Processing Module: Added speech_processor.py with Faster-Whisper for real-time audio transcription
- Voice Recording Client: Enhanced Jetson client with VoiceRecorder class that auto-detects USB microphones
- Speech API Endpoint: New /speech endpoint for audio processing and transcription
- Debug Tools: Added debug_speech.py for testing and troubleshooting speech processing

### Retrieval-Augmented Generation (RAG)
- Vector Database: New db.py module with FAISS for semantic search and MongoDB for conversation logging
- Context-Aware Responses: Chat now uses past conversation context for more relevant responses
- Embedding Storage: Automatic storage and retrieval of conversation embeddings

### Topic Analysis & Visualisation
- Topic Extraction: New topics_controller.py for automatic topic discovery from conversation logs
- Word Cloud Generation: Visual representation of conversation topics
- REST API: /topics endpoint for retrieving analyzed topics and visualisations

## Improvements
### Server
- Async Event Loop: Better handling of concurrent requests
- Enhanced Health Checks: Added speech processor status monitoring
- Improved Error Handling: More detailed error messages for speech processing
- Increased Buffer Size: Support for larger audio files (2MB)

## Client
- Auto USB Mic Detection: Voice recorder automatically finds and configures USB microphones
- Enhanced Chat Interface: Voice and text input in unified interface
- Better Status Indicators: Visual feedback for recording and processing states
