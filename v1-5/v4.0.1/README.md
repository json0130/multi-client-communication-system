# Release Notes v4.0.1: Major Performance & RAG Update
## Performance & Architecture
- Monitor Streaming is 300% Faster: Achieved by reducing frame processing to 15 FPS, implementing smart caching (60% reduction in re-encoding), and adding broadcast throttling (max 5 updates/sec).
- Reduced CPU Load: Emotion processing is 50% less CPU intensive by increasing the interval to 200ms.
- Enhanced Client Monitoring: Introduced high-quality, configurable monitor streams (up to 1280×720,85% quality), adaptive frame rate adjustment, and WebSocket rooms for isolated monitoring.
- Modular Architecture: Redesigned Plugin System with abstract base classes for easy creation of new Input (Camera, Voice) and Output (TTS, Arduino) modules.

## New Features: RAG & Database
- RAG Integration for Memory: Added a new RAG module using FAISS and Supabase to provide Per-User Memory by automatically retrieving Top-8 relevant past conversations as context for new queries.
- Supabase postgresql Backend: Complete SQL database integration for user management, chat persistence, and GPT-powered Interest Tracking and Topic Inference.
