# Version 4.0.0 Release Notes

## Major Architecture Overhaul
### Multi-Client Server Architecture
- Complete Server Rewrite: New modular server controller supporting multiple concurrent clients
- Individual Client Isolation: Each client gets dedicated server instance with separate resources
- Client Manager: Automated client registration, lifecycle management, and resource allocation
- Request Router: Intelligent routing of requests to appropriate client server instances
- WebSocket Manager: Enhanced real-time communication with session-based client tracking

### Modular Client Architecture
- Plugin-Based System: Completely redesigned client with interchangeable input/output modules
- Generic Image Input Module: Universal camera support (USB, built-in, IP cameras) with frame preprocessing
- RealSense Input Module: Optimised Intel RealSense camera integration with web streaming
- Speech Input Module: Enhanced USB microphone auto-detection and audio processing
- Arduino Output Module: Complete emotion display integration with auto-reconnection
- TTS Output Module: Multi-engine text-to-speech with emotion-based voice adaptation

## Key New Features
### Individual Client Monitoring
- Dedicated Dashboards: /client/{id}/monitor for each robot
- Isolated Video Streams: Separate live streams per client
- Resource Isolation: No cross-client interference

### Modules
- Image Processing: Multi-camera support with background encoding
- RealSense Optimised: Native D400 support with web streaming
- Speech Processing: Enhanced audio with auto-device detection
- Output Systems: Arduino emotion display + multi-engine TTS
