# Release Notes v4.1.1: Face Recognition & Identification (WIP)

## New Core Processing Module
- Face Recognition Processor: Added a new FaceRecoProcessor module. This module integrates dlib's powerful face recognition capabilities directly into the system.
- Face Detection & Encoding: The new processor can take an image (e.g., from the CameraInputModule or RealSenseInputModule) and perform two key actions:
- Detect all human faces present in the frame.
- Compute a unique 128-bit encoding (a "faceprint") for each detected face.
- Face Identification: The module includes a recognize_faces function that can compare detected faces against a list of known encodings. This enables the system to identify specific, "known" individuals in real-time.
