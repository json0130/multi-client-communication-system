# realsense_stream.py
import pyrealsense2 as rs
import cv2
import numpy as np
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
import socketserver
import json
from queue import Queue, Empty
import os

class RealSenseStreamer:
    def __init__(self, web_port=8080, frame_processor=None):
        self.web_port = web_port
        self.latest_frame = None
        self.latest_jpg = None
        self.metadata = {
            "emotions": [("Unknown", 0.0), ("Unknown", 0.0), ("Unknown", 0.0)],  # list of (emotion, confidence)
            "fps": 0.0
        }
        self.running = False
        self.server = None
        self.pipeline = None
        self.server_thread = None
        self.stream_thread = None
        self.frame_processor = frame_processor  # Optional callback for frame processing
        self.frame_queue = Queue(maxsize=1)  # Only keep the most recent frame
        self.processed_queue = Queue(maxsize=1)  # Only keep the most recent processed frame
        
    def start_pipeline(self):
        """Initialize and start the RealSense camera pipeline"""
        # print("Initializing RealSense pipeline...")
        self.pipeline = rs.pipeline()
        config = rs.config()
        
        # Enable streams at a higher frame rate for better performance
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # Start streaming
        try:
            pipeline_profile = self.pipeline.start(config)
            device = pipeline_profile.get_device()
            # print(f"Using device: {device.get_info(rs.camera_info.name)} (S/N: {device.get_info(rs.camera_info.serial_number)})")
            return True
        except Exception as e:
            print(f"Error starting pipeline: {e}")
            return False
            
    def get_frame(self):
        """Get the latest frame from RealSense camera"""
        if not self.pipeline:
            return None
            
        try:
            frames = self.pipeline.wait_for_frames(5000)  # 5 second timeout
            color_frame = frames.get_color_frame()
            
            if not color_frame:
                return None
                
            # Convert to numpy array (this is already a copy, no need for another)
            return np.asanyarray(color_frame.get_data())
        except Exception as e:
            print(f"Error getting frame: {e}")
            return None
            
    def update_metadata(self, key, value):
        """Update metadata for the frame"""
        self.metadata[key] = value
        
    def capture_frames(self):
        """Thread for capturing frames from the camera and putting them in the queue"""
        fps_frame_count = 0
        fps_start_time = time.time()
        
        while self.running:
            frame = self.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue
                
            # Update FPS calculation
            fps_frame_count += 1
            current_time = time.time()
            elapsed = current_time - fps_start_time
            
            if elapsed >= 1.0:
                fps = fps_frame_count / elapsed
                self.update_metadata("fps", round(fps, 1))
                fps_frame_count = 0
                fps_start_time = current_time
                
            # Save this frame as the latest
            self.latest_frame = frame
            
            # Put in queue for processing, replacing any existing frame
            if not self.frame_queue.full():
                self.frame_queue.put(frame)
            else:
                try:
                    self.frame_queue.get_nowait()  # Remove old frame
                    self.frame_queue.put(frame)  # Add new frame
                except Empty:
                    pass
                
            # Avoid using 100% CPU
            time.sleep(0.001)
    
    def process_frames(self):
        """Process frames using the provided processor or default processing"""
        while self.running:
            try:
                # Get a frame to process
                frame = None
                try:
                    frame = self.frame_queue.get(timeout=0.1)
                except Empty:
                    continue
                    
                # Add FPS text
                display_frame = frame.copy()
                cv2.putText(display_frame, f"FPS: {self.metadata['fps']}", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Use custom processor if provided
                if self.frame_processor:
                    processed = self.frame_processor(frame, display_frame)
                    if processed is not None:
                        display_frame = processed
                
                # Convert to JPEG
                ret, jpeg = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    self.latest_jpg = jpeg.tobytes()
                
                # Put processed frame in queue
                if not self.processed_queue.full():
                    self.processed_queue.put(display_frame)
                else:
                    try:
                        self.processed_queue.get_nowait()  # Remove old frame
                        self.processed_queue.put(display_frame)  # Add new frame
                    except Empty:
                        pass
                
            except Exception as e:
                print(f"Error processing frame: {e}")
                time.sleep(0.01)
    
    def update_emotions(self, emotions_list):
        """
        emotions_list: list of tuples like [("happy", 80.5), ("sad", 10.0), ("neutral", 5.5)]
        """
        # truncate or pad the list to length 3
        emotions_list = emotions_list[:3] + [("Unknown", 0.0)]*(3 - len(emotions_list))
        self.metadata["emotions"] = emotions_list

                
    class StreamingHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            streamer = self.server.streamer
            
            if self.path == '/':
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                
                html = f'''
                <!DOCTYPE html>
                <html>
                <head>
                    <title>RealSense Camera Stream</title>
                    <meta name="viewport" content="width=device-width, initial-scale=1">
                    <style>
                        body {{
                            font-family: Arial, sans-serif;
                            margin: 0;
                            padding: 20px;
                            text-align: center;
                            background-color: #f0f0f0;
                        }}
                        .container {{
                            max-width: 800px;
                            margin: 0 auto;
                            background: white;
                            padding: 20px;
                            border-radius: 10px;
                            box-shadow: 0 0 10px rgba(0,0,0,0.1);
                        }}
                        h1 {{
                            color: #333;
                        }}
                        .video-container {{
                            margin: 20px 0;
                            position: relative;
                        }}
                        .stats {{
                            background: rgba(0,0,0,0.7);
                            color: white;
                            position: absolute;
                            top: 10px;
                            left: 10px;
                            padding: 5px 10px;
                            border-radius: 5px;
                            font-size: 14px;
                        }}
                        .emotion {{
                            font-size: 24px;
                            margin: 20px 0;
                            font-weight: bold;
                            color: #2c3e50;
                        }}
                        img {{
                            max-width: 100%;
                            border: 3px solid #333;
                            border-radius: 5px;
                        }}
                    </style>
                    <script>
                        function refreshImage() {{
                            var img = document.getElementById('camera');
                            if (img.complete) {{
                                img.src = '/frame.jpg?t=' + new Date().getTime();
                            }}
                        }}
                        
                        function updateStats() {{
                            fetch('/stats')
                            .then(response => response.json())
                            .then(data => {{
                                document.getElementById('emotion').innerText = 'Detected Emotion: ' + data.emotion;
                                document.getElementById('fps').innerText = 'FPS: ' + data.fps;
                            }})
                            .catch(error => console.error('Error fetching stats:', error));
                        }}
                        
                        // Refresh image and stats periodically
                        setInterval(refreshImage, 33);  // ~30 fps
                        setInterval(updateStats, 200);  // 5 Hz
                    </script>
                </head>
                <body>
                    <div class="container">
                        <h1>RealSense Camera Stream</h1>
                        <div class="video-container">
                            <img id="camera" src="/frame.jpg" alt="Camera Feed">
                            <div class="stats">
                                <div id="fps">FPS: {streamer.metadata.get("fps", 0)}</div>
                            </div>
                        </div>
                        <div id="emotion" class="emotion">Detected Emotion: {streamer.metadata.get("emotion", "Unknown")}</div>
                    </div>
                </body>
                </html>
                '''
                self.wfile.write(html.encode())
                
            elif self.path.startswith('/frame.jpg'):
                if streamer.latest_jpg is not None:
                    self.send_response(200)
                    self.send_header('Content-type', 'image/jpeg')
                    self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
                    self.end_headers()
                    self.wfile.write(streamer.latest_jpg)
                else:
                    self.send_response(503)
                    self.end_headers()
                    
            elif self.path == '/stats':
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Cache-Control', 'no-cache')
                self.end_headers()
                self.wfile.write(json.dumps(streamer.metadata).encode())
                
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format, *args):
            # Suppress log messages
            return

    class StreamingServer(socketserver.ThreadingMixIn, HTTPServer):
        allow_reuse_address = True
        daemon_threads = True
        
        def __init__(self, server_address, RequestHandlerClass, streamer):
            self.streamer = streamer
            super().__init__(server_address, RequestHandlerClass)
            
    def start_server(self):
        """Start the HTTP server for streaming"""
        server_address = ('0.0.0.0', self.web_port)
        self.server = self.StreamingServer(server_address, self.StreamingHandler, self)
        # print(f"Starting web server on port {self.web_port}...")
        # print(f"Access the stream at http://localhost:{self.web_port}")
        self.server.serve_forever()
        
    def start(self):
        """Start both the camera pipeline and streaming server"""
        if self.running:
            # print("Streamer is already running")
            return False
            
        if not self.start_pipeline():
            print("Failed to start RealSense pipeline")
            return False
            
        self.running = True
        
        # Start capture thread
        capture_thread = threading.Thread(target=self.capture_frames)
        capture_thread.daemon = True
        capture_thread.start()
        
        # Start processing thread
        processing_thread = threading.Thread(target=self.process_frames)
        processing_thread.daemon = True
        processing_thread.start()
        
        # Start web server thread
        server_thread = threading.Thread(target=self.start_server)
        server_thread.daemon = True
        server_thread.start()
        
        # Save thread references
        self.stream_thread = capture_thread
        self.processing_thread = processing_thread
        self.server_thread = server_thread
        
        return True
        
    def stop(self):
        """Stop the streamer and release resources"""
        # print("Stopping streamer...")
        self.running = False
        
        # Give threads time to clean up
        time.sleep(0.5)
        
        if self.server:
            try:
                self.server.shutdown()
                self.server.server_close()
            except Exception as e:
                print(f"Error shutting down server: {e}")
            
        if self.pipeline:
            try:
                self.pipeline.stop()
            except Exception as e:
                print(f"Error stopping pipeline: {e}")
            
        # print("Streamer stopped")
        
    def get_latest_frame(self):
        """Get the latest frame for processing"""
        # return self.latest_frame
        
    def get_latest_processed_frame(self):
        """Get the latest processed frame"""
        try:
            return self.processed_queue.get_nowait()
        except Empty:
            return None

if __name__ == "__main__":
    # Simple test when run directly
    streamer = RealSenseStreamer()
    
    try:
        if streamer.start():
            # print("Streamer started successfully")
            print("Press Ctrl+C to exit")
            # Keep the main thread running
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        streamer.stop()
