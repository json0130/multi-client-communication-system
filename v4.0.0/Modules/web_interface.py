# web_interface.py - Web UI and Monitoring Interface
from flask import Response

class WebInterface:
    """Web interface for monitoring and live streaming"""
    
    def __init__(self, stream_fps=30):
        self.stream_fps = stream_fps
    
    def get_monitor_html(self):
        """Generate the monitoring interface HTML"""
        return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Emotion Detection Monitor</title>
    <link href="https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&family=Roboto:wght@300;400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --blue-primary: #1a73e8;
            --blue-light: #4285f4;
            --blue-lighter: #e8f0fe;
            --gray-50: #f8f9fa;
            --gray-100: #f1f3f4;
            --gray-200: #e8eaed;
            --gray-300: #dadce0;
            --gray-500: #9aa0a6;
            --gray-700: #5f6368;
            --gray-900: #202124;
            --white: #ffffff;
            --green: #34a853;
            --yellow: #fbbc04;
            --red: #ea4335;
            --shadow: 0 1px 2px 0 rgba(60,64,67,0.3), 0 1px 3px 1px rgba(60,64,67,0.15);
            --shadow-hover: 0 1px 3px 0 rgba(60,64,67,0.3), 0 4px 8px 3px rgba(60,64,67,0.15);
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Google Sans', 'Roboto', sans-serif;
            background-color: var(--gray-50);
            color: var(--gray-900);
            line-height: 1.5;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 24px;
        }

        .header {
            background-color: var(--white);
            box-shadow: var(--shadow);
            margin-bottom: 24px;
            border-radius: 8px;
            padding: 24px;
        }

        .header h1 {
            font-size: 28px;
            font-weight: 500;
            color: var(--gray-900);
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 4px 12px;
            background-color: var(--green);
            color: var(--white);
            border-radius: 16px;
            font-size: 14px;
            font-weight: 500;
        }

        .status-badge.error {
            background-color: var(--red);
        }

        .pulse {
            width: 8px;
            height: 8px;
            background-color: var(--white);
            border-radius: 50%;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.5; transform: scale(1.2); }
            100% { opacity: 1; transform: scale(1); }
        }

        .content {
            display: grid;
            grid-template-columns: 1fr 500px;
            gap: 24px;
            margin-bottom: 24px;
        }

        .video-section {
            background-color: var(--white);
            border-radius: 8px;
            box-shadow: var(--shadow);
            padding: 24px;
        }

        .video-container {
            position: relative;
            background-color: var(--gray-900);
            border-radius: 8px;
            overflow: hidden;
            aspect-ratio: 16/9;
            display: flex;
            justify-content: center;
            align-items: center;
        }

        #videoStream {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }

        .video-placeholder {
            color: var(--gray-500);
            text-align: center;
            padding: 48px;
        }

        .video-placeholder svg {
            width: 64px;
            height: 64px;
            margin-bottom: 16px;
            opacity: 0.5;
        }

        .emotion-display {
            position: absolute;
            top: 20px;
            left: 20px;
            background-color: rgba(255, 255, 255, 0.95);
            padding: 16px 20px;
            border-radius: 8px;
            box-shadow: var(--shadow);
            backdrop-filter: blur(10px);
        }

        .emotion-label {
            font-size: 14px;
            color: var(--gray-700);
            margin-bottom: 4px;
        }

        .emotion-value {
            font-size: 24px;
            font-weight: 500;
            color: var(--blue-primary);
            text-transform: capitalize;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .confidence-bar {
            width: 200px;
            height: 8px;
            background-color: var(--gray-200);
            border-radius: 4px;
            overflow: hidden;
            margin-top: 8px;
        }

        .confidence-fill {
            height: 100%;
            background-color: var(--blue-primary);
            transition: width 0.3s ease;
        }

        .confidence-text {
            font-size: 14px;
            color: var(--gray-700);
            margin-top: 4px;
        }

        .chat-section {
            background-color: var(--white);
            border-radius: 8px;
            box-shadow: var(--shadow);
            display: flex;
            flex-direction: column;
            height: 700px;
        }

        .chat-header {
            padding: 24px;
            border-bottom: 1px solid var(--gray-200);
        }

        .chat-header h2 {
            font-size: 22px;
            font-weight: 500;
            color: var(--gray-900);
        }

        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 24px;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }

        .message {
            display: flex;
            gap: 16px;
            animation: fadeIn 0.3s ease;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .message-avatar {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
            flex-shrink: 0;
        }

        .user-message .message-avatar {
            background-color: var(--blue-lighter);
        }

        .bot-message .message-avatar {
            background-color: var(--gray-100);
        }

        .message-content {
            flex: 1;
        }

        .message-header {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 8px;
        }

        .message-author {
            font-size: 16px;
            font-weight: 500;
            color: var(--gray-700);
        }

        .message-time {
            font-size: 14px;
            color: var(--gray-500);
        }

        .message-text {
            font-size: 16px;
            color: var(--gray-900);
            line-height: 1.6;
            padding: 16px 20px;
            background-color: var(--gray-50);
            border-radius: 12px;
            display: inline-block;
            max-width: 100%;
            word-wrap: break-word;
        }

        .user-message .message-text {
            background-color: var(--blue-lighter);
            color: var(--blue-primary);
        }

        .bot-message .message-text {
            background-color: var(--gray-100);
        }

        .no-messages {
            text-align: center;
            color: var(--gray-500);
            padding: 48px;
            font-size: 16px;
        }

        @media (max-width: 1024px) {
            .content {
                grid-template-columns: 1fr;
            }

            .chat-section {
                height: 500px;
            }
        }

        .chat-messages::-webkit-scrollbar {
            width: 8px;
        }

        .chat-messages::-webkit-scrollbar-track {
            background: var(--gray-100);
            border-radius: 4px;
        }

        .chat-messages::-webkit-scrollbar-thumb {
            background: var(--gray-300);
            border-radius: 4px;
        }

        .chat-messages::-webkit-scrollbar-thumb:hover {
            background: var(--gray-500);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>
                🐱 Local Emotion Detection Monitor
                <span class="status-badge" id="connectionStatus">
                    <span class="pulse"></span>
                    <span id="statusText">Connecting...</span>
                </span>
            </h1>
        </div>

        <div class="content">
            <div class="video-section">
                <div class="video-container">
                    <div class="video-placeholder" id="videoPlaceholder">
                        <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"></path>
                        </svg>
                        <p>Waiting for video stream...</p>
                    </div>
                    <img id="videoStream" style="display: none;" />
                    <div class="emotion-display" id="emotionDisplay" style="display: none;">
                        <div class="emotion-label">Detected Emotion</div>
                        <div class="emotion-value" id="emotionValue">
                            <span id="emotionText">neutral</span>
                            <span id="emotionEmoji">😐</span>
                        </div>
                        <div class="confidence-bar">
                            <div class="confidence-fill" id="confidenceFill"></div>
                        </div>
                        <div class="confidence-text" id="confidenceText">Confidence: 0%</div>
                    </div>
                </div>
            </div>

            <div class="chat-section">
                <div class="chat-header">
                    <h2>Live Chat</h2>
                </div>
                <div class="chat-messages" id="chatMessages">
                    <div class="no-messages">
                        No messages yet. Chat activity will appear here.
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <script>
        const socket = io();
        let messageCount = 0;
        let totalConfidence = 0;
        let emotionCounts = {};

        const statusBadge = document.getElementById('connectionStatus');
        const statusText = document.getElementById('statusText');
        const videoStream = document.getElementById('videoStream');
        const videoPlaceholder = document.getElementById('videoPlaceholder');
        const emotionDisplay = document.getElementById('emotionDisplay');
        const chatMessages = document.getElementById('chatMessages');

        const emotionEmojis = {
            'happy': '😊',
            'sad': '😢',
            'angry': '😠',
            'fear': '😨',
            'surprise': '😲',
            'disgust': '🤢',
            'contempt': '😤',
            'neutral': '😐'
        };

        videoStream.src = window.location.origin + '/live_stream';
        videoStream.onload = () => {
            videoPlaceholder.style.display = 'none';
            videoStream.style.display = 'block';
            emotionDisplay.style.display = 'block';
        };

        socket.on('connect', () => {
            console.log('Connected to local server');
            statusBadge.classList.remove('error');
            statusText.textContent = 'Connected (Local)';
            socket.emit('join_stream');
        });

        socket.on('disconnect', () => {
            console.log('Disconnected from server');
            statusBadge.classList.add('error');
            statusText.textContent = 'Disconnected';
        });

        socket.on('live_frame_update', (data) => {
            if (data.emotion && data.confidence) {
                updateEmotionDisplay(data.emotion, data.confidence);
                updateStats(data.emotion, data.confidence);
            }
        });

        socket.on('chat_message', (data) => {
            addChatMessage(data);
        });

        function updateEmotionDisplay(emotion, confidence) {
            const emotionText = document.getElementById('emotionText');
            const emotionEmoji = document.getElementById('emotionEmoji');
            const confidenceFill = document.getElementById('confidenceFill');
            const confidenceText = document.getElementById('confidenceText');

            emotionText.textContent = emotion;
            emotionEmoji.textContent = emotionEmojis[emotion] || '😐';
            confidenceFill.style.width = confidence + '%';
            confidenceText.textContent = `Confidence: ${Math.round(confidence)}%`;

            if (confidence > 70) {
                confidenceFill.style.backgroundColor = 'var(--green)';
            } else if (confidence > 40) {
                confidenceFill.style.backgroundColor = 'var(--yellow)';
            } else {
                confidenceFill.style.backgroundColor = 'var(--red)';
            }
        }

        function updateStats(emotion, confidence) {
            totalConfidence += confidence;
            messageCount++;

            if (!emotionCounts[emotion]) {
                emotionCounts[emotion] = 0;
            }
            emotionCounts[emotion]++;
        }

        function addChatMessage(data) {
            const noMessages = chatMessages.querySelector('.no-messages');
            if (noMessages) {
                noMessages.remove();
            }

            const message = document.createElement('div');
            message.className = `message ${data.type === 'user' ? 'user-message' : 'chatbox-message'}`;

            const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

            message.innerHTML = `
                <div class="message-avatar">
                    ${data.type === 'user' ? '👤' : '🐱'}
                </div>
                <div class="message-content">
                    <div class="message-header">
                        <span class="message-author">${data.type === 'user' ? 'User' : 'ChatBox'}</span>
                        <span class="message-time">${time}</span>
                        ${data.emotion ? `<span>${emotionEmojis[data.emotion] || ''}</span>` : ''}
                    </div>
                    <div class="message-text">${data.content}</div>
                </div>
            `;

            chatMessages.appendChild(message);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
    </script>
</body>
</html>
        '''
    
    def generate_live_stream(self, get_latest_frame):
        """Generate live video stream"""
        def generate():
            import time
            import cv2
            
            while True:
                frame = get_latest_frame()
                if frame is not None:
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    frame_bytes = buffer.tobytes()

                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

                time.sleep(1.0 / self.stream_fps)

        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')