import requests

# Jay's code - Do not change unless modified on his end
class LocalHTTPClient:
    """HTTP client for chat messages to local server"""
   
    def __init__(self, server_url, api_key="emotion_recognition_key_123"):
        self.server_url = server_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({'Authorization': f'Bearer {api_key}'})
   
    def send_chat_message(self, message):
        """Send text chat message to local server"""
        try:
            payload = {"message": message}
            response = self.session.post(
                f"{self.server_url}/chat",
                json=payload,
                timeout=15
            )
           
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Chat request failed: {response.status_code}")
                return None
               
        except Exception as e:
            print(f"Error sending chat message: {e}")
            return None

    def send_speech_message(self, audio_data):
        """Send speech audio to local server for transcription and chat"""
        try:
            # Encode audio as base64
            audio_b64 = base64.b64encode(audio_data).decode('utf-8')
            
            payload = {"audio": audio_b64}
            response = self.session.post(
                f"{self.server_url}/speech",
                json=payload,
                timeout=30  # Longer timeout for speech processing
            )

            if response.status_code == 200:
                return response.json()
            else:
                print(f"Speech request failed: {response.status_code}")
                if response.status_code == 400:
                    try:
                        error_data = response.json()
                        print(f"   Error details: {error_data.get('details', 'Unknown error')}")
                    except:
                        pass
                return None

        except Exception as e:
            print(f"Error sending speech message: {e}")
            return None
    
    # Function for images implemented for iRobi
    def send_image(self, image_bytes, filename="capture.jpg"):
        files = {'file': (filename, image_bytes, 'image/jpeg')}
        try:
            response = requests.post(f"{self.server_url}/image", files=files)
            return response.json()

        except Exception as e:
            return {"response" : f"Error sending image message: {str(e)}"}
        

#test_image = "test.jpg"
            if not os.path.exists(test_image):
                self.add_message("Test Image not found", "bot")
                return
            frame = cv2.imread(test_image)
            if frame is None:
                self.add_message("Frame not found", "bot")
                return
            # Encode image
            success, buffer = cv2.imencode('.jpg', frame)
            if not success:
                self.add_message("Failed to encode image", "bot")
                return
            img_bytes = buffer.tobytes()
            reply = self.http_client.send_image(img_bytes)
            text = reply.get("response", "(No response)")