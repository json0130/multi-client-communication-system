# chatbot.py - Updated for espeak-based TTS in Docker
from openai import OpenAI
import os
import time
import serial
import re
import threading
import subprocess
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
EMOTION_FILE = "/app/emotion_file.txt"
FALLBACK_EMOTION_FILE = "emotion_file.txt"
DEFAULT_EMOTION = "neutral"
EMOTION_CACHE_TIME = 2  # seconds
SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE = 9600
DEBUG_MODE = False

TTS_ENABLED = True
TTS_RATE = 150  # Approximate words per minute
TTS_VOLUME = 100  # espeak volume from 0 to 200

# Globals
last_emotion_read_time = 0
cached_emotion = DEFAULT_EMOTION
cached_confidence = 0.0

# Setup OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Setup serial connection
try:
    arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    print(f"Connected to Arduino on {SERIAL_PORT}")
except serial.SerialException:
    arduino = None
    print(f"⚠️  Warning: Could not connect to Arduino on {SERIAL_PORT}")

def get_current_emotion():
    global last_emotion_read_time, cached_emotion, cached_confidence

    current_time = time.time()
    if current_time - last_emotion_read_time < EMOTION_CACHE_TIME:
        return cached_emotion, cached_confidence

    try:
        if os.path.exists(EMOTION_FILE):
            with open(EMOTION_FILE, "r") as f:
                content = f.read().strip()
            if ":" in content:
                emotion, confidence_str = content.split(":", 1)
                confidence = float(confidence_str) if confidence_str else 0.0
            else:
                emotion, confidence = content, 0.0

            cached_emotion, cached_confidence = emotion, confidence
            last_emotion_read_time = current_time
            return emotion, confidence
    except Exception as e:
        if DEBUG_MODE:
            print(f"Error reading primary emotion file: {e}")

    try:
        if os.path.exists(FALLBACK_EMOTION_FILE):
            with open(FALLBACK_EMOTION_FILE, "r") as f:
                content = f.read().strip()
            if ":" in content:
                emotion, confidence_str = content.split(":", 1)
                confidence = float(confidence_str) if confidence_str else 0.0
            else:
                emotion, confidence = content, 0.0

            cached_emotion, cached_confidence = emotion, confidence
            last_emotion_read_time = current_time
            return emotion, confidence
    except Exception as e:
        print(f"Error reading fallback emotion file: {e}")

    return DEFAULT_EMOTION, 0.0

def extract_emotion_tag(text):
    match = re.match(r"\[(.*?)\]", text)
    return match.group(1) if match else DEFAULT_EMOTION

def remove_emotion_tag(text):
    return re.sub(r"^\[(.*?)\]\s*", "", text)

def speak_text(text):
    """Speak text using espeak in a non-blocking thread"""
    if not TTS_ENABLED:
        return

    speech_text = remove_emotion_tag(text)

    def speak_worker():
        try:
            cmd = [
                "espeak",
                f"-s{TTS_RATE}",
                f"-a{TTS_VOLUME}",
                speech_text
            ]
            subprocess.run(cmd)
        except Exception as e:
            print(f"🔊 Speech error (espeak): {e}")

    thread = threading.Thread(target=speak_worker)
    thread.daemon = True
    thread.start()

def send_emotion_to_arduino(emotion):
    if arduino and arduino.is_open:
        try:
            arduino.write(f"{emotion}\n".encode('utf-8'))
            print(f"🛰️  Sent to Arduino: {emotion}")
        except Exception as e:
            print(f"⚠️  Serial write failed: {e}")

def ask_chatgpt(prompt, detected_emotion=None, user_emotion=None):
    emotion = user_emotion if user_emotion else detected_emotion
    try:
        tagged_prompt = f"[{emotion}] {prompt}"

        system_prompt = """You are a helpful assistant that responds to users based on their emotional state and always tags your response with one of the following emotion categories and their meaning:

- [GREETING] 
- [WAVE] 
- [POINT] 
- [CONFUSED] 
- [SHRUG] 
- [ANGRY] 
- [SAD] 
- [SLEEP] 
- [DEFAULT] 
- [POSE]

IMPORTANT:
- Always start your response with one of the above emotion tags in square brackets, like [SAD] or [POSE].
- Do NOT invent new emotion tags.
- Choose the tag that best reflects the tone of your response, not necessarily the user's input emotion.
- Respond naturally after the tag. For example:
  - [GREETING] Hi there! How can I assist you today?
  - [CONFUSED] I'm not quite sure I follow — could you rephrase that?
  - [ANGRY] That doesn't seem fair at all!
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": tagged_prompt},
            ]
        )

        return response.choices[0].message.content
    except Exception as e:
        print(f"Error: {type(e).__name__}: {str(e)}")
        return "Sorry, I encountered an error while processing your request."

def main():
    print("\033[1;36m" + "=" * 50 + "\033[0m")
    print("\033[1;36m     Cloud-Powered Emotion-Aware Chatbot\033[0m")
    print("\033[1;36m" + "=" * 50 + "\033[0m")
    print("Type 'exit' to end the conversation.")
    print("You can specify emotion like [happy], or it will use detected one.")

    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set in environment.")
        return

    while True:
        detected_emotion, confidence = get_current_emotion()

        if confidence > 10:
            print(f"\033[0;90mDetected emotion: {detected_emotion} ({confidence:.1f}%)\033[0m")

        user_input = input("\033[1;35mYou: \033[0m")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        user_specified_emotion = None
        if user_input.startswith("[") and "]" in user_input:
            end_idx = user_input.find("]")
            user_specified_emotion = user_input[1:end_idx]
            user_input = user_input[end_idx + 1:].strip()
            print(f"\033[0;90mUsing user-specified emotion: {user_specified_emotion}\033[0m")
        
        final_emotion = user_specified_emotion if user_specified_emotion else detected_emotion
        print(f"\033[0;94m>>> Sending to GPT: [{final_emotion}] {user_input}\033[0m")

        response = ask_chatgpt(user_input, detected_emotion, user_specified_emotion)
        print(f"\033[1;32mGPT-4o-mini: \033[0m{response}")

        response_emotion = extract_emotion_tag(response)
        send_emotion_to_arduino(response_emotion)

        if TTS_ENABLED:
            speak_text(response)

if __name__ == "__main__":
    main()

