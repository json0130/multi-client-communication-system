import multiprocessing
import subprocess
import time
import os

# Optional: Set paths for logs or emotion file
EMOTION_FILE = "emotion_file.txt"

def run_emotion_detection():
    """Run the emotion detection system (test.py)."""
    try:
        print("[Emotion Detection] Starting...")
        subprocess.run(["python3", "test.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"[Emotion Detection] Error: {e}")

def run_chatbot():
    """Run the chatbot system (chatbot.py)."""
    try:
        print("[Chatbot] Starting...")
        subprocess.run(["python3", "chatbot.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"[Chatbot] Error: {e}")

def main():
    # Ensure the emotion file exists
    if not os.path.exists(EMOTION_FILE):
        with open(EMOTION_FILE, "w") as f:
            f.write("neutral:0.0")
        print(f"[Main] Created {EMOTION_FILE}")

    # Use multiprocessing to run both processes concurrently
    emotion_process = multiprocessing.Process(target=run_emotion_detection)
    chatbot_process = multiprocessing.Process(target=run_chatbot)

    # Start both processes
    emotion_process.start()
    chatbot_process.start()

    # Wait for both processes to finish
    try:
        emotion_process.join()
        chatbot_process.join()
    except KeyboardInterrupt:
        print("[Main] Interrupted by user. Terminating processes...")
        emotion_process.terminate()
        chatbot_process.terminate()
    finally:
        emotion_process.join()
        chatbot_process.join()
        print("[Main] Shutdown complete.")

if __name__ == "__main__":
    main()

