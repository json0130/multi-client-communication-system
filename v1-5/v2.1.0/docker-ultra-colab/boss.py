# boss.py - Updated to use client_test.py for cloud emotion detection
import multiprocessing
import subprocess
import time
import os

# Optional: Set paths for logs or emotion file
EMOTION_FILE = "/app/emotion_file.txt"  # Use absolute path for Docker

def run_emotion_detection():
    """Run the cloud-based emotion detection system (client_test.py)."""
    try:
        print("[Emotion Detection] Starting cloud-based emotion detection...")
        # Use client_test.py instead of test.py
        subprocess.run(["python3", "client_test.py"], check=True)
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
    # Ensure the emotion file exists with proper permissions
    if not os.path.exists(EMOTION_FILE):
        try:
            with open(EMOTION_FILE, "w") as f:
                f.write("neutral:0.0")
            print(f"[Main] Created {EMOTION_FILE}")
            # Try to set permissions
            try:
                os.chmod(EMOTION_FILE, 0o666)
                print(f"[Main] Set permissions for {EMOTION_FILE}")
            except Exception as e:
                print(f"[Main] Warning: Could not set permissions: {e}")
        except Exception as e:
            print(f"[Main] Error creating emotion file: {e}")
            print("[Main] Will try creating it in the current directory...")
            try:
                with open("emotion_file.txt", "w") as f:
                    f.write("neutral:0.0")
                print("[Main] Created emotion_file.txt in current directory")
            except Exception as e2:
                print(f"[Main] Error creating alternative emotion file: {e2}")

    # Display information about cloud connection
    print("[Main] Using cloud-based emotion recognition")
    
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
