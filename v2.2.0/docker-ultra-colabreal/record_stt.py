import subprocess
import whisper

# Step 1: Record from mic
subprocess.run([
    "arecord", "-D", "plughw:2,0", "-f", "cd", "-t", "wav", "-d", "10", "input.wav"
])

# Step 2: Transcribe using Whisper
model = whisper.load_model("tiny")  # or "base", "small" if Jetson has enough RAM
result = model.transcribe("input.wav")
print("You said:", result["text"])

