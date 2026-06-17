import argparse
import whisper
from openai import OpenAI
import threading
import socket
import time
import json
import queue
import wave
import re
import numpy as np
import soundfile as sf
import sounddevice as sd
from pynput import keyboard


import scipy.signal
from utils.connection import Connection
from utils.colors import bcolors

class WhisperSTT():
    def __init__(self, model_size: str="small") -> None:
        self.model = whisper.load_model(model_size)
    
    def get_transcribe(self, audio: str, language: str = "en"):
        result = self.model.transcribe(audio=audio, language=language, verbose=False)
        return result.get('text', '')
    
class LLM():
    def __init__(self, ip="0.0.0.0", port=11434, model="llama2:13b", token_limit=500):
        self.client = OpenAI(
            base_url = f'http://{ip}:{port}/v1',
            api_key='ollama', # required, but unused
        )

        self.model=model
        self.temperature=0.5
        self.token_limit=token_limit
        self.history=[]
        self.system_prompt=""" 
                            Don't think, you are a helpful assistant. Keep the responses short please. Never use emojis in any response whatsoever.
                        """

        self.max_history_turns=5            # UNUSED
        self.emoji_pattern=re.compile("["
                u"\U0001F600-\U0001F64F"    # emoticons
                u"\U0001F300-\U0001F5FF"    # symbols & pictographs
                u"\U0001F680-\U0001F6FF"    # transport & map symbols
                u"\U0001F1E0-\U0001F1FF"    # flags (iOS)
                u"\U00002700-\U000027BF"    # Dingbats
                u"\U0001F900-\U0001F9FF"    # Supplemental Symbols and Pictographs
                u"\U00002600-\U000026FF"    # Misc symbols
                u"\U00002B00-\U00002BFF"    # Arrows
                u"\U0001FA70-\U0001FAFF"    # Symbols and Pictographs Extended-A
                u"\U000025A0-\U000025FF"    # Geometric Shapes
                                "]+", flags=re.UNICODE)


    # Remove special characters/emojis
    def clean(self, sentence):
        for sentence_char in sentence:
            if not sentence_char.isalnum() and not " ":
                sentence = sentence.replace(sentence_char, "")
        sentence = self.emoji_pattern.sub(r'', sentence)

        return sentence
    
    
    
    def send_query(self, query):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt}]+
                # + history +
                [{"role": "user", "content": query}],
            stream=False,  # Enable streaming
            temperature=self.temperature,
            max_tokens=self.token_limit
        )

        self.history.append({"role": "user", "content": query})
        return response.choices[0].message.content
    
    def send_query_stream(self, query):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt}]+
                # + history +
                [{"role": "user", "content": query}],
            stream=True,  # Enable streaming
            temperature=self.temperature,
            max_tokens=self.token_limit
        )

        self.history.append({"role": "user", "content": query})
        
        # Handle streaming response
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
                
        return response
    
    def send_query_sentence(self, query):
        sentence_buffer = ""
        sentence = ""
        full_response = ""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt}]+
                # + history +
                [{"role": "user", "content": query}],
            stream=True,  # Enable streaming
            temperature=self.temperature,
            max_tokens=self.token_limit
        )

        self.history.append({"role": "user", "content": query})
        
        # Handle streaming response
        for chunk in response:
            if (chunk.choices[0].delta.content is not None):
                sentence_buffer = sentence_buffer + (chunk.choices[0].delta.content)
                
                # Split sentences as the response is streamed
                try:
                    for sentence_stop in [". ", "?", "!", ";", ":", "\n"]:
                        if (sentence_stop in sentence_buffer):
                            try:
                                [sentence, sentence_buffer] = sentence_buffer.split(sentence_stop)
                            except ValueError:
                                [sentence, sentence_buffer] = [sentence_buffer, ""]

                            sentence = self.clean(sentence)
                            yield sentence
                            full_response = full_response + sentence
                            sentence = ""
                            continue
                except IndexError:
                    print(f">>>{bcolors.FAIL}INDEX ERROR{bcolors.OKCYAN}<<< ", end='', flush=True)

        if sentence_buffer != "":
            sentence = self.clean(sentence_buffer)
            yield sentence

        self.history.append({"role": "assistant", "content": full_response})
        return full_response

def on_press(key):
    print(f"Key {key} pressed.")
    return False  # Stop listener after first key press

# print("Waiting for a key press...")


# print("Key detected, continuing execution.")

def main(args):
    wspr = WhisperSTT()
    llm = LLM(args.llm_ip, args.llm_port)
    conn = Connection(ip=args.ip, port=args.port, type='client')

    while True:
        # conn.send(json.dumps({"command": "record", "content": None}).encode())
        print("Press spacebar to start recording")
        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()
        conn.send(json.dumps({"command": "record", "content": "start"}).encode())
        res = conn.receive()

        _ = conn.queue.get()
        print("Press spacebar to stop recording")

        # Communication error
        if res == -1:
            print("{}ERROR: Communuication unsuccessful{}".format(bcolors.FAIL, bcolors.ENDC)) 
            return
        

        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()
        conn.send(json.dumps({"command": "record", "content": "stop"}).encode())
        res = conn.receive()
        
        audio = conn.queue.get()

        # Recording error
        if audio == b'-1':
            print("{}ERROR: Audio failed to record due to timeout{}".format(bcolors.FAIL, bcolors.ENDC)) 
            continue

        audio = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        audio_np = scipy.signal.resample_poly(audio, 16000, 48000)
        
        wspr_response = wspr.get_transcribe(audio=audio_np)
        print(wspr_response)
        # if "pepper" not in wspr_response.lower():
        #     continue

        # conn.send(json.dumps({"command": "tts", "content": "What's up?"}).encode())
        # conn.send(json.dumps({"command": "record", "content": None}).encode())
        # res = conn.receive()
        # audio = conn.queue.get()
        # if audio == b'-1':
        #     print("{}ERROR: Audio failed to record due to timeout{}".format(bcolors.FAIL, bcolors.ENDC)) 
        #     continue

        # audio = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        # audio_np = scipy.signal.resample_poly(audio, 16000, 48000)
        
        # wspr_response = wspr.get_transcribe(audio=audio_np)
        # print(wspr_response)
        llm_response = llm.send_query_sentence(wspr_response)
        for sentence in llm_response:
            conn.send(json.dumps({"command": "tts", "content": sentence}).encode())


    conn.shut_down()


def save_wav(filename, data, rate=48000, channels=1):
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(2)
    wf.setframerate(rate)
    wf.writeframes(data)
    wf.close()
    print("Saved audio to", filename)

def read_wav(filename, channels=1):
    with wave.open(filename, 'rb') as wav_file:
        n_frames = wav_file.getnframes()
        frames = wav_file.readframes(n_frames)
        sample_width = wav_file.getsampwidth()
        dtype = {1: np.int8, 2: np.int16, 4: np.int32}[sample_width]
        rate = wav_file.getframerate()
        actual_channels = wav_file.getnchannels()

    # Convert bytes to numpy array
    audio = np.frombuffer(frames, dtype=dtype)

    # Reshape if stereo or more
    if actual_channels > 1:
        audio = audio.reshape(-1, actual_channels)

    return audio, rate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="192.168.0.3",
                        help="Robot IP address")
    parser.add_argument("--port", type=int, default=9559,
                        help="Naoqi port number")
    
    parser.add_argument("--llm_ip", type=str, default="130.216.239.52",
                        help="Robot IP address")
    parser.add_argument("--llm_port", type=int, default=11434,
                        help="Naoqi port number")
    args = parser.parse_args()

    main(args)