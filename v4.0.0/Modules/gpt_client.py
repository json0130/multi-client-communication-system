# gpt_client.py - OpenAI GPT Integration
import os
import re
from openai import OpenAI

class GPTClient:
    """OpenAI GPT client for emotion-aware chat responses"""
    
    def __init__(self):
        self.client = None
        self.available = False
        
    def setup_openai(self):
        """Setup OpenAI client"""
        try:
            # Try to get API key from environment variables
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                print("Warning: OPENAI_API_KEY not set in environment")
                print("Please set your OpenAI API key: export OPENAI_API_KEY='your-key-here'")
                return False

            self.client = OpenAI(api_key=api_key)
            self.available = True
            print("OpenAI client initialized")
            return True
        except Exception as e:
            print(f"OpenAI setup failed: {e}")
            self.available = False
            return False
    
    def ask_chatgpt_optimized(self, prompt, detected_emotion, confidence):
        """Optimized ChatGPT request with timeout"""
        if self.client is None:
            return "[DEFAULT] ChatGPT is not available."

        try:
            tagged_prompt = f"[{detected_emotion}] {prompt}"

            system_prompt = """Your name is ChatBox. You are a gentle, kind, and supportive robot designed to be a companion for children with mental health challenges. You always speak in a calm and friendly tone, using simple and concise language so children can easily understand and stay focused. Your responses are meant to make children feel safe, heard, and supported.

IMPORTANT:
- Always start your response with one of the following emotion tags in square brackets, like [SAD] or [POSE].
  Tags: [GREETING], [WAVE], [POINT], [CONFUSED], [SHRUG], [ANGRY], [SAD], [SLEEP], [DEFAULT], [POSE]
- Do NOT invent new emotion tags.
- Choose the tag that best reflects the tone of your response, not necessarily the user's input emotion.
- Respond naturally after the tag."""

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": tagged_prompt},
                ],
                timeout=10
            )

            return response.choices[0].message.content

        except Exception as e:
            return "[DEFAULT] Sorry, I encountered an error."
    
    def extract_emotion_tag(self, text):
        """Extract emotion tag from response"""
        match = re.match(r"\[(.*?)\]", text)
        return match.group(1) if match else "DEFAULT"
    
    def is_available(self):
        """Check if GPT client is available"""
        return self.available
    
    def get_status(self):
        """Get GPT client status"""
        return {
            'available': self.available,
            'client_initialized': self.client is not None
        }