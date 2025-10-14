# gpt_client.py - OpenAI GPT Integration
import os
import re
from openai import OpenAI
from collections import deque
from dataclasses import dataclass
from typing import Deque

import requests

@dataclass
class Message:
    role: str
    content: str

class GPTClient:
    """OpenAI/OpenRouter GPT client for emotion-aware chat responses"""
    
    def __init__(self):
        self.client = None
        self.available = False
        self.conversation_history: Deque[Message] = deque(maxlen=20)

        # OpenRouter specific settings
        self.is_openrouter = self.config.get('provider', '').lower() == 'openrouter'
        self.openrouter_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model = self.config.get('model', 'openai/gpt-oss-20b:free')    # default fallback model
        
    def setup_openai(self):
        """Setup OpenAI/OpenRouter client"""
        try:
            # Try to get API key from environment variables
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                print("Warning: OPENAI_API_KEY not set in environment")
                print("Please set your OpenAI API key: export OPENAI_API_KEY='your-key-here'")
                return False

            if self.is_openrouter:
                # For OpenRouter we don't need the OpenAI client
                self.available = True
                print("OpenRouter selected, using custom endpoint")
                return True
            else:
                self.client = OpenAI()
                self.available = True
                print("OpenAI selected, client initialized")
                return True
            
        except Exception as e:
            print(f"OpenAI setup failed: {e}")
            self.available = False
            return False
    
    def ask_chatgpt_optimized(self, user_prompt: str, system_prompt: str):
        """
        Generic, optimized ChatGPT request that accepts a custom system prompt.
        """
        if self.client is None:
            return "[DEFAULT] ChatGPT is not available."

        try:
            # Add user message to history
            self.conversation_history.append(Message("user", user_prompt))
            # Convert history to message format
            messages = [{"role": "system", "content": system_prompt}]
            # Add conversation history
            for msg in self.conversation_history:
                messages.append({"role": msg.role, "content": msg.content})
            
            if self.is_openrouter:
                # Use OpenRouter API
                headers = {
                    "Authorization": f"Bearer {self.config.get('api_key')}",
                    "HTTP-Referer": "https://github.com/your-repo",  # Replace with your repo
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.config.get('temperature', 0.7),
                    "max_tokens": self.config.get('max_tokens', 1000)
                }
                
                response = requests.post(
                    self.openrouter_url,
                    headers=headers,
                    json=payload
                )
                
                if response.status_code == 200:
                    assistant_response = response.json()['choices'][0]['message']['content']
                else:
                    raise Exception(f"OpenRouter API error: {response.text}")
            
            else:
                # Use original OpenAI API
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    timeout=10
                )

                assistant_response = response.choices[0].message.content
                
            # Store assistant's response in history
            self.conversation_history.append(Message("assistant", assistant_response))
            return assistant_response

        except Exception as e:
            print(f"❌ Error in optimized GPT request: {e}")
            return "[DEFAULT] Sorry, I encountered an error."
    
    def ask_with_dynamic_prompt(self, full_prompt: str):
        """
        Sends a pre-formatted, complex prompt to the LLM.
        This bypasses the default system prompt and emotion tagging.
        Used for advanced logic like task delegation.
        """
        if self.client is None:
            return "[DEFAULT] ChatGPT is not available."

        try:
            # Our full_prompt contains both the system instructions and the user message.
            # It looks for the last occurrence of 'User:' to make the split.
            if '\nUser: ' in full_prompt:
                parts = full_prompt.rsplit('\nUser: ', 1)
                system_content = parts[0].replace('System: ', '').strip()
                user_content = parts[1].strip()
            else:
                # Fallback if the format is unexpected
                system_content = "You are a helpful AI assistant."
                user_content = full_prompt

            # Add user message to history
            self.conversation_history.append(Message("user", user_content))

            # Convert history to message format
            messages = [{"role": "system", "content": system_content}]
            # Add conversation history
            for msg in self.conversation_history:
                messages.append({"role": msg.role, "content": msg.content})

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                timeout=20
            )

            assistant_response = response.choices[0].message.content
            # Store assistant's response in history
            self.conversation_history.append(Message("assistant", assistant_response))

            return assistant_response

        except Exception as e:
            print(f"❌ Error in dynamic prompt request: {e}")
            return "[DEFAULT] Sorry, I encountered an error during the advanced request."
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history.clear()

    def get_conversation_history(self) -> list[Message]:
        """Get current conversation history"""
        return list(self.conversation_history)
    
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
        
    def infer_topics_and_conditions(self, messages: list[str]) -> tuple[list[str], list[str]]:
        """
        Given a list of user messages → return (interests[], health_conditions[]).
        Uses a short JSON-only response format to keep parsing trivial.
        """
        if not self.available:
            raise RuntimeError("OpenAI client not initialised")

        joined = "\n".join(f"- {m}" for m in messages[:100])  # safety cap
        sys_prompt = (
            "You are an analyst. From the USER’S sentences below, extract:\n"
            "1. up to 10 distinct long-term interests or hobbies\n"
            "2. any explicit or strongly implied health conditions.\n"
            "Return **ONLY** valid JSON like:\n"
            '{ "interests": ["..."], "health_conditions": ["..."] }'
        )

        resp = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": joined}
            ],
            temperature=0.3,
            timeout=15,
        )
        import json, re
        txt = resp.choices[0].message.content.strip()
        # tolerate code-block fencing
        txt = re.sub(r"^```(json)?|```$", "", txt).strip()
        data = json.loads(txt)
        return data.get("interests", []), data.get("health_conditions", [])