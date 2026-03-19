# openrouter_client.py - OpenRouter Integration
import os
import re
from openai import OpenAI
from collections import deque
from dataclasses import dataclass
from typing import Deque

@dataclass
class Message:
    role: str
    content: str

class OpenRouterClient:
    """OpenRouter client for emotion-aware chat responses"""
    
    def __init__(self):
        self.client = None
        self.available = False
        self.conversation_history: Deque[Message] = deque(maxlen=20)

        
    def setup_openai(self):
        """Setup OpenRouter client"""
        try:
            # Try to get API key from environment variables
            api_key = os.getenv('OPENROUTER_API_KEY')
            if not api_key:
                print("Warning: OPENROUTER_API_KEY not set in environment")
                print("Please set your OpenRouter API key: export OPENROUTER_API_KEY='your-key-here'")
                return False

            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )
            self.available = True
            print("OpenRouter client initialized")
            return True
        except Exception as e:
            print(f"OpenRouter setup failed: {e}")
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
            # for msg in self.conversation_history:
            #     messages.append({"role": msg.role, "content": msg.content})
            
            # This is now a clean, generic function. It just passes the prompts along.
            response = self.client.chat.completions.create(
                model="arcee-ai/trinity-large-preview:free",
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
    
    def ask_with_dynamic_prompt(self, full_prompt: str, temperature: float = 0.7):
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
            #messages = [{"role": "system", "content": system_content}, {"role": "user", "content": user_content}]
            # Add conversation history
            for msg in self.conversation_history:
                messages.append({"role": msg.role, "content": msg.content})

            response = self.client.chat.completions.create(
                model="arcee-ai/trinity-large-preview:free",
                messages=messages,
                temperature=temperature,
                timeout=20
            )

            assistant_response = response.choices[0].message.content
            # Store assistant's response in history
            self.conversation_history.append(Message("assistant", assistant_response))

            return assistant_response

        except Exception as e:
            print(f"❌ Error in dynamic prompt request: {e}")
            return "[DEFAULT, OpenRouter] Sorry, I encountered an error during the advanced request."
    
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
            model="arcee-ai/trinity-large-preview:free",
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