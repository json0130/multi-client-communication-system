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
    
    def ask_chatgpt_optimized(self, user_prompt: str, system_prompt: str):
        """
        Generic, optimized ChatGPT request that accepts a custom system prompt.
        """
        if self.client is None:
            return "[DEFAULT] ChatGPT is not available."

        try:
            # This is now a clean, generic function. It just passes the prompts along.
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                timeout=10
            )

            return response.choices[0].message.content

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

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content},
                ],
                timeout=20
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"❌ Error in dynamic prompt request: {e}")
            return "[DEFAULT] Sorry, I encountered an error during the advanced request."
    
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