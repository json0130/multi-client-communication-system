from openai import OpenAI
import re

class OllamaClient:
    def __init__(self, model_name="qwen2.5:7b", host="127.0.0.1", port=11434):
        self.model_name = model_name
        self.available = False
        
        self.history = []
        self.max_history_turns = 7 # Remembers the last 7 back-and-forths (14 messages total)
        
        self.client = OpenAI(
            base_url=f"http://{host}:{port}/v1",
            api_key="ollama" 
        )

    def setup_client(self) -> bool:
        """Check if Ollama is running and the model is available."""
        try:
            print("🔍 Checking connection to local Ollama via OpenAI API...")
            self.client.models.list()
            self.available = True
            print(f"✅ Connected to Ollama! Using model: {self.model_name}")
            return True
        except Exception as e:
            print(f"❌ Could not connect to Ollama: {e}")
        return False

    def is_available(self) -> bool:
        return self.available

    def ask_model_optimized(self, message: str, user_emotion: str = "neutral", confidence: float = 0.0, robot_id: str = "Robot", robot_role: str = "a companion robot", allowed_tags: str = "[DEFAULT]") -> str:
        """Send prompt with conversation history via OpenAI API format."""
        
        # Make the identity dynamic!
        system_prompt = (
            f"You are {robot_id}, {robot_role}. "
            
            "*** STRICT MANDATORY FORMATTING RULES ***\n"
            "1. THE VERY FIRST CHARACTER of your response MUST be an open bracket '['. Never start with a word, greeting, or space.\n"
            f"2. You MUST use exactly ONE emotion tag from this exact list for entire response: {allowed_tags}.\n"
            "3. ALWAYS choose the tag that best matches the emotion or physical action of the dialogue you are generating. For example, use [SAD] if expressing empathy, [SHRUG] if you don't know the answer, or [WAVE] if saying hello.\n"
            "4. Keep your spoken response to 1 or 2 sentences maximum. Be casual and conversational.\n\n"
            
            "*** EXAMPLES OF PERFECT RESPONSES ***\n"
            "[WAVE] Hello there! It's so nice to meet you.\n"
            "[CONFUSED] I'm not sure I understand what you mean.\n"
            "[SAD] I'm so sorry you are having a hard day.\n\n"
            
            "*** EXAMPLES OF INCORRECT RESPONSES (NEVER DO THIS) ***\n"
            "Hello! [WAVE] How are you? (Error: Text before the tag)\n"
            "[HAPPY] I feel great! (Error: Tag is not in the allowed list)\n"
            "Sure, I can help. [DEFAULT] Let's go. (Error: Text before the tag)\n\n"
            
            "Respond to the user's next message following these exact rules."
        )
        
        current_user_msg = {"role": "user", "content": f"[User Emotion: {user_emotion}] {message}"}
        self.history.append(current_user_msg)
        
        if len(self.history) > (self.max_history_turns * 2):
            self.history = self.history[-(self.max_history_turns * 2):]
            
        messages_payload = [{"role": "system", "content": system_prompt}] + self.history
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages_payload,
                stream=False,
                temperature=0.6
            )
            
            clean_response = response.choices[0].message.content.strip()
            self.history.append({"role": "assistant", "content": clean_response})
            
            print(f"--- DEBUG: FINAL OUTPUT ---\n{clean_response}\n---------------------------\n")
            return clean_response
            
        except Exception as e:
            if self.history:
                self.history.pop()
            print(f"❌ Failed to get response from local LLM: {e}")
            return "[SAD] Sorry, my local brain is having trouble right now."

    def ask_with_dynamic_prompt(self, full_prompt: str) -> str:
        """Sends a pre-formatted, complex prompt to the local LLM."""
        if not self.available:
            return "[DEFAULT] Local LLM is not available."

        try:
            if '\nUser: ' in full_prompt:
                parts = full_prompt.rsplit('\nUser: ', 1)
                system_content = parts[0].replace('System: ', '').strip()
                user_content = parts[1].strip()
            else:
                system_content = "You are a helpful AI assistant."
                user_content = full_prompt

            self.history.append({"role": "user", "content": user_content})

            if len(self.history) > (self.max_history_turns * 2):
                self.history = self.history[-(self.max_history_turns * 2):]

            messages_payload = [{"role": "system", "content": system_content}] + self.history

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages_payload,
                temperature=0.6, 
                timeout=30 
            )

            assistant_response = response.choices[0].message.content.strip()
            self.history.append({"role": "assistant", "content": assistant_response})

            return assistant_response

        except Exception as e:
            if self.history and self.history[-1].get("role") == "user":
                self.history.pop()
            print(f"❌ Error in dynamic prompt request via Ollama: {e}")
            return "[DEFAULT] Sorry, I encountered a brain freeze during the advanced request."

    def extract_emotion_tag(self, text: str) -> str:
        """Extract the bracketed emotion tag from the response."""
        match = re.search(r"\[([A-Z_]+)\]", text)
        if match:
            return match.group(1)
        return "DEFAULT"
    
    def clean_response_text(self, text: str) -> str:
        """Removes the bracketed [EMOTION] tags and any literal unicode emojis."""
        cleaned = re.sub(r"\[[A-Z_]+\]", "", text)
        cleaned = re.sub(r'[\U0001F600-\U0001F64F]', '', cleaned) 
        cleaned = re.sub(r'[\U0001F300-\U0001F5FF]', '', cleaned) 
        cleaned = re.sub(r'[\U0001F680-\U0001F6FF]', '', cleaned) 
        cleaned = re.sub(r'[\U0001F900-\U0001F9FF]', '', cleaned) 
        cleaned = re.sub(r'[\u2600-\u26FF]', '', cleaned)         
        cleaned = re.sub(r'[\u2700-\u27BF]', '', cleaned)         
        return cleaned.strip()
    
    def infer_topics_and_conditions(self, messages: list[str]) -> tuple[list[str], list[str]]:
        """Given a list of user messages → return (interests[], health_conditions[])."""
        if not self.available:
            return [], []

        joined = "\n".join(f"- {m}" for m in messages[:100]) 
        sys_prompt = (
            "You are an analytical system. Extract from the user's messages:\n"
            "1. up to 10 distinct long-term interests or hobbies\n"
            "2. any explicit or strongly implied health conditions.\n"
            "You MUST return ONLY valid JSON in this exact format, with no markdown, no conversational text, and no extra words:\n"
            '{"interests": ["..."], "health_conditions": ["..."]}'
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": joined}
                ],
                temperature=0.1,  
                timeout=20
            )
            
            import json
            txt = response.choices[0].message.content.strip()
            txt = re.sub(r"^```(json)?|```$", "", txt).strip()
            data = json.loads(txt)
            return data.get("interests", []), data.get("health_conditions", [])
            
        except Exception as e:
            print(f"❌ Failed to infer topics with local LLM: {e}")
            return [], []