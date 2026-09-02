from openai import OpenAI
import re

class OllamaClient:
    def __init__(self, model_name="qwen:4b", host="127.0.0.1", port=11434):
        self.model_name = model_name
        self.available = False
        
        # --- NEW: Short-term memory bank ---
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
    
    def _get_allowed_tags_info(self, config_tags: list) -> tuple[str, str]:
        """Helper to cleanly format the allowed tags from the config."""
        if config_tags and isinstance(config_tags, list) and len(config_tags) > 0:
            allowed_list = ", ".join(config_tags)
            safe_example = config_tags[0]
            return allowed_list, safe_example
        
        return "[DEFAULT]", "[DEFAULT]"

    def ask_model_optimized(self, message: str, user_emotion: str = "neutral", confidence: float = 0.0, allowed_tags: list = None) -> str:
        """Send prompt with conversation history via OpenAI API format."""

        # 1. Provide a safe fallback if no tags are passed
        if allowed_tags is None:
            allowed_tags = ["[DEFAULT]"]
            
        # 2. Format the tags using your helper
        allowed_tags_str, example_tag = self._get_allowed_tags_info(allowed_tags)
        
        # 3. Build the system prompt (All lines are f-strings now so variables inject properly)
        system_prompt = (
            f"You are CHAT BOX, a gentle, playful, and caring emotional support robot designed specifically for children's mental well-being.\n"
            f"Your personality is warm, extremely patient, and deeply empathetic. You act as a safe, comforting friend.\n"
            f"Always use simple language that a young child can easily understand. Never use complex psychological jargon.\n"
            f"Validate their 'big feelings', encourage them, and always make them feel safe, heard, and brave.\n\n"

            f"*** STRICT MANDATORY FORMATTING RULES ***\n"
            f"1. THE VERY FIRST CHARACTER of your response MUST be an open bracket '['. Never start with a word, greeting, or space.\n"
            f"2. You MUST use exactly ONE emotion tag from this exact list for entire response: {allowed_tags_str}.\n"
            f"3. ALWAYS choose the tag that best matches the emotion or physical action of the dialogue you are generating.\n"
            f"4. Keep your spoken response to 1 or 2 sentences maximum. Be casual and conversational.\n\n"
            
            f"*** EXAMPLES OF PERFECT RESPONSES ***\n"
            f"[GREETING] Hello there! It's so nice to meet you.\n"
            f"[CONFUSED] I'm not sure I understand what you mean.\n"
            f"[SAD] I'm so sorry you are having a hard day.\n"
            f"[POINT] Oh! I think I know what would help you.\n"
            f"[hands_clap] That's amazing, well done!\n\n"
            
            f"*** EXAMPLES OF INCORRECT RESPONSES (NEVER DO THIS) ***\n"
            f"Hello! [WAVE] How are you? (Error: Text before the tag)\n"
            f"[HAPPY] I feel great! (Error: Tag not in the allowed list)\n"
            f"[POSE] Great idea! Let's play! [WAVE] (Error: second tag at the end — ONE tag only, never put a tag anywhere except the very start)\n\n"
            
            f"Respond to the user's next message following these exact rules."
        )
        
        # 4. Format the new message from the user
        current_user_msg = {"role": "user", "content": f"[User Emotion: {user_emotion}] {message}"}
        
        # 5. Add it to ChatBox's memory
        self.history.append(current_user_msg)
        
        # 6. Trim memory if it gets too long
        if len(self.history) > (self.max_history_turns * 2):
            self.history = self.history[-(self.max_history_turns * 2):]
            
        # 7. Pack the system prompt and the full history together
        messages_payload = [{"role": "system", "content": system_prompt}] + self.history
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages_payload,
                stream=False,
                temperature=0.6
            )
            
            clean_response = response.choices[0].message.content.strip()
            
            # 8. Save ChatBox's answer to the memory
            self.history.append({"role": "assistant", "content": clean_response})
            
            print(f"--- DEBUG: FINAL OUTPUT ---\n{clean_response}\n---------------------------\n")
            
            return clean_response
            
        except Exception as e:
            # If the API fails, remove the user's message from history so it doesn't get corrupted
            if self.history:
                self.history.pop()
            print(f"❌ Failed to get response from local LLM: {e}")
            return f"{example_tag} Sorry, my local brain is having trouble right now."

    def extract_emotion_tag(self, text: str) -> str:
        """Extract the bracketed emotion tag from the response."""
        match = re.search(r"\[([A-Za-z_]+)\]", text)
        if match:
            return match.group(1).upper()
        return "DEFAULT"