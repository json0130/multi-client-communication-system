# inputModules/text_input.py - Text input module
import threading
import logging
from typing import Optional, Dict
from client import InputModule

logger = logging.getLogger(__name__)

class TextInputModule(InputModule):
    """Simple text input module for interactive chat"""
    
    def __init__(self, name: str = "text_input", config: Dict = None):
        super().__init__(name, config)
        self.input_thread = None
        self.stop_event = threading.Event()
    
    def initialize(self) -> bool:
        """Initialize text input module"""
        logger.info("🔤 Initializing text input module")
        return True
    
    def start(self) -> bool:
        """Start text input thread"""
        if not self.enabled:
            self.enabled = True
            self.stop_event.clear()
            self.input_thread = threading.Thread(target=self._input_loop, daemon=True)
            self.input_thread.start()
            logger.info("🔤 Text input started - type messages and press Enter")
            return True
        return False
    
    def stop(self):
        """Stop text input"""
        if self.enabled:
            self.enabled = False
            self.stop_event.set()
            if self.input_thread:
                self.input_thread.join(timeout=1)
            logger.info("🔤 Text input stopped")
    
    def get_data(self) -> Optional[str]:
        """Get text input from user"""
        try:
            return input("💬 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
    
    def _input_loop(self):
        """Main input loop for text"""
        logger.info("💬 Text input ready. Type 'exit' to quit.")
        
        while not self.stop_event.is_set() and self.enabled:
            try:
                user_input = self.get_data()
                
                if user_input is None:
                    break
                
                if user_input.lower() in ['exit', 'quit', 'stop']:
                    logger.info("🛑 Exit command received")
                    if self.client:
                        self.client.running = False
                    break
                
                if user_input:
                    # Send to server via client
                    if self.client:
                        response = self.client.send_to_server('chat', user_input)
                        self.client.process_server_response(response, 'chat')
                
            except Exception as e:
                logger.error(f"❌ Text input error: {e}")
                break