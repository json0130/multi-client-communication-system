# modules/output/console_output.py - Console text output module
import logging
from typing import Dict, Any
from client import OutputModule

logger = logging.getLogger(__name__)

class ConsoleOutputModule(OutputModule):
    """Simple console output module for displaying text responses"""
    
    def __init__(self, name: str = "console_output", config: Dict = None):
        super().__init__(name, config)
        
        # Configuration
        self.show_timestamps = self.config.get('show_timestamps', True)
        self.show_response_type = self.config.get('show_response_type', True)
        self.prefix = self.config.get('prefix', '🤖')
        self.max_length = self.config.get('max_length', None)  # Truncate long responses
    
    def initialize(self) -> bool:
        """Initialize console output module"""
        # logger.info("🖥️ Initializing console output module")
        return True
    
    def start(self) -> bool:
        """Start console output module"""
        if not self.enabled:
            self.enabled = True
            # logger.info("🖥️ Console output started")
            return True
        return False
    
    def stop(self):
        """Stop console output module"""
        if self.enabled:
            self.enabled = False
            # logger.info("🖥️ Console output stopped")
    
    def process_output(self, data: Any) -> bool:
        """Process output with debugging info"""
        if not self.enabled:
            return False
    
        try:
        
            # Extract text for display
            if isinstance(data, dict):
                text = data.get('text', data.get('response', data.get('content', str(data))))
            else:
                text = str(data)
        
            # Clean text for display
            display_text = text
            if hasattr(self, 'remove_emotion_tags') and self.config.get('remove_emotion_tags', True):
                display_text = re.sub(r"^\[(.*?)\]\s*", "", display_text)
        
            # Display the text
            timestamp = ""
            if self.config.get('show_timestamps', False):
                from datetime import datetime
                timestamp = f"[{datetime.now().strftime('%H:%M:%S')}] "
        
            response_type = ""
            if self.config.get('show_response_type', False) and isinstance(data, dict):
                resp_type = data.get('type', data.get('input_type', 'unknown'))
                response_type = f"[{resp_type}] "
        
            prefix = self.config.get('prefix', '🤖')
        
            # print(f"{prefix} {timestamp}{response_type}{display_text}")
        
            return True
        
        except Exception as e:
            print(f"❌ Console output error: {e}")
            return False
