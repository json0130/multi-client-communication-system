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
        logger.info("🖥️ Initializing console output module")
        return True
    
    def start(self) -> bool:
        """Start console output module"""
        if not self.enabled:
            self.enabled = True
            logger.info("🖥️ Console output started")
            return True
        return False
    
    def stop(self):
        """Stop console output module"""
        if self.enabled:
            self.enabled = False
            logger.info("🖥️ Console output stopped")
    
    def process_output(self, data: Any) -> bool:
        """Process and display output data"""
        if not self.enabled:
            return False
        
        try:
            # Extract text from data
            if isinstance(data, dict):
                text = data.get('text', str(data))
                response_type = data.get('type', 'unknown')
            elif isinstance(data, str):
                text = data
                response_type = 'text'
            else:
                text = str(data)
                response_type = 'other'
            
            # Truncate if needed
            if self.max_length and len(text) > self.max_length:
                text = text[:self.max_length] + "..."
            
            # Build display message
            display_parts = []
            
            if self.prefix:
                display_parts.append(self.prefix)
            
            if self.show_response_type:
                display_parts.append(f"[{response_type}]")
            
            display_parts.append(text)
            
            # Display the message
            display_message = " ".join(display_parts)
            print(display_message)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Console output error: {e}")
            return False