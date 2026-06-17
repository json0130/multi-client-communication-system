# OutputModules/console_output.py - Console text output module
import logging
from typing import Dict, Any
from client import OutputModule

logger = logging.getLogger(__name__)


class ConsoleOutputModule(OutputModule):
    """Displays robot responses to the terminal."""

    def __init__(self, name: str = "console_output", config: Dict = None):
        super().__init__(name, config)
        self.prefix = self.config.get('prefix', 'iRobi')

    def initialize(self) -> bool:
        return True

    def start(self) -> bool:
        if not self.enabled:
            self.enabled = True
            return True
        return False

    def stop(self):
        if self.enabled:
            self.enabled = False

    def process_output(self, data: Any) -> bool:
        if not self.enabled:
            return False
        try:
            if isinstance(data, dict):
                text = data.get('text', data.get('response', data.get('content', str(data))))
            else:
                text = str(data)

            if self.config.get('show_timestamps', False):
                from datetime import datetime
                ts = f"[{datetime.now().strftime('%H:%M:%S')}] "
            else:
                ts = ""

            print(f"\n{self.prefix}: {ts}{text}")
            return True
        except Exception as e:
            print(f"Console output error: {e}")
            return False
