"""
OutputModules/console_output.py
=================================
Prints server responses to stdout. Useful for debugging without audio hardware.
"""

import re
import logging
from typing import Any, Dict

from client import OutputModule

logger = logging.getLogger(__name__)


class ConsoleOutputModule(OutputModule):

    def __init__(self, name: str = "console_output", config: Dict = None):
        super().__init__(name, config)
        self._prefix = self.config.get('prefix', '[ChatBox]')

    def initialize(self) -> bool:
        return True

    def start(self) -> bool:
        if not self.enabled:
            self.enabled = True
            return True
        return False

    def stop(self):
        self.enabled = False

    def process_output(self, data: Any) -> bool:
        if not self.enabled:
            return False
        try:
            if isinstance(data, dict):
                text = data.get('text', data.get('response', str(data)))
            else:
                text = str(data)
            # Strip emotion tags for clean display
            display = re.sub(r'\[.*?\]', '', text).strip()
            if display:
                print(f"{self._prefix} {display}", flush=True)
            return True
        except Exception as e:
            logger.error(f"[Console] Output error: {e}")
            return False
