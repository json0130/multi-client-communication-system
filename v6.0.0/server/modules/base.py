"""
modules/base.py
===============
Abstract base class that every pluggable module must implement.

Adding a new module:
  1. Subclass BaseModule
  2. Implement initialize(), is_available(), get_status()
  3. Register it in robot/robot_instance.py

The robot layer only ever calls these three methods on a module reference —
it never knows or cares what type the module is.
"""

from __future__ import annotations
from abc import ABC, abstractmethod


class BaseModule(ABC):

    @abstractmethod
    def initialize(self) -> bool:
        """
        Set up the module (load models, open connections, etc.).
        Returns True if ready to use, False if something failed.
        Called once when the robot instance is created.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """
        Returns True if the module is initialised and ready to handle requests.
        Cheap to call — no I/O.
        """
        ...

    @abstractmethod
    def get_status(self) -> dict:
        """
        Return a plain dict describing the module's current state.
        Used by the health-check endpoint.
        """
        ...