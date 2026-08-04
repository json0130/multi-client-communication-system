"""
robot/robot_registry.py
========================
Central store for all active RobotInstance objects.

Responsibilities:
  - Create a RobotInstance when the server connects to a robot
  - Initialize its modules based on the robot's DB configuration
  - Look up instances by client_id
  - Mark robots active/inactive in Supabase on connect/disconnect
  - Clean up instances that have been idle too long

The gateway layer (HTTP + WebSocket) only ever calls:
  - registry.connect(client_id)      → creates instance
  - registry.get(client_id)          → returns instance
  - registry.disconnect(client_id)   → tears down instance
"""

from __future__ import annotations
import time
import threading
from typing import Optional

from data import robot_repo, user_repo
from core.rbac import GrantStore, RBACFilter, Visibility
from robot.robot_instance import RobotInstance


# How long (seconds) before an idle instance is cleaned up
IDLE_TIMEOUT = 30 * 60   # 30 minutes


class RobotRegistry:

    def __init__(
        self,
        rbac: Optional[RBACFilter] = None,
        grants: Optional[GrantStore] = None,
        profiles=None,
    ):
        self._instances: dict[str, RobotInstance] = {}
        self._lock = threading.RLock()
        self._cleanup_running = False

        # Shared across every instance so a grant issued by one robot is visible
        # to the robot it was granted to, and all decisions land in one audit log.
        # `is not None` rather than `or`: GrantStore defines __len__, so an empty
        # store is falsy and `or` would quietly hand every robot its own store.
        self._rbac = rbac if rbac is not None else RBACFilter()
        self._grants = grants if grants is not None else GrantStore()
        # Optional ProfileRegistry — supplies default_visibility per robot.
        self._profiles = profiles

    @property
    def rbac(self) -> RBACFilter:
        return self._rbac

    @property
    def grants(self) -> GrantStore:
        return self._grants

    # ── Public API ────────────────────────────────────────────────────────────

    def connect(self, client_id: str) -> Optional[RobotInstance]:
        """
        Called when the server successfully opens a WebSocket connection to a robot.

        1. Fetches robot config from Supabase
        2. Creates a RobotInstance
        3. Initializes modules
        4. Marks robot active in DB
        5. Stores instance in registry

        Returns the instance, or None if the robot isn't registered in the DB.
        """
        with self._lock:
            # Already connected
            if client_id in self._instances:
                print(f"[Registry] {client_id} already connected, reusing instance.")
                return self._instances[client_id]

            robot = robot_repo.get_robot(client_id)
            if not robot:
                print(f"[Registry] {client_id} not found in DB — register it via the web UI first.")
                return None

            # Ensure a user row exists for this robot
            user_id = self._ensure_user(client_id, robot.robot_name)

            # RBAC identity comes from the DB row, which ProfileRegistry has
            # already reconciled against the scenario profile at boot.
            default_visibility = self._default_visibility_for(client_id)

            instance = RobotInstance(
                client_id=client_id,
                robot_name=robot.robot_name,
                user_id=user_id,
                enabled_modules=set(robot.modules),
                rbac=self._rbac,
                grants=self._grants,
                access_level=robot.access_level,
                scenario_id=robot.scenario_id,
                default_visibility=default_visibility,
            )

            ok = self._init_modules(instance, robot)
            if not ok:
                print(f"[Registry] {client_id} module init had failures — "
                      "instance still created with available modules.")

            self._instances[client_id] = instance
            robot_repo.set_active(client_id, True)
            print(f"[Registry] {client_id} connected. "
                  f"Modules: {list(instance.enabled_modules)}")
            return instance

    def get(self, client_id: str) -> Optional[RobotInstance]:
        """Return an existing instance, or None if not connected."""
        with self._lock:
            return self._instances.get(client_id)

    def get_all(self, exclude_id: Optional[str] = None) -> list[RobotInstance]:
        """Return all active instances, optionally excluding one."""
        with self._lock:
            return [
                inst for cid, inst in self._instances.items()
                if cid != exclude_id
            ]

    def disconnect(self, client_id: str):
        """
        Called when the WebSocket connection to a robot closes.
        Marks robot inactive in DB and removes the instance.
        """
        with self._lock:
            instance = self._instances.pop(client_id, None)
            if instance:
                print(f"[Registry] {client_id} disconnected.")
            robot_repo.set_active(client_id, False)

    def is_connected(self, client_id: str) -> bool:
        with self._lock:
            return client_id in self._instances

    # ── Cleanup task ──────────────────────────────────────────────────────────

    def start_cleanup_task(self):
        """Start background thread that removes idle instances every 5 minutes."""
        if self._cleanup_running:
            return
        self._cleanup_running = True

        def _worker():
            while self._cleanup_running:
                time.sleep(300)
                self._cleanup_idle()

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        print("[Registry] Cleanup task started.")

    def stop_cleanup_task(self):
        self._cleanup_running = False

    def shutdown(self):
        """Mark all robots inactive and clear the registry."""
        with self._lock:
            for client_id in list(self._instances.keys()):
                robot_repo.set_active(client_id, False)
            self._instances.clear()
        print("[Registry] Shutdown complete.")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _default_visibility_for(self, client_id: str) -> str:
        """
        Visibility stamped on records this robot writes.

        Comes from the scenario profile when one is loaded; otherwise 'local',
        so a deployment without a profile keeps every robot isolated.
        """
        if self._profiles is not None:
            try:
                entry = self._profiles.get_robot(client_id)
                if entry is not None:
                    return entry.default_visibility
            except Exception as e:
                print(f"[Registry] profile lookup failed for {client_id}: {e}")
        return Visibility.LOCAL.value

    def _ensure_user(self, client_id: str, robot_name: str) -> int:
        """
        Return the user_id for this robot.
        Creates a new user row if one doesn't exist yet.
        Uses a simple naming convention to avoid duplicate users.
        """
        # Check if we already have a user by looking at recent robots
        # For simplicity, we create a user each session if not cached.
        # In production you'd store user_id on the robots table.
        try:
            user_id = user_repo.create_user(name=robot_name)
            return user_id
        except Exception as e:
            print(f"[Registry] Could not create user for {client_id}: {e}")
            return 0

    def _init_modules(self, instance: RobotInstance, robot) -> bool:
        """
        Initialize each enabled module and attach it to the instance.
        Continues even if individual modules fail.
        Returns True if all enabled modules initialized successfully.
        """
        all_ok = True
        modules = instance.enabled_modules

        if "gpt" in modules or "llm" in modules:
            try:
                from modules.llm.llm_module import LLMModule
                mod = LLMModule()
                if mod.initialize():
                    instance.llm = mod
                    print(f"  [Registry] {instance.client_id} — LLM ready "
                          f"({mod.provider_name})")
                else:
                    print(f"  [Registry] {instance.client_id} — LLM init failed")
                    all_ok = False
            except Exception as e:
                print(f"  [Registry] LLM error: {e}")
                all_ok = False

        if "speech" in modules:
            try:
                from modules.speech.speech_module import SpeechModule
                mod = SpeechModule()
                if mod.initialize():
                    instance.speech = mod
                    print(f"  [Registry] {instance.client_id} — Speech ready")
                else:
                    print(f"  [Registry] {instance.client_id} — Speech init failed")
                    all_ok = False
            except Exception as e:
                print(f"  [Registry] Speech error: {e}")
                all_ok = False

        if "emotion" in modules:
            try:
                from modules.emotion.emotion_module import EmotionModule
                mod = EmotionModule()
                if mod.initialize():
                    instance.emotion = mod
                    print(f"  [Registry] {instance.client_id} — Emotion ready")
                else:
                    print(f"  [Registry] {instance.client_id} — Emotion init failed")
                    all_ok = False
            except Exception as e:
                print(f"  [Registry] Emotion error: {e}")
                all_ok = False

        if "rag" in modules:
            try:
                from modules.rag.rag_module import RagModule
                # Provenance is threaded in so records this robot writes are
                # attributable, and so a legacy v1 sidecar can be backfilled
                # with the robot that owns the index.
                mod = RagModule(
                    user_id=instance.user_id,
                    client_id=instance.client_id,
                    scenario_id=getattr(robot, "scenario_id", None),
                    session_id=instance.identity.session_id,
                    default_visibility=self._default_visibility_for(instance.client_id),
                )
                if mod.initialize():
                    instance.rag = mod
                    print(f"  [Registry] {instance.client_id} — RAG ready")
                else:
                    print(f"  [Registry] {instance.client_id} — RAG init failed")
                    all_ok = False
            except Exception as e:
                print(f"  [Registry] RAG error: {e}")
                all_ok = False

        return all_ok

    def _cleanup_idle(self):
        """Remove instances that haven't been active for IDLE_TIMEOUT seconds."""
        now = time.time()
        with self._lock:
            idle = [
                cid for cid, inst in self._instances.items()
                if now - inst.last_active > IDLE_TIMEOUT
            ]
            for cid in idle:
                print(f"[Registry] Cleaning up idle instance: {cid}")
                self._instances.pop(cid, None)
                robot_repo.set_active(cid, False)