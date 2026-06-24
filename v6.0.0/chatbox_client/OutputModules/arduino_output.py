"""
OutputModules/arduino_output.py
================================
TCP/Serial output module for the ChatBox ESP32 (ChatBoxPlus_ESP32.ino).

Transport priority: Serial (USB, if detected) → TCP/WiFi
Reconnect: exponential backoff on both transports.

Valid ESP32 commands (22):
  greeting, wave, point, confused, shrug, angry, sad, sleep, default, pose,
  idle, dance_sway, dance_arms, dance_groove, hands_clap, hands_wave_both,
  head_nod, head_shake, ears_wiggle, ears_perk, hands_only_wave, hands_only_tap

Protocol (TCP + Serial):
  Client → ESP32 : "{command}\n"
  ESP32 → Client : "OK:{command}", "DONE:{command}", "ERR:{command}"
"""

import glob
import logging
import socket
import struct
import subprocess
import threading
import time
from typing import Any, Callable, Dict, Optional

from client import OutputModule

logger = logging.getLogger(__name__)

try:
    import serial as _serial_mod
    _SERIAL_AVAILABLE = True
except ImportError:
    _serial_mod = None
    _SERIAL_AVAILABLE = False
    logger.debug("[Arduino] pyserial not installed — serial transport unavailable")


class ArduinoOutputModule(OutputModule):
    """
    Sends text commands to the ChatBox ESP32 over TCP (WiFi) or Serial (USB).

    Config keys (all optional):
      host                 : str   mDNS name or IP (default "chatbox.local")
      port                 : int   TCP port (default 8888)
      bind_address         : str   local IP to bind (for Docker --network host)
      reconnect_delay      : float initial retry delay in seconds (default 1.0)
      max_reconnect_delay  : float max retry delay in seconds (default 30.0)
      serial.port          : str   hint for serial port (e.g. "/dev/ttyUSB0")
      serial.baudrate      : int   (default 115200)
      serial.timeout       : float (default 1.0)
    """

    def __init__(self, name: str = "arduino_output", config: Dict[str, Any] = None):
        super().__init__(name, config)

        self.host              = self.config.get('host', 'chatbox.local')
        self.port              = self.config.get('port', 8888)
        self._bind_address     = self.config.get('bind_address')
        self._reconnect_delay  = self.config.get('reconnect_delay', 1.0)
        self._max_delay        = self.config.get('max_reconnect_delay', 30.0)
        self._serial_cfg: Dict = self.config.get('serial', {})

        self._sock: Optional[socket.socket] = None
        self._serial                         = None
        self._mode: str                      = "tcp"
        self.connected                       = False
        self._running                        = False
        self._resolved_ip: Optional[str]     = None
        self._serial_port: Optional[str]     = None
        self.last_esp32_message              = ""

        # Optional callbacks set by ChatBoxClient
        self.on_connected:        Optional[Callable]       = None
        self.on_disconnected:     Optional[Callable]       = None
        self.on_connection_error: Optional[Callable[[str], None]] = None

    # ── BaseModule interface ──────────────────────────────────────────────────

    def initialize(self) -> bool:
        return True

    def process_output(self, data: Any) -> bool:
        return True

    def is_connected(self) -> bool:
        return self.connected

    def start(self) -> bool:
        self._running = True
        self._serial_port = self._detect_serial()

        if self._serial_port:
            self._mode = "serial"
            logger.info(f"[Arduino] USB serial detected ({self._serial_port}) — serial mode")
            threading.Thread(target=self._serial_loop, daemon=True, name="arduino-serial").start()
        else:
            self._mode = "tcp"
            logger.info(f"[Arduino] No serial device — TCP/WiFi mode ({self.host}:{self.port})")
            threading.Thread(target=self._tcp_loop, daemon=True, name="arduino-tcp").start()
        return True

    def stop(self):
        self._running = False
        if self._mode == "serial" and self._serial:
            try:
                self._serial.close()
            except Exception:
                pass
            self._serial = None
        elif self._mode == "tcp" and self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
        self.connected = False
        logger.info("[Arduino] Stopped")

    # ── Command send ──────────────────────────────────────────────────────────

    def send_command(self, command: str) -> bool:
        if not self.connected:
            logger.warning(f"[Arduino] Cannot send '{command}' — not connected")
            return False
        payload = f"{command.strip()}\n".encode('utf-8')
        try:
            if self._mode == "serial":
                self._serial.write(payload)
                self._serial.flush()
            else:
                self._sock.sendall(payload)
            logger.info(f"[Arduino] → {command.strip()}")
            return True
        except Exception as e:
            logger.error(f"[Arduino] Send failed: {e}")
            self.connected = False
            return False

    # ── Serial detection ──────────────────────────────────────────────────────

    def _detect_serial(self) -> Optional[str]:
        if not _SERIAL_AVAILABLE:
            return None
        candidates = []
        hint = self._serial_cfg.get("port", "")
        if hint:
            candidates.append(hint)
        candidates += sorted(glob.glob("/dev/ttyUSB*")) + sorted(glob.glob("/dev/ttyACM*"))
        for port in candidates:
            try:
                s = _serial_mod.Serial(port, timeout=0.5)
                s.close()
                return port
            except Exception:
                continue
        return None

    # ── TCP helpers ───────────────────────────────────────────────────────────

    def _resolve_host(self) -> str:
        if self._resolved_ip:
            return self._resolved_ip
        if self.host.endswith('.local'):
            ip = self._mdns_resolve(self.host)
            if ip:
                self._resolved_ip = ip
                return ip
            try:
                result = subprocess.run(
                    ['avahi-resolve-host-name', self.host],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    parts = result.stdout.strip().split()
                    if len(parts) >= 2:
                        self._resolved_ip = parts[1]
                        logger.info(f"[Arduino] Resolved {self.host} → {self._resolved_ip} (avahi)")
                        return self._resolved_ip
            except Exception:
                pass
        return self.host

    def _mdns_resolve(self, hostname: str, timeout: float = 3.0) -> Optional[str]:
        labels = hostname.rstrip('.').encode('ascii').split(b'.')
        qname  = b''.join(bytes([len(l)]) + l for l in labels) + b'\x00'
        packet = struct.pack('!HHHHHH', 0, 0, 1, 0, 0, 0) + qname + struct.pack('!HH', 1, 1)

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 255)
        sock.settimeout(1.0)
        if self._bind_address:
            try:
                sock.bind((self._bind_address, 0))
                sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF,
                                socket.inet_aton(self._bind_address))
            except OSError as e:
                logger.warning(f"[Arduino] Cannot bind mDNS to {self._bind_address}: {e}")
        try:
            sock.sendto(packet, ('224.0.0.251', 5353))
            deadline = time.time() + timeout
            while time.time() < deadline:
                try:
                    data, _ = sock.recvfrom(4096)
                except socket.timeout:
                    continue
                if len(data) < 12:
                    continue
                flags = struct.unpack('!H', data[2:4])[0]
                if not (flags & 0x8000):
                    continue
                qdcount = struct.unpack('!H', data[4:6])[0]
                ancount = struct.unpack('!H', data[6:8])[0]
                if ancount == 0:
                    continue
                off = 12
                for _ in range(qdcount):
                    while off < len(data):
                        n = data[off]
                        if n == 0:    off += 1; break
                        if n >= 0xC0: off += 2; break
                        off += 1 + n
                    off += 4
                for _ in range(ancount):
                    while off < len(data):
                        n = data[off]
                        if n == 0:    off += 1; break
                        if n >= 0xC0: off += 2; break
                        off += 1 + n
                    if off + 10 > len(data):
                        break
                    rtype, _, _, rdlen = struct.unpack('!HHIH', data[off:off + 10])
                    off += 10
                    if rtype == 1 and rdlen == 4:
                        ip = '.'.join(str(b) for b in data[off:off + 4])
                        logger.info(f"[Arduino] mDNS: {hostname} → {ip}")
                        return ip
                    off += rdlen
        except Exception as e:
            logger.debug(f"[Arduino] mDNS error: {e}")
        finally:
            sock.close()
        return None

    def _tcp_connect(self) -> bool:
        host = self._resolve_host()
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            if self._bind_address:
                try:
                    s.bind((self._bind_address, 0))
                except OSError as e:
                    logger.warning(f"[Arduino] Cannot bind TCP to {self._bind_address}: {e}")
            s.settimeout(5.0)
            s.connect((host, self.port))
            s.settimeout(1.0)
            self._sock     = s
            self.connected = True
            logger.info(f"[Arduino] TCP connected to {host}:{self.port}")
            if self.on_connected:
                self.on_connected()
            return True
        except socket.gaierror:
            self._resolved_ip = None
            logger.error(f"[Arduino] Cannot resolve '{self.host}'")
            if self.on_connection_error:
                self.on_connection_error(f"Cannot resolve {self.host}")
            return False
        except Exception as e:
            logger.error(f"[Arduino] TCP connect failed: {e}")
            if self.on_connection_error:
                self.on_connection_error(str(e))
            return False

    def _serial_connect(self) -> bool:
        baudrate = self._serial_cfg.get("baudrate", 115200)
        timeout  = self._serial_cfg.get("timeout", 1.0)
        try:
            self._serial = _serial_mod.Serial(
                self._serial_port, baudrate=baudrate,
                timeout=timeout, write_timeout=1.0
            )
            self.connected = True
            logger.info(f"[Arduino] Serial connected to {self._serial_port} @ {baudrate}")
            if self.on_connected:
                self.on_connected()
            return True
        except Exception as e:
            logger.error(f"[Arduino] Serial connect failed: {e}")
            if self.on_connection_error:
                self.on_connection_error(str(e))
            return False

    # ── Reconnect loops ───────────────────────────────────────────────────────

    def _tcp_loop(self):
        delay = self._reconnect_delay
        first = True
        while self._running:
            if not self.connected:
                if not first:
                    logger.info(f"[Arduino] TCP reconnecting in {delay:.0f}s...")
                    time.sleep(delay)
                first = False
                if self._tcp_connect():
                    delay = self._reconnect_delay
                else:
                    delay = min(delay * 2, self._max_delay)
                continue
            try:
                data = self._sock.recv(256)
                if not data:
                    raise ConnectionResetError("EOF")
                for line in data.decode('utf-8', errors='ignore').splitlines():
                    line = line.strip()
                    if line:
                        self.last_esp32_message = line
                        logger.debug(f"[Arduino] ESP32: {line}")
            except socket.timeout:
                pass
            except Exception as e:
                if self._running:
                    logger.error(f"[Arduino] TCP lost: {e}")
                    self.connected = False
                    self._resolved_ip = None
                    try:
                        self._sock.close()
                    except Exception:
                        pass
                    self._sock = None
                    if self.on_disconnected:
                        self.on_disconnected()

    def _serial_loop(self):
        delay = self._reconnect_delay
        first = True
        while self._running:
            if not self.connected:
                if not first:
                    logger.info(f"[Arduino] Serial reconnecting in {delay:.0f}s...")
                    time.sleep(delay)
                first = False
                if self._serial_connect():
                    delay = self._reconnect_delay
                else:
                    delay = min(delay * 2, self._max_delay)
                continue
            try:
                line = self._serial.readline()
                if line:
                    text = line.decode('utf-8', errors='ignore').strip()
                    if text:
                        self.last_esp32_message = text
                        logger.debug(f"[Arduino] ESP32: {text}")
            except Exception as e:
                if self._running:
                    logger.error(f"[Arduino] Serial lost: {e}")
                    self.connected = False
                    try:
                        self._serial.close()
                    except Exception:
                        pass
                    self._serial = None
                    if self.on_disconnected:
                        self.on_disconnected()
