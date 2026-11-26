"""Zero-dependency RFC 6455 WebSocket framing, handshake, and broadcast hub."""

from __future__ import annotations

import base64
import hashlib
import json
import queue
import socket
import struct
import threading
from dataclasses import dataclass
from typing import Any

_WS_MAGIC_GUID = "258EAFA5-E914-47CF-95CA-C5AB0DC85B11"

OPCODE_CONTINUATION = 0x0
OPCODE_TEXT = 0x1
OPCODE_BINARY = 0x2
OPCODE_CLOSE = 0x8
OPCODE_PING = 0x9
OPCODE_PONG = 0xA

_MAX_CLIENT_FRAME_BYTES = 65_536


def compute_accept_key(sec_websocket_key: str) -> str:
    """Compute RFC 6455 Sec-WebSocket-Accept header value."""
    try:
        key_bytes = (sec_websocket_key.strip() + _WS_MAGIC_GUID).encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError("Sec-WebSocket-Key must be valid ASCII") from exc
    digest = hashlib.sha1(key_bytes).digest()
    return base64.b64encode(digest).decode("ascii")


def encode_frame(opcode: int, payload: bytes) -> bytes:
    """Encode an unmasked server-to-client RFC 6455 WebSocket frame."""
    header = bytearray()
    header.append(0x80 | (opcode & 0x0F))  # FIN=1
    length = len(payload)
    if length < 126:
        header.append(length)
    elif length < 65_536:
        header.append(126)
        header.extend(struct.pack("!H", length))
    else:
        header.append(127)
        header.extend(struct.pack("!Q", length))
    return bytes(header) + payload


def encode_text(text: str) -> bytes:
    return encode_frame(OPCODE_TEXT, text.encode("utf-8"))


def encode_pong(payload: bytes = b"") -> bytes:
    return encode_frame(OPCODE_PONG, payload[:125])


def encode_close(code: int = 1000) -> bytes:
    return encode_frame(OPCODE_CLOSE, struct.pack("!H", code))


def _recv_exact(sock: socket.socket, n: int, *, allow_initial_timeout: bool = False) -> bytes | None:
    chunks = bytearray()
    while len(chunks) < n:
        try:
            data = sock.recv(n - len(chunks))
        except TimeoutError:
            if allow_initial_timeout and len(chunks) == 0:
                raise
            return None
        except OSError:
            return None
        if not data:
            return None
        chunks.extend(data)
    return bytes(chunks)


def read_frame(sock: socket.socket, *, allow_idle_timeout: bool = False) -> tuple[int, bytes] | None:
    """Read one RFC 6455 WebSocket frame from a client socket.

    Raises TimeoutError if `allow_idle_timeout=True` and the socket times out before
    the first byte arrives. Returns None on clean EOF or socket error.
    """
    head = _recv_exact(sock, 2, allow_initial_timeout=allow_idle_timeout)
    if head is None:
        return None
    b1, b2 = head[0], head[1]
    opcode = b1 & 0x0F
    masked = bool(b2 & 0x80)
    length = b2 & 0x7F

    if length == 126:
        ext = _recv_exact(sock, 2)
        if ext is None:
            return None
        length = struct.unpack("!H", ext)[0]
    elif length == 127:
        ext = _recv_exact(sock, 8)
        if ext is None:
            return None
        length = struct.unpack("!Q", ext)[0]

    if length > _MAX_CLIENT_FRAME_BYTES:
        return (OPCODE_CLOSE, struct.pack("!H", 1009))

    mask_key = b""
    if masked:
        raw_mask = _recv_exact(sock, 4)
        if raw_mask is None:
            return None
        mask_key = raw_mask

    payload = _recv_exact(sock, length) if length > 0 else b""
    if payload is None:
        return None

    if masked and payload:
        unmasked = bytes(payload[i] ^ mask_key[i & 3] for i in range(length))
        return opcode, unmasked
    return opcode, payload


@dataclass
class _ClientSubscription:
    channel: str  # "live" or "alerts"
    send_queue: queue.Queue[bytes | None]


class WebSocketHub:
    """Thread-safe broadcast hub feeding `/ws/live` and `/ws/alerts` clients from EventBus."""

    def __init__(self, *, max_queue_size: int = 256) -> None:
        self._lock = threading.RLock()
        self._clients: list[_ClientSubscription] = []
        self._max_queue_size = max_queue_size
        self._closed = False

    def client_count(self) -> int:
        with self._lock:
            return len(self._clients)

    def register(self, channel: str) -> _ClientSubscription:
        sub = _ClientSubscription(
            channel=channel,
            send_queue=queue.Queue(maxsize=self._max_queue_size),
        )
        with self._lock:
            if self._closed:
                sub.send_queue.put_nowait(None)
            else:
                self._clients.append(sub)
        return sub

    def unregister(self, sub: _ClientSubscription) -> None:
        with self._lock:
            if sub in self._clients:
                self._clients.remove(sub)

    def publish_event(self, event: dict[str, Any]) -> None:
        """Broadcast an EventBus dictionary to matching WebSocket subscribers."""
        event_type = event.get("type")
        frame = encode_text(json.dumps(event, separators=(",", ":")))

        with self._lock:
            targets = [
                sub
                for sub in self._clients
                if sub.channel == "live" or (sub.channel == "alerts" and event_type == "alert")
            ]

        for sub in targets:
            try:
                sub.send_queue.put_nowait(frame)
            except queue.Full:
                # Drop oldest frame so a slow network client never stalls the trading loop.
                try:
                    sub.send_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    sub.send_queue.put_nowait(frame)
                except queue.Full:
                    pass

    def shutdown(self) -> None:
        """Signal all connected WebSocket handler threads to close cleanly."""
        with self._lock:
            self._closed = True
            clients = list(self._clients)
            self._clients.clear()
        for sub in clients:
            try:
                sub.send_queue.put_nowait(None)
            except queue.Full:
                try:
                    sub.send_queue.get_nowait()
                    sub.send_queue.put_nowait(None)
                except (queue.Empty, queue.Full):
                    pass
