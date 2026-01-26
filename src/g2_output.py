"""Even G2 smart glasses output module for real-time translation display."""

from __future__ import annotations

import asyncio
import json
import struct
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np

try:
    from bleak import BleakClient, BleakScanner
    BLEAK_AVAILABLE = True
except ImportError:
    BLEAK_AVAILABLE = False


@dataclass
class G2Config:
    """Configuration for Even G2 output."""

    enabled: bool = False  # Enable G2 output
    mode: str = "notification"  # "notification" or "teleprompter"
    auto_connect: bool = True  # Auto-connect on startup
    use_right: bool = False  # Use right eye (default: left)
    display_format: str = "both"  # "original", "translated", "both"


# BLE UUIDs for Even G2
UUID_BASE = "00002760-08c2-11e1-9073-0e8ac72e{:04x}"
CHAR_WRITE = UUID_BASE.format(0x5401)
CHAR_NOTIFY = UUID_BASE.format(0x5402)
CHAR_NOTIF_WRITE = UUID_BASE.format(0x7401)
CHAR_NOTIF_NOTIFY = UUID_BASE.format(0x7402)


# =============================================================================
# CRC Functions
# =============================================================================

def crc16_ccitt(data: bytes) -> int:
    """CRC-16/CCITT for packet framing."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) if crc & 0x8000 else (crc << 1)
            crc &= 0xFFFF
    return crc


def encode_varint(value: int) -> bytes:
    """Encode integer as protobuf varint."""
    result = []
    while value > 0x7F:
        result.append((value & 0x7F) | 0x80)
        value >>= 7
    result.append(value & 0x7F)
    return bytes(result)


def build_packet(seq: int, svc_hi: int, svc_lo: int, payload: bytes,
                 total_pkts: int = 1, pkt_num: int = 1) -> bytes:
    """Build a G2 protocol packet with header and CRC."""
    header = bytes([0xAA, 0x21, seq, len(payload) + 2, total_pkts, pkt_num, svc_hi, svc_lo])
    crc = crc16_ccitt(payload)
    return header + payload + bytes([crc & 0xFF, (crc >> 8) & 0xFF])


# =============================================================================
# Authentication
# =============================================================================

async def authenticate(client) -> None:
    """Send authentication sequence to G2 glasses."""
    ts = int(time.time())
    ts_var = encode_varint(ts)
    txid = bytes([0xE8] + [0xFF]*8 + [0x01])

    auth_pkts = [
        build_packet(1, 0x80, 0x00, bytes([0x08,0x04,0x10,0x0D,0x1A,0x04,0x08,0x01,0x10,0x04])),
        build_packet(2, 0x80, 0x20, bytes([0x08,0x05,0x10,0x0E,0x22,0x02,0x08,0x02])),
        build_packet(3, 0x80, 0x20, bytes([0x08,0x80,0x01,0x10,0x0F,0x82,0x08,0x11,0x08]) + ts_var + bytes([0x10]) + txid),
    ]
    for p in auth_pkts:
        await client.write_gatt_char(CHAR_WRITE, p, response=False)
        await asyncio.sleep(0.1)


# =============================================================================
# Notification Display
# =============================================================================

CRC32C_TABLE = [
    0, 0x1edc6f41, 0x3db8de82, 0x2364b1c3, 0x7db96c84, 0x656503c5, 0x46010a06, 0x58dd6547,
    0xf6b3d908, 0xe86fb649, 0xcb0bbf8a, 0xd5d7d0cb, 0x880ab58c, 0x96d6dacd, 0xb5b2d30e, 0xab6ebc4f,
    0xfac7c671, 0xe41ba930, 0xc77fa0f3, 0xd9a3cfb2, 0x847eaaf5, 0x9aa2c5b4, 0xb9c6cc77, 0xa71aa336,
    0x9741f79, 0x17a87038, 0x34cc79fb, 0x2a1016ba, 0x77cd73fd, 0x69111cbc, 0x4a75157f, 0x54a97a3e,
    0xf25f9ca3, 0xec83f3e2, 0xcfe7fa21, 0xd13b9560, 0x8ce6f027, 0x923a9f66, 0xb15e96a5, 0xaf82f9e4,
    0x1ec45ab, 0x1f3047ea, 0x3c544e29, 0x22882168, 0x7f55442f, 0x61892b6e, 0x42ed22ad, 0x5c314dec,
    0x8985ad2, 0x16445593, 0x35205c50, 0x2bfc3311, 0x76215656, 0x68fd3917, 0x4b9930d4, 0x55455f95,
    0xfb2bccda, 0xe5f7a39b, 0xc693aa58, 0xd84fc519, 0x8592a05e, 0x9b4ecf1f, 0xb82ac6dc, 0xa6f6a99d,
    0xf6453420, 0xe8995b61, 0xcbfd52a2, 0xd5213de3, 0x88fc58a4, 0x962037e5, 0xb5443e26, 0xab985167,
    0x5f6ed28, 0x1b2a8e69, 0x384e87aa, 0x2692e8eb, 0x7b4f8dac, 0x6593e2ed, 0x46f7eb2e, 0x582b846f,
    0x1ebcf2, 0x1fd7a1b3, 0x3cb3a870, 0x226fc731, 0x7fb2a276, 0x616ecd37, 0x420ac4f4, 0x5cd6abb5,
    0xf2b818fa, 0xec6477bb, 0xcf007e78, 0xd1dc1139, 0x8c01747e, 0x92dd1b3f, 0xb1b912fc, 0xaf657dbd,
    0xf4bf8e40, 0xea63e101, 0xc907e8c2, 0xd7db8783, 0x8a06e2c4, 0x94da8d85, 0xb7be8446, 0xa962eb07,
    0x70c5748, 0x19d03809, 0x3ab431ca, 0x24685e8b, 0x79b53bcc, 0x6769548d, 0x440d5d4e, 0x5ad1320f,
    0xe9e46ba, 0x104290fb, 0x33269938, 0x2dfaf679, 0x7027933e, 0x6efbfc7f, 0x4d9ff5bc, 0x53439afd,
    0xfd2d29b2, 0xe3f146f3, 0xc0954f30, 0xde492071, 0x83944536, 0x9d482a77, 0xbe2c23b4, 0xa0f04cf5,
    0xc8efb7db, 0xd633d89a, 0xf557d159, 0xeb8bbe18, 0xb656db5f, 0xa88ab41e, 0x8beebddd, 0x9532d29c,
    0x3b5c61d3, 0x25800e92, 0x6e40751, 0x18386810, 0x45e50d57, 0x5b396216, 0x785d6bd5, 0x66810494,
    0x36287109, 0x28f41e48, 0xb90178b, 0x156468ca, 0x48b90d8d, 0x566562cc, 0x75016b0f, 0x6bdd044e,
    0xc5b3b701, 0xdb6fd840, 0xf80bd183, 0xe6d7bec2, 0xbb0adb85, 0xa5d6b4c4, 0x86b2bd07, 0x986ed246,
    0x3a982b79, 0x24444438, 0x7204dfb, 0x19fc22ba, 0x442147fd, 0x5afd28bc, 0x7999217f, 0x67454e3e,
    0xc92bfd71, 0xd7f79230, 0xf4939bf3, 0xea4ff4b2, 0xb79291f5, 0xa94efeb4, 0x8a2af777, 0x94f69836,
    0xdefa7bf8, 0xc02614b9, 0xe3421d7a, 0xfd9e723b, 0xa043177c, 0xbe9f783d, 0x9dfb71fe, 0x83271ebf,
    0x2d49adf0, 0x3395c2b1, 0x10f1cb72, 0xe2da433, 0x53f08174, 0x4d2cee35, 0x6e48e7f6, 0x709488b7,
    0x203d4f88, 0x3ee120c9, 0x1d85290a, 0x359464b, 0x5e84230c, 0x40584c4d, 0x633c458e, 0x7de02acf,
    0xd38e9980, 0xcd52f6c1, 0xee36ff02, 0xf0ea9043, 0xad37f504, 0xb3eb9a45, 0x908f9386, 0x8e53fcc7,
    0x2ca73a5a, 0x327b551b, 0x111f5cd8, 0xfc33399, 0x521e56de, 0x4cc2399f, 0x6fa6305c, 0x717a5f1d,
    0xdf14ec52, 0xc1c88313, 0xe2ac8ad0, 0xfc70e591, 0xa1ad80d6, 0xbf71ef97, 0x9c15e654, 0x82c98915,
    0xd2604e0a, 0xccbc214b, 0xefd82888, 0xf10447c9, 0xacd9228e, 0xb2054dcf, 0x9161440c, 0x8fbd2b4d,
    0x21d39802, 0x3f0ff743, 0x1c6bfe80, 0x2b791c1, 0x5f6af486, 0x41b69bc7, 0x62d29204, 0x7c0efd45,
]


def calc_crc32c(data: bytes) -> int:
    """Calculate CRC32C (Castagnoli) checksum."""
    crc = 0
    for b in data:
        idx = b ^ ((crc >> 24) & 0xFF)
        crc = ((crc << 8) & 0xFFFFFFFF) ^ CRC32C_TABLE[idx]
    return crc


def calc_file_check_fields(data: bytes) -> tuple[int, int, int]:
    """Calculate file check header fields for notifications."""
    crc = calc_crc32c(data)
    return len(data) * 256, (crc << 8) & 0xFFFFFFFF, (crc >> 24) & 0xFF


def build_notification_json(title: str, message: str, subtitle: str = "") -> bytes:
    """Build notification JSON payload."""
    ts = int(time.time())
    notif = {
        "android_notification": {
            "msg_id": 10000 + (ts % 10000),
            "action": 0,
            "app_identifier": "com.live.translator",
            "title": title,
            "subtitle": subtitle,
            "message": message,
            "time_s": ts,
            "date": datetime.now().strftime("%Y%m%dT%H%M%S"),
            "display_name": "Live Translator"
        }
    }
    return json.dumps(notif, separators=(',', ':')).encode()


# =============================================================================
# G2 Output Handler
# =============================================================================

class G2Output:
    """Even G2 smart glasses output handler."""

    def __init__(self, config: G2Config):
        """Initialize G2 output.

        Args:
            config: G2 configuration.
        """
        if not BLEAK_AVAILABLE:
            raise ImportError("bleak library required for G2 output. Install with: pip install bleak")

        self.config = config
        self._left_client: Optional[BleakClient] = None
        self._right_client: Optional[BleakClient] = None
        self._connected = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._seq = 0x49  # Sequence number for packets

    async def connect(self) -> bool:
        """Connect to G2 glasses.

        Returns:
            True if connected successfully.
        """
        if self._connected:
            return True

        print("[G2] Scanning for Even G2 glasses...")
        devices = await BleakScanner.discover(timeout=10.0)

        left_dev = next((d for d in devices if d.name and "G2" in d.name and "_L_" in d.name), None)
        right_dev = next((d for d in devices if d.name and "G2" in d.name and "_R_" in d.name), None)

        if not left_dev or not right_dev:
            print("[G2] ERROR: Could not find both G2 eyes!")
            return False

        print(f"[G2] Found LEFT:  {left_dev.name}")
        print(f"[G2] Found RIGHT: {right_dev.name}")

        try:
            self._left_client = BleakClient(left_dev)
            self._right_client = BleakClient(right_dev)

            await self._left_client.connect()
            await self._right_client.connect()

            # Enable notifications
            await self._left_client.start_notify(CHAR_NOTIF_NOTIFY, lambda s, d: None)
            await self._right_client.start_notify(CHAR_NOTIF_NOTIFY, lambda s, d: None)
            await self._left_client.start_notify(CHAR_NOTIFY, lambda s, d: None)
            await self._right_client.start_notify(CHAR_NOTIFY, lambda s, d: None)

            # Authenticate
            print("[G2] Authenticating...")
            await authenticate(self._left_client)
            await authenticate(self._right_client)
            await asyncio.sleep(0.5)

            self._connected = True
            print("[G2] Connected and authenticated!")
            return True

        except Exception as e:
            print(f"[G2] Connection failed: {e}")
            if self._left_client:
                await self._left_client.disconnect()
            if self._right_client:
                await self._right_client.disconnect()
            return False

    async def disconnect(self) -> None:
        """Disconnect from G2 glasses."""
        if self._left_client:
            await self._left_client.disconnect()
        if self._right_client:
            await self._right_client.disconnect()
        self._connected = False
        print("[G2] Disconnected")

    async def send_notification(self, title: str, message: str, subtitle: str = "") -> None:
        """Send a notification to the glasses.

        Args:
            title: Notification title.
            message: Notification message.
            subtitle: Optional subtitle.
        """
        if not self._connected or not self._right_client or not self._left_client:
            return

        json_bytes = build_notification_json(title, message, subtitle)
        size, checksum, extra = calc_file_check_fields(json_bytes)

        filename = b"user/notify_whitelist.json"

        # FILE_CHECK
        fc_payload = (
            struct.pack('<I', 0x100) +
            struct.pack('<I', size) +
            struct.pack('<I', checksum) +
            bytes([extra]) +
            filename + bytes(80 - len(filename))
        )
        await self._right_client.write_gatt_char(CHAR_NOTIF_WRITE,
            build_packet(0x10, 0xC4, 0x00, fc_payload), response=False)
        await asyncio.sleep(0.2)

        # START
        await self._right_client.write_gatt_char(CHAR_NOTIF_WRITE,
            build_packet(0x49, 0xC4, 0x00, bytes([0x01])), response=False)
        await asyncio.sleep(0.05)

        # DATA chunks
        chunks = [json_bytes[i:i+234] for i in range(0, len(json_bytes), 234)]
        for i, chunk in enumerate(chunks):
            pkt = build_packet(0x49, 0xC5, 0x00, chunk, len(chunks), i+1)
            await self._right_client.write_gatt_char(CHAR_NOTIF_WRITE, pkt, response=False)
            await asyncio.sleep(0.03)
        await asyncio.sleep(0.2)

        # END
        await self._right_client.write_gatt_char(CHAR_NOTIF_WRITE,
            build_packet(0xDA, 0xC4, 0x00, bytes([0x02])), response=False)

        # Heartbeat to left eye
        await asyncio.sleep(0.1)
        await self._left_client.write_gatt_char(CHAR_WRITE,
            bytes.fromhex("aa210e0601018020080e106b6a00e174"), response=False)

    def update(self, original: str, translated: str, speaker: Optional[str] = None) -> None:
        """Update display on glasses.

        Args:
            original: Original text.
            translated: Translated text.
            speaker: Optional speaker name.
        """
        if not self._connected:
            return

        # Format message based on display_format
        if self.config.display_format == "original":
            title = speaker or "Speech"
            message = original
        elif self.config.display_format == "translated":
            title = speaker or "Translation"
            message = translated
        else:  # both
            title = speaker or "Translation"
            message = f"{original}\n→ {translated}" if translated else original

        # Schedule async send
        if self._loop and not self._loop.is_closed():
            try:
                asyncio.run_coroutine_threadsafe(
                    self.send_notification(title, message),
                    self._loop
                )
            except Exception as e:
                print(f"[G2] Error sending update: {e}")

    def start(self) -> None:
        """Start G2 output (connect in background)."""
        import threading

        def run_async_loop():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

            # Connect
            if self.config.auto_connect:
                self._loop.run_until_complete(self.connect())

            # Keep loop running
            self._loop.run_forever()

        thread = threading.Thread(target=run_async_loop, daemon=True)
        thread.start()

        # Wait for connection
        import time
        time.sleep(3.0)  # Give time to connect

    def stop(self) -> None:
        """Stop G2 output and disconnect."""
        if self._loop and not self._loop.is_closed():
            asyncio.run_coroutine_threadsafe(self.disconnect(), self._loop).result(timeout=5.0)
            self._loop.call_soon_threadsafe(self._loop.stop)

    @property
    def is_connected(self) -> bool:
        """Check if connected to glasses."""
        return self._connected
