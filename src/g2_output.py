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

def build_auth_packets() -> list:
    """Build 7-packet authentication sequence."""
    timestamp = int(time.time())
    ts_varint = encode_varint(timestamp)
    txid = bytes([0xE8, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01])

    packets = []

    # Auth 1-2: Capability exchange
    p1 = bytes([0xAA, 0x21, 0x01, 0x0C, 0x01, 0x01, 0x80, 0x00,
                0x08, 0x04, 0x10, 0x0C, 0x1A, 0x04, 0x08, 0x01, 0x10, 0x04])
    packets.append(p1 + bytes([crc16_ccitt(p1[8:]) & 0xFF, (crc16_ccitt(p1[8:]) >> 8) & 0xFF]))

    p2 = bytes([0xAA, 0x21, 0x02, 0x0A, 0x01, 0x01, 0x80, 0x20,
                0x08, 0x05, 0x10, 0x0E, 0x22, 0x02, 0x08, 0x02])
    packets.append(p2 + bytes([crc16_ccitt(p2[8:]) & 0xFF, (crc16_ccitt(p2[8:]) >> 8) & 0xFF]))

    # Auth 3: Time sync
    payload3 = bytes([0x08, 0x80, 0x01, 0x10, 0x0F, 0x82, 0x08, 0x11, 0x08]) + ts_varint + bytes([0x10]) + txid
    p3 = bytes([0xAA, 0x21, 0x03, len(payload3) + 2, 0x01, 0x01, 0x80, 0x20]) + payload3
    packets.append(p3 + bytes([crc16_ccitt(payload3) & 0xFF, (crc16_ccitt(payload3) >> 8) & 0xFF]))

    # Auth 4-6: Additional exchanges
    p4 = bytes([0xAA, 0x21, 0x04, 0x0C, 0x01, 0x01, 0x80, 0x00,
                0x08, 0x04, 0x10, 0x10, 0x1A, 0x04, 0x08, 0x01, 0x10, 0x04])
    packets.append(p4 + bytes([crc16_ccitt(p4[8:]) & 0xFF, (crc16_ccitt(p4[8:]) >> 8) & 0xFF]))

    p5 = bytes([0xAA, 0x21, 0x05, 0x0C, 0x01, 0x01, 0x80, 0x00,
                0x08, 0x04, 0x10, 0x11, 0x1A, 0x04, 0x08, 0x01, 0x10, 0x04])
    packets.append(p5 + bytes([crc16_ccitt(p5[8:]) & 0xFF, (crc16_ccitt(p5[8:]) >> 8) & 0xFF]))

    p6 = bytes([0xAA, 0x21, 0x06, 0x0A, 0x01, 0x01, 0x80, 0x20,
                0x08, 0x05, 0x10, 0x12, 0x22, 0x02, 0x08, 0x01])
    packets.append(p6 + bytes([crc16_ccitt(p6[8:]) & 0xFF, (crc16_ccitt(p6[8:]) >> 8) & 0xFF]))

    # Auth 7: Final time sync
    payload7 = bytes([0x08, 0x80, 0x01, 0x10, 0x13, 0x82, 0x08, 0x11, 0x08]) + ts_varint + bytes([0x10]) + txid
    p7 = bytes([0xAA, 0x21, 0x07, len(payload7) + 2, 0x01, 0x01, 0x80, 0x20]) + payload7
    packets.append(p7 + bytes([crc16_ccitt(payload7) & 0xFF, (crc16_ccitt(payload7) >> 8) & 0xFF]))

    return packets


async def authenticate(client) -> None:
    """Send 7-packet authentication sequence to G2 glasses."""
    for pkt in build_auth_packets():
        await client.write_gatt_char(CHAR_WRITE, pkt, response=False)
        await asyncio.sleep(0.1)


# =============================================================================
# Notification Display
# =============================================================================

# CRC32C (Castagnoli) lookup table for G2 protocol
# Polynomial: 0x1EDC6F41, Init: 0, Non-reflected (MSB-first)
CRC32C_TABLE = [
    0, 0x1edc6f41, 0x3db8de82, 0x2364b1c3,
    2071051524, 1705890373, 1187603334, 1477774535,
    4142103048, 3896448329, 3411780746, 3582446539,
    2375206668, 2471405645, 2955549070, 2935387855,
    4078607185, 3989238800, 3466741203, 3497929362,
    2288723541, 2528594196, 3050567895, 2869925782,
    0x5f9e159, 0x1b258e18, 0x38413fdb, 0x269d509a,
    2122865757, 1616130844, 1127252703, 1575808414,
    4176042467, 3862247074, 3310454625, 3683510304,
    2207835367, 2638515110, 3189783141, 2700891428,
    0xe0a23eb, 0x10d64caa, 0x33b2fd69, 0x2d6e9228,
    1971035887, 1806168494, 1220755565, 1444884268,
    0xbf3c2b2, 0x152fadf3, 0x364b1c30, 0x28977371,
    1887600566, 1851658487, 1295687988, 1407635061,
    4245731514, 3821852667, 3232261688, 3732146553,
    2254505406, 2562550527, 3151616828, 2768614525,
    4010728583, 4057117638, 3535143429, 3429526852,
    2491376003, 2325941954, 2848440065, 3072053312,
    0x19eda68f, 0x731c9ce, 0x2455780d, 0x3a89174c,
    1654397835, 2084598986, 1596245257, 1106815560,
    0x1c1447d6, 0x2c82897, 0x21ac9954, 0x3f70f615,
    1734736594, 2042205587, 1524442192, 1140935441,
    3942071774, 4096479903, 3612336988, 3381890077,
    2441511130, 2405101467, 2889768536, 3001168153,
    0x17e78564, 0x93bea25, 0x2a5f5be6, 0x348334a7,
    1821784160, 1917474593, 1362028258, 1341295011,
    3775201132, 4292382765, 3703316974, 3261091503,
    2591375976, 2225679657, 2815270122, 3104961451,
    3841793589, 4196495732, 3645227191, 3348738038,
    2676794161, 2169556080, 2721349043, 3169325810,
    0x121e643d, 0xcc20b7c, 0x2fa6babf, 0x317ad5fe,
    1768937785, 2008266360, 1423378363, 1242261754,
    3233928783, 3726489870, 4252567757, 3819267980,
    3148901195, 2775319562, 2248717769, 2564086408,
    0x3622ac47, 0x28fec306, 0xb9a72c5, 0x15461d84,
    1297289539, 1401912834, 1894502337, 1849139328,
    0x33db4d1e, 0x2d07225f, 0xe63939c, 0x10bffcdd,
    1219162138, 1450614619, 1964125848, 1808679385,
    3308795670, 3689175127, 4169197972, 3864823509,
    3192490514, 2694178131, 2213631120, 2636987345,
    0x38288fac, 0x26f4e0ed, 0x590512e, 0x1b4c3e6f,
    1129919144, 1569021417, 2128735274, 1614644075,
    3469473188, 3491207909, 4084411174, 3987686503,
    3048884384, 2875598817, 2281870882, 2531195235,
    3409057021, 3589176252, 4136290943, 3897992510,
    2957224441, 2929706680, 2382067579, 2468812858,
    0x3dd16ef5, 0x230d01b4, 0x69b077, 0x1eb5df36,
    1184945137, 1484569776, 2065173875, 1707369010,
    0x2fcf0ac8, 0x31136589, 0x1277d44a, 0xcabbb0b,
    1421785036, 1247991949, 1762027854, 2010777103,
    3643568320, 3354402689, 3834949186, 4199072003,
    2724056516, 3162612357, 2682590022, 2168028167,
    3704983961, 3255434968, 3782037275, 4289798234,
    2812554397, 3111666652, 2585588255, 2227215710,
    0x2a36eb91, 0x34ea84d0, 0x178e3513, 0x9525a52,
    1363629717, 1335572948, 1828685847, 1914955606,
    3609613099, 3388619882, 3936259497, 4098024168,
    2891443759, 2995487086, 2448371885, 2402508780,
    0x21c52923, 0x3f194662, 0x1c7df7a1, 0x2a198e0,
    1521783847, 1147730790, 1728858789, 2043684324,
    0x243cc87a, 0x3ae0a73b, 0x198416f8, 0x75879b9,
    1598911870, 1100028479, 1660267516, 2083112125,
    3537875570, 3422805299, 4016532720, 4055565233,
    2846756726, 3077726263, 2484523508, 2328542901,
]


def calc_crc32c(data: bytes) -> int:
    """Calculate CRC32C (Castagnoli) checksum using G2's MSB-first algorithm."""
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
    """Build notification JSON payload.

    Note: Uses com.google.android.gm (Gmail) as app_identifier for compatibility.
    The G2 glasses whitelist certain apps, and Gmail is confirmed working.
    Message must be <234 bytes total JSON to avoid multi-packet issues.
    """
    ts = int(time.time())
    notif = {
        "android_notification": {
            "msg_id": 10000 + (ts % 10000),
            "action": 0,
            "app_identifier": "com.google.android.gm",
            "title": title,
            "subtitle": subtitle,
            "message": message,
            "time_s": ts,
            "date": datetime.now().strftime("%Y%m%dT%H%M%S"),
            "display_name": "Gmail"
        }
    }
    return json.dumps(notif, separators=(',', ':')).encode()


# =============================================================================
# Teleprompter Display (Confirmed Working)
# =============================================================================

def build_display_config(seq: int, msg_id: int) -> bytes:
    """Service 0x0E-20: Display configuration."""
    config = bytes.fromhex(
        "0801121308021090" "4E1D00E094442500" "000000280030001213"
        "0803100D0F1D0040" "8D44250000000028" "0030001212080410"
        "001D0000884225" "00000000280030" "001212080510001D"
        "00009242250000" "A242280030001212" "080610001D0000C6"
        "42250000C4422800" "30001800"
    )
    payload = bytes([0x08, 0x02, 0x10]) + encode_varint(msg_id) + bytes([0x22, 0x6A]) + config
    return build_packet(seq, 0x0E, 0x20, payload)


def build_teleprompter_init(seq: int, msg_id: int, total_lines: int = 10) -> bytes:
    """Service 0x06-20 type=1: Initialize teleprompter (manual mode)."""
    content_height = max(1, (total_lines * 2665) // 140)

    display = (
        bytes([0x08, 0x01, 0x10, 0x00, 0x18, 0x00, 0x20, 0x8B, 0x02]) +
        bytes([0x28]) + encode_varint(content_height) +
        bytes([0x30, 0xE6, 0x01]) +
        bytes([0x38, 0x8E, 0x0A]) +
        bytes([0x40, 0x05, 0x48, 0x00])  # Font size=5, mode=manual
    )

    settings = bytes([0x08, 0x01, 0x12, len(display)]) + display
    payload = bytes([0x08, 0x01, 0x10]) + encode_varint(msg_id) + bytes([0x1A, len(settings)]) + settings
    return build_packet(seq, 0x06, 0x20, payload)


def build_content_page(seq: int, msg_id: int, page_num: int, text: str) -> bytes:
    """Service 0x06-20 type=3: Content page."""
    text_bytes = ("\n" + text).encode('utf-8')

    inner = (
        bytes([0x08]) + encode_varint(page_num) +
        bytes([0x10, 0x0A]) +
        bytes([0x1A]) + encode_varint(len(text_bytes)) + text_bytes
    )

    content = bytes([0x2A]) + encode_varint(len(inner)) + inner
    payload = bytes([0x08, 0x03, 0x10]) + encode_varint(msg_id) + content
    return build_packet(seq, 0x06, 0x20, payload)


def build_sync(seq: int, msg_id: int) -> bytes:
    """Service 0x80-00 type=14: Sync/trigger."""
    payload = bytes([0x08, 0x0E, 0x10]) + encode_varint(msg_id) + bytes([0x6A, 0x00])
    return build_packet(seq, 0x80, 0x00, payload)


# =============================================================================
# Even AI Display (Confirmed Working - Real-time Q&A display)
# =============================================================================

def build_evenai_ctrl_enter(seq: int, magic: int) -> bytes:
    """Service 0x07-20: CTRL command to enter AI mode - REQUIRED first!"""
    payload = bytes([
        0x08, 0x01,           # commandId = 1 (CTRL)
        0x10, magic & 0xFF,   # magicRandom
        0x1a, 0x02,           # ctrl field (field 3)
        0x08, 0x02            # status = 2 (EVEN_AI_ENTER)
    ])
    return build_packet(seq, 0x07, 0x20, payload)


def build_evenai_config(seq: int, magic: int, stream_speed: int = 255) -> bytes:
    """Service 0x07-20: CONFIG command - set streaming speed.

    Args:
        stream_speed: Higher = faster text appearance. 255 = instant, 32 = default slow.
    """
    config = bytes([
        0x08, 0x00,           # voiceSwitch = 0 (off)
        0x10,                 # streamSpeed field
    ]) + encode_varint(stream_speed)

    payload = bytes([
        0x08, 0x0A,           # commandId = 10 (CONFIG)
        0x10, magic & 0xFF,   # magicRandom
        0x6a,                 # config field (field 13)
    ]) + encode_varint(len(config)) + config

    return build_packet(seq, 0x07, 0x20, payload)


def build_evenai_reply(seq: int, magic: int, text: str, stream_mode: bool = True) -> bytes:
    """Service 0x07-20: REPLY command - displays answer/translation text.

    Args:
        stream_mode: If True, use streaming mode (may show text faster).
    """
    text_bytes = text.encode('utf-8')

    replyinfo = bytes([
        0x08, 0x00,           # cmdCnt = 0
        0x10, 0x01 if stream_mode else 0x00,  # streamEnable = 1 for streaming
        0x18, 0x00,           # textMode = 0
        0x22,                 # text field
    ]) + encode_varint(len(text_bytes)) + text_bytes

    payload = bytes([
        0x08, 0x05,           # commandId = 5 (REPLY)
        0x10, magic & 0xFF,   # magicRandom
        0x3a,                 # replyInfo field (field 7)
    ]) + encode_varint(len(replyinfo)) + replyinfo

    return build_packet(seq, 0x07, 0x20, payload)


def format_teleprompter_text(text: str, chars_per_line: int = 25, lines_per_page: int = 10) -> list[str]:
    """Format text into pages of wrapped lines for teleprompter."""
    text = text.replace("\\n", "\n")

    wrapped = []
    for line in text.split("\n"):
        if not line.strip():
            wrapped.append("")
            continue

        words = line.split()
        current = ""
        for word in words:
            if len(current) + len(word) + 1 > chars_per_line:
                if current:
                    wrapped.append(current.strip())
                current = word + " "
            else:
                current += word + " "
        if current.strip():
            wrapped.append(current.strip())

    if not wrapped:
        wrapped = [text]

    # Pad to at least 10 lines
    while len(wrapped) < lines_per_page:
        wrapped.append(" ")

    # Split into pages
    pages = []
    for i in range(0, len(wrapped), lines_per_page):
        page_lines = wrapped[i:i + lines_per_page]
        while len(page_lines) < lines_per_page:
            page_lines.append(" ")
        pages.append("\n".join(page_lines) + " \n")

    # Pad to minimum 14 pages
    while len(pages) < 14:
        pages.append("\n".join([" "] * lines_per_page) + " \n")

    return pages


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
        self._seq = 0x08  # Sequence number for packets
        self._msg_id = 0x14  # Message ID / magic number for packets
        self._teleprompter_initialized = False
        self._evenai_initialized = False

    async def connect(self) -> bool:
        """Connect to G2 glasses.

        For teleprompter mode: only one eye needed.
        For notification mode: both eyes needed.

        Returns:
            True if connected successfully.
        """
        if self._connected:
            return True

        print(f"[G2] Scanning for Even G2 glasses (mode: {self.config.mode})...")
        devices = await BleakScanner.discover(timeout=10.0)

        left_dev = next((d for d in devices if d.name and "G2" in d.name and "_L_" in d.name), None)
        right_dev = next((d for d in devices if d.name and "G2" in d.name and "_R_" in d.name), None)

        # Single-eye modes: teleprompter, evenai
        if self.config.mode in ("teleprompter", "evenai"):
            target_dev = right_dev if self.config.use_right else left_dev
            if not target_dev:
                print(f"[G2] ERROR: Could not find G2 {'right' if self.config.use_right else 'left'} eye!")
                return False
            print(f"[G2] Found: {target_dev.name}")

            try:
                if self.config.use_right:
                    self._right_client = BleakClient(target_dev)
                    await self._right_client.connect()
                    await self._right_client.start_notify(CHAR_NOTIFY, lambda s, d: None)
                    await authenticate(self._right_client)
                else:
                    self._left_client = BleakClient(target_dev)
                    await self._left_client.connect()
                    await self._left_client.start_notify(CHAR_NOTIFY, lambda s, d: None)
                    await authenticate(self._left_client)

                await asyncio.sleep(0.5)
                self._connected = True
                print(f"[G2] Connected and authenticated ({self.config.mode} mode)!")
                return True

            except Exception as e:
                print(f"[G2] Connection failed: {e}")
                return False

        # Notification mode needs both eyes
        if not left_dev or not right_dev:
            print("[G2] ERROR: Could not find both G2 eyes (required for notification mode)!")
            print("[G2] TIP: Try mode='teleprompter' which only needs one eye and works more reliably.")
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
            print("[G2] Connected and authenticated (notification mode)!")
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
        if self._left_client and self._left_client.is_connected:
            await self._left_client.disconnect()
        if self._right_client and self._right_client.is_connected:
            await self._right_client.disconnect()
        self._connected = False
        print("[G2] Disconnected")

    async def send_teleprompter(self, text: str) -> None:
        """Send text to glasses using teleprompter protocol.

        Full reinitialization required for each update (protocol limitation).

        Args:
            text: Text to display on glasses.
        """
        client = self._right_client if self.config.use_right else self._left_client
        if not client or not client.is_connected:
            return

        pages = format_teleprompter_text(text, lines_per_page=5)  # Fewer lines per page
        total_lines = min(15, len(text.split("\n")) + 5)

        # Display config
        await client.write_gatt_char(CHAR_WRITE, build_display_config(self._seq, self._msg_id), response=False)
        self._seq += 1
        self._msg_id += 1
        await asyncio.sleep(0.05)

        # Teleprompter init
        await client.write_gatt_char(CHAR_WRITE, build_teleprompter_init(self._seq, self._msg_id, total_lines), response=False)
        self._seq += 1
        self._msg_id += 1
        await asyncio.sleep(0.1)

        # Send first 3 content pages only (for speed)
        for i in range(min(3, len(pages))):
            await client.write_gatt_char(CHAR_WRITE, build_content_page(self._seq, self._msg_id, i, pages[i]), response=False)
            self._seq += 1
            self._msg_id += 1
            await asyncio.sleep(0.02)

        # Sync trigger
        await client.write_gatt_char(CHAR_WRITE, build_sync(self._seq, self._msg_id), response=False)
        self._seq += 1
        self._msg_id += 1

        # Wrap sequence numbers
        if self._seq > 0xFF:
            self._seq = 0x08
        if self._msg_id > 0xFF:
            self._msg_id = 0x14

    async def send_evenai(self, text: str) -> None:
        """Send text using Even AI protocol (confirmed working).

        Requires CTRL(ENTER) once to enter AI mode, then REPLY packets for updates.

        Args:
            text: Text to display.
        """
        client = self._right_client if self.config.use_right else self._left_client
        if not client or not client.is_connected:
            return

        # Enter AI mode on first use (REQUIRED!)
        if not self._evenai_initialized:
            await client.write_gatt_char(
                CHAR_WRITE,
                build_evenai_ctrl_enter(self._seq, self._msg_id),
                response=False
            )
            self._seq += 1
            self._msg_id += 1
            await asyncio.sleep(0.05)  # Minimal delay
            self._evenai_initialized = True
            print("[G2] Even AI mode entered")

        # Send REPLY packet with text
        await client.write_gatt_char(
            CHAR_WRITE,
            build_evenai_reply(self._seq, self._msg_id, text),
            response=False
        )
        self._seq += 1
        self._msg_id += 1

        # Wrap sequence numbers
        if self._seq > 0xFF:
            self._seq = 0x08
        if self._msg_id > 0xFF:
            self._msg_id = 0x14

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
        await asyncio.sleep(0.3)

        # START
        await self._right_client.write_gatt_char(CHAR_NOTIF_WRITE,
            build_packet(0x49, 0xC4, 0x00, bytes([0x01])), response=False)
        await asyncio.sleep(0.1)

        # DATA chunks
        chunks = [json_bytes[i:i+234] for i in range(0, len(json_bytes), 234)]
        for i, chunk in enumerate(chunks):
            pkt = build_packet(0x49, 0xC5, 0x00, chunk, len(chunks), i+1)
            await self._right_client.write_gatt_char(CHAR_NOTIF_WRITE, pkt, response=False)
            await asyncio.sleep(0.05)
        await asyncio.sleep(0.3)

        # END
        await self._right_client.write_gatt_char(CHAR_NOTIF_WRITE,
            build_packet(0xDA, 0xC4, 0x00, bytes([0x02])), response=False)

        # Heartbeat to left eye
        await asyncio.sleep(0.2)
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

        # Schedule async send based on mode
        if self._loop and not self._loop.is_closed():
            try:
                if self.config.mode == "evenai":
                    # Even AI mode: confirmed working, single packet updates
                    display_text = f"{message}" if message else original
                    asyncio.run_coroutine_threadsafe(
                        self.send_evenai(display_text),
                        self._loop
                    )
                elif self.config.mode == "teleprompter":
                    # Teleprompter mode: multi-line text display
                    display_text = f"{title}\n\n{message}" if title else message
                    asyncio.run_coroutine_threadsafe(
                        self.send_teleprompter(display_text),
                        self._loop
                    )
                else:
                    # Notification mode (may not work reliably)
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
