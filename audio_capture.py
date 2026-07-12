"""
App audio capture via BlackHole with optional real-time playback.
48 kHz, low-latency pass-through when playback is enabled.
"""
from __future__ import annotations

import queue
import threading
from typing import Callable

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 48000
BLOCK_SIZE = 512  # ~10.7 ms at 48 kHz for low latency
CHANNELS = 2
DTYPE = np.float32

# Known audio-capable app names (process names on macOS)
AUDIO_APP_NAMES = frozenset({
    "Zoom", "zoom", "FaceTime", "Google Chrome", "Chromium",
    "Safari", "Microsoft Teams", "Slack", "Discord", "Spotify",
    "Music", "QuickTime Player", "VLC", "Meet", "Brave Browser",
    "Firefox", "Arc", "Electron",  # Electron can be Meet/Spotify etc.
})


def get_running_audio_apps() -> list[str]:
    """Return list of running application names (audio-capable apps first)."""
    import subprocess
    try:
        result = subprocess.run(
            ["osascript", "-e", (
                'tell application "System Events" to get name of every process '
                'whose background only is false'
            )],
            capture_output=True,
            text=True,
            timeout=5,
            env={**__import__("os").environ, "LANG": "en_US.UTF-8"},
        )
        if result.returncode != 0 or not result.stdout:
            return []
        names = [n.strip() for n in result.stdout.split(",") if n.strip()]
        seen = set()
        audio_first = []
        others = []
        for n in names:
            if n in seen:
                continue
            seen.add(n)
            if n in AUDIO_APP_NAMES or _looks_like_audio_app(n):
                audio_first.append(n)
            else:
                others.append(n)
        return sorted(audio_first) + sorted(others)
    except Exception:
        return []


def _looks_like_audio_app(name: str) -> bool:
    """Heuristic: include common meeting/browser apps."""
    lower = name.lower()
    if "meet" in lower or "chrome" in lower or "zoom" in lower:
        return True
    if "face" in lower and "time" in lower:
        return True
    return False


def find_blackhole_input_device() -> int | None:
    """Return sounddevice device index for BlackHole input, or None."""
    for i, dev in enumerate(sd.query_devices()):
        name = (dev.get("name") or "").strip()
        if "BlackHole" in name and dev.get("max_input_channels", 0) >= 1:
            return i
    return None


def get_default_output_device() -> int | None:
    """Return device index for default output (speakers)."""
    try:
        return sd.query_devices(kind="output")["index"]
    except Exception:
        return None


class RingBuffer:
    """Single producer, single consumer float32 ring buffer for audio blocks."""

    def __init__(self, num_blocks: int = 32, block_samples: int = BLOCK_SIZE * CHANNELS):
        self._block_samples = block_samples
        self._buf = np.zeros((num_blocks, block_samples), dtype=DTYPE)
        self._write_idx = 0
        self._read_idx = 0
        self._count = 0
        self._lock = threading.Lock()

    def write(self, block: np.ndarray) -> bool:
        """Write one block. Drops oldest if full. Returns True if written."""
        with self._lock:
            if self._count >= len(self._buf):
                self._read_idx = (self._read_idx + 1) % len(self._buf)
                self._count -= 1
            n = min(block.size, self._block_samples)
            self._buf[self._write_idx, :n] = block.flat[:n]
            if n < self._block_samples:
                self._buf[self._write_idx, n:] = 0
            self._write_idx = (self._write_idx + 1) % len(self._buf)
            self._count += 1
            return True

    def read(self, out: np.ndarray) -> bool:
        """Read one block into out. Returns True if data was available."""
        with self._lock:
            if self._count == 0:
                out.fill(0)
                return False
            out_flat = out.ravel()
            n = min(self._block_samples, out_flat.size)
            out_flat[:n] = self._buf[self._read_idx, :n]
            if n < out_flat.size:
                out_flat[n:] = 0
            self._read_idx = (self._read_idx + 1) % len(self._buf)
            self._count -= 1
            return True


class AppAudioCapture:
    """Captures from BlackHole; optionally plays back to default output."""

    def __init__(
        self,
        on_level: Callable[[float], None] | None = None,
        on_status: Callable[[str], None] | None = None,
        transcription_queue: queue.Queue[np.ndarray] | None = None,
    ):
        self._on_level = on_level
        self._on_status = on_status
        self._transcription_queue = transcription_queue
        self._input_stream: sd.InputStream | None = None
        self._output_stream: sd.OutputStream | None = None
        self._ring: RingBuffer | None = None
        self._playback_enabled = False
        self._muted = False
        self._running = False
        self._input_device: int | None = None
        self._output_device: int | None = None
        self._lock = threading.Lock()

    def _report_status(self, msg: str) -> None:
        if self._on_status:
            self._on_status(msg)

    def _report_level(self, rms: float) -> None:
        if self._on_level:
            self._on_level(rms)

    def _input_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            self._report_status(f"Input: {status}")
        block = indata.copy()
        if self._muted:
            block.fill(0)
        rms = float(np.sqrt(np.mean(block ** 2))) if block.size else 0.0
        self._report_level(rms)
        if self._ring and self._playback_enabled:
            self._ring.write(block)
        if self._transcription_queue is not None:
            try:
                self._transcription_queue.put_nowait(block.copy())
            except queue.Full:
                pass

    def _output_callback(self, outdata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            self._report_status(f"Output: {status}")
        if self._ring:
            self._ring.read(outdata)
        else:
            outdata.fill(0)

    def start_capture(self, input_device: int | None = None) -> bool:
        """Start capturing from the given input device (BlackHole). Returns True on success."""
        with self._lock:
            if self._running:
                return True
            dev = input_device if input_device is not None else find_blackhole_input_device()
            if dev is None:
                self._report_status("Failed to capture: BlackHole not found. Install BlackHole and select it as the app's output.")
                return False
            self._ring = RingBuffer(num_blocks=16, block_samples=BLOCK_SIZE * CHANNELS)
            try:
                self._input_stream = sd.InputStream(
                    device=dev,
                    channels=CHANNELS,
                    samplerate=SAMPLE_RATE,
                    blocksize=BLOCK_SIZE,
                    dtype=DTYPE,
                    callback=self._input_callback,
                )
                self._input_stream.start()
                self._input_device = dev
                self._running = True
                self._report_status("Capturing audio")
                return True
            except Exception as e:
                self._report_status(f"Failed to capture: {e}")
                return False

    def stop_capture(self) -> None:
        with self._lock:
            self._running = False
            if self._output_stream:
                try:
                    self._output_stream.stop()
                    self._output_stream.close()
                except Exception:
                    pass
                self._output_stream = None
            if self._input_stream:
                try:
                    self._input_stream.stop()
                    self._input_stream.close()
                except Exception:
                    pass
                self._input_stream = None
            self._ring = None
            self._report_status("Stopped")

    def set_playback(self, enabled: bool) -> None:
        """Turn real-time playback to speakers on or off."""
        with self._lock:
            if enabled == self._playback_enabled:
                return
            self._playback_enabled = enabled
            if not self._running or not self._input_stream:
                return
            if enabled:
                out_dev = get_default_output_device()
                if out_dev is None:
                    self._report_status("Playback failed: no output device")
                    self._playback_enabled = False
                    return
                if self._ring is None:
                    self._ring = RingBuffer(num_blocks=16, block_samples=BLOCK_SIZE * CHANNELS)
                try:
                    self._output_stream = sd.OutputStream(
                        device=out_dev,
                        channels=CHANNELS,
                        samplerate=SAMPLE_RATE,
                        blocksize=BLOCK_SIZE,
                        dtype=DTYPE,
                        callback=self._output_callback,
                    )
                    self._output_stream.start()
                    self._output_device = out_dev
                except Exception as e:
                    self._report_status(f"Playback failed: {e}")
                    self._playback_enabled = False
            else:
                if self._output_stream:
                    try:
                        self._output_stream.stop()
                        self._output_stream.close()
                    except Exception:
                        pass
                    self._output_stream = None

    def set_muted(self, muted: bool) -> None:
        self._muted = muted

    @property
    def is_playback_enabled(self) -> bool:
        return self._playback_enabled

    @property
    def is_muted(self) -> bool:
        return self._muted

    @property
    def is_running(self) -> bool:
        return self._running
