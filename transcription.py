"""
Continuous voice-to-text via AssemblyAI (free online API).
Consumes audio from capture (48 kHz stereo float32), buffers 1.2 s chunks,
sends to AssemblyAI (4 in flight, fast polling), writes as results arrive.
Supports English and Italian.
"""
from __future__ import annotations

import os
import queue
import struct
import tempfile
import threading
from concurrent.futures import Future
from typing import Callable

import numpy as np

# Audio format for capture (must match audio_capture)
SAMPLE_RATE = 48000
CHANNELS_CAPTURE = 2
CHUNK_DURATION_SEC = 1.2  # 1.2 s chunks for very fast, continuous output
SAMPLES_PER_CHUNK = int(SAMPLE_RATE * CHUNK_DURATION_SEC)
MAX_IN_FLIGHT = 4  # Up to 4 at API at once; drain results quickly (~1 s delay)
QUEUE_POLL_TIMEOUT = 0.06  # Wake often to drain in_flight and show results fast


def _float32_stereo_to_linear16_mono(block: np.ndarray) -> np.ndarray:
    """Convert (frames, 2) float32 [-1,1] to (frames,) int16 mono (L+R)/2."""
    if block.ndim == 2:
        mono = block.mean(axis=1)
    else:
        mono = block.ravel()
    mono = np.clip(mono, -1.0, 1.0)
    return (mono * 32767).astype(np.int16)


def _wav_bytes_from_mono_int16(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Build a minimal WAV file (bytes) from mono int16 samples."""
    n = len(samples)
    data = samples.tobytes()
    # WAV header: 44 bytes
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + len(data),
        b"WAVE",
        b"fmt ",
        16,  # fmt chunk size
        1,   # PCM
        1,   # mono
        sample_rate,
        sample_rate * 2,
        2,   # block align
        16,  # bits per sample
        b"data",
        len(data),
    )
    return header + data


def _submit_transcribe_async(audio_wav_bytes: bytes, language_code: str):
    """
    Submit WAV to AssemblyAI asynchronously. Returns (Future, temp_path).
    Caller must delete temp_path when done.
    """
    import assemblyai as aai
    key = os.environ.get("ASSEMBLYAI_API_KEY")
    if not key:
        return None, None
    aai.settings.api_key = key
    aai.settings.polling_interval = 0.5  # Check API status every 0.5 s (faster than default 3 s)
    lang = "en" if language_code.startswith("en") else "it"
    config = aai.TranscriptionConfig(
        language_code=lang,
        speech_models=["universal-2"],
    )
    transcriber = aai.Transcriber(config=config)
    f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    f.write(audio_wav_bytes)
    f.flush()
    path = f.name
    f.close()
    future = transcriber.transcribe_async(path)
    return future, path


def _result_from_future(future: Future) -> str | None:
    """Get transcript text from a completed Future; returns None on error or empty."""
    import assemblyai as aai
    try:
        transcript = future.result(timeout=0)
        if transcript.status == aai.TranscriptStatus.error:
            return f"[Transcription error: {transcript.error}]"
        return (transcript.text or "").strip() or None
    except Exception as e:
        return f"[Transcription error: {e}]"


class Transcriber:
    """
    Consumes raw audio blocks from a queue, buffers chunks (duration set at init),
    sends to AssemblyAI (up to 4 in flight), drains results every ~60 ms.
    """

    def __init__(
        self,
        audio_queue: queue.Queue[np.ndarray],
        on_transcription: Callable[[str], None],
        language_code: str = "en-US",
        chunk_duration_sec: float = CHUNK_DURATION_SEC,
    ):
        self._queue = audio_queue
        self._on_transcription = on_transcription
        self._language_code = language_code
        self._samples_per_chunk = int(SAMPLE_RATE * max(0.5, min(10.0, chunk_duration_sec)))
        self._buffer: list[np.ndarray] = []
        self._buffer_samples = 0
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def set_language_code(self, code: str) -> None:
        with self._lock:
            self._language_code = code

    def _drain_in_flight(self, in_flight: list[tuple[Future, str]]) -> None:
        """Collect completed futures, push results, delete temp files, remove from list."""
        done_idxs = []
        for i, (future, path) in enumerate(in_flight):
            if not future.done():
                continue
            try:
                if path and os.path.exists(path):
                    try:
                        os.unlink(path)
                    except OSError:
                        pass
            finally:
                pass
            text = _result_from_future(future)
            if text:
                self._on_transcription(text)
            done_idxs.append(i)
        for i in reversed(done_idxs):
            in_flight.pop(i)

    def _worker(self) -> None:
        if not os.environ.get("ASSEMBLYAI_API_KEY"):
            self._on_transcription(
                "[Transcription: set ASSEMBLYAI_API_KEY (get free key at https://www.assemblyai.com/app/account)]"
            )
            return
        in_flight: list[tuple[Future, str]] = []
        while self._running:
            try:
                self._drain_in_flight(in_flight)
                block = self._queue.get(timeout=QUEUE_POLL_TIMEOUT)
                if block is None:
                    break
                with self._lock:
                    self._buffer.append(block)
                    self._buffer_samples += block.shape[0]
                while self._running and self._buffer_samples >= self._samples_per_chunk and len(in_flight) < MAX_IN_FLIGHT:
                    with self._lock:
                        need = self._samples_per_chunk
                        chunks_to_use = []
                        taken = 0
                        while taken < need and self._buffer:
                            b = self._buffer.pop(0)
                            self._buffer_samples -= b.shape[0]
                            take = min(b.shape[0], need - taken)
                            if take < b.shape[0]:
                                leftover = b[take:]
                                b = b[:take]
                                self._buffer.insert(0, leftover)
                                self._buffer_samples += leftover.shape[0]
                            chunks_to_use.append(b)
                            taken += b.shape[0]
                    if not chunks_to_use:
                        break
                    combined = np.concatenate(chunks_to_use, axis=0)
                    mono_int16 = _float32_stereo_to_linear16_mono(combined)
                    wav_bytes = _wav_bytes_from_mono_int16(mono_int16)
                    with self._lock:
                        lang = self._language_code
                    try:
                        future, path = _submit_transcribe_async(wav_bytes, lang)
                        if future is not None and path:
                            in_flight.append((future, path))
                    except Exception as e:
                        self._on_transcription(f"[Transcription error: {e}]")
            except queue.Empty:
                self._drain_in_flight(in_flight)
                continue
            except Exception as e:
                self._on_transcription(f"[Transcription error: {e}]")
        # Drain any remaining in-flight on exit
        for future, path in in_flight:
            try:
                if path and os.path.exists(path):
                    os.unlink(path)
            except OSError:
                pass

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
