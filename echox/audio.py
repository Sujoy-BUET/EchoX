"""File I/O, playback and recording.

Playback and recording run in PortAudio callback threads. The GUI only reads
`Player.pos` and drains `Recorder` blocks from a queue, so audio never waits
for drawing."""
from __future__ import annotations

import os
import queue
import shutil
import subprocess
import tempfile
import threading

import numpy as np
import soundfile as sf

from .dsp.common import to_mono

try:
    import sounddevice as sd
    AUDIO_ERROR = None
except OSError as exc:  # PortAudio missing (Linux without libportaudio2)
    sd = None
    AUDIO_ERROR = str(exc)

FILE_FILTER = ("Audio files (*.wav *.flac *.ogg *.mp3 *.aiff *.aif *.m4a *.aac *.opus *.wma);;"
               "All files (*)")


def _decode_with_ffmpeg(path: str):
    """Formats libsndfile cannot read (m4a/aac, wma, ...) go through ffmpeg into
    a temporary float WAV. Sample rate is kept."""
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise RuntimeError("This format needs ffmpeg (Mac: brew install ffmpeg, "
                           "Ubuntu: sudo apt install ffmpeg).")
    fd, tmp = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        proc = subprocess.run([exe, "-v", "error", "-y", "-i", path, "-vn",
                               "-acodec", "pcm_f32le", tmp], capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg could not decode the file: {proc.stderr.strip()[:200]}")
        return sf.read(tmp, always_2d=False)
    finally:
        os.remove(tmp)


def load_audio(path: str) -> tuple[np.ndarray, int]:
    """wav / flac / ogg / mp3 / aiff through libsndfile, anything else via ffmpeg."""
    try:
        data, sr = sf.read(path, always_2d=False)
    except (sf.LibsndfileError, RuntimeError, TypeError):
        data, sr = _decode_with_ffmpeg(path)
    return to_mono(data), int(sr)


def save_audio(path: str, x: np.ndarray, sr: int):
    sf.write(path, np.clip(x, -1.0, 1.0).astype(np.float32), sr)


class Player:
    def __init__(self):
        self.data = np.zeros(0, dtype=np.float32)
        self.sr = 44100
        self.pos = 0
        self.playing = False
        self.loop: tuple[int, int] | None = None
        self.volume = 1.0
        self.time_scale = 1.0          # source samples per played sample (speed change)
        self.boost_ceiling = 10 ** (-1.0 / 20)      # soft ceiling for volume above 100 %
        self.stream = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------- control
    def set_source(self, data: np.ndarray, sr: int, time_scale: float = 1.0):
        """`data` is what gets played. With a time-stretched buffer, time_scale
        says how many source samples one played sample stands for, so the rest
        of the app can keep working in source samples."""
        if sr != self.sr:          # stream must be reopened; never close while holding the lock
            self.playing = False
            self._close()
            self.sr = sr
        with self._lock:
            self.data = np.ascontiguousarray(data, dtype=np.float32)
            self.time_scale = float(time_scale)
            self.pos = min(self.pos, len(self.data))

    def play(self):
        if sd is None:
            raise RuntimeError(f"Audio output unavailable: {AUDIO_ERROR}")
        if len(self.data) == 0:
            return
        if self.stream is None:
            self.stream = sd.OutputStream(samplerate=self.sr, channels=1, dtype="float32",
                                          blocksize=1024, latency="low", callback=self._callback)
            self.stream.start()
        with self._lock:
            if self.loop and not (self.loop[0] <= self.pos < self.loop[1]):
                self.pos = self.loop[0]
            elif self.pos >= len(self.data):
                self.pos = 0
            self.playing = True

    def pause(self):
        self.playing = False

    def stop(self):
        self.playing = False
        self.pos = self.loop[0] if self.loop else 0

    def seek(self, sample: int):
        with self._lock:
            self.pos = int(np.clip(sample, 0, max(len(self.data) - 1, 0)))

    def set_loop(self, region: tuple[int, int] | None):
        """Region in source samples."""
        if region and region[1] - region[0] < 1024:
            region = None
        if region is not None:
            region = (int(region[0] / self.time_scale), int(region[1] / self.time_scale))
        self.loop = region

    # ---- positions in source samples, whatever buffer is actually playing
    @property
    def pos_source(self) -> int:
        return int(self.pos * self.time_scale)

    @property
    def length_source(self) -> int:
        return int(len(self.data) * self.time_scale)

    def seek_source(self, sample: int):
        self.seek(int(sample / self.time_scale))

    def latency_samples(self) -> int:
        if self.stream is None:
            return 0
        return int(self.stream.latency * self.sr)

    def close(self):
        self.playing = False
        self._close()

    def _close(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    # ------------------------------------------------------ audio callback
    def _callback(self, out, frames, time_info, status):
        out.fill(0)
        if not self.playing:
            return
        with self._lock:
            data, pos, loop = self.data, self.pos, self.loop
            end = loop[1] if loop else len(data)
            written = 0
            while written < frames:
                n = min(frames - written, end - pos)
                if n <= 0:
                    if loop:
                        pos = loop[0]
                        continue
                    self.playing = False
                    break
                block = data[pos:pos + n] * self.volume
                if self.volume > 1.0:
                    # tanh knee: below the ceiling nothing changes, above it the
                    # peaks bend over instead of being chopped flat
                    c = self.boost_ceiling
                    loud = np.abs(block) > c * 0.7
                    if loud.any():
                        b = block[loud]
                        block[loud] = np.sign(b) * (c * 0.7 + (c - c * 0.7)
                                                    * np.tanh((np.abs(b) - c * 0.7) / (c - c * 0.7)))
                out[written:written + n, 0] = block
                pos += n
                written += n
            self.pos = pos


class Recorder:
    def __init__(self):
        self.stream = None
        self.sr = 44100
        self._q: queue.Queue = queue.Queue()

    def start(self, sr: int = 44100):
        if sd is None:
            raise RuntimeError(f"Microphone unavailable: {AUDIO_ERROR}")
        try:
            self.stream = sd.InputStream(samplerate=sr, channels=1, dtype="float32",
                                         blocksize=1024, callback=self._callback)
        except Exception:
            sr = int(sd.query_devices(kind="input")["default_samplerate"])
            self.stream = sd.InputStream(samplerate=sr, channels=1, dtype="float32",
                                         blocksize=1024, callback=self._callback)
        self.sr = sr
        self._q = queue.Queue()
        self.stream.start()

    def read(self) -> np.ndarray:
        blocks = []
        while True:
            try:
                blocks.append(self._q.get_nowait())
            except queue.Empty:
                break
        return np.concatenate(blocks) if blocks else np.zeros(0, dtype=np.float32)

    def stop(self) -> np.ndarray:
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        return self.read()

    @property
    def active(self) -> bool:
        return self.stream is not None

    def _callback(self, indata, frames, time_info, status):
        self._q.put(indata[:, 0].copy())
