"""
Audio capture with hardened VAD and Faster Whisper STT.
RESTORED: Multi-device hardware binding (Mic/System/WASAPI).
LAYER 6: Integrated restart() for hot-swapping audio modes in settings.
"""

import queue
import threading
import time
import collections
import numpy as np
from typing import Optional, List, Tuple
from PyQt6.QtCore import QObject, pyqtSignal
from utils.logger import setup_logger

logger = setup_logger(__name__)


class AudioCapture(QObject):
    transcription_ready = pyqtSignal(str)
    level = pyqtSignal(float)

    def __init__(self, config):
        super().__init__()
        self.config = config
        self._running = False
        self._paused = False
        self._muted = False # RESTORED: Required for UI sync
        self.q = queue.Queue(maxsize=100)
        self.transcripts = []
        self.sr = config.get("capture.audio.sample_rate", 16000)
        self.capture_mode = config.get("capture.audio.mode", "system")
        
        self.block_ms = 200
        self.block_size = int(self.sr * self.block_ms / 1000)
        self.silence_blocks = int(1500 / self.block_ms)

        self.model = None
        self._model_name = config.get("capture.audio.whisper_model", "base")
        self._model_loaded = False
        self._model_lock = threading.Lock()
        self._current_rms = 0.0
        
        # Stream management
        self._active_streams = []

    def _ensure_whisper_loaded(self):
        with self._model_lock:
            if self._model_loaded: return
            try:
                from faster_whisper import WhisperModel
                self.model = WhisperModel(self._model_name, device="auto", compute_type="int8")
                self._model_loaded = True
                logger.info(f"✅ Whisper Ready: {self._model_name}")
            except Exception as e:
                logger.error(f"Whisper Error: {e}")
                self._model_loaded = False 

    def start(self):
        if self._running: return
        self._running = True
        self._paused = self._muted # Sync with current mute state
        threading.Thread(target=self._capture_loop, daemon=True, name="audio-cap").start()
        threading.Thread(target=self._process_loop, daemon=True, name="audio-proc").start()

    def stop(self):
        """Harden cleanup: Stop threads and safely purge streams."""
        self._running = False
        self._close_streams()
        time.sleep(0.1)

    def restart(self, new_mode: str = None):
        """Layer 6: Hot-swaps the hardware streams without app reboot. Added cooldown."""
        logger.info(f"🔄 Restarting audio pipeline (Mode: {new_mode or self.capture_mode})")
        self.stop()
        if new_mode: self.capture_mode = new_mode
        time.sleep(1.5) # Hardware settle time
        self.start()

    def _close_streams(self):
        """Thread-safe stream purging."""
        for s in list(self._active_streams):
            try: 
                if hasattr(s, 'stop'): s.stop()
                if hasattr(s, 'close'): s.close()
            except: pass
        self._active_streams.clear()

    def toggle(self):
        self._muted = not self._muted
        self._paused = self._muted
        return self._muted

    def _find_system_audio_source(self):
        import sounddevice as sd
        try:
            # 1. Prioritize WASAPI Loopback (Most stable on Win10/11)
            apis = sd.query_hostapis()
            wasapi_idx = next((i for i, a in enumerate(apis) if "WASAPI" in a['name']), None)
            
            if wasapi_idx is not None:
                devices = sd.query_devices()
                for i, d in enumerate(devices):
                    if d['hostapi'] == wasapi_idx and d['max_output_channels'] > 0:
                        # On WASAPI, the output device's index is used with 'loopback=True'
                        return i, d['name'], True

            # 2. Fallback to Virtual Cable (MME/DirectSound)
            devices = sd.query_devices()
            for i, d in enumerate(devices):
                if d['max_input_channels'] > 0 and any(x in d['name'].lower() for x in ["cable", "vb-audio", "stereo mix"]):
                    return i, d['name'], False
        except Exception as e:
            logger.debug(f"Source discovery error: {e}")
        return None, "Default", False

    def _capture_loop(self):
        import sounddevice as sd
        def make_cb():
            def cb(indata, frames, time, status):
                if status: logger.debug(f"Audio Status: {status}")
                if not self._paused and self._running:
                    data = np.mean(indata, axis=1, keepdims=True).astype(np.float32) if indata.shape[1] > 1 else indata.copy().astype(np.float32)
                    self.q.put_nowait(data)
                    self._current_rms = float(np.sqrt(np.mean(indata**2)))
            return cb

        self._close_streams()
        mode = self.capture_mode
        try:
            # AGGRESSIVE RETRY LOGIC: Give hardware a chance to unstick
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    if mode in ["mic", "both"]:
                        # Try to open default mic
                        s_mic = sd.InputStream(samplerate=self.sr, channels=1, callback=make_cb())
                        self._active_streams.append(s_mic)
                        s_mic.start()
                    
                    if mode in ["system", "both"]:
                        idx, name, is_loopback = self._find_system_audio_source()
                        if idx is not None:
                            logger.info(f"🎤 Binding to System Audio: {name} (Loopback: {is_loopback})")
                            kwargs = {"device": idx, "samplerate": self.sr, "channels": 1, "callback": make_cb()}
                            if is_loopback: 
                                kwargs["loopback"] = True # sounddevice WASAPI standard
                            s_sys = sd.InputStream(**kwargs)
                            self._active_streams.append(s_sys)
                            s_sys.start()
                    
                    logger.info("🎙️ Audio Hardware Successfully Synchronized.")
                    break # Success!
                except Exception as e:
                    self._close_streams()
                    if attempt < max_retries - 1:
                        logger.warning(f"⚠️ Hardware Busy (Attempt {attempt+1}/{max_retries}). Retrying WASAPI reset...")
                        time.sleep(2.0)
                    else: raise e

            while self._running:
                self.level.emit(self._current_rms)
                time.sleep(0.1)
        except Exception as e:
            logger.error(f"❌ Final Capture Loop Failure: {e}")
        finally:
            self._close_streams()

    def _process_loop(self):
        speech_buffer = []
        is_speaking = False
        silence_count = 0
        while self._running:
            try: data = self.q.get(timeout=0.5)
            except: continue
            rms = np.sqrt(np.mean(data**2))
            has_speech = rms > 0.005
            if has_speech or is_speaking: speech_buffer.append(data)
            if has_speech:
                silence_count = 0
                is_speaking = True
            elif is_speaking:
                silence_count += 1
                if silence_count >= self.silence_blocks:
                    self._transcribe(speech_buffer)
                    speech_buffer = []
                    is_speaking = False

    def _transcribe(self, buffer):
        self._ensure_whisper_loaded()
        if not self.model or len(buffer) < 5: return
        try:
            audio = np.concatenate(buffer, axis=0).flatten()
            segments, _ = self.model.transcribe(audio, language="en")
            text = " ".join([s.text.strip() for s in segments if not self._is_hall(s.text)])
            if text.strip():
                self.transcription_ready.emit(text.strip())
        except: pass

    def _is_hall(self, t): return any(x in t.lower() for x in ["thank you", "watching", "♪", "[music]"])
    def get_transcript(self): return " ".join(self.transcripts)
    def clear(self): self.transcripts.clear()
