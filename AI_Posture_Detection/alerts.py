"""
============================================================
alerts.py
AI-Based Real-Time Human Posture Detection and Correction System
============================================================
Manages all alert systems:

1. Audio beep alert  (winsound / pygame fallback)
2. Voice alert       (pyttsx3 text-to-speech)
3. Cooldown logic    (prevents alert spam)
4. Sound toggle      (on/off control)
5. Custom sound file support
============================================================
"""

import os
import time
import threading
import logging

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
SOUNDS_DIR = os.path.join(BASE_DIR, "sounds")
os.makedirs(SOUNDS_DIR, exist_ok=True)

ALERT_MP3_PATH = os.path.join(SOUNDS_DIR, "alert.mp3")
BEEP_WAV_PATH  = os.path.join(SOUNDS_DIR, "beep.wav")


# ─────────────────────────────────────────────────────────────
# SECTION 1: SOUND GENERATION (create fallback .wav if missing)
# ─────────────────────────────────────────────────────────────

def generate_beep_wav(filepath=BEEP_WAV_PATH, frequency=880, duration_ms=400):
    """
    Generate a simple sine-wave beep and save as WAV file.

    This is used when no custom alert.mp3 exists, ensuring the alert
    system always works out of the box.

    Parameters:
        filepath    (str): Path to save the .wav file
        frequency   (int): Beep frequency in Hz (880 = A5 note)
        duration_ms (int): Duration in milliseconds
    """
    try:
        import struct
        import math
        import wave

        sample_rate = 44100
        num_samples = int(sample_rate * duration_ms / 1000)
        amplitude   = 32767 * 0.6  # ~60% volume

        samples = []
        for i in range(num_samples):
            t = i / sample_rate
            # Fade in/out to avoid clicks
            fade = 1.0
            fade_samples = int(sample_rate * 0.02)  # 20ms fade
            if i < fade_samples:
                fade = i / fade_samples
            elif i > num_samples - fade_samples:
                fade = (num_samples - i) / fade_samples

            sample = int(amplitude * fade * math.sin(2 * math.pi * frequency * t))
            samples.append(struct.pack("<h", sample))

        with wave.open(filepath, "wb") as wf:
            wf.setnchannels(1)     # mono
            wf.setsampwidth(2)     # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(b"".join(samples))

        logger.info(f"[Alerts] Generated beep.wav: {filepath}")
        return True

    except Exception as e:
        logger.warning(f"[Alerts] Could not generate beep.wav: {e}")
        return False


# Auto-generate beep.wav on import if it doesn't exist
if not os.path.exists(BEEP_WAV_PATH):
    generate_beep_wav()


# ─────────────────────────────────────────────────────────────
# SECTION 2: ALERT MANAGER CLASS
# ─────────────────────────────────────────────────────────────

class AlertManager:
    """
    Central alert manager for the posture detection system.

    Manages:
    - Audio alerts (beep sound)
    - Voice alerts (pyttsx3 TTS)
    - Cooldown timers to prevent spam
    - Sound enable/disable toggle

    Usage:
        manager = AlertManager()
        manager.trigger_bad_posture_alert("Please sit straight!")
        manager.trigger_good_posture_clear()
    """

    def __init__(self,
                 sound_enabled: bool = True,
                 voice_enabled: bool = True,
                 alert_cooldown: float = 5.0,
                 bad_posture_threshold: float = 3.0):
        """
        Initialize the AlertManager.

        Parameters:
            sound_enabled         (bool): Play beep sound on bad posture
            voice_enabled         (bool): Play voice message on bad posture
            alert_cooldown        (float): Minimum seconds between repeated alerts
            bad_posture_threshold (float): Seconds of continuous bad posture
                                           before alert triggers
        """
        self.sound_enabled         = sound_enabled
        self.voice_enabled         = voice_enabled
        self.alert_cooldown        = alert_cooldown
        self.bad_posture_threshold = bad_posture_threshold

        self._last_alert_time   = 0.0   # timestamp of last alert
        self._bad_posture_start = None  # when bad posture was first detected
        self._is_alerting       = False  # prevent overlapping threads
        self._voice_engine      = None   # pyttsx3 engine (lazy init)

        # Voice message
        self._voice_messages = [
            "Please sit straight.",
        ]
        self._msg_index = 0

        logger.info(f"[Alerts] AlertManager initialized | "
                    f"Sound: {sound_enabled} | Voice: {voice_enabled} | "
                    f"Cooldown: {alert_cooldown}s | Threshold: {bad_posture_threshold}s")

    # ── Public API ────────────────────────────────────────────

    def update_bad_posture(self, is_bad: bool) -> bool:
        """
        Call this every frame to track bad posture duration.

        Automatically triggers alert if bad posture persists beyond threshold.

        Parameters:
            is_bad (bool): True if current frame shows bad posture

        Returns:
            bool: True if alert was just triggered this call
        """
        now = time.time()

        if is_bad:
            # Start timer if not already running
            if self._bad_posture_start is None:
                self._bad_posture_start = now

            elapsed = now - self._bad_posture_start

            # Trigger alert if threshold exceeded AND cooldown passed
            if (elapsed >= self.bad_posture_threshold and
                    now - self._last_alert_time >= self.alert_cooldown):
                self._fire_alert()
                return True

        else:
            # Good posture → reset timer
            self._bad_posture_start = None

        return False

    def trigger_bad_posture_alert(self, message: str = None):
        """
        Immediately fire a bad posture alert (bypasses threshold timer).

        Parameters:
            message (str): Optional custom voice message
        """
        self._fire_alert(message=message)

    def trigger_good_posture_clear(self):
        """Signal that good posture is detected. Resets bad posture timer."""
        self._bad_posture_start = None

    def get_bad_posture_duration(self) -> float:
        """
        Return how long (in seconds) bad posture has been detected.

        Returns:
            float: Duration in seconds, or 0 if currently good posture
        """
        if self._bad_posture_start is None:
            return 0.0
        return round(time.time() - self._bad_posture_start, 1)

    def toggle_sound(self):
        """Toggle sound alert on/off."""
        self.sound_enabled = not self.sound_enabled
        state = "ON" if self.sound_enabled else "OFF"
        logger.info(f"[Alerts] Sound toggled: {state}")
        return self.sound_enabled

    def toggle_voice(self):
        """Toggle voice alert on/off."""
        self.voice_enabled = not self.voice_enabled
        state = "ON" if self.voice_enabled else "OFF"
        logger.info(f"[Alerts] Voice toggled: {state}")
        return self.voice_enabled

    # ── Internal Methods ──────────────────────────────────────

    def _fire_alert(self, message: str = None):
        """
        Dispatch alert on a background thread to avoid blocking the main loop.

        Parameters:
            message (str): Optional voice message override
        """
        if self._is_alerting:
            return  # Don't overlap alerts

        self._last_alert_time = time.time()
        self._is_alerting     = True

        voice_msg = message or self._get_next_message()

        thread = threading.Thread(
            target=self._alert_worker,
            args=(voice_msg,),
            daemon=True
        )
        thread.start()

    def _alert_worker(self, voice_message: str):
        """
        Worker thread: plays beep then voice message sequentially.

        Parameters:
            voice_message (str): Text to speak
        """
        try:
            if self.sound_enabled:
                self._play_beep()
            if self.voice_enabled:
                time.sleep(0.1)  # brief pause between beep and voice
                self._speak(voice_message)
        except Exception as e:
            logger.error(f"[Alerts] Alert worker error: {e}")
        finally:
            self._is_alerting = False

    def _play_beep(self):
        """
        Play the alert sound using the best available method.

        Priority:
            1. pygame.mixer (mp3 or wav)
            2. winsound (Windows beep)
            3. print fallback
        """
        try:
            import pygame
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)

            # Prefer custom alert.mp3 if it exists, otherwise use generated wav
            sound_file = ALERT_MP3_PATH if os.path.exists(ALERT_MP3_PATH) else BEEP_WAV_PATH

            if os.path.exists(sound_file):
                pygame.mixer.music.load(sound_file)
                pygame.mixer.music.set_volume(0.8)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.05)
            else:
                # Fallback: winsound beep (Windows-only)
                self._winsound_beep()

        except Exception as pygame_err:
            logger.warning(f"[Alerts] pygame failed ({pygame_err}), trying winsound...")
            self._winsound_beep()

    def _winsound_beep(self):
        """Windows winsound fallback beep."""
        try:
            import winsound
            winsound.Beep(880, 400)
            time.sleep(0.1)
            winsound.Beep(660, 300)
        except Exception as e:
            logger.warning(f"[Alerts] winsound failed: {e}")
            print("\a")  # terminal bell as last resort

    def _speak(self, text: str):
        """
        Speak the given text using win32com or pyttsx3 text-to-speech.

        Parameters:
            text (str): Text to convert to speech
        """
        # Try win32com.client SpVoice first (highly robust on Windows background threads)
        try:
            import pythoncom
            pythoncom.CoInitialize()
            import win32com.client
            # Dispatch SAPI voice
            speaker = win32com.client.Dispatch("SAPI.SpVoice")
            speaker.Speak(text)
            logger.info(f"[Alerts] Spoken successfully via SAPI: {text}")
            return
        except Exception as sapi_err:
            logger.warning(f"[Alerts] SAPI Speak failed ({sapi_err}), falling back to pyttsx3...")

        # Fallback to pyttsx3 with CoInitialize
        try:
            import pythoncom
            pythoncom.CoInitialize()
            import pyttsx3

            if self._voice_engine is None:
                self._voice_engine = pyttsx3.init()
                self._voice_engine.setProperty("rate", 150)    # words per minute
                self._voice_engine.setProperty("volume", 0.9)  # 0–1

                # Try to set a female voice if available
                voices = self._voice_engine.getProperty("voices")
                for voice in voices:
                    if "female" in voice.name.lower() or "zira" in voice.name.lower():
                        self._voice_engine.setProperty("voice", voice.id)
                        break

            self._voice_engine.say(text)
            self._voice_engine.runAndWait()
            logger.info(f"[Alerts] Spoken successfully via pyttsx3: {text}")

        except Exception as e:
            logger.warning(f"[Alerts] Both SAPI and pyttsx3 voice failed: {e}")

    def _get_next_message(self) -> str:
        """
        Return the next voice message in rotation.

        Returns:
            str: Next alert message
        """
        msg = self._voice_messages[self._msg_index % len(self._voice_messages)]
        self._msg_index += 1
        return msg


# ─────────────────────────────────────────────────────────────
# SECTION 3: QUICK TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("[Alerts] Testing AlertManager...")

    manager = AlertManager(
        sound_enabled=True,
        voice_enabled=True,
        bad_posture_threshold=2.0,
        alert_cooldown=3.0
    )

    # Simulate 5 seconds of bad posture
    print("Simulating bad posture for 5 seconds...")
    for i in range(50):
        triggered = manager.update_bad_posture(is_bad=True)
        if triggered:
            print(f"  ▶ ALERT TRIGGERED at t={manager.get_bad_posture_duration()}s")
        time.sleep(0.1)

    # Simulate good posture
    print("Good posture detected. Clearing alerts...")
    manager.trigger_good_posture_clear()
    print("Done.")
