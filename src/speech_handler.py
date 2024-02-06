import logging
import sched
import time
import wave
from pynput import keyboard
from transformers import pipeline
from transformers.pipelines.audio_utils import ffmpeg_microphone_live

class SpeechHandler:
    """ Automatic Speech Recognition 
    Args:
        model_id: Model ID for the automatic speech recognition, try openai/whisper-tiny.en"""
    def __init__(self, model_id, trigger_key, llama_handler):
        self.trigger_key = trigger_key
        self.key_pressed = False
        self.recording_started = False
        self.recording_stopped = False
        self.key_listener = keyboard.Listener(self._on_press, self._on_release)
        self.task_scheduler = sched.scheduler(time.time, time.sleep)
        self.transcriber = pipeline("automatic-speech-recognition",
                                    model=model_id,
                                    device="cpu")
        self.llama_handler = llama_handler
                
    def reset(self):
        self.key_pressed = False
        self.recording_started = False
        self.recording_stopped = False

    def _on_press(self, key):
        if key == self.trigger_key:
            self.key_pressed = True
        return True
    
    def _on_release(self, key):
        if key == self.trigger_key:
            self.key_pressed = False
        if key == keyboard.Key.esc:
                # Stop listener
                return False
        return True
    
    def keychek_loop(self):
        if self.key_pressed and not self.recording_started:
            logging.info("Speak while you keep the key pressed.")
            self.recording_started = True
            speech_input = self.transcribe_mic(chunk_length_s=5.0)
            self.recording_started = False
            if len(speech_input) > 0:
                print(speech_input)
                self.llama_handler.prompt(speech_input)
        elif not self.key_pressed and self.recording_started:
            self.mic.close()
            self.recording_started = False
            self.recording_stopped = True
            logging.debug("Recording stopped.")
            
        self.task_scheduler.enter(delay=.1, priority=1, action=self.keychek_loop)

    def listen(self):
        logging.debug("Waiting for any key")

        self.reset()
        self.key_listener.start()

        self.task_scheduler.enter(delay=.1, priority=1, action=self.keychek_loop)
        self.task_scheduler.run()
        
    def transcribe_mic(self, chunk_length_s: float) -> str:
        """ Transcribe the audio from a microphone """
        sampling_rate = self.transcriber.feature_extractor.sampling_rate
        self.mic = ffmpeg_microphone_live(
                sampling_rate=sampling_rate,
                chunk_length_s=chunk_length_s,
                stream_chunk_s=chunk_length_s,
            )
        
        result = ""
        for item in self.transcriber(self.mic):
            result = item["text"]
            if not item["partial"][0]:
                break
        return result.strip()

