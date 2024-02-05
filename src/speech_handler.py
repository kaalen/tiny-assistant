from transformers import pipeline
from transformers.pipelines.audio_utils import ffmpeg_microphone_live

# asr_model_id = "openai/whisper-tiny.en"

class SpeechHandler:
    """ Automatic Speech Recognition 
    Args:
        model_id: Model ID for the automatic speech recognition, try openai/whisper-tiny.en"""
    def __init__(self, model_id):
        self.transcriber = pipeline("automatic-speech-recognition",
                                    model=model_id,
                                    device="cpu")
        
    def transcribe_mic(self, chunk_length_s: float) -> str:
        """ Transcribe the audio from a microphone """
        sampling_rate = self.transcriber.feature_extractor.sampling_rate
        mic = ffmpeg_microphone_live(
                sampling_rate=sampling_rate,
                chunk_length_s=chunk_length_s,
                stream_chunk_s=chunk_length_s,
            )
        
        result = ""
        for item in self.transcriber(mic):
            result = item["text"]
            if not item["partial"][0]:
                break
        return result.strip()

