import logging
from pynput import keyboard
from llm_handler import LLMHandler
from speech_handler import SpeechHandler

model_file = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"  # OR "llama-2-7b-chat.Q4_K_M.gguf"
template_tiny = """<|system|>
                   You are a smart mini computer named Raspberry Pi. 
                   Write a short but funny answer.</s>
                   <|user|>
                   {question}</s>
                   <|assistant|>"""
template_llama = """<s>[INST] <<SYS>>
                    You are a smart mini computer named Raspberry Pi.
                    Write a short but funny answer.</SYS>>
                    {question} [/INST]"""

if __name__ == "__main__":
    logger = logging.getLogger() 
    logger.setLevel(logging.INFO)

    llama_handler = LLMHandler(model_file, template_tiny, enable_tts=True)

    logger.info("Init automatic speech recogntion...")
    speech_handler = SpeechHandler("openai/whisper-tiny.en", keyboard.Key.space, llama_handler)
    speech_handler.listen()

