import logging
from typing import Any, Dict, List, Optional
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser

from speech_handler import SpeechHandler
from llm_handler import LLMHandler


llm: Optional[LlamaCpp] = None

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

    logger.info("Init automatic speech recogntion...")
    speech_handler = SpeechHandler("openai/whisper-tiny.en")

    logger.info("Init LLaMA GPT...")
    llama_handler = LLMHandler(model_file, template_tiny)

    while True:
        # Q-A loop:
        # add_display_line("Start speaking")
        print("Start typing your question:")
        # add_display_line("")
        question = input() #speech_handler.transcribe_mic(chunk_length_s=5.0)

        if len(question) > 0:
            # print(f"> {question}")
            # print()
            # SAMPLE QUESTION: "Write a short quirky story about a female wizard, who accidentally creates a time machine and travels back in time to meet her long-lost love."

            llama_handler.prompt(question)