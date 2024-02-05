import logging
from typing import Any, Dict, List, Optional
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser

class LLMHandler():
    def __init__(self, model_file, prompt_template):
        """ Load the LLM model of your choice
         Args:
            model_file (str): path to the model file
            prompt_template (str): prompt template"""
        self.model_file = model_file
        self.prompt_template = prompt_template
        self.callback_manager = CallbackManager([StreamingCallbackHandler()])
        self.llm = LlamaCpp(
            model_path=model_file,
            temperature=0.1,
            n_gpu_layers=0,
            n_batch=256,
            callback_manager=self.callback_manager,
            verbose=True,
        )

    def prompt(self, user_input: str):
        """ Give LLM an input prompt """
        prompt = PromptTemplate(template=self.prompt_template, input_variables=["question"])
        chain = prompt | self.llm | StrOutputParser()
        chain.invoke({"question": user_input}, config={})


class StreamingCallbackHandler(StreamingStdOutCallbackHandler):
    """ Callback handler for streaming """

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when LLM starts running."""
        logging.debug("DEBUG: LLM started")

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Run when LLM ends running."""
        logging.debug("DEBUG: LLM ended")

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        print(f"{token}", end="")
