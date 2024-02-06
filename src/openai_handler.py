from typing import Callable
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain.schema.output_parser import StrOutputParser

class OpenAIHandler:
    """ OpenAI API Handler """
    def __init__(self, 
                 openai_endpoint, 
                 openai_api_key, 
                 prompt_template="You're a helpful assistant. Provide a short answer to the question.\nQuestion: {question}\nAnswer:"):
        self.llm = OpenAI(openai_api_key=openai_api_key,
                          base_url=openai_endpoint)
        self.prompt_template = prompt_template

    def inference(self, question: str, callback: Callable):
        """ Ask OpenAI model a question """
        prompt = PromptTemplate(template=self.template,
                                input_variables=["question"])

        chain = self.prompt_template | self.llm | StrOutputParser()
        for token in chain.stream({"question": question}):
            callback(token)