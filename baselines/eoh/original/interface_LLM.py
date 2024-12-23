import http.client
import json
import tiktoken
import logging
from utils.utils import chat_completion, format_messages


class ConfigEoH:
    def __init__(self, model):
        self.model = model


class InterfaceAPI:
    def __init__(self, api_endpoint, api_key, model_LLM, debug_mode):
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model_LLM = model_LLM
        self.debug_mode = debug_mode
        self.n_trial = 5
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def cal_usage_LLM(self, lst_prompt, lst_completion, encoding_name="cl100k_base"):
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.get_encoding(encoding_name)
        for i in range(len(lst_prompt)):
            for message in lst_prompt[i]:
                for key, value in message.items():
                    self.prompt_tokens += len(encoding.encode(value))

            self.completion_tokens += len(encoding.encode(lst_completion[i]))

    def get_response(self, prompt_content):
        pre_messages = {"system": "", "user": prompt_content}
        cfg = ConfigEoH(model=self.model_LLM)
        messages = format_messages(cfg, pre_messages)
        #logging.info(f"Prompt: {prompt_content}")
        response = chat_completion(1, [messages[1]], self.model_LLM, temperature=1.)
        response = response[0].message.content
        self.cal_usage_LLM([messages], [response])
        logging.info(f"LLM usage: prompt_tokens = {self.prompt_tokens}, completion_tokens = {self.completion_tokens}")

        return response