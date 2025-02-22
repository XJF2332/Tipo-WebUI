import os
import random

from llama_cpp import Llama
from kgen.metainfo import SPECIAL, POSSIBLE_QUALITY_TAGS, RATING_TAGS
from kgen.formatter import seperate_tags, apply_format


class TipoGen:
    def __init__(self):
        self.response = []
        self.prompt = ""
        self.tag_map = {}
        self.formatted = ""
        self.model_list = os.listdir("models")
        self.model = self.model_list[0]
        self.llm = Llama(model_path=f"models/{self.model_list[0]}", n_gpu_layers=-1, n_ctx=2048)
        self.temperature = 0.5
        self.top_p = 0.95
        self.top_k = 45
        self.repetition_penalty = 1.17
        self.max_new_tokens = 128
        self.seed_max = 2**32-1
        self.default_format = """<|special|>, 
<|characters|>, <|copyrights|>, 
<|artist|>, 

<|general|>,

<|extended|>.

<|quality|>, <|meta|>, <|rating|>"""

    def unload_model(self):
        self.llm = None


    def load_model(self, index):
        self.unload_model()
        self.model = self.model_list[index]
        self.llm = Llama(model_path=f"models/{self.model}", n_gpu_layers=-1, n_ctx=2048)


    def gen_prompt(self):
        self.response = self.llm.create_completion(
            prompt=self.prompt,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repeat_penalty=self.repetition_penalty,
            max_tokens=self.max_new_tokens,
            seed=random.randint(0, self.seed_max)
        )
        self.tag_map = seperate_tags(self.response["choices"][0]["text"])
        self.formatted = apply_format(self.tag_map, self.default_format)

# test
Generator=TipoGen()
Generator.prompt = """
1girl, solo, 
"""
Generator.max_new_tokens = 1024
Generator.model_index = 1
Generator.load_model(1)
print(Generator.model)
print("-"*50)
Generator.gen_prompt()
print(Generator.response["choices"][0]["text"])
print("-"*50)
print(Generator.tag_map)
print("-"*50)
print(Generator.formatted)
