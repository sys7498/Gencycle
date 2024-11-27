from transformers import AutoModelForCausalLM, AutoTokenizer


class ObjectGenerator:
    def __init__(self):
        self.model_path = "Zhengyi/LLaMA-Mesh"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path)