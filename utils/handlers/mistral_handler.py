from utils.handlers.base_model_handler import BaseModelHandler
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class MistralHandler(BaseModelHandler):
    def initialize_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def format_messages(self, messages):
        processed_messages = []
        for role, content in messages:
            processed_messages.append({"role": role, "content": content})
        return processed_messages

    def generate_response(self, formatted_input, **kwargs):
        formatted_input = self.tokenizer.apply_chat_template(
            formatted_input,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True
        )
        with torch.no_grad():
            generated_ids = self.model.generate(
                **formatted_input,
                **kwargs
            )
        result = [self.tokenizer.decode(generated_id, skip_special_tokens=True) for generated_id in generated_ids]
        return result