import re
from utils.generator.chat_response_generator import ChatResponseGenerator

class CoTGenerator(ChatResponseGenerator):
    """ Generate a list of thoughts for the given question using the specified model. 
        Either OpenAI's GPT-4o's or Together's Models' API can be used.
        Each thought in a CoT list is represented as (sentence, label) where label is one of [evidence], [reasoning], and [claim].
    """
    def __init__(self, model_name, chat_history=[], api_key=None, local=False):
        super().__init__(model_name, chat_history, api_key, local)
        
    def generate_cot_list(self, query, **kwargs):
        
        def extract_sentence(input_string):
            # Regular expression to match the format with the number, sentence, and tag
            pattern = re.compile(r'^(\d+\.\s)(.*?)$')
            
            # Search the pattern in the input string
            match = pattern.search(input_string)
            
            if match:
                # Extract the sentence
                sentence = match.group(2)
                return sentence
            else:
                # Return the input string as it is
                return input_string
            
        cot_texts = self.generate_response(query, **kwargs)
        multi_cot_list = {}
        
        for i, cot_text in enumerate(cot_texts):
            cot_list = [extract_sentence(cot) for cot in cot_text.split("\n")]
        
            # Remove None values from the list
            cot_list = [cot.strip() for cot in cot_list if cot is not None]
            cot_list = [cot for cot in cot_list if len(cot) > 10]
            
            multi_cot_list[f"c_{i}"] = cot_list
        
        return multi_cot_list
