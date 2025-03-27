from utils.generator.chat_response_generator import ChatResponseGenerator

# First generate the premises for the given question using the GPT-4o model
class PremiseGenerator(ChatResponseGenerator):
    """ Generate a list of premises for the given question using the specified model. 
        Either OpenAI's GPT-4o's or Together's Models' API can be used.
    """
    def __init__(self, model_name, chat_history=[], api_key=None, local=False):
        self.chat_history = chat_history
        super().__init__(model_name, self.chat_history, api_key, local)
        
    def generate_premises(self, query, **kwargs):
        return self.generate_response(query, **kwargs)