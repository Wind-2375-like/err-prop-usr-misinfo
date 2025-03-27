import re
import pandas as pd
from test_data_generation.processors.base_processor import BaseProcessor

class DatasetMathProcessor(BaseProcessor):
    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df['question'] = self.formulate_question(df)
        df['operation'] = self.formulate_operation(df)
        # Apply the retrieval_from_answer once and unpack the results
        df[['correct_answer', 'rationale']] = df['solution'].apply(self.retrieve_answer_parts).apply(pd.Series)
        df = self.remove_correct_answer_with_na_and_multiple_numbers(df)
        
        # Returning only the necessary columns
        df = df[['question', 'correct_answer', 'operation', 'rationale']]
        return df
    
    def formulate_question(self, df: pd.DataFrame) -> pd.Series:
        # Custom logic for Dataset MATH's question formulation
        return df['problem']
    
    def retrieve_answer_parts(self, answer: str) -> tuple:
        # Get the answer from \box{...}
        # Then replace \box{...} to ... to generate rationale
        try:
            ans = re.search(r'\\boxed{(.+?)}', answer).group(1)
            rat = re.sub(r'\\boxed{(.+?)}', r'\1', answer)
        except:
            ans = self.na_str
            rat = self.na_str

        # Return a tuple with all the required information (answer, rationale)
        return ans, rat
