import pandas as pd
import re
from test_data_generation.processors.base_processor import BaseProcessor

class DatasetGsm8kProcessor(BaseProcessor):
    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df['question'] = self.formulate_question(df)
        # Apply the retrieval_from_answer once and unpack the results
        df[['correct_answer', 'operation', 'rationale']] = df['answer'].apply(self.retrieve_answer_parts).apply(pd.Series)
        df = self.remove_correct_answer_with_na_and_multiple_numbers(df)
        
        # Returning only the necessary columns
        df = df[['question', 'correct_answer', 'operation', 'rationale']]
        return df
    
    def retrieve_answer_parts(self, answer: str) -> tuple:
        # Get the answer, rationale, and operations from the answer string
        try:
            assert len(answer.split("\n#### ")) == 2
            rat, ans = answer.split("\n#### ")
            ans = ans.strip()
            rat = rat.strip()
        except:
            ans = self.na_str
            rat = self.na_str

        # Get the operation from the answer string enclosed in <<...>>
        try:
            operations = re.findall(r'<<.*?>>', rat)
            oprt = self.concat_str.join([op[2:-2] for op in operations])
            # Remove all texts in <<...>> from the answer string
            rat = re.sub(r'<<.*?>>', '', rat)
        except:
            oprt = self.na_str

        # Return a tuple with all the required information (answer, operation, rationale)
        return ans, oprt, rat
    
    def formulate_question(self, df: pd.DataFrame) -> pd.Series:
        # Custom logic for Dataset GSM8K's question formulation
        return df['question']
