import re
import pandas as pd
from test_data_generation.processors.base_processor import BaseProcessor

class DatasetMetamathProcessor(BaseProcessor):
    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df['question'] = self.formulate_question(df)
        df_math = df[df['type'].str.contains('MATH')].copy()
        df_gsm = df[df['type'].str.contains('GSM')].copy()

        # Process the MATH type questions
        df_math['operation'] = self.formulate_operation(df_math)
        # Apply the retrieval_from_answer once and unpack the results
        df_math[['correct_answer', 'rationale']] = df_math['response'].apply(self.retrieve_answer_parts_math).apply(pd.Series)
        
        # Process the GSM type questions
        df_gsm[['correct_answer', 'operation', 'rationale']] = df_gsm['response'].apply(self.retrieve_answer_parts_gsm).apply(pd.Series)

        # Concatenate the two dataframes
        df = pd.concat([df_math, df_gsm])
        df = self.remove_correct_answer_with_na_and_multiple_numbers(df)
        
        # Returning only the necessary columns
        df = df[['question', 'correct_answer', 'operation', 'rationale']]
        return df
    
    def formulate_question(self, df: pd.DataFrame) -> pd.Series:
        # Custom logic for Dataset MATH's question formulation
        return df['query']
    
    def retrieve_answer_parts_math(self, answer: str) -> tuple:
        # Get the answer from \box{...}
        # Then replace \box{...} to ... to generate rationale
        try:
            assert len(answer.split("The answer is: ")) == 2
            rat, ans = answer.split("The answer is: ")
            ans = ans.strip()
            rat = re.sub(r'\\boxed{(.+?)}', r'\1', rat)
            rat = rat.strip()
        except:
            ans = self.na_str
            rat = self.na_str

        # Return a tuple with all the required information (answer, rationale)
        return ans, rat
    
    def retrieve_answer_parts_gsm(self, answer: str) -> tuple:
        # Get the answer, rationale, and operations from the answer string
        try:
            assert len(answer.split("The answer is: ")) == 2
            rat, ans = answer.split("The answer is: ")
            ans = ans.strip()
            assert len(rat.split("\n#### ")) == 2
            rat = rat.split("\n#### ")[0]
            rat = rat.strip()
        except:
            ans = self.na_str
            rat = self.na_str

        oprt = self.na_str

        # Return a tuple with all the required information (answer, operation, rationale)
        return ans, oprt, rat