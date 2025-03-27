import pandas as pd
import re

class BaseProcessor:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.na_str = "n/a"
        self.concat_str = "|"

    def remove_correct_answer_with_na_and_multiple_numbers(self, df: pd.DataFrame) -> pd.DataFrame:
        # Drop rows where 'correct_answer' is self.na_str
        df = df[df['correct_answer'] != self.na_str]
        
        # Define the function to check if 'correct_answer' is a single number
        def is_single_number(s):
            s = s.strip()
            if '(' in s or ')' in s:
                return False
            pattern = r'^-?\d{1,3}(,\d{3})*(\.\d+)?$'
            return re.match(pattern, s) is not None
        
        # Apply the function to the 'correct_answer' column
        df = df[df['correct_answer'].apply(is_single_number)]
        
        return df
    
    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df['question'] = self.formulate_question(df)
        df['correct_answer'] = self.formulate_correct_answer(df)
        df = self.remove_correct_answer_with_na_and_multiple_numbers(df)
        df['operation'] = self.formulate_operation(df)
        df['rationale'] = self.formulate_rationale(df)
        # Returning only the necessary columns
        df = df[['question', 'correct_answer', 'operation', 'rationale']]
        return df
    
    def formulate_question(self, df: pd.DataFrame) -> pd.Series:
        # Default implementation for formulating questions, if applicable
        return df.apply(lambda x: f"{self.na_str}", axis=1)

    def formulate_correct_answer(self, df: pd.DataFrame) -> pd.Series:
        # Default implementation for formulating questions, if applicable
        return df.apply(lambda x: f"{self.na_str}", axis=1)
    
    def formulate_operation(self, df: pd.DataFrame) -> pd.Series:
        # Default implementation for formulating questions, if applicable
        return df.apply(lambda x: f"{self.na_str}", axis=1)
    
    def formulate_rationale(self, df: pd.DataFrame) -> pd.Series:
        # Default implementation for formulating questions, if applicable
        return df.apply(lambda x: f"{self.na_str}", axis=1)
