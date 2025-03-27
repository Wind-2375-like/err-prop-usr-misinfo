import pandas as pd
from test_data_generation.processors.base_processor import BaseProcessor

class DatasetMathqaProcessor(BaseProcessor):
    def formulate_question(self, df: pd.DataFrame) -> pd.Series:
        # Custom logic for Dataset MathQA's question formulation
        return df['Problem'].apply(lambda x: x.strip())

    def formulate_correct_answer(self, df: pd.DataFrame) -> pd.Series:
        # Custom logic for Dataset MathQA's correct answer formulation
        def extract_correct_answer(row):
            for i, option in enumerate(row["options"].split(", ")):
                try:
                    # Split by a ) or b ) etc.
                    assert len(option.split(f"{chr(97+i)} )")) in [2, 3]
                    text = option.split(f"{chr(97+i)} )")[-1]
                    if chr(97+i) == row["correct"]:
                        return text.strip()
                except:
                    return self.na_str
        return df.apply(extract_correct_answer, axis=1).apply(pd.Series)
    
    def formulate_operation(self, df: pd.DataFrame) -> pd.Series:
        # Custom logic for Dataset MathQA's operation formulation
        def concatenate_formula(row):
            return f"{row['annotated_formula']}{self.concat_str}{row['linear_formula']}".strip()
        return df.apply(concatenate_formula, axis=1)
    
    def formulate_rationale(self, df: pd.DataFrame) -> pd.Series:
        # Custom logic for Dataset MathQA's rationale formulation
        return df['Rationale'].apply(lambda x: x.strip())
