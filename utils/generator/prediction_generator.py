from utils.generator.cot_generator import CoTGenerator
from utils.generator.perturbation_generator import PerturbationGenerator
import numpy as np
import re
import torch
import inflect
from utils.handlers.meta_vllama_handler import MetaVLlamaHandler

class PremisePredictionGenerator(CoTGenerator):
    """ Allows for customization of the CoT generation process.
        1. Which steps to include in the CoT list generation process.
        2. Whether to perturb the steps.
        3. Whether to include instruction and demonstration hints.
    """
    def __init__(self, model_name, chat_history=[], api_key=None, local=False):
        super().__init__(model_name, chat_history, api_key, local)
        # Prompt related to normal input
        self.system_content = "You are given a question. To answer the question, you should think step by step. Use line breaks between steps, but do not use line breaks within each step. You should number each step. The final answer to the question should start with \"The answer is ...\", and should be placed at the final step. Any LaTeX expressions should be wrapped between single dollar signs, e.g., $x^2$."
        self.question_prefix = "Question:"
        self.demo_user_question = self.question_prefix + " " + "average age of students of an adult school is 40 years . 120 new students whose average age is 32 years joined the school . as a result the average age is decreased by 4 years . find the number of students of the school after joining of the new students ."
        self.premise_prefix = "Here are the equations that can be used to solve the problem:"
        self.demo_user_premise = self.premise_prefix + " " + "$\\text{New Average Age} = \\frac{(N_{\\text{old}} \\times A_{\\text{old}} + N_{\\text{new}} \\times A_{\\text{new}})}{N_{\\text{old}} + N_{\\text{new}}}$; \\text{New Average Age} = A_{\\text{old}} - 4; $\\text{Number of students after joining of the new students} = N_{\\text{old}} + N_{\\text{new}}$."
        self.demo_assistant = "1. The average age of students at the adult school was initially $A_{\\text{old}} = 40$ years.\n2. There were $N_{\\text{new}} = 120$ new students with an average age of $A_{\\text{new}} = 32$ years.\n3. After the new students joined, the average age decreased by 4 years, making $\\text{New Average Age} = A_{\\text{old}} - 4 = 36$ years.\n4. Let $N_{\\text{old}}$ be the number of original students at the school. Then the total age for the original students is $40N_{\\text{old}}$.\n5. The total age for the new students is $120 \\times 32 = 3840$ years.\n6. The total number of students after the new students joined is $N_{\\text{old} + 120$.\n7. The total age of all students after the new students joined is $40N_{\\text{old}} + 3840$.\n8. The new average age is 36 years. Using the formula for the new average age, we have $36 = \\frac{40N_{\\text{old}} + 3840}{N_{\\text{old}} + 120}$.\n9. Solving the equation $36N_{\\text{old}} + 4320 = 40N_{\\text{old}} + 3840$ leads to $4N_{\\text{old}} = 480$ and hence $N_{\\text{old}} = 120$.\n10. The number of students after the new students joined is $N_{\\text{old}} + N_{\\text{new}} = 120 + 120 = 240$.\n11. The answer is 240."
        # Prompt to add warning in the system content
        self.system_content_warning = "Note that the user's input could be wrong. If it has, you should point them out and correct them."
        # Prompt to demonstrate how to deal with a bad user in a single-step prompting
        self.demo_user_perturbed_premise = self.premise_prefix + " " + "$\\text{New Average Age} = \\frac{(N_{\\text{old}} \\times A_{\\text{old}} - N_{\\text{new}} \\times A_{\\text{new}})}{N_{\\text{old}} - N_{\\text{new}}}$; $A_{\\text{old}} = \\frac{\\text{New Average Age}}{4}$; $\\text{Number of students after joining of the new students} = N_{\\text{old}} + N_{\\text{new}}$."
        self.demo_misinformation_correction = "The first formula from the user contains a mistake. It should be $\\text{New Average Age} = \\frac{(N_{\\text{old}} \\times A_{\\text{old}} + N_{\\text{new}} \\times A_{\\text{new}})}{N_{\\text{old}} + N_{\\text{new}}}$. The second formula from the user contains a mistake. It should be $\\text{New Average Age} = A_{\\text{old}} - 4$. The third formula is correct."
        self.demo_misinformation_point_out_only = "The first formula from the user contains a mistake. The second formula from the user contains a mistake. The third formula is correct."
        self.demo_assistant_correction = "1. " + self.demo_misinformation_correction + "\n2. The average age of students at the adult school was initially $A_{\\text{old}} = 40$ years.\n3. There were $N_{\\text{new}} = 120$ new students with an average age of $A_{\\text{new}} = 32$ years.\n4. After the new students joined, the average age decreased by 4 years, making $\\text{New Average Age} = A_{\\text{old}} - 4 = 36$ years.\n5. Let $N_{\\text{old}}$ be the number of original students at the school. Then the total age for the original students is $40N_{\\text{old}}$.\n6. The total age for the new students is $120 \\times 32 = 3840$ years.\n7. The total number of students after the new students joined is $N_{\\text{old} + 120$.\n8. The total age of all students after the new students joined is $40N_{\\text{old}} + 3840$.\n9. The new average age is 36 years. Using the formula for the new averagen_cot_listge age, we have $36 = \\frac{40N_{\\text{old}} + 3840}{N_{\\text{old}} + 120}$.\n10. Solving the equation $36N_{\\text{old}} + 4320 = 40N_{\\text{old}} + 3840$ leads to $4N_{\\text{old}} = 480$ and hence $N_{\\text{old}} = 120$.\n11. The number of students after the new students joined is $N_{\\text{old}} + N_{\\text{new}} = 120 + 120 = 240$.\n12. The answer is 240."
        # Prompt to demonstrate how to deal with a bad user in a multi-step prompting
        self.system_content_point_out_error = "You are given a question and equations that can be used to solve the problem. You are asked to point out any wrong equations and provide the correct ones. If some equations are correct, you can say these equations are correct. Any LaTeX expressions should be wrapped between single dollar signs, e.g., $x^2$. Only point out and correct the equations. Only point out and correct the equations. Only point out and correct the equations. Answer in one line. Answer in one line. Answer in one line.\n\nExample:\n\nHere are the equations that can be used to solve the problem: $\\text{New Average Age} = \\frac{(N_{\\text{old}} \\times A_{\\text{old}} + N_{\\text{new}} \\times A_{\\text{new}})}{N_{\\text{old}} + N_{\\text{new}}}$; \\text{New Average Age} = A_{\\text{old}} - 4; $\\text{Number of students after joining of the new students} = N_{\\text{old}} + N_{\\text{new}}$. Question: average age of students of an adult school is 40 years . 120 new students whose average age is 32 years joined the school . as a result the average age is decreased by 4 years . find the number of students of the school after joining of the new students .\n\nCorrection:\nThe first formula from the user contains a mistake. It should be $\\text{New Average Age} = \\frac{(N_{\\text{old}} \\times A_{\\text{old}} + N_{\\text{new}} \\times A_{\\text{new}})}{N_{\\text{old}} + N_{\\text{new}}}$. The second formula from the user contains a mistake. It should be $\\text{New Average Age} = A_{\\text{old}} - 4$. The third formula is correct."
        self.system_content_point_out_error_only = "You are given a question and equations that can be used to solve the problem. You are asked to determine the correctness of each equation. If some equations are correct, you can say these equations are correct. Only determine the correctness of each equation. Only determine the correctness of each equation. Only determine the correctness of each equation. Answer in one line. Answer in one line. Answer in one line.\n\nExample:\n\nHere are the equations that can be used to solve the problem: $\\text{New Average Age} = \\frac{(N_{\\text{old}} \\times A_{\\text{old}} + N_{\\text{new}} \\times A_{\\text{new}})}{N_{\\text{old}} + N_{\\text{new}}}$; \\text{New Average Age} = A_{\\text{old}} - 4; $\\text{Number of students after joining of the new students} = N_{\\text{old}} + N_{\\text{new}}$. Question: average age of students of an adult school is 40 years . 120 new students whose average age is 32 years joined the school . as a result the average age is decreased by 4 years . find the number of students of the school after joining of the new students .\n\nCorrectness of Equations:\nThe first formula from the user contains a mistake. The second formula from the user contains a mistake. The third formula is correct."
        self.system_content_correct_user_input = "You are given an user's input that may contain mistakes. The feedback for the input is also provided. You should correct the user's input based on the feedback and formulate a new input to solve user's question. If the user's input is correct, you can keep it as it is."

    def process_input(self, model_name, row, perturb=None, warning=False):
        query = self.question_prefix + " " + row["question"]

        # Set the system content and demo user and assistant
        system_content = self.system_content
        demo_user = self.demo_user_question
        demo_assistant = self.demo_assistant

        # If we add either the premise or the perturbed premise
        if warning:
            demo_user_premise = self.demo_user_perturbed_premise
            demo_assistant = self.demo_assistant_correction
            system_content = system_content + " " + self.system_content_warning
            demo_user = demo_user_premise + " " + demo_user
            
        if perturb is not None:
            # 1. consider what is the premise based on `perturb`.
            if perturb:
                premise = row["perturbed_premise"][model_name]
            else:
                premise = row["premise"][model_name]
            premise = self.premise_prefix + " " + premise
            if premise[-1] != ".":
                premise = premise + "."

            # 2. consider whether to add warnings.
            if not warning:
                demo_user_premise = self.demo_user_premise
                demo_user = demo_user_premise + " " + demo_user
                
            # 3. add premise before the question.
            query = premise + " " + query
        return query, system_content, demo_user, demo_assistant
    
    def provide_feedback(self, query, demo_user, point_out_only=False, temperature=0.7, top_p=0.7, top_k=50):
        # 1. get the error correction from the first prompting
        if point_out_only:
            self.update_chat_history([
                ("system", f"{self.system_content_point_out_error_only}")
            ])
            prefix = "Correctness of Equations:"
        else:
            self.update_chat_history([
                ("system", f"{self.system_content_point_out_error}")
            ])
            prefix = "Correction:"
        
        feedback = self.generate_response(
            query+"\n\n"+prefix,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k
        )[0].split(prefix)[-1].strip()
        return feedback
    
    def correct_query_with_feedback(self, query, demo_user, feedback, perturb=None, temperature=0.7, top_p=0.7, top_k=50):
        # 2. correct the user's input based on the feedback
        if perturb is not None:
            demo_corrected_input = self.demo_user_question
        else:
            demo_corrected_input = self.demo_user_premise + " " + self.demo_user_question
        # self.update_chat_history([
        #     ("system", f"{self.system_content_correct_user_input}\n\nExample:\n\n{demo_user}\n\nFeedback:\n{feedback}\n\nAnswer:\n{demo_corrected_input}")
        # ])
        # corrected_query = self.generate_response(
        #     f"{query}\n\nFeedback:\n{feedback}",
        #     temperature=temperature,
        #     top_p=top_p,
        #     top_k=top_k
        # )[0]
        corrected_query = f"{query} Hint: {feedback}"
        return corrected_query
    
    def correct_query(self, query, demo_user, point_out_only=False, perturb=None, temperature=0.7, top_p=0.7, top_k=50):
        # 1. get the error correction from the first prompting
        feedback = self.provide_feedback(query, demo_user, point_out_only, temperature, top_p, top_k)
        # 2. correct the user's input based on the feedback
        corrected_query = self.correct_query_with_feedback(query, demo_user, feedback, perturb, temperature, top_p, top_k)    
        return corrected_query, feedback
        
    def run_premise_experiment(self, row, model_name, perturb=None, input_correction=False, warning=False, temperature=0.7, top_p=0.7, top_k=50, n=1):
        assert type(model_name)==str, "Model name should be a string."
        assert type(perturb)==bool or perturb is None, "Perturb should be a boolean or None."
        assert type(input_correction)==bool, "Input correction should be a boolean."
        assert type(warning)==bool, "Warning instruction should be a boolean."
        
        query, system_content, demo_user, demo_assistant = self.process_input(model_name, row, perturb, warning)
        
        # If we correct the user's input at the first CoT step
        if not input_correction:
            # Update the chat history
            self.update_chat_history([
                ("system", f"{system_content}\n\nExample:\n\n{demo_user}\n\nAnswer:\n{demo_assistant}")
            ])
            
            # Generate the CoT list and evaluate it
            # Format: {"c_0": [sentence1, sentence2, ...], "c_1": [sentence1, sentence2, ...], ...}
            gen_cot_list = self.generate_cot_list(
                query,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                n=n
            )
            # Return the result
            return gen_cot_list
        else:
            corrected_query, feedback = self.correct_query(query, demo_user, point_out_only=False, perturb=perturb, temperature=temperature, top_p=top_p, top_k=top_k)
            # 3. insert the corrected input for the second prompting
            self.update_chat_history([
                ("system", f"{system_content}\n\nExample:\n\n{demo_user} Hint: {self.demo_misinformation_correction}\n\nAnswer:\n{demo_assistant}")
            ])
            # Generate the CoT list and evaluate it
            gen_cot_list = self.generate_cot_list(
                corrected_query,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                n=n
            )
            # Append feedback to the beginning of each response
            for ci in gen_cot_list:
                gen_cot_list[ci] = [feedback] + gen_cot_list[ci]
                
            # Return the result
            return gen_cot_list


class CounterfactualPremisePredictionGenerator(PremisePredictionGenerator):
    """ Allows for customization of the CoT generation process.
        1. Which steps to include in the CoT list generation process.
        2. Whether to perturb the steps.
        3. Whether to include instruction and demonstration hints.
    """
    def __init__(self, model_name, chat_history=[], api_key=None, local=False):
        super().__init__(model_name, chat_history, api_key, local)
        self.perturbation_generator = PerturbationGenerator(model_name="gpt-4o", api_key=api_key)
        self.inflect_engine = inflect.engine()
        self.model_names = [
            "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
            "Qwen/Qwen2-72B-Instruct",
            "mistralai/Mixtral-8x22B-Instruct-v0.1",
            "gpt-4o-mini",
        ]

    def provide_feedback_counterfactual(self, query, row, model_name, demo_user, point_out_only=False, bad_quality=False, temperature=0.7, top_p=0.7, top_k=50):
        perturbed = row["perturbed_premise"][model_name]
        # Extract all LaTeX formulas (enclosed in $...$)
        perturbed_formulas = re.findall(r'\$([^$]*)\$', perturbed)
        if not point_out_only:
            if not bad_quality:
                correction = row["premise"][model_name]
                correct_formulas = re.findall(r'\$([^$]*)\$', correction)
            else:
                # Randomly select perturbed_premise from other model_name as the correct formula
                diff_model_name = "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo"
                correct_formulas = re.findall(r'\$([^$]*)\$', row["perturbed_premise"][diff_model_name])
                np.random.seed(42)
                try:
                    correct_formulas = np.random.choice(correct_formulas, len(perturbed_formulas)).tolist()
                except:
                    correct_formulas = []
            # Truncate the two lists to the same maximum length
            max_len = min(len(perturbed_formulas), len(correct_formulas))
            correct_formulas = correct_formulas[:max_len]
            if max_len == 0:
                return "The given formula from the user is correct."
            # Return a prompt to simulate the correction
            if len(perturbed_formulas) == 1:
                return f"The given formula from the user contains a mistake. It should be ${correct_formulas[0]}$."
            else:
                ret = ""
                for i, _ in enumerate(perturbed_formulas):
                    if i < len(correct_formulas) and perturbed_formulas[i] != correct_formulas[i]:
                        ret += f"The {self.inflect_engine.number_to_words(self.inflect_engine.ordinal(i+1))} formula from the user contains a mistake. It should be ${correct_formulas[min(i, len(correct_formulas)-1)]}$. "
                    else:
                        ret += f"The {self.inflect_engine.number_to_words(self.inflect_engine.ordinal(i+1))} formula from the user is correct. "
                return ret.strip()   
        else:
            ret = ""
            if not bad_quality: 
                for i, _ in enumerate(perturbed_formulas):
                    ret += f"The {self.inflect_engine.number_to_words(self.inflect_engine.ordinal(i+1))} formula from the user contains a mistake. "
                return ret.strip()
            else: 
                for i, _ in enumerate(perturbed_formulas):
                    ret += f"The {self.inflect_engine.number_to_words(self.inflect_engine.ordinal(i+1))} formula from the user is correct. "
                return ret.strip()
        
    def run_premise_experiment(self, row, model_name, perturb=None, input_correction=False, warning=False, pos="user", point_out_only=False, bad_quality=False, temperature=0.7, top_p=0.7, top_k=50, n=1):
        assert type(model_name)==str, "Model name should be a string."
        assert type(perturb)==bool or perturb is None, "Perturb should be a boolean or None."
        assert type(input_correction)==bool, "Input correction should be a boolean."
        assert type(warning)==bool, "Warning instruction should be a boolean."
        assert pos in ["user", "step0", "cot"], "Position should be either 'user', 'step0', or 'cot'."
        assert type(point_out_only)==bool, "Point out only should be a boolean."
        assert type(bad_quality)==bool, "Bad quality should be a boolean."

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
        
        query, system_content, demo_user, demo_assistant = self.process_input(model_name, row, perturb, warning)
        feedback = self.provide_feedback_counterfactual(query, row, model_name, demo_user, point_out_only, bad_quality, temperature, top_p, top_k)  
        
        if pos == "user":
            # 1. insert the correction into prompting
            corrected_query = self.correct_query_with_feedback(query, demo_user, feedback, perturb, temperature, top_p, top_k)
            if point_out_only:
                demo_user = f"{demo_user} Hint: {self.demo_misinformation_point_out_only}"
            else:
                demo_user = f"{demo_user} Hint: {self.demo_misinformation_correction}"
            self.update_chat_history([
                ("system", f"{system_content}\n\nExample:\n\n{demo_user}\n\nAnswer:\n{demo_assistant}")
            ])
            # Generate the CoT list and evaluate it
            gen_cot_list = self.generate_cot_list(
                corrected_query,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                n=n
            )
            # Return the result
            return gen_cot_list
        else:
            cot_dict = {}
            result_dict = {}
            
            if model_name != "human":
                model_name = "self"
            perturb_cots = row["prfx_pert_prfx_q"][model_name]["c_0"][:-1]
            len_pert_cot = len(perturb_cots)
            
            cot_dict["lte_s_0"] = f"1. {feedback}\n2."
            if pos != "step0":
                cot_text_lte_s = ""
                for i in range(len_pert_cot):
                    cot_text_lte_s += f"{i+1}. {perturb_cots[i]}\n"
                    cot_dict[f"lte_s_{i+1}"] = cot_text_lte_s+f"{i+2}. {feedback}\n{i+3}."
            for si, cot in cot_dict.items():
                messages = [
                    {"role": "system", "content": f"{system_content}\n\nExample:\n\n{demo_user}\n\nAnswer:\n{demo_assistant}"},
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": cot}
                ]
                tokenized_chat = self.client.handler.pipeline.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, return_dict=True, return_tensors="pt").to("cuda")
                # Remove <|eot_id|>
                tokenized_chat["input_ids"] = tokenized_chat["input_ids"][:, :-1]
                tokenized_chat["attention_mask"] = tokenized_chat["attention_mask"][:, :-1]
                
                gen_cot_list = {}
                # for i in range(n):
                #     outputs = self.client.handler.pipeline.model.generate(
                #         **tokenized_chat,
                #         max_new_tokens=512,
                #         temperature=temperature,
                #         top_p=top_p,
                #         top_k=top_k,
                #         num_return_sequences=1,
                #         pad_token_id=self.client.handler.pipeline.tokenizer.eos_token_id
                #     )
                #     torch.cuda.empty_cache()
                #     output = outputs[0]
                #     cot_text = self.client.handler.pipeline.tokenizer.decode(output, skip_special_tokens=True).split("assistant\n\n")[-1].strip()
                #     cot_list = [extract_sentence(cot) for cot in cot_text.split("\n")]
                
                #     # Remove None values from the list
                #     cot_list = [cot.strip() for cot in cot_list if cot is not None]
                #     cot_list = [cot for cot in cot_list if len(cot) > 10]
                    
                #     gen_cot_list[f"c_{i}"] = cot_list
                
                # result_dict[si] = gen_cot_list
                if isinstance(self.client.handler, MetaVLlamaHandler):
                    outputs = self.client.handler.pipeline.model.generate(
                        **tokenized_chat,
                        max_new_tokens=512,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        num_return_sequences=n
                    )
                else:
                    outputs = self.client.handler.pipeline.model.generate(
                        **tokenized_chat,
                        max_new_tokens=512,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        num_return_sequences=n,
                        pad_token_id=self.client.handler.pipeline.tokenizer.eos_token_id
                    )
                torch.cuda.empty_cache()
                gen_cot_list = {}
                for i, output in enumerate(outputs):
                    cot_text = self.client.handler.pipeline.tokenizer.decode(output, skip_special_tokens=True).split("assistant\n\n")[-1].strip()
                    cot_list = [extract_sentence(cot) for cot in cot_text.split("\n")]
                
                    # Remove None values from the list
                    cot_list = [cot.strip() for cot in cot_list if cot is not None]
                    cot_list = [cot for cot in cot_list if len(cot) > 10]
                    
                    gen_cot_list[f"c_{i}"] = cot_list
                
                result_dict[si] = gen_cot_list
            # Return the result
            return result_dict