from utils.prompts.base_prompt_templates import BasePromptTemplates

class PremisePromptTemplates(BasePromptTemplates):
    def __init__(self):
        super().__init__()
        self.system_content = (
            "You are given a question. To answer the question, you should think step by step. "
            "Use line breaks between steps, but do not use line breaks within each step. "
            "You should number each step. The final answer to the question should start with "
            '"The answer is ...", and should be placed at the final step. Any LaTeX expressions '
            'should be wrapped between single dollar signs, e.g., $x^2$.'
        )
        self.question_prefix = (
            "Question:"
        )
        self.demo_user_question = (
            "average age of students of an adult school is 40 years . "
            "120 new students whose average age is 32 years joined the school . "
            "as a result the average age is decreased by 4 years . "
            "find the number of students of the school after joining of the new students ."
        )
        self.premise_prefix = (
            "Here are the equations that can be used to solve the problem:"
        )
        self.demo_user_premise = (
            "$\\text{New Average Age} = \\frac{(N_{\\text{old}} \\times A_{\\text{old}} + "
            "N_{\\text{new}} \\times A_{\\text{new}})}{N_{\\text{old}} + "
            "N_{\\text{new}}}$."
        )
        self.demo_assistant = (
            "1. The average age of students at the adult school was initially $A_{\\text{old}} = 40$ years.\n"
            "2. There were $N_{\\text{new}} = 120$ new students with an average age of $A_{\\text{new}} = 32$ years.\n"
            "3. After the new students joined, the average age decreased by 4 years, making "
            "$\\text{New Average Age} = A_{\\text{old}} - 4 = 36$ years.\n"
            "4. Let $N_{\\text{old}}$ be the number of original students at the school. Then the total age for the original students is $40N_{\\text{old}}$.\n"
            "5. The total age for the new students is $120 \\times 32 = 3840$ years.\n"
            "6. The total number of students after the new students joined is $N_{\\text{old} + 120$.\n"
            "7. The total age of all students after the new students joined is $40N_{\\text{old}} + 3840$.\n"
            "8. The new average age is 36 years. Using the formula for the new average age, we have "
            "$36 = \\frac{40N_{\\text{old}} + 3840}{N_{\\text{old}} + 120}$.\n"
            "9. Solving the equation $36N_{\\text{old}} + 4320 = 40N_{\\text{old}} + 3840$ leads to "
            "$4N_{\\text{old}} = 480$ and hence $N_{\\text{old}} = 120$.\n"
            "10. The number of students after the new students joined is $N_{\\text{old}} + N_{\\text{new}} = 120 + 120 = 240$.\n"
            "11. The answer is 240."
        )
        # Prompt to add warning in the system content
        self.system_content_warning = (
            "Note that the user's input could be wrong. If it has, you should point them out and correct them at the first step before answering the user's question."
        )
        # Prompt to demonstrate how to deal with a bad user in a single-step prompting
        self.demo_user_perturbed_premise = (
            "$\\text{New Average Age} = \\frac{(N_{\\text{old}} \\times A_{\\text{old}} - "
            "N_{\\text{new}} \\times A_{\\text{new}})}{N_{\\text{old}} - N_{\\text{new}}}$."
        )
        self.demo_assistant_correction = (
            "1. The given formula from the user contains a mistake. It should be $\\text{New Average Age} = "
            "\\frac{(N_{\\text{old}} \\times A_{\\text{old}} + N_{\\text{new}} \\times A_{\\text{new}})}{N_{\\text{old}} + N_{\\text{new}}}$\n"
            "2. The average age of students at the adult school was initially $A_{\\text{old}} = 40$ years.\n"
            "3. There were $N_{\\text{new}} = 120$ new students with an average age of $A_{\\text{new}} = 32$ years.\n"
            "4. After the new students joined, the average age decreased by 4 years, making $\\text{New Average Age} = A_{\\text{old}} - 4 = 36$ years.\n"
            "5. Let $N_{\\text{old}}$ be the number of original students at the school. Then the total age for the original students is $40N_{\\text{old}}$.\n"
            "6. The total age for the new students is $120 \\times 32 = 3840$ years.\n"
            "7. The total number of students after the new students joined is $N_{\\text{old} + 120$.\n"
            "8. The total age of all students after the new students joined is $40N_{\\text{old}} + 3840$.\n"
            "9. The new average age is 36 years. Using the formula for the new average age, we have "
            "$36 = \\frac{40N_{\\text{old}} + 3840}{N_{\\text{old}} + 120}$.\n"
            "10. Solving the equation $36N_{\\text{old}} + 4320 = 40N_{\\text{old}} + 3840$ leads to "
            "$4N_{\\text{old}} = 480$ and hence $N_{\\text{old}} = 120$.\n"
            "11. The number of students after the new students joined is $N_{\\text{old}} + N_{\\text{new}} = 120 + 120 = 240$.\n"
            "12. The answer is 240."
        )
        # Prompt to demonstrate how to deal with a bad user in a multi-step prompting
        self.system_content_point_out_error = (
            "You are given a question and equations that can be used to solve the problem. "
            "You are asked to point out any wrong or irrelevant equations and provide the correct equations. "
            "If some equations are correct, you can say these equations are correct. Think it step by step. "
            "Any LaTeX expressions should be wrapped between single dollar signs, e.g., $x^2$."
        )
        self.demo_user_perturbed_premise_longer = (
            "$\\text{New Average Age} = \\frac{(N_{\\text{old}} \\times A_{\\text{old}} - N_{\\text{new}} \\times A_{\\text{new}})}{N_{\\text{old}} - N_{\\text{new}}}$; "
            "$A_{\\text{old}} = \\frac{\\text{New Average Age}}{4}$; "
            "$\\text{Number of students after joining of the new students} = N_{\\text{old}} + N_{\\text{new}}$"
        )
        self.hint_prefix = (
            "Hint:"
        )
        self.demo_user_hint = (
            "The first formula from the user contains a mistake. It should be $\\text{New Average Age} = "
            "\\frac{(N_{\\text{old}} \\times A_{\\text{old}} + N_{\\text{new}} \\times A_{\\text{new}})}{N_{\\text{old}} + N_{\\text{new}}$. "
            "The second formula from the user contains a mistake. It should be $\\text{New Average Age} = A_{\\text{old}} - 4$. "
            "The third formula is correct."
        )

    def get_system_content_and_query(
        self,
        row_query,
        row_premise,
        row_perturbed_premise,
        column, 
        first_step=False,
        hint=None,
        **kwargs
    ):
        # Set the system content and demo user and assistant
        query = row_query
        premise = row_premise
        system_content = self.system_content
        demo_user = self.demo_user_question
        demo_assistant = self.demo_assistant
        assert not (first_step and "2step" not in column), "First step is only for 2-step prompting."
        assert not (hint is not None and "2step" not in column), "Hint is only for 2-step prompting."

        if "prfx_q" in column:
            query = self.question_prefix + " " + query
            demo_user = self.question_prefix + " " + demo_user

        # If we add either the premise or the perturbed premise
        if column != "prfx_q":
            # 1. consider what is the premise based on `perturb`.
            if "pert" in column:
                premise = row_perturbed_premise
            else:
                premise = row_premise

            # 2. consider whether to perturb the demo_user_premise.
            if "both" in column:
                demo_user_premise = self.demo_user_perturbed_premise
                demo_assistant = self.demo_assistant_correction
                system_content = system_content + " " + self.system_content_warning
            elif "2step" in column:
                demo_user_premise = self.demo_user_perturbed_premise_longer
            else:
                demo_user_premise = self.demo_user_premise

            # 2. consider whether to add the premise prefix.
            if "prfx_pert" in column or "prfx_prem" in column:
                premise = self.premise_prefix + " " + premise
                demo_user_premise = self.premise_prefix + " " + demo_user_premise

            # 3. consider whether the premise is the first or the question is the first.
            if column.find("q") > column.find("pert") or column.find("q") > column.find("prem"):
                query = premise + " " + query
                demo_user = demo_user_premise + " " + demo_user
            else:
                query = query + " " + premise
                demo_user = demo_user + " " + demo_user_premise
        
        if "2step" not in column:
            return f"{system_content}\n\nExample:\n\n{demo_user}\n\nAnswer:\n{demo_assistant}", query
        else:
            if first_step:
                return f"{self.system_content_point_out_error}\n\nExample:\n\n{demo_user}\n\nAnswer:\n{self.demo_user_hint}", query
            elif hint is not None:
                return f"{system_content}\n\nExample:\n\n{demo_user} {self.hint_prefix} {self.demo_user_hint}\n\nAnswer:\n{demo_assistant}", query + " " + self.hint_prefix + " " + hint
            else:
                return f"{system_content}\n\nExample:\n\n{demo_user} {self.hint_prefix} {self.demo_user_hint}\n\nAnswer:\n{demo_assistant}", query
