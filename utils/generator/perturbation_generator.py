import re
import random
from sympy.parsing.latex import parse_latex
import sympy as sp
from utils.generator.chat_response_generator import ChatResponseGenerator
from decimal import Decimal, ROUND_HALF_UP, getcontext

# Perturb the inputs
class PerturbationGenerator(ChatResponseGenerator):
    """ Generate perturbed premises and thoughts for the given question using the specified model."""
    def __init__(self, model_name, chat_history=[], api_key=None, local=False):
        super().__init__(model_name, chat_history, api_key, local)
        self.base_system_content = "You are given a sentence that may contain some LaTeX expressions. You are required to"
        self.change_ops = []
        self.change_nums = []
        self.swap_ops = []
        
    def clear_history(self):
        self.change_ops = []
        self.change_nums = []
        self.swap_ops = []

    def format_float(self, f):
        """ Format the float to a string with a fixed number of decimal places.
            Args:
                f (float): The float to format.
            Returns:
                str: The formatted float as a string.
        """
        # Convert the float to a string, then to Decimal for precise control
        d = Decimal(str(f))

        # Get current context and set a sufficiently large precision
        context = getcontext()
        context.prec = 50  # Set precision high to handle the input float's precision

        # Scale the decimal to minimize floating-point errors effectively
        # The scale is adjusted to the number of decimal places in the input
        as_str = f"{d:f}"
        num_decimal_places = as_str[::-1].find('.')
        quantize_scale = '1' + ('.' + '0' * num_decimal_places if num_decimal_places != -1 else '')

        quantized = d.quantize(Decimal(quantize_scale), rounding=ROUND_HALF_UP)

        # Convert to string and remove trailing zeros
        str_quantized = str(quantized).rstrip('0')

        # Ensure no dangling decimal point
        if str_quantized.endswith('.'):
            str_quantized = str_quantized[:-1]

        return str_quantized
        
    def replace_text_with_symbols(self, formula_latex):
        """ Replace the \\text{...}_* or \\text{...}_{...} strings with symbols.
            Args:
                formula_latex (str): The formula in LaTeX format.
            Returns:
                str: The formula with the \\text{...}_* or \\text{...}_{...} strings replaced by symbols.
                dict: The mapping from text to symbol.
        """
        # Pattern to match \text{...}_* or \text{...}_{...}
        pattern = r'(\\text\{(?:[^{}]|\{[^{}]*\})*\}(?:_\{(?:[^{}]|\{[^{}]*\})*\}|_[^{}])?)'

        # Retrieve all matches
        text_matches = re.findall(pattern, formula_latex)
        unique_texts = set(text_matches)

        # Create a mapping from text to symbol
        def hash_text(text):
            return str(hash(text))[:5]
        
        text_to_symbol = {text: "\\Tau_{" + hash_text(text) + "}" for text in unique_texts}

        # Replace \text{...} with \Tau_{...}
        for text, symbol in text_to_symbol.items():
            formula_latex = formula_latex.replace(text, symbol)

        return formula_latex, text_to_symbol

    def extract_all_latex_expressions(self, query):
        """ Extract LaTeX expressions from the last formula in the query.
            Args:
                query (str): The query containing LaTeX expressions.
            Returns:
                list: The list of LaTeX expressions.
        """
        def replace_structures(expr):
            """
            Replace \sum_{...} and \text{...} in the expression with placeholders.
            Args:
                expr (str): The LaTeX expression.
            Returns:
                tuple:
                    expr_replaced (str): The expression with placeholders.
                    placeholder_mapping (dict): Mapping from placeholder to original structure.
            """
            placeholder_mapping = {}
            # Updated pattern to handle nested braces in \sum and \text
            sum_pattern = r'(\\sum_\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}(?:\^\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})?)'
            text_pattern = r'(\\text\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})'
            
            # Function to generate a unique placeholder
            def generate_placeholder(structure_type, index):
                return f'__{structure_type.upper()}_{index}__'
            
            # Replace all \sum_{...} first
            def replace_sums(expr):
                sums = re.findall(sum_pattern, expr)
                for idx, sum_expr in enumerate(sums):
                    placeholder = generate_placeholder('sum', idx)
                    placeholder_mapping[placeholder] = sum_expr
                    expr = expr.replace(sum_expr, placeholder)
                return expr

            # Then replace all \text{...}
            def replace_texts(expr):
                texts = re.findall(text_pattern, expr)
                for idx, text_expr in enumerate(texts):
                    placeholder = generate_placeholder('text', idx)
                    placeholder_mapping[placeholder] = text_expr
                    expr = expr.replace(text_expr, placeholder)
                return expr
            
            expr_replaced = replace_sums(expr)
            expr_replaced = replace_texts(expr_replaced)
            
            return expr_replaced, placeholder_mapping
        
        def extract_expressions(formula):
            """
            Recursively extract all individual expressions from the formula.
            Args:
                formula (str): The LaTeX formula to extract expressions from.
            Returns:
                list: A list of individual expressions.
            """
            expressions = []
            i = 0
            n = len(formula)
            current_expr = ''
            stack = []
            while i < n:
                c = formula[i]
                if c == ',' and not stack:
                    # At top level, split on comma
                    if current_expr.strip():
                        expressions.append(current_expr.strip())
                    current_expr = ''
                    i += 1
                else:
                    if c in '({[':
                        stack.append(c)
                    elif c in ')}]':
                        if not stack:
                            break
                        opening = stack.pop()
                        # Check that the parentheses/brackets/braces match
                        if (opening == '(' and c != ')') or \
                        (opening == '[' and c != ']') or \
                        (opening == '{' and c != '}'):
                            break
                    current_expr += c
                    i += 1
            if current_expr.strip():
                expressions.append(current_expr.strip())

            final_expressions = []
            for expr in expressions:
                expr = expr.strip()
                if not expr:
                    continue
                if (expr.startswith('(') and expr.endswith(')') or \
                expr.startswith('[') and expr.endswith(']') or \
                expr.startswith('{') and expr.endswith('}')) and \
                    "," in expr:
                    # Remove the outer parentheses/brackets/braces
                    inner_expr = expr[1:-1]
                    # Recursively extract expressions from the inner content
                    nested_expressions = extract_expressions(inner_expr)
                    final_expressions.extend(nested_expressions)
                else:
                    # Remove \sum_{...} and \text{...} when checking for "="
                    expr_replaced, placeholder_mapping = replace_structures(expr)
                    if "=" in expr_replaced:
                        # Split on "="
                        sub_exprs = expr_replaced.split("=")
                        for sub_expr in sub_exprs:
                            sub_expr = sub_expr.strip()
                            if sub_expr:
                                # Restore the original structures in sub_expr
                                for placeholder, original in placeholder_mapping.items():
                                    sub_expr = sub_expr.replace(placeholder, original)
                                final_expressions.append(sub_expr)
                    else:
                        final_expressions.append(expr)
                        
            return final_expressions
        
        # Replace \leq, \geq, \neq, >, <, to "="
        query = query.replace(r'\leq', '=').replace(r'\geq', '=').replace(r'\neq', '=').replace('>', '=').replace('<', '=').replace(r'\quad', ' ')
        query = re.sub(r'\\begin{[^{}]*}', '', query)
        query = re.sub(r'\\end{[^{}]*}', '', query)

        # Extract all LaTeX formulas (enclosed in $...$)
        formulas_latex = re.findall(r'\$([^$]*)\$', query)

        latex_expressions = []
        for formula_latex in formulas_latex:
            formula_latex = formula_latex.strip()
            if formula_latex != "":
                if " \\ " not in formula_latex:
                    if "; " not in formula_latex:
                        # Extract all expressions from the formula
                        expressions = extract_expressions(formula_latex)

                        for expr in expressions:
                            expr = expr.strip()
                            if expr == "":
                                continue
                            latex_expressions.append(expr)
                    else:
                        # Split the formula on "; "
                        sub_formulas = formula_latex.split("; ")
                        for sub_formula in sub_formulas:
                            sub_formula = sub_formula.strip()
                            if sub_formula == "":
                                continue
                            # Extract all expressions from the sub-formula
                            expressions = extract_expressions(sub_formula)
                            for expr in expressions:
                                expr = expr.strip()
                                if expr == "":
                                    continue
                                latex_expressions.append(expr)
                else:
                    # Split the formula on " \\ "
                    sub_formulas = formula_latex.split(" \\ ")
                    for sub_formula in sub_formulas:
                        sub_formula = sub_formula.strip()
                        if "; " not in sub_formula:
                            # Extract all expressions from the sub-formula
                            expressions = extract_expressions(sub_formula)
                            for expr in expressions:
                                expr = expr.strip()
                                if expr == "":
                                    continue
                                latex_expressions.append(expr)
                        else:
                            # Split the sub-formula on "; "
                            sub_sub_formulas = sub_formula.split("; ")
                            for sub_sub_formula in sub_sub_formulas:
                                sub_sub_formula = sub_sub_formula.strip()
                                if sub_sub_formula == "":
                                    continue
                                # Extract all expressions from the sub-sub-formula
                                expressions = extract_expressions(sub_sub_formula)
                                for expr in expressions:
                                    expr = expr.strip()
                                    if expr == "":
                                        continue
                                    latex_expressions.append(expr)
        # Remove duplicates while preserving order
        seen = set()
        unique_expressions = []
        for expr in latex_expressions:
            if expr not in seen:
                seen.add(expr)
                unique_expressions.append(expr)
        return unique_expressions    
        
    def change_operators(self, query, requirements, **kwargs):
        """ Perturb the given query by changing operators from the requirements.
            Args:
                query (str): The query to perturb.
                requirements (list): The list of requirements to change the operators.
            Returns:
                str: The perturbed query.
        """
        system_content = self.base_system_content + " ONLY change the operators with minimal text changes as follows:"
        for requirement in requirements:
            original_operation = requirement["original_operation"]
            new_operation = requirement["new_operation"]
            arg_1 = requirement["arg_1"]
            arg_2 = requirement["arg_2"]
            system_content += f"\nchange the operator from {original_operation} to {new_operation} between {arg_1} and {arg_2}"
        system_content += "\nReturn the new sentence only."
        self.update_chat_history([
            ("system", system_content)
        ])
        return self.generate_response(query, **kwargs)

    def change_numbers(self, query, requirements, **kwargs):
        """ Perturb the given query by changing the numbers from the requirements.
            Args:
                query (str): The query to perturb.
                requirements (list): The list of requirements to change the numbers.
            Returns:
                str: The perturbed query
        """
        system_content = self.base_system_content + " ONLY change the values with minimal text changes as follows:"
        for requirement in requirements:
            original_number = requirement["original_number"]
            new_number = requirement["new_number"]
            system_content += f"\nchange the value {original_number} to {new_number}"
        system_content += "\nReturn the new sentence only."
        self.update_chat_history([
            ("system", system_content)
        ])
        return self.generate_response(query, **kwargs)
    
    def swap_operands(self, query, requirements, **kwargs):
        """ Perturb the given query by swapping the operands of the operators with from the requirements.
            Args:
                query (str): The query to perturb.
                requirements (list): The list of requirements to swap the operands.
            Returns:
                str: The perturbed query.
        """
        system_content = self.base_system_content + " ONLY swap the operands with minimal text changes as follows:"
        for requirement in requirements:
            operand_1 = requirement["operand_1"]
            operand_2 = requirement["operand_2"]
            system_content += f"\nswap the operands {operand_1} and {operand_2}"
        system_content += "\nReturn the new sentence only."
        self.update_chat_history([
            ("system", system_content)
        ])
        return self.generate_response(query, **kwargs)
    
    def contradict_query(self, query, **kwargs):
        """ Perturb the given query by contradicting the query.
            Args:
                query (str): The query to perturb.
            Returns:
                str: The perturbed query.
        """
        system_content = self.base_system_content + " change the sentence with minimal text changes to make it contradictory with the original query. Return the new sentence only."
        self.update_chat_history([
            ("system", system_content)
        ])
        return self.generate_response(query, **kwargs)
    
    def actions_to_change_operators(self, query):
        """ Generate the actions to change the operators in the query.
            Args:
                query (str): The query to generate the actions for.
            Returns:
                list: The list of actions to change the operators in the query. Each action is a dictionary containing the original operation type, the new operation type, and the arguments
        """
        def extract_operations(expr):
            """ Extract the operations from the given expression.
                Args:
                    expr (Sympy expression): The Sympy expression to extract the operations from.
                Returns:
                    list: The list of operations in tuples. Each tuple contains the operation type and the arguments. The operation types are "add", "subtract", "multiply", and "divide".
            """
            operations = []
            
            # Recursive function to explore and record the operations in tuples
            def recurse(node):
                if isinstance(node, sp.Mul):  # Check for multiplication
                    # Check if it's actually a division (one arg is a reciprocal)
                    if any(isinstance(arg, sp.Pow) and arg.exp == -1 for arg in node.args) and any(not(isinstance(arg, sp.Pow) and arg.exp == -1) for arg in node.args):
                        # Identify the dividend and the divisor
                        for arg in node.args:
                            if not (isinstance(arg, sp.Pow) and arg.exp == -1):
                                dividend = arg
                            else:
                                divisor = arg.base
                        operations.append(["divide", dividend, divisor])
                        recurse(dividend)
                        recurse(divisor)
                    else:
                        operations.append(["multiply"] + list(node.args))
                        for arg in node.args:
                            recurse(arg)
                        
                elif isinstance(node, sp.Add):  # Check for addition
                    # Check if it's actually a subtraction (one arg is a negative)
                    if any(isinstance(arg, sp.Mul) and any(
                            isinstance(arg_arg, sp.Number) and arg_arg.is_negative for arg_arg in arg.args) for arg in node.args):
                        # Identify the minuend and the subtrahend
                        # Need to consider the case where there are multiple subtrahends
                        minuend = None
                        for arg in node.args:
                            if not (isinstance(arg, sp.Mul) and any(
                                    isinstance(arg_arg, sp.Number) and arg_arg.is_negative for arg_arg in arg.args)):
                                minuend = arg
                            else:
                                subtrahend = -arg
                        if minuend is None:
                            minuend = node.args[0]
                        operations.append(["subtract", minuend, subtrahend])
                        recurse(minuend)
                        recurse(subtrahend)
                    else:
                        operations.append(["add"] + list(node.args))
                        for arg in node.args:
                            recurse(arg)
                        
                elif isinstance(node, sp.Pow):  # Check for power
                    args = node.args
                    operations.append(["pow"] + list(args))
                    for arg in args:
                        recurse(arg)
            
            recurse(expr)
            
            # Remove the operation with "pow"
            return [operation for operation in operations if operation[0] != "pow"]
        
        latex_expressions = self.extract_all_latex_expressions(query)
        if len(latex_expressions) == 0:
            return None
        
        operations = []
        text_to_symbol = {}
        # Iterate through all the LaTeX expressions
        for formula_latex in latex_expressions:
            # Replace the \text{...} strings with symbols
            formula_latex, t2s = self.replace_text_with_symbols(formula_latex)
            text_to_symbol.update(t2s)
            # Replace the \%, !, \left, \right symbol with nothing
            formula_latex = re.sub(r'\\sum(?![_^{])', '', formula_latex.replace(r'\%', '').replace('%', '').replace('!', '').replace('\\left', '').replace('\\right', ''))
            # Replace the ×, ÷, and / symbols with \times, \div
            formula_latex = formula_latex.replace('×', r'\times ').replace('÷', r'\div ').replace('/', r'\div ')
            # Remove any trailing backslashes
            formula_latex = formula_latex.rstrip("\\")
            # Deal with combination numbers
            formula_latex = re.sub(r'\^(\{[^{}]+\}|[^\s{}])C(_\{[^{}]+\}|_\w)', lambda x: f'C{x.group(2)}^{x.group(1)}', formula_latex)
            # Parse the LaTeX expression to a Sympy expression
            try:
                formula = parse_latex(formula_latex)
            except:
                continue
            # Extract the operations from the formula
            operations.extend(extract_operations(formula))
        
        # Convert the symbol to text, and remove the trailing zeros
        if len(text_to_symbol) > 0:
            for operation in operations:
                for i in range(1, len(operation)):
                    if isinstance(operation[i], sp.Number):
                        # If it is a float, remove its trailing zeros
                        operation[i] = self.format_float(operation[i])
                    for text, symbol in text_to_symbol.items():
                        operation[i] = str(operation[i]).replace(symbol[1:], text)
        
        actions = []
        if operations:
            # Randomly select 1-all of the operations to change, keep the order
            num_ops_to_change = random.randint(1, len(operations))
            indices_to_change = random.sample(range(len(operations)), num_ops_to_change)
            indices_to_change.sort()
            for index in indices_to_change:
                operation = operations[index]
                oprt = operation[0]
                # Randomly select a new operation type
                oprts = ["add", "subtract", "multiply", "divide"]
                oprts.remove(oprt)
                new_oprt = random.choice(oprts)
                
                actions.append(
                    {
                        "original_operation": oprt,
                        "new_operation": new_oprt,
                        "arg_1": str(operation[1]),
                        "arg_2": str(operation[2])
                    }
                )
            
        return actions if len(actions) > 0 else None
    
    def actions_to_change_numbers(self, query):
        """ Generate the actions to change the numbers in the query.
            Args:
                query (str): The query to generate the actions for.
            Returns:
                list: The list of actions to change the numbers in the query. Each action is a dictionary containing the original number and the new number.
        """
        def insert_or_delete_a_random_number(value):
            """ Randomly insert or delete a number from the given value.
                Args:
                    value (int or float): The value to insert or delete a number from.
                Returns:
                    int or float: The new value with a number inserted or deleted.
            """
            value_str = str(value)
            minus = False
            if value_str[0] == "-":
                value_str = value_str[1:]
                minus = True
            if random.random() > 0.5 and len(value_str) > 1:
                # Randomly delete a number
                index = random.randint(0, len(value_str) - 1)
                new_value = value_str[:index] + value_str[index+1:]
            else:
                # Randomly insert a number
                index = random.randint(0, len(value_str))
                new_value = value_str[:index] + str(random.randint(0, 9)) + value_str[index:]
            if minus:
                new_value = "-" + new_value
            return int(new_value) if isinstance(value, int) else float(new_value)
        
        numbers = re.compile(
            r'-?[\d,]*\.?\d+',
            re.MULTILINE | re.DOTALL | re.IGNORECASE,
        ).findall(query)
        
        if numbers:
            last_number = numbers[-1]
            # Remove duplicate numbers but keep the last_number in the last position
            numbers = list(set(numbers))
            numbers.remove(last_number)
            numbers.append(last_number)
            
            # Select 1-all of the numbers to change, ensure the last one is selected
            num_to_change = random.randint(0, len(numbers)-1)
            numbers_to_change = random.sample(numbers[:-1], num_to_change) + [numbers[-1]]
            
            actions = []
            for number in numbers_to_change:
                has_comma = "," in number
                num_to_change = number.replace(",", "")
                num_to_change = num_to_change.strip()

                # detect if the number is a float or an integer
                if "." in num_to_change:
                    num_digit = len(num_to_change.split(".")[1])
                    value = float(num_to_change)
                    new_value = insert_or_delete_a_random_number(value) if random.random() > 0.8 else round(value + random.uniform(-10, 10) / 100 * value, num_digit)
                else:
                    value = int(num_to_change)
                    delta = lambda f: max(int(f), 1) if f >= 0 else min(int(f), -1)
                    new_value = insert_or_delete_a_random_number(value) if random.random() > 0.8 else value + delta(random.uniform(-10, 10) / 100 * value)

                if has_comma:
                    new_pattern = str("{:,}".format(new_value))
                else:
                    new_pattern = str(new_value)
                    
                actions.append(
                    {
                        "original_number": number,
                        "new_number": new_pattern
                    }
                )
            return actions         
        else:
            return None
        
    def actions_to_swap_operands(self, query):
        """ Generate the actions to swap the operands in the query.
            Args:
                query (str): The query to generate the actions for.
            Returns:
                list: The list of actions to swap the operands in the query. Each action is a dictionary containing the two operands to swap.
        """
        latex_expressions = self.extract_all_latex_expressions(query)
        if len(latex_expressions) == 0:
            return None
        
        operands = []
        text_to_symbol = {}
        # Iterate through all the LaTeX expressions
        for formula_latex in latex_expressions:
            # Replace the \text{...} strings with symbols
            formula_latex, t2s = self.replace_text_with_symbols(formula_latex)
            text_to_symbol.update(t2s)
            # Replace the \%, !, \left, \right symbol with nothing
            formula_latex = re.sub(r'\\sum(?![_^{])', '', formula_latex.replace(r'\%', '').replace('%', '').replace('!', '').replace('\\left', '').replace('\\right', ''))
            # Replace the ×, ÷, and / symbols with \times, \div
            formula_latex = formula_latex.replace('×', r'\times').replace('÷', r'\div').replace('/', r'\div')
            # Remove any trailing backslashes
            formula_latex = formula_latex.rstrip("\\")
            # Deal with combination numbers
            formula_latex = re.sub(r'\^(\{[^{}]+\}|[^\s{}])C(_\{[^{}]+\}|_\w)', lambda x: f'C{x.group(2)}^{x.group(1)}', formula_latex)
            # Parse the LaTeX expression to a Sympy expression
            try:
                formula = parse_latex(formula_latex)
            except:
                continue
            # Extract all the operands from the formula
            operands.extend(list(formula.free_symbols))
        operands = list(set(operands))
            
        # Convert the symbol to text
        if len(text_to_symbol) > 0:
            for i in range(len(operands)):
                for text, symbol in text_to_symbol.items():
                    operands[i] = str(operands[i]).replace(symbol[1:], text)
                    
        # Randomly select 1-all pairs of the operands to swap
        actions = []
        if len(operands)>=2:
            # num_elems_to_swap should be even
            num_elems_to_swap = random.randint(1, max(len(operands) // 2, 1)) * 2
            elems_to_swap = random.sample(operands, num_elems_to_swap)
            for i in range(0, len(elems_to_swap), 2):
                actions.append(
                    {
                        "operand_1": elems_to_swap[i],
                        "operand_2": elems_to_swap[i+1]
                    }
                )
            return actions
        else:
            return None
        
    def test_perturbation(self, query, **kwargs):
        """ Generate the perturbations for the given query.
            Args:
                query (str): The query to generate the perturbations for.
                temperature (float): The temperature for sampling from the model.
                top_p (float): The nucleus sampling ratio.
                model_name (str): The name of the model to use for generating the response.
            Returns:
                str: The perturbed query.
        """
        
        # 1. Generate the actions to change the operators
        change_ops = self.actions_to_change_operators(query)

        # 2. Generate the actions to change the numbers
        change_nums = self.actions_to_change_numbers(query)

        # 3. Generate the actions to swap the operands
        swap_ops = self.actions_to_swap_operands(query)

        return change_ops, change_nums, swap_ops
    
    def generate_perturbation(self, query, **kwargs):
        """ Generate the perturbations for the given query.
            Args:
                query (str): The query to generate the perturbations for.
                temperature (float): The temperature for sampling from the model.
                top_p (float): The nucleus sampling ratio.
                model_name (str): The name of the model to use for generating the response.
            Returns:
                str: The perturbed query.
        """
        original_query = query
        
        # 1. Generate the actions to change the operators
        change_ops = self.actions_to_change_operators(query)
        if change_ops is not None:
            for action_todo in change_ops:
                discard = False
                for action_hist in self.change_ops:
                    if action_todo["original_operation"] == action_hist["original_operation"] and action_todo["arg_1"] == action_hist["arg_1"] and action_todo["arg_2"] == action_hist["arg_2"]:
                        discard = True
                        break
                if not discard:
                    self.change_ops.append(action_todo)
            query = self.change_operators(query, self.change_ops, **kwargs)[0]
        
        # 2. Generate the actions to change the numbers
        change_nums = self.actions_to_change_numbers(query)
        if change_nums is not None:
            for action_todo in change_nums:
                discard = False
                for action_hist in self.change_nums:
                    if action_todo["original_number"] == action_hist["original_number"]:
                        discard = True
                        break
                if not discard:
                    self.change_nums.append(action_todo)
            query = self.change_numbers(query, self.change_nums, **kwargs)[0]
            
        # 3. Generate the actions to swap the operands
        swap_ops = self.actions_to_swap_operands(query)
        if swap_ops is not None:
            for action_todo in swap_ops:
                discard = False
                for action_hist in self.swap_ops:
                    if action_todo["operand_1"] == action_hist["operand_1"] and action_todo["operand_2"] == action_hist["operand_2"]:
                        discard = True
                        break
                if not discard:
                    self.swap_ops.append(action_todo)
            query = self.swap_operands(query, self.swap_ops, **kwargs)[0]
            
        # 4. If the first three steps don't work, generate the contradictions
        if query == original_query:
            query = self.contradict_query(query, **kwargs)[0]
            
        return query