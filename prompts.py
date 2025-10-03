import csv
import json
import os
from datetime import datetime
from itertools import product
import ast
from ast import unparse 
from utilities import api_call, get_python
from typing import List, Optional
import signal

# =====================
# Timeout helper
# =====================
class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Execution timed out")

# Register the handler once
signal.signal(signal.SIGALRM, timeout_handler)

# =====================
# Results initialization
# =====================
def init_results(reset=False):
    files = {
        "easy": "./results/results_easy_test.json",
        "medium": "./results/results_medium_test.json",
        "hard": "./results/results_hard_test.json"
    }
    if reset:
        for f in files.values():
            with open(f, "w") as out:
                json.dump([], out)
    else:
        for f in files.values():
            if not os.path.exists(f):
                with open(f, "w") as out:
                    json.dump([], out)
    return files


# =====================
# 1. Prompt dimensions
# =====================
problem_framing = {
    "Natural language": "Write a Python function named {func_name} that {description}",
    "Docstring + signature": "\"\"\"{description}\"\"\"\ndef {func_name}({signature}):",
    "Test-driven": "Write Python code that passes these tests:\n{tests}"
}

reasoning_scaffolds = {
    "Direct": "{problem}",
    "Step-by-step": "Reason step by step, then provide code:\n{problem}",
    "Self-checking": "Explain your approach, then provide final code:\n{problem}"
}

verbosity = {
    "Minimal": "{problem}",
    "Medium": "{problem}\nFunction signature: def {func_name}({signature})",
    "Verbose": "Problem:\n{problem}\nConstraints:\n{constraints}\nExamples:\n{examples}\nFunction name: {func_name}"
}

output_control = {
    "Code only": (
        "Return only code, no explanation. "
        "Put all code inside fenced code blocks like this:\n```python\n# code here\n```"
    ),
    "Code + explanation": (
        "Provide code inside a fenced Python code block, then explain briefly outside the block. "
        "Example:\n```python\n# code\n```\nExplanation here."
    ),
    "Code + tests + explanation": (
        "Provide code and tests inside fenced Python code blocks, then explain briefly outside the block. "
        "Always format code like:\n```python\n# code\n```"
    )
}


# =====================
# 2. Problems
# =====================
problems = [
    {
        "name": "Easy – Two Sum",
        "func_name": "two_sum",
        "description": "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.",
        "signature": "nums: list[int], target: int -> list[int]",
        "constraints": "Each input has exactly one solution. Do not use the same element twice.",
        "examples": "Input: nums = [2,7,11,15], target = 9 → Output: [0,1]",
        "tests": "assert two_sum([2,7,11,15], 9) == [0,1]"
    },
    {
        "name": "Medium – Product of Array Except Self",
        "func_name": "product_except_self",
        "description": "Given an array nums of length n, return an array output such that output[i] is equal to the product of all elements of nums except nums[i].",
        "signature": "nums: list[int] -> list[int]",
        "constraints": "Do not use division. Solve in O(n) time complexity and O(1) extra space (output array does not count).",
        "examples": "Input: [1,2,3,4] → Output: [24,12,8,6]\nInput: [-1,1,0,-3,3] → Output: [0,0,9,0,0]",
        "tests": """
assert product_except_self([1,2,3,4]) == [24,12,8,6]
assert product_except_self([-1,1,0,-3,3]) == [0,0,9,0,0]
"""
    },
    {
        "name": "Hard – Median of Two Sorted Arrays",
        "func_name": "find_median_sorted_arrays",
        "description": "Given two sorted arrays nums1 and nums2 of size m and n respectively, return the median of the two sorted arrays.",
        "signature": "nums1: list[int], nums2: list[int] -> float",
        "constraints": "The overall run time complexity should be O(log (m+n)).",
        "examples": "Input: nums1 = [1,3], nums2 = [2] → Output: 2.0",
        "tests": "assert find_median_sorted_arrays([1,3], [2]) == 2.0"
    }
]

problems2 = [
    {
        "name": "Easy – Find Words Containing Character",
        "func_name": "find_words_containing",
        "description": "Given a 0-indexed array of strings words and a character x, return an array of indices representing the words that contain x.",
        "signature": "words: list[str], x: str -> list[int]",
        "constraints": "1 <= words.length <= 50; 1 <= words[i].length <= 50; x is a lowercase English letter; words[i] consists only of lowercase English letters.",
        "examples": 'Input: words = ["leet","code"], x = "e" → Output: [0,1]',
        "tests": """
assert find_words_containing(["leet","code"], "e") == [0,1]
assert find_words_containing(["abc","bcd","aaaa","cbc"], "a") == [0,2]
assert find_words_containing(["abc","bcd","aaaa","cbc"], "z") == []
"""
    },
    {
        "name": "Medium – Coin Change",
        "func_name": "coin_change",
        "description": "You are given an integer array coins representing coins of different denominations and an integer amount representing a total amount of money. Return the fewest number of coins that you need to make up that amount. If that amount of money cannot be made up by any combination of the coins, return -1.",
        "signature": "coins: list[int], amount: int -> int",
        "constraints": "1 <= coins.length <= 12; 1 <= coins[i] <= 2^31 - 1; 0 <= amount <= 10^4",
        "examples": "Input: coins = [1,2,5], amount = 11 → Output: 3\nInput: coins = [2], amount = 3 → Output: -1\nInput: coins = [1], amount = 0 → Output: 0",
        "tests": """
assert coin_change([1,2,5], 11) == 3
assert coin_change([2], 3) == -1
assert coin_change([1], 0) == 0
"""
    },
    {
    "name": "Hard – Trapping Rain Water",
    "func_name": "trap",
    "description": "Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it can trap after raining.",
    "signature": "height: list[int] -> int",
    "constraints": "1 <= len(height) <= 2 * 10^4, 0 <= height[i] <= 10^5",
    "examples": "Input: height = [0,1,0,2,1,0,1,3,2,1,2,1] → Output: 6; Input: height = [4,2,0,3,2,5] → Output: 9",
    "tests": """
assert trap([4,2,0,3,2,5]) == 9
assert trap([1,0,1]) == 1
assert trap([2,0,2]) == 2
"""
}

]

# =====================
# 3. Code cleaner
# =====================
def clean_and_normalize_code(code_str, expected_name):
    """Clean code: remove non-code text, handle class Solution, rename funcs."""
    prelude = "from typing import List, Optional\n"
    code_str = prelude + code_str

    # Defensive: strip out lines that clearly aren't code
    filtered_lines = []
    for line in code_str.splitlines():
        if line.strip().startswith(("def ", "class ", "import ", "from ", "@")):
            filtered_lines.append(line)
        elif line.strip().startswith(("return", "if", "for", "while", "try", "except", "with", "#", "")):
            filtered_lines.append(line)
        # else drop natural language junk like "Explanation:", "Reason step by step", etc.
    code_str = "\n".join(filtered_lines)

    try:
        signal.alarm(3)  # timeout parse
        tree = ast.parse(code_str)
    except SyntaxError:
        # Fallback: regex capture just function bodies
        import re
        matches = re.findall(r"(def\s+" + re.escape(expected_name) + r"\s*\(.*?\):[\s\S]+?)(?=\ndef|\Z)", code_str)
        if matches:
            return prelude + matches[0]
        raise
    finally:
        signal.alarm(0)

    new_body = []
    has_solution_class = False

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef, ast.Assign)):
            if isinstance(node, ast.ClassDef) and node.name == "Solution":
                has_solution_class = True
                for subnode in node.body:
                    if isinstance(subnode, ast.FunctionDef):
                        subnode.name = expected_name
                        new_body.append(subnode)
            elif isinstance(node, ast.FunctionDef):
                if node.name == expected_name:
                    new_body.append(node)

    tree.body = new_body
    cleaned_code = unparse(tree)

    if has_solution_class:
        wrapper = (
            f"\ndef {expected_name}(*args, **kwargs):\n"
            f"    return Solution().{expected_name}(*args, **kwargs)\n"
        )
        cleaned_code += wrapper

    return cleaned_code


# =====================
# 4. Prompt generator
# =====================
def generate_prompts(problem):
    all_variants = []
    for (pf_name, pf_template), (rs_name, rs_template), (vb_name, vb_template), (oc_name, oc_template) in product(
        problem_framing.items(),
        reasoning_scaffolds.items(),
        verbosity.items(),
        output_control.items()
    ):
        base_problem = pf_template.format(
            description=problem["description"],
            signature=problem["signature"],
            tests=problem["tests"],
            func_name=problem["func_name"]
        )
        scaffolded = rs_template.format(problem=base_problem)
        verbose = vb_template.format(
            problem=scaffolded,
            signature=problem["signature"],
            constraints=problem["constraints"],
            examples=problem["examples"],
            func_name=problem["func_name"]
        )
        final_prompt = f"{verbose}\n\n{oc_template}"

        all_variants.append({
            "framing": pf_name,
            "scaffold": rs_name,
            "verbosity": vb_name,
            "output": oc_name,
            "prompt": final_prompt
        })
    return all_variants

# =====================
# 5. Run experiment
# =====================
def run_experiment(difficulties=None, reset=False):
    model = "qwen2.5-coder:0.5b"
    #model = "starcoder:1b"
    files = init_results(reset=reset)

    for problem in problems:
        difficulty = problem["name"].split("–")[0].strip().lower()

        if difficulties and difficulty not in difficulties:
            continue

        print(f"\n=== Running {problem['name']} ===")
        variants = generate_prompts(problem)

        for i, variant in enumerate(variants):
            print(f"\n--- Variant {i+1}/{len(variants)} ---")
            print(f"Framing: {variant['framing']}, Scaffold: {variant['scaffold']}, "
                  f"Verbosity: {variant['verbosity']}, Output: {variant['output']}")

            result = {
                "timestamp": datetime.now().isoformat(),
                "problem": problem["name"],
                "framing": variant["framing"],
                "scaffold": variant["scaffold"],
                "verbosity": variant["verbosity"],
                "output": variant["output"],
                "passed": False,
                "error": "",
                "raw_code": "",
                "code": "",
                "test_results": []
            }

            try:
                # Protect external API calls with a process-level timeout so the whole
                # experiment doesn't hang if the model/client blocks.
                from utilities import call_with_timeout
                try:
                    response = call_with_timeout(api_call, args=(variant["prompt"], model), timeout=30)
                except TimeoutError:
                    raise TimeoutError("API call timed out")
                raw_code = get_python(response)
                result["raw_code"] = raw_code

                expected_name = problem["func_name"]
                try:
                    code = clean_and_normalize_code(raw_code, expected_name)
                    result["code"] = code
                except Exception as e:
                    result["error"] = f"Cleaning error: {e}"
                    print(f"Cleaning error: {e}")
                    continue  # skip bad variant

                local_ns = {
                    "List": List,
                    "Optional": Optional
                }

                try:
                    signal.alarm(5)  # timeout for exec + tests
                    exec(code, local_ns, local_ns)

                    tree = ast.parse(problem["tests"])
                    all_passed = True
                    for node in tree.body:
                        if isinstance(node, ast.Assert):
                            actual_val = eval(compile(ast.Expression(node.test.left), "<ast>", "eval"), local_ns)
                            expected_val = eval(compile(ast.Expression(node.test.comparators[0]), "<ast>", "eval"), local_ns)
                            passed = (actual_val == expected_val)
                            if not passed:
                                all_passed = False
                            result["test_results"].append({
                                "test": ast.unparse(node.test),
                                "actual": actual_val,
                                "expected": expected_val,
                                "passed": passed
                            })

                    result["passed"] = all_passed
                    print("✅ Passed all tests" if all_passed else "❌ Some tests failed")
                except TimeoutError:
                    result["error"] = "Execution timed out"
                    print("⏱️ Execution timed out")
                except Exception as e:
                    result["error"] = str(e)
                    print("❌ Failed:", e)
                finally:
                    signal.alarm(0)

            except Exception as e:
                print(f"Error during API call: {e}")
                result["error"] = f"API error: {e}"

            json_file = files[difficulty]
            with open(json_file, "r+") as f:
                data = json.load(f)
                data.append(result)
                f.seek(0)
                json.dump(data, f, indent=2)


if __name__ == "__main__":
    run_experiment(difficulties=["easy"], reset=False)
