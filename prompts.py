import json
import os
from datetime import datetime
from itertools import product
from pathlib import Path
from utilities.utilities import api_call, call_with_timeout, get_python, TimeoutError
from utilities.checker import test_problem
from utilities.results import ProblemResult

# =====================
# Initialize result file
# =====================
def init_results(reset=False):
    file_path = "./results/results_kattis.json"
    if reset or not os.path.exists(file_path):
        with open(file_path, "w") as out:
            json.dump([], out)
    return file_path


# =====================
# Base instructions
# =====================
base_instructions = (
    "You are a Python programming expert who writes clean, efficient code for competitive-programming style problems.\n"
    "When given a problem statement and test cases, produce a single Python script that:\n"
    "1. Uses only the Python standard library (no external imports).\n"
    "2. Reads input silently from stdin using input() without any prompts or additional text.\n"
    "3. Chooses descriptive, non-conflicting variable and function names.\n"
    "4. Correctly handles edge cases (empty inputs, minimum/maximum values, etc.).\n"
    "5. Does not hard-code any test-specific values (your solution must generalize).\n"
    "6. Make sure to print the result and nothing else besides the result!\n"
)

# =====================
# Prompt variant components
# =====================

problem_framing = {
    "Natural language": "{description}",
    #"Test-driven": "Write Python code that passes these tests:\n{tests}"
}


reasoning_scaffolds = {
    "Direct": "{problem}",
    "Chain-of-Thought": "Reason step by step, then provide code:\n{problem}",
    "Program-Aided": "First generate pseudo-code or comments as intermediate reasoning steps, then the full code:\n{problem}"
}

decomposition = {
    "None": "{problem}",
    "Basic": (
        "Break down the problem into simpler sub-tasks:\n"
        "1. Understand inputs and outputs.\n"
        "2. Handle edge cases.\n"
        "3. Implement core logic.\n"
        "Then solve each sub-task in code:\n{problem}"
    )
}

output_control = {
    "Code only": (
        "Return only code inside fenced Python code blocks like this:\n```python\n# code here\n```"
    ),
    "Explanation + Code": (
        "Provide an explanation for your solution, then provide code inside a fenced Python code block."
    )
}


# =====================
# Prompt generator
# =====================
def load_kattis_problems(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def generate_prompt_variants(problem_id: str, problem_data: dict):
    desc = problem_data.get("description", "")
    tests = problem_data.get("tests", "")

    variants = []
    for framing_key, reasoning_key, decomp_key, output_key in product(
        problem_framing.keys(),
        reasoning_scaffolds.keys(),
        decomposition.keys(),
        output_control.keys()
    ):
        framing = problem_framing[framing_key].format(description=desc, tests=tests)
        decomp_problem = decomposition[decomp_key].format(problem=framing)
        reason_problem = reasoning_scaffolds[reasoning_key].format(problem=decomp_problem)
        full_prompt = base_instructions + "\nBelow is the full problem. Write only as instructed.\n\n" + reason_problem + "\n" + output_control[output_key]

        variants.append({
            "framing": framing_key,
            "reasoning": reasoning_key,
            "decomposition": decomp_key,
            "output_control": output_key,
            "prompt": full_prompt
        })

    return variants


# =====================
# Experiment runner
# =====================
def run_experiment(model="qwen2.5-coder:0.5b", reset=False):
    results_file = init_results(reset=reset)

    kattis_path = Path("./datasets/kattis_problems.json")
    if not kattis_path.exists():
        raise FileNotFoundError("kattis_problems.json not found in workspace")

    kattis = load_kattis_problems(str(kattis_path))

    for problem_id, pdata in kattis.items():
        if problem_id not in ["twostones"]:  # Example filter
            continue

        print(f"\n=== Running {problem_id} ===")

        prompt_variants = generate_prompt_variants(problem_id, pdata)

        for variant in prompt_variants:
            variant_name = f"{variant['framing']}-{variant['reasoning']}-{variant['decomposition']}-{variant['output_control']}"
            print(f"\n--- Variant: {variant_name} ---")

            result = {
                "timestamp": datetime.now().isoformat(),
                "problem": problem_id,
                "prompt_variant": {
                    "framing": variant["framing"],
                    "reasoning": variant["reasoning"],
                    "decomposition": variant["decomposition"],
                    "output_control": variant["output_control"]
                },
                "prompt": variant["prompt"],
                "passed": False,
                "error": "",
                "raw_code": "",
                "test_summary": {},
            }

            try:
                response = call_with_timeout(api_call, args=(variant["prompt"], model), timeout=60)
                raw_code = get_python(response)
                result["raw_code"] = raw_code

                tmp_path = "./temp_solution.py"
                with open(tmp_path, "w") as f:
                    f.write(raw_code)

                prob_result: ProblemResult = test_problem(problem_id=problem_id, solution_path=tmp_path)

                passed = prob_result.passed_count == prob_result.total_count
                result["passed"] = passed
                result["test_summary"] = {
                    "passed": prob_result.passed_count,
                    "total": prob_result.total_count
                }

                prob_result.print_cases()
                print("✅ All tests passed" if passed else "❌ Some tests failed")

            except TimeoutError:
                result["error"] = "API call or execution timed out"
                print("⏱️ Timeout occurred")
            except Exception as e:
                result["error"] = str(e)
                print(f"❌ Error: {e}")

            with open(results_file, "r+") as f:
                data = json.load(f)
                data.append(result)
                f.seek(0)
                json.dump(data, f, indent=2)


if __name__ == "__main__":
    run_experiment(reset=False)
