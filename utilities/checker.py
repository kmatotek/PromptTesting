import json, subprocess, time
from typing import List
from utilities.results import TestCaseResult, ProblemResult, Status

def strip_whitespace(s: str) -> str:
    # Strip whitespace from each line (Kattis doesn't care about leading & trailing whitesapce)
    return "\n".join(line.strip() for line in s.strip().splitlines())


def run_solution_on_input(solution_path, test_case_str, timeout = 5.0):
    # Run the model's solution on a single test case
   
    proc = subprocess.run(
        ["python3", solution_path],  
        input=test_case_str.encode(),     
        stdout=subprocess.PIPE,      
        stderr=subprocess.PIPE,       
        timeout=timeout               
    )
   
    return proc

def test_problem(problem_id, solution_path, parsed_tests_path = "./datasets/parsed_tests.json") -> ProblemResult:
    """
    Load test cases for a problem, run the solution on each, and collect results.
    """

    # Load all parsed test cases from JSON
    data = json.load(open(parsed_tests_path))
    cases_json = data[problem_id]

    results: List[TestCaseResult] = []

    # Iterate through each test case
    for idx, case in enumerate(cases_json, start=1):
        inp, exp = case["input"], case["output"]

        try:
            proc = run_solution_on_input(solution_path, inp)
        except subprocess.TimeoutExpired:
            # Solution took too long; timeout
            results.append(TestCaseResult(
                case_number=idx,
                status=Status.TIMEOUT,
                input_str=inp,
                expected_str=exp,
                error_msg="Timed out"
            ))
            continue

        out = proc.stdout.decode().rstrip()

        if proc.returncode != 0:
            # Runtime error 
            results.append(TestCaseResult(
                case_number=idx,
                status=Status.RUNTIME_ERROR,
                input_str=inp,
                expected_str=exp,
                error_msg=proc.stderr.decode().strip()
            ))
        elif strip_whitespace(out) == strip_whitespace(exp):
            # Test passed
            results.append(TestCaseResult(
                case_number=idx,
                status=Status.PASS,
                input_str=inp,
                expected_str=exp,
                actual_str=out
            ))
        else:
            # Test failed; actual differs from expected
            results.append(TestCaseResult(
                case_number=idx,
                status=Status.FAIL,
                input_str=inp,
                expected_str=exp,
                actual_str=out
            ))

    # Return a summary object containing all case results
    return ProblemResult(problem_id=problem_id, cases=results)
