import ollama
import re
import multiprocessing
import traceback

# ---------------------
# Ollama API helpers
# ---------------------

def api_call(input_text, model, client=None):
    """
    Call Ollama with the given input_text.
    """
    prompt = input_text
    print(prompt)
    if client:
        response = client.chat(
            model=model, 
            messages=[{"role": "user", "content": prompt}]
        )
    else:
        # fallback: use global ollama if client not provided
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
    return response["message"]["content"]


def _call_target(func, q, args, kwargs):
    """Target wrapper for running a function in a separate process.

    Puts a tuple (True, result) on the queue on success, or (False, exc_str) on failure.
    """
    try:
        res = func(*args, **(kwargs or {}))
        q.put((True, res))
    except Exception:
        q.put((False, traceback.format_exc()))


def call_with_timeout(func, args=(), kwargs=None, timeout: int = 30):
    """Call `func(*args, **kwargs)` but kill it if it doesn't complete within `timeout` seconds.

    Uses multiprocessing to ensure the child can be terminated if it blocks in C code.
    Returns the function's return value on success. Raises TimeoutError on timeout and
    re-raises exceptions from the child process as RuntimeError with the child's traceback.
    """
    if kwargs is None:
        kwargs = {}

    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=_call_target, args=(func, q, args, kwargs))
    p.start()
    p.join(timeout)
    if p.is_alive():
        try:
            p.terminate()
        except Exception:
            pass
        p.join()
        raise TimeoutError("Function call timed out")

    if q.empty():
        raise RuntimeError("Child process exited without returning a result")

    ok, payload = q.get()
    if ok:
        return payload
    else:
        # payload is a traceback string from child
        raise RuntimeError(f"Child process raised an exception:\n{payload}")


import re

def get_python(message):
    start_code_block = message.find("```Python")
    if start_code_block == -1:
        start_code_block = message.find("```python")
    if start_code_block == -1:
        start_code_block = message.find("```")
        cut_string = message[start_code_block + 3:]
    else:
        cut_string = message[start_code_block + 10:]
    end_code_block = cut_string.find("```")
    return cut_string[:end_code_block].strip()


def get_last_non_empty_line(text: str) -> str:
    clean_text = re.sub(r'\x1b\[[0-9;]*m', '', text)
    lines = clean_text.splitlines()
    for line in reversed(lines):
        if line.strip():
            return line.strip().split('(')[0].strip()
    return ""


def get_test_cases_info(feedback: str):
    matches = re.findall(r"Test cases:\s*\[.*?\]\s*(\d+)\s*/\s*(\d+)", feedback)
    if matches:
        last_match = matches[-1]
        passed = int(last_match[0])
        total = int(last_match[1])
        return passed, total
    return 0, 0


def normalize_failed_cases(failures):
    return [
      {
        "case": getattr(c, "case_number", None),
        "input": getattr(c, "input_str", None),
        "expected": getattr(c, "expected_str", None),
        "got": getattr(c, "actual_str", "<no output>"),
        "error": getattr(c, "error_msg", "")
      }
      for c in failures
    ]


def normalize_function_name(code: str, expected_name: str) -> str:
    match = re.search(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)", code)
    if not match:
        return code
    found_name = match.group(1)
    if found_name != expected_name:
        code = re.sub(rf"\bdef\s+{found_name}\b", f"def {expected_name}", code)
        code = re.sub(rf"\b{found_name}\s*\(", f"{expected_name}(", code)
    return code

import textwrap
import re

def preclean_code(code_str: str) -> str:
    """
    Normalize messy LLM Python output:
    - Strip trailing semicolons
    - Dedent uniformly
    - Collapse excessive spaces before code
    - Remove stray leading/trailing whitespace
    """
    # Remove any trailing semicolons
    code_str = re.sub(r";\s*$", "", code_str, flags=re.MULTILINE)

    # Replace tabs with 4 spaces just in case
    code_str = code_str.replace("\t", "    ")

    # Dedent (handles weird alignment like 59 spaces)
    code_str = textwrap.dedent(code_str)

    # Strip extra leading/trailing whitespace
    code_str = code_str.strip()

    return code_str
