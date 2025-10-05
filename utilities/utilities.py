import ollama
import re
import multiprocessing as mp
import traceback
from typing import Any, Callable

# ---------------------
# Ollama API helpers
# ---------------------

def api_call(input_text, model, client=None):
    """
    Call Ollama with the given input_text.
    """
    prompt = input_text
    #print(prompt)
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


class TimeoutError(Exception):
    """Raised when a call times out."""
    pass


def _worker(func: Callable, args: tuple, kwargs: dict, out_q: mp.Queue):
    """Worker executed in a child process to run func and return its result or exception."""
    try:
        res = func(*args, **(kwargs or {}))
        out_q.put((True, res))
    except Exception as e:
        # Send traceback string to parent for better debugging
        tb = traceback.format_exc()
        out_q.put((False, (e, tb)))


def call_with_timeout(func: Callable, args: tuple = (), kwargs: dict = None, timeout: float = 30) -> Any:
    """Run func(*args, **kwargs) in a separate process and return its result.

    If the function doesn't return within `timeout` seconds, terminate the process and raise TimeoutError.
    Any exception raised inside the function is re-raised in the parent with original traceback attached.
    """
    ctx = mp.get_context("fork") if hasattr(mp, "get_context") else mp
    q: mp.Queue = ctx.Queue()
    p = ctx.Process(target=_worker, args=(func, args, kwargs or {}, q))
    p.start()
    try:
        success, payload = q.get(timeout=timeout)
    except Exception:
        # Timeout or queue empty; ensure process is terminated
        if p.is_alive():
            p.terminate()
            p.join(timeout=1)
        raise TimeoutError(f"Function call timed out after {timeout} seconds")
    finally:
        if p.is_alive():
            p.join(timeout=1)

    if success:
        return payload
    else:
        exc, tb = payload
        # Raise the original exception but attach remote traceback for debugging
        raise Exception(f"Child process exception: {exc}\nRemote traceback:\n{tb}")


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
