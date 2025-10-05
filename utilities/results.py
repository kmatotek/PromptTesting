from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

class Status(Enum):
    PASS = auto()
    FAIL = auto()
    TIMEOUT = auto()
    RUNTIME_ERROR = auto()

@dataclass
class TestCaseResult:
    case_number: int
    status: Status
    input_str: str
    expected_str: str
    actual_str: Optional[str] = None
    error_msg: Optional[str] = None

@dataclass
class ProblemResult:
    problem_id: str
    cases: list[TestCaseResult]

    @property
    def passed_count(self) -> int:
        return sum(1 for c in self.cases if c.status == Status.PASS)

    @property
    def total_count(self) -> int:
        return len(self.cases)

    
    def print_cases(self) -> None:
        # Header block
        sep = "=" * 60
        print(sep)
        print(f"Problem: {self.problem_id}")
        print(f"Passed: {self.passed_count}/{self.total_count}")
        print(sep)

        # Per-case details
        for c in self.cases:
            header = f"Case #{c.case_number}: {c.status.name.replace('_', ' ').title()}"
            print(header)

            # If not passed, show input, expected, and actual/error
            if c.status != Status.PASS:
                print("  Input:")
                for line in c.input_str.splitlines():
                    print(f"    {line}")

                print("  Expected:")
                for line in c.expected_str.splitlines():
                    print(f"    {line}")

                if c.status == Status.FAIL:
                    print("  Got:")
                    for line in c.actual_str.splitlines():
                        print(f"    {line}")
                elif c.status == Status.RUNTIME_ERROR:
                    print("  Runtime Error:")
                    for line in (c.error_msg or "").splitlines():
                        print(f"    {line}")
                elif c.status == Status.TIMEOUT:
                    print("  Timeout occurred")

            # Separator between cases
            print("-" * 60)
