"""Code execution scorer - extracts and runs code, checks output."""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

from ai_evals.config import EvalTask
from ai_evals.scorers.base import ScoreResult


class CodeExecutionScorer:
    def __init__(
        self,
        expected: str | list[str] | None = None,
        timeout_seconds: int = 30,
    ) -> None:
        self.expected = expected
        self.timeout_seconds = timeout_seconds

    def score(self, response: str, task: EvalTask) -> ScoreResult:
        code = _extract_code(response)
        if not code:
            return ScoreResult(
                score=0.0,
                passed=False,
                details={"error": "No code block found in response"},
            )

        try:
            result = _run_code(code, timeout=self.timeout_seconds)
        except subprocess.TimeoutExpired:
            return ScoreResult(
                score=0.0,
                passed=False,
                details={"error": f"Code execution timed out after {self.timeout_seconds}s"},
            )
        except Exception as e:
            return ScoreResult(
                score=0.0,
                passed=False,
                details={"error": f"Code execution failed: {e}"},
            )

        if result.returncode != 0:
            return ScoreResult(
                score=0.0,
                passed=False,
                details={
                    "error": "Code exited with non-zero status",
                    "stderr": result.stderr[:500],
                    "returncode": result.returncode,
                },
            )

        output = result.stdout.strip()

        if self.expected is None:
            # No expected output — just check it ran successfully
            return ScoreResult(
                score=1.0,
                passed=True,
                details={"output": output[:500]},
            )

        expected_list = [self.expected] if isinstance(self.expected, str) else self.expected
        matched = any(exp.strip() in output for exp in expected_list)

        return ScoreResult(
            score=1.0 if matched else 0.0,
            passed=matched,
            details={
                "output": output[:500],
                "expected": expected_list,
                "matched": matched,
            },
        )


def _extract_code(response: str) -> str | None:
    """Extract the first Python code block from a response."""
    # Try fenced code block with python tag
    match = re.search(r"```python\s*\n(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try generic fenced code block
    match = re.search(r"```\s*\n(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()

    return None


def _run_code(code: str, timeout: int = 30) -> subprocess.CompletedProcess:
    """Run Python code in a subprocess."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", dir=Path(tempfile.gettempdir()), delete=True
    ) as f:
        f.write(code)
        f.flush()
        return subprocess.run(
            ["python3", f.name],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
