import unittest
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

import torch
from torch.export import export, ExportedProgram

# Suppose you have some helper code for:
#  1) Checking IR or strings, similarly to "FileCheck"
#  2) Creating & verifying an SMT encoding
#  3) Possibly running & comparing outputs
# We illustrate them here as placeholders:


class FileCheck:
    """Minimal placeholder for file-check-like usage."""

    def __init__(self):
        self.patterns = []

    def check(self, s: str):
        self.patterns.append(("check", s))
        return self

    def check_not(self, s: str):
        self.patterns.append(("check_not", s))
        return self

    def check_count(self, s: str, count: int, exactly: bool = True):
        self.patterns.append(("check_count", s, count, exactly))
        return self

    def run(self, text: str):
        pass


class SmtTester:

    def __init__(
        self,
        module: torch.nn.Module,
        example_inputs: Tuple[torch.Tensor, ...],
        dynamic_shapes: Optional[Any] = None,
    ):
        module.eval()
        self.original_module = module
        self.example_inputs = example_inputs
        self.dynamic_shapes = dynamic_shapes

        self.artifacts: Dict[str, Any] = OrderedDict()
        self.stage_flow = OrderedDict()

        self.reference_outputs = None

    def export(self):
        ep: ExportedProgram = export(self.original_module, self.example_inputs)
        self.artifacts["export"] = ep

        with torch.no_grad():
            self.reference_outputs = self.original_module(*self.example_inputs)

        return self

    def check_count(self, needed: Dict[str, int]):
        if "export" not in self.artifacts:
            raise RuntimeError("Must run .export() before .check_count()")
        ep: ExportedProgram = self.artifacts["export"]

        code_str = ep.graph_module.code
        for pattern, expected_count in needed.items():
            actual = code_str.count(pattern)
            if actual != expected_count:
                raise AssertionError(
                    f"Expected {expected_count} occurrences of '{pattern}' but got {actual}"
                )
        return self

    def encode_smt(self):
        if "export" not in self.artifacts:
            raise RuntimeError("Must run .export() before .encode_smt()")

        ep: ExportedProgram = self.artifacts["export"]

        smt_encoding = f"<FakeSmtEncoding for {ep}>"
        self.artifacts["smt_encoding"] = smt_encoding
        return self

    def solve_smt(self):
        if "smt_encoding" not in self.artifacts:
            raise RuntimeError("Must run .encode_smt() before .solve_smt()")

        encoding = self.artifacts["smt_encoding"]

        solver_result = "<FakeSolverResult: sat>"
        self.artifacts["smt_result"] = solver_result
        return self

    def run_method_and_compare_outputs(
        self,
        tolerance: float = 1e-4,
    ):
        if self.reference_outputs is None:
            raise RuntimeError(
                "No reference outputs to compare with. Did you .export()?"
            )
        pass

        return self

    def check(self, patterns: list[str]):
        if "export" not in self.artifacts:
            raise RuntimeError("Must run .export() before .check()")

        code_str = self.artifacts["export"].graph_module.code
        for pat in patterns:
            if pat not in code_str:
                raise AssertionError(f"Pattern '{pat}' not found in exported code.")
        return self

    def check_not(self, patterns: list[str]):
        if "export" not in self.artifacts:
            raise RuntimeError("Must run .export() before .check_not()")

        code_str = self.artifacts["export"].graph_module.code
        for pat in patterns:
            if pat in code_str:
                raise AssertionError(f"Pattern '{pat}' was unexpectedly found in code.")
        return self

    def assert_no_unsupported_ops(self):
        if "export" not in self.artifacts:
            raise RuntimeError("Must run .export() first.")
        return self
