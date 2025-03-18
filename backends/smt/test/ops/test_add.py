import unittest
import torch

import sys
import os

# Ensure the 'executorch/' directory is in sys.path:
current_file = os.path.abspath(__file__)
ops_dir = os.path.dirname(current_file)  # .../executorch/backends/smt/test/ops
smt_test_dir = os.path.dirname(ops_dir)  # .../executorch/backends/smt/test
smt_dir = os.path.dirname(smt_test_dir)  # .../executorch/backends/smt
backends_dir = os.path.dirname(smt_dir)  # .../executorch/backends
executorch_dir = os.path.dirname(backends_dir)  # .../executorch
if executorch_dir not in sys.path:
    sys.path.insert(0, executorch_dir)

# Now import with the fully-qualified module path:
from backends.smt.test.tester.tester import SmtTester

# Use SmtTester below...


# from backends.smt.test.tester.tester import SmtTester
from tester.tester import SmtTester


class TestAdd(unittest.TestCase):
    """
    A test suite for Add ops in an SMT backend, akin to the XNNPack-based test suite.
    It uses a hypothetical 'SmtTester' that might:
      - export the model
      - optionally quantize if relevant
      - produce an SMT encoding
      - check constraints or do some solver-level verifications
    """

    class AddModule(torch.nn.Module):
        def forward(self, x, y):
            z = x + y
            z = z + x
            z = z + x
            z = z + z
            return z

    class AddModule2(torch.nn.Module):
        def forward(self, x):
            z = x + x
            return z

    class AddConstant(torch.nn.Module):
        def __init__(self, cst):
            super().__init__()
            # We'll store a few forms of constants similarly
            self._const1 = cst
            self.register_buffer("_const2", cst, persistent=False)
            self.register_parameter("_const3", torch.nn.Parameter(cst))

        def forward(self, x):
            out1 = x + self._const1 + torch.ones(1, 1, 1)
            out2 = x + self._const2 + self._const3
            return out1, out2

    def _test_add(self, inputs, module):
        """
        A helper method that sets up SmtTester steps for an 'Add'-like module,
        parallel to how XNNPack's test code does repeated steps.
        """
        (
            SmtTester(module, inputs)
            .export()  # Exports the module via torch.export
            .check_symbolic_count({"aten.add.Tensor": 4})
            #   or any custom checks on the IR before encoding
            .encode_smt()  # Convert to an SMT-based representation
            .assert_no_unsupported_ops()  # Possibly a step to ensure no leftover ops
            .solve_smt()  # Invoke solver or do a partial check
            .run_and_compare_outputs()  # Possibly run the original PyTorch and some model to compare
        )

    def test_fp32_add(self):
        # Test the basic AddModule in fp32
        inputs = (torch.randn(1), torch.randn(1))
        self._test_add(inputs, self.AddModule())

    def test_fp16_add(self):
        # Similarly test in float16
        inputs = (
            torch.randn(1, dtype=torch.float16),
            torch.randn(1, dtype=torch.float16),
        )
        self._test_add(inputs, self.AddModule())

    def test_add_constant(self):
        # A test for a module with constants
        inputs = (torch.randn(4, 4, 4),)
        module = self.AddConstant(torch.randn(4, 4, 4))
        (
            SmtTester(module, inputs)
            .export()
            .check_symbolic_count({"aten.add.Tensor": 4})
            .encode_smt()
            .solve_smt()
            .run_and_compare_outputs()
        )

    def test_add_module2(self):
        # Another example with a single input
        inputs = (torch.randn(2, 2),)
        (
            SmtTester(self.AddModule2(), inputs)
            .export()
            .check_symbolic_count({"aten.add.Tensor": 1})
            .encode_smt()
            .solve_smt()
            .run_and_compare_outputs()
        )

    class AddRelu(torch.nn.Module):
        def forward(self, x, y):
            z = x + y
            return torch.nn.functional.relu(z)

    def test_add_relu(self):
        # Test an add followed by relu
        inputs = (torch.randn(1, 1, 4, 4), torch.randn(1, 1, 4, 4))
        (
            SmtTester(self.AddRelu(), inputs)
            .export()
            .check_symbolic_count({"aten.add.Tensor": 1, "aten.relu.default": 1})
            .encode_smt()
            .solve_smt()
            .run_and_compare_outputs()
        )


if __name__ == "__main__":
    unittest.main()
