import unittest
import torch

from executorch.backends.smt.test.tester.tester import SmtTester


class TestAdd(unittest.TestCase):

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
            self._const1 = cst
            self.register_buffer("_const2", cst, persistent=False)
            self.register_parameter("_const3", torch.nn.Parameter(cst))

        def forward(self, x):
            out1 = x + self._const1 + torch.ones(1, 1, 1)
            out2 = x + self._const2 + self._const3
            return out1, out2

    def _test_add(self, inputs, module):
        (
            SmtTester(module, inputs)
            .export()  # Exports the module via torch.export
            .check_count({"torch.ops.aten.add.Tensor": 4})
            # .to_edge_transform_and_lower()
            #   or any custom checks on the IR before encoding
            # .encode_smt()  # Convert to an SMT-based representation
            # .assert_no_unsupported_ops()  # Possibly a step to ensure no leftover ops
            # .solve_smt()  # Invoke solver or do a partial check
            # .run_method_and_compare_outputs()  # Possibly run the original PyTorch and some model to compare
        )

    def test_fp32_add(self):
        inputs = (torch.randn(1), torch.randn(1))
        self._test_add(inputs, self.AddModule())

    # def test_fp16_add(self):
    #     inputs = (
    #         torch.randn(1, dtype=torch.float16),
    #         torch.randn(1, dtype=torch.float16),
    #     )
    #     self._test_add(inputs, self.AddModule())

    # def test_add_constant(self):
    #     inputs = (torch.randn(4, 4, 4),)
    #     module = self.AddConstant(torch.randn(4, 4, 4))
    #     (
    #         SmtTester(module, inputs)
    #         .export()
    #         .check_count({"aten.add.Tensor": 4})
    #         .encode_smt()
    #         .solve_smt()
    #         .run_method_and_compare_outputs()
    #     )

    # def test_add_module2(self):
    #     inputs = (torch.randn(2, 2),)
    #     (
    #         SmtTester(self.AddModule2(), inputs)
    #         .export()
    #         .check_count({"aten.add.Tensor": 1})
    #         .encode_smt()
    #         .solve_smt()
    #         .run_method_and_compare_outputs()
    #     )

    # class AddRelu(torch.nn.Module):
    #     def forward(self, x, y):
    #         z = x + y
    #         return torch.nn.functional.relu(z)

    # def test_add_relu(self):
    #     inputs = (torch.randn(1, 1, 4, 4), torch.randn(1, 1, 4, 4))
    #     (
    #         SmtTester(self.AddRelu(), inputs)
    #         .export()
    #         .check_count({"aten.add.Tensor": 1, "aten.relu.default": 1})
    #         .encode_smt()
    #         .solve_smt()
    #         .run_method_and_compare_outputs()
    #     )


if __name__ == "__main__":
    unittest.main()
