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

    def _test_add(
        self,
        inputs,
        module,
        check_count_dict=None,
        expected_smt_expr=None,
    ):
        """
        General test harness for any module. 
        - `check_count_dict`: dict of e.g. {"torch.ops.aten.add.Tensor": 4}
        - `expected_smt_expr`: a string that should appear in the final SMT expr
        """
        tester = SmtTester(module, inputs).export()  # exports the module via torch.export

        if check_count_dict:
            tester.check_count(check_count_dict)

        # Convert to edge + partition/lower
        tester.to_edge_transform_and_lower()

        # If user wants to check final SMT expression, do it here
        if expected_smt_expr is not None:
            tester.check_smt_expression(expected_smt_expr)

    def test_fp32_add(self):
        """
        Example: The AddModule has 4 add.Tensor ops and
        an expected final SMT expression "x + y + x + x + x + y + x + x"
        """
        inputs = (torch.randn(1), torch.randn(1))
        self._test_add(
            inputs,
            self.AddModule(),
            check_count_dict={"torch.ops.aten.add.Tensor": 4},
            expected_smt_expr="x + y + x + x + x + y + x + x",
        )

    def test_add_constant(self):
        inputs = (torch.randn(4, 4, 4),)
        module = self.AddConstant(torch.randn(4, 4, 4))
        self._test_add(
            inputs,
            module,
            check_count_dict={"aten.add.Tensor": 4},
            expected_smt_expr=None,  # or some partial string, if you want
        )

    def test_add_module2(self):
        inputs = (torch.randn(2, 2),)
        self._test_add(
            inputs,
            self.AddModule2(),
            check_count_dict={"torch.ops.aten.add.Tensor": 1},
            expected_smt_expr="x + x",
        )

if __name__ == "__main__":
    unittest.main()
