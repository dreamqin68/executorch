import unittest

import torch
from executorch.backends.smt.test.tester.tester import SmtTester


class TestDiv(unittest.TestCase):
    class Div(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            z = x / y
            return z

    class DivSingleInput(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            z = x / x
            return z

    def _test_div(
        self,
        inputs,
        module,
        check_count_dict=None,
        expected_smt_expr=None,
    ):
        tester = SmtTester(module, inputs).export()

        if check_count_dict:
            tester.check_count(check_count_dict)

        tester.to_edge_transform_and_lower()

        if expected_smt_expr is not None:
            tester.check_smt_expression(expected_smt_expr)

    def test_fp16_div(self):
        inputs = (
            (torch.randn(1) + 4).to(torch.float16),
            (torch.randn(1) + 4).to(torch.float16),
        )
        self._test_div(
            inputs,
            self.Div(),
            check_count_dict={"torch.ops.aten.div.Tensor": 1},
            expected_smt_expr="x/y",
        )

    def test_fp32_div(self):
        # Adding 4 to move distribution away from 0, 4 Std Dev should be far enough
        inputs = (torch.randn(1) + 4, torch.randn(1) + 4)
        self._test_div(
            inputs,
            self.Div(),
            check_count_dict={"torch.ops.aten.div.Tensor": 1},
            expected_smt_expr="x/y",
        )

    def test_fp32_div_single_input(self):
        # Adding 4 to move distribution away from 0, 4 Std Dev should be far enough
        inputs = (torch.randn(1) + 4,)
        self._test_div(
            inputs,
            self.DivSingleInput(),
            check_count_dict={"torch.ops.aten.div.Tensor": 1},
            expected_smt_expr="x/x",
        )


if __name__ == "__main__":
    unittest.main()
