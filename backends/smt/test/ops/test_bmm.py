import unittest

import torch
from executorch.backends.smt.test.tester.tester import SmtTester


class TestBMM(unittest.TestCase):
    class BMM(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.bmm(x, y)

    def _test_bmm(
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

    def test_fp16_bmm(self):
        inputs = (
            torch.randn(2, 3, 4).to(torch.float16),
            torch.randn(2, 4, 6).to(torch.float16),
        )
        self._test_bmm(
            inputs,
            self.BMM(),
            check_count_dict={"torch.ops.aten.bmm.default": 1},
            expected_smt_expr="bmm(x, y)",
        )

    def test_fp32_bmm(self):
        inputs = (
            torch.randn(2, 3, 4),
            torch.randn(2, 4, 6),
        )
        self._test_bmm(
            inputs,
            self.BMM(),
            check_count_dict={"torch.ops.aten.bmm.default": 1},
            expected_smt_expr="bmm(x, y)",
        )


if __name__ == "__main__":
    unittest.main()
