import unittest
import torch

from executorch.backends.smt.test.tester.tester import SmtTester


class TestSub(unittest.TestCase):
    class Sub(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            z = x - y
            return z

    class Sub2(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            z = x - x
            return z

    def _test_sub(
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

    def test_fp16_sub(self):
        inputs = (
            torch.randn((1, 3)).to(torch.float16),
            torch.randn((4, 3)).to(torch.float16),
        )
        self._test_sub(
            inputs,
            self.Sub(),
            check_count_dict={"torch.ops.aten.sub.Tensor": 1},
            expected_smt_expr="x - y",
        )
        self._test_sub(inputs)


if __name__ == "__main__":
    unittest.main()
