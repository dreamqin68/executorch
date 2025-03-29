import unittest
import torch

from executorch.backends.smt.test.tester.tester import SmtTester


class TestMul(unittest.TestCase):
    class Mul(torch.nn.Module):
        def forward(self, x, y):
            z = x * y
            return z

    class Mul2(torch.nn.Module):
        def forward(self, x):
            z = x * x
            return z

    class MulFunctional(torch.nn.Module):
        def forward(self, x, y):
            z = torch.mul(x, y) * torch.functional.torch.mul(x, y)
            return z

    class MulRelu(torch.nn.Module):
        def forward(self, x, y):
            z = x * y
            return torch.nn.functional.relu(z)

    def _test_mul(
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

    def test_fp16_mul(self):
        inputs = (
            torch.randn((1, 3)).to(torch.float16),
            torch.randn((4, 3)).to(torch.float16),
        )
        self._test_mul(
            inputs,
            self.Mul(),
            check_count_dict={"torch.ops.aten.mul.Tensor": 1},
            expected_smt_expr="x*y",
        )

    def test_fp32_mul(self):
        inputs = (torch.randn((1, 3)), torch.randn((4, 3)))
        self._test_mul(
            inputs,
            self.Mul(),
            check_count_dict={"torch.ops.aten.mul.Tensor": 1},
            expected_smt_expr="x*y",
        )

    def test_qs8_mul2(self):
        inputs = (torch.randn(1, 1, 4, 4),)
        self._test_mul(
            inputs,
            self.Mul2(),
            check_count_dict={"torch.ops.aten.mul.Tensor": 1},
            expected_smt_expr="x*x",
        )

    def test_qs8_mul_functional(self):
        inputs = (torch.randn(1, 1, 4, 4), torch.randn(1, 1, 4, 4))
        self._test_mul(
            inputs,
            self.MulFunctional(),
            check_count_dict={"torch.ops.aten.mul.Tensor": 3},
            expected_smt_expr="x*y*x*y",
        )


if __name__ == "__main__":
    unittest.main()
