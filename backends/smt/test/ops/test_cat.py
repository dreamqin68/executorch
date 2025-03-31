import unittest

import torch
from executorch.backends.smt.test.tester.tester import SmtTester


class TestCat(unittest.TestCase):
    class Cat2(torch.nn.Module):
        def forward(self, arg1, arg2):
            xs = [arg1, arg2]
            x = torch.cat(xs)
            return x + x  # Quantize by propagation.

    class Cat3(torch.nn.Module):
        def forward(self, arg1, arg2, arg3):
            xs = [arg1, arg2, arg3]
            x = torch.cat(xs)
            return x + x  # Quantize by propagation.

    class Cat4(torch.nn.Module):
        def forward(self, arg1, arg2, arg3, arg4):
            xs = [arg1, arg2, arg3, arg4]
            x = torch.cat(xs)
            return x + x  # Quantize by propagation.

    class Cat5(torch.nn.Module):
        def forward(self, arg1, arg2, arg3, arg4, arg5):
            xs = [arg1, arg2, arg3, arg4, arg5]
            x = torch.cat(xs)
            return x + x  # Quantize by propagation.

    def _test_cat(
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

    def test_fp16_cat2(self):
        # """
        # Using Clamp2 because fp16 add is done in fp32 ATM. Need to fix that first.
        # """
        inputs = (
            torch.randn(1, 2, 3).to(torch.float16),
            torch.randn(3, 2, 3).to(torch.float16),
        )
        self._test_cat(
            inputs,
            self.Cat2(),
            check_count_dict={"torch.ops.aten.cat": 1},
            expected_smt_expr="concat_axis_0_inputs_2(arg1, arg2) + concat_axis_0_inputs_2(arg1, arg2)",
        )


if __name__ == "__main__":
    unittest.main()
