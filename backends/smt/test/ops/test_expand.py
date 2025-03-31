import unittest
from typing import Sequence, Tuple
import torch
from executorch.backends.smt.test.tester.tester import SmtTester


class TestSimpleExpand(unittest.TestCase):
    """Tests the Tensor.expand which should be converted to a repeat op by a pass."""

    class Expand(torch.nn.Module):
        # (input tensor, multiples)
        test_parameters = [
            (torch.ones(1), (2,)),
            (torch.ones(1, 4), (1, -1)),
            (torch.ones(1, 1, 2, 2), (4, 3, -1, 2)),
            (torch.ones(1), (2, 2, 4)),
            (torch.ones(3, 2, 4, 1), (-1, -1, -1, 3)),
            (torch.ones(1, 1, 192), (1, -1, -1)),
        ]

        def forward(self, x: torch.Tensor, multiples: Sequence):
            return x.expand(multiples)

    def _test_expand(
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


if __name__ == "__main__":
    unittest.main()
