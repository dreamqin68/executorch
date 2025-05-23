# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024-2025 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from typing import Tuple

import torch
from executorch.backends.arm.quantizer import (
    EthosUQuantizer,
    get_symmetric_quantization_config,
    TOSAQuantizer,
)
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.arm_tester import ArmTester
from executorch.backends.arm.tosa_specification import TosaSpecification
from executorch.backends.xnnpack.test.tester.tester import Quantize
from executorch.exir.backend.backend_details import CompileSpec
from parameterized import parameterized


test_data_suite = [
    # (test_name, test_data)
    ("zeros", torch.zeros(1, 10, 10, 10)),
    ("ones", torch.ones(10, 10, 10)),
    ("rand", torch.rand(10, 10) - 0.5),
    ("randn_pos", torch.randn(10) + 10),
    ("randn_neg", torch.randn(10) - 10),
    ("ramp", torch.arange(-16, 16, 0.2)),
]


class TestRelu(unittest.TestCase):
    class Relu(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            return self.relu(x)

    def _test_relu_tosa_MI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.tensor]
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80+MI"),
            )
            .export()
            .check(["torch.ops.aten.relu.default"])
            .check_not(["torch.ops.quantized_decomposed"])
            .to_edge()
            .partition()
            .check_not(["executorch_exir_dialects_edge__ops_aten_relu_default"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data)
        )

    def _test_relu_tosa_BI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.tensor]
    ):
        tosa_spec = TosaSpecification.create_from_string("TOSA-0.80+BI")
        compile_spec = common.get_tosa_compile_spec(tosa_spec)
        quantizer = TOSAQuantizer(tosa_spec).set_io(get_symmetric_quantization_config())
        (
            ArmTester(module, example_inputs=test_data, compile_spec=compile_spec)
            .quantize(Quantize(quantizer, get_symmetric_quantization_config()))
            .export()
            .check_count({"torch.ops.aten.relu.default": 1})
            .check(["torch.ops.quantized_decomposed"])
            .to_edge()
            .partition()
            .check_not(["executorch_exir_dialects_edge__ops_aten_relu_default"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data)
        )

    def _test_relu_ethosu_BI_pipeline(
        self,
        compile_spec: CompileSpec,
        module: torch.nn.Module,
        test_data: Tuple[torch.tensor],
    ):
        quantizer = EthosUQuantizer(compile_spec).set_io(
            get_symmetric_quantization_config()
        )
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=compile_spec,
            )
            .quantize(Quantize(quantizer, get_symmetric_quantization_config()))
            .export()
            .check_count({"torch.ops.aten.relu.default": 1})
            .check(["torch.ops.quantized_decomposed"])
            .to_edge()
            .partition()
            .check_not(["executorch_exir_dialects_edge__ops_aten_relu_default"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
        )

    @parameterized.expand(test_data_suite)
    def test_relu_tosa_MI(
        self,
        test_name: str,
        test_data: torch.Tensor,
    ):
        self._test_relu_tosa_MI_pipeline(self.Relu(), (test_data,))

    @parameterized.expand(test_data_suite)
    def test_relu_tosa_BI(self, test_name: str, test_data: torch.Tensor):
        self._test_relu_tosa_BI_pipeline(self.Relu(), (test_data,))

    @parameterized.expand(test_data_suite)
    def test_relu_u55_BI(self, test_name: str, test_data: torch.Tensor):
        self._test_relu_ethosu_BI_pipeline(
            common.get_u55_compile_spec(), self.Relu(), (test_data,)
        )

    @parameterized.expand(test_data_suite)
    def test_relu_u85_BI(self, test_name: str, test_data: torch.Tensor):
        self._test_relu_ethosu_BI_pipeline(
            common.get_u85_compile_spec(), self.Relu(), (test_data,)
        )
