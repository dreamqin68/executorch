import _operator

from executorch.exir.dialects._ops import ops as exir_ops

not_supported_operator = [
    exir_ops.edge.aten.arange.start_step,
    exir_ops.edge.aten.clone.default,
    exir_ops.edge.aten.full.default,
    exir_ops.edge.aten.slice_scatter.default,
    exir_ops.edge.aten.copy.default,
    exir_ops.edge.quantized_decomposed.embedding_4bit.dtype,
]

to_be_implemented_operator = [
    exir_ops.edge.aten.any.dim,
    exir_ops.edge.aten.eq.Scalar,
    exir_ops.edge.aten.full_like.default,
    exir_ops.edge.aten.logical_not.default,
    exir_ops.edge.aten.where.self,
]

allow_list_operator = [
    exir_ops.edge.aten.add.Tensor,
    exir_ops.edge.aten.sub.Tensor,
    exir_ops.edge.aten.permute_copy.default,
    exir_ops.edge.aten.view_copy.default,
    exir_ops.edge.aten.mm.default,
    exir_ops.edge.aten.slice_copy.Tensor,
    exir_ops.edge.aten.mul.Tensor,
    exir_ops.edge.aten.cat.default,
    exir_ops.edge.aten.index_put.default,
    exir_ops.edge.aten.clone.default,
    exir_ops.edge.aten.unsqueeze_copy.default,
    exir_ops.edge.aten.expand_copy.default,
    exir_ops.edge.aten.bmm.default,
    exir_ops.edge.aten.div.Tensor,
    exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
    exir_ops.edge.aten._softmax.default,
]
