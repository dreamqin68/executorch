import z3
from typing import Dict, Optional, Any, List
import collections


class SMTExpr:

    def __init__(self, expr: z3.ExprRef):
        self.expr = expr

    def __and__(self, other: "SMTExpr") -> "SMTExpr":
        return SMTExpr(z3.And(self.expr, other.expr))

    def __rand__(self, other: "SMTExpr") -> "SMTExpr":
        return self.__and__(other)

    # def __add__(self, other: "SMTExpr") -> "SMTExpr":
    #     return SMTExpr(self.expr + other.expr)

    def __add__(self, other: "SMTExpr") -> "SMTExpr":
        if isinstance(self.expr, tuple) and isinstance(other.expr, tuple):
            if len(self.expr) != len(other.expr):
                raise ValueError("Cannot add tuples of different lengths")
            return SMTExpr(tuple(a + b for a, b in zip(self.expr, other.expr)))
        elif isinstance(self.expr, tuple):
            return SMTExpr(tuple(a + other.expr for a in self.expr))
        elif isinstance(other.expr, tuple):
            return SMTExpr(tuple(self.expr + a for a in other.expr))
        else:
            return SMTExpr(self.expr + other.expr)

    def __sub__(self, other: "SMTExpr") -> "SMTExpr":
        return SMTExpr(self.expr - other.expr)

    def __mul__(self, other: "SMTExpr") -> "SMTExpr":
        return SMTExpr(self.expr * other.expr)

    def __truediv__(self, other: "SMTExpr") -> "SMTExpr":
        return SMTExpr(self.expr / other.expr)

    def __eq__(self, other: "SMTExpr") -> "SMTExpr":
        # Returns an SMT expression representing equality.
        return SMTExpr(self.expr == other.expr)

    def __ne__(self, other: "SMTExpr") -> "SMTExpr":
        return SMTExpr(self.expr != other.expr)

    def __repr__(self):
        return f"SMTExpr({self.expr})"

    @staticmethod
    def mkBool(val: bool) -> "SMTExpr":
        """Wraps a python bool in a z3.BoolVal."""
        return SMTExpr(z3.BoolVal(val))

    # @staticmethod
    # def var(name: str, bitwidth: int = 32) -> "SMTExpr":
    #     return SMTExpr(z3.BitVec(name, bitwidth))

    @staticmethod
    def var(name: str) -> "SMTExpr":
        return SMTExpr(z3.Real(name))

    @staticmethod
    def mkConst(val: Any) -> "SMTExpr":
        if hasattr(val, "sexpr"):
            return SMTExpr(val)
        if isinstance(val, int):
            return SMTExpr(z3.IntVal(val))
        elif isinstance(val, float):
            return SMTExpr(z3.RealVal(val))
        else:
            raise TypeError(
                f"Cannot create SMT constant from value of type {type(val)}"
            )

    @staticmethod
    def gather(weight_expr: "SMTExpr", indices_expr: "SMTExpr") -> "SMTExpr":
        return SMTExpr(z3.Select(weight_expr.expr, indices_expr.expr))

    @staticmethod
    def transpose(self, shape: List[int], perm: List[int]) -> "SMTExpr":
        name_hint = "trans_of_"

        if self.expr.decl().kind() == z3.Z3_OP_UNINTERPRETED:
            base_name = self.expr.decl().name()
            fun_name = f"{name_hint}{base_name}"
        else:
            fun_name = f"{name_hint}arr"

        trans_fn = z3.Function(fun_name, z3.IntSort(), z3.RealSort())
        return SMTExpr(trans_fn)

    @staticmethod
    def global_avg_pool_2d(input_expr: "SMTExpr", shape_4d: tuple) -> "SMTExpr":
        gap_fn = z3.Function("gap", z3.IntSort(), z3.RealSort())
        expr = gap_fn(input_expr)
        return SMTExpr(expr)

    @staticmethod
    def slice(
        input_expr: "SMTExpr",
        shape: list[int],
        dim: int,
        start: int,
        size: int,
        stride: int = 1,
    ) -> "SMTExpr":
        slice_fn = z3.Function("slice", z3.IntSort(), z3.RealSort())
        return SMTExpr(slice_fn)

    @staticmethod
    def sdpa(
        q_expr: "SMTExpr",
        k_expr: "SMTExpr",
        v_expr: "SMTExpr",
        mask_expr: "SMTExpr",
        scale_expr: "SMTExpr",
    ) -> "SMTExpr":
        sdpa_fn = z3.Function("sdpa", z3.IntSort(), z3.RealSort())
        return SMTExpr(sdpa_fn)

    @staticmethod
    def select_dim(
        input_expr: "SMTExpr", shape: list[int], dim: int, index: int
    ) -> "SMTExpr":
        select_fn = z3.Function("select_dim", z3.IntSort(), z3.RealSort())
        return SMTExpr(select_fn)

    @staticmethod
    def concat(inputs: List["SMTExpr"], axis: int) -> "SMTExpr":
        fun_name = f"concat_axis_{axis}_inputs_{len(inputs)}"
        domain_sorts = [z3.RealSort() for _ in inputs]
        range_sort = z3.RealSort()

        cat_fn = z3.Function(fun_name, *(domain_sorts + [range_sort]))
        expr = cat_fn(*[inp.expr for inp in inputs])
        return SMTExpr(expr)

    @staticmethod
    def matmul(a_expr: "SMTExpr", b_expr: "SMTExpr") -> "SMTExpr":
        matmul_fn = z3.Function("matmul", z3.IntSort(), z3.RealSort())
        return SMTExpr(matmul_fn)

    @staticmethod
    def reshape(
        input_expr: "SMTExpr", old_shape: list[int], new_shape: list[int]
    ) -> "SMTExpr":
        fn_name = (
            f"reshape_{'_'.join(map(str, old_shape))}_to_{'_'.join(map(str,new_shape))}"
        )
        reshape_fn = z3.Function(fn_name, z3.IntSort(), z3.RealSort())
        return SMTExpr(reshape_fn)

    @staticmethod
    def transpose_nd(input_expr: "SMTExpr", perm: list[int]) -> "SMTExpr":
        fn_name = "transpose_nd_" + "_".join(map(str, perm))
        trans_fn = z3.Function(fn_name, z3.IntSort(), z3.RealSort())
        return SMTExpr(trans_fn)

    @staticmethod
    def mm(a_expr: "SMTExpr", b_expr: "SMTExpr") -> "SMTExpr":
        mm_fn = z3.Function("mm", z3.RealSort(), z3.RealSort(), z3.RealSort())
        expr = mm_fn(a_expr, b_expr)
        return SMTExpr(expr)

    @staticmethod
    def scatter_nd(
        input_expr: "SMTExpr", indices_expr: "SMTExpr", value_expr: "SMTExpr"
    ) -> "SMTExpr":
        scatter_fn = z3.Function("scatter_nd", z3.IntSort(), z3.RealSort())
        return SMTExpr(scatter_fn)

    @staticmethod
    def unsqueeze(input_expr: "SMTExpr", dim: int) -> "SMTExpr":

        fn_name = f"unsqueeze_dim_{dim}"
        unsq_fn = z3.Function(fn_name, z3.IntSort(), z3.RealSort())
        return SMTExpr(unsq_fn)

    @staticmethod
    def expand(
        input_expr: "SMTExpr", old_shape: list[int], new_shape: list[int]
    ) -> "SMTExpr":

        fn_name = (
            f"expand_{'_'.join(map(str,old_shape))}_to_{'_'.join(map(str,new_shape))}"
        )
        expand_fn = z3.Function(fn_name, z3.IntSort(), z3.RealSort())
        return SMTExpr(expand_fn)

    @staticmethod
    def bmm(a_expr: "SMTExpr", b_expr: "SMTExpr") -> "SMTExpr":
        # fn = z3.Function("bmm", z3.IntSort(), z3.RealSort())
        fn = z3.Function("bmm", z3.RealSort(), z3.RealSort(), z3.RealSort())
        expr = fn(a_expr.expr, b_expr.expr)
        return SMTExpr(expr)

    @staticmethod
    def dim_order_copy(input_expr: "SMTExpr") -> "SMTExpr":
        fn = z3.Function("dim_order_copy", z3.IntSort(), z3.RealSort())
        expr = fn(input_expr)
        return SMTExpr(expr)

    @staticmethod
    def softmax(input_expr: "SMTExpr", dim: int) -> "SMTExpr":
        fn_name = f"softmax_dim_{dim}"
        softmax_fn = z3.Function(fn_name, z3.IntSort(), z3.RealSort())
        return SMTExpr(softmax_fn)

    @property
    def z3_expr(self) -> z3.ExprRef:
        return self.expr


class ValueTy:
    def __init__(self, value: SMTExpr, vtype: str = "Unknown"):
        self.value = value
        self.vtype = vtype

    def __repr__(self):
        return f"ValueTy(type={self.vtype}, value={self.value})"


class RegFile:

    def __init__(self):
        self.m: Dict[Any, ValueTy] = {}

    def addValueTy(self, v, valty: ValueTy):
        if v in self.m:
            raise ValueError(f"Key {v} already in RegFile.")
        self.m[v] = valty

    def addExpr(self, v, expr: SMTExpr, vtype: str):
        if v in self.m:
            raise ValueError(f"Key {v} already in RegFile.")
        self.m[v] = ValueTy(expr, vtype)

    def findOrCrash(self, v) -> ValueTy:
        if v not in self.m:
            raise KeyError(f"Cannot find key: {v}")
        return self.m[v]

    def contains(self, v) -> bool:
        return v in self.m

    def getExpr(self, v) -> SMTExpr:
        valty = self.findOrCrash(v)
        return valty.value

    def __iter__(self):
        return iter(self.m.items())


class State:

    def __init__(self, init_mem: Optional[Any] = None):
        self.precond: SMTExpr = SMTExpr.mkBool(True)
        self.welldef: Dict[Any, Dict[str, SMTExpr]] = collections.defaultdict(dict)
        self.regs = RegFile()
        self.retValues = []
        self.hasQuantifier = False
        self.hasConstArray = False
        self.m = init_mem

    def addPrecondition(self, e: SMTExpr):
        self.precond = self.precond & e

    def wellDefined(self, op, e: SMTExpr, desc: str = ""):
        if desc not in self.welldef[op]:
            self.welldef[op][desc] = e
        else:
            self.welldef[op][desc] = self.welldef[op][desc] & e

    def preconditionExpr(self) -> SMTExpr:
        return self.precond

    def isWellDefined(self) -> SMTExpr:
        ret = SMTExpr.mkBool(True)
        for op_map in self.welldef.values():
            for expr in op_map.values():
                ret = ret & expr
        return ret

    def isOpWellDefined(self, op) -> SMTExpr:
        if op not in self.welldef:
            return SMTExpr.mkBool(True)
        ret = SMTExpr.mkBool(True)
        for expr in self.welldef[op].values():
            ret = ret & expr
        return ret

    def getOpWellDefinedness(self, op) -> Dict[str, SMTExpr]:
        if op not in self.welldef:
            return {}
        return dict(self.welldef[op])

    def get_sort_from_shape(self, shape: Optional[tuple]) -> z3.Sort:
        if shape is None or len(shape) == 0:
            return z3.RealSort()
        return z3.ArraySort(z3.IntSort(), z3.RealSort())

    def __repr__(self):
        lines = []
        lines.append("=== State Debug ===")
        lines.append(f"Precondition: {self.precond}")
        lines.append("Register File:")
        for k, v in self.regs:
            lines.append(f"  Key: {k}, Value: {v}")
        lines.append("Well-Definedness Conditions:")
        for op, desc_map in self.welldef.items():
            for desc, expr in desc_map.items():
                lines.append(f"  Op: {op}, Desc: {desc}, Expr: {expr}")
        return "\n".join(lines)
