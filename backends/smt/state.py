import z3
from typing import Dict, Optional, Any, List
import collections


class SMTExpr:
    """A wrapper for a Z3 expression, often representing a tensor as an array."""

    def __init__(self, expr: z3.ExprRef):
        # expr is a z3.ExprRef, either a Bool, Int, Real, or Array, etc.
        self.expr = expr

    def __and__(self, other: "SMTExpr") -> "SMTExpr":
        return SMTExpr(z3.And(self.expr, other.expr))

    def __rand__(self, other: "SMTExpr") -> "SMTExpr":
        return self.__and__(other)

    def __add__(self, other: "SMTExpr") -> "SMTExpr":
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
        # Returns an SMT expression representing inequality.
        return SMTExpr(self.expr != other.expr)

    def __repr__(self):
        return f"SMTExpr({self.expr})"

    @staticmethod
    def mkBool(val: bool) -> "SMTExpr":
        """Wraps a python bool in a z3.BoolVal."""
        return SMTExpr(z3.BoolVal(val))

    @staticmethod
    def var(name: str, bitwidth: int = 32) -> "SMTExpr":
        """
        Creates a fresh variable of BitVec sort (for demonstration).
        In a real system, you might prefer Real or a more specialized sort.
        """
        return SMTExpr(z3.BitVec(name, bitwidth))

    @staticmethod
    def mkConst(val: Any) -> "SMTExpr":
        """
        Create a Z3 constant from a Python value.
          - int -> IntVal
          - float -> RealVal
          - otherwise assume val is already a z3.ExprRef or we treat it as is.
        """
        if isinstance(val, int):
            return SMTExpr(z3.IntVal(val))
        elif isinstance(val, float):
            return SMTExpr(z3.RealVal(val))
        else:
            # For other types (like array expressions), assume it's already z3.
            return SMTExpr(val)

    @staticmethod
    def gather(weight_expr: "SMTExpr", indices_expr: "SMTExpr") -> "SMTExpr":
        """
        Implements a simple gather operation.
        For demonstration, we assume weight_expr is an Array, and indices_expr is a scalar
        that references the index. We model gather as z3.Select(weight_expr, indices_expr).

        For a real system that handles vectorized indices, you'd expand this to a loop or
        a lambda expression mapping each index in indices_expr to a Select call.
        """
        return SMTExpr(z3.Select(weight_expr.expr, indices_expr.expr))

    @staticmethod
    def transpose(self, shape: List[int], perm: List[int]) -> "SMTExpr":
        """
        Symbolically transpose (or permute dimensions of) this array expression.
        In a real system, you'd:
          1) represent 'self.expr' as an array from int -> real,
          2) break the index i into multi-dimensional indices,
          3) reorder them according to 'perm',
          4) flatten them back to a 1D index,
          5) build a lambda expression mapping i -> Select(old_expr, new_index).

        For demonstration, we define a placeholder approach that uses an
        uninterpreted function representing the transposed array.
        """
        # For a real approach, you'd define a new array e.g.:
        #   i = z3.Int("i_tr")
        #   lam = z3.Lambda( i, z3.Select(self.expr, reorderIndex(i, shape, perm)) )
        # then return SMTExpr(lam)
        # Here we do a minimal placeholder:

        # Create an uninterpreted function name that references this expression.
        name_hint = "trans_of_"
        # If self.expr is a function or array, we might have a name:
        if self.expr.decl().kind() == z3.Z3_OP_UNINTERPRETED:
            base_name = self.expr.decl().name()
            fun_name = f"{name_hint}{base_name}"
        else:
            fun_name = f"{name_hint}arr"

        # Create an uninterpreted function from Int -> Real,
        # modeling the transposed array as a new function.
        # A real system would do a lambda re-mapping.
        trans_fn = z3.Function(fun_name, z3.IntSort(), z3.RealSort())
        return SMTExpr(trans_fn)

    @staticmethod
    def global_avg_pool_2d(input_expr: "SMTExpr", shape_4d: tuple) -> "SMTExpr":
        """
        Placeholder for a 'global average pooling over last two dims' of a 4D tensor.
        shape_4d is expected to be (N, C, H, W).

        In a real system, you'd define a new lambda expression or
        an uninterpreted function that asserts for each (n, c)
        the result = sum_{h,w} input_expr[n,c,h,w] / (H*W).
        For demonstration, we produce an uninterpreted function 'gap' of type
        (Int, Int) -> Real
        that stands in for that result for each (n,c).
        """
        # This is a minimal placeholder approach.
        # We'll create a new Z3 function with name 'gap_placeholder'
        # from Int->Real, ignoring that we actually have 2 dims for (n,c).
        gap_fn = z3.Function("gap_placeholder", z3.IntSort(), z3.RealSort())
        return SMTExpr(gap_fn)

    @staticmethod
    def slice(
        input_expr: "SMTExpr",
        shape: list[int],
        dim: int,
        start: int,
        size: int,
        stride: int = 1,
    ) -> "SMTExpr":
        """
        Symbolically slice a single dimension of 'input_expr' array from 'start' (inclusive)
        with length 'size' along 'dim', using 'stride'. For demonstration, we create an
        uninterpreted function to stand in for the sliced array. In a real system, you'd
        define a lambda expression that re-maps output indices to input indices.

        e.g. if out_idx is flattened or multi-d, you'd shift the dimension 'dim' by start
        and skip by stride.

        We'll do a minimal placeholder approach returning a new z3.Function to show the concept.
        """
        # For a more thorough approach, you'd do something like:
        # lam = z3.Lambda(..., z3.Select(...))
        # mapping each index -> input_expr( index modified by offset & stride).
        # We'll just do a minimal placeholder:
        slice_fn = z3.Function("slice_placeholder", z3.IntSort(), z3.RealSort())
        return SMTExpr(slice_fn)

    @staticmethod
    def sdpa(
        q_expr: "SMTExpr",
        k_expr: "SMTExpr",
        v_expr: "SMTExpr",
        mask_expr: "SMTExpr",
        scale_expr: "SMTExpr",
    ) -> "SMTExpr":
        """
        Placeholder for scaled dot-product attention:
        result = softmax((q x k^T)*scale + mask) x v
        A real system might create a function or lambda that does a matrix multiply and
        a softmax. Here we do a minimal uninterpreted function returning an array.
        """
        sdpa_fn = z3.Function("sdpa_placeholder", z3.IntSort(), z3.RealSort())
        return SMTExpr(sdpa_fn)

    @staticmethod
    def select_dim(
        input_expr: "SMTExpr", shape: list[int], dim: int, index: int
    ) -> "SMTExpr":
        """
        Symbolically select along dimension `dim` at a single `index`.
        For demonstration, we create an uninterpreted function that stands for
        'input_expr sliced at dim=dim, index=index'.
        A real system might define a reindexing lambda.

        shape is the original shape of input_expr.
        We note that dimension `dim` is removed from the shape by 1.
        """

        select_fn = z3.Function("select_dim_placeholder", z3.IntSort(), z3.RealSort())
        return SMTExpr(select_fn)

    @staticmethod
    def concat(inputs: List["SMTExpr"], axis: int) -> "SMTExpr":
        """
        Placeholder for a 'concatenate' op across 'axis'.
        For a real system, you'd define a new array expression that, for each index i,
        decides which input array it comes from based on i's offset in dimension axis.
        We do a minimal placeholder approach returning an uninterpreted function:
        concat_placeholder_axis_{axis}: Int -> Real
        If you want to handle multi-dimensional indexing precisely, you'd do a lambda
        that re-maps indices for each input in the specified 'axis' dimension.
        """

        # We'll just name a function based on how many inputs and the axis
        fun_name = f"concat_placeholder_axis_{axis}_inputs_{len(inputs)}"
        cat_fn = z3.Function(fun_name, z3.IntSort(), z3.RealSort())
        return SMTExpr(cat_fn)

    @staticmethod
    def matmul(a_expr: "SMTExpr", b_expr: "SMTExpr") -> "SMTExpr":
        """
        Placeholder for matrix multiply of two 2D (or higher) arrays.
        In a real system, you'd define constraints or a lambda for each
        output element = sum(a_expr[i,k] * b_expr[k,j]).
        For demonstration, we do an uninterpreted function from Int -> Real.
        """

        # We'll just create an uninterpreted function – a placeholder approach.
        matmul_fn = z3.Function("matmul_placeholder", z3.IntSort(), z3.RealSort())
        return SMTExpr(matmul_fn)

    @staticmethod
    def reshape(
        input_expr: "SMTExpr", old_shape: list[int], new_shape: list[int]
    ) -> "SMTExpr":
        """
        Symbolically "reshape" input_expr from old_shape to new_shape.
        In a real system, you'd define a lambda that re-maps flattened indices from
        [0..prod(old_shape)) to [0..prod(new_shape)), or at least assert that
        prod(old_shape) == prod(new_shape).
        For demonstration, we do a placeholder uninterpreted function from Int->Real.
        """
        import z3

        # We'll name an uninterpreted function with the shapes embedded, for debug:
        fn_name = (
            f"reshape_{'_'.join(map(str, old_shape))}_to_{'_'.join(map(str,new_shape))}"
        )
        reshape_fn = z3.Function(fn_name, z3.IntSort(), z3.RealSort())
        return SMTExpr(reshape_fn)

    @property
    def z3_expr(self) -> z3.ExprRef:
        return self.expr


class ValueTy:
    def __init__(self, value: SMTExpr, vtype: str = "Unknown"):
        """
        :param value: The underlying expression (SMTExpr).
        :param vtype: A string describing the 'type' (e.g. "Float", "Integer").
        """
        self.value = value
        self.vtype = vtype

    def __repr__(self):
        return f"ValueTy(type={self.vtype}, value={self.value})"


class RegFile:
    """
    Registry mapping IR values or nodes -> ValueTy (with an SMTExpr).
    """

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
    """
    Symbolic state storing:
      - A precondition
      - A registry of node->SMTExpr
      - Well-definedness constraints
      - Possibly memory state or other structures
    """

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
        """
        In a real system, for shape (d1, d2, ..., d_n), we'd represent a flattened array
        from Int -> Real. If shape is None or empty, we can use RealSort for scalars.
        """
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
