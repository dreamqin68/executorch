import z3


class Sort:

    def __init__(self, z3_sort):
        self.z3_sort = z3_sort

    @staticmethod
    def bvSort(bw: int):
        return Sort(z3.BitVecSort(bw))

    @staticmethod
    def boolSort():
        return Sort(z3.BoolSort())

    def isBV(self):
        return self.z3_sort.kind() == z3.Z3_SORT_BV

    def bitwidth(self):
        if not self.isBV():
            raise TypeError("Not a bitvector sort!")
        return self.z3_sort.size()

    def isBool(self):
        return self.z3_sort.kind() == z3.Z3_SORT_BOOL

    def __repr__(self):
        return f"Sort({self.z3_sort})"


class Expr:

    def __init__(self, z3_expr):
        self.z3_expr = z3_expr

    @staticmethod
    def mkBV(val: int, bitwidth: int):
        return Expr(z3.BitVecVal(val, bitwidth))

    @staticmethod
    def mkBool(val: bool):
        return Expr(z3.BoolVal(val))

    @staticmethod
    def mkFreshVar(sort: Sort, prefix: str = "var"):
        if sort.isBV():
            return Expr(z3.BitVec(prefix, sort.bitwidth()))
        elif sort.isBool():
            return Expr(z3.Bool(prefix))
        else:
            raise NotImplementedError("Only BV and Bool are implemented here.")

    def sort(self) -> Sort:
        return Sort(self.z3_expr.sort())

    def bitwidth(self) -> int:
        return self.sort().bitwidth()

    def isUInt(self):
        if self.isNumeral():
            try:
                val = self.z3_expr.as_long()
                return (True, val)
            except:
                pass
        return (False, None)

    def isNumeral(self):
        return self.z3_expr.is_const() and len(self.z3_expr.children()) == 0

    def isFalse(self):
        return self.z3_expr.eq(z3.BoolVal(False))

    def isTrue(self):
        return self.z3_expr.eq(z3.BoolVal(True))

    def __neg__(self):
        return Expr(-self.z3_expr)

    def __invert__(self):
        if self.sort().isBool():
            return Expr(z3.Not(self.z3_expr))
        elif self.sort().isBV():
            return Expr(z3.BitVecVal(-1, self.bitwidth()) ^ self.z3_expr)
        else:
            raise TypeError("~ operator not supported for this sort")

    def __add__(self, other: "Expr"):
        return Expr(self.z3_expr + other.z3_expr)

    def __sub__(self, other: "Expr"):
        return Expr(self.z3_expr - other.z3_expr)

    def __mul__(self, other: "Expr"):
        return Expr(self.z3_expr * other.z3_expr)

    def __and__(self, other: "Expr"):
        if self.sort().isBool() and other.sort().isBool():
            return Expr(z3.And(self.z3_expr, other.z3_expr))
        elif self.sort().isBV() and other.sort().isBV():
            return Expr(self.z3_expr & other.z3_expr)
        else:
            raise TypeError("& not supported between these sorts")

    def __or__(self, other: "Expr"):
        if self.sort().isBool() and other.sort().isBool():
            return Expr(z3.Or(self.z3_expr, other.z3_expr))
        elif self.sort().isBV() and other.sort().isBV():
            return Expr(self.z3_expr | other.z3_expr)
        else:
            raise TypeError("| not supported between these sorts")

    def __xor__(self, other: "Expr"):
        if self.sort().isBool() and other.sort().isBool():
            return Expr(z3.Xor(self.z3_expr, other.z3_expr))
        elif self.sort().isBV() and other.sort().isBV():
            return Expr(self.z3_expr ^ other.z3_expr)
        else:
            raise TypeError("^ not supported between these sorts")

    def __eq__(self, other):
        return Expr(self.z3_expr == other.z3_expr)

    def __ne__(self, other):
        return Expr(self.z3_expr != other.z3_expr)

    def __lt__(self, other: "Expr"):
        if self.sort().isBV():
            return Expr(self.z3_expr < other.z3_expr)
        else:
            raise TypeError("< not supported for non-BV sorts")

    def __le__(self, other: "Expr"):
        if self.sort().isBV():
            return Expr(self.z3_expr <= other.z3_expr)
        else:
            raise TypeError("<= not supported for non-BV sorts")

    def __gt__(self, other: "Expr"):
        return Expr(self.z3_expr > other.z3_expr)

    def __ge__(self, other: "Expr"):
        return Expr(self.z3_expr >= other.z3_expr)

    def shl(self, other: "Expr"):
        return Expr(z3.LShR(self.z3_expr, -other.z3_expr))  # or z3.SHL if available?

    def lshr(self, other: "Expr"):
        return Expr(z3.LShR(self.z3_expr, other.z3_expr))

    def ashr(self, other: "Expr"):
        return Expr(self.z3_expr >> other.z3_expr)

    def extract(self, high: int, low: int):
        if not self.sort().isBV():
            raise TypeError("extract called on a non-BV expression")
        return Expr(z3.Extract(high, low, self.z3_expr))

    def zext(self, bits: int):
        if not self.sort().isBV():
            raise TypeError("zext on a non-BV expression")
        bw = self.bitwidth()
        return Expr(z3.ZeroExt(bits, self.z3_expr))

    def sext(self, bits: int):
        if not self.sort().isBV():
            raise TypeError("sext on a non-BV expression")
        return Expr(z3.SignExt(bits, self.z3_expr))

    def __str__(self):
        return str(self.z3_expr)

    def __repr__(self):
        return f"Expr({self.z3_expr})"


class Model:

    def __init__(self, z3_model):
        # a z3.ModelRef
        self.z3_model = z3_model

    def eval(self, expr: Expr, model_completion: bool = False) -> Expr:
        """Evaluates `expr` in this model, returning a new Expr (constant)."""
        val = self.z3_model.eval(expr.z3_expr, model_completion)
        return Expr(val)

    def __repr__(self):
        return f"Model({self.z3_model})"


class CheckResult:

    def __init__(self, z3_result):
        self.z3_result = z3_result  # typically z3 sat, unsat, or unknown

    def hasSat(self) -> bool:
        return self.z3_result == z3.sat

    def hasUnsat(self) -> bool:
        return self.z3_result == z3.unsat

    def isUnknown(self) -> bool:
        return self.z3_result == z3.unknown

    def isInconsistent(self) -> bool:
        return False

    def __repr__(self):
        return f"CheckResult({self.z3_result})"


class Solver:

    def __init__(self):
        self.solver = z3.Solver()

    def add(self, e: Expr):
        self.solver.add(e.z3_expr)

    def check(self) -> CheckResult:
        result = self.solver.check()
        return CheckResult(result)

    def getModel(self) -> Model:
        if self.solver.model() is None:
            raise RuntimeError("No model available. Call check() first.")
        return Model(self.solver.model())

    def reset(self):
        self.solver.reset()

    def __repr__(self):
        return f"Solver({self.solver})"
