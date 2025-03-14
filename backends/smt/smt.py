import z3


# -----------------------------------------------------------------------------
# Sort Class
# -----------------------------------------------------------------------------
class Sort:
    """
    Python analog for an SMT sort (bitvectors, booleans, etc.).
    """

    def __init__(self, z3_sort):
        """
        :param z3_sort: a z3.SortRef
        """
        self.z3_sort = z3_sort

    @staticmethod
    def bvSort(bw: int):
        """
        Creates a bitvector Sort with the given bitwidth.
        """
        return Sort(z3.BitVecSort(bw))

    @staticmethod
    def boolSort():
        """
        Creates a boolean Sort.
        """
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


# -----------------------------------------------------------------------------
# Expr Class
# -----------------------------------------------------------------------------
class Expr:
    """
    Python analog for an SMT expression. Here, we store a z3 expression (z3.ExprRef).
    We mimic many of the operator overloads and methods from the C++ version.
    """

    def __init__(self, z3_expr):
        """
        :param z3_expr: a z3.ExprRef representing the expression
        """
        self.z3_expr = z3_expr

    # --- Some static constructors (similar to C++ "mkBV", "mkBool", etc.) ---

    @staticmethod
    def mkBV(val: int, bitwidth: int):
        """
        Creates a bitvector constant of `bitwidth` bits.
        """
        return Expr(z3.BitVecVal(val, bitwidth))

    @staticmethod
    def mkBool(val: bool):
        """
        Creates a boolean constant (True or False).
        """
        return Expr(z3.BoolVal(val))

    @staticmethod
    def mkFreshVar(sort: Sort, prefix: str = "var"):
        """
        Creates a fresh variable of type `sort` with name starting by `prefix`.
        """
        if sort.isBV():
            return Expr(z3.BitVec(prefix, sort.bitwidth()))
        elif sort.isBool():
            return Expr(z3.Bool(prefix))
        else:
            raise NotImplementedError("Only BV and Bool are implemented here.")

    # --- Accessors ---

    def sort(self) -> Sort:
        return Sort(self.z3_expr.sort())

    def bitwidth(self) -> int:
        return self.sort().bitwidth()

    # --- Queries ---

    def isUInt(self):
        """
        Returns (True, value) if this expression is a concrete unsigned integer
        (in Python's sense), or (False, None) otherwise.
        """
        if self.isNumeral():
            # Try to retrieve the value (this works if the expression is a concrete BV)
            try:
                val = (
                    self.z3_expr.as_long()
                )  # If it’s out of range, can raise an exception
                return (True, val)
            except:
                pass
        return (False, None)

    def isNumeral(self):
        """True if the expression is a constant numeral (like 42, 0b101, etc.)."""
        return self.z3_expr.is_const() and len(self.z3_expr.children()) == 0

    def isFalse(self):
        return self.z3_expr.eq(z3.BoolVal(False))

    def isTrue(self):
        return self.z3_expr.eq(z3.BoolVal(True))

    # --- Operator Overloads ---

    def __neg__(self):
        return Expr(-self.z3_expr)

    def __invert__(self):
        """
        For a Boolean expression, ~ is logical NOT.
        For a BV expression, ~ is bitwise NOT.
        """
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
        """
        For booleans, & is logical AND.
        For bitvectors, & is bitwise AND.
        """
        if self.sort().isBool() and other.sort().isBool():
            return Expr(z3.And(self.z3_expr, other.z3_expr))
        elif self.sort().isBV() and other.sort().isBV():
            return Expr(self.z3_expr & other.z3_expr)
        else:
            raise TypeError("& not supported between these sorts")

    def __or__(self, other: "Expr"):
        """
        For booleans, | is logical OR.
        For bitvectors, | is bitwise OR.
        """
        if self.sort().isBool() and other.sort().isBool():
            return Expr(z3.Or(self.z3_expr, other.z3_expr))
        elif self.sort().isBV() and other.sort().isBV():
            return Expr(self.z3_expr | other.z3_expr)
        else:
            raise TypeError("| not supported between these sorts")

    def __xor__(self, other: "Expr"):
        """
        For booleans, ^ is logical XOR.
        For bitvectors, ^ is bitwise XOR.
        """
        if self.sort().isBool() and other.sort().isBool():
            return Expr(z3.Xor(self.z3_expr, other.z3_expr))
        elif self.sort().isBV() and other.sort().isBV():
            return Expr(self.z3_expr ^ other.z3_expr)
        else:
            raise TypeError("^ not supported between these sorts")

    def __eq__(self, other):
        # Return an Expr of bool sort for “=”
        return Expr(self.z3_expr == other.z3_expr)

    def __ne__(self, other):
        return Expr(self.z3_expr != other.z3_expr)

    def __lt__(self, other: "Expr"):
        # For bitvectors, use BVULT (unsigned).
        if self.sort().isBV():
            return Expr(self.z3_expr < other.z3_expr)
        else:
            # Could handle Int sorts, etc.
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

    # --- Bitvector Shifts and Extracts ---

    def shl(self, other: "Expr"):
        """Logical shift-left for bitvectors."""
        return Expr(z3.LShR(self.z3_expr, -other.z3_expr))  # or z3.SHL if available?

    def lshr(self, other: "Expr"):
        """Logical shift-right (unsigned) for bitvectors."""
        return Expr(z3.LShR(self.z3_expr, other.z3_expr))

    def ashr(self, other: "Expr"):
        """Arithmetic (signed) shift-right."""
        return Expr(self.z3_expr >> other.z3_expr)

    def extract(self, high: int, low: int):
        """
        Equivalent to the Z3 operator e[high:low].
        """
        if not self.sort().isBV():
            raise TypeError("extract called on a non-BV expression")
        return Expr(z3.Extract(high, low, self.z3_expr))

    # --- Conversions / Rewrites ---

    def zext(self, bits: int):
        """Zero-extend a bitvector by `bits` bits."""
        if not self.sort().isBV():
            raise TypeError("zext on a non-BV expression")
        bw = self.bitwidth()
        return Expr(z3.ZeroExt(bits, self.z3_expr))

    def sext(self, bits: int):
        """Sign-extend a bitvector by `bits` bits."""
        if not self.sort().isBV():
            raise TypeError("sext on a non-BV expression")
        return Expr(z3.SignExt(bits, self.z3_expr))

    # --- Overloads so that str(...) and print(...) look nice ---

    def __str__(self):
        return str(self.z3_expr)

    def __repr__(self):
        return f"Expr({self.z3_expr})"


# -----------------------------------------------------------------------------
# Model Class
# -----------------------------------------------------------------------------
class Model:
    """
    Python analog for an SMT model. In z3, this is basically a reference
    to the solver's model. We'll store it as a z3.ModelRef.
    """

    def __init__(self, z3_model):
        # a z3.ModelRef
        self.z3_model = z3_model

    def eval(self, expr: Expr, model_completion: bool = False) -> Expr:
        """Evaluates `expr` in this model, returning a new Expr (constant)."""
        val = self.z3_model.eval(expr.z3_expr, model_completion)
        return Expr(val)

    def __repr__(self):
        return f"Model({self.z3_model})"


# -----------------------------------------------------------------------------
# CheckResult Class
# -----------------------------------------------------------------------------
class CheckResult:
    """
    For the Python version, we’ll just store the result from solver.check().
    """

    def __init__(self, z3_result):
        self.z3_result = z3_result  # typically z3 sat, unsat, or unknown

    def hasSat(self) -> bool:
        return self.z3_result == z3.sat

    def hasUnsat(self) -> bool:
        return self.z3_result == z3.unsat

    def isUnknown(self) -> bool:
        return self.z3_result == z3.unknown

    def isInconsistent(self) -> bool:
        """
        Our simplified version can't truly have both SAT and UNSAT.
        We just return False here.
        """
        return False

    def __repr__(self):
        return f"CheckResult({self.z3_result})"


# -----------------------------------------------------------------------------
# Solver Class
# -----------------------------------------------------------------------------
class Solver:
    """
    Python analog for an SMT solver. This uses z3.Solver() under the hood.
    """

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


# -----------------------------------------------------------------------------
# Simple Demonstration of Usage
# -----------------------------------------------------------------------------
# if __name__ == "__main__":
#     # Example usage
#     s = Solver()

#     # x, y: fresh bitvector vars of width 8
#     x = Expr.mkFreshVar(Sort.bvSort(8), "x")
#     y = Expr.mkFreshVar(Sort.bvSort(8), "y")

#     # Add constraint: x + y == 10
#     s.add((x + y) == Expr.mkBV(10, 8))

#     # Check
#     cr = s.check()
#     print("Check result:", cr)

#     if cr.hasSat():
#         m = s.getModel()
#         valx = m.eval(x)
#         valy = m.eval(y)
#         print("Model: x =", valx, ", y =", valy)
#     else:
#         print("No model or unknown.")
