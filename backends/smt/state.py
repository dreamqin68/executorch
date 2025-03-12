import z3
from typing import Dict, Optional, Any
import collections

class SMTExpr:
    """A wrapper for a Z3 expression."""
    def __init__(self, expr):
        # expr is a z3.ExprRef
        self.expr = expr

    def __and__(self, other):
        return SMTExpr(z3.And(self.expr, other.expr))

    def __rand__(self, other):
        return self.__and__(other)

    def __add__(self, other):
        return SMTExpr(self.expr + other.expr)

    def __mul__(self, other):
        return SMTExpr(self.expr * other.expr)

    def __eq__(self, other):
        # Returns an SMT expression representing equality.
        return SMTExpr(self.expr == other.expr)

    def __ne__(self, other):
        # Returns an SMT expression representing inequality.
        return SMTExpr(self.expr != other.expr)

    def __repr__(self):
        return f"SMTExpr({self.expr})"

    @staticmethod
    def mkBool(val: bool):
        return SMTExpr(z3.BoolVal(val))

    @staticmethod
    def var(name: str, bitwidth: int = 32):
        # For simplicity, we create a BitVec variable.
        return SMTExpr(z3.BitVec(name, bitwidth))

    @property
    def z3_expr(self):
        return self.expr


class ValueTy:
    def __init__(self, value, vtype: str = "Unknown"):
        """
        :param value: The underlying expression or data (an SMTExpr).
        :param vtype: A string describing the type (e.g. "Float", "Integer", etc.)
        """
        self.value = value
        self.vtype = vtype

    def __repr__(self):
        return f"ValueTy(type={self.vtype}, value={self.value})"


class RegFile:
    """
    A simple registry mapping IR values (or nodes) to SMT expressions.
    """
    def __init__(self):
        # Keys are arbitrary (often strings or node objects); values are ValueTy.
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
    The symbolic state that holds:
      - A precondition (an SMTExpr)
      - A registry (RegFile) mapping IR nodes/values to SMT expressions (ValueTy)
      - Additional bookkeeping such as well-definedness conditions.
    """
    def __init__(self, init_mem):
        # Start with a true precondition.
        self.precond: SMTExpr = SMTExpr.mkBool(True)
        self.welldef: Dict[Any, Dict[str, SMTExpr]] = collections.defaultdict(dict)
        self.regs = RegFile()
        self.retValues = []  # List of ValueTy if needed.
        self.hasQuantifier = False
        self.hasConstArray = False
        self.m = init_mem  # Some memory representation (could be None).

    def addPrecondition(self, e: SMTExpr):
        self.precond = self.precond & e

    def wellDefined(self, op, e: SMTExpr, desc: str = ""):
        if desc not in self.welldef[op]:
            self.welldef[op][desc] = e
        else:
            combined = self.welldef[op][desc] & e
            self.welldef[op][desc] = combined

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
