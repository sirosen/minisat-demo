import dataclasses
import enum
import functools
import sys
import time
import typing as t

from pyrsistent import PMap, pmap

# allow deeper parsing
sys.setrecursionlimit(2000)


class Operator(enum.Enum):
    AND = enum.auto()
    OR = enum.auto()
    NOT = enum.auto()


class Paren(enum.Enum):
    LEFT = enum.auto()
    RIGHT = enum.auto()


@dataclasses.dataclass
class Atom:
    name: str


type Token = Atom | Operator | Paren


@dataclasses.dataclass
class BinaryOperator:
    operator: Operator
    lhs: "ParseTreeElement"
    rhs: "ParseTreeElement"

    def __post_init__(self) -> None:
        self._saved_str = _str_of_tree(self)
        self._hashval = hash(self._saved_str)

    def __hash__(self) -> int:
        return self._hashval


@dataclasses.dataclass
class UnaryOperator:
    operator: Operator
    value: "ParseTreeElement"

    def __post_init__(self) -> None:
        self._saved_str = _str_of_tree(self)
        self._hashval = hash(self._saved_str)

    def __hash__(self) -> int:
        return self._hashval


@dataclasses.dataclass
class ParsedAtom:
    name: str

    def __post_init__(self) -> None:
        self._saved_str = _str_of_tree(self)
        self._hashval = hash(self._saved_str)

    def __hash__(self) -> int:
        return self._hashval


type ParseTreeElement = ParsedAtom | UnaryOperator | BinaryOperator


def _str_of_tree(tree: ParseTreeElement) -> str:
    if (saved_str := getattr(tree, "_saved_str", None)) is not None:
        return saved_str
    match tree:
        case ParsedAtom(name=n):
            return n
        case UnaryOperator(Operator.NOT, value):
            return f"~({_str_of_tree(value)})"
        case BinaryOperator(Operator.OR, lhs=l, rhs=r):
            return f"({_str_of_tree(l)}|{_str_of_tree(r)})"
        case BinaryOperator(Operator.AND, lhs=l, rhs=r):
            return f"({_str_of_tree(l)}&{_str_of_tree(r)})"
        case _:
            raise NotImplementedError("unreachable")


class _Parser:
    """Itty-bitty recursive-descent parser"""

    def __init__(self, statement: str) -> None:
        self.statement = statement
        self.tokens = self._tokenize()

        self._parse_idx = 0
        self.parsed = self._parse()

        if not self._done():
            raise ValueError("parsing failed: not all input consumed")

    def _done(self) -> bool:
        return self._parse_idx >= len(self.tokens)

    @property
    def _current_token(self) -> Token:
        if self._done():
            raise ValueError("parsing failed: unterminated expression")
        return self.tokens[self._parse_idx]

    def _advance(self) -> None:
        self._parse_idx += 1

    def _tokenize(self) -> list[Token]:
        tokens: list[Token] = []
        current_atom: list[str] = []

        def lex_atom():
            nonlocal current_atom
            if current_atom:
                tokens.append(Atom("".join(current_atom)))
                current_atom = []

        for char in self.statement:
            match char:
                case " ":
                    lex_atom()
                case "&":
                    lex_atom()
                    tokens.append(Operator.AND)
                case "|":
                    lex_atom()
                    tokens.append(Operator.OR)
                case "~":
                    lex_atom()
                    tokens.append(Operator.NOT)
                case "(":
                    lex_atom()
                    tokens.append(Paren.LEFT)
                case ")":
                    lex_atom()
                    tokens.append(Paren.RIGHT)
                case _:
                    current_atom.append(char)
        lex_atom()
        return tokens

    def _parse(self) -> ParseTreeElement:
        return self._parse_or()

    def _parse_or(self) -> ParseTreeElement:
        lhs = self._parse_and()
        if not self._done() and self._current_token == Operator.OR:
            self._advance()
            rhs = self._parse_or()
            return BinaryOperator(Operator.OR, lhs, rhs)
        return lhs

    def _parse_and(self) -> ParseTreeElement:
        lhs = self._parse_not()
        if not self._done() and self._current_token == Operator.AND:
            self._advance()
            rhs = self._parse_and()
            return BinaryOperator(Operator.AND, lhs, rhs)
        return lhs

    def _parse_not(self) -> ParseTreeElement:
        if self._current_token == Operator.NOT:
            self._advance()
            inner = self._parse_not()
            return UnaryOperator(Operator.NOT, inner)
        return self._parse_expr()

    def _parse_expr(self) -> ParseTreeElement:
        if self._current_token == Paren.LEFT:
            self._advance()
            ret = self._parse_or()
            if self._current_token != Paren.RIGHT:
                raise ValueError("parsing failed: unbalanced parens")
            self._advance()
            return ret
        if isinstance(self._current_token, Atom):
            ret = ParsedAtom(self._current_token.name)
            self._advance()
            return ret
        raise ValueError(f"parsing failed, unexpected terminal: {self._current_token}")


def parse(statement: str) -> ParseTreeElement:
    return _Parser(statement).parsed


@functools.cache
def all_atom_names(tree: ParseTreeElement) -> set[str]:
    if isinstance(tree, ParsedAtom):
        return {tree.name}
    elif isinstance(tree, UnaryOperator):
        return all_atom_names(tree.value)
    elif isinstance(tree, BinaryOperator):
        return all_atom_names(tree.lhs).union(all_atom_names(tree.rhs))
    else:
        t.assert_never(tree)


# --- end parser ---


class SATHypothesis:
    def __init__(self, hypotheses: t.Mapping[str, bool]) -> None:
        self.values: PMap[str, bool] = pmap(hypotheses)
        self._hashval = hash((self.__class__, self.values))

    def set(self, name: str, value: bool) -> "SATHypothesis":
        return SATHypothesis({name: value, **self.values})

    def __hash__(self) -> int:
        return self._hashval


@functools.cache
def _evaluate_hypothesis(tree: ParseTreeElement, hypothesis: SATHypothesis) -> bool:
    match tree:
        case ParsedAtom(name=n):
            return hypothesis.values[n]
        case UnaryOperator(operator=Operator.NOT, value=value):
            return not _evaluate_hypothesis(value, hypothesis)
        case BinaryOperator(operator=Operator.OR, lhs=l, rhs=r):
            return _evaluate_hypothesis(l, hypothesis) or _evaluate_hypothesis(
                r, hypothesis
            )
        case BinaryOperator(operator=Operator.AND, lhs=l, rhs=r):
            return _evaluate_hypothesis(l, hypothesis) and _evaluate_hypothesis(
                r, hypothesis
            )
        case _:
            raise NotImplementedError("unreachable: bad parse")


def _evolutions(
    names: set[str], hypothesis: SATHypothesis
) -> t.Iterable[SATHypothesis]:
    missing_names = names.difference(hypothesis.values.keys())
    for name in missing_names:
        for potential_value in (True, False):
            yield hypothesis.set(name, potential_value)


def _is_satisfiable_under_hypothesis(
    tree: ParseTreeElement, hypothesis: SATHypothesis
) -> bool:
    all_names = all_atom_names(tree)
    if all_names <= set(hypothesis.values.keys()):
        return _evaluate_hypothesis(tree, hypothesis)
    else:
        return any(
            _is_satisfiable_under_hypothesis(tree, evolved_hypothesis)
            for evolved_hypothesis in _evolutions(all_names, hypothesis)
        )


def satisfiable(statement: str) -> bool:
    tree = parse(statement)
    return _is_satisfiable_under_hypothesis(tree, SATHypothesis({}))


if __name__ == "__main__":
    statement = sys.argv[1].strip()
    if statement == "-":
        statement = sys.stdin.read().strip()
    before = time.time()
    result = satisfiable(statement)
    after = time.time()
    display_statement = statement if len(statement) < 1000 else f"{statement[:1000]}..."
    print()
    print(f"statement: {display_statement}")
    print()
    print("    is it satisfiable?")
    print(f"    ... {'yes' if result else 'no'}")
    print(f"    ... solved in {after - before}s")
