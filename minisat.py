#!/usr/bin/env python3.13
import dataclasses
import enum
import sys


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


@dataclasses.dataclass
class UnaryOperator:
    operator: Operator
    value: "ParseTreeElement"


@dataclasses.dataclass
class ParsedAtom:
    name: str


type ParseTreeElement = ParsedAtom | UnaryOperator | BinaryOperator


class Parser:
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
        raise ValueError(
            f"parsing failed, unexpected terminal: {self._current_token}"
        )


def solve(statement: str) -> str: ...


if __name__ == "__main__":
    statement = sys.argv[1]
    print(Parser(statement).parsed)
