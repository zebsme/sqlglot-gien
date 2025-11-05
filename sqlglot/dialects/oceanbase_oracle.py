from __future__ import annotations

import typing as t

from sqlglot import exp, tokens
from sqlglot.dialects.oracle import Oracle
from sqlglot.generator import Generator
from sqlglot.tokens import Tokenizer, TokenType


class OceanBase_Oracle(Oracle):  # 继承自 MySQL 方言
    class Tokenizer(Oracle.Tokenizer):
        pass

    class Parser(Oracle.Parser):
        pass

    class Generator(Oracle.Generator):
        pass



