from sqlglot import exp, tokens
from sqlglot.dialects.mysql import MySQL
from sqlglot.generator import Generator
from sqlglot.tokens import Tokenizer, TokenType
import re


class GoldenDB(MySQL):  # 继承自 MySQL 方言
    class Tokenizer(MySQL.Tokenizer):
        pass

    class Parser(MySQL.Parser):
        pass

    class Generator(MySQL.Generator):
        pass