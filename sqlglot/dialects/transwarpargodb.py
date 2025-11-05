from sqlglot import exp, tokens
from sqlglot.dialects.databricks import Databricks
from sqlglot.generator import Generator
from sqlglot.tokens import Tokenizer, TokenType


class TranswarpArgoDB(Databricks):  # 继承自 MySQL 方言
    class Tokenizer(Databricks.Tokenizer):
        pass

    class Parser(Databricks.Parser):
        pass

    class Generator(Databricks.Generator):
        pass
