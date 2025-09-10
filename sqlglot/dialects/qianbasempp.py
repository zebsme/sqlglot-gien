from sqlglot import exp, tokens
from sqlglot.dialects.postgres import Postgres
from sqlglot.generator import Generator
from sqlglot.tokens import Tokenizer, TokenType


class QianBaseMPP(Postgres):  # 继承自 PostgreSQL 方言
    class Tokenizer(Postgres.Tokenizer):
        # 保留 PostgreSQL 的原始标识符和引号规则
        IDENTIFIERS = ['"']
        QUOTES = ["'"]

        # 扩展关键字映射
        KEYWORDS = {
            **Postgres.Tokenizer.KEYWORDS,
            "FLOAT8": TokenType.DOUBLE,
            "DOUBLE": TokenType.DOUBLE,
        }

    class Generator(Postgres.Generator):
        # 覆盖类型映射
        TYPE_MAPPING = {
            **Postgres.Generator.TYPE_MAPPING,  # 继承原有映射
            exp.DataType.Type.DECIMAL: "NUMERIC",  # DECIMAL → NUMERIC
            exp.DataType.Type.INT: "INT4",  # INT → INT4
            exp.DataType.Type.BIGINT: "INT8",  # BIGINT → INT8
            exp.DataType.Type.SMALLINT: "INT2",  # SMALLINT → INT2
            exp.DataType.Type.DOUBLE: "FLOAT8",  # DOUBLE → FLOAT8
            exp.DataType.Type.FLOAT: "FLOAT4",  # FLOAT → FLOAT4
        }

        


