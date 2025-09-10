from sqlpolyglot import exp, tokens
from sqlpolyglot.dialects.postgres import Postgres
from sqlpolyglot.generator import Generator
from sqlpolyglot.tokens import Tokenizer, TokenType

# 实际应当衍生自ClickHouse
class TCHouse(Postgres): 
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

        def datatype_sql(self, expression: exp.DataType) -> str:
            if expression.is_type(exp.DataType.Type.ARRAY):
                if expression.expressions:
                    values = self.expressions(expression, key="values", flat=True)
                    return f"{self.expressions(expression, flat=True)}[{values}]"
                return "ARRAY"

            if (
                expression.is_type(exp.DataType.Type.DOUBLE)
                and expression.expressions
            ):
                # Keep DOUBLE type with precision
                return f"DOUBLE({self.expressions(expression, flat=True)})"

            return super().datatype_sql(expression)



