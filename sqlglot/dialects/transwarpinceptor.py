from sqlglot import exp, tokens
from sqlglot.dialects.mysql import MySQL
from sqlglot.generator import Generator
from sqlglot.tokens import Tokenizer, TokenType


class TranswarpInceptor(MySQL):  # 继承自 MySQL 方言
    class Tokenizer(MySQL.Tokenizer):
        # 扩展关键字映射
        KEYWORDS = {
            **MySQL.Tokenizer.KEYWORDS,
        }

    class Generator(MySQL.Generator):
        # 覆盖类型映射
        TYPE_MAPPING = {
            **MySQL.Generator.TYPE_MAPPING,  # 继承原有映射
            exp.DataType.Type.TEXT: "STRING",
            exp.DataType.Type.TIMESTAMP: "TIMESTAMP",

        }

        def datatype_sql(self, expression: exp.DataType) -> str:
            if (
                self.VARCHAR_REQUIRES_SIZE
                and expression.is_type(exp.DataType.Type.VARCHAR)
                and not expression.expressions
            ):
                # `VARCHAR` must always have a size - if it doesn't, we always generate `TEXT`
                return "VARCHAR"

            # https://dev.mysql.com/doc/refman/8.0/en/numeric-type-syntax.html
            result = super().datatype_sql(expression)
            if expression.this in self.UNSIGNED_TYPE_MAPPING:
                result = f"{result} UNSIGNED"

            return result



