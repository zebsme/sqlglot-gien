from sqlglot import exp, tokens
from sqlglot.dialects.athena import Athena
from sqlglot.generator import Generator
from sqlglot.tokens import Tokenizer, TokenType


class Informix(Athena):  # 继承自 Athena 方言
    class Tokenizer(Athena.Tokenizer):
        # 保留 Athena 的原始标识符和引号规则
        IDENTIFIERS = ['"']
        QUOTES = ["'"]

        # 扩展关键字映射
        KEYWORDS = {
            **Athena.Tokenizer.KEYWORDS,
        }

    class Generator(Athena.Generator):
        # 覆盖类型映射
        TYPE_MAPPING = {
            **Athena.Generator.TYPE_MAPPING,  # 继承原有映射
            exp.DataType.Type.FLOAT: "FLOAT",  # 保持 FLOAT 类型
            exp.DataType.Type.TEXT: "TEXT",    # 保持 TEXT 类型
            exp.DataType.Type.DOUBLE: "DOUBLE PRECISION",  # 保持 DOUBLE PRECISION 类型
            exp.DataType.Type.NVARCHAR: "NVARCHAR",  # 保持 NVARCHAR 类型
        }

        def datatype_sql(self, expression: exp.DataType) -> str:
            if expression.is_type(exp.DataType.Type.ARRAY):
                if expression.expressions:
                    values = self.expressions(expression, key="values", flat=True)
                    return f"{self.expressions(expression, flat=True)}[{values}]"
                return "ARRAY"


            return super().datatype_sql(expression)



