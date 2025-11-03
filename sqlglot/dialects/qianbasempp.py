from sqlglot import exp, tokens
from sqlglot.dialects.postgres import Postgres
from sqlglot.generator import Generator
from sqlglot.parser import Parser
from sqlglot.tokens import Tokenizer, TokenType
import typing as t


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

    class Parser(Parser):
        PROPERTY_PARSERS = {
            **Parser.PROPERTY_PARSERS,
            "DISTRIBUTED BY": lambda self: self._parse_distributed_property(),
        }
        
        def _parse_distributed_property(self) -> exp.DistributedByProperty:
            """
            解析DISTRIBUTED BY语法，支持qianBaseMPP的DISTRIBUTED BY (column)语法。
            
            关键逻辑：
            - 匹配DISTRIBUTED BY关键字
            - 解析括号内的列名列表
            - 默认分布方式为HASH
            """
            kind = "HASH"
            expressions: t.Optional[t.List[exp.Expression]] = None
            
            # 解析括号内的列名列表
            expressions = self._parse_wrapped_csv(self._parse_id_var)
            
            return self.expression(
                exp.DistributedByProperty,
                expressions=expressions,
                kind=kind,
                buckets=None,
                order=None,
            )

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

        def distributedbyproperty_sql(self, expression: exp.DistributedByProperty) -> str:
            """
            生成DISTRIBUTED BY属性的SQL语句。
            """
            if not expression.expressions:
                return ""
            
            expressions_sql = ", ".join(self.sql(e) for e in expression.expressions)
            return f"DISTRIBUTED BY ({expressions_sql})"


