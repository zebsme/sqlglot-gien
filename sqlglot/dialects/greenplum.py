import typing as t
from sqlglot import exp, tokens
from sqlglot.dialects.postgres import Postgres
from sqlglot.generator import Generator
from sqlglot.tokens import Tokenizer, TokenType
import re


class Greenplum(Postgres):  # 继承自 PostgreSQL 方言
    class Tokenizer(Postgres.Tokenizer):
        # 保留 PostgreSQL 的原始标识符和引号规则
        IDENTIFIERS = ['"']
        QUOTES = ["'"]

    class Parser(Postgres.Parser):
        PROPERTY_PARSERS = {
            **Postgres.Parser.PROPERTY_PARSERS,
            "DISTRIBUTE BY": lambda self: self._parse_distributed_property(),
        }

        def _parse_distributed_property(self) -> exp.DistributedByProperty:
            """
            解析DISTRIBUTE BY HASH语法，支持Greenplum的分布语法。
            """
            kind = "HASH"
            expressions: t.Optional[t.List[exp.Expression]] = None
            
            if self._match_text_seq("BY", "HASH"):
                expressions = self._parse_wrapped_csv(self._parse_id_var)
            elif self._match_text_seq("HASH"):
                expressions = self._parse_wrapped_csv(self._parse_id_var)
            elif self._match_text_seq("BY", "RANDOM"):
                kind = "RANDOM"

            return self.expression(
                exp.DistributedByProperty,
                expressions=expressions,
                kind=kind,
                buckets=None,
                order=None,
            )

        def _parse_types(
            self, check_func: bool = False, schema: bool = False, allow_identifiers: bool = True
        ) -> t.Optional[exp.Expression]:
            """重写类型解析方法，特殊处理CHARACTER VARYING"""
            # 先尝试使用父类的解析方法
            result = super()._parse_types(check_func, schema, allow_identifiers)
            
            # 如果解析成功且是VARCHAR类型，检查原始文本是否是CHARACTER VARYING
            if result and isinstance(result, exp.DataType) and result.this == exp.DataType.Type.VARCHAR:
                # 检查当前token的原始文本
                if hasattr(self, '_prev') and hasattr(self._prev, 'text') and self._prev.text == "CHARACTER VARYING":
                    # 设置自定义名称属性
                    result.args["custom_name"] = "CHARACTER VARYING"
            
            return result

    class Generator(Postgres.Generator):
        # 覆盖类型映射
        TYPE_MAPPING = {
            **Postgres.Generator.TYPE_MAPPING,  # 继承原有映射
        }

        """Greenplum方言的生成器，支持生成CHARACTER VARYING语法"""
        
        def datatype_sql(self, expression: exp.DataType) -> str:
            """重写数据类型生成方法，特殊处理CHARACTER VARYING"""
            # 检查是否是自定义的CHARACTER VARYING类型
            if hasattr(expression, 'args') and expression.args.get('custom_name') == "CHARACTER VARYING":
                # 生成CHARACTER VARYING格式
                expressions = self.expressions(expression, flat=True)
                if expressions:
                    return f"CHARACTER VARYING({expressions})"
                else:
                    return "CHARACTER VARYING"
            
            # 对于其他类型，使用父类的生成方法
            return super().datatype_sql(expression)

        def distributedbyproperty_sql(self, expression: exp.DistributedByProperty) -> str:
            """生成DISTRIBUTE BY HASH语法的SQL"""
            kind = expression.args.get("kind", "HASH")
            expressions = self.expressions(expression, flat=True)
            
            if kind == "HASH" and expressions:
                return f"DISTRIBUTE BY HASH({expressions})"
            elif kind == "RANDOM":
                return "DISTRIBUTE BY RANDOM"
            else:
                return f"DISTRIBUTE BY {kind}({expressions})"



