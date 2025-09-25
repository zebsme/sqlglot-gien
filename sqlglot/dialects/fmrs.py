from sqlglot import exp, tokens, parser
from sqlglot.dialects.databricks import Databricks
from sqlglot.generator import Generator
from sqlglot.tokens import Tokenizer, TokenType


class FMRS(Databricks):  # 继承自 Databricks 方言
    class Tokenizer(Databricks.Tokenizer):
        # 保留 Databricks 的原始标识符和引号规则
        IDENTIFIERS = ['`']
        QUOTES = ["'"]

        # 扩展关键字映射
        KEYWORDS = {
            **Databricks.Tokenizer.KEYWORDS,
            "DISABLE": TokenType.DISABLE,
            "NOVALIDATE": TokenType.NOVALIDATE,
        }

    class Generator(Databricks.Generator):
        # 覆盖类型映射
        TYPE_MAPPING = {
            **Databricks.Generator.TYPE_MAPPING,
            # 华为FMRS特有的类型映射
            exp.DataType.Type.TEXT: "string",
            exp.DataType.Type.DOUBLE: "double",
            exp.DataType.Type.INT: "int",
            exp.DataType.Type.BIGINT: "bigint",
            exp.DataType.Type.SMALLINT: "smallint",
            exp.DataType.Type.TINYINT: "tinyint",
            exp.DataType.Type.BOOLEAN: "boolean",
            exp.DataType.Type.DATE: "date",
            exp.DataType.Type.TIMESTAMP: "timestamp",
            exp.DataType.Type.DECIMAL: "decimal",
            exp.DataType.Type.FLOAT: "float",
            exp.DataType.Type.VARCHAR: "varchar",
            exp.DataType.Type.CHAR: "char",
        }

        def primarykey_sql(self, expression: exp.PrimaryKey) -> str:
            """生成主键约束，支持DISABLE NOVALIDATE"""
            columns = self.expressions(expression, key="expressions")
            sql = f"PRIMARY KEY ({columns})"
            
            # 检查是否有DISABLE NOVALIDATE属性
            options = expression.args.get('options', [])
            if options and isinstance(options, list):
                if "DISABLE" in options:
                    sql += " DISABLE"
                    if "NOVALIDATE" in options:
                        sql += " NOVALIDATE"
            
            return sql

    class Parser(Databricks.Parser):
        """FMRS方言的解析器，支持DISABLE NOVALIDATE语法"""
        
        def _parse_primary_key(self):
            """解析主键约束，支持DISABLE NOVALIDATE"""
            self._match(TokenType.PRIMARY_KEY)
            
            # 解析主键列
            columns = self._parse_wrapped_csv(self._parse_expression)
            
            # 检查是否有DISABLE NOVALIDATE
            options = []
            if self._match(TokenType.DISABLE):
                options.append("DISABLE")
                if self._match(TokenType.NOVALIDATE):
                    options.append("NOVALIDATE")
            
            return self.expression(
                exp.PrimaryKey,
                expressions=columns,
                options=options if options else None
            )





