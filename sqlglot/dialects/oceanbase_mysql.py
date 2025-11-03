from __future__ import annotations

import typing as t

from sqlglot import exp, tokens
from sqlglot.dialects.mysql import MySQL
from sqlglot.generator import Generator
from sqlglot.tokens import Tokenizer, TokenType


class OceanBase_MySQL(MySQL):  # 继承自 MySQL 方言
    class Tokenizer(MySQL.Tokenizer):
        # 扩展关键字映射
        KEYWORDS = {
            **MySQL.Tokenizer.KEYWORDS,
        }

    class Parser(MySQL.Parser):
        def _parse_primary_key(
            self, wrapped_optional: bool = False, in_props: bool = False
        ) -> exp.PrimaryKeyColumnConstraint | exp.PrimaryKey:
            """解析 PRIMARY KEY，支持 OceanBase 的 USING BTREE 语法"""
            desc = (
                self._match_set((TokenType.ASC, TokenType.DESC))
                and self._prev.token_type == TokenType.DESC
            )

            # 检查是否有 USING 子句（OceanBase 特有语法）
            index_type = None
            if self._match(TokenType.USING):
                index_type = self._advance_any() and self._prev.text

            # 解析其他键约束选项
            options = self._parse_key_constraint_options()
            
            # 如果有 index_type，将其添加到选项中
            if index_type:
                options.append(exp.IndexConstraintOption(using=index_type))

            if not in_props and not self._match(TokenType.L_PAREN, advance=False):
                return self.expression(
                    exp.PrimaryKeyColumnConstraint,
                    desc=desc,
                    options=options,
                )

            expressions = self._parse_wrapped_csv(
                self._parse_primary_key_part, optional=wrapped_optional
            )

            return self.expression(
                exp.PrimaryKey,
                expressions=expressions,
                include=self._parse_index_params(),
                options=options,
            )

        def _parse_index_constraint(
            self, kind: t.Optional[str] = None
        ) -> exp.IndexColumnConstraint:
            if kind:
                self._match_texts(("INDEX", "KEY"))

            this = self._parse_id_var(any_token=False)
            index_type = self._match(TokenType.USING) and self._advance_any() and self._prev.text
            expressions = self._parse_wrapped_csv(self._parse_ordered)

            options = []
            while True:
                # GLOBAL | LOCAL - 全局或本地索引选项
                if self._match_text_seq("GLOBAL"):
                    opt = exp.IndexConstraintOption(global_index=True)
                elif self._match_text_seq("LOCAL"):
                    opt = exp.IndexConstraintOption(local_index=True)
                # block_size - 块大小选项
                elif self._match_text_seq("BLOCK_SIZE"):  # OceanBase uses BLOCK_SIZE instead of KEY_BLOCK_SIZE
                    self._match(TokenType.EQ)
                    block_size = self._parse_number()
                    opt = exp.IndexConstraintOption(key_block_size=block_size)
                elif self._match_text_seq("KEY_BLOCK_SIZE"):  # Also support MySQL's KEY_BLOCK_SIZE for compatibility
                    self._match(TokenType.EQ)
                    block_size = self._parse_number()
                    opt = exp.IndexConstraintOption(key_block_size=block_size)
                # compression - 压缩选项
                elif self._match_text_seq("COMPRESSION"):
                    self._match(TokenType.EQ)
                    compression = self._parse_string() or self._parse_var()
                    opt = exp.IndexConstraintOption(compression=compression)
                # STORING(column_name_list) - 存储列选项
                elif self._match_text_seq("STORING"):
                    self._match(TokenType.L_PAREN)
                    storing_columns = self._parse_csv(self._parse_id_var)
                    self._match(TokenType.R_PAREN)
                    opt = exp.IndexConstraintOption(storing=storing_columns)
                # comment - 注释选项
                elif self._match(TokenType.COMMENT):
                    opt = exp.IndexConstraintOption(comment=self._parse_string())
                # 其他兼容性选项
                elif self._match(TokenType.USING):
                    opt = exp.IndexConstraintOption(using=self._advance_any() and self._prev.text)
                elif self._match_text_seq("WITH", "PARSER"):
                    opt = exp.IndexConstraintOption(parser=self._parse_var(any_token=True))
                elif self._match_text_seq("VISIBLE"):
                    opt = exp.IndexConstraintOption(visible=True)
                elif self._match_text_seq("INVISIBLE"):
                    opt = exp.IndexConstraintOption(visible=False)
                elif self._match_text_seq("ENGINE_ATTRIBUTE"):
                    self._match(TokenType.EQ)
                    opt = exp.IndexConstraintOption(engine_attr=self._parse_string())
                elif self._match_text_seq("SECONDARY_ENGINE_ATTRIBUTE"):
                    self._match(TokenType.EQ)
                    opt = exp.IndexConstraintOption(secondary_engine_attr=self._parse_string())
                else:
                    opt = None

                if not opt:
                    break

                options.append(opt)

            return self.expression(
                exp.IndexColumnConstraint,
                this=this,
                expressions=expressions,
                kind=kind,
                index_type=index_type,
                options=options,
            )

    class Generator(MySQL.Generator):
        # 覆盖类型映射
        TYPE_MAPPING = {
            **MySQL.Generator.TYPE_MAPPING,  # 继承原有映射
            # exp.DataType.Type.TIMESTAMP: "TIMESTAMP",

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

        def primarykey_sql(self, expression: exp.PrimaryKey) -> str:
            """生成 PRIMARY KEY 约束的 SQL，支持 USING 子句"""
            expressions = self.expressions(expression, "expressions")
            
            sql = f"PRIMARY KEY ({expressions})"
            
            # 从 options 中查找 USING 子句
            options = expression.args.get("options", [])
            for option in options:
                if isinstance(option, exp.IndexConstraintOption):
                    using = option.args.get("using")
                    if using:
                        sql += f" USING {using}"
                        break
            
            # 生成其他选项
            other_options = []
            for option in options:
                if isinstance(option, exp.IndexConstraintOption):
                    using = option.args.get("using")
                    if not using:  # 跳过 USING 选项，已经处理过了
                        option_sql = self.indexconstraintoption_sql(option)
                        if option_sql:
                            other_options.append(option_sql)
            
            if other_options:
                sql += f" {' '.join(other_options)}"
                
            return sql

        def primarykeycolumnconstraint_sql(self, expression: exp.PrimaryKeyColumnConstraint) -> str:
            """生成列级 PRIMARY KEY 约束的 SQL，支持 USING 子句"""
            sql = "PRIMARY KEY"
            
            # 从 options 中查找 USING 子句
            options = expression.args.get("options", [])
            for option in options:
                if isinstance(option, exp.IndexConstraintOption):
                    using = option.args.get("using")
                    if using:
                        sql += f" USING {using}"
                        break
            
            # 生成其他选项
            other_options = []
            for option in options:
                if isinstance(option, exp.IndexConstraintOption):
                    using = option.args.get("using")
                    if not using:  # 跳过 USING 选项，已经处理过了
                        option_sql = self.indexconstraintoption_sql(option)
                        if option_sql:
                            other_options.append(option_sql)
            
            if other_options:
                sql += f" {' '.join(other_options)}"
                
            return sql

        def indexconstraintoption_sql(self, expression: exp.IndexConstraintOption) -> str:
            # GLOBAL | LOCAL - 全局或本地索引选项
            global_index = expression.args.get("global_index")
            if global_index:
                return "GLOBAL"
            
            local_index = expression.args.get("local_index")
            if local_index:
                return "LOCAL"
            
            # block_size - 块大小选项
            key_block_size = self.sql(expression, "key_block_size")
            if key_block_size:
                return f"BLOCK_SIZE = {key_block_size}"
            
            # compression - 压缩选项
            compression = self.sql(expression, "compression")
            if compression:
                return f"COMPRESSION = {compression}"
            
            # STORING(column_name_list) - 存储列选项
            storing = expression.args.get("storing")
            if storing:
                columns = self.expressions(expression, "storing")
                return f"STORING({columns})"
            
            # comment - 注释选项
            comment = self.sql(expression, "comment")
            if comment:
                return f"COMMENT {comment}"
            
            # 其他兼容性选项
            using = self.sql(expression, "using")
            if using:
                return f"USING {using}"

            parser = self.sql(expression, "parser")
            if parser:
                return f"WITH PARSER {parser}"

            visible = expression.args.get("visible")
            if visible is not None:
                return "VISIBLE" if visible else "INVISIBLE"

            engine_attr = self.sql(expression, "engine_attr")
            if engine_attr:
                return f"ENGINE_ATTRIBUTE = {engine_attr}"

            secondary_engine_attr = self.sql(expression, "secondary_engine_attr")
            if secondary_engine_attr:
                return f"SECONDARY_ENGINE_ATTRIBUTE = {secondary_engine_attr}"

            return ""



