from sqlglot import exp, tokens
from sqlglot.dialects.postgres import Postgres
from sqlglot.dialects.dialect import build_formatted_time
from sqlglot.generator import Generator
from sqlglot.tokens import TokenType
from sqlglot.helper import apply_index_offset, ensure_list, seq_get
import typing as t


class GaussDB(Postgres):  # 继承自 PostgreSQL 方言

    
    class Tokenizer(Postgres.Tokenizer):
        # 保留 PostgreSQL 的原始标识符和引号规则
        IDENTIFIERS = ['"']
        QUOTES = ["'"]

        # 扩展关键字映射
        KEYWORDS = {
            **Postgres.Tokenizer.KEYWORDS,
            "FLOAT8": TokenType.DOUBLE,
            "DOUBLE": TokenType.DOUBLE,
            "MINUS": TokenType.EXCEPT,
            "SERVER": TokenType.SERVER,
            "FOREIGN": TokenType.EXTERNAL,
            "LOG INTO": TokenType.LOG_INTO,
            "PER NODE REJECT LIMIT": TokenType.REJECT_LIMIT,
        }

    class Parser(Postgres.Parser):

        PROPERTY_PARSERS = {
            # "LOG INTO": lambda self: self.expression(exp.Property, this="LOG INTO", value=self._parse_id_var(any_token=True)),
            **Postgres.Parser.PROPERTY_PARSERS,
            "DISTRIBUTE BY": lambda self: self._parse_distributed_property(),
            "LOCAL": lambda self: (self._match_text_seq("TEMPORARY") or self._match_text_seq("TEMP"))
            and self.expression(exp.TemporaryProperty, this="LOCAL"),
            "PARTITION BY": lambda self: self._parse_partitioned_by_with_list(),
            "PARTITIONED BY": lambda self: self._parse_partitioned_by(),
            "PARTITIONED_BY": lambda self: self._parse_partitioned_by(),
            "FOREIGN": lambda self: self.expression(exp.ExternalProperty),
            "SERVER": lambda self: self._parse_server_property(),
            "OPTIONS": lambda self: self._parse_option_properties(),
            "LOG INTO": lambda self: self.expression(exp.WithJournalTableProperty, this=self._parse_table_parts()),
            "PER NODE REJECT LIMIT":lambda self: self.parse_kv_property(key="PER NODE REJECT LIMIT", quoted=True)
        }
        
        # OPTIONS中各类参数的解析器
        OPTION_PARSERS = {
            "LOCATION":lambda self: self._parse_property_assignment(exp.LocationProperty),
            "FORMAT": lambda self: self._parse_property_assignment(exp.FileFormatProperty),
            "HEADER":lambda self: self.parse_kv_property(key="HEADER", quoted=True),
            "FILEHEADER":lambda self: self.parse_kv_property(key="FILEHEADER", quoted=True),
            "OUT_FILENAME_PREFIX":lambda self: self.parse_kv_property(key="OUT_FILENAME_PREFIX", quoted=True),
            "DELIMITER":lambda self: self.parse_kv_property(key="DELIMITER", quoted=True),
            "QUOTE":lambda self: self.parse_kv_property(key="QUOTE", quoted=True),
            "ESCAPE":lambda self: self.parse_kv_property(key="ESCAPE", quoted=True),
            "NULL":lambda self: self.parse_kv_property(key="NULL", quoted=True),
            "BLANK_NUMBER_STR_TO_NUL":lambda self: self.parse_kv_property(key="BLANK_NUMBER_STR_TO_NUL", quoted=True),
            "NOESCAPING":lambda self: self.parse_kv_property(key="NOESCAPING", quoted=True),
            "ENCODING":lambda self: self.parse_kv_property(key="ENCODING", quoted=True),
            "DATAENCODING":lambda self: self.parse_kv_property(key="DATAENCODING", quoted=True),
            "MODE":lambda self: self.parse_kv_property(key="MODE", quoted=True),
            "EOL":lambda self: self.parse_kv_property(key="EOL", quoted=True),
            "CONFLICT_DELIMITER":lambda self: self.parse_kv_property(key="CONFLICT_DELIMITER", quoted=True),
            "FILE_TYPE":lambda self: self.parse_kv_property(key="FILE_TYPE", quoted=True),
            "AUTO_CREATE_PIPE":lambda self: self.parse_kv_property(key="AUTO_CREATE_PIPE", quoted=True),
            "DEL_PIPE":lambda self: self.parse_kv_property(key="DEL_PIPE", quoted=True),
            "GDS_COMPRESS":lambda self: self.parse_kv_property(key="GDS_COMPRESS", quoted=True),
            "PRESERVE_BLANKS":lambda self: self.parse_kv_property(key="PRESERVE_BLANKS", quoted=True),
            "FIX":lambda self: self.parse_kv_property(key="FIX", quoted=True),
            "OUT_FIX_ALIGNMENT":lambda self: self.parse_kv_property(key="OUT_FIX_ALIGNMENT", quoted=True),
            "OUT_FIX_NUM_ALIGNMENT":lambda self: self.parse_kv_property(key="OUT_FIX_NUM_ALIGNMENT", quoted=True),
            "DATE_FORMAT":lambda self: self.parse_kv_property(key="DATE_FORMAT", quoted=True),
            "TIME_FORMAT":lambda self: self.parse_kv_property(key="TIME_FORMAT", quoted=True),
            "TIMESTAMP_FORMAT":lambda self: self.parse_kv_property(key="TIMESTAMP_FORMAT", quoted=True),
            "SMALLDATETIME_FORMAT":lambda self: self.parse_kv_property(key="SMALLDATETIME_FORMAT", quoted=True),
            "FILL_MISSING_FIELDS":lambda self: self.parse_kv_property(key="FILL_MISSING_FIELDS", quoted=True),
            "IGNORE_EXTRA_DATA":lambda self: self.parse_kv_property(key="IGNORE_EXTRA_DATA", quoted=True),
            "REJECT_LIMIT":lambda self: self.parse_kv_property(key="PER NODE REJECT LIMIT", quoted=True),
            "COMPATIBLE_ILLEGAL_CHARS":lambda self: self.parse_kv_property(key="COMPATIBLE_ILLEGAL_CHARS", quoted=True),
            "REPLACE_ILLEGAL_CHARS":lambda self: self.parse_kv_property(key="REPLACE_ILLEGAL_CHARS", quoted=True),
            "WITH ERROR_TABLE_NAME":lambda self: self.parse_kv_property(key="WITH ERROR_TABLE_NAME", quoted=True),
            "LOG INTO ERROR_TABLE_NAME":lambda self: self.parse_kv_property(key="LOG INTO ERROR_TABLE_NAME", quoted=True),
            "REMOTE LOG":lambda self: self.parse_kv_property(key="REMOTE LOG", quoted=True),
            "PER NODE REJECT LIMIT":lambda self: self.parse_kv_property(key="PER NODE REJECT LIMIT", quoted=True),
            "FILE_SEQUENCE":lambda self: self.parse_kv_property(key="FILE_SEQUENCE", quoted=True),
        }
        
        # 解析OPTIONS中的参数取值，包括各类编码的字符串和数字
        OPTION_VALUE_PARSERS = {
            **Postgres.Parser.STRING_PARSERS,
            **Postgres.Parser.NUMERIC_PARSERS,
        }
            
        
        ALTER_PARSERS = {
            **Postgres.Parser.ALTER_PARSERS,
            "ADD": lambda self: self._parse_alter_table_add(),
        }
        
        FUNCTIONS = {
            **Postgres.Parser.FUNCTIONS,
            "TO_CHAR": build_formatted_time(exp.TimeToStr, "postgres",default = True),
            "TO_DATE": build_formatted_time(exp.StrToDate, "postgres",default = True),
        }
        FUNC_TOKENS = {
            *Postgres.Parser.FUNC_TOKENS,
            TokenType.VALUES,
        }
        
        
        def parse_kv_property(self, key: str, quoted: True) -> exp.Property:
            """解析形如 `KEY "VALUE"` 的K-V属性。"""
            if self._match_set(self.OPTION_VALUE_PARSERS):
                value = self.OPTION_VALUE_PARSERS[self._prev.token_type](self, self._prev)
                return self.expression(exp.Property, this=key, value=value)
            return self._parse_placeholder()
        
        
        def  _parse_option_properties(self) -> t.List[exp.Expression]:
            """解析OPTIONS形如 `(KEY1 "VALUE1", KEY2 "VALUE2", ...)` 的 *括号包裹* 属性列表。"""
            return self._parse_wrapped_csv(self._parse_option_property)
        
        def      _parse_option_property(self) -> t.Optional[exp.Expression]:
            """
            OPTIONS的通用属性解析入口。
            只解释``OPTION_PARSERS`` 中的参数，直接调用对应解析器。
            """
            if self._match_texts(self.OPTION_PARSERS):
                return self.OPTION_PARSERS[self._prev.text.upper()](self)
        
        def _parse_alter_table_add(self) -> t.List[exp.Expression]:
            """
            解析ALTER TABLE ADD 语法，支持分区、约束、列定义等。
            """
            def _parse_add_alteration() -> t.Optional[exp.Expression]:
                # 消费 ADD 关键字，随后分支解析具体对象
                self._match_text_seq("ADD")
                # 优先解析约束（避免与列定义产生歧义）
                if self._match_set(self.ADD_CONSTRAINT_TOKENS, advance=False):
                    return self.expression(
                        exp.AddConstraint, expressions=self._parse_csv(self._parse_constraint)
                    )

                # 解析 IF [NOT] EXISTS，用于分区添加
                exists = self._parse_exists(not_=True)
                
                # 新增：PostgreSQL分区语法支持
                if self._match(TokenType.PARTITION):
                    partition_name = self._parse_id_var()
                    
                    # 解析 VALUES 子句
                    if self._match_text_seq("VALUES"):
                        self._match(TokenType.L_PAREN)
                        values = self._parse_csv(self._parse_expression)
                        self._match(TokenType.R_PAREN)
                                                
                        return self.expression(
                            exp.AddGaussDBPartition,
                            this=partition_name,
                            expressions=values,
                            exists=exists
                        )
                    
                    # 解析 FOR VALUES IN 子句
                    elif self._match_text_seq("FOR", "VALUES", "IN"):
                        self._match(TokenType.L_PAREN)
                        values = self._parse_csv(self._parse_expression)
                        self._match(TokenType.R_PAREN)
                        
                        return self.expression(
                            exp.AddGaussDBPartition,
                            this=partition_name,
                            expressions=values,
                            exists=exists
                        )
                    
                    # 解析 FOR VALUES FROM ... TO 子句
                    elif self._match_text_seq("FOR", "VALUES", "FROM"):
                        self._match(TokenType.L_PAREN)
                        from_values = self._parse_csv(self._parse_expression)
                        self._match(TokenType.R_PAREN)
                        
                        if self._match_text_seq("TO"):
                            self._match(TokenType.L_PAREN)
                            to_values = self._parse_csv(self._parse_expression)
                            self._match(TokenType.R_PAREN)
                            
                            return self.expression(
                                exp.AddPartition,
                                this=partition_name,
                                expressions=from_values + to_values,  # 合并范围值
                                exists=exists
                            )

                # Hive/Athena 风格：ADD [IF NOT EXISTS] PARTITION (...) [LOCATION '...']
                if self._match_pair(TokenType.PARTITION, TokenType.L_PAREN, advance=False):
                    return self.expression(
                        exp.AddPartition,
                        exists=exists,
                        this=self._parse_field(any_token=True),
                        # 可选 LOCATION 属性，指定分区外部路径/存储位置
                        location=self._match_text_seq("LOCATION", advance=False)
                        and self._parse_property(),
                    )
                    
                # 其次尝试解析列定义（支持 [NOT] EXISTS）
                column_def = self._parse_add_column()
                if isinstance(column_def, exp.ColumnDef):
                    return column_def
                # 未命中任何 ADD 子分支：返回 None 交由上层处理
                return None

            # 处理多列添加的方言差异：
            # - 某些方言要求每列前都写 ADD（ALTER_TABLE_ADD_REQUIRED_FOR_EACH_COLUMN=True）
            # - 另一些支持 "ADD COLUMNS (...)" 或 "ADD (...)" 一次性添加多列
            if not self._match_set(self.ADD_CONSTRAINT_TOKENS, advance=False) and (
                not self.dialect.ALTER_TABLE_ADD_REQUIRED_FOR_EACH_COLUMN
                or self._match_text_seq("COLUMNS")
            ):
                # 若方言支持整体 schema 形式，优先解析 schema；否则回退到列定义 CSV
                schema = self._parse_schema()

                return (
                    # schema 可能为单个或列表：统一转为列表，便于上层处理
                    ensure_list(schema)
                    if schema
                    else self._parse_csv(self._parse_column_def_with_exists)
                )

            # 需要对每个项分别解析（列/约束/分区），以逗号分隔
            return self._parse_csv(_parse_add_alteration)
        
        def _parse_distributed_property(self) -> exp.DistributedByProperty:
            """
            解析DISTRIBUTED 语法，支持HASH、RANDOM、BUCKETS等。
            """
            kind = "HASH"
            expressions: t.Optional[t.List[exp.Expression]] = None
            if self._match_text_seq("BY", "HASH"):
                expressions = self._parse_wrapped_csv(self._parse_id_var)
            elif self._match_text_seq("HASH"):
                expressions = self._parse_wrapped_csv(self._parse_id_var)            
            elif self._match_text_seq("BY", "RANDOM"):
                kind = "RANDOM"

            # If the BUCKETS keyword is not present, the number of buckets is AUTO
            buckets: t.Optional[exp.Expression] = None
            if self._match_text_seq("BUCKETS") and not self._match_text_seq("AUTO"):
                buckets = self._parse_number()

            return self.expression(
                exp.DistributedByProperty,
                expressions=expressions,
                kind=kind,
                buckets=buckets,
                order=self._parse_order(),
            )
            
        def _parse_partition(self) -> t.Optional[exp.Partition]:
            """
            解析PARTITION PARTITION / SUBPARTITION 子句, 支持括号包裹的PARTITION / SUBPARTITION 子句。
            """
            # 解析 PARTITION / SUBPARTITION 子句
            if not self._match_texts(self.PARTITION_KEYWORDS):
                return None  # 未出现分区关键字则不进入该分支

            wrapped = self._match(TokenType.L_PAREN)
            return self.expression(
                exp.Partition,
                subpartition=self._prev.text.upper() == "SUBPARTITION",  # 区分是否为二级分区
                expressions=self._parse_wrapped_csv(self._parse_assignment) if wrapped else self._parse_csv(self._parse_assignment),  # 括号包裹的分区表达式列表
            )
            
            
        def _parse_partition_list(self) -> t.List[exp.Expression]:
            """解析分区列表，如 (PARTITION p1 VALUES ('val1'), PARTITION p2 VALUES ('val2'))"""
            partitions = []
            while True:
                if not self._match(TokenType.PARTITION):
                    break
                
                partition_name = self._parse_id_var()
                
                if self._match_text_seq("VALUES"):
                    self._match(TokenType.L_PAREN)
                    values = self._parse_csv(self._parse_expression)
                    self._match(TokenType.R_PAREN)
                    
                    partitions.append(self.expression(
                        exp.PartitionBoundSpec,
                        this=values
                    ))
                
                if not self._match(TokenType.COMMA):
                    break
            
            return partitions

        def _parse_partitioned_by_with_list(self) -> exp.PartitionedByProperty:
            """解析PARTITION BY语法，支持后续的分区列表"""
            # 先解析PARTITION BY部分
            partition_by = self._parse_partitioned_by()
            
            # 检查是否有后续的分区列表
            if self._match(TokenType.L_PAREN):
                partition_list = self._parse_partition_list()
                self._match(TokenType.R_PAREN)
                
                # 将分区列表添加到属性中
                partition_by.set("partition_list", partition_list)
            
            return partition_by        
                   
                
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


        def partitionlistproperty_sql(self, expression: exp.PartitionListProperty) -> str:
            """生成分区列表的SQL"""
            partition_by = self.sql(expression, "this")
            partition_list = self.expressions(expression, key="partition_list", flat=True)
            return f"PARTITION BY {partition_by} ({partition_list})"
