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
            "IGNORE": TokenType.IGNORE,
        }

    class Parser(Postgres.Parser):
        # MERGE WHEN INSERT feature flags for GaussDB
        MERGE_INSERT_DEFAULT_VALUES_SUPPORTED = True
        MERGE_INSERT_WHERE_SUPPORTED = True
        MERGE_INSERT_OVERRIDING_SUPPORTED = False

        PROPERTY_PARSERS = {
            **Postgres.Parser.PROPERTY_PARSERS,
            "DISTRIBUTE BY": lambda self: self._parse_distributed_property(),
            "LOCAL": lambda self: (
                self._match_text_seq("TEMPORARY") or self._match_text_seq("TEMP")
            )
            and self.expression(exp.TemporaryProperty, this="LOCAL"),
            "PARTITION BY": lambda self: self._parse_partition_by_opt_range(),
            "PARTITIONED BY": lambda self: self._parse_partitioned_by(),
            "PARTITIONED_BY": lambda self: self._parse_partitioned_by(),
            "FOREIGN": lambda self: self.expression(exp.ExternalProperty),
            "SERVER": lambda self: self._parse_server_property(),
            "OPTIONS": lambda self: self._parse_options_property(),
            "LOG INTO": lambda self: self.expression(
                exp.WithJournalTableProperty, this=self._parse_table_parts()
            ),
            "PER NODE REJECT LIMIT": lambda self: self._parse_per_node_reject_limit_property(),
            "ENABLE": lambda self: self._parse_enable_row_movement_property(),
            "TO": lambda self: self._parse_to_group_or_node(),
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
            "TO": lambda self: self._parse_alter_table_to(),  # 新增TO解析器
            "OWNER": lambda self: self._parse_alter_owner(),
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

        def _parse_per_node_reject_limit_property(self) -> t.Optional[exp.Expression]:
            """解析 PER NODE REJECT LIMIT 属性。"""
            index = self._index
            value = self._parse_var_or_string() or self._parse_number()
            if value:
                has_rows = self._match_text_seq("ROWS")
                return self.expression(
                    exp.PerNodeRejectLimitProperty,
                    this=value,
                    rows=has_rows,
                )
            self._retreat(index)
            return self._parse_placeholder()

        def _parse_enable_row_movement_property(self) -> t.Optional[exp.Expression]:
            """解析 ENABLE ROW MOVEMENT 属性。"""
            index = self._index
            if self._match_text_seq("ROW", "MOVEMENT"):
                return self.expression(exp.EnableRowMovementProperty)
            self._retreat(index)
            return self._parse_placeholder()
        

        def _parse_options_property(self) -> t.List[exp.Expression]:
            """解析 OPTIONS (...) 子句，直接返回属性列表以复用现有属性节点。"""
            return [opt for opt in self._parse_wrapped_csv(self._parse_option_property) if opt]
        
        def _parse_option_property(self) -> t.Optional[exp.Expression]:
            """
            OPTIONS的通用属性解析入口。
            只解释``OPTION_PARSERS`` 中的参数，直接调用对应解析器。
            """
            if self._match_texts(self.OPTION_PARSERS):
                return self.OPTION_PARSERS[self._prev.text.upper()](self)
            return self._parse_placeholder()
        
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
                            exists=exists,
                            values=exp.var("VALUES"),
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
                            exists=exists,
                            values=exp.var("FOR_VALUES_IN"),
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
                                exp.AddGaussDBPartition,
                                this=partition_name,
                                exists=exists,
                                values=exp.var("FOR_VALUES_RANGE"),
                                range_from=self.expression(exp.Tuple, expressions=from_values),
                                range_to=self.expression(exp.Tuple, expressions=to_values),
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
            解析 PARTITION 子句，支持 GaussDB 的 PARTITION FOR 语法。
            
            支持的语法格式：
            - 标准语法：PARTITION(partition_name) 或 PARTITION(col=value)
            - GaussDB 语法：PARTITION FOR(value1, value2, ...)
            """
            # 解析 PARTITION / SUBPARTITION 子句
            if not self._match_texts(self.PARTITION_KEYWORDS):
                return None  # 未出现分区关键字则不进入该分支

            subpartition = self._prev.text.upper() == "SUBPARTITION"
            
            # 检查是否为 PARTITION FOR 语法
            if self._match_text_seq("FOR"):
                # 解析 PARTITION FOR(value1, value2, ...)
                values = self._parse_wrapped_csv(self._parse_expression)
                return self.expression(
                    exp.Partition,
                    subpartition=subpartition,
                    expressions=values,  # 直接将分区值作为 expressions
                )
            else:
                # 标准 PARTITION 语法
                wrapped = self._match(TokenType.L_PAREN, advance=False)
                return self.expression(
                    exp.Partition,
                    subpartition=subpartition,
                    expressions=self._parse_wrapped_csv(self._parse_assignment) if wrapped else self._parse_csv(self._parse_assignment),
                )     
                   
                   
        def _parse_to_group_or_node(self) -> t.Optional[exp.Expression]:
            """
            Parse TO GROUP groupname or TO NODE (nodename [, ...]) syntax.

            Supports:
            - TO GROUP groupname 
            - TO NODE (nodename1, nodename2, ...)
            """
            if self._match_text_seq("GROUP"):
                # Parse TO GROUP groupname
                group_name = self._parse_id_var()
                if group_name:
                    return self.expression(exp.ToGroupProperty, this=group_name)
            elif self._match_text_seq("NODE"):
                # Parse TO NODE (nodename [, ...])
                if self._match(TokenType.L_PAREN):
                    node_names = self._parse_csv(self._parse_id_var)
                    self._match(TokenType.R_PAREN)
                    return self.expression(exp.ToNodeProperty, expressions=node_names)
                else:
                    # Single node without parentheses
                    node_name = self._parse_id_var()
                    if node_name:
                        return self.expression(exp.ToNodeProperty, expressions=[node_name])


        def _parse_alter_table_to(self) -> t.Optional[exp.Expression]:
            """
            解析ALTER TABLE TO GROUP/NODE语法。
            
            支持语法：
            - ALTER TABLE table_name TO GROUP groupname
            - ALTER TABLE table_name TO NODE (nodename [, ...])
            - ALTER TABLE table_name TO NODE nodename  # 单节点简写
            
            Returns:
                AlterToGroup或AlterToNode表达式，解析失败返回None
            """
            if self._match_text_seq("GROUP"):
                # 解析 TO GROUP groupname
                group_name = self._parse_id_var()
                if group_name:
                    return self.expression(exp.AlterToGroup, this=group_name)
            elif self._match_text_seq("NODE"):
                # 解析 TO NODE (nodename [, ...]) 或 TO NODE nodename
                if self._match(TokenType.L_PAREN):
                    # 括号包裹的多节点语法
                    node_names = self._parse_csv(self._parse_id_var)
                    self._match(TokenType.R_PAREN)
                    return self.expression(exp.AlterToNode, expressions=node_names)
                else:
                    # 单节点简写语法
                    node_name = self._parse_id_var()
                    if node_name:
                        return self.expression(exp.AlterToNode, expressions=[node_name])
            
            # 未匹配到GROUP或NODE关键字
            self.raise_error("Expected GROUP or NODE after TO in ALTER TABLE statement")
            return None

        def _parse_alter_owner(self) -> exp.AlterOwner:
            """解析 ALTER ... OWNER TO 语法。"""
            if not self._match_text_seq("TO"):
                self.raise_error("Expected TO after OWNER")
            owner = self._parse_id_var(any_token=True)
            return self.expression(exp.AlterOwner, expression=owner)
        
        # 参考doris逻辑解析PARTITION BY RANGE/LIST的逻辑
        def _parse_partitioning_granularity_dynamic(self) -> exp.PartitionByRangePropertyDynamic:
            self._match_text_seq("START")
            start = self._parse_wrapped(self._parse_expression)
            self._match_text_seq("END")
            end = self._parse_wrapped(self._parse_expression)
            self._match_text_seq("EVERY")
            every = self._parse_wrapped(self._parse_expression)
            return self.expression(
                exp.PartitionByRangePropertyDynamic, start=start, end=end, every=every
            )

        def _parse_partition_definition(self) -> exp.Partition:
            self._match_text_seq("PARTITION")

            if self._match_text_seq("FOR"):
                values = self._parse_wrapped_csv(self._parse_expression)
                return self.expression(exp.Partition, expressions=values)

            name = self._parse_id_var()

            if self._match_text_seq("FOR"):
                values = self._parse_wrapped_csv(self._parse_expression)
                return self.expression(exp.Partition, expressions=values)

            self._match_text_seq("VALUES")

            if self._match_text_seq("LESS", "THAN"):
                values = self._parse_wrapped_csv(self._parse_expression)
                if len(values) == 1 and values[0].name.upper() == "MAXVALUE":
                    values = [exp.var("MAXVALUE")]

                part_range = self.expression(exp.PartitionRange, this=name, expressions=values)
                return self.expression(exp.Partition, expressions=[part_range])

            self._match(TokenType.L_BRACKET)
            values = self._parse_csv(lambda: self._parse_wrapped_csv(self._parse_expression))

            self._match(TokenType.R_BRACKET)
            self._match(TokenType.R_PAREN)

            part_range = self.expression(exp.PartitionRange, this=name, expressions=values)
            return self.expression(exp.Partition, expressions=[part_range])

        def _parse_partition_definition_list(self) -> exp.Partition:
            # PARTITION <name> VALUES IN (<value_csv>)
            self._match_text_seq("PARTITION")
            name = self._parse_id_var()
            self._match_text_seq("VALUES")
            self._match_text_seq("IN")
            values = self._parse_wrapped_csv(self._parse_expression)
            part_list = self.expression(exp.PartitionList, this=name, expressions=values)
            return self.expression(exp.Partition, expressions=[part_list])

        def _parse_partition_by_opt_range(
            self,
        ) -> exp.PartitionedByProperty | exp.PartitionByRangeProperty | exp.PartitionByListProperty:
            if self._match_text_seq("LIST"):
                return self.expression(
                    exp.PartitionByListProperty,
                    partition_expressions=self._parse_wrapped_id_vars(),
                    create_expressions=self._parse_wrapped_csv(
                        self._parse_partition_definition_list
                    ),
                )

            if not self._match_text_seq("RANGE"):
                return super()._parse_partitioned_by()

            partition_expressions = self._parse_wrapped_id_vars()
            self._match_l_paren()

            if self._match_text_seq("START", advance=False):
                create_expressions = self._parse_csv(self._parse_partitioning_granularity_dynamic)
            elif self._match_text_seq("PARTITION", advance=False):
                create_expressions = self._parse_csv(self._parse_partition_definition)
            else:
                create_expressions = None

            self._match_r_paren()

            return self.expression(
                exp.PartitionByRangeProperty,
                partition_expressions=partition_expressions,
                create_expressions=create_expressions,
            )

        def _parse_alter(self) -> exp.Alter | exp.Command:
            alter = super()._parse_alter()
            if isinstance(alter, exp.Alter):
                actions = alter.args.get("actions") or []
                if actions and not alter.args.get("expressions"):
                    alter.set("expressions", actions)

                table = alter.args.get("this")
                if table:
                    table_copy = table.copy()
                    for action in actions:
                        if isinstance(action, exp.AlterOwner) and not action.args.get("this"):
                            action.set("this", table_copy.copy())
            return alter

                
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

        PROPERTIES_LOCATION = {
            **Postgres.Generator.PROPERTIES_LOCATION,
            exp.TablespaceProperty: exp.Properties.Location.POST_NAME,
            exp.ServerProperty: exp.Properties.Location.POST_SCHEMA,
            exp.TableReadWriteProperty: exp.Properties.Location.POST_SCHEMA,
            exp.WithJournalTableProperty: exp.Properties.Location.POST_SCHEMA,
            exp.EnableRowMovementProperty: exp.Properties.Location.POST_SCHEMA,
            exp.PerNodeRejectLimitProperty: exp.Properties.Location.POST_SCHEMA,
        }

        TRANSFORMS = {
            **Postgres.Generator.TRANSFORMS,
            exp.ExternalProperty: lambda *_: "FOREIGN",
            exp.AlterToGroup: lambda self, e: self.altertogroup_sql(e),
            exp.AlterToNode: lambda self, e: self.altertonode_sql(e),
            exp.WithJournalTableProperty: lambda self, e: self.withjournaltableproperty_sql(e),
            exp.AddGaussDBPartition: lambda self, e: self.addgaussdbpartition_sql(e),
            exp.AlterOwner: lambda self, e: self.alterowner_sql(e),
        }

        def createable_sql(
            self, expression: exp.Create, locations: t.DefaultDict
        ) -> str:
            post_name = locations.pop(exp.Properties.Location.POST_NAME, None)
            this_sql = super().createable_sql(expression, locations)

            if post_name:
                props_sql = self.properties(
                    exp.Properties(expressions=post_name),
                    wrapped=False,
                    prefix=" ",
                )
                return f"{this_sql}{props_sql}"

            return this_sql

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


        def addgaussdbpartition_sql(self, expression: exp.AddGaussDBPartition) -> str:
            exists = " IF NOT EXISTS" if expression.args.get("exists") else ""
            name = self.sql(expression, "this")
            variant = expression.args.get("values")
            variant_name = variant.name if isinstance(variant, exp.Var) else None

            if variant_name == "FOR_VALUES_IN":
                values = self.expressions(expression, flat=True)
                values_sql = f" ({values})" if values else ""
                return f"ADD{exists} PARTITION {name} FOR VALUES IN{values_sql}"
            if variant_name == "FOR_VALUES_RANGE":
                from_sql = self.sql(expression, "range_from")
                to_sql = self.sql(expression, "range_to")
                to_clause = f" TO {to_sql}" if to_sql else ""
                return f"ADD{exists} PARTITION {name} FOR VALUES FROM {from_sql}{to_clause}"

            values = self.expressions(expression, flat=True)
            values_sql = f" ({values})" if values else ""
            if isinstance(variant, exp.Var):
                keyword = "VALUES" if variant.name == "VALUES" else "VALUES"
            else:
                keyword = "VALUES"
            return f"ADD{exists} PARTITION {name} {keyword}{values_sql}"

        def partitionrange_sql(self, expression: exp.PartitionRange) -> str:
            name = self.sql(expression, "this")
            name_sql = f"{name} " if name else ""
            if expression.expressions:
                values = self.expressions(expression, flat=True)
                return f"{name_sql}VALUES LESS THAN ({values})"
            high = self.sql(expression, "expression")
            if high:
                return f"{name_sql}TO {high}"
            return name_sql.strip()

        def partitionbylistproperty_sql(self, expression: exp.PartitionByListProperty) -> str:
            partition_expressions = self.expressions(expression, "partition_expressions")
            create_expressions = self.expressions(expression, "create_expressions")
            return f"PARTITION BY LIST {self.wrap(partition_expressions)} {self.wrap(create_expressions)}"

        def altertogroup_sql(self, expression: exp.AlterToGroup) -> str:
            return f"TO GROUP {self.sql(expression, 'this')}"

        def altertonode_sql(self, expression: exp.AlterToNode) -> str:
            nodes = self.expressions(expression, flat=True)
            if len(expression.expressions) > 1:
                return f"TO NODE ({nodes})"
            return f"TO NODE {nodes}"

        def alterowner_sql(self, expression: exp.AlterOwner) -> str:
            owner = self.sql(expression, "expression")
            return f"OWNER TO {owner}"

        def tablereadwriteproperty_sql(self, expression: exp.TableReadWriteProperty) -> str:
            return self.sql(expression, "this")

        def serverproperty_sql(self, expression: exp.ServerProperty) -> str:
            value = self.sql(expression, "this")
            return f"SERVER {value}"

        def withjournaltableproperty_sql(self, expression: exp.WithJournalTableProperty) -> str:
            table = self.sql(expression, "this")
            return f"LOG INTO {table}"

        def enablerowmovementproperty_sql(self, expression: exp.EnableRowMovementProperty) -> str:
            return "ENABLE ROW MOVEMENT"

        def pernoderejectlimitproperty_sql(self, expression: exp.PerNodeRejectLimitProperty) -> str:
            value = self.sql(expression, "this")
            suffix = " ROWS" if expression.args.get("rows") else ""
            return f"PER NODE REJECT LIMIT {value}{suffix}"

        def partitionlist_sql(self, expression: exp.PartitionList) -> str:
            name = self.sql(expression, "this")
            values = self.expressions(expression, "expressions")
            return f"PARTITION {name} VALUES IN {self.wrap(values)}"

        def partition_sql(self, expression: exp.Partition) -> str:
            partition_keyword = "SUBPARTITION" if expression.args.get("subpartition") else "PARTITION"
            expressions = expression.expressions or []

            if len(expressions) == 1:
                child = expressions[0]
                if isinstance(child, exp.Column) and not child.args.get("table"):
                    return f"{partition_keyword} {self.sql(child)}"
                if isinstance(child, exp.PartitionList):
                    return self.sql(child)
                if isinstance(child, exp.PartitionRange):
                    return f"{partition_keyword} {self.sql(child)}"
            if expressions and all(
                not isinstance(e, (exp.PartitionList, exp.PartitionRange, exp.Column))
                for e in expressions
            ):
                values = self.expressions(expression, flat=True)
                return f"{partition_keyword} FOR ({values})"

            return super().partition_sql(expression)
