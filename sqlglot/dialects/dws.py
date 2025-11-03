from sqlglot import exp, tokens
from sqlglot.dialects.postgres import Postgres
from sqlglot.generator import Generator
from sqlglot.tokens import Tokenizer, TokenType
import typing as t


class DWS(Postgres):  # 继承自 Postgres 方言
    class Tokenizer(Postgres.Tokenizer):
        # 保留 MySQL 的原始标识符和引号规则
        IDENTIFIERS = ['"']
        QUOTES = ["'"]

        # 扩展关键字映射
        KEYWORDS = {
            **Postgres.Tokenizer.KEYWORDS,
            "FLOAT8": TokenType.DOUBLE,
            "LIST": TokenType.LIST,
        }

    class Parser(Postgres.Parser):
        PROPERTY_PARSERS = {
            **Postgres.Parser.PROPERTY_PARSERS,
            "DISTRIBUTE BY": lambda self: self._parse_distributed_property(),
            "PARTITION BY": lambda self: self._parse_partitioned_by_with_list(),
            "LOCAL": lambda self: self._parse_local_property(),
        }

        STATEMENT_PARSERS = {
            **Postgres.Parser.STATEMENT_PARSERS,
            TokenType.CREATE: lambda self: self._parse_create(),
        }

        def _parse_distributed_property(self) -> exp.DistributedByProperty:
            """
            解析DISTRIBUTE BY HASH语法，支持DWS的分布语法。
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

        def _parse_partition_list(self) -> t.List[exp.Expression]:
            """解析分区列表，支持 VALUES ('value') 和 VALUES LESS THAN (value)"""
            partitions = []
            while True:
                if not self._match(TokenType.PARTITION):
                    break
                
                partition_name = self._parse_id_var()
                
                if self._match_text_seq("VALUES"):
                    if self._match_text_seq("LESS", "THAN"):
                        # VALUES LESS THAN (value) - RANGE 分区
                        self._match(TokenType.L_PAREN)
                        values = self._parse_csv(self._parse_expression)
                        self._match(TokenType.R_PAREN)
                        
                        # 创建分区范围表达式，标记为 LESS THAN 类型
                        partition_def = self.expression(
                            exp.PartitionRange,
                            this=partition_name,
                            expressions=values
                        )
                        # 添加元数据来标记这是 LESS THAN 类型
                        partition_def.set("is_less_than", True)
                        partitions.append(partition_def)
                    else:
                        # VALUES ('value') - LIST 分区
                        self._match(TokenType.L_PAREN)
                        values = self._parse_csv(self._parse_expression)
                        self._match(TokenType.R_PAREN)
                        
                        # 创建分区范围表达式
                        partition_def = self.expression(
                            exp.PartitionRange,
                            this=partition_name,
                            expressions=values
                        )
                        partitions.append(partition_def)
                
                if not self._match(TokenType.COMMA):
                    break
            
            return partitions

        def _parse_index_partition_list(self) -> t.List[exp.Expression]:
            """解析索引分区列表，支持PARTITION和SUBPARTITION以及TABLESPACE"""
            partitions = []
            while True:
                if self._match_text_seq("PARTITION"):
                    partition_name = self._parse_id_var()
                    partition_expr = self.expression(exp.Identifier, this=partition_name)
                    
                    # 检查是否有TABLESPACE
                    if self._match_text_seq("TABLESPACE"):
                        tablespace = self._parse_id_var()
                        partition_expr.set("tablespace", tablespace)
                    
                    partitions.append(partition_expr)
                elif self._match_text_seq("SUBPARTITION"):
                    subpartition_name = self._parse_id_var()
                    subpartition_expr = self.expression(exp.Identifier, this=f"SUBPARTITION {subpartition_name}")
                    
                    # 检查是否有TABLESPACE
                    if self._match_text_seq("TABLESPACE"):
                        tablespace = self._parse_id_var()
                        subpartition_expr.set("tablespace", tablespace)
                    
                    partitions.append(subpartition_expr)
                else:
                    break
                
                if not self._match(TokenType.COMMA):
                    break
            
            return partitions

        def _parse_partitioned_by_with_list(self) -> exp.PartitionedByProperty:
            """解析PARTITION BY语法，支持后续的分区列表"""
            # 解析PARTITION BY后面的内容
            if self._match(TokenType.LIST):
                partition_type = "LIST"
            elif self._match(TokenType.RANGE):
                partition_type = "RANGE"
            else:
                partition_type = self._parse_id_var()
            
            # 解析分区列
            self._match(TokenType.L_PAREN)
            partition_columns = self._parse_csv(self._parse_id_var)
            self._match(TokenType.R_PAREN)
            
            # 创建分区表达式，包含分区类型
            partition_expr = self.expression(
                exp.Partition,
                expressions=partition_columns
            )
            # 将分区类型存储到分区表达式中
            partition_expr.set("partition_type", partition_type)
            
            # 检查是否有后续的分区列表（在分区列括号之后）
            # 注意：这里需要检查是否还有额外的左括号（分区列表的括号）
            if self._match(TokenType.L_PAREN, advance=False):
                self._match(TokenType.L_PAREN)  # 消费左括号
                partition_list = self._parse_partition_list()
                self._match(TokenType.R_PAREN)  # 消费右括号
                
                # 使用PartitionListProperty来存储分区列表
                partition_by = self.expression(
                    exp.PartitionListProperty,
                    this=partition_expr,
                    partition_list=partition_list
                )
            else:
                # 使用普通的PartitionedByProperty
                partition_by = self.expression(
                    exp.PartitionedByProperty,
                    this=partition_expr
                )
            
            return partition_by

        def _parse_index_column(self) -> exp.Expression:
            """解析索引列，支持长度、COLLATE、opclass等选项"""
            # 解析列名或表达式
            if self._match(TokenType.L_PAREN):
                # 表达式索引 (expression)
                expr = self._parse_expression()
                self._match(TokenType.R_PAREN)
                column_expr = expr
            else:
                # 简单列名
                column_name = self._parse_id_var()
                column_expr = self.expression(exp.Column, this=column_name)
                
                # 解析长度 (length)
                if self._match(TokenType.L_PAREN):
                    length = self._parse_expression()
                    self._match(TokenType.R_PAREN)
                    # 将长度信息存储到列表达式中
                    column_expr.set("length", length)
                
                # 解析opclass（使用内置的opclass解析方法）
                opclass_expr = self._parse_opclass()
                if opclass_expr:
                    column_expr.set("opclass", opclass_expr)
            
            # 解析COLLATE
            if self._match_text_seq("COLLATE"):
                collation = self._parse_id_var()
                column_expr.set("collation", collation)
            
            # 解析ASC/DESC
            if self._match_text_seq("ASC"):
                column_expr.set("ascending", True)
            elif self._match_text_seq("DESC"):
                column_expr.set("ascending", False)
            
            # 解析NULLS FIRST/LAST
            if self._match_text_seq("NULLS"):
                if self._match_text_seq("FIRST"):
                    column_expr.set("nulls_first", True)
                elif self._match_text_seq("LAST"):
                    column_expr.set("nulls_first", False)
            
            return column_expr

        def _parse_local_property(self) -> exp.Property:
            """解析LOCAL属性，支持分区列表"""
            # 检查是否有分区列表
            if self._match(TokenType.L_PAREN):
                partition_list = self._parse_partition_list()
                self._match(TokenType.R_PAREN)
                
                return self.expression(
                    exp.Property,
                    this="LOCAL",
                    value=partition_list if partition_list else []
                )
            else:
                return self.expression(exp.Property, this="LOCAL", value=[])

        def _parse_create(self) -> exp.Create:
            """重写CREATE语句解析，支持DWS的索引创建语法"""
            # 检查是否是CREATE INDEX
            if self._match_text_seq("UNIQUE"):
                unique = True
            else:
                unique = False
                
            if self._match_text_seq("INDEX"):
                return self._parse_create_index(unique=unique)
            else:
                # 回退到父类的CREATE解析
                return super()._parse_create()

        def _parse_create_index(self, unique: bool = False) -> exp.Create:
            """解析CREATE INDEX语句，支持DWS的分区索引语法"""
            # 解析CONCURRENTLY关键字
            concurrently = self._match_text_seq("CONCURRENTLY")
            
            # 解析索引名称（可能包含schema）
            this = self._parse_id_var()
            # 处理带schema的索引名
            if self._match(TokenType.DOT):
                schema_name = this
                this = self._parse_id_var()
                # 存储完整的索引名
                full_index_name = f"{schema_name}.{this}"
            
            # 解析ON table_name (可能包含schema)
            self._match_text_seq("ON")
            # 解析表名，可能包含schema
            table_name = self._parse_id_var()
            if self._match(TokenType.DOT):
                # 处理 schema.table 格式
                schema_name = table_name
                table_name = self._parse_id_var()
                table = self.expression(exp.Table, this=table_name, db=schema_name)
            else:
                # 处理简单表名
                table = self.expression(exp.Table, this=table_name)
            
            # 解析USING method (可选)
            using = None
            if self._match_text_seq("USING"):
                using = self._parse_id_var()
            
            # 解析索引列
            self._match(TokenType.L_PAREN)
            expressions = self._parse_csv(self._parse_index_column)
            self._match(TokenType.R_PAREN)
            
            # 解析索引选项
            properties = []
            
            # LOCAL/GLOBAL选项
            if self._match_text_seq("LOCAL"):
                # 解析LOCAL选项，可能包含分区列表
                if self._match(TokenType.L_PAREN):
                    # 解析索引分区列表（支持TABLESPACE）
                    partition_list = self._parse_index_partition_list()
                    self._match(TokenType.R_PAREN)
                    # 将分区列表包装成Array表达式
                    partition_array = self.expression(exp.Array, expressions=partition_list)
                    properties.append(self.expression(exp.Property, this="LOCAL", value=partition_array))
                else:
                    properties.append(self.expression(exp.Property, this="LOCAL", value=self.expression(exp.Array, expressions=[])))
            elif self._match_text_seq("GLOBAL"):
                properties.append(self.expression(exp.Property, this="GLOBAL", value=self.expression(exp.Array, expressions=[])))
            
            # INCLUDE选项
            if self._match_text_seq("INCLUDE"):
                self._match(TokenType.L_PAREN)
                include_columns = self._parse_csv(self._parse_id_var)
                self._match(TokenType.R_PAREN)
                # 将包含列包装成Array表达式
                include_array = self.expression(exp.Array, expressions=include_columns)
                properties.append(self.expression(exp.Property, this="INCLUDE", value=include_array))
            
            # WITH选项
            if self._match_text_seq("WITH"):
                self._match(TokenType.L_PAREN)
                storage_params = self._parse_csv(self._parse_assignment)
                self._match(TokenType.R_PAREN)
                # 将存储参数包装成Array表达式
                with_array = self.expression(exp.Array, expressions=storage_params)
                properties.append(self.expression(exp.Property, this="WITH", value=with_array))
            
            # TABLESPACE选项
            if self._match_text_seq("TABLESPACE"):
                tablespace = self._parse_id_var()
                # 将表空间包装成Array表达式
                tablespace_array = self.expression(exp.Array, expressions=[tablespace])
                properties.append(self.expression(exp.Property, this="TABLESPACE", value=tablespace_array))
            
            # COMMENT选项
            if self._match_text_seq("COMMENT"):
                comment = self._parse_string()
                # 将注释包装成Array表达式
                comment_array = self.expression(exp.Array, expressions=[comment])
                properties.append(self.expression(exp.Property, this="COMMENT", value=comment_array))
            
            # VISIBLE/INVISIBLE选项
            if self._match_text_seq("VISIBLE"):
                properties.append(self.expression(exp.Property, this="VISIBLE", value=None))
            elif self._match_text_seq("INVISIBLE"):
                properties.append(self.expression(exp.Property, this="INVISIBLE", value=None))
            
            # 将表名和列信息存储到properties中
            if table:
                properties.append(self.expression(exp.Property, this="TABLE", value=table))
            if expressions:
                # 将列表达式列表包装成Array表达式
                columns_array = self.expression(exp.Array, expressions=expressions)
                properties.append(self.expression(exp.Property, this="COLUMNS", value=columns_array))
            
            # 将properties包装成Expression对象
            if properties:
                properties_expr = self.expression(exp.Properties, expressions=properties)
            else:
                properties_expr = None
            
            return self.expression(
                exp.Create,
                this=this,
                kind="INDEX",
                unique=unique,
                concurrently=concurrently,
                properties=properties_expr
            )

    class Generator(Postgres.Generator):
        # 覆盖类型映射
        TYPE_MAPPING = {
            **Postgres.Generator.TYPE_MAPPING,  # 继承原有映射
            exp.DataType.Type.INT: "INT4",  # INT → INTEGER
            exp.DataType.Type.BIGINT: "INT8",  # BIGINT → INT8
            exp.DataType.Type.SMALLINT: "INT2",  # SMALLINT → INT2
            exp.DataType.Type.DOUBLE: "FLOAT8",  # DOUBLE → FLOAT8
            exp.DataType.Type.FLOAT: "FLOAT4",  # FLOAT → FLOAT4
            # exp.DataType.Type.VARCHAR: "CHARACTER VARYING",  # VARCHAR → CHARACTER VARYING
        }
        

        # 属性位置映射
        PROPERTIES_LOCATION = {
            **Postgres.Generator.PROPERTIES_LOCATION,
            exp.PartitionListProperty: exp.Properties.Location.POST_SCHEMA,
        }

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

        def partitionedbyproperty_sql(self, expression: exp.PartitionedByProperty) -> str:
            """生成PARTITION BY语法的SQL"""
            partition_by = self.sql(expression, "this")
            return f"PARTITION BY {partition_by}"

        def partitionlistproperty_sql(self, expression: exp.PartitionListProperty) -> str:
            """生成PARTITION BY语法的SQL，支持分区列表"""
            partition_by = self.sql(expression, "this")
            partition_list = expression.args.get("partition_list")
            
            if partition_list:
                # partition_list是一个列表，需要直接处理
                partition_list_sql = ", ".join([self.sql(part) for part in partition_list])
                return f"PARTITION BY {partition_by} ({partition_list_sql})"
            else:
                return f"PARTITION BY {partition_by}"

        def partitionrange_sql(self, expression: exp.PartitionRange) -> str:
            """生成分区范围的SQL，支持 VALUES 和 VALUES LESS THAN"""
            if expression.this and expression.expressions:
                values = self.expressions(expression, flat=True)
                
                # 检查是否应该使用 VALUES LESS THAN
                if expression.args.get("is_less_than"):
                    return f"PARTITION {expression.this} VALUES LESS THAN ({values})"
                else:
                    return f"PARTITION {expression.this} VALUES ({values})"
            elif expression.expressions:
                # 只有VALUES
                values = self.expressions(expression, flat=True)
                if expression.args.get("is_less_than"):
                    return f"VALUES LESS THAN ({values})"
                else:
                    return f"VALUES ({values})"
            else:
                # 只有分区名称
                return f"PARTITION {expression.this}"

        def partition_sql(self, expression: exp.Partition) -> str:
            """生成分区定义的SQL"""
            partition_type = expression.args.get("partition_type", "LIST")
            
            if expression.this and expression.expressions:
                # 分区名称 + VALUES
                values = self.expressions(expression, flat=True)
                return f"PARTITION {expression.this} VALUES ({values})"
            elif expression.expressions:
                # 分区类型 + 分区列
                columns = self.expressions(expression, flat=True)
                return f"{partition_type}({columns})"
            else:
                # 只有分区名称
                return f"PARTITION {expression.this}"

        def create_sql(self, expression: exp.Create) -> str:
            """生成CREATE语句的SQL，特别处理INDEX类型"""
            kind = expression.args.get("kind")
            
            if kind == "INDEX":
                return self._create_index_sql(expression)
            else:
                # 回退到父类的CREATE处理
                return super().create_sql(expression)

        def _create_index_sql(self, expression: exp.Create) -> str:
            """生成CREATE INDEX语句的SQL"""
            unique = "UNIQUE " if expression.args.get("unique") else ""
            this = self.sql(expression, "this")
            
            # 生成属性部分
            properties_sql = ""
            if expression.args.get("properties"):
                properties = []
                for prop in expression.args["properties"]:
                    prop_sql = self.sql(prop)
                    if prop_sql:
                        properties.append(prop_sql)
                if properties:
                    properties_sql = " " + " ".join(properties)
            
            # 简化版本，不包含表名、列名和USING子句
            return f"CREATE {unique}INDEX {this}{properties_sql}"

        def property_sql(self, expression: exp.Property) -> str:
            """生成属性SQL，特别处理LOCAL属性的分区列表"""
            this = expression.args.get("this")
            
            if this == "LOCAL":
                partition_list = expression.args.get("partition_list")
                if partition_list:
                    partition_list_sql = ", ".join([self.sql(part) for part in partition_list])
                    return f"LOCAL ({partition_list_sql})"
                else:
                    return "LOCAL"
            elif this == "GLOBAL":
                return "GLOBAL"
            elif this == "INCLUDE":
                expressions = self.expressions(expression, flat=True)
                return f"INCLUDE ({expressions})"
            elif this == "WITH":
                expressions = self.expressions(expression, flat=True)
                return f"WITH ({expressions})"
            elif this == "TABLESPACE":
                expressions = self.expressions(expression, flat=True)
                return f"TABLESPACE {expressions}"
            elif this == "COMMENT":
                expressions = self.expressions(expression, flat=True)
                return f"COMMENT {expressions}"
            elif this in ("VISIBLE", "INVISIBLE"):
                return this
            else:
                return super().property_sql(expression)
