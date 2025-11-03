from __future__ import annotations

import typing as t

from sqlglot import exp
from sqlglot.dialects.dialect import (
    Version,
    rename_func,
    unit_to_var,
    timestampdiff_sql,
    build_date_delta,
    groupconcat_sql,
)
from sqlglot.dialects.hive import _build_with_ignore_nulls
from sqlglot.dialects.spark2 import Spark2, temporary_storage_provider, _build_as_cast
from sqlglot.helper import ensure_list, seq_get
from sqlglot.tokens import TokenType
from sqlglot.transforms import (
    ctas_with_tmp_tables_to_create_tmp_view,
    remove_unique_constraints,
    preprocess,
    move_partitioned_by_to_schema_columns,
)


def _build_datediff(args: t.List) -> exp.Expression:
    """
    Although Spark docs don't mention the "unit" argument, Spark3 added support for
    it at some point. Databricks also supports this variant (see below).

    For example, in spark-sql (v3.3.1):
    - SELECT DATEDIFF('2020-01-01', '2020-01-05') results in -4
    - SELECT DATEDIFF(day, '2020-01-01', '2020-01-05') results in 4

    See also:
    - https://docs.databricks.com/sql/language-manual/functions/datediff3.html
    - https://docs.databricks.com/sql/language-manual/functions/datediff.html
    """
    unit = None
    this = seq_get(args, 0)
    expression = seq_get(args, 1)

    if len(args) == 3:
        unit = exp.var(t.cast(exp.Expression, this).name)
        this = args[2]

    return exp.DateDiff(
        this=exp.TsOrDsToDate(this=this), expression=exp.TsOrDsToDate(this=expression), unit=unit
    )


def _build_dateadd(args: t.List) -> exp.Expression:
    expression = seq_get(args, 1)

    if len(args) == 2:
        # DATE_ADD(startDate, numDays INTEGER)
        # https://docs.databricks.com/en/sql/language-manual/functions/date_add.html
        return exp.TsOrDsAdd(
            this=seq_get(args, 0), expression=expression, unit=exp.Literal.string("DAY")
        )

    # DATE_ADD / DATEADD / TIMESTAMPADD(unit, value integer, expr)
    # https://docs.databricks.com/en/sql/language-manual/functions/date_add3.html
    return exp.TimestampAdd(this=seq_get(args, 2), expression=expression, unit=seq_get(args, 0))


def _normalize_partition(e: exp.Expression) -> exp.Expression:
    """Normalize the expressions in PARTITION BY (<expression>, <expression>, ...)"""
    if isinstance(e, str):
        return exp.to_identifier(e)
    if isinstance(e, exp.Literal):
        return exp.to_identifier(e.name)
    return e


def _dateadd_sql(self: Hudi_Spark.Generator, expression: exp.TsOrDsAdd | exp.TimestampAdd) -> str:
    if not expression.unit or (
        isinstance(expression, exp.TsOrDsAdd) and expression.text("unit").upper() == "DAY"
    ):
        # Coming from Hive/Spark2 DATE_ADD or roundtripping the 2-arg version of Spark3/DB
        return self.func("DATE_ADD", expression.this, expression.expression)

    this = self.func(
        "DATE_ADD",
        unit_to_var(expression),
        expression.expression,
        expression.this,
    )

    if isinstance(expression, exp.TsOrDsAdd):
        # The 3 arg version of DATE_ADD produces a timestamp in Spark3/DB but possibly not
        # in other dialects
        return_type = expression.return_type
        if not return_type.is_type(exp.DataType.Type.TIMESTAMP, exp.DataType.Type.DATETIME):
            this = f"CAST({this} AS {return_type})"

    return this


def _groupconcat_sql(self: Hudi_Spark.Generator, expression: exp.GroupConcat) -> str:
    if self.dialect.version < Version("4.0.0"):
        expr = exp.ArrayToString(
            this=exp.ArrayAgg(this=expression.this),
            expression=expression.args.get("separator") or exp.Literal.string(""),
        )
        return self.sql(expr)

    return groupconcat_sql(self, expression)


class Hudi_Spark(Spark2):
    SUPPORTS_ORDER_BY_ALL = True

    class Tokenizer(Spark2.Tokenizer):
        STRING_ESCAPES_ALLOWED_IN_RAW_STRINGS = False

        RAW_STRINGS = [
            (prefix + q, q)
            for q in t.cast(t.List[str], Spark2.Tokenizer.QUOTES)
            for prefix in ("r", "R")
        ]

    class Parser(Spark2.Parser):
        FUNCTIONS = {
            **Spark2.Parser.FUNCTIONS,
            "ANY_VALUE": _build_with_ignore_nulls(exp.AnyValue),
            "DATE_ADD": _build_dateadd,
            "DATEADD": _build_dateadd,
            "TIMESTAMPADD": _build_dateadd,
            "TIMESTAMPDIFF": build_date_delta(exp.TimestampDiff),
            "DATEDIFF": _build_datediff,
            "DATE_DIFF": _build_datediff,
            "LISTAGG": exp.GroupConcat.from_arg_list,
            "TIMESTAMP_LTZ": _build_as_cast("TIMESTAMP_LTZ"),
            "TIMESTAMP_NTZ": _build_as_cast("TIMESTAMP_NTZ"),
            "TRY_ELEMENT_AT": lambda args: exp.Bracket(
                this=seq_get(args, 0),
                expressions=ensure_list(seq_get(args, 1)),
                offset=1,
                safe=True,
            ),
        }

        # Hudi 特定的属性解析器
        PROPERTY_PARSERS = {
            **Spark2.Parser.PROPERTY_PARSERS,
            "OPTIONS": lambda self: self._parse_hudi_options(),
            "TBLPROPERTIES": lambda self: self._parse_hudi_tblproperties(),
        }

        # Hudi OPTIONS 中支持的配置项
        HUDI_OPTION_PARSERS = {
            "TYPE": lambda self: self._parse_hudi_property_assignment("type"),
            "PRIMARYKEY": lambda self: self._parse_hudi_property_assignment("primaryKey"),
            "PRIMARY_KEY": lambda self: self._parse_hudi_property_assignment("primaryKey"),
            "PRECOMBINEFIELD": lambda self: self._parse_hudi_property_assignment("preCombineField"),
            "PRECOMBINE_FIELD": lambda self: self._parse_hudi_property_assignment("preCombineField"),
            "HOODIE_INDEX_TYPE": lambda self: self._parse_hudi_property_assignment("hoodie.index.type"),
            "HOODIE_BLOOM_INDEX_UPDATE_PARTITION_PATH": lambda self: self._parse_hudi_property_assignment("hoodie.bloom.index.update.partition.path"),
            "HOODIE_BLOOM_INDEX_FILTER_TYPE": lambda self: self._parse_hudi_property_assignment("hoodie.bloom.index.filter.type"),
            "HOODIE_BUCKET_INDEX_NUM_BUCKETS": lambda self: self._parse_hudi_property_assignment("hoodie.bucket.index.num.buckets"),
            "HOODIE_BUCKET_INDEX_HASH_FIELD": lambda self: self._parse_hudi_property_assignment("hoodie.bucket.index.hash.field"),
            "HOODIE_QUERY_AS_RO_TABLE": lambda self: self._parse_hudi_property_assignment("hoodie.query.as.ro.table"),
            "HOODIE_DATASOURCE_WRITE_RECORDKEY_FIELD": lambda self: self._parse_hudi_property_assignment("hoodie.datasource.write.recordkey.field"),
            "HOODIE_SCHEMA_EVOLUTION_ENABLE": lambda self: self._parse_hudi_property_assignment("hoodie.schema.evolution.enable"),
            "HOODIE_WRITE_LOCK_ZOOKEEPER_BASE_PATH": lambda self: self._parse_hudi_property_assignment("hoodie.write.lock.zookeeper.base.path"),
            "LAST_COMMIT_COMPLETION_TIME_SYNC": lambda self: self._parse_hudi_property_assignment("last_commit_completion_time_sync"),
            "LAST_COMMIT_TIME_SYNC": lambda self: self._parse_hudi_property_assignment("last_commit_time_sync"),
            "HOODIE_PARQUET_MAX_FILE_SIZE": lambda self: self._parse_hudi_property_assignment("hoodie.parquet.max.file.size"),
            "HOODIE_TABLE_DESCRIPTION": lambda self: self._parse_hudi_property_assignment("hoodie.table.description"),
        }

        PLACEHOLDER_PARSERS = {
            **Spark2.Parser.PLACEHOLDER_PARSERS,
            TokenType.L_BRACE: lambda self: self._parse_query_parameter(),
        }

        def _parse_query_parameter(self) -> t.Optional[exp.Expression]:
            this = self._parse_id_var()
            self._match(TokenType.R_BRACE)
            return self.expression(exp.Placeholder, this=this, widget=True)

        def _parse_generated_as_identity(
            self,
        ) -> (
            exp.GeneratedAsIdentityColumnConstraint
            | exp.ComputedColumnConstraint
            | exp.GeneratedAsRowColumnConstraint
        ):
            this = super()._parse_generated_as_identity()
            if this.expression:
                return self.expression(exp.ComputedColumnConstraint, this=this.expression)
            return this

        def _parse_hudi_options(self) -> t.List[exp.Expression]:
            """解析 Hudi OPTIONS 子句，格式为 OPTIONS (key1 = 'value1', key2 = 'value2', ...)"""
            return self._parse_wrapped_csv(self._parse_hudi_option_property)

        def _parse_hudi_tblproperties(self) -> t.List[exp.Expression]:
            """解析 Hudi TBLPROPERTIES 子句，格式为 TBLPROPERTIES (key1 = 'value1', key2 = 'value2', ...)"""
            return self._parse_wrapped_csv(self._parse_hudi_tblproperty)

        def _parse_hudi_option_property(self) -> t.Optional[exp.Expression]:
            """解析单个 Hudi OPTIONS 属性"""
            if self._match_texts(self.HUDI_OPTION_PARSERS):
                return self.HUDI_OPTION_PARSERS[self._prev.text.upper()](self)
            
            # 处理单引号包围的属性名，如 'hoodie.query.as.ro.table'
            if self._match(TokenType.STRING):
                key = self._prev.text
                if self._match(TokenType.EQ):
                    # 使用更通用的表达式解析，支持布尔值、字符串、数字等
                    value = self._parse_bitwise() or self._parse_var(any_token=True)
                    return self.expression(exp.Property, this=key, value=value)
            
            # 处理带点号的属性名，如 hoodie.index.type
            # 检查各种可能的标识符类型，包括关键字
            if self._match_set((TokenType.IDENTIFIER, TokenType.VAR, TokenType.INDEX, TokenType.UPDATE, TokenType.PARTITION, TokenType.SCHEMA, TokenType.ENABLE)):
                key = self._prev.text
                # 检查是否有更多的点号分隔的标识符
                while self._match(TokenType.DOT) and self._match_set((TokenType.IDENTIFIER, TokenType.VAR, TokenType.INDEX, TokenType.UPDATE, TokenType.PARTITION, TokenType.SCHEMA, TokenType.ENABLE)):
                    key += "." + self._prev.text
                
                if self._match(TokenType.EQ):
                    # 使用更通用的表达式解析，支持布尔值、字符串、数字等
                    value = self._parse_bitwise() or self._parse_var(any_token=True)
                    return self.expression(exp.Property, this=key, value=value)
            
            # 如果没有匹配到任何属性，返回 None 让上层处理
            return None

        def _parse_hudi_tblproperty(self) -> t.Optional[exp.Expression]:
            """解析单个 Hudi TBLPROPERTIES 属性"""
            # 处理单引号包围的属性名，如 'hoodie.query.as.ro.table'
            if self._match(TokenType.STRING):
                key = self._prev.text
                if self._match(TokenType.EQ):
                    # 使用更通用的表达式解析，支持布尔值、字符串、数字等
                    value = self._parse_bitwise() or self._parse_var(any_token=True)
                    return self.expression(exp.Property, this=key, value=value)
            
            if self._match_set((TokenType.IDENTIFIER, TokenType.VAR, TokenType.INDEX, TokenType.UPDATE, TokenType.PARTITION, TokenType.SCHEMA, TokenType.ENABLE)):
                key = self._prev.text
                # 检查是否有更多的点号分隔的标识符
                while self._match(TokenType.DOT) and self._match_set((TokenType.IDENTIFIER, TokenType.VAR, TokenType.INDEX, TokenType.UPDATE, TokenType.PARTITION, TokenType.SCHEMA, TokenType.ENABLE)):
                    key += "." + self._prev.text
                
                if self._match(TokenType.EQ):
                    # 使用更通用的表达式解析，支持布尔值、字符串、数字等
                    value = self._parse_bitwise() or self._parse_var(any_token=True)
                    return self.expression(exp.Property, this=key, value=value)
            
            return None

        def _parse_hudi_property_assignment(self, key: str) -> exp.Property:
            """解析 Hudi 属性赋值，格式为 key = value"""
            if self._match(TokenType.EQ):
                value = self._parse_expression()
                return self.expression(exp.Property, this=key, value=value)
            return self.expression(exp.Property, this=key)

    class Generator(Spark2.Generator):
        SUPPORTS_TO_NUMBER = True
        PAD_FILL_PATTERN_IS_REQUIRED = False
        SUPPORTS_CONVERT_TIMEZONE = True
        SUPPORTS_MEDIAN = True
        SUPPORTS_UNIX_SECONDS = True
        SUPPORTS_DECODE_CASE = True

        TYPE_MAPPING = {
            **Spark2.Generator.TYPE_MAPPING,
            exp.DataType.Type.MONEY: "DECIMAL(15, 4)",
            exp.DataType.Type.SMALLMONEY: "DECIMAL(6, 4)",
            exp.DataType.Type.UUID: "STRING",
            exp.DataType.Type.TIMESTAMPLTZ: "TIMESTAMP_LTZ",
            exp.DataType.Type.TIMESTAMPNTZ: "TIMESTAMP_NTZ",
        }

        TRANSFORMS = {
            **Spark2.Generator.TRANSFORMS,
            exp.ArrayConstructCompact: lambda self, e: self.func(
                "ARRAY_COMPACT", self.func("ARRAY", *e.expressions)
            ),
            exp.Create: preprocess(
                [
                    remove_unique_constraints,
                    lambda e: ctas_with_tmp_tables_to_create_tmp_view(
                        e, temporary_storage_provider
                    ),
                    move_partitioned_by_to_schema_columns,
                ]
            ),
            exp.DateFromUnixDate: rename_func("DATE_FROM_UNIX_DATE"),
            exp.GroupConcat: _groupconcat_sql,
            exp.EndsWith: rename_func("ENDSWITH"),
            exp.PartitionedByProperty: lambda self,
            e: f"PARTITIONED BY {self.wrap(self.expressions(sqls=[_normalize_partition(e) for e in e.this.expressions], skip_first=True))}",
            exp.StartsWith: rename_func("STARTSWITH"),
            exp.TsOrDsAdd: _dateadd_sql,
            exp.TimestampAdd: _dateadd_sql,
            exp.DatetimeDiff: timestampdiff_sql,
            exp.TimestampDiff: timestampdiff_sql,
            exp.TryCast: lambda self, e: (
                self.trycast_sql(e) if e.args.get("safe") else self.cast_sql(e)
            ),
        }
        TRANSFORMS.pop(exp.AnyValue)
        TRANSFORMS.pop(exp.DateDiff)
        TRANSFORMS.pop(exp.Group)

        def bracket_sql(self, expression: exp.Bracket) -> str:
            if expression.args.get("safe"):
                key = seq_get(self.bracket_offset_expressions(expression, index_offset=1), 0)
                return self.func("TRY_ELEMENT_AT", expression.this, key)

            return super().bracket_sql(expression)

        def computedcolumnconstraint_sql(self, expression: exp.ComputedColumnConstraint) -> str:
            return f"GENERATED ALWAYS AS ({self.sql(expression, 'this')})"

        def anyvalue_sql(self, expression: exp.AnyValue) -> str:
            return self.function_fallback_sql(expression)

        def datediff_sql(self, expression: exp.DateDiff) -> str:
            end = self.sql(expression, "this")
            start = self.sql(expression, "expression")

            if expression.unit:
                return self.func("DATEDIFF", unit_to_var(expression), start, end)

            return self.func("DATEDIFF", end, start)

        def placeholder_sql(self, expression: exp.Placeholder) -> str:
            if not expression.args.get("widget"):
                return super().placeholder_sql(expression)

            return f"{{{expression.name}}}"

        def property_sql(self, expression: exp.Property) -> str:
            """生成属性赋值的 SQL，格式为 key = value"""
            if expression.value:
                return f"{expression.this} = {self.sql(expression, 'value')}"
            return str(expression.this)
