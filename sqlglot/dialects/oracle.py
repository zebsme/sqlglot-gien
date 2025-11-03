from __future__ import annotations

import typing as t

from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
    Dialect,
    NormalizationStrategy,
    build_timetostr_or_tochar,
    build_formatted_time,
    no_ilike_sql,
    rename_func,
    strposition_sql,
    to_number_with_nls_param,
    trim_sql,
)
from sqlglot.helper import seq_get
from sqlglot.parser import OPTIONS_TYPE, build_coalesce
from sqlglot.tokens import TokenType

if t.TYPE_CHECKING:
    from sqlglot._typing import E


def _trim_sql(self: Oracle.Generator, expression: exp.Trim) -> str:
    position = expression.args.get("position")

    if position and position.upper() in ("LEADING", "TRAILING"):
        return self.trim_sql(expression)

    return trim_sql(self, expression)


def _build_to_timestamp(args: t.List) -> exp.StrToTime | exp.Anonymous:
    if len(args) == 1:
        return exp.Anonymous(this="TO_TIMESTAMP", expressions=args)

    return build_formatted_time(exp.StrToTime, "oracle")(args)


class Oracle(Dialect):
    ALIAS_POST_TABLESAMPLE = True
    LOCKING_READS_SUPPORTED = True
    TABLESAMPLE_SIZE_IS_PERCENT = True
    NULL_ORDERING = "nulls_are_large"
    ON_CONDITION_EMPTY_BEFORE_ERROR = False
    ALTER_TABLE_ADD_REQUIRED_FOR_EACH_COLUMN = False

    # See section 8: https://docs.oracle.com/cd/A97630_01/server.920/a96540/sql_elements9a.htm
    NORMALIZATION_STRATEGY = NormalizationStrategy.UPPERCASE

    # https://docs.oracle.com/database/121/SQLRF/sql_elements004.htm#SQLRF00212
    # https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes
    TIME_MAPPING = {
        "D": "%u",  # Day of week (1-7)
        "DAY": "%A",  # name of day
        "DD": "%d",  # day of month (1-31)
        "DDD": "%j",  # day of year (1-366)
        "DY": "%a",  # abbreviated name of day
        "HH": "%I",  # Hour of day (1-12)
        "HH12": "%I",  # alias for HH
        "HH24": "%H",  # Hour of day (0-23)
        "IW": "%V",  # Calendar week of year (1-52 or 1-53), as defined by the ISO 8601 standard
        "MI": "%M",  # Minute (0-59)
        "MM": "%m",  # Month (01-12; January = 01)
        "MON": "%b",  # Abbreviated name of month
        "MONTH": "%B",  # Name of month
        "SS": "%S",  # Second (0-59)
        "WW": "%W",  # Week of year (1-53)
        "YY": "%y",  # 15
        "YYYY": "%Y",  # 2015
        "FF6": "%f",  # only 6 digits are supported in python formats
    }

    class Tokenizer(tokens.Tokenizer):
        VAR_SINGLE_TOKENS = {"@", "$", "#"}

        UNICODE_STRINGS = [
            (prefix + q, q)
            for q in t.cast(t.List[str], tokens.Tokenizer.QUOTES)
            for prefix in ("U", "u")
        ]

        NESTED_COMMENTS = False

        KEYWORDS = {
            **tokens.Tokenizer.KEYWORDS,
            "(+)": TokenType.JOIN_MARKER,
            "BINARY_DOUBLE": TokenType.DOUBLE,
            "BINARY_FLOAT": TokenType.FLOAT,
            "BULK COLLECT INTO": TokenType.BULK_COLLECT_INTO,
            "COLUMNS": TokenType.COLUMN,
            "MATCH_RECOGNIZE": TokenType.MATCH_RECOGNIZE,
            "MINUS": TokenType.EXCEPT,
            "NVARCHAR2": TokenType.NVARCHAR,
            "ORDER SIBLINGS BY": TokenType.ORDER_SIBLINGS_BY,
            "SAMPLE": TokenType.TABLE_SAMPLE,
            "START": TokenType.BEGIN,
            "TOP": TokenType.TOP,
            "VARCHAR2": TokenType.VARCHAR,
            "ENABLE": TokenType.ENABLE,
            "TIMESTAMP": TokenType.TIMESTAMP,
        }

    class Parser(parser.Parser):
        WINDOW_BEFORE_PAREN_TOKENS = {TokenType.OVER, TokenType.KEEP}
        VALUES_FOLLOWED_BY_PAREN = False

        FUNCTIONS = {
            **parser.Parser.FUNCTIONS,
            "CONVERT": exp.ConvertToCharset.from_arg_list,
            "L2_DISTANCE": exp.EuclideanDistance.from_arg_list,
            "NVL": lambda args: build_coalesce(args, is_nvl=True),
            "SQUARE": lambda args: exp.Pow(this=seq_get(args, 0), expression=exp.Literal.number(2)),
            "TO_CHAR": build_timetostr_or_tochar,
            "TO_TIMESTAMP": _build_to_timestamp,
            "TO_DATE": build_formatted_time(exp.StrToDate, "oracle"),
            "TRUNC": lambda args: exp.DateTrunc(
                unit=seq_get(args, 1) or exp.Literal.string("DD"),
                this=seq_get(args, 0),
                unabbreviate=False,
            ),
        }

        NO_PAREN_FUNCTION_PARSERS = {
            **parser.Parser.NO_PAREN_FUNCTION_PARSERS,
            "NEXT": lambda self: self._parse_next_value_for(),
            "PRIOR": lambda self: self.expression(exp.Prior, this=self._parse_bitwise()),
            "SYSDATE": lambda self: self.expression(exp.CurrentTimestamp, sysdate=True),
            "DBMS_RANDOM": lambda self: self._parse_dbms_random(),
        }

        FUNCTION_PARSERS: t.Dict[str, t.Callable] = {
            **parser.Parser.FUNCTION_PARSERS,
            "JSON_ARRAY": lambda self: self._parse_json_array(
                exp.JSONArray,
                expressions=self._parse_csv(lambda: self._parse_format_json(self._parse_bitwise())),
            ),
            "JSON_ARRAYAGG": lambda self: self._parse_json_array(
                exp.JSONArrayAgg,
                this=self._parse_format_json(self._parse_bitwise()),
                order=self._parse_order(),
            ),
            "JSON_EXISTS": lambda self: self._parse_json_exists(),
        }
        FUNCTION_PARSERS.pop("CONVERT")

        PROPERTY_PARSERS = {
            **parser.Parser.PROPERTY_PARSERS,
            "GLOBAL": lambda self: self._match_text_seq("TEMPORARY")
            and self.expression(exp.TemporaryProperty, this="GLOBAL"),
            "PRIVATE": lambda self: self._match_text_seq("TEMPORARY")
            and self.expression(exp.TemporaryProperty, this="PRIVATE"),
            "FORCE": lambda self: self.expression(exp.ForceProperty),
            "PARTITION BY": lambda self: self._parse_partitioned_by_oracle(),
        }

        STATEMENT_PARSERS = {
            **parser.Parser.STATEMENT_PARSERS,
            "COMMENT": lambda self: self._parse_comment_on(),
        }

        QUERY_MODIFIER_PARSERS = {
            **parser.Parser.QUERY_MODIFIER_PARSERS,
            TokenType.ORDER_SIBLINGS_BY: lambda self: ("order", self._parse_order()),
            TokenType.WITH: lambda self: ("options", [self._parse_query_restrictions()]),
        }

        TYPE_LITERAL_PARSERS = {
            exp.DataType.Type.DATE: lambda self, this, _: self.expression(
                exp.DateStrToDate, this=this
            )
        }

        def _parse_types(self, check_func: bool = False, schema: bool = False, allow_identifiers: bool = True) -> t.Optional[exp.Expression]:
            """Override to handle Oracle-specific data types."""
            # Handle INT(0) -> INT (ignore parameter)
            if self._match_text_seq("INT"):
                if self._match(TokenType.L_PAREN):
                    # Parse INT(0) as INT - consume the parameter but ignore it
                    self._parse_number()  # Parse the 0
                    self._match(TokenType.R_PAREN)
                    return self.expression(exp.DataType, this=exp.DataType.Type.INT)
                else:
                    return self.expression(exp.DataType, this=exp.DataType.Type.INT)

            # Handle FLOAT(256) -> FLOAT with length parameter
            if self._match_text_seq("FLOAT"):
                if self._match(TokenType.L_PAREN):
                    # Parse FLOAT(256) and preserve the length parameter
                    length_value = self._parse_number()  # Parse the 256
                    self._match(TokenType.R_PAREN)
                    # Create a literal expression for the length
                    length_expr = self.expression(exp.Literal, this=length_value.this, is_string=False)
                    return self.expression(exp.DataType, this=exp.DataType.Type.FLOAT, expressions=[length_expr])
                else:
                    return self.expression(exp.DataType, this=exp.DataType.Type.FLOAT)

            # Handle DOUBLE PRECISION -> BINARY_DOUBLE
            if self._match_text_seq("DOUBLE", "PRECISION"):
                return self.expression(exp.DataType, this=exp.DataType.Type.BINARY_DOUBLE)

            # For all other cases, use the parent implementation
            return super()._parse_types(check_func, schema, allow_identifiers)

        def _parse_field_def(self) -> t.Optional[exp.Expression]:
            """Override field definition parsing for Oracle."""
            # Use the parent implementation but with Oracle-specific handling
            return super()._parse_field_def()

        def _parse_schema(self, this: t.Optional[exp.Expression] = None) -> t.Optional[exp.Expression]:
            """Override schema parsing for Oracle."""
            # Use the parent implementation but with Oracle-specific handling
            return super()._parse_schema(this)

        def _parse_column_def(self, this: t.Optional[exp.Expression], computed_column: bool = True) -> t.Optional[exp.Expression]:
            """Override column definition parsing for Oracle."""
            # Use the parent implementation but with Oracle-specific handling
            return super()._parse_column_def(this, computed_column)

        # SELECT UNIQUE .. is old-style Oracle syntax for SELECT DISTINCT ..
        # Reference: https://stackoverflow.com/a/336455
        DISTINCT_TOKENS = {TokenType.DISTINCT, TokenType.UNIQUE}

        QUERY_RESTRICTIONS: OPTIONS_TYPE = {
            "WITH": (
                ("READ", "ONLY"),
                ("CHECK", "OPTION"),
            ),
        }

        def _parse_dbms_random(self) -> t.Optional[exp.Expression]:
            if self._match_text_seq(".", "VALUE"):
                lower, upper = None, None
                if self._match(TokenType.L_PAREN, advance=False):
                    lower_upper = self._parse_wrapped_csv(self._parse_bitwise)
                    if len(lower_upper) == 2:
                        lower, upper = lower_upper

                return exp.Rand(lower=lower, upper=upper)

            self._retreat(self._index - 1)
            return None

        def _parse_json_array(self, expr_type: t.Type[E], **kwargs) -> E:
            return self.expression(
                expr_type,
                null_handling=self._parse_on_handling("NULL", "NULL", "ABSENT"),
                return_type=self._match_text_seq("RETURNING") and self._parse_type(),
                strict=self._match_text_seq("STRICT"),
                **kwargs,
            )

        def _parse_hint_function_call(self) -> t.Optional[exp.Expression]:
            if not self._curr or not self._next or self._next.token_type != TokenType.L_PAREN:
                return None

            this = self._curr.text

            self._advance(2)
            args = self._parse_hint_args()
            this = self.expression(exp.Anonymous, this=this, expressions=args)
            self._match_r_paren(this)
            return this

        def _parse_hint_args(self):
            args = []
            result = self._parse_var()

            while result:
                args.append(result)
                result = self._parse_var()

            return args

        def _parse_query_restrictions(self) -> t.Optional[exp.Expression]:
            kind = self._parse_var_from_options(self.QUERY_RESTRICTIONS, raise_unmatched=False)

            if not kind:
                return None

            return self.expression(
                exp.QueryOption,
                this=kind,
                expression=self._match(TokenType.CONSTRAINT) and self._parse_field(),
            )

        def _parse_json_exists(self) -> exp.JSONExists:
            this = self._parse_format_json(self._parse_bitwise())
            self._match(TokenType.COMMA)
            return self.expression(
                exp.JSONExists,
                this=this,
                path=self.dialect.to_json_path(self._parse_bitwise()),
                passing=self._match_text_seq("PASSING")
                and self._parse_csv(lambda: self._parse_alias(self._parse_bitwise())),
                on_condition=self._parse_on_condition(),
            )

        def _parse_into(self) -> t.Optional[exp.Into]:
            # https://docs.oracle.com/en/database/oracle/oracle-database/19/lnpls/SELECT-INTO-statement.html
            bulk_collect = self._match(TokenType.BULK_COLLECT_INTO)
            if not bulk_collect and not self._match(TokenType.INTO):
                return None

            index = self._index

            expressions = self._parse_expressions()
            if len(expressions) == 1:
                self._retreat(index)
                self._match(TokenType.TABLE)
                return self.expression(
                    exp.Into, this=self._parse_table(schema=True), bulk_collect=bulk_collect
                )

            return self.expression(exp.Into, bulk_collect=bulk_collect, expressions=expressions)

        def _parse_connect_with_prior(self):
            return self._parse_assignment()

        def _parse_partitioned_by_oracle(self) -> t.Optional[exp.Expression]:
            """Parse Oracle PARTITION BY syntax with a simple approach."""
            # Parse the partition type and column
            if self._match_text_seq("LIST"):
                partition_expressions = self._parse_wrapped_csv(self._parse_expression)
                
                # Skip the partition definitions for now - just consume the tokens
                if self._match(TokenType.L_PAREN):
                    # Find the matching closing paren
                    paren_count = 1
                    while paren_count > 0 and self._curr:
                        if self._curr.token_type == TokenType.L_PAREN:
                            paren_count += 1
                        elif self._curr.token_type == TokenType.R_PAREN:
                            paren_count -= 1
                        self._advance()
                
                # Return a simple partitioned by property
                return self.expression(
                    exp.PartitionedByProperty,
                    this=self.expression(exp.Partition, expressions=partition_expressions)
                )
            else:
                # Fall back to the base parser for other partition types
                return super()._parse_partitioned_by()

        def _parse_partition_by_oracle(self) -> t.Optional[exp.Expression]:
            """Parse Oracle PARTITION BY syntax."""
            if not self._match_text_seq("PARTITION", "BY"):
                return None
            
            # Parse partition type (RANGE, LIST, HASH, INTERVAL)
            if self._match_text_seq("RANGE"):
                return self._parse_partition_by_range_oracle()
            elif self._match_text_seq("LIST"):
                return self._parse_partition_by_list_oracle()
            elif self._match_text_seq("HASH"):
                return self._parse_partition_by_hash_oracle()
            elif self._match_text_seq("INTERVAL"):
                return self._parse_partition_by_interval_oracle()
            else:
                # Fall back to generic partition parsing
                return super()._parse_partitioned_by()

        def _parse_partition_by_range_oracle(self) -> exp.Expression:
            """Parse Oracle PARTITION BY RANGE syntax."""
            partition_expressions = self._parse_wrapped_csv(self._parse_expression)
            create_expressions = self._parse_wrapped_csv(self._parse_partition_definition_oracle)
            
            return self.expression(
                exp.PartitionByRangeProperty,
                partition_expressions=partition_expressions,
                create_expressions=create_expressions,
            )

        def _parse_partition_by_list_oracle(self) -> exp.Expression:
            """Parse Oracle PARTITION BY LIST syntax."""
            # Parse the partition column(s)
            partition_expressions = self._parse_wrapped_csv(self._parse_expression)
            
            # Parse the partition definitions
            create_expressions = []
            if self._match(TokenType.L_PAREN):
                # Parse each partition definition
                create_expressions = self._parse_csv(self._parse_partition_definition_oracle)
                self._match(TokenType.R_PAREN)
            
            return self.expression(
                exp.PartitionByListProperty,
                partition_expressions=partition_expressions,
                create_expressions=create_expressions,
            )

        def _parse_partition_by_hash_oracle(self) -> exp.Expression:
            """Parse Oracle PARTITION BY HASH syntax."""
            partition_expressions = self._parse_wrapped_csv(self._parse_expression)
            
            return self.expression(
                exp.PartitionedByProperty,
                this=self.expression(exp.Partition, expressions=partition_expressions),
            )

        def _parse_partition_by_interval_oracle(self) -> exp.Expression:
            """Parse Oracle PARTITION BY INTERVAL syntax."""
            # Parse the interval expression (e.g., INTERVAL (NUMTODSINTERVAL(1,'MONTH')))
            interval_expression = self._parse_expression()
            
            # Parse the range partition key
            partition_expressions = self._parse_wrapped_csv(self._parse_expression)
            
            # Parse partition definitions
            create_expressions = self._parse_wrapped_csv(self._parse_partition_definition_oracle)
            
            return self.expression(
                exp.PartitionByIntervalProperty,
                interval_expression=interval_expression,
                partition_expressions=partition_expressions,
                create_expressions=create_expressions,
            )

        def _parse_partition_definition_oracle(self) -> t.Optional[exp.Expression]:
            """Parse Oracle partition definition like 'PARTITION p1 VALUES (10)'."""
            if not self._match_text_seq("PARTITION"):
                return None
            
            name = self._parse_id_var()
            values = None
            
            # Parse VALUES clause
            if self._match_text_seq("VALUES"):
                values = self._parse_wrapped_csv(self._parse_expression)
            
            # Create a simple partition list expression
            return self.expression(
                exp.PartitionList,
                this=name,
                expressions=values,
            )

        def _parse_storage_clause(self) -> t.Optional[exp.Expression]:
            """Parse Oracle STORAGE clause."""
            storage_options = {}
            
            while not self._match(TokenType.R_PAREN, advance=False):
                if self._match_text_seq("INITIAL"):
                    storage_options["INITIAL"] = self._parse_number()
                elif self._match_text_seq("NEXT"):
                    storage_options["NEXT"] = self._parse_number()
                elif self._match_text_seq("MINEXTENTS"):
                    storage_options["MINEXTENTS"] = self._parse_number()
                elif self._match_text_seq("MAXEXTENTS"):
                    storage_options["MAXEXTENTS"] = self._parse_number()
                elif self._match_text_seq("PCTINCREASE"):
                    storage_options["PCTINCREASE"] = self._parse_number()
                elif self._match_text_seq("FREELISTS"):
                    storage_options["FREELISTS"] = self._parse_number()
                elif self._match_text_seq("FREELIST"):
                    storage_options["FREELIST"] = self._parse_number()
                else:
                    break
            
            return self.expression(exp.Anonymous, this="STORAGE", expressions=list(storage_options.values()))

        def _parse_subpartition_definition_oracle(self) -> t.Optional[exp.Expression]:
            """Parse Oracle subpartition definition."""
            if not self._match_text_seq("SUBPARTITION"):
                return None
            
            name = self._parse_id_var()
            values = None
            tablespace = None
            storage_options = []
            
            # Parse VALUES clause for LIST subpartitions
            if self._match_text_seq("VALUES"):
                if self._match(TokenType.L_PAREN):
                    values = self._parse_wrapped_csv(self._parse_expression)
                else:
                    # Handle VALUES IN syntax
                    if self._match_text_seq("IN"):
                        values = self._parse_wrapped_csv(self._parse_expression)
            
            # Parse subpartition attributes
            while True:
                if self._match_text_seq("TABLESPACE"):
                    tablespace = self._parse_id_var()
                elif self._match_text_seq("STORAGE"):
                    if self._match(TokenType.L_PAREN):
                        storage_options.append(self._parse_storage_clause())
                        self._match(TokenType.R_PAREN)
                elif self._match_text_seq("LOGGING") or self._match_text_seq("NOLOGGING"):
                    storage_options.append(self._prev.text.upper())
                elif self._match_text_seq("COMPRESS") or self._match_text_seq("NOCOMPRESS"):
                    storage_options.append(self._prev.text.upper())
                else:
                    break
            
            # Use PartitionList for subpartition definitions as well
            return self.expression(
                exp.PartitionList,
                this=name,
                expressions=values,
            )

        def _parse_comment_on(self) -> t.Optional[exp.Expression]:
            """Parse Oracle COMMENT ON syntax."""
            if not self._match_text_seq("COMMENT", "ON"):
                return None
            
            # Parse object type (TABLE, COLUMN, etc.)
            kind = self._parse_id_var()
            if not kind:
                return None
            
            # Parse object name (table.column or just table)
            this = self._parse_table_parts()
            if not this:
                return None
            
            # Parse IS clause
            if not self._match_text_seq("IS"):
                return None
            
            # Parse comment text
            comment_text = self._parse_string()
            if not comment_text:
                return None
            
            return self.expression(
                exp.Comment,
                this=comment_text,
                kind=kind,
                expression=this,
            )

        def _parse_column_constraint(self) -> t.Optional[exp.Expression]:
            this = self._match(TokenType.CONSTRAINT) and self._parse_id_var()

            procedure_option_follows = (
                self._match(TokenType.WITH, advance=False)
                and self._next
                and self._next.text.upper() in self.PROCEDURE_OPTIONS
            )

            if not procedure_option_follows and self._match_texts(self.CONSTRAINT_PARSERS):
                constraint = self.CONSTRAINT_PARSERS[self._prev.text.upper()](self)
                
                # 检查是否有 ENABLE 关键字
                if self._match(TokenType.ENABLE):
                    if hasattr(constraint, 'set'):
                        constraint.set('enabled', True)
                
                return self.expression(
                    exp.ColumnConstraint,
                    this=this,
                    kind=constraint,
                )

            return this

    class Generator(generator.Generator):
        LOCKING_READS_SUPPORTED = True
        JOIN_HINTS = False
        TABLE_HINTS = False
        DATA_TYPE_SPECIFIERS_ALLOWED = True
        ALTER_TABLE_INCLUDE_COLUMN_KEYWORD = False
        LIMIT_FETCH = "FETCH"
        TABLESAMPLE_KEYWORDS = "SAMPLE"
        LAST_DAY_SUPPORTS_DATE_PART = False
        SUPPORTS_SELECT_INTO = True
        TZ_TO_WITH_TIME_ZONE = True
        SUPPORTS_WINDOW_EXCLUDE = True
        QUERY_HINT_SEP = " "
        SUPPORTS_DECODE_CASE = True

        TYPE_MAPPING = {
            **generator.Generator.TYPE_MAPPING,
            exp.DataType.Type.TINYINT: "SMALLINT",
            exp.DataType.Type.SMALLINT: "SMALLINT",
            exp.DataType.Type.INT: "INT",
            exp.DataType.Type.BIGINT: "INT",
            exp.DataType.Type.DECIMAL: "NUMBER",
            exp.DataType.Type.DOUBLE: "DOUBLE PRECISION",
            exp.DataType.Type.VARCHAR: "VARCHAR2",
            exp.DataType.Type.NVARCHAR: "NVARCHAR2",
            exp.DataType.Type.NCHAR: "NCHAR",
            exp.DataType.Type.TEXT: "CLOB",
            exp.DataType.Type.TIMETZ: "TIME",
            exp.DataType.Type.TIMESTAMPNTZ: "TIMESTAMP",
            exp.DataType.Type.TIMESTAMPTZ: "TIMESTAMP",
            exp.DataType.Type.BINARY: "BLOB",
            exp.DataType.Type.VARBINARY: "BLOB",
            exp.DataType.Type.ROWVERSION: "BLOB",
        }
        TYPE_MAPPING.pop(exp.DataType.Type.BLOB)

        TRANSFORMS = {
            **generator.Generator.TRANSFORMS,
            exp.DateStrToDate: lambda self, e: self.func(
                "TO_DATE", e.this, exp.Literal.string("YYYY-MM-DD")
            ),
            exp.DateTrunc: lambda self, e: self.func("TRUNC", e.this, e.unit),
            exp.EuclideanDistance: rename_func("L2_DISTANCE"),
            exp.ILike: no_ilike_sql,
            exp.LogicalOr: rename_func("MAX"),
            exp.LogicalAnd: rename_func("MIN"),
            exp.Mod: rename_func("MOD"),
            exp.Rand: rename_func("DBMS_RANDOM.VALUE"),
            exp.Select: transforms.preprocess(
                [
                    transforms.eliminate_distinct_on,
                    transforms.eliminate_qualify,
                ]
            ),
            exp.StrPosition: lambda self, e: (
                strposition_sql(
                    self, e, func_name="INSTR", supports_position=True, supports_occurrence=True
                )
            ),
            exp.StrToTime: lambda self, e: self.func("TO_TIMESTAMP", e.this, self.format_time(e)),
            exp.StrToDate: lambda self, e: self.func("TO_DATE", e.this, self.format_time(e)),
            exp.Subquery: lambda self, e: self.subquery_sql(e, sep=" "),
            exp.Substring: rename_func("SUBSTR"),
            exp.Table: lambda self, e: self.table_sql(e, sep=" "),
            exp.TableSample: lambda self, e: self.tablesample_sql(e),
            exp.TemporaryProperty: lambda _, e: f"{e.name or 'GLOBAL'} TEMPORARY",
            exp.TimeToStr: lambda self, e: self.func("TO_CHAR", e.this, self.format_time(e)),
            exp.ToChar: lambda self, e: self.function_fallback_sql(e),
            exp.ToNumber: to_number_with_nls_param,
            exp.Trim: _trim_sql,
            exp.Unicode: lambda self, e: f"ASCII(UNISTR({self.sql(e.this)}))",
            exp.UnixToTime: lambda self,
            e: f"TO_DATE('1970-01-01', 'YYYY-MM-DD') + ({self.sql(e, 'this')} / 86400)",
            exp.UtcTimestamp: rename_func("UTC_TIMESTAMP"),
            exp.UtcTime: rename_func("UTC_TIME"),
        }

        PROPERTIES_LOCATION = {
            **generator.Generator.PROPERTIES_LOCATION,
            exp.VolatileProperty: exp.Properties.Location.UNSUPPORTED,
            exp.PartitionByRangeProperty: exp.Properties.Location.POST_SCHEMA,
            exp.PartitionByListProperty: exp.Properties.Location.POST_SCHEMA,
            exp.PartitionedByProperty: exp.Properties.Location.POST_SCHEMA,
        }

        def currenttimestamp_sql(self, expression: exp.CurrentTimestamp) -> str:
            if expression.args.get("sysdate"):
                return "SYSDATE"

            this = expression.this
            return self.func("CURRENT_TIMESTAMP", this) if this else "CURRENT_TIMESTAMP"

        def offset_sql(self, expression: exp.Offset) -> str:
            return f"{super().offset_sql(expression)} ROWS"

        def add_column_sql(self, expression: exp.Expression) -> str:
            return f"ADD {self.sql(expression)}"

        def queryoption_sql(self, expression: exp.QueryOption) -> str:
            option = self.sql(expression, "this")
            value = self.sql(expression, "expression")
            value = f" CONSTRAINT {value}" if value else ""

            return f"{option}{value}"

        def coalesce_sql(self, expression: exp.Coalesce) -> str:
            func_name = "NVL" if expression.args.get("is_nvl") else "COALESCE"
            return rename_func(func_name)(self, expression)

        def into_sql(self, expression: exp.Into) -> str:
            into = "INTO" if not expression.args.get("bulk_collect") else "BULK COLLECT INTO"
            if expression.this:
                return f"{self.seg(into)} {self.sql(expression, 'this')}"

            return f"{self.seg(into)} {self.expressions(expression)}"

        def hint_sql(self, expression: exp.Hint) -> str:
            expressions = []

            for expression in expression.expressions:
                if isinstance(expression, exp.Anonymous):
                    formatted_args = self.format_args(*expression.expressions, sep=" ")
                    expressions.append(f"{self.sql(expression, 'this')}({formatted_args})")
                else:
                    expressions.append(self.sql(expression))

            return f" /*+ {self.expressions(sqls=expressions, sep=self.QUERY_HINT_SEP).strip()} */"

        def isascii_sql(self, expression: exp.IsAscii) -> str:
            return f"NVL(REGEXP_LIKE({self.sql(expression.this)}, '^[' || CHR(1) || '-' || CHR(127) || ']*$'), TRUE)"


        def columnconstraint_sql(self, expression: exp.ColumnConstraint) -> str:
            this = self.sql(expression, "this")
            kind_sql = self.sql(expression, "kind").strip()
            
            # 检查是否有 ENABLE 属性
            enabled = getattr(expression.kind, 'args', {}).get('enabled', False)
            enable_sql = " ENABLE" if enabled else ""
            
            return f"CONSTRAINT {this} {kind_sql}{enable_sql}" if this else f"{kind_sql}{enable_sql}"

        def partitionbyrangeproperty_sql(self, expression: exp.PartitionByRangeProperty) -> str:
            """Generate Oracle PARTITION BY RANGE syntax."""
            partition_expressions = self.sql(expression, "partition_expressions")
            create_expressions = self.sql(expression, "create_expressions")
            return f"PARTITION BY RANGE {self.wrap(partition_expressions)} {self.wrap(create_expressions)}"

        def partitionbylistproperty_sql(self, expression: exp.PartitionByListProperty) -> str:
            """Generate Oracle PARTITION BY LIST syntax."""
            partition_expressions = self.sql(expression, "partition_expressions")
            create_expressions = self.sql(expression, "create_expressions")
            return f"PARTITION BY LIST {self.wrap(partition_expressions)} {self.wrap(create_expressions)}"

        def partitionlist_sql(self, expression: exp.PartitionList) -> str:
            """Generate Oracle partition definition syntax."""
            name = self.sql(expression, "this")
            values = expression.args.get("expressions")

            if values:
                return f"PARTITION {name} VALUES {self.wrap(values)}"
            else:
                return f"PARTITION {name}"
