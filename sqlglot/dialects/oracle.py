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
            # Oracle storage parameters
            "PCTFREE": TokenType.PCTFREE,
            "PCTUSED": TokenType.PCTUSED,
            "INITRANS": TokenType.INITRANS,
            "MAXTRANS": TokenType.MAXTRANS,
            "SEGMENT": TokenType.SEGMENT,
            "CREATION": TokenType.CREATION,
            "IMMEDIATE": TokenType.IMMEDIATE,
            "DEFERRED": TokenType.DEFERRED,
            "NOCOMPRESS": TokenType.NOCOMPRESS,
            "COMPRESS": TokenType.COMPRESS,
            "LOGGING": TokenType.LOGGING,
            "NOLOGGING": TokenType.NOLOGGING,
            "FREELISTS": TokenType.FREELISTS,
            "FREELIST": TokenType.FREELIST,
            "GROUPS": TokenType.GROUPS,
            "BUFFER_POOL": TokenType.BUFFER_POOL,
            "FLASH_CACHE": TokenType.FLASH_CACHE,
            "CELL_FLASH_CACHE": TokenType.CELL_FLASH_CACHE,
            "DEFAULT": TokenType.DEFAULT,
            "KEEP": TokenType.KEEP,
            "RECYCLE": TokenType.RECYCLE,
            "NONE": TokenType.NONE,
            "COMPUTE": TokenType.COMPUTE,
            "STATISTICS": TokenType.STATISTICS,
            "TABLESPACE": TokenType.TABLESPACE,
        }

    class Parser(parser.Parser):
        WINDOW_BEFORE_PAREN_TOKENS = {TokenType.OVER, TokenType.KEEP}
        VALUES_FOLLOWED_BY_PAREN = False

        FUNCTIONS = {
            **parser.Parser.FUNCTIONS,
            "CONVERT": exp.ConvertToCharset.from_arg_list,
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
            # Oracle storage parameters
            "SEGMENT": lambda self: self._parse_segment_creation(),
            "PCTFREE": lambda self: self._parse_pctfree(),
            "PCTUSED": lambda self: self._parse_pctused(),
            "INITRANS": lambda self: self._parse_inittrans(),
            "MAXTRANS": lambda self: self._parse_maxtrans(),
            "NOCOMPRESS": lambda self: self.expression(exp.CompressProperty, this="NOCOMPRESS"),
            "COMPRESS": lambda self: self.expression(exp.CompressProperty, this="COMPRESS"),
            "LOGGING": lambda self: self.expression(exp.LoggingProperty, this="LOGGING"),
            "NOLOGGING": lambda self: self.expression(exp.LoggingProperty, this="NOLOGGING"),
            "STORAGE": lambda self: self._parse_storage(),
            "FREELISTS": lambda self: self._parse_freelists(),
            "FREELIST": lambda self: self._parse_freelist_groups(),
            "BUFFER_POOL": lambda self: self._parse_buffer_pool(),
            "FLASH_CACHE": lambda self: self._parse_flash_cache(),
            "CELL_FLASH_CACHE": lambda self: self._parse_cell_flash_cache(),
            "COMPUTE": lambda self: self._parse_compute_statistics(),
            "TABLESPACE": lambda self: self._parse_tablespace(),
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

        def _parse_segment_creation(self) -> exp.SegmentCreationProperty:
            """Parse SEGMENT CREATION IMMEDIATE/DEFERRED"""
            self._match_text_seq("CREATION")
            creation_type = "IMMEDIATE"
            if self._match(TokenType.IMMEDIATE):
                creation_type = "IMMEDIATE"
            elif self._match(TokenType.DEFERRED):
                creation_type = "DEFERRED"
            return self.expression(exp.SegmentCreationProperty, this=creation_type)

        def _parse_pctfree(self) -> exp.PctFreeProperty:
            """Parse PCTFREE value"""
            value = self._parse_number()
            return self.expression(exp.PctFreeProperty, this=value)

        def _parse_pctused(self) -> exp.PctUsedProperty:
            """Parse PCTUSED value"""
            value = self._parse_number()
            return self.expression(exp.PctUsedProperty, this=value)

        def _parse_inittrans(self) -> exp.InitTransProperty:
            """Parse INITRANS value"""
            value = self._parse_number()
            return self.expression(exp.InitTransProperty, this=value)

        def _parse_maxtrans(self) -> exp.MaxTransProperty:
            """Parse MAXTRANS value"""
            value = self._parse_number()
            return self.expression(exp.MaxTransProperty, this=value)

        def _parse_storage(self) -> exp.StorageProperty:
            """Parse STORAGE clause with parameters"""
            self._match(TokenType.L_PAREN)
            storage_params = []
            
            # Parse storage parameters like INITIAL, NEXT, MINEXTENTS, etc.
            while not self._match(TokenType.R_PAREN):
                if self._match_text_seq("INITIAL"):
                    value = self._parse_number()
                    storage_params.append(("INITIAL", value))
                elif self._match_text_seq("NEXT"):
                    value = self._parse_number()
                    storage_params.append(("NEXT", value))
                elif self._match_text_seq("MINEXTENTS"):
                    value = self._parse_number()
                    storage_params.append(("MINEXTENTS", value))
                elif self._match_text_seq("MAXEXTENTS"):
                    value = self._parse_number()
                    storage_params.append(("MAXEXTENTS", value))
                elif self._match_text_seq("PCTINCREASE"):
                    value = self._parse_number()
                    storage_params.append(("PCTINCREASE", value))
                elif self._match_text_seq("FREELISTS"):
                    value = self._parse_number()
                    storage_params.append(("FREELISTS", value))
                elif self._match_text_seq("FREELIST", "GROUPS"):
                    value = self._parse_number()
                    storage_params.append(("FREELIST_GROUPS", value))
                elif self._match_text_seq("BUFFER_POOL"):
                    if self._match(TokenType.DEFAULT):
                        storage_params.append(("BUFFER_POOL", "DEFAULT"))
                    elif self._match(TokenType.KEEP):
                        storage_params.append(("BUFFER_POOL", "KEEP"))
                    elif self._match(TokenType.RECYCLE):
                        storage_params.append(("BUFFER_POOL", "RECYCLE"))
                elif self._match_text_seq("FLASH_CACHE"):
                    if self._match(TokenType.DEFAULT):
                        storage_params.append(("FLASH_CACHE", "DEFAULT"))
                    elif self._match(TokenType.KEEP):
                        storage_params.append(("FLASH_CACHE", "KEEP"))
                    elif self._match(TokenType.NONE):
                        storage_params.append(("FLASH_CACHE", "NONE"))
                elif self._match_text_seq("CELL_FLASH_CACHE"):
                    if self._match(TokenType.DEFAULT):
                        storage_params.append(("CELL_FLASH_CACHE", "DEFAULT"))
                    elif self._match(TokenType.KEEP):
                        storage_params.append(("CELL_FLASH_CACHE", "KEEP"))
                    elif self._match(TokenType.NONE):
                        storage_params.append(("CELL_FLASH_CACHE", "NONE"))
                
                # Skip comma if present
                self._match(TokenType.COMMA)
            
            return self.expression(exp.StorageProperty, this=storage_params)

        def _parse_freelists(self) -> exp.FreelistsProperty:
            """Parse FREELISTS value"""
            value = self._parse_number()
            return self.expression(exp.FreelistsProperty, this=value)

        def _parse_freelist_groups(self) -> exp.FreelistGroupsProperty:
            """Parse FREELIST GROUPS value"""
            self._match_text_seq("GROUPS")
            value = self._parse_number()
            return self.expression(exp.FreelistGroupsProperty, this=value)

        def _parse_buffer_pool(self) -> exp.BufferPoolProperty:
            """Parse BUFFER_POOL type"""
            if self._match(TokenType.DEFAULT):
                return self.expression(exp.BufferPoolProperty, this="DEFAULT")
            elif self._match(TokenType.KEEP):
                return self.expression(exp.BufferPoolProperty, this="KEEP")
            elif self._match(TokenType.RECYCLE):
                return self.expression(exp.BufferPoolProperty, this="RECYCLE")
            return self.expression(exp.BufferPoolProperty, this="DEFAULT")

        def _parse_flash_cache(self) -> exp.FlashCacheProperty:
            """Parse FLASH_CACHE type"""
            if self._match(TokenType.DEFAULT):
                return self.expression(exp.FlashCacheProperty, this="DEFAULT")
            elif self._match(TokenType.KEEP):
                return self.expression(exp.FlashCacheProperty, this="KEEP")
            elif self._match(TokenType.NONE):
                return self.expression(exp.FlashCacheProperty, this="NONE")
            return self.expression(exp.FlashCacheProperty, this="DEFAULT")

        def _parse_cell_flash_cache(self) -> exp.CellFlashCacheProperty:
            """Parse CELL_FLASH_CACHE type"""
            if self._match(TokenType.DEFAULT):
                return self.expression(exp.CellFlashCacheProperty, this="DEFAULT")
            elif self._match(TokenType.KEEP):
                return self.expression(exp.CellFlashCacheProperty, this="KEEP")
            elif self._match(TokenType.NONE):
                return self.expression(exp.CellFlashCacheProperty, this="NONE")
            return self.expression(exp.CellFlashCacheProperty, this="DEFAULT")

        def _parse_compute_statistics(self) -> exp.ComputeStatisticsProperty:
            """Parse COMPUTE STATISTICS"""
            self._match_text_seq("STATISTICS")
            return self.expression(exp.ComputeStatisticsProperty, this="COMPUTE STATISTICS")

        def _parse_tablespace(self) -> exp.TablespaceProperty:
            """Parse TABLESPACE name"""
            tablespace_name = self._parse_id_var()
            return self.expression(exp.TablespaceProperty, this=tablespace_name)

        def _parse_index_params(self) -> exp.IndexParameters:
            """Override to handle Oracle-specific index parameters"""
            # Parse standard index parameters first
            using = self._parse_var(any_token=True) if self._match(TokenType.USING) else None

            # Parse columns
            if self._match(TokenType.L_PAREN, advance=False):
                columns = self._parse_wrapped_csv(self._parse_with_operator)
            else:
                columns = None

            include = self._parse_wrapped_id_vars() if self._match_text_seq("INCLUDE") else None
            partition_by = self._parse_partition_by()
            
            # Parse Oracle-specific index storage parameters
            oracle_storage_props = []
            
            # Parse PCTFREE
            if self._match(TokenType.PCTFREE):
                value = self._parse_number()
                oracle_storage_props.append(self.expression(exp.PctFreeProperty, this=value))
            
            # Parse INITRANS
            if self._match(TokenType.INITRANS):
                value = self._parse_number()
                oracle_storage_props.append(self.expression(exp.InitTransProperty, this=value))
            
            # Parse MAXTRANS
            if self._match(TokenType.MAXTRANS):
                value = self._parse_number()
                oracle_storage_props.append(self.expression(exp.MaxTransProperty, this=value))
            
            # Parse COMPUTE STATISTICS
            if self._match(TokenType.COMPUTE):
                self._match_text_seq("STATISTICS")
                oracle_storage_props.append(self.expression(exp.ComputeStatisticsProperty, this="COMPUTE STATISTICS"))
            
            # Parse STORAGE clause
            if self._match_text_seq("STORAGE"):
                storage_prop = self._parse_storage()
                oracle_storage_props.append(storage_prop)
            
            # Parse TABLESPACE
            tablespace = None
            if self._match(TokenType.TABLESPACE):
                tablespace_name = self._parse_id_var()
                tablespace = tablespace_name
                oracle_storage_props.append(self.expression(exp.TablespaceProperty, this=tablespace_name))
            
            # Convert Oracle storage properties to with_storage format
            with_storage = oracle_storage_props if oracle_storage_props else None
            
            where = self._parse_where()
            on = self._parse_field() if self._match(TokenType.ON) else None

            return self.expression(
                exp.IndexParameters,
                using=using,
                columns=columns,
                include=include,
                partition_by=partition_by,
                where=where,
                with_storage=with_storage,
                tablespace=tablespace,
                on=on,
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
            exp.Group: transforms.preprocess([transforms.unalias_group]),
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
            # Oracle storage properties
            exp.SegmentCreationProperty: lambda self, e: f"SEGMENT CREATION {self.sql(e, 'this')}",
            exp.PctFreeProperty: lambda self, e: f"PCTFREE {self.sql(e, 'this')}",
            exp.PctUsedProperty: lambda self, e: f"PCTUSED {self.sql(e, 'this')}",
            exp.InitTransProperty: lambda self, e: f"INITRANS {self.sql(e, 'this')}",
            exp.MaxTransProperty: lambda self, e: f"MAXTRANS {self.sql(e, 'this')}",
            exp.CompressProperty: lambda self, e: self.sql(e, 'this'),
            exp.LoggingProperty: lambda self, e: self.sql(e, 'this'),
            exp.StorageProperty: lambda self, e: self._storage_sql(e),
            exp.FreelistsProperty: lambda self, e: f"FREELISTS {self.sql(e, 'this')}",
            exp.FreelistGroupsProperty: lambda self, e: f"FREELIST GROUPS {self.sql(e, 'this')}",
            exp.BufferPoolProperty: lambda self, e: f"BUFFER_POOL {self.sql(e, 'this')}",
            exp.FlashCacheProperty: lambda self, e: f"FLASH_CACHE {self.sql(e, 'this')}",
            exp.CellFlashCacheProperty: lambda self, e: f"CELL_FLASH_CACHE {self.sql(e, 'this')}",
            exp.ComputeStatisticsProperty: lambda self, e: self.sql(e, 'this'),
            exp.TablespaceProperty: lambda self, e: f"TABLESPACE {self.sql(e, 'this')}",
        }

        PROPERTIES_LOCATION = {
            **generator.Generator.PROPERTIES_LOCATION,
            exp.VolatileProperty: exp.Properties.Location.UNSUPPORTED,
            # Oracle storage properties
            exp.SegmentCreationProperty: exp.Properties.Location.POST_SCHEMA,
            exp.PctFreeProperty: exp.Properties.Location.POST_SCHEMA,
            exp.PctUsedProperty: exp.Properties.Location.POST_SCHEMA,
            exp.InitTransProperty: exp.Properties.Location.POST_SCHEMA,
            exp.MaxTransProperty: exp.Properties.Location.POST_SCHEMA,
            exp.CompressProperty: exp.Properties.Location.POST_SCHEMA,
            exp.LoggingProperty: exp.Properties.Location.POST_SCHEMA,
            exp.StorageProperty: exp.Properties.Location.POST_SCHEMA,
            exp.FreelistsProperty: exp.Properties.Location.POST_SCHEMA,
            exp.FreelistGroupsProperty: exp.Properties.Location.POST_SCHEMA,
            exp.BufferPoolProperty: exp.Properties.Location.POST_SCHEMA,
            exp.FlashCacheProperty: exp.Properties.Location.POST_SCHEMA,
            exp.CellFlashCacheProperty: exp.Properties.Location.POST_SCHEMA,
            exp.ComputeStatisticsProperty: exp.Properties.Location.POST_SCHEMA,
            exp.TablespaceProperty: exp.Properties.Location.POST_SCHEMA,
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

        def _storage_sql(self, expression: exp.StorageProperty) -> str:
            """Generate SQL for Oracle STORAGE clause"""
            storage_params = expression.this
            if not storage_params:
                return "STORAGE()"
            
            param_parts = []
            for param_name, param_value in storage_params:
                if param_name == "FREELIST_GROUPS":
                    param_parts.append(f"FREELIST GROUPS {param_value}")
                else:
                    param_parts.append(f"{param_name} {param_value}")
            
            return f"STORAGE({', '.join(param_parts)})"

        def indexparameters_sql(self, expression: exp.IndexParameters) -> str:
            """Generate SQL for Oracle IndexParameters"""
            parts = []
            
            # Handle columns
            if expression.args.get("columns"):
                columns_sql = self.expressions(expression, key="columns")
                parts.append(f"({columns_sql})")
            
            # Handle Oracle-specific storage parameters (not in WITH clause)
            if expression.args.get("with_storage"):
                storage_props = expression.args["with_storage"]
                for prop in storage_props:
                    if isinstance(prop, exp.TablespaceProperty):
                        # Skip tablespace here, it will be handled separately
                        continue
                    parts.append(self.sql(prop))
            
            # Handle tablespace separately
            if expression.args.get("tablespace"):
                parts.append(f"TABLESPACE {self.sql(expression, 'tablespace')}")
            
            return " ".join(parts)

        def columnconstraint_sql(self, expression: exp.ColumnConstraint) -> str:
            this = self.sql(expression, "this")
            kind_sql = self.sql(expression, "kind").strip()
            
            # 检查是否有 ENABLE 属性
            enabled = getattr(expression.kind, 'args', {}).get('enabled', False)
            enable_sql = " ENABLE" if enabled else ""
            
            return f"CONSTRAINT {this} {kind_sql}{enable_sql}" if this else f"{kind_sql}{enable_sql}"
