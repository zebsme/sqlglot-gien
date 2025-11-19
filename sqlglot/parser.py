from __future__ import annotations

import itertools
import logging
import re
import typing as t
from collections import defaultdict

from sqlglot import exp
from sqlglot.errors import ErrorLevel, ParseError, TokenError, concat_messages, merge_errors
from sqlglot.helper import apply_index_offset, ensure_list, seq_get
from sqlglot.time import format_time
from sqlglot.tokens import Token, Tokenizer, TokenType
from sqlglot.trie import TrieResult, in_trie, new_trie

if t.TYPE_CHECKING:
    from sqlglot._typing import E, Lit
    from sqlglot.dialects.dialect import Dialect, DialectType

    T = t.TypeVar("T")
    TCeilFloor = t.TypeVar("TCeilFloor", exp.Ceil, exp.Floor)

logger = logging.getLogger("sqlglot")

OPTIONS_TYPE = t.Dict[str, t.Sequence[t.Union[t.Sequence[str], str]]]

# Used to detect alphabetical characters and +/- in timestamp literals
TIME_ZONE_RE: t.Pattern[str] = re.compile(r":.*?[a-zA-Z\+\-]")


def build_var_map(args: t.List) -> exp.StarMap | exp.VarMap:
    """
    构建变量映射表达式

    根据参数列表构建星号映射或变量映射表达式。
    如果只有一个星号参数，返回StarMap；否则返回VarMap，包含键值对数组。

    Args:
        args: 参数列表，可以是星号表达式或键值对

    Returns:
        exp.StarMap | exp.VarMap: 星号映射或变量映射表达式
    """
    if len(args) == 1 and args[0].is_star:
        return exp.StarMap(this=args[0])

    keys = []
    values = []
    for i in range(0, len(args), 2):
        keys.append(args[i])
        values.append(args[i + 1])

    return exp.VarMap(keys=exp.array(*keys, copy=False), values=exp.array(*values, copy=False))


def build_like(args: t.List) -> exp.Escape | exp.Like:
    """
    构建LIKE表达式，支持可选的ESCAPE子句

    创建LIKE比较表达式，如果提供了转义字符，则包装在ESCAPE表达式中。

    Args:
        args: 参数列表，[表达式, 模式, 转义字符(可选)]

    Returns:
        exp.Escape | exp.Like: 转义表达式或LIKE表达式
    """
    like = exp.Like(this=seq_get(args, 1), expression=seq_get(args, 0))
    return exp.Escape(this=like, expression=seq_get(args, 2)) if len(args) > 2 else like


def binary_range_parser(
    expr_type: t.Type[exp.Expression], reverse_args: bool = False
) -> t.Callable[[Parser, t.Optional[exp.Expression]], t.Optional[exp.Expression]]:
    """
    创建二元范围解析器的工厂函数

    返回一个解析器函数，用于解析二元范围表达式（如BETWEEN、IN等）。
    可以配置参数顺序是否反转。

    Args:
        expr_type: 要创建的表达式类型
        reverse_args: 是否反转参数顺序

    Returns:
        解析器函数，接受Parser实例和可选的表达式，返回解析后的表达式
    """
    def _parse_binary_range(
        self: Parser, this: t.Optional[exp.Expression]
    ) -> t.Optional[exp.Expression]:
        """
        解析二元范围表达式的内部函数

        Args:
            self: Parser实例
            this: 当前表达式

        Returns:
            解析后的二元范围表达式
        """
        expression = self._parse_bitwise()
        if reverse_args:
            this, expression = expression, this
        return self._parse_escape(self.expression(expr_type, this=this, expression=expression))

    return _parse_binary_range


def build_logarithm(args: t.List, dialect: Dialect) -> exp.Func:
    """
    构建对数函数表达式

    根据参数数量和数据库方言构建对数函数。支持不同方言的参数顺序。

    Args:
        args: 参数列表
        dialect: 数据库方言对象

    Returns:
        exp.Func: 对数函数表达式（Log或Ln）
    """
    # Default argument order is base, expression
    this = seq_get(args, 0)
    expression = seq_get(args, 1)

    if expression:
        if not dialect.LOG_BASE_FIRST:
            this, expression = expression, this
        return exp.Log(this=this, expression=expression)

    return (exp.Ln if dialect.parser_class.LOG_DEFAULTS_TO_LN else exp.Log)(this=this)


def build_hex(args: t.List, dialect: Dialect) -> exp.Hex | exp.LowerHex:
    """
    构建十六进制函数表达式

    根据数据库方言返回大写或小写的十六进制表达式。

    Args:
        args: 参数列表
        dialect: 数据库方言对象

    Returns:
        exp.Hex | exp.LowerHex: 十六进制或小写十六进制表达式
    """
    arg = seq_get(args, 0)
    return exp.LowerHex(this=arg) if dialect.HEX_LOWERCASE else exp.Hex(this=arg)


def build_lower(args: t.List) -> exp.Lower | exp.Hex:
    """
    构建LOWER函数表达式，支持LOWER(HEX(..))的简化

    如果参数是HEX表达式，则简化为LowerHex以简化转译过程。

    Args:
        args: 参数列表

    Returns:
        exp.Lower | exp.Hex: LOWER表达式或十六进制表达式
    """
    # LOWER(HEX(..)) can be simplified to LowerHex to simplify its transpilation
    arg = seq_get(args, 0)
    return exp.LowerHex(this=arg.this) if isinstance(arg, exp.Hex) else exp.Lower(this=arg)


def build_upper(args: t.List) -> exp.Upper | exp.Hex:
    """
    构建UPPER函数表达式，支持UPPER(HEX(..))的简化

    如果参数是HEX表达式，则简化为Hex以简化转译过程。

    Args:
        args: 参数列表

    Returns:
        exp.Upper | exp.Hex: UPPER表达式或十六进制表达式
    """
    # UPPER(HEX(..)) can be simplified to Hex to simplify its transpilation
    arg = seq_get(args, 0)
    return exp.Hex(this=arg.this) if isinstance(arg, exp.Hex) else exp.Upper(this=arg)


def build_extract_json_with_path(expr_type: t.Type[E]) -> t.Callable[[t.List, Dialect], E]:
    """
    创建JSON路径提取函数的构建器

    返回一个构建器函数，用于创建JSON路径提取表达式。

    Args:
        expr_type: 要创建的表达式类型

    Returns:
        构建器函数，接受参数列表和方言，返回指定类型的表达式
    """
    def _builder(args: t.List, dialect: Dialect) -> E:
        """
        构建JSON路径提取表达式的内部函数

        Args:
            args: 参数列表
            dialect: 数据库方言对象

        Returns:
            指定类型的JSON路径提取表达式
        """
        expression = expr_type(
            this=seq_get(args, 0), expression=dialect.to_json_path(seq_get(args, 1))
        )
        if len(args) > 2 and expr_type is exp.JSONExtract:
            expression.set("expressions", args[2:])

        return expression

    return _builder


def build_mod(args: t.List) -> exp.Mod:
    """
    构建取模函数表达式

    创建取模运算表达式，如果操作数是二元节点则用括号包装。

    Args:
        args: 参数列表，包含被除数和除数

    Returns:
        exp.Mod: 取模表达式
    """
    this = seq_get(args, 0)
    expression = seq_get(args, 1)

    # Wrap the operands if they are binary nodes, e.g. MOD(a + 1, 7) -> (a + 1) % 7
    this = exp.Paren(this=this) if isinstance(this, exp.Binary) else this
    expression = exp.Paren(this=expression) if isinstance(expression, exp.Binary) else expression

    return exp.Mod(this=this, expression=expression)


def build_pad(args: t.List, is_left: bool = True):
    """
    构建填充函数表达式（LPAD或RPAD）

    创建左填充或右填充表达式，用于在字符串前后添加填充字符。

    Args:
        args: 参数列表，[字符串, 长度, 填充模式]
        is_left: 是否为左填充（LPAD）

    Returns:
        exp.Pad: 填充表达式
    """
    return exp.Pad(
        this=seq_get(args, 0),
        expression=seq_get(args, 1),
        fill_pattern=seq_get(args, 2),
        is_left=is_left,
    )


def build_array_constructor(
    exp_class: t.Type[E], args: t.List, bracket_kind: TokenType, dialect: Dialect
) -> exp.Expression:
    """
    构建数组构造函数表达式

    根据表达式类和数据库方言创建数组构造函数。

    Args:
        exp_class: 表达式类
        args: 参数列表
        bracket_kind: 括号类型
        dialect: 数据库方言对象

    Returns:
        exp.Expression: 数组构造函数表达式
    """
    array_exp = exp_class(expressions=args)

    if exp_class == exp.Array and dialect.HAS_DISTINCT_ARRAY_CONSTRUCTORS:
        array_exp.set("bracket_notation", bracket_kind == TokenType.L_BRACKET)

    return array_exp


def build_convert_timezone(
    args: t.List, default_source_tz: t.Optional[str] = None
) -> t.Union[exp.ConvertTimezone, exp.Anonymous]:
    """
    构建时区转换函数表达式

    创建时区转换表达式，支持不同的参数组合。

    Args:
        args: 参数列表
        default_source_tz: 默认源时区

    Returns:
        exp.ConvertTimezone | exp.Anonymous: 时区转换表达式或匿名表达式
    """
    if len(args) == 2:
        source_tz = exp.Literal.string(default_source_tz) if default_source_tz else None
        return exp.ConvertTimezone(
            source_tz=source_tz, target_tz=seq_get(args, 0), timestamp=seq_get(args, 1)
        )

    return exp.ConvertTimezone.from_arg_list(args)


def build_trim(args: t.List, is_left: bool = True):
    """
    构建TRIM函数表达式

    创建字符串修剪表达式，支持前导或尾随字符的修剪。

    Args:
        args: 参数列表，[字符串, 要修剪的字符]
        is_left: 是否为前导修剪

    Returns:
        exp.Trim: TRIM表达式
    """
    return exp.Trim(
        this=seq_get(args, 0),
        expression=seq_get(args, 1),
        position="LEADING" if is_left else "TRAILING",
    )


def build_coalesce(
    args: t.List, is_nvl: t.Optional[bool] = None, is_null: t.Optional[bool] = None
) -> exp.Coalesce:
    """
    构建COALESCE函数表达式

    创建COALESCE表达式，支持NVL和NULL函数的变体。

    Args:
        args: 参数列表
        is_nvl: 是否为NVL函数
        is_null: 是否为NULL函数

    Returns:
        exp.Coalesce: COALESCE表达式
    """
    return exp.Coalesce(this=seq_get(args, 0), expressions=args[1:], is_nvl=is_nvl, is_null=is_null)


def build_locate_strposition(args: t.List):
    """
    构建字符串位置查找函数表达式

    创建字符串位置查找表达式，用于查找子字符串在目标字符串中的位置。

    Args:
        args: 参数列表，[子字符串, 目标字符串, 起始位置(可选)]

    Returns:
        exp.StrPosition: 字符串位置表达式
    """
    return exp.StrPosition(
        this=seq_get(args, 1),
        substr=seq_get(args, 0),
        position=seq_get(args, 2),
    )


class _Parser(type):
    """
    Parser类的元类

    在类创建时自动构建SHOW_TRIE和SET_TRIE字典，用于快速查找解析器。
    """
    def __new__(cls, clsname, bases, attrs):
        """
        创建Parser类时的钩子函数

        Args:
            cls: 元类
            clsname: 类名
            bases: 基类
            attrs: 类属性

        Returns:
            创建的类
        """
        klass = super().__new__(cls, clsname, bases, attrs)

        # 构建SHOW和SET命令的Trie树，用于快速查找解析器
        klass.SHOW_TRIE = new_trie(key.split(" ") for key in klass.SHOW_PARSERS)
        klass.SET_TRIE = new_trie(key.split(" ") for key in klass.SET_PARSERS)

        return klass


class Parser(metaclass=_Parser):
    """
    Parser consumes a list of tokens produced by the Tokenizer and produces a parsed syntax tree.

    Args:
        error_level: The desired error level.
            Default: ErrorLevel.IMMEDIATE
        error_message_context: The amount of context to capture from a query string when displaying
            the error message (in number of characters).
            Default: 100
        max_errors: Maximum number of error messages to include in a raised ParseError.
            This is only relevant if error_level is ErrorLevel.RAISE.
            Default: 3
    """

    FUNCTIONS: t.Dict[str, t.Callable] = {
        **{name: func.from_arg_list for name, func in exp.FUNCTION_BY_NAME.items()},
        **dict.fromkeys(("COALESCE", "IFNULL", "NVL"), build_coalesce),
        "ARRAY": lambda args, dialect: exp.Array(expressions=args),
        "ARRAYAGG": lambda args, dialect: exp.ArrayAgg(
            this=seq_get(args, 0), nulls_excluded=dialect.ARRAY_AGG_INCLUDES_NULLS is None or None
        ),
        "ARRAY_AGG": lambda args, dialect: exp.ArrayAgg(
            this=seq_get(args, 0), nulls_excluded=dialect.ARRAY_AGG_INCLUDES_NULLS is None or None
        ),
        "CHAR": lambda args: exp.Chr(expressions=args),
        "CHR": lambda args: exp.Chr(expressions=args),
        "COUNT": lambda args: exp.Count(this=seq_get(args, 0), expressions=args[1:], big_int=True),
        "CONCAT": lambda args, dialect: exp.Concat(
            expressions=args,
            safe=not dialect.STRICT_STRING_CONCAT,
            coalesce=dialect.CONCAT_COALESCE,
        ),
        "CONCAT_WS": lambda args, dialect: exp.ConcatWs(
            expressions=args,
            safe=not dialect.STRICT_STRING_CONCAT,
            coalesce=dialect.CONCAT_COALESCE,
        ),
        "CONVERT_TIMEZONE": build_convert_timezone,
        "DATE_TO_DATE_STR": lambda args: exp.Cast(
            this=seq_get(args, 0),
            to=exp.DataType(this=exp.DataType.Type.TEXT),
        ),
        "GENERATE_DATE_ARRAY": lambda args: exp.GenerateDateArray(
            start=seq_get(args, 0),
            end=seq_get(args, 1),
            step=seq_get(args, 2) or exp.Interval(this=exp.Literal.string(1), unit=exp.var("DAY")),
        ),
        "GLOB": lambda args: exp.Glob(this=seq_get(args, 1), expression=seq_get(args, 0)),
        "HEX": build_hex,
        "JSON_EXTRACT": build_extract_json_with_path(exp.JSONExtract),
        "JSON_EXTRACT_SCALAR": build_extract_json_with_path(exp.JSONExtractScalar),
        "JSON_EXTRACT_PATH_TEXT": build_extract_json_with_path(exp.JSONExtractScalar),
        "LIKE": build_like,
        "LOG": build_logarithm,
        "LOG2": lambda args: exp.Log(this=exp.Literal.number(2), expression=seq_get(args, 0)),
        "LOG10": lambda args: exp.Log(this=exp.Literal.number(10), expression=seq_get(args, 0)),
        "LOWER": build_lower,
        "LPAD": lambda args: build_pad(args),
        "LEFTPAD": lambda args: build_pad(args),
        "LTRIM": lambda args: build_trim(args),
        "MOD": build_mod,
        "RIGHTPAD": lambda args: build_pad(args, is_left=False),
        "RPAD": lambda args: build_pad(args, is_left=False),
        "RTRIM": lambda args: build_trim(args, is_left=False),
        "SCOPE_RESOLUTION": lambda args: exp.ScopeResolution(expression=seq_get(args, 0))
        if len(args) != 2
        else exp.ScopeResolution(this=seq_get(args, 0), expression=seq_get(args, 1)),
        "STRPOS": exp.StrPosition.from_arg_list,
        "CHARINDEX": lambda args: build_locate_strposition(args),
        "INSTR": exp.StrPosition.from_arg_list,
        "LOCATE": lambda args: build_locate_strposition(args),
        "TIME_TO_TIME_STR": lambda args: exp.Cast(
            this=seq_get(args, 0),
            to=exp.DataType(this=exp.DataType.Type.TEXT),
        ),
        "TO_HEX": build_hex,
        "TS_OR_DS_TO_DATE_STR": lambda args: exp.Substring(
            this=exp.Cast(
                this=seq_get(args, 0),
                to=exp.DataType(this=exp.DataType.Type.TEXT),
            ),
            start=exp.Literal.number(1),
            length=exp.Literal.number(10),
        ),
        "UNNEST": lambda args: exp.Unnest(expressions=ensure_list(seq_get(args, 0))),
        "UPPER": build_upper,
        "VAR_MAP": build_var_map,
    }

    NO_PAREN_FUNCTIONS = {
        TokenType.CURRENT_DATE: exp.CurrentDate,
        TokenType.CURRENT_DATETIME: exp.CurrentDate,
        TokenType.CURRENT_TIME: exp.CurrentTime,
        TokenType.CURRENT_TIMESTAMP: exp.CurrentTimestamp,
        TokenType.CURRENT_USER: exp.CurrentUser,
    }

    STRUCT_TYPE_TOKENS = {
        TokenType.NESTED,
        TokenType.OBJECT,
        TokenType.STRUCT,
        TokenType.UNION,
    }

    NESTED_TYPE_TOKENS = {
        TokenType.ARRAY,
        TokenType.LIST,
        TokenType.LOWCARDINALITY,
        TokenType.MAP,
        TokenType.NULLABLE,
        TokenType.RANGE,
        *STRUCT_TYPE_TOKENS,
    }

    ENUM_TYPE_TOKENS = {
        TokenType.DYNAMIC,
        TokenType.ENUM,
        TokenType.ENUM8,
        TokenType.ENUM16,
    }

    AGGREGATE_TYPE_TOKENS = {
        TokenType.AGGREGATEFUNCTION,
        TokenType.SIMPLEAGGREGATEFUNCTION,
    }

    TYPE_TOKENS = {
        TokenType.BIT,
        TokenType.BOOLEAN,
        TokenType.TINYINT,
        TokenType.UTINYINT,
        TokenType.SMALLINT,
        TokenType.USMALLINT,
        TokenType.INT,
        TokenType.UINT,
        TokenType.BIGINT,
        TokenType.UBIGINT,
        TokenType.INT128,
        TokenType.UINT128,
        TokenType.INT256,
        TokenType.UINT256,
        TokenType.MEDIUMINT,
        TokenType.UMEDIUMINT,
        TokenType.FIXEDSTRING,
        TokenType.FLOAT,
        TokenType.DOUBLE,
        TokenType.UDOUBLE,
        TokenType.CHAR,
        TokenType.NCHAR,
        TokenType.VARCHAR,
        TokenType.NVARCHAR,
        TokenType.BPCHAR,
        TokenType.TEXT,
        TokenType.MEDIUMTEXT,
        TokenType.LONGTEXT,
        TokenType.BLOB,
        TokenType.MEDIUMBLOB,
        TokenType.LONGBLOB,
        TokenType.BINARY,
        TokenType.VARBINARY,
        TokenType.JSON,
        TokenType.JSONB,
        TokenType.INTERVAL,
        TokenType.TINYBLOB,
        TokenType.TINYTEXT,
        TokenType.TIME,
        TokenType.TIMETZ,
        TokenType.TIMESTAMP,
        TokenType.TIMESTAMP_S,
        TokenType.TIMESTAMP_MS,
        TokenType.TIMESTAMP_NS,
        TokenType.TIMESTAMPTZ,
        TokenType.TIMESTAMPLTZ,
        TokenType.TIMESTAMPNTZ,
        TokenType.DATETIME,
        TokenType.DATETIME2,
        TokenType.DATETIME64,
        TokenType.SMALLDATETIME,
        TokenType.DATE,
        TokenType.DATE32,
        TokenType.INT4RANGE,
        TokenType.INT4MULTIRANGE,
        TokenType.INT8RANGE,
        TokenType.INT8MULTIRANGE,
        TokenType.NUMRANGE,
        TokenType.NUMMULTIRANGE,
        TokenType.TSRANGE,
        TokenType.TSMULTIRANGE,
        TokenType.TSTZRANGE,
        TokenType.TSTZMULTIRANGE,
        TokenType.DATERANGE,
        TokenType.DATEMULTIRANGE,
        TokenType.DECIMAL,
        TokenType.DECIMAL32,
        TokenType.DECIMAL64,
        TokenType.DECIMAL128,
        TokenType.DECIMAL256,
        TokenType.UDECIMAL,
        TokenType.BIGDECIMAL,
        TokenType.UUID,
        TokenType.GEOGRAPHY,
        TokenType.GEOGRAPHYPOINT,
        TokenType.GEOMETRY,
        TokenType.POINT,
        TokenType.RING,
        TokenType.LINESTRING,
        TokenType.MULTILINESTRING,
        TokenType.POLYGON,
        TokenType.MULTIPOLYGON,
        TokenType.HLLSKETCH,
        TokenType.HSTORE,
        TokenType.PSEUDO_TYPE,
        TokenType.SUPER,
        TokenType.SERIAL,
        TokenType.SMALLSERIAL,
        TokenType.BIGSERIAL,
        TokenType.XML,
        TokenType.YEAR,
        TokenType.USERDEFINED,
        TokenType.MONEY,
        TokenType.SMALLMONEY,
        TokenType.ROWVERSION,
        TokenType.IMAGE,
        TokenType.VARIANT,
        TokenType.VECTOR,
        TokenType.VOID,
        TokenType.OBJECT,
        TokenType.OBJECT_IDENTIFIER,
        TokenType.INET,
        TokenType.IPADDRESS,
        TokenType.IPPREFIX,
        TokenType.IPV4,
        TokenType.IPV6,
        TokenType.UNKNOWN,
        TokenType.NOTHING,
        TokenType.NULL,
        TokenType.NAME,
        TokenType.TDIGEST,
        TokenType.DYNAMIC,
        TokenType.LONG_VARCHAR,
        TokenType.DECFLOAT,
        TokenType.LONG_VARGRAPHIC,
        TokenType.BYTEINT,
        # TokenType.INTEGER,
        *ENUM_TYPE_TOKENS,
        *NESTED_TYPE_TOKENS,
        *AGGREGATE_TYPE_TOKENS,
    }

    SIGNED_TO_UNSIGNED_TYPE_TOKEN = {
        TokenType.BIGINT: TokenType.UBIGINT,
        TokenType.INT: TokenType.UINT,
        TokenType.MEDIUMINT: TokenType.UMEDIUMINT,
        TokenType.SMALLINT: TokenType.USMALLINT,
        TokenType.TINYINT: TokenType.UTINYINT,
        TokenType.DECIMAL: TokenType.UDECIMAL,
        TokenType.DOUBLE: TokenType.UDOUBLE,
    }

    SUBQUERY_PREDICATES = {
        TokenType.ANY: exp.Any,
        TokenType.ALL: exp.All,
        TokenType.EXISTS: exp.Exists,
        TokenType.SOME: exp.Any,
    }

    RESERVED_TOKENS = {
        *Tokenizer.SINGLE_TOKENS.values(),
        TokenType.SELECT,
    } - {TokenType.IDENTIFIER}

    DB_CREATABLES = {
        TokenType.DATABASE,
        TokenType.DICTIONARY,
        TokenType.FILE_FORMAT,
        TokenType.MODEL,
        TokenType.NAMESPACE,
        TokenType.SCHEMA,
        TokenType.SEMANTIC_VIEW,
        TokenType.SEQUENCE,
        TokenType.SINK,
        TokenType.SOURCE,
        TokenType.STAGE,
        TokenType.STORAGE_INTEGRATION,
        TokenType.STREAMLIT,
        TokenType.TABLE,
        TokenType.TAG,
        TokenType.VIEW,
        TokenType.WAREHOUSE,
    }

    CREATABLES = {
        TokenType.COLUMN,
        TokenType.CONSTRAINT,
        TokenType.FOREIGN_KEY,
        TokenType.FUNCTION,
        TokenType.INDEX,
        TokenType.PROCEDURE,
        *DB_CREATABLES,
    }

    ALTERABLES = {
        TokenType.INDEX,
        TokenType.TABLE,
        TokenType.VIEW,
        TokenType.SESSION,
    }

    # Tokens that can represent identifiers
    ID_VAR_TOKENS = {
        TokenType.ALL,
        TokenType.ANALYZE,
        TokenType.ATTACH,
        TokenType.VAR,
        TokenType.ANTI,
        TokenType.APPLY,
        TokenType.ASC,
        TokenType.ASOF,
        TokenType.AUTO_INCREMENT,
        TokenType.BEGIN,
        TokenType.BPCHAR,
        TokenType.CACHE,
        TokenType.CASE,
        TokenType.COLLATE,
        TokenType.COMMAND,
        TokenType.COMMENT,
        TokenType.COMMIT,
        TokenType.CONSTRAINT,
        TokenType.COPY,
        TokenType.CUBE,
        TokenType.CURRENT_SCHEMA,
        TokenType.DEFAULT,
        TokenType.DELETE,
        TokenType.DESC,
        TokenType.DESCRIBE,
        TokenType.DETACH,
        TokenType.DICTIONARY,
        TokenType.DIV,
        TokenType.END,
        TokenType.EXECUTE,
        TokenType.EXPORT,
        TokenType.ESCAPE,
        TokenType.FALSE,
        TokenType.FIRST,
        TokenType.FILTER,
        TokenType.FINAL,
        TokenType.FORMAT,
        TokenType.FULL,
        TokenType.GET,
        TokenType.IDENTIFIER,
        TokenType.IS,
        TokenType.ISNULL,
        TokenType.INTERVAL,
        TokenType.KEEP,
        TokenType.KILL,
        TokenType.LEFT,
        TokenType.LIMIT,
        TokenType.LOAD,
        TokenType.LOCK,
        TokenType.MERGE,
        TokenType.NATURAL,
        TokenType.NEXT,
        TokenType.OFFSET,
        TokenType.OPERATOR,
        TokenType.ORDINALITY,
        TokenType.OVERLAPS,
        TokenType.OVERWRITE,
        TokenType.PARTITION,
        TokenType.PERCENT,
        TokenType.PIVOT,
        TokenType.PRAGMA,
        TokenType.PUT,
        TokenType.RANGE,
        TokenType.RECURSIVE,
        TokenType.REFERENCES,
        TokenType.REFRESH,
        TokenType.RENAME,
        TokenType.REPLACE,
        TokenType.RIGHT,
        TokenType.ROLLUP,
        TokenType.ROW,
        TokenType.ROWS,
        TokenType.SEMI,
        TokenType.SET,
        TokenType.SETTINGS,
        TokenType.SHOW,
        TokenType.TEMPORARY,
        TokenType.TOP,
        TokenType.TRUE,
        TokenType.TRUNCATE,
        TokenType.UNIQUE,
        TokenType.UNNEST,
        TokenType.UNPIVOT,
        TokenType.UPDATE,
        TokenType.USE,
        TokenType.VOLATILE,
        TokenType.WINDOW,
        *ALTERABLES,
        *CREATABLES,
        *SUBQUERY_PREDICATES,
        *TYPE_TOKENS,
        *NO_PAREN_FUNCTIONS,
    }
    ID_VAR_TOKENS.remove(TokenType.UNION)

    TABLE_ALIAS_TOKENS = ID_VAR_TOKENS - {
        TokenType.ANTI,
        TokenType.ASOF,
        TokenType.FULL,
        TokenType.LEFT,
        TokenType.LOCK,
        TokenType.NATURAL,
        TokenType.RIGHT,
        TokenType.SEMI,
        TokenType.WINDOW,
    }

    ALIAS_TOKENS = ID_VAR_TOKENS

    COLON_PLACEHOLDER_TOKENS = ID_VAR_TOKENS

    ARRAY_CONSTRUCTORS = {
        "ARRAY": exp.Array,
        "LIST": exp.List,
    }

    COMMENT_TABLE_ALIAS_TOKENS = TABLE_ALIAS_TOKENS - {TokenType.IS}

    UPDATE_ALIAS_TOKENS = TABLE_ALIAS_TOKENS - {TokenType.SET}

    TRIM_TYPES = {"LEADING", "TRAILING", "BOTH"}

    FUNC_TOKENS = {
        TokenType.COLLATE,
        TokenType.COMMAND,
        TokenType.CURRENT_DATE,
        TokenType.CURRENT_DATETIME,
        TokenType.CURRENT_SCHEMA,
        TokenType.CURRENT_TIMESTAMP,
        TokenType.CURRENT_TIME,
        TokenType.CURRENT_USER,
        TokenType.FILTER,
        TokenType.FIRST,
        TokenType.FORMAT,
        TokenType.GET,
        TokenType.GLOB,
        TokenType.IDENTIFIER,
        TokenType.INDEX,
        TokenType.ISNULL,
        TokenType.ILIKE,
        TokenType.INSERT,
        TokenType.LIKE,
        TokenType.MERGE,
        TokenType.NEXT,
        TokenType.OFFSET,
        TokenType.PRIMARY_KEY,
        TokenType.RANGE,
        TokenType.REPLACE,
        TokenType.RLIKE,
        TokenType.ROW,
        TokenType.UNNEST,
        TokenType.VAR,
        TokenType.LEFT,
        TokenType.RIGHT,
        TokenType.SEQUENCE,
        TokenType.DATE,
        TokenType.DATETIME,
        TokenType.TABLE,
        TokenType.TIMESTAMP,
        TokenType.TIMESTAMPTZ,
        TokenType.TRUNCATE,
        TokenType.UTC_DATE,
        TokenType.UTC_TIME,
        TokenType.UTC_TIMESTAMP,
        TokenType.WINDOW,
        TokenType.XOR,
        *TYPE_TOKENS,
        *SUBQUERY_PREDICATES,
    }

    CONJUNCTION: t.Dict[TokenType, t.Type[exp.Expression]] = {
        TokenType.AND: exp.And,
    }

    ASSIGNMENT: t.Dict[TokenType, t.Type[exp.Expression]] = {
        TokenType.COLON_EQ: exp.PropertyEQ,
    }

    DISJUNCTION: t.Dict[TokenType, t.Type[exp.Expression]] = {
        TokenType.OR: exp.Or,
    }

    EQUALITY = {
        TokenType.EQ: exp.EQ,
        TokenType.NEQ: exp.NEQ,
        TokenType.NULLSAFE_EQ: exp.NullSafeEQ,
    }

    COMPARISON = {
        TokenType.GT: exp.GT,
        TokenType.GTE: exp.GTE,
        TokenType.LT: exp.LT,
        TokenType.LTE: exp.LTE,
    }

    BITWISE = {
        TokenType.AMP: exp.BitwiseAnd,
        TokenType.CARET: exp.BitwiseXor,
        TokenType.PIPE: exp.BitwiseOr,
    }

    TERM = {
        TokenType.DASH: exp.Sub,
        TokenType.PLUS: exp.Add,
        TokenType.MOD: exp.Mod,
        TokenType.COLLATE: exp.Collate,
    }

    FACTOR = {
        TokenType.DIV: exp.IntDiv,
        TokenType.LR_ARROW: exp.Distance,
        TokenType.SLASH: exp.Div,
        TokenType.STAR: exp.Mul,
    }

    EXPONENT: t.Dict[TokenType, t.Type[exp.Expression]] = {}

    TIMES = {
        TokenType.TIME,
        TokenType.TIMETZ,
    }

    TIMESTAMPS = {
        TokenType.TIMESTAMP,
        TokenType.TIMESTAMPNTZ,
        TokenType.TIMESTAMPTZ,
        TokenType.TIMESTAMPLTZ,
        *TIMES,
    }

    SET_OPERATIONS = {
        TokenType.UNION,
        TokenType.INTERSECT,
        TokenType.EXCEPT,
    }

    JOIN_METHODS = {
        TokenType.ASOF,
        TokenType.NATURAL,
        TokenType.POSITIONAL,
    }

    JOIN_SIDES = {
        TokenType.LEFT,
        TokenType.RIGHT,
        TokenType.FULL,
    }

    JOIN_KINDS = {
        TokenType.ANTI,
        TokenType.CROSS,
        TokenType.INNER,
        TokenType.OUTER,
        TokenType.SEMI,
        TokenType.STRAIGHT_JOIN,
    }

    JOIN_HINTS: t.Set[str] = set()

    LAMBDAS = {
        TokenType.ARROW: lambda self, expressions: self.expression(
            exp.Lambda,
            this=self._replace_lambda(
                self._parse_assignment(),
                expressions,
            ),
            expressions=expressions,
        ),
        TokenType.FARROW: lambda self, expressions: self.expression(
            exp.Kwarg,
            this=exp.var(expressions[0].name),
            expression=self._parse_assignment(),
        ),
    }

    COLUMN_OPERATORS = {
        TokenType.DOT: None,
        TokenType.DOTCOLON: lambda self, this, to: self.expression(
            exp.JSONCast,
            this=this,
            to=to,
        ),
        TokenType.DCOLON: lambda self, this, to: self.build_cast(
            strict=self.STRICT_CAST, this=this, to=to
        ),
        TokenType.ARROW: lambda self, this, path: self.expression(
            exp.JSONExtract,
            this=this,
            expression=self.dialect.to_json_path(path),
            only_json_types=self.JSON_ARROWS_REQUIRE_JSON_TYPE,
        ),
        TokenType.DARROW: lambda self, this, path: self.expression(
            exp.JSONExtractScalar,
            this=this,
            expression=self.dialect.to_json_path(path),
            only_json_types=self.JSON_ARROWS_REQUIRE_JSON_TYPE,
        ),
        TokenType.HASH_ARROW: lambda self, this, path: self.expression(
            exp.JSONBExtract,
            this=this,
            expression=path,
        ),
        TokenType.DHASH_ARROW: lambda self, this, path: self.expression(
            exp.JSONBExtractScalar,
            this=this,
            expression=path,
        ),
        TokenType.PLACEHOLDER: lambda self, this, key: self.expression(
            exp.JSONBContains,
            this=this,
            expression=key,
        ),
    }

    CAST_COLUMN_OPERATORS = {
        TokenType.DOTCOLON,
        TokenType.DCOLON,
    }

    EXPRESSION_PARSERS = {
        exp.Cluster: lambda self: self._parse_sort(exp.Cluster, TokenType.CLUSTER_BY),
        exp.Column: lambda self: self._parse_column(),
        exp.ColumnDef: lambda self: self._parse_column_def(self._parse_column()),
        exp.Condition: lambda self: self._parse_assignment(),
        exp.DataType: lambda self: self._parse_types(allow_identifiers=False, schema=True),
        exp.Expression: lambda self: self._parse_expression(),
        exp.From: lambda self: self._parse_from(joins=True),
        exp.GrantPrincipal: lambda self: self._parse_grant_principal(),
        exp.GrantPrivilege: lambda self: self._parse_grant_privilege(),
        exp.Group: lambda self: self._parse_group(),
        exp.Having: lambda self: self._parse_having(),
        exp.Hint: lambda self: self._parse_hint_body(),
        exp.Identifier: lambda self: self._parse_id_var(),
        exp.Join: lambda self: self._parse_join(),
        exp.Lambda: lambda self: self._parse_lambda(),
        exp.Lateral: lambda self: self._parse_lateral(),
        exp.Limit: lambda self: self._parse_limit(),
        exp.Offset: lambda self: self._parse_offset(),
        exp.Order: lambda self: self._parse_order(),
        exp.Ordered: lambda self: self._parse_ordered(),
        exp.Properties: lambda self: self._parse_properties(),
        exp.PartitionedByProperty: lambda self: self._parse_partitioned_by(),
        exp.Qualify: lambda self: self._parse_qualify(),
        exp.Returning: lambda self: self._parse_returning(),
        exp.Select: lambda self: self._parse_select(),
        exp.Sort: lambda self: self._parse_sort(exp.Sort, TokenType.SORT_BY),
        exp.Table: lambda self: self._parse_table_parts(),
        exp.TableAlias: lambda self: self._parse_table_alias(),
        exp.Tuple: lambda self: self._parse_value(values=False),
        exp.Whens: lambda self: self._parse_when_matched(),
        exp.Where: lambda self: self._parse_where(),
        exp.Window: lambda self: self._parse_named_window(),
        exp.With: lambda self: self._parse_with(),
        "JOIN_TYPE": lambda self: self._parse_join_parts(),
    }

    STATEMENT_PARSERS = {
        TokenType.ALTER: lambda self: self._parse_alter(),
        TokenType.ANALYZE: lambda self: self._parse_analyze(),
        TokenType.BEGIN: lambda self: self._parse_transaction(),
        TokenType.CACHE: lambda self: self._parse_cache(),
        TokenType.COMMENT: lambda self: self._parse_comment(),
        TokenType.COMMIT: lambda self: self._parse_commit_or_rollback(),
        TokenType.COPY: lambda self: self._parse_copy(),
        TokenType.CREATE: lambda self: self._parse_create(),
        TokenType.DELETE: lambda self: self._parse_delete(),
        TokenType.DESC: lambda self: self._parse_describe(),
        TokenType.DESCRIBE: lambda self: self._parse_describe(),
        TokenType.DROP: lambda self: self._parse_drop(),
        TokenType.GRANT: lambda self: self._parse_grant(),
        TokenType.REVOKE: lambda self: self._parse_revoke(),
        TokenType.INSERT: lambda self: self._parse_insert(),
        TokenType.KILL: lambda self: self._parse_kill(),
        TokenType.LOAD: lambda self: self._parse_load(),
        TokenType.MERGE: lambda self: self._parse_merge(),
        TokenType.PIVOT: lambda self: self._parse_simplified_pivot(),
        TokenType.PRAGMA: lambda self: self.expression(exp.Pragma, this=self._parse_expression()),
        TokenType.REFRESH: lambda self: self._parse_refresh(),
        TokenType.ROLLBACK: lambda self: self._parse_commit_or_rollback(),
        TokenType.SET: lambda self: self._parse_set(),
        TokenType.TRUNCATE: lambda self: self._parse_truncate_table(),
        TokenType.UNCACHE: lambda self: self._parse_uncache(),
        TokenType.UNPIVOT: lambda self: self._parse_simplified_pivot(is_unpivot=True),
        TokenType.UPDATE: lambda self: self._parse_update(),
        TokenType.USE: lambda self: self._parse_use(),
        TokenType.SEMICOLON: lambda self: exp.Semicolon(),
    }

    UNARY_PARSERS = {
        TokenType.PLUS: lambda self: self._parse_unary(),  # Unary + is handled as a no-op
        TokenType.NOT: lambda self: self.expression(exp.Not, this=self._parse_equality()),
        TokenType.TILDA: lambda self: self.expression(exp.BitwiseNot, this=self._parse_unary()),
        TokenType.DASH: lambda self: self.expression(exp.Neg, this=self._parse_unary()),
        TokenType.PIPE_SLASH: lambda self: self.expression(exp.Sqrt, this=self._parse_unary()),
        TokenType.DPIPE_SLASH: lambda self: self.expression(exp.Cbrt, this=self._parse_unary()),
    }

    STRING_PARSERS = {
        TokenType.HEREDOC_STRING: lambda self, token: self.expression(
            exp.RawString, this=token.text
        ),
        TokenType.NATIONAL_STRING: lambda self, token: self.expression(
            exp.National, this=token.text
        ),
        TokenType.RAW_STRING: lambda self, token: self.expression(exp.RawString, this=token.text),
        TokenType.STRING: lambda self, token: self.expression(
            exp.Literal, this=token.text, is_string=True
        ),
        TokenType.UNICODE_STRING: lambda self, token: self.expression(
            exp.UnicodeString,
            this=token.text,
            escape=self._match_text_seq("UESCAPE") and self._parse_string(),
        ),
    }

    NUMERIC_PARSERS = {
        TokenType.BIT_STRING: lambda self, token: self.expression(exp.BitString, this=token.text),
        TokenType.BYTE_STRING: lambda self, token: self.expression(exp.ByteString, this=token.text),
        TokenType.HEX_STRING: lambda self, token: self.expression(
            exp.HexString,
            this=token.text,
            is_integer=self.dialect.HEX_STRING_IS_INTEGER_TYPE or None,
        ),
        TokenType.NUMBER: lambda self, token: self.expression(
            exp.Literal, this=token.text, is_string=False
        ),
    }

    PRIMARY_PARSERS = {
        **STRING_PARSERS,
        **NUMERIC_PARSERS,
        TokenType.INTRODUCER: lambda self, token: self._parse_introducer(token),
        TokenType.NULL: lambda self, _: self.expression(exp.Null),
        TokenType.TRUE: lambda self, _: self.expression(exp.Boolean, this=True),
        TokenType.FALSE: lambda self, _: self.expression(exp.Boolean, this=False),
        TokenType.SESSION_PARAMETER: lambda self, _: self._parse_session_parameter(),
        TokenType.STAR: lambda self, _: self._parse_star_ops(),
    }

    PLACEHOLDER_PARSERS = {
        TokenType.PLACEHOLDER: lambda self: self.expression(exp.Placeholder),
        TokenType.PARAMETER: lambda self: self._parse_parameter(),
        TokenType.COLON: lambda self: (
            self.expression(exp.Placeholder, this=self._prev.text)
            if self._match_set(self.COLON_PLACEHOLDER_TOKENS)
            else None
        ),
    }

    RANGE_PARSERS = {
        TokenType.AT_GT: binary_range_parser(exp.ArrayContainsAll),
        TokenType.BETWEEN: lambda self, this: self._parse_between(this),
        TokenType.GLOB: binary_range_parser(exp.Glob),
        TokenType.ILIKE: binary_range_parser(exp.ILike),
        TokenType.IN: lambda self, this: self._parse_in(this),
        TokenType.IRLIKE: binary_range_parser(exp.RegexpILike),
        TokenType.IS: lambda self, this: self._parse_is(this),
        TokenType.LIKE: binary_range_parser(exp.Like),
        TokenType.LT_AT: binary_range_parser(exp.ArrayContainsAll, reverse_args=True),
        TokenType.OVERLAPS: binary_range_parser(exp.Overlaps),
        TokenType.RLIKE: binary_range_parser(exp.RegexpLike),
        TokenType.SIMILAR_TO: binary_range_parser(exp.SimilarTo),
        TokenType.FOR: lambda self, this: self._parse_comprehension(this),
        TokenType.QMARK_AMP: binary_range_parser(exp.JSONBContainsAllTopKeys),
        TokenType.QMARK_PIPE: binary_range_parser(exp.JSONBContainsAnyTopKeys),
        TokenType.HASH_DASH: binary_range_parser(exp.JSONBDeleteAtPath),
    }

    PIPE_SYNTAX_TRANSFORM_PARSERS = {
        "AGGREGATE": lambda self, query: self._parse_pipe_syntax_aggregate(query),
        "AS": lambda self, query: self._build_pipe_cte(
            query, [exp.Star()], self._parse_table_alias()
        ),
        "EXTEND": lambda self, query: self._parse_pipe_syntax_extend(query),
        "LIMIT": lambda self, query: self._parse_pipe_syntax_limit(query),
        "ORDER BY": lambda self, query: query.order_by(
            self._parse_order(), append=False, copy=False
        ),
        "PIVOT": lambda self, query: self._parse_pipe_syntax_pivot(query),
        "SELECT": lambda self, query: self._parse_pipe_syntax_select(query),
        "TABLESAMPLE": lambda self, query: self._parse_pipe_syntax_tablesample(query),
        "UNPIVOT": lambda self, query: self._parse_pipe_syntax_pivot(query),
        "WHERE": lambda self, query: query.where(self._parse_where(), copy=False),
    }

    PROPERTY_PARSERS: t.Dict[str, t.Callable] = {
        "ALLOWED_VALUES": lambda self: self.expression(
            exp.AllowedValuesProperty, expressions=self._parse_csv(self._parse_primary)
        ),
        "ALGORITHM": lambda self: self._parse_property_assignment(exp.AlgorithmProperty),
        "AUTO": lambda self: self._parse_auto_property(),
        "AUTO_INCREMENT": lambda self: self._parse_property_assignment(exp.AutoIncrementProperty),
        "BACKUP": lambda self: self.expression(
            exp.BackupProperty, this=self._parse_var(any_token=True)
        ),
        "BLOCKCOMPRESSION": lambda self: self._parse_blockcompression(),
        "CHARSET": lambda self, **kwargs: self._parse_character_set(**kwargs),
        "CHARACTER SET": lambda self, **kwargs: self._parse_character_set(**kwargs),
        "CHECKSUM": lambda self: self._parse_checksum(),
        "CLUSTER BY": lambda self: self._parse_cluster(),
        "CLUSTERED": lambda self: self._parse_clustered_by(),
        "COLLATE": lambda self, **kwargs: self._parse_property_assignment(
            exp.CollateProperty, **kwargs
        ),
        "COMMENT": lambda self: self._parse_property_assignment(exp.SchemaCommentProperty),
        "CONTAINS": lambda self: self._parse_contains_property(),
        "COPY": lambda self: self._parse_copy_property(),
        "DATABLOCKSIZE": lambda self, **kwargs: self._parse_datablocksize(**kwargs),
        "DATA_DELETION": lambda self: self._parse_data_deletion_property(),
        "DEFINER": lambda self: self._parse_definer(),
        "DETERMINISTIC": lambda self: self.expression(
            exp.StabilityProperty, this=exp.Literal.string("IMMUTABLE")
        ),
        "DISTRIBUTED": lambda self: self._parse_distributed_property(),
        "DUPLICATE": lambda self: self._parse_composite_key_property(exp.DuplicateKeyProperty),
        "DYNAMIC": lambda self: self.expression(exp.DynamicProperty),
        "DISTKEY": lambda self: self._parse_distkey(),
        "DISTSTYLE": lambda self: self._parse_property_assignment(exp.DistStyleProperty),
        "EMPTY": lambda self: self.expression(exp.EmptyProperty),
        "ENGINE": lambda self: self._parse_property_assignment(exp.EngineProperty),
        "ENVIRONMENT": lambda self: self.expression(
            exp.EnviromentProperty, expressions=self._parse_wrapped_csv(self._parse_assignment)
        ),
        "EXECUTE": lambda self: self._parse_property_assignment(exp.ExecuteAsProperty),
        "EXTERNAL": lambda self: self.expression(exp.ExternalProperty),
        "FALLBACK": lambda self, **kwargs: self._parse_fallback(**kwargs),
        "FORMAT": lambda self: self._parse_property_assignment(exp.FileFormatProperty),
        "FREESPACE": lambda self: self._parse_freespace(),
        "GLOBAL": lambda self: self.expression(exp.GlobalProperty),
        "HEAP": lambda self: self.expression(exp.HeapProperty),
        "ICEBERG": lambda self: self.expression(exp.IcebergProperty),
        "IMMUTABLE": lambda self: self.expression(
            exp.StabilityProperty, this=exp.Literal.string("IMMUTABLE")
        ),
        "INHERITS": lambda self: self.expression(
            exp.InheritsProperty, expressions=self._parse_wrapped_csv(self._parse_table)
        ),
        "INPUT": lambda self: self.expression(exp.InputModelProperty, this=self._parse_schema()),
        "JOURNAL": lambda self, **kwargs: self._parse_journal(**kwargs),
        "LANGUAGE": lambda self: self._parse_property_assignment(exp.LanguageProperty),
        "LAYOUT": lambda self: self._parse_dict_property(this="LAYOUT"),
        "LIFETIME": lambda self: self._parse_dict_range(this="LIFETIME"),
        "LIKE": lambda self: self._parse_create_like(),
        "LOCATION": lambda self: self._parse_property_assignment(exp.LocationProperty),
        "LOCK": lambda self: self._parse_locking(),
        "LOCKING": lambda self: self._parse_locking(),
        "LOG": lambda self, **kwargs: self._parse_log(**kwargs),
        "MATERIALIZED": lambda self: self.expression(exp.MaterializedProperty),
        "MERGEBLOCKRATIO": lambda self, **kwargs: self._parse_mergeblockratio(**kwargs),
        "MODIFIES": lambda self: self._parse_modifies_property(),
        "MULTISET": lambda self: self.expression(exp.SetProperty, multi=True),
        "NO": lambda self: self._parse_no_property(),
        "ON": lambda self: self._parse_on_property(),
        "ORDER BY": lambda self: self._parse_order(skip_order_token=True),
        "OUTPUT": lambda self: self.expression(exp.OutputModelProperty, this=self._parse_schema()),
        "PARTITION": lambda self: self._parse_partitioned_of(),
        "PARTITION BY": lambda self: self._parse_partitioned_by(),
        "PARTITIONED BY": lambda self: self._parse_partitioned_by(),
        "PARTITIONED_BY": lambda self: self._parse_partitioned_by(),
        "PRIMARY KEY": lambda self: self._parse_primary_key(in_props=True),
        "RANGE": lambda self: self._parse_dict_range(this="RANGE"),
        "READS": lambda self: self._parse_reads_property(),
        "REMOTE": lambda self: self._parse_remote_with_connection(),
        "RETURNS": lambda self: self._parse_returns(),
        "STRICT": lambda self: self.expression(exp.StrictProperty),
        "STREAMING": lambda self: self.expression(exp.StreamingTableProperty),
        "ROW": lambda self: self._parse_row(),
        "ROW_FORMAT": lambda self: self._parse_property_assignment(exp.RowFormatProperty),
        "SAMPLE": lambda self: self.expression(
            exp.SampleProperty, this=self._match_text_seq("BY") and self._parse_bitwise()
        ),
        "SECURE": lambda self: self.expression(exp.SecureProperty),
        "SECURITY": lambda self: self._parse_security(),
        "SET": lambda self: self.expression(exp.SetProperty, multi=False),
        "SETTINGS": lambda self: self._parse_settings_property(),
        "SHARING": lambda self: self._parse_property_assignment(exp.SharingProperty),
        "SORTKEY": lambda self: self._parse_sortkey(),
        "SOURCE": lambda self: self._parse_dict_property(this="SOURCE"),
        "STABLE": lambda self: self.expression(
            exp.StabilityProperty, this=exp.Literal.string("STABLE")
        ),
        "STORED": lambda self: self._parse_stored(),
        "SYSTEM_VERSIONING": lambda self: self._parse_system_versioning_property(),
        "TBLPROPERTIES": lambda self: self._parse_wrapped_properties(),
        "TEMP": lambda self: self.expression(exp.TemporaryProperty),
        "TEMPORARY": lambda self: self.expression(exp.TemporaryProperty),
        "TO": lambda self: self._parse_to_table(),
        "TRANSIENT": lambda self: self.expression(exp.TransientProperty),
        "TRANSFORM": lambda self: self.expression(
            exp.TransformModelProperty, expressions=self._parse_wrapped_csv(self._parse_expression)
        ),
        "TTL": lambda self: self._parse_ttl(),
        "USING": lambda self: self._parse_property_assignment(exp.FileFormatProperty),
        "UNLOGGED": lambda self: self.expression(exp.UnloggedProperty),
        "VOLATILE": lambda self: self._parse_volatile_property(),
        "WITH": lambda self: self._parse_with_property(),
        "TABLESPACE": lambda self: self._parse_tablespace_property(),
	    "OPTIONS": lambda self: self._parse_wrapped_properties(),
    }

    CONSTRAINT_PARSERS = {
        "AUTOINCREMENT": lambda self: self._parse_auto_increment(),
        "AUTO_INCREMENT": lambda self: self._parse_auto_increment(),
        "CASESPECIFIC": lambda self: self.expression(exp.CaseSpecificColumnConstraint, not_=False),
        "CHARACTER SET": lambda self: self.expression(
            exp.CharacterSetColumnConstraint, this=self._parse_var_or_string()
        ),
        "CHECK": lambda self: self.expression(
            exp.CheckColumnConstraint,
            this=self._parse_wrapped(self._parse_assignment),
            enforced=self._match_text_seq("ENFORCED"),
        ),
        "COLLATE": lambda self: self.expression(
            exp.CollateColumnConstraint,
            this=self._parse_identifier() or self._parse_column(),
        ),
        "COMMENT": lambda self: self.expression(
            exp.CommentColumnConstraint, this=self._parse_string()
        ),
        "COMPRESS": lambda self: self._parse_compress(),
        "CLUSTERED": lambda self: self.expression(
            exp.ClusteredColumnConstraint, this=self._parse_wrapped_csv(self._parse_ordered)
        ),
        "NONCLUSTERED": lambda self: self.expression(
            exp.NonClusteredColumnConstraint, this=self._parse_wrapped_csv(self._parse_ordered)
        ),
        "DEFAULT": lambda self: self.expression(
            exp.DefaultColumnConstraint, this=self._parse_bitwise()
        ),
        "ENCODE": lambda self: self.expression(exp.EncodeColumnConstraint, this=self._parse_var()),
        "EPHEMERAL": lambda self: self.expression(
            exp.EphemeralColumnConstraint, this=self._parse_bitwise()
        ),
        "EXCLUDE": lambda self: self.expression(
            exp.ExcludeColumnConstraint, this=self._parse_index_params()
        ),
        "FOREIGN KEY": lambda self: self._parse_foreign_key(),
        "FORMAT": lambda self: self.expression(
            exp.DateFormatColumnConstraint, this=self._parse_var_or_string()
        ),
        "KEY": lambda self: self._parse_key_constraint(),
        "GENERATED": lambda self: self._parse_generated_as_identity(),
        "IDENTITY": lambda self: self._parse_auto_increment(),
        "INLINE": lambda self: self._parse_inline(),
        "LIKE": lambda self: self._parse_create_like(),
        "NOT": lambda self: self._parse_not_constraint(),
        "NULL": lambda self: self.expression(exp.NotNullColumnConstraint, allow_null=True),
        "ON": lambda self: (
            self._match(TokenType.UPDATE)
            and self.expression(exp.OnUpdateColumnConstraint, this=self._parse_function())
        )
        or self.expression(exp.OnProperty, this=self._parse_id_var()),
        "PATH": lambda self: self.expression(exp.PathColumnConstraint, this=self._parse_string()),
        "PERIOD": lambda self: self._parse_period_for_system_time(),
        "PRIMARY KEY": lambda self: self._parse_primary_key(),
        "REFERENCES": lambda self: self._parse_references(match=False),
        "TITLE": lambda self: self.expression(
            exp.TitleColumnConstraint, this=self._parse_var_or_string()
        ),
        "TTL": lambda self: self.expression(exp.MergeTreeTTL, expressions=[self._parse_bitwise()]),
        "UNIQUE": lambda self: self._parse_unique(),
        "UPPERCASE": lambda self: self.expression(exp.UppercaseColumnConstraint),
        "WITH": lambda self: self.expression(
            exp.Properties, expressions=self._parse_wrapped_properties()
        ),
        "BUCKET": lambda self: self._parse_partitioned_by_bucket_or_truncate(),
        "TRUNCATE": lambda self: self._parse_partitioned_by_bucket_or_truncate(),
    }

    def _parse_partitioned_by_bucket_or_truncate(self) -> t.Optional[exp.Expression]:
        # 解析 PARTITION BY (BUCKET(..) | TRUNCATE(..)) 语法：
        # - 如果没有括号，则解析为标识符
        # - 如果有括号，则解析为 PartitionedByBucket 或 PartitionByTruncate
        if not self._match(TokenType.L_PAREN, advance=False):
            # Partitioning by bucket or truncate follows the syntax:
            # PARTITION BY (BUCKET(..) | TRUNCATE(..))
            # If we don't have parenthesis after each keyword, we should instead parse this as an identifier
            self._retreat(self._index - 1)
            return None

        klass = (
            exp.PartitionedByBucket
            if self._prev.text.upper() == "BUCKET"
            else exp.PartitionByTruncate
        )

        args = self._parse_wrapped_csv(lambda: self._parse_primary() or self._parse_column())
        this, expression = seq_get(args, 0), seq_get(args, 1)

        if isinstance(this, exp.Literal):
            # 检查 Iceberg 分区转换（bucket / truncate）并确保它们的参数顺序正确
            #  - 对于 Hive，它是 `bucket(<num buckets>, <col name>)` 或 `truncate(<num_chars>, <col_name>)`
            #  - 对于 Trino，它是相反的 - `bucket(<col name>, <num buckets>)` 或 `truncate(<col_name>, <num_chars>)`
            # 两种变体都被规范化为后者 i.e `bucket(<col name>, <num buckets>)`
            #
            # Hive ref: https://docs.aws.amazon.com/athena/latest/ug/querying-iceberg-creating-tables.html#querying-iceberg-partitioning
            # Trino ref: https://docs.aws.amazon.com/athena/latest/ug/create-table-as.html#ctas-table-properties
            this, expression = expression, this

        return self.expression(klass, this=this, expression=expression)

    ALTER_PARSERS = {
        "ADD": lambda self: self._parse_alter_table_add(),
        "AS": lambda self: self._parse_select(),
        "ALTER": lambda self: self._parse_alter_table_alter(),
        "CLUSTER BY": lambda self: self._parse_cluster(wrapped=True),
        "DELETE": lambda self: self.expression(exp.Delete, where=self._parse_where()),
        "DROP": lambda self: self._parse_alter_table_drop(),
        "RENAME": lambda self: self._parse_alter_table_rename(),
        "SET": lambda self: self._parse_alter_table_set(),
        "SWAP": lambda self: self.expression(
            exp.SwapTable, this=self._match(TokenType.WITH) and self._parse_table(schema=True)
        ),
    }

    ALTER_ALTER_PARSERS = {
        "DISTKEY": lambda self: self._parse_alter_diststyle(),
        "DISTSTYLE": lambda self: self._parse_alter_diststyle(),
        "SORTKEY": lambda self: self._parse_alter_sortkey(),
        "COMPOUND": lambda self: self._parse_alter_sortkey(compound=True),
    }

    SCHEMA_UNNAMED_CONSTRAINTS = {
        "CHECK",
        "EXCLUDE",
        "FOREIGN KEY",
        "LIKE",
        "PERIOD",
        "PRIMARY KEY",
        "UNIQUE",
        "BUCKET",
        "TRUNCATE",
    }

    NO_PAREN_FUNCTION_PARSERS = {
        "ANY": lambda self: self.expression(exp.Any, this=self._parse_bitwise()),
        "CASE": lambda self: self._parse_case(),
        "CONNECT_BY_ROOT": lambda self: self.expression(
            exp.ConnectByRoot, this=self._parse_column()
        ),
        "IF": lambda self: self._parse_if(),
    }

    INVALID_FUNC_NAME_TOKENS = {
        TokenType.IDENTIFIER,
        TokenType.STRING,
    }

    FUNCTIONS_WITH_ALIASED_ARGS = {"STRUCT"}

    KEY_VALUE_DEFINITIONS = (exp.Alias, exp.EQ, exp.PropertyEQ, exp.Slice)

    FUNCTION_PARSERS = {
        **{
            name: lambda self: self._parse_max_min_by(exp.ArgMax) for name in exp.ArgMax.sql_names()
        },
        **{
            name: lambda self: self._parse_max_min_by(exp.ArgMin) for name in exp.ArgMin.sql_names()
        },
        "CAST": lambda self: self._parse_cast(self.STRICT_CAST),
        "CEIL": lambda self: self._parse_ceil_floor(exp.Ceil),
        "CONVERT": lambda self: self._parse_convert(self.STRICT_CAST),
        "DECODE": lambda self: self._parse_decode(),
        "EXTRACT": lambda self: self._parse_extract(),
        "FLOOR": lambda self: self._parse_ceil_floor(exp.Floor),
        "GAP_FILL": lambda self: self._parse_gap_fill(),
        "JSON_OBJECT": lambda self: self._parse_json_object(),
        "JSON_OBJECTAGG": lambda self: self._parse_json_object(agg=True),
        "JSON_TABLE": lambda self: self._parse_json_table(),
        "MATCH": lambda self: self._parse_match_against(),
        "NORMALIZE": lambda self: self._parse_normalize(),
        "OPENJSON": lambda self: self._parse_open_json(),
        "OVERLAY": lambda self: self._parse_overlay(),
        "POSITION": lambda self: self._parse_position(),
        "SAFE_CAST": lambda self: self._parse_cast(False, safe=True),
        "STRING_AGG": lambda self: self._parse_string_agg(),
        "SUBSTRING": lambda self: self._parse_substring(),
        "TRIM": lambda self: self._parse_trim(),
        "TRY_CAST": lambda self: self._parse_cast(False, safe=True),
        "TRY_CONVERT": lambda self: self._parse_convert(False, safe=True),
        "XMLELEMENT": lambda self: self.expression(
            exp.XMLElement,
            this=self._match_text_seq("NAME") and self._parse_id_var(),
            expressions=self._match(TokenType.COMMA) and self._parse_csv(self._parse_expression),
        ),
        "XMLTABLE": lambda self: self._parse_xml_table(),
    }

    QUERY_MODIFIER_PARSERS = {
        TokenType.MATCH_RECOGNIZE: lambda self: ("match", self._parse_match_recognize()),
        TokenType.PREWHERE: lambda self: ("prewhere", self._parse_prewhere()),
        TokenType.WHERE: lambda self: ("where", self._parse_where()),
        TokenType.GROUP_BY: lambda self: ("group", self._parse_group()),
        TokenType.HAVING: lambda self: ("having", self._parse_having()),
        TokenType.QUALIFY: lambda self: ("qualify", self._parse_qualify()),
        TokenType.WINDOW: lambda self: ("windows", self._parse_window_clause()),
        TokenType.ORDER_BY: lambda self: ("order", self._parse_order()),
        TokenType.LIMIT: lambda self: ("limit", self._parse_limit()),
        TokenType.FETCH: lambda self: ("limit", self._parse_limit()),
        TokenType.OFFSET: lambda self: ("offset", self._parse_offset()),
        TokenType.FOR: lambda self: ("locks", self._parse_locks()),
        TokenType.LOCK: lambda self: ("locks", self._parse_locks()),
        TokenType.TABLE_SAMPLE: lambda self: ("sample", self._parse_table_sample(as_modifier=True)),
        TokenType.USING: lambda self: ("sample", self._parse_table_sample(as_modifier=True)),
        TokenType.CLUSTER_BY: lambda self: (
            "cluster",
            self._parse_sort(exp.Cluster, TokenType.CLUSTER_BY),
        ),
        TokenType.DISTRIBUTE_BY: lambda self: (
            "distribute",
            self._parse_sort(exp.Distribute, TokenType.DISTRIBUTE_BY),
        ),
        TokenType.SORT_BY: lambda self: ("sort", self._parse_sort(exp.Sort, TokenType.SORT_BY)),
        TokenType.CONNECT_BY: lambda self: ("connect", self._parse_connect(skip_start_token=True)),
        TokenType.START_WITH: lambda self: ("connect", self._parse_connect()),
    }
    QUERY_MODIFIER_TOKENS = set(QUERY_MODIFIER_PARSERS)

    SET_PARSERS = {
        "GLOBAL": lambda self: self._parse_set_item_assignment("GLOBAL"),
        "LOCAL": lambda self: self._parse_set_item_assignment("LOCAL"),
        "SESSION": lambda self: self._parse_set_item_assignment("SESSION"),
        "TRANSACTION": lambda self: self._parse_set_transaction(),
    }

    SHOW_PARSERS: t.Dict[str, t.Callable] = {}

    TYPE_LITERAL_PARSERS = {
        exp.DataType.Type.JSON: lambda self, this, _: self.expression(exp.ParseJSON, this=this),
    }

    TYPE_CONVERTERS: t.Dict[exp.DataType.Type, t.Callable[[exp.DataType], exp.DataType]] = {}

    DDL_SELECT_TOKENS = {TokenType.SELECT, TokenType.WITH, TokenType.L_PAREN}

    PRE_VOLATILE_TOKENS = {TokenType.CREATE, TokenType.REPLACE, TokenType.UNIQUE}

    TRANSACTION_KIND = {"DEFERRED", "IMMEDIATE", "EXCLUSIVE"}
    TRANSACTION_CHARACTERISTICS: OPTIONS_TYPE = {
        "ISOLATION": (
            ("LEVEL", "REPEATABLE", "READ"),
            ("LEVEL", "READ", "COMMITTED"),
            ("LEVEL", "READ", "UNCOMITTED"),
            ("LEVEL", "SERIALIZABLE"),
        ),
        "READ": ("WRITE", "ONLY"),
    }

    CONFLICT_ACTIONS: OPTIONS_TYPE = dict.fromkeys(
        ("ABORT", "FAIL", "IGNORE", "REPLACE", "ROLLBACK", "UPDATE"), tuple()
    )
    CONFLICT_ACTIONS["DO"] = ("NOTHING", "UPDATE")

    CREATE_SEQUENCE: OPTIONS_TYPE = {
        "SCALE": ("EXTEND", "NOEXTEND"),
        "SHARD": ("EXTEND", "NOEXTEND"),
        "NO": ("CYCLE", "CACHE", "MAXVALUE", "MINVALUE"),
        **dict.fromkeys(
            (
                "SESSION",
                "GLOBAL",
                "KEEP",
                "NOKEEP",
                "ORDER",
                "NOORDER",
                "NOCACHE",
                "CYCLE",
                "NOCYCLE",
                "NOMINVALUE",
                "NOMAXVALUE",
                "NOSCALE",
                "NOSHARD",
            ),
            tuple(),
        ),
    }

    ISOLATED_LOADING_OPTIONS: OPTIONS_TYPE = {"FOR": ("ALL", "INSERT", "NONE")}

    USABLES: OPTIONS_TYPE = dict.fromkeys(
        ("ROLE", "WAREHOUSE", "DATABASE", "SCHEMA", "CATALOG"), tuple()
    )

    CAST_ACTIONS: OPTIONS_TYPE = dict.fromkeys(("RENAME", "ADD"), ("FIELDS",))

    SCHEMA_BINDING_OPTIONS: OPTIONS_TYPE = {
        "TYPE": ("EVOLUTION",),
        **dict.fromkeys(("BINDING", "COMPENSATION", "EVOLUTION"), tuple()),
    }

    PROCEDURE_OPTIONS: OPTIONS_TYPE = {}

    EXECUTE_AS_OPTIONS: OPTIONS_TYPE = dict.fromkeys(("CALLER", "SELF", "OWNER"), tuple())

    KEY_CONSTRAINT_OPTIONS: OPTIONS_TYPE = {
        "NOT": ("ENFORCED",),
        "MATCH": (
            "FULL",
            "PARTIAL",
            "SIMPLE",
        ),
        "INITIALLY": ("DEFERRED", "IMMEDIATE"),
        "USING": (
            "BTREE",
            "HASH",
        ),
        **dict.fromkeys(("DEFERRABLE", "NORELY", "RELY"), tuple()),
    }

    WINDOW_EXCLUDE_OPTIONS: OPTIONS_TYPE = {
        "NO": ("OTHERS",),
        "CURRENT": ("ROW",),
        **dict.fromkeys(("GROUP", "TIES"), tuple()),
    }

    INSERT_ALTERNATIVES = {"ABORT", "FAIL", "IGNORE", "REPLACE", "ROLLBACK"}

    CLONE_KEYWORDS = {"CLONE", "COPY"}
    HISTORICAL_DATA_PREFIX = {"AT", "BEFORE", "END"}
    HISTORICAL_DATA_KIND = {"OFFSET", "STATEMENT", "STREAM", "TIMESTAMP", "VERSION"}

    OPCLASS_FOLLOW_KEYWORDS = {"ASC", "DESC", "NULLS", "WITH"}

    OPTYPE_FOLLOW_TOKENS = {TokenType.COMMA, TokenType.R_PAREN}

    TABLE_INDEX_HINT_TOKENS = {TokenType.FORCE, TokenType.IGNORE, TokenType.USE}

    VIEW_ATTRIBUTES = {"ENCRYPTION", "SCHEMABINDING", "VIEW_METADATA"}

    WINDOW_ALIAS_TOKENS = ID_VAR_TOKENS - {TokenType.RANGE, TokenType.ROWS}
    WINDOW_BEFORE_PAREN_TOKENS = {TokenType.OVER}
    WINDOW_SIDES = {"FOLLOWING", "PRECEDING"}

    JSON_KEY_VALUE_SEPARATOR_TOKENS = {TokenType.COLON, TokenType.COMMA, TokenType.IS}

    FETCH_TOKENS = ID_VAR_TOKENS - {TokenType.ROW, TokenType.ROWS, TokenType.PERCENT}

    ADD_CONSTRAINT_TOKENS = {
        TokenType.CONSTRAINT,
        TokenType.FOREIGN_KEY,
        TokenType.INDEX,
        TokenType.KEY,
        TokenType.PRIMARY_KEY,
        TokenType.UNIQUE,
    }

    DISTINCT_TOKENS = {TokenType.DISTINCT}

    UNNEST_OFFSET_ALIAS_TOKENS = TABLE_ALIAS_TOKENS - SET_OPERATIONS

    SELECT_START_TOKENS = {TokenType.L_PAREN, TokenType.WITH, TokenType.SELECT}

    COPY_INTO_VARLEN_OPTIONS = {"FILE_FORMAT", "COPY_OPTIONS", "FORMAT_OPTIONS", "CREDENTIAL"}

    IS_JSON_PREDICATE_KIND = {"VALUE", "SCALAR", "ARRAY", "OBJECT"}

    ODBC_DATETIME_LITERALS: t.Dict[str, t.Type[exp.Expression]] = {}

    ON_CONDITION_TOKENS = {"ERROR", "NULL", "TRUE", "FALSE", "EMPTY"}

    PRIVILEGE_FOLLOW_TOKENS = {TokenType.ON, TokenType.COMMA, TokenType.L_PAREN}

    # The style options for the DESCRIBE statement
    DESCRIBE_STYLES = {"ANALYZE", "EXTENDED", "FORMATTED", "HISTORY"}

    # The style options for the ANALYZE statement
    ANALYZE_STYLES = {
        "BUFFER_USAGE_LIMIT",
        "FULL",
        "LOCAL",
        "NO_WRITE_TO_BINLOG",
        "SAMPLE",
        "SKIP_LOCKED",
        "VERBOSE",
    }

    ANALYZE_EXPRESSION_PARSERS = {
        "ALL": lambda self: self._parse_analyze_columns(),
        "COMPUTE": lambda self: self._parse_analyze_statistics(),
        "DELETE": lambda self: self._parse_analyze_delete(),
        "DROP": lambda self: self._parse_analyze_histogram(),
        "ESTIMATE": lambda self: self._parse_analyze_statistics(),
        "LIST": lambda self: self._parse_analyze_list(),
        "PREDICATE": lambda self: self._parse_analyze_columns(),
        "UPDATE": lambda self: self._parse_analyze_histogram(),
        "VALIDATE": lambda self: self._parse_analyze_validate(),
    }

    PARTITION_KEYWORDS = {"PARTITION", "SUBPARTITION"}

    AMBIGUOUS_ALIAS_TOKENS = (TokenType.LIMIT, TokenType.OFFSET)

    OPERATION_MODIFIERS: t.Set[str] = set()

    RECURSIVE_CTE_SEARCH_KIND = {"BREADTH", "DEPTH", "CYCLE"}

    MODIFIABLES = (exp.Query, exp.Table, exp.TableFromRows, exp.Values)

    STRICT_CAST = True

    PREFIXED_PIVOT_COLUMNS = False
    IDENTIFY_PIVOT_STRINGS = False

    LOG_DEFAULTS_TO_LN = False

    # Whether the table sample clause expects CSV syntax
    TABLESAMPLE_CSV = False

    # The default method used for table sampling
    DEFAULT_SAMPLING_METHOD: t.Optional[str] = None

    # Whether the SET command needs a delimiter (e.g. "=") for assignments
    SET_REQUIRES_ASSIGNMENT_DELIMITER = True

    # Whether the TRIM function expects the characters to trim as its first argument
    TRIM_PATTERN_FIRST = False

    # Whether string aliases are supported `SELECT COUNT(*) 'count'`
    STRING_ALIASES = False

    # Whether query modifiers such as LIMIT are attached to the UNION node (vs its right operand)
    MODIFIERS_ATTACHED_TO_SET_OP = True
    SET_OP_MODIFIERS = {"order", "limit", "offset"}

    # Whether to parse IF statements that aren't followed by a left parenthesis as commands
    NO_PAREN_IF_COMMANDS = True

    # Whether the -> and ->> operators expect documents of type JSON (e.g. Postgres)
    JSON_ARROWS_REQUIRE_JSON_TYPE = False

    # Whether the `:` operator is used to extract a value from a VARIANT column
    COLON_IS_VARIANT_EXTRACT = False

    # Whether or not a VALUES keyword needs to be followed by '(' to form a VALUES clause.
    # If this is True and '(' is not found, the keyword will be treated as an identifier
    VALUES_FOLLOWED_BY_PAREN = True

    # Whether implicit unnesting is supported, e.g. SELECT 1 FROM y.z AS z, z.a (Redshift)
    SUPPORTS_IMPLICIT_UNNEST = False

    # Whether or not interval spans are supported, INTERVAL 1 YEAR TO MONTHS
    INTERVAL_SPANS = True

    # Whether a PARTITION clause can follow a table reference
    SUPPORTS_PARTITION_SELECTION = False

    # Whether the `name AS expr` schema/column constraint requires parentheses around `expr`
    WRAPPED_TRANSFORM_COLUMN_CONSTRAINT = True

    # Whether the 'AS' keyword is optional in the CTE definition syntax
    OPTIONAL_ALIAS_TOKEN_CTE = True

    # Whether renaming a column with an ALTER statement requires the presence of the COLUMN keyword
    ALTER_RENAME_REQUIRES_COLUMN = True

    # Whether Alter statements are allowed to contain Partition specifications
    ALTER_TABLE_PARTITIONS = False

    # Whether all join types have the same precedence, i.e., they "naturally" produce a left-deep tree.
    # In standard SQL, joins that use the JOIN keyword take higher precedence than comma-joins. That is
    # to say, JOIN operators happen before comma operators. This is not the case in some dialects, such
    # as BigQuery, where all joins have the same precedence.
    JOINS_HAVE_EQUAL_PRECEDENCE = False

    # Whether TIMESTAMP <literal> can produce a zone-aware timestamp
    ZONE_AWARE_TIMESTAMP_CONSTRUCTOR = False

    # Whether map literals support arbitrary expressions as keys.
    # When True, allows complex keys like arrays or literals: {[1, 2]: 3}, {1: 2} (e.g. DuckDB).
    # When False, keys are typically restricted to identifiers.
    MAP_KEYS_ARE_ARBITRARY_EXPRESSIONS = False

    # Whether JSON_EXTRACT requires a JSON expression as the first argument, e.g this
    # is true for Snowflake but not for BigQuery which can also process strings
    JSON_EXTRACT_REQUIRES_JSON_EXPRESSION = False

    # MERGE WHEN INSERT capability flags (dialect-specific; default disabled)
    # - MERGE_INSERT_DEFAULT_VALUES_SUPPORTED: supports "INSERT DEFAULT VALUES" form
    # - MERGE_INSERT_OVERRIDING_SUPPORTED: supports "OVERRIDING {SYSTEM|USER} VALUE"
    # - MERGE_INSERT_WHERE_SUPPORTED: supports a trailing WHERE clause inside INSERT
    MERGE_INSERT_DEFAULT_VALUES_SUPPORTED = False
    MERGE_INSERT_OVERRIDING_SUPPORTED = False
    MERGE_INSERT_WHERE_SUPPORTED = False
    
    # 索引名称中是否允许出现.
    SUPPORT_INDEX_NAME_WITH_DOT = False

    # Dialects like Databricks support JOINS without join criteria
    # Adding an ON TRUE, makes transpilation semantically correct for other dialects
    ADD_JOIN_ON_TRUE = False

    # Whether INTERVAL spans with literal format '\d+ hh:[mm:[ss[.ff]]]'
    # can omit the span unit `DAY TO MINUTE` or `DAY TO SECOND`
    SUPPORTS_OMITTED_INTERVAL_SPAN_UNIT = False

    __slots__ = (
        "error_level",
        "error_message_context",
        "max_errors",
        "dialect",
        "sql",
        "errors",
        "_tokens",
        "_index",
        "_curr",
        "_next",
        "_prev",
        "_prev_comments",
        "_pipe_cte_counter",
    )

    # Autofilled
    SHOW_TRIE: t.Dict = {}
    SET_TRIE: t.Dict = {}

    def __init__(
        self,
        error_level: t.Optional[ErrorLevel] = None,
        error_message_context: int = 100,
        max_errors: int = 3,
        dialect: DialectType = None,
    ):
        from sqlglot.dialects import Dialect

        self.error_level = error_level or ErrorLevel.IMMEDIATE
        self.error_message_context = error_message_context
        self.max_errors = max_errors
        self.dialect = Dialect.get_or_raise(dialect)
        self.reset()

    def reset(self):
        self.sql = ""
        self.errors = []
        self._tokens = []
        self._index = 0
        self._curr = None
        self._next = None
        self._prev = None
        self._prev_comments = None
        self._pipe_cte_counter = 0

    def parse(
        self, raw_tokens: t.List[Token], sql: t.Optional[str] = None
    ) -> t.List[t.Optional[exp.Expression]]:
        """
        Parses a list of tokens and returns a list of syntax trees, one tree
        per parsed SQL statement.

        Args:
            raw_tokens: The list of tokens.
            sql: The original SQL string, used to produce helpful debug messages.

        Returns:
            The list of the produced syntax trees.
        """
        return self._parse(
            parse_method=self.__class__._parse_statement, raw_tokens=raw_tokens, sql=sql
        )

    def parse_into(
        self,
        expression_types: exp.IntoType,
        raw_tokens: t.List[Token],
        sql: t.Optional[str] = None,
    ) -> t.List[t.Optional[exp.Expression]]:
        """
        Parses a list of tokens into a given Expression type. If a collection of Expression
        types is given instead, this method will try to parse the token list into each one
        of them, stopping at the first for which the parsing succeeds.

        Args:
            expression_types: The expression type(s) to try and parse the token list into.
            raw_tokens: The list of tokens.
            sql: The original SQL string, used to produce helpful debug messages.

        Returns:
            The target Expression.
        """
        errors = []
        for expression_type in ensure_list(expression_types):
            parser = self.EXPRESSION_PARSERS.get(expression_type)
            if not parser:
                raise TypeError(f"No parser registered for {expression_type}")

            try:
                return self._parse(parser, raw_tokens, sql)
            except ParseError as e:
                e.errors[0]["into_expression"] = expression_type
                errors.append(e)

        raise ParseError(
            f"Failed to parse '{sql or raw_tokens}' into {expression_types}",
            errors=merge_errors(errors),
        ) from errors[-1]

    def _parse(
        self,
        parse_method: t.Callable[[Parser], t.Optional[exp.Expression]],
        raw_tokens: t.List[Token],
        sql: t.Optional[str] = None,
    ) -> t.List[t.Optional[exp.Expression]]:
        self.reset()
        self.sql = sql or ""

        total = len(raw_tokens)
        chunks: t.List[t.List[Token]] = [[]]

        for i, token in enumerate(raw_tokens):
            if token.token_type == TokenType.SEMICOLON:
                if token.comments:
                    chunks.append([token])

                if i < total - 1:
                    chunks.append([])
            else:
                chunks[-1].append(token)

        expressions = []

        for tokens in chunks:
            self._index = -1
            self._tokens = tokens
            self._advance()

            expressions.append(parse_method(self))

            if self._index < len(self._tokens):
                self.raise_error("Invalid expression / Unexpected token")

            self.check_errors()

        return expressions

    def check_errors(self) -> None:
        """Logs or raises any found errors, depending on the chosen error level setting."""
        if self.error_level == ErrorLevel.WARN:
            for error in self.errors:
                logger.error(str(error))
        elif self.error_level == ErrorLevel.RAISE and self.errors:
            raise ParseError(
                concat_messages(self.errors, self.max_errors),
                errors=merge_errors(self.errors),
            )

    def raise_error(self, message: str, token: t.Optional[Token] = None) -> None:
        """
        Appends an error in the list of recorded errors or raises it, depending on the chosen
        error level setting.
        """
        token = token or self._curr or self._prev or Token.string("")
        start = token.start
        end = token.end + 1
        start_context = self.sql[max(start - self.error_message_context, 0) : start]
        highlight = self.sql[start:end]
        end_context = self.sql[end : end + self.error_message_context]

        error = ParseError.new(
            f"{message}. Line {token.line}, Col: {token.col}.\n"
            f"  {start_context}\033[4m{highlight}\033[0m{end_context}",
            description=message,
            line=token.line,
            col=token.col,
            start_context=start_context,
            highlight=highlight,
            end_context=end_context,
        )

        if self.error_level == ErrorLevel.IMMEDIATE:
            raise error

        self.errors.append(error)

    def expression(
        self, exp_class: t.Type[E], comments: t.Optional[t.List[str]] = None, **kwargs
    ) -> E:
        """
        创建并验证一个新的表达式对象

        这是Parser类的核心方法，用于创建各种SQL表达式节点。
        会自动添加注释并验证表达式的完整性。

        Args:
            exp_class: 要实例化的表达式类
            comments: 可选的注释列表，将附加到表达式上
            kwargs: 表达式的参数及其对应的值

        Returns:
            目标表达式实例
        """
        instance = exp_class(**kwargs)
        instance.add_comments(comments) if comments else self._add_comments(instance)
        return self.validate_expression(instance)

    def _add_comments(self, expression: t.Optional[exp.Expression]) -> None:
        """
        将前一个注释添加到表达式中

        用于将解析过程中收集的注释附加到当前表达式上。

        Args:
            expression: 要添加注释的表达式
        """
        if expression and self._prev_comments:
            expression.add_comments(self._prev_comments)
            self._prev_comments = None

    def validate_expression(self, expression: E, args: t.Optional[t.List] = None) -> E:
        """
        验证表达式的完整性

        检查表达式的所有必需参数是否都已设置。
        如果错误级别不是IGNORE，则会检查并报告错误。

        Args:
            expression: 要验证的表达式
            args: 可选的参数列表，如果表达式是Func类型则使用

        Returns:
            验证后的表达式
        """
        if self.error_level != ErrorLevel.IGNORE:
            for error_message in expression.error_messages(args):
                self.raise_error(error_message)

        return expression

    def _find_sql(self, start: Token, end: Token) -> str:
        """
        从原始SQL字符串中提取指定范围的SQL文本

        Args:
            start: 起始token
            end: 结束token

        Returns:
            指定范围内的SQL文本
        """
        return self.sql[start.start : end.end + 1]

    def _is_connected(self) -> bool:
        """
        检查前一个token和当前token是否在SQL中连续

        Returns:
            如果token连续则返回True
        """
        return self._prev and self._curr and self._prev.end + 1 == self._curr.start

    def _advance(self, times: int = 1) -> None:
        """
        向前推进token指针

        更新解析器的内部状态，包括当前token、下一个token和前一个token。
        同时处理注释的传递。

        Args:
            times: 向前推进的步数
        """
        self._index += times
        self._curr = seq_get(self._tokens, self._index)
        self._next = seq_get(self._tokens, self._index + 1)

        # 更新前一个token和注释
        if self._index > 0:
            self._prev = self._tokens[self._index - 1]
            self._prev_comments = self._prev.comments
        else:
            self._prev = None
            self._prev_comments = None

    def _retreat(self, index: int) -> None:
        """
        回退到指定的token索引位置

        Args:
            index: 目标索引位置
        """
        if index != self._index:
            self._advance(index - self._index)

    def _warn_unsupported(self) -> None:
        """
        警告用户遇到了不支持的SQL语法

        当解析器遇到不支持的语法时，会记录警告信息。
        只对当前正在处理的SQL块发出警告。
        """
        if len(self._tokens) <= 1:
            return

        # 使用_find_sql是因为self.sql可能包含多个块，我们只对当前正在处理的块发出警告
        sql = self._find_sql(self._tokens[0], self._tokens[-1])[: self.error_message_context]

        logger.warning(
            f"'{sql}' contains unsupported syntax. Falling back to parsing as a 'Command'."
        )

    def _parse_command(self) -> exp.Command:
        """
        解析不支持的SQL语法作为命令

        当遇到不支持的语法时，将其解析为通用的Command表达式。

        Returns:
            Command表达式对象
        """
        self._warn_unsupported()
        return self.expression(
            exp.Command,
            comments=self._prev_comments,
            this=self._prev.text.upper(),
            expression=self._parse_string(),
        )

    def _try_parse(self, parse_method: t.Callable[[], T], retreat: bool = False) -> t.Optional[T]:
        """
        尝试解析，如果失败则回退

        这个方法解决了内部包含try/catch的解析函数在遇到错误时的回退问题。
        根据用户设置的ErrorLevel，行为可能不同，_try_parse通过设置和重置解析器状态来解决这个问题。

        Args:
            parse_method: 要尝试的解析方法
            retreat: 是否在失败时回退

        Returns:
            解析结果，如果失败则返回None
        """
        index = self._index
        error_level = self.error_level

        self.error_level = ErrorLevel.IMMEDIATE
        try:
            this = parse_method()
        except ParseError:
            this = None
        finally:
            if not this or retreat:
                self._retreat(index)
            self.error_level = error_level

        return this

    def _parse_comment(self, allow_exists: bool = True) -> exp.Expression:
        """
        解析COMMENT语句

        支持对表、列、函数等对象添加注释的SQL语句。

        Args:
            allow_exists: 是否允许IF EXISTS子句

        Returns:
            Comment表达式对象
        """
        start = self._prev
        # 解析可选的IF EXISTS子句
        exists = self._parse_exists() if allow_exists else None

        # 匹配ON关键字
        self._match(TokenType.ON)

        # 检查是否为MATERIALIZED注释
        materialized = self._match_text_seq("MATERIALIZED")
        # 获取要注释的对象类型
        kind = self._match_set(self.CREATABLES) and self._prev
        if not kind:
            return self._parse_as_command(start)

        if kind.token_type in (TokenType.FUNCTION, TokenType.PROCEDURE):
            this = self._parse_user_defined_function(kind=kind.token_type)
        elif kind.token_type == TokenType.TABLE:
            this = self._parse_table(alias_tokens=self.COMMENT_TABLE_ALIAS_TOKENS)
        elif kind.token_type == TokenType.COLUMN:
            this = self._parse_column()
        elif kind.token_type == TokenType.INDEX:
            # 为 INDEX 类型添加专门的处理逻辑，支持带 schema 的索引名
            if self.SUPPORT_INDEX_NAME_WITH_DOT:
                this = self._parse_index_identifier()  # 解析索引名称
            else:
                this = self._parse_id_var()  # 解析索引名称
        else:
            this = self._parse_id_var()

        self._match(TokenType.IS)

        return self.expression(
            exp.Comment,
            this=this,
            kind=kind.text,
            expression=self._parse_string(),
            exists=exists,
            materialized=materialized,
        )

    def _parse_to_table(
        self,
    ) -> exp.ToTableProperty:
        """
        解析TO TABLE属性

        用于解析指向特定表的属性。

        Returns:
            ToTableProperty表达式对象
        """
        table = self._parse_table_parts(schema=True)
        return self.expression(exp.ToTableProperty, this=table)

    def _parse_index_identifier(self) -> t.Optional[exp.Expression]:
        """解析索引标识符，支持带 schema 的格式，如 schema.index_name"""
        # 解析第一个标识符部分（可能是 schema 或索引名）
        first_part = self._parse_table_part(schema=True)
        if not first_part:
            return self._parse_id_var()

        # 检查是否有点号，表示有 schema
        if self._match(TokenType.DOT):
            # 解析第二个标识符部分（索引名）
            second_part = self._parse_table_part(schema=True)
            if second_part:
                # 创建带 schema 的标识符
                return self.expression(
                    exp.Identifier,
                    this=f"{first_part.this}.{second_part.this}",
                    quoted=False
                )
            else:
                # 如果点号后面没有标识符，回退
                self._advance(-1)
                return first_part
        else:
            # 没有 schema，直接返回第一个部分
            return first_part

    # https://clickhouse.com/docs/en/engines/table-engines/mergetree-family/mergetree#mergetree-table-ttl
    def _parse_ttl(self) -> exp.Expression:
        """
        解析TTL（Time To Live）表达式

        支持ClickHouse的MergeTree表引擎的TTL功能。
        可以设置删除、重压缩、移动到磁盘/卷等操作。

        Returns:
            MergeTreeTTL表达式对象
        """
        def _parse_ttl_action() -> t.Optional[exp.Expression]:
            """
            解析TTL动作的内部函数

            支持多种TTL动作：DELETE、RECOMPRESS、TO DISK、TO VOLUME等
            """
            this = self._parse_bitwise()

            # 解析不同的TTL动作类型
            if self._match_text_seq("DELETE"):
                return self.expression(exp.MergeTreeTTLAction, this=this, delete=True)
            if self._match_text_seq("RECOMPRESS"):
                return self.expression(
                    exp.MergeTreeTTLAction, this=this, recompress=self._parse_bitwise()
                )
            if self._match_text_seq("TO", "DISK"):
                return self.expression(
                    exp.MergeTreeTTLAction, this=this, to_disk=self._parse_string()
                )
            if self._match_text_seq("TO", "VOLUME"):
                return self.expression(
                    exp.MergeTreeTTLAction, this=this, to_volume=self._parse_string()
                )

            return this

        expressions = self._parse_csv(_parse_ttl_action)
        where = self._parse_where()
        group = self._parse_group()

        aggregates = None
        if group and self._match(TokenType.SET):
            aggregates = self._parse_csv(self._parse_set_item)

        return self.expression(
            exp.MergeTreeTTL,
            expressions=expressions,
            where=where,
            group=group,
            aggregates=aggregates,
        )

    def _parse_statement(self) -> t.Optional[exp.Expression]:
        """
        解析SQL语句的主入口

        根据当前token类型选择相应的解析器，支持标准SQL语句和命令。
        如果遇到不支持的语法，会回退到Command解析。

        Returns:
            解析后的表达式对象，如果没有更多内容则返回None
        """
        if self._curr is None:
            return None

        # 尝试匹配标准SQL语句解析器
        if self._match_set(self.STATEMENT_PARSERS):
            comments = self._prev_comments
            stmt = self.STATEMENT_PARSERS[self._prev.token_type](self)
            stmt.add_comments(comments, prepend=True)
            return stmt

        # 如果标准解析器失败，尝试作为命令解析
        if self._match_set(self.dialect.tokenizer_class.COMMANDS):
            return self._parse_command()

        # 最后尝试解析为表达式或SELECT语句
        expression = self._parse_expression()
        expression = self._parse_set_operations(expression) if expression else self._parse_select()
        return self._parse_query_modifiers(expression)

    def _parse_drop(self, exists: bool = False) -> exp.Drop | exp.Command:
        """
        解析DROP语句

        支持删除表、列、索引、数据库等对象。
        包含各种方言特定的选项如CONCURRENTLY、CASCADE等。

        Args:
            exists: 是否已经解析了IF EXISTS子句

        Returns:
            Drop表达式或Command表达式
        """
        start = self._prev
        # 解析TEMPORARY关键字
        temporary = self._match(TokenType.TEMPORARY)
        # 解析EXTERNAL关键字
        external = self._match(TokenType.EXTERNAL)
        # 解析MATERIALIZED关键字
        materialized = self._match_text_seq("MATERIALIZED")

        # 获取要删除的对象类型
        kind = self._match_set(self.CREATABLES) and self._prev.text.upper()
        if not kind:
            return self._parse_as_command(start)

        # 解析CONCURRENTLY关键字（PostgreSQL特性）
        concurrently = self._match_text_seq("CONCURRENTLY")
        # 解析IF EXISTS子句
        if_exists = exists or self._parse_exists()

        # 根据类型解析不同的内容
        if kind == "COLUMN":
            this = self._parse_column()
        else:
            this = self._parse_table_parts(
                schema=True, is_db_reference=self._prev.token_type == TokenType.SCHEMA
            )

        # 解析ON子句（用于集群等）
        cluster = self._parse_on_property() if self._match(TokenType.ON) else None

        # 解析可选的类型列表（用括号包围）
        if self._match(TokenType.L_PAREN, advance=False):
            expressions = self._parse_wrapped_csv(self._parse_types)
        else:
            expressions = None

        return self.expression(
            exp.Drop,
            exists=if_exists,
            this=this,
            expressions=expressions,
            kind=self.dialect.CREATABLE_KIND_MAPPING.get(kind) or kind,
            temporary=temporary,
            external=external,
            materialized=materialized,
            cascade=self._match_text_seq("CASCADE"),
            constraints=self._match_text_seq("CONSTRAINTS"),
            purge=self._match_text_seq("PURGE"),
            cluster=cluster,
            concurrently=concurrently,
        )

    def _parse_exists(self, not_: bool = False) -> t.Optional[bool]:
        """
        解析IF EXISTS或IF NOT EXISTS子句

        Args:
            not_: 是否期望NOT关键字

        Returns:
            如果匹配成功则返回True，否则返回None
        """
        # 记录进入函数时的位置，用于失败时回退
        start_index = self._index

        # 检查是否匹配"IF"文本序列
        if not self._match_text_seq("IF"):
            # 失败时回退到进入函数时的位置
            self._retreat(start_index)
            return False

        # 如果not_为False，则不需要检查NOT标记
        # 如果not_为True，则需要检查是否存在NOT标记
        if not_ and not self._match(TokenType.NOT):
            # 失败时回退到进入函数时的位置
            self._retreat(start_index)
            return False

        # 最后检查是否存在EXISTS标记
        if not self._match(TokenType.EXISTS):
            # 失败时回退到进入函数时的位置
            self._retreat(start_index)
            return False

        return True

        # return (
        #     self._match_text_seq("IF")
        #     and (not not_ or self._match(TokenType.NOT))
        #     and self._match(TokenType.EXISTS)
        # )

    def _parse_create(self) -> exp.Create | exp.Command:
        """
        解析CREATE语句

        这是最复杂的解析方法之一，支持创建表、视图、函数、索引、数据库等。
        包含多种方言特定的语法和选项。

        Returns:
            Create表达式或Command表达式
        """
        # 注意：这里不能为None，因为我们已经匹配了一个语句解析器
        start = self._prev

        # 解析REPLACE关键字（CREATE OR REPLACE）
        replace = (
            start.token_type == TokenType.REPLACE
            or self._match_pair(TokenType.OR, TokenType.REPLACE)
            or self._match_pair(TokenType.OR, TokenType.ALTER)
        )
        # 解析REFRESH关键字
        refresh = self._match_pair(TokenType.OR, TokenType.REFRESH)

        # 解析UNIQUE关键字
        unique = self._match(TokenType.UNIQUE)

        # 解析列存储相关的集群选项（SQL Server特性）
        # CLUSTERED COLUMNSTORE - 表示聚集列存储索引
        # NONCLUSTERED COLUMNSTORE - 表示非聚集列存储索引
        if self._match_text_seq("CLUSTERED", "COLUMNSTORE"):
            clustered = True
        elif self._match_text_seq("NONCLUSTERED", "COLUMNSTORE") or self._match_text_seq(
            "COLUMNSTORE"
        ):
            clustered = False
        else:
            clustered = None

        # 处理TABLE FUNCTION的特殊情况
        # 某些方言允许"CREATE TABLE FUNCTION"语法，这里需要特殊处理
        if self._match_pair(TokenType.TABLE, TokenType.FUNCTION, advance=False):
            self._advance()

        # 初始化属性和创建对象类型标记
        properties = None
        create_token = self._match_set(self.CREATABLES) and self._prev

        # 如果没有匹配到标准的可创建对象类型，尝试解析属性
        if not create_token:
            # exp.Properties.Location.POST_CREATE
            # 某些方言允许在CREATE和对象类型之间有属性
            # 例如：CREATE EXTERNAL TABLE（exp.Properties.Location.POST_CREATE）
            properties = self._parse_properties()
            create_token = self._match_set(self.CREATABLES) and self._prev

            # 如果仍然没有找到有效的创建类型，则作为命令处理
            if not properties or not create_token:
                return self._parse_as_command(start)

        # 解析CONCURRENTLY关键字（PostgreSQL特性，用于并发创建索引）
        concurrently = self._match_text_seq("CONCURRENTLY")
        # 解析IF NOT EXISTS子句，防止重复创建
        exists = self._parse_exists(not_=True)

        # 初始化各种可能的CREATE语句组件
        this = None                    # 要创建的对象（表名、函数名等）
        expression: t.Optional[exp.Expression] = None  # 创建语句的主体表达式
        indexes = None                 # 表的索引列表
        no_schema_binding = None       # 视图的无模式绑定选项
        begin = None                   # 函数体的BEGIN标记
        end = None                     # 函数体的END标记
        clone = None                   # 克隆选项

        def extend_props(temp_props: t.Optional[exp.Properties]) -> None:
            """
            扩展属性的内部函数

            将临时属性合并到主属性列表中。这是必要的，因为CREATE语句
            可以在多个位置包含属性，需要将它们合并到一个列表中。

            Args:
                temp_props: 要合并的临时属性
            """
            nonlocal properties
# DEBUG: extend_props called
            # print(f"DEBUG: extend_props called with: {temp_props}")
            # print(f"DEBUG: current properties: {properties}")
            if properties and temp_props:
                # 如果已有属性，则扩展现有列表
                properties.expressions.extend(temp_props.expressions)
                # print(f"DEBUG: Extended existing properties")
            elif temp_props:
                # 如果没有现有属性，则直接使用新属性
                properties = temp_props
                # print(f"DEBUG: Set new properties")

        # ======= 根据创建的对象类型进行不同的解析分支 =======

        if create_token.token_type in (TokenType.FUNCTION, TokenType.PROCEDURE):
            # === 分支1: 创建函数或存储过程 ===
            # 解析函数签名（名称、参数、返回类型等）
            this = self._parse_user_defined_function(kind=create_token.token_type)

            # 解析函数类型签名后的属性（exp.Properties.Location.POST_SCHEMA）
            # "schema"在这里指的是UDF的类型签名，例如：LANGUAGE SQL, DETERMINISTIC等
            extend_props(self._parse_properties())

            # 解析ALIAS子句（主要用于BigQuery等云数据库）
            # 例如：CREATE FUNCTION f() RETURNS INT ALIAS 'some_alias'
            expression = self._match(TokenType.ALIAS) and self._parse_heredoc()
            extend_props(self._parse_properties())

            # 如果没有ALIAS表达式，需要解析函数体
            if not expression:
                if self._match(TokenType.COMMAND):
                    # 如果遇到COMMAND token，作为命令处理
                    expression = self._parse_as_command(self._prev)
                else:
                    # 解析函数体的开始和结束标记
                    begin = self._match(TokenType.BEGIN)  # SQL函数的BEGIN关键字
                    return_ = self._match_text_seq("RETURN")  # RETURN语句

                    if self._match(TokenType.STRING, advance=False):
                        # 处理BigQuery的JavaScript UDF定义，函数体是字符串，后面跟OPTIONS属性
                        # 例如：CREATE FUNCTION f() RETURNS STRING LANGUAGE js AS "return 'hello';" OPTIONS(...)
                        # 参考：https://cloud.google.com/bigquery/docs/reference/standard-sql/data-definition-language#create_function_statement
                        expression = self._parse_string()
                        extend_props(self._parse_properties())
                    else:
                        # 解析标准的函数表达式（SQL函数体）
                        expression = self._parse_user_defined_function_expression()

                    # 匹配函数体的结束标记
                    end = self._match_text_seq("END")

                    # 如果有RETURN语句，将表达式包装为Return节点
                    if return_:
                        expression = self.expression(exp.Return, this=expression)
        elif create_token.token_type == TokenType.INDEX:
            # === 分支2: 创建索引 ===
            # PostgreSQL允许匿名索引，例如：CREATE INDEX IF NOT EXISTS ON t(c)
            # 这种情况下没有索引名称，直接跟ON子句
            if not self._match(TokenType.ON):
                # 标准索引语法：CREATE INDEX index_name ON table_name(columns)
                if self.SUPPORT_INDEX_NAME_WITH_DOT:
                    index = self._parse_index_identifier()  # 解析索引名称
                else:
                    index = self._parse_id_var()  # 解析索引名称
                anonymous = False
            else:
                # 匿名索引语法：CREATE INDEX ON table_name(columns)
                index = None
                anonymous = True

            # 解析完整的索引定义（表名、列、选项等）
            this = self._parse_index(index=index, anonymous=anonymous)
        elif create_token.token_type in self.DB_CREATABLES:
            # === 分支3: 创建数据库对象（表、视图、数据库、模式等） ===
            # DB_CREATABLES包括：DATABASE, SCHEMA, TABLE, VIEW等数据库级别的对象

            # 解析对象名称（可能包含模式限定符）
            # 例如：schema.table_name 或者 database.schema.table_name
            table_parts = self._parse_table_parts(
                schema=True, is_db_reference=create_token.token_type == TokenType.SCHEMA
            )

            # 解析名称后的属性（exp.Properties.Location.POST_NAME）
            # 某些方言允许在对象名称后直接跟属性，用逗号分隔
            self._match(TokenType.COMMA)
            extend_props(self._parse_properties(before=True))

            # 解析对象的结构定义（列定义、约束等）
            # 对于表：列定义和约束；对于视图：查询定义等
            this = self._parse_schema(this=table_parts)

            # 解析结构定义后的属性（exp.Properties.Location.POST_SCHEMA and POST_WITH）
            # 例如：表的存储引擎、字符集、分区等属性
            extend_props(self._parse_properties())

            # 检查是否有ALIAS关键字
            has_alias = self._match(TokenType.ALIAS)
            if not self._match_set(self.DDL_SELECT_TOKENS, advance=False):
                # 解析别名后的属性（exp.Properties.Location.POST_ALIAS）
                extend_props(self._parse_properties())

            # 根据具体的对象类型解析不同的内容
            if create_token.token_type == TokenType.SEQUENCE:
                # 序列需要数据类型定义
                # 例如：CREATE SEQUENCE seq_name AS INTEGER
                expression = self._parse_types()
                props = self._parse_properties()
                if props:
                    sequence_props = exp.SequenceProperties()
                    options = []
                    for prop in props:
                        if isinstance(prop, exp.SequenceProperties):
                            for arg, value in prop.args.items():
                                if arg == "options":
                                    options.extend(value)
                                else:
                                    sequence_props.set(arg, value)
                            prop.pop()

                    if options:
                        sequence_props.set("options", options)

                    props.append("expressions", sequence_props)
                    extend_props(props)
            else:
                # 解析DDL SELECT语句（主要用于CREATE TABLE AS SELECT）
                # 例如：CREATE TABLE new_table AS SELECT * FROM old_table
                expression = self._parse_ddl_select()

                # 某些方言也支持使用表作为别名的替代方案，而不是SELECT语句
                # 这里我们将其作为替代方案回退
                # 例如：CREATE TABLE new_table LIKE old_table
                if not expression and has_alias:
                    expression = self._try_parse(self._parse_table_parts)

            # === 针对不同对象类型的特殊处理 ===
            if create_token.token_type == TokenType.TABLE:
                # 表创建的特殊处理：解析表达式后的属性和索引定义
                # 解析表达式后的属性（exp.Properties.Location.POST_EXPRESSION）
                # 例如：分区定义、存储选项等
                extend_props(self._parse_properties())

                # 解析表的索引定义列表
                # 某些方言允许在CREATE TABLE语句中直接定义索引
                indexes = []
                while True:
                    index = self._parse_index()

                    # 解析索引后的属性（exp.Properties.Location.POST_INDEX）
                    extend_props(self._parse_properties())
                    if not index:
                        # 没有更多索引，退出循环
                        break
                    else:
                        # 索引之间用逗号分隔
                        self._match(TokenType.COMMA)
                        indexes.append(index)
            elif create_token.token_type == TokenType.VIEW:
                # 视图创建的特殊处理
                # 解析"WITH NO SCHEMA BINDING"选项（SQL Server特性）
                # 表示视图不绑定到底层表的模式，允许底层表结构变化
                if self._match_text_seq("WITH", "NO", "SCHEMA", "BINDING"):
                    no_schema_binding = True
            elif create_token.token_type in (TokenType.SINK, TokenType.SOURCE):
                # 流处理系统（如Apache Flink）的SINK和SOURCE对象
                # 需要额外的连接器属性定义
                extend_props(self._parse_properties())

            # 解析SHALLOW关键字（用于克隆操作）
            # SHALLOW表示浅克隆，只复制结构不复制数据
            shallow = self._match_text_seq("SHALLOW")

            # 解析CLONE或COPY关键字
            # 支持从现有对象克隆创建新对象
            if self._match_texts(self.CLONE_KEYWORDS):
                # 判断是COPY还是CLONE操作
                copy = self._prev.text.lower() == "copy"
                # 解析要克隆的源表
                clone = self.expression(
                    exp.Clone, this=self._parse_table(schema=True), shallow=shallow, copy=copy
                )

        # ======= 最终验证和CREATE表达式构建 =======

        # 检查是否还有未处理的token
        # 如果当前token不是右括号或逗号，说明有无法解析的语法，回退到命令模式
        if self._curr and not self._match_set((TokenType.R_PAREN, TokenType.COMMA), advance=False):
            return self._parse_as_command(start)

        # 获取创建对象类型的文本表示，并转换为大写
        create_kind_text = create_token.text.upper()

        # 构建并返回完整的CREATE表达式
        # 包含所有解析到的组件和选项
        # print(f"DEBUG: Final properties before Create: {properties}")
        return self.expression(
            exp.Create,
            this=this,                    # 要创建的对象（表名、函数名等）
            kind=self.dialect.CREATABLE_KIND_MAPPING.get(create_kind_text) or create_kind_text,  # 对象类型
            replace=replace,              # 是否为CREATE OR REPLACE
            refresh=refresh,              # 是否为CREATE OR REFRESH
            unique=unique,                # 是否为UNIQUE索引
            expression=expression,        # 主体表达式（函数体、SELECT语句等）
            exists=exists,                # IF NOT EXISTS选项
            properties=properties,        # 属性列表
            indexes=indexes,              # 索引定义列表
            no_schema_binding=no_schema_binding,  # 无模式绑定选项
            begin=begin,                  # 函数体BEGIN标记
            end=end,                      # 函数体END标记
            clone=clone,                  # 克隆选项
            concurrently=concurrently,    # 并发创建选项
            clustered=clustered,          # 聚集/非聚集选项
        )

    def _parse_sequence_properties(self) -> t.Optional[exp.SequenceProperties]:
        """
        解析`CREATE SEQUENCE`语句中的属性部分。

        支持如下属性（部分为方言专属）：
        - INCREMENT [ BY ] <value>
        - MINVALUE / MAXVALUE <value>
        - START [ WITH ] <value>
        - CACHE <value>  （T-SQL 中CACHE可省略数值，表示动态初始化）
        - OWNED BY <table.column> / OWNED BY NONE
        - 其他方言特定选项（通过 `CREATE_SEQUENCE` 字典控制）

        解析过程中会持续尝试匹配属性，直到不再命中已知关键字。
        所有解析到的键值对最终封装为 `exp.SequenceProperties` 节点返回。

        Returns:
            SequenceProperties 节点，若无属性被解析则返回 ``None``。
        """
        seq = exp.SequenceProperties()

        options = []
        index = self._index

        while self._curr:
            self._match(TokenType.COMMA)
            if self._match_text_seq("INCREMENT"):
                self._match_text_seq("BY")
                self._match_text_seq("=")
                seq.set("increment", self._parse_term())
            elif self._match_text_seq("MINVALUE"):
                seq.set("minvalue", self._parse_term())
            elif self._match_text_seq("MAXVALUE"):
                seq.set("maxvalue", self._parse_term())
            elif self._match(TokenType.START_WITH) or self._match_text_seq("START"):
                self._match_text_seq("=")
                seq.set("start", self._parse_term())
            elif self._match_text_seq("CACHE"):
                # T-SQL allows empty CACHE which is initialized dynamically
                seq.set("cache", self._parse_number() or True)
            elif self._match_text_seq("OWNED", "BY"):
                # "OWNED BY NONE" is the default
                seq.set("owned", None if self._match_text_seq("NONE") else self._parse_column())
            else:
                opt = self._parse_var_from_options(self.CREATE_SEQUENCE, raise_unmatched=False)
                if opt:
                    options.append(opt)
                else:
                    break

        seq.set("options", options if options else None)
        return None if self._index == index else seq

    def _parse_property_before(self) -> t.Optional[exp.Expression]:
        """
        解析位于 Teradata `CREATE TABLE ...` 语句 *前置位置* 的属性。

        Teradata 允许在列定义 **之前** 使用类似 `, NO BEFORE ...` 的关键字对表进行属性修饰，
        这些关键字往往和常规属性解析逻辑不同。

        该函数的设计思路：
        1. 先消耗一个可选的逗号分隔符，此时已知上一个循环读取完毕。
        2. 通过 ``kwargs`` 记录一组可能出现且需要额外语义信息的前缀关键字。
           例如 `NO`, `DUAL`, `BEFORE/AFTER` 等。
        3. 如果当前 token 命中 ``PROPERTY_PARSERS`` 中已注册的属性解析器，
           则按照是否命中上述关键字，将其作为命名参数传递给属性解析器。
           这样可以避免在具体属性解析函数中再去判断上下文关键字。
        4. 若当前 token 不是受支持的属性名，则返回 ``None``，
           由调用方决定是否继续其他 fall-back 解析逻辑。

        Returns:
            解析得到的属性表达式，或 ``None``（无法解析）。
        """
        # only used for teradata currently
        self._match(TokenType.COMMA)

        kwargs = {
            "no": self._match_text_seq("NO"),
            "dual": self._match_text_seq("DUAL"),
            "before": self._match_text_seq("BEFORE"),
            "default": self._match_text_seq("DEFAULT"),
            "local": (self._match_text_seq("LOCAL") and "LOCAL")
            or (self._match_text_seq("NOT", "LOCAL") and "NOT LOCAL"),
            "after": self._match_text_seq("AFTER"),
            "minimum": self._match_texts(("MIN", "MINIMUM")),
            "maximum": self._match_texts(("MAX", "MAXIMUM")),
        }

        if self._match_texts(self.PROPERTY_PARSERS):
            parser = self.PROPERTY_PARSERS[self._prev.text.upper()]
            try:
                return parser(self, **{k: v for k, v in kwargs.items() if v})
            except TypeError:
                self.raise_error(f"Cannot parse property '{self._prev.text}'")

        return None

    def _parse_wrapped_properties(self) -> t.List[exp.Expression]:
        """解析形如 `(prop1, prop2, ...)` 的 *括号包裹* 属性列表。"""
        return self._parse_wrapped_csv(self._parse_property)

    def _parse_property(self) -> t.Optional[exp.Expression]:
        """
        通用属性解析入口。

        按照以下优先级尝试解析当前属性：
        1. 若 token 在 ``PROPERTY_PARSERS`` 中，直接调用对应解析器。
        2. 若以 `DEFAULT` 开头，则将 `default=True` 传入解析器（Snowflake/Hive 等方言）。
        3. 处理 ClickHouse `COMPOUND SORTKEY` 等复合关键字属性。
        4. 处理 `SQL SECURITY [DEFINER]` 特性（MySQL / Oracle）。
        5. 如果以上均不匹配，则尝试解析 *键=值* 形式的通用属性；
           若键后无 `=`，则回退为 *SEQUENCE* 属性解析。
        """
        if self._match_texts(self.PROPERTY_PARSERS):
            # print(f"DEBUG: Matched property parser for: {self._prev.text.upper()}")
            return self.PROPERTY_PARSERS[self._prev.text.upper()](self)

        if self._match(TokenType.DEFAULT) and self._match_texts(self.PROPERTY_PARSERS):
            return self.PROPERTY_PARSERS[self._prev.text.upper()](self, default=True)

        if self._match_text_seq("COMPOUND", "SORTKEY"):
            return self._parse_sortkey(compound=True)

        if self._match_text_seq("SQL", "SECURITY"):
            return self.expression(exp.SqlSecurityProperty, definer=self._match_text_seq("DEFINER"))

        # 解析表的读写属性,目前仅针对Gaussdb的外表
        if self._match_text_seq("READ", "ONLY"):
            return self.expression(exp.TableReadWriteProperty, this="READ ONLY")
        if self._match_text_seq("WRITE", "ONLY"):
            return self.expression(exp.TableReadWriteProperty, this="WRITE ONLY")
        if self._match_text_seq("READ", "WRITE"):
            return self.expression(exp.TableReadWriteProperty, this="READ WRITE")

        index = self._index

        seq_props = self._parse_sequence_properties()
        if seq_props:
            return seq_props

        self._retreat(index)
        key = self._parse_column()

        if not self._match(TokenType.EQ):
            self._retreat(index)
            return None

        # Transform the key to exp.Dot if it's dotted identifiers wrapped in exp.Column or to exp.Var otherwise
        if isinstance(key, exp.Column):
            key = key.to_dot() if len(key.parts) > 1 else exp.var(key.name)

        value = self._parse_bitwise() or self._parse_var(any_token=True)

        # Transform the value to exp.Var if it was parsed as exp.Column(exp.Identifier())
        if isinstance(value, exp.Column):
            value = exp.var(value.name)

        return self.expression(exp.Property, this=key, value=value)

    def _parse_stored(self) -> t.Union[exp.FileFormatProperty, exp.StorageHandlerProperty]:
        """
        解析 `STORED AS/ BY` 存储格式属性（Hive / Spark）。

        - `STORED BY <handler>` => `StorageHandlerProperty`
        - `STORED AS INPUTFORMAT ... OUTPUTFORMAT ...` => 解析为 `InputOutputFormat` 节点
        - 兼容旧语法 `STORED AS PARQUET` 等简单格式字符串
        """
        if self._match_text_seq("BY"):
            return self.expression(exp.StorageHandlerProperty, this=self._parse_var_or_string())

        self._match(TokenType.ALIAS)
        input_format = self._parse_string() if self._match_text_seq("INPUTFORMAT") else None
        output_format = self._parse_string() if self._match_text_seq("OUTPUTFORMAT") else None

        return self.expression(
            exp.FileFormatProperty,
            this=(
                self.expression(
                    exp.InputOutputFormat,
                    input_format=input_format,
                    output_format=output_format,
                )
                if input_format or output_format
                else self._parse_var_or_string() or self._parse_number() or self._parse_id_var()
            ),
            hive_format=True,
        )

    def _parse_unquoted_field(self) -> t.Optional[exp.Expression]:
        """
        解析一个字段名称；若为未加引号的标识符，则规范化为变量(exp.Var)。
        这么做的原因：不同方言中未加引号的标识符通常与变量/关键字解析一致，
        统一成 Var 可降低后续生成与比较的歧义。
        """
        field = self._parse_field()
        # 若是未加引号的标识符，转换为 Var，避免引号语义差异带来的歧义
        if isinstance(field, exp.Identifier) and not field.quoted:
            field = exp.var(field)

        return field

    def _parse_property_assignment(self, exp_class: t.Type[E], **kwargs: t.Any) -> E:
        """
        解析通用的 `KEY = VALUE` 形式的属性赋值。
        逻辑说明：
        - 先匹配等号 `=`，再可选匹配 `AS/ALIAS`（部分方言允许写成 `KEY = ALIAS value`）。
        - 值部分通过 `_parse_unquoted_field` 保证未加引号的标识符被规范化为 Var。
        """
        self._match(TokenType.EQ)
        self._match(TokenType.ALIAS)

        return self.expression(exp_class, this=self._parse_unquoted_field(), **kwargs)

    def _parse_properties(self, before: t.Optional[bool] = None) -> t.Optional[exp.Properties]:
        """
        解析属性列表，并将其聚合为 `exp.Properties`。

        关键逻辑：
        - `before=True` 时，走 Teradata 等方言的"前置属性"分支；否则走常规属性分支。
        - 单个解析结果可能返回一个或多个属性（如复合属性），因此统一用 `ensure_list` 展开。
        - 空列表返回 `None`，避免生成空的 Properties 节点，便于上层判空处理。
        """
        properties = []
        while True:
            if before:
                prop = self._parse_property_before()
            else:
                prop = self._parse_property()
            if not prop:
                break
            for p in ensure_list(prop):
                properties.append(p)

        if properties:
            return self.expression(exp.Properties, expressions=properties)

        return None

    def _parse_fallback(self, no: bool = False) -> exp.FallbackProperty:
        """
        解析 `FALLBACK [PROTECTION]`（Teradata）属性。
        - `no=True` 表示 `NO FALLBACK`。
        - `PROTECTION` 为可选修饰，表示启用保护模式。
        """
        return self.expression(
            exp.FallbackProperty, no=no, protection=self._match_text_seq("PROTECTION")
        )

    def _parse_security(self) -> t.Optional[exp.SecurityProperty]:
        """
        解析安全属性，如 `SQL SECURITY { DEFINER | INVOKER }` 或 `NONE`。

        含义与原因：
        - 不同方言（如 MySQL、Oracle）允许为过程/函数/视图指定执行权限的主体。
        - 这里直接匹配受支持的枚举值，命中后用大写规范化，构造 `SecurityProperty`。
        """
        if self._match_texts(("NONE", "DEFINER", "INVOKER")):
            # _match_texts 命中后，_prev 即为刚刚匹配到的那个 token
            security_specifier = self._prev.text.upper()
            return self.expression(exp.SecurityProperty, this=security_specifier)
        return None

    def _parse_settings_property(self) -> exp.SettingsProperty:
        """
        解析 `SETTINGS (k = v, ...)` 或等价语法的键值对属性集合。
        关键点：使用 `_parse_csv(self._parse_assignment)` 以支持以逗号分隔的 `k=v` 列表。
        """
        return self.expression(
            exp.SettingsProperty, expressions=self._parse_csv(self._parse_assignment)
        )

    def _parse_volatile_property(self) -> exp.VolatileProperty | exp.StabilityProperty:
        """
        解析 VOLATILE/稳定性相关属性（主要见于 Teradata）。

        关键逻辑说明：
        - 通过回看前两个 token 来判断上下文是否允许 `VOLATILE` 作为属性出现
          （一些方言需要 `CREATE [VOLATILE] TABLE` 这样的上下文）。
        - 如果上下文满足 `PRE_VOLATILE_TOKENS`，返回 `VolatileProperty`，
          否则退化为一般的 `StabilityProperty("VOLATILE")`（便于统一下游生成）。
        """
        if self._index >= 2:
            pre_volatile_token = self._tokens[self._index - 2]
        else:
            pre_volatile_token = None

        # 仅当前序 token 属于允许 VOLATILE 出现的位置时，认定为 VOLATILE 属性
        if pre_volatile_token and pre_volatile_token.token_type in self.PRE_VOLATILE_TOKENS:
            return exp.VolatileProperty()

        # 否则作为稳定性属性对待，保留原始字面量以便方言生成
        return self.expression(exp.StabilityProperty, this=exp.Literal.string("VOLATILE"))

    def _parse_retention_period(self) -> exp.Var:
        """
        解析 T-SQL 的 `HISTORY_RETENTION_PERIOD` 值：
        形如 `{INFINITE | <number> DAY | DAYS | MONTH ...}`。

        关键逻辑：
        - 数值部分可选，缺省时表示 `INFINITE` 一类的文字；
        - 单位使用 `_parse_var(any_token=True)`，原因是单位允许是标识符枚举而非普通字符串。
        - 返回合成后的 `exp.Var`，便于后续直接序列化为 `<number> <UNIT>`。
        """
        # Parse TSQL's HISTORY_RETENTION_PERIOD: {INFINITE | <number> DAY | DAYS | MONTH ...}
        number = self._parse_number()
        number_str = f"{number} " if number else ""
        unit = self._parse_var(any_token=True)
        return exp.var(f"{number_str}{unit}")

    def _parse_system_versioning_property(
        self, with_: bool = False
    ) -> exp.WithSystemVersioningProperty:
        """
        解析 `SYSTEM_VERSIONING` 属性（T-SQL）：
        - 形如：`WITH (SYSTEM_VERSIONING = ON ( ... ))` 或 `WITH (SYSTEM_VERSIONING = OFF)`。

        关键逻辑与原因：
        - 默认 `on=True`，若显式匹配到 `OFF`，则置为 False 并提前返回；
        - `ON` 分支内支持括号包裹的详细配置：
          - `HISTORY_TABLE = <schema.table>`：系统版本表；
          - `DATA_CONSISTENCY_CHECK = { ON | OFF }`：数据一致性检查；
          - `HISTORY_RETENTION_PERIOD = <number> <unit>`：历史保留期；
        - 每项之间允许逗号分隔，循环读取直到遇到右括号。
        """
        self._match(TokenType.EQ)
        prop = self.expression(
            exp.WithSystemVersioningProperty,
            **{  # type: ignore
                "on": True,
                "with": with_,
            },
        )

        # 显式关闭：直接返回，后续不解析细项
        if self._match_text_seq("OFF"):
            prop.set("on", False)
            return prop

        self._match(TokenType.ON)
        if self._match(TokenType.L_PAREN):
            while self._curr and not self._match(TokenType.R_PAREN):
                if self._match_text_seq("HISTORY_TABLE", "="):
                    prop.set("this", self._parse_table_parts())
                elif self._match_text_seq("DATA_CONSISTENCY_CHECK", "="):
                    # 这里用 _advance_any 读取枚举值（ON/OFF），并统一为大写字符串
                    prop.set("data_consistency", self._advance_any() and self._prev.text.upper())
                elif self._match_text_seq("HISTORY_RETENTION_PERIOD", "="):
                    prop.set("retention_period", self._parse_retention_period())

                # 允许逗号分隔多个选项
                self._match(TokenType.COMMA)

        return prop

    def _parse_data_deletion_property(self) -> exp.DataDeletionProperty:
        """
        解析 `DATA DELETION`（删除/保留策略）属性。

        关键逻辑：
        - 等号后先判断是 ON 还是 OFF，默认视为 ON（若未命中 OFF）。
        - 可选括号体内支持：
          - `FILTER_COLUMN = <col>`：策略作用的列；
          - `RETENTION_PERIOD = <number> <unit>`：保留期；
        - 多项之间使用逗号分隔，循环读取直到右括号。
        """
        self._match(TokenType.EQ)
        on = self._match_text_seq("ON") or not self._match_text_seq("OFF")
        prop = self.expression(exp.DataDeletionProperty, on=on)

        if self._match(TokenType.L_PAREN):
            while self._curr and not self._match(TokenType.R_PAREN):
                if self._match_text_seq("FILTER_COLUMN", "="):
                    prop.set("filter_column", self._parse_column())
                elif self._match_text_seq("RETENTION_PERIOD", "="):
                    prop.set("retention_period", self._parse_retention_period())

                # 允许逗号分隔多个选项
                self._match(TokenType.COMMA)

        return prop

    def _parse_distributed_property(self) -> exp.DistributedByProperty:
        """
        解析分布属性（典型于数据仓库/MPP，如 `DISTRIBUTED BY HASH(...) BUCKETS n`）。

        关键逻辑与原因：
        - 默认分布方式为 `HASH`，若匹配到 `BY RANDOM` 则切换为 `RANDOM`（随机分布）。
        - `BY HASH` 时解析括号内的分布键列表，使用 `_parse_wrapped_csv` 统一处理逗号表达式。
        - `BUCKETS` 数量：若未显式给出，则视为 `AUTO`，因此只有在遇到 `BUCKETS` 且未匹配 `AUTO` 时才解析数字。
        - 同时解析可能存在的 `ORDER BY`（部分方言支持）。
        """
        kind = "HASH"
        expressions: t.Optional[t.List[exp.Expression]] = None
        if self._match_text_seq("BY", "HASH"):
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

    def _parse_composite_key_property(self, expr_type: t.Type[E]) -> E:
        """
        解析复合键属性，例如 `PRIMARY KEY(col1, col2)` 或 `UNIQUE KEY(...)` 等。

        关键逻辑：
        - 显式匹配关键字 `KEY` 以确保语义明确；
        - 使用 `_parse_wrapped_id_vars()` 解析括号中的标识符列表，
          统一处理逗号分隔及标识符大小写/引号问题；
        - 返回传入的目标表达式类型（`expr_type`），以便在调用处复用为不同键约束。
        """
        self._match_text_seq("KEY")
        expressions = self._parse_wrapped_id_vars()
        return self.expression(expr_type, expressions=expressions)

    def _parse_with_property(self) -> t.Optional[exp.Expression] | t.List[exp.Expression]:
        """
        解析 WITH 子句下的诸多属性，按优先级逐项尝试：
        1) `(SYSTEM_VERSIONING ...)`：优先匹配带括号的系统版本化配置；
        2) `(<props...>)`：通用括号包裹属性列表；
        3) `JOURNAL`、视图特性、`DATA/NO DATA`、`SERDE_PROPERTIES`；
        4) `SCHEMA` 绑定、`PROCEDURE OPTIONS`；
        5) 否则尝试解析 `WITH ISOLATED LOADING`（Teradata）。

        说明：匹配顺序很重要，越具体/越有歧义的形态优先解析，避免被通用分支吞掉。
        """
        if self._match_text_seq("(", "SYSTEM_VERSIONING"):
            prop = self._parse_system_versioning_property(with_=True)
            self._match_r_paren()
            return prop

        if self._match(TokenType.L_PAREN, advance=False):
            return self._parse_wrapped_properties()

        if self._match_text_seq("JOURNAL"):
            return self._parse_withjournaltable()

        if self._match_texts(self.VIEW_ATTRIBUTES):
            return self.expression(exp.ViewAttributeProperty, this=self._prev.text.upper())

        if self._match_text_seq("DATA"):
            return self._parse_withdata(no=False)
        elif self._match_text_seq("NO", "DATA"):
            return self._parse_withdata(no=True)

        if self._match(TokenType.SERDE_PROPERTIES, advance=False):
            return self._parse_serde_properties(with_=True)

        if self._match(TokenType.SCHEMA):
            return self.expression(
                exp.WithSchemaBindingProperty,
                this=self._parse_var_from_options(self.SCHEMA_BINDING_OPTIONS),
            )

        if self._match_texts(self.PROCEDURE_OPTIONS, advance=False):
            return self.expression(
                exp.WithProcedureOptions, expressions=self._parse_csv(self._parse_procedure_option)
            )


        withisolatedloading = self._parse_withisolatedloading()
        if withisolatedloading:
            return withisolatedloading
        # 补充构建外表时 WITH作为LOG INTO用的情况
        elif self._curr.token_type == TokenType.VAR and self._next.text == "LOG":
            return self.expression(exp.WithJournalTableProperty, this=self._parse_table_parts())
            # self._advance()
            # return self.expression(
            #     exp.Property, this = "LOG INTO", value = exp.Var(self._prev.text))

        if not self._next:
            return None
        
        return None

    def _parse_procedure_option(self) -> exp.Expression | None:
        """
        解析存储过程选项，目前主要处理 `EXECUTE AS`：
        - 优先尝试 `EXECUTE AS { CALLER | SELF | OWNER }`；
        - 如果不是枚举值，则允许使用字符串指定主体（兼容某些方言写法）。
        否则退回到通用的 `PROCEDURE_OPTIONS` 列表中解析。
        """
        if self._match_text_seq("EXECUTE", "AS"):
            return self.expression(
                exp.ExecuteAsProperty,
                this=self._parse_var_from_options(self.EXECUTE_AS_OPTIONS, raise_unmatched=False)
                or self._parse_string(),
            )

        return self._parse_var_from_options(self.PROCEDURE_OPTIONS)

    # https://dev.mysql.com/doc/refman/8.0/en/create-view.html
    def _parse_definer(self) -> t.Optional[exp.DefinerProperty]:
        """
        解析 MySQL `DEFINER = user@host` 语法（常用于 `CREATE VIEW` 等）。

        关键逻辑：
        - `user` 与 `host` 之间由 `@` 连接；为兼容不同 tokenizer，这里用 `PARAMETER`（通常对应 `@`）进行匹配；
        - `host` 既可以是标识符，也允许 `*`（部分语法使用 `%` 等通配，源码中以 `MOD` 兼容读取上一 token 文本）。
        - 缺少 `user` 或 `host` 则返回 `None`，交由上层决定错误处理或容错。
        """
        self._match(TokenType.EQ)

        user = self._parse_id_var()
        self._match(TokenType.PARAMETER)
        host = self._parse_id_var() or (self._match(TokenType.MOD) and self._prev.text)

        if not user or not host:
            return None

        return exp.DefinerProperty(this=f"{user}@{host}")

    def _parse_withjournaltable(self) -> exp.WithJournalTableProperty:
        """
        解析 `WITH JOURNAL TABLE = <table>`（Teradata）配置。
        说明：严格匹配 `TABLE =` 以避免与其它 WITH 变体产生歧义。
        """
        self._match(TokenType.TABLE)
        self._match(TokenType.EQ)
        return self.expression(exp.WithJournalTableProperty, this=self._parse_table_parts())

    def _parse_log(self, no: bool = False) -> exp.LogProperty:
        """
        解析 `LOG`/`NO LOG` 属性。
        - `no=True` 表示 `NO LOG`；否则为 `LOG`。
        用显式布尔标记承载语义，便于生成端处理。
        """
        return self.expression(exp.LogProperty, no=no)

    def _parse_journal(self, **kwargs) -> exp.JournalProperty:
        """
        解析 `JOURNAL` 相关属性，参数通过 **kwargs 传入以复用多处调用的差异化标记。
        这样可以避免在此处硬编码过多方言分支。
        """
        return self.expression(exp.JournalProperty, **kwargs)

    def _parse_checksum(self) -> exp.ChecksumProperty:
        """
        解析 `CHECKSUM = { ON | OFF | DEFAULT }`（SQL Server 等）。

        关键逻辑：
        - 等号后尝试匹配 `ON` 或 `OFF`，否则保持 `None` 表示未显式设置；
        - `DEFAULT` 独立匹配，表示采用引擎默认的校验策略。
        """
        self._match(TokenType.EQ)

        on = None
        if self._match(TokenType.ON):
            on = True
        elif self._match_text_seq("OFF"):
            on = False

        return self.expression(exp.ChecksumProperty, on=on, default=self._match(TokenType.DEFAULT))

    def _parse_cluster(self, wrapped: bool = False) -> exp.Cluster:
        """
        解析 `CLUSTER BY`/`CLUSTER` 类语法的列序列表。
        - `wrapped=True` 时表示形如 `CLUSTER BY (a, b)` 的括号形式；
        - 否则解析为非括号的逗号分隔序列。
        统一输出为 `exp.Cluster`，其 `expressions` 字段承载顺序列表。
        """
        return self.expression(
            exp.Cluster,
            expressions=(
                self._parse_wrapped_csv(self._parse_ordered)
                if wrapped
                else self._parse_csv(self._parse_ordered)
            ),
        )

    def _parse_clustered_by(self) -> exp.ClusteredByProperty:
        """
        解析 `CLUSTERED BY (cols) [SORTED BY (...)] INTO <n> BUCKETS`（Hive/Trino 等）。

        关键逻辑：
        - 先匹配 `BY` 后的列清单；
        - 可选 `SORTED BY` 指定桶内排序列；
        - 强制匹配 `INTO <number> BUCKETS`，保证语法完整性；
        - 全部信息落入 `ClusteredByProperty` 的字段中，便于生成端序列化。
        """
        self._match_text_seq("BY")

        self._match_l_paren()
        expressions = self._parse_csv(self._parse_column)
        self._match_r_paren()

        if self._match_text_seq("SORTED", "BY"):
            self._match_l_paren()
            sorted_by = self._parse_csv(self._parse_ordered)
            self._match_r_paren()
        else:
            sorted_by = None

        self._match(TokenType.INTO)
        buckets = self._parse_number()
        self._match_text_seq("BUCKETS")

        return self.expression(
            exp.ClusteredByProperty,
            expressions=expressions,
            sorted_by=sorted_by,
            buckets=buckets,
        )

    def _parse_copy_property(self) -> t.Optional[exp.CopyGrantsProperty]:
        """
        解析 `COPY GRANTS` 语义（Snowflake/Redshift 等方言用于复制对象权限）。
        若未成功匹配 `GRANTS`，回退一格索引并返回 `None`，交给上层选择其它属性解析路径。
        """
        if not self._match_text_seq("GRANTS"):
            # 未命中 `GRANTS`，将索引回退到进入该分支前的位置，避免吞掉 token
            self._retreat(self._index - 1)
            return None

        return self.expression(exp.CopyGrantsProperty)

    def _parse_freespace(self) -> exp.FreespaceProperty:
        """
        解析 `FREESPACE = <number> [%]` 属性（常见于 Teradata）。

        关键逻辑：
        - 等号后读取数值；
        - 若紧跟百分号，则设置 `percent=True` 表示百分比单位，否则为绝对值；
        - 统一封装为 `FreespaceProperty`，由生成端决定具体序列化形式。
        """
        self._match(TokenType.EQ)
        # 若出现百分号，表示百分比；否则为具体空间数值
        return self.expression(
            exp.FreespaceProperty, this=self._parse_number(), percent=self._match(TokenType.PERCENT)
        )

    def _parse_mergeblockratio(
        self, no: bool = False, default: bool = False
    ) -> exp.MergeBlockRatioProperty:
        """
        解析 `MERGEBLOCKRATIO = <number> [%]` 或 `NO MERGEBLOCKRATIO`/`DEFAULT MERGEBLOCKRATIO`。

        关键逻辑：
        - 命中等号分支时，解析数值和可选百分号；
        - 否则根据调用方传入的 `no/default` 标志构造语义（避免在此处重复判词）。
        """
        if self._match(TokenType.EQ):
            return self.expression(
                exp.MergeBlockRatioProperty,
                this=self._parse_number(),
                percent=self._match(TokenType.PERCENT),
            )

        return self.expression(exp.MergeBlockRatioProperty, no=no, default=default)

    def _parse_datablocksize(
        self,
        default: t.Optional[bool] = None,
        minimum: t.Optional[bool] = None,
        maximum: t.Optional[bool] = None,
    ) -> exp.DataBlocksizeProperty:
        """
        解析 `DATABLOCKSIZE = <size> [BYTES|KBYTES|KILOBYTES]` 以及可选的
        `DEFAULT | MINIMUM | MAXIMUM` 标志位（通过参数传入）。

        关键逻辑：
        - 等号后读取数值；
        - 单位可选，命中后保留原始文本（`_prev.text`）以保证方言原样生成；
        - `default/minimum/maximum` 由上层在进入该函数前根据关键字设置布尔位。
        """
        self._match(TokenType.EQ)
        size = self._parse_number()

        units = None
        if self._match_texts(("BYTES", "KBYTES", "KILOBYTES")):
            units = self._prev.text

        return self.expression(
            exp.DataBlocksizeProperty,
            size=size,
            units=units,
            default=default,
            minimum=minimum,
            maximum=maximum,
        )

    def _parse_blockcompression(self) -> exp.BlockCompressionProperty:
        """
        解析块压缩策略：`BLOCKCOMPRESSION = { ALWAYS | MANUAL | NEVER | DEFAULT } [AUTOTEMP <schema>]`。

        关键逻辑：
        - 等号后尝试匹配四种策略之一，分别用布尔位承载；
        - 可选 `AUTOTEMP` 后跟目标 schema，使用 `_parse_schema()` 解析成结构化节点；
        - 统一封装为 `BlockCompressionProperty`，生成端根据布尔位选择具体输出。
        """
        self._match(TokenType.EQ)
        always = self._match_text_seq("ALWAYS")
        manual = self._match_text_seq("MANUAL")
        never = self._match_text_seq("NEVER")
        default = self._match_text_seq("DEFAULT")

        autotemp = None
        if self._match_text_seq("AUTOTEMP"):
            autotemp = self._parse_schema()

        return self.expression(
            exp.BlockCompressionProperty,
            always=always,
            manual=manual,
            never=never,
            default=default,
            autotemp=autotemp,
        )

    def _parse_withisolatedloading(self) -> t.Optional[exp.IsolatedLoadingProperty]:
        """
        解析 Teradata 的 `WITH [NO] [CONCURRENT] ISOLATED LOADING FOR ...` 属性。

        关键逻辑：
        - 记录进入分支时的索引 `index`，若后续未成功匹配完整关键字串，则回退（避免吞掉前瞻 token）。
        - 可选前缀 `NO` 与 `CONCURRENT` 分别以布尔标记记录，便于生成端序列化。
        - 必须命中连续关键字 `ISOLATED LOADING` 才视为有效属性，否则回退并返回 None。
        - 目标 `FOR` 子句通过 `ISOLATED_LOADING_OPTIONS` 控制的枚举进行解析，未命中可为空。
        """
        index = self._index
        no = self._match_text_seq("NO")
        concurrent = self._match_text_seq("CONCURRENT")

        if not self._match_text_seq("ISOLATED", "LOADING"):
            # 未完整命中关键字串，回退到进入前的位置，交由其它属性分支解析
            self._retreat(index)
            return None

        target = self._parse_var_from_options(self.ISOLATED_LOADING_OPTIONS, raise_unmatched=False)
        return self.expression(
            exp.IsolatedLoadingProperty, no=no, concurrent=concurrent, target=target
        )

    def _parse_locking(self) -> exp.LockingProperty:
        """
        解析锁语义（Teradata 等）：`LOCKING [DATABASE|TABLE|VIEW|ROW] <obj> FOR/IN <mode> [OVERRIDE]`。

        关键逻辑：
        - 锁作用对象 `kind` 优先匹配固定关键字；DATABASE/TABLE/VIEW 需要跟具体对象名；ROW 级别不需要对象；
        - 介词 `FOR|IN` 不是所有方言都一致，保留原词以供生成；
        - 锁类型支持 `ACCESS/EXCLUSIVE/SHARE/READ/WRITE/CHECKSUM` 等，未命中则保持 None；
        - `OVERRIDE` 为附加修饰符（如允许覆盖默认锁策略）。
        """
        if self._match(TokenType.TABLE):
            kind = "TABLE"
        elif self._match(TokenType.VIEW):
            kind = "VIEW"
        elif self._match(TokenType.ROW):
            kind = "ROW"
        elif self._match_text_seq("DATABASE"):
            kind = "DATABASE"
        else:
            kind = None

        if kind in ("DATABASE", "TABLE", "VIEW"):
            this = self._parse_table_parts()
        else:
            this = None

        if self._match(TokenType.FOR):
            for_or_in = "FOR"
        elif self._match(TokenType.IN):
            for_or_in = "IN"
        else:
            for_or_in = None

        if self._match_text_seq("ACCESS"):
            lock_type = "ACCESS"
        elif self._match_texts(("EXCL", "EXCLUSIVE")):
            lock_type = "EXCLUSIVE"
        elif self._match_text_seq("SHARE"):
            lock_type = "SHARE"
        elif self._match_text_seq("READ"):
            lock_type = "READ"
        elif self._match_text_seq("WRITE"):
            lock_type = "WRITE"
        elif self._match_text_seq("CHECKSUM"):
            lock_type = "CHECKSUM"
        else:
            lock_type = None

        override = self._match_text_seq("OVERRIDE")

        return self.expression(
            exp.LockingProperty,
            this=this,
            kind=kind,
            for_or_in=for_or_in,
            lock_type=lock_type,
            override=override,
        )

    def _parse_partition_by(self) -> t.List[exp.Expression]:
        """
        解析 `PARTITION BY` 列表，返回表达式序列；若未出现则返回空列表。
        说明：使用 `_parse_assignment` 以支持复杂表达式（如 `col ASC NULLS LAST`）。
        """
        if self._match(TokenType.PARTITION_BY):
            return self._parse_csv(self._parse_assignment)
        return []

    def _parse_partition_bound_spec(self) -> exp.PartitionBoundSpec:
        """
        解析分区边界说明（PostgreSQL RANGE/LIST 分区语义）：
        - `IN (<exprs>)`：LIST 分区值集合；
        - `FROM (...) TO (...)`：RANGE 分区闭开区间；
        - `WITH (MODULUS m, REMAINDER r)`：哈希分区指定模与余数；
        - 特殊字面量 `MINVALUE | MAXVALUE` 被规范为 `exp.Var` 便于生成。

        关键逻辑：不同形式互斥，按分支匹配，命中则采集对应字段，其它置空。
        未命中任一分支视为语法错误。
        """
        def _parse_partition_bound_expr() -> t.Optional[exp.Expression]:
            if self._match_text_seq("MINVALUE"):
                return exp.var("MINVALUE")
            if self._match_text_seq("MAXVALUE"):
                return exp.var("MAXVALUE")
            return self._parse_bitwise()

        this: t.Optional[exp.Expression | t.List[exp.Expression]] = None
        expression = None
        from_expressions = None
        to_expressions = None

        if self._match(TokenType.IN):
            this = self._parse_wrapped_csv(self._parse_bitwise)
        elif self._match(TokenType.FROM):
            from_expressions = self._parse_wrapped_csv(_parse_partition_bound_expr)
            self._match_text_seq("TO")
            to_expressions = self._parse_wrapped_csv(_parse_partition_bound_expr)
        elif self._match_text_seq("WITH", "(", "MODULUS"):
            this = self._parse_number()
            self._match_text_seq(",", "REMAINDER")
            expression = self._parse_number()
            self._match_r_paren()
        else:
            self.raise_error("Failed to parse partition bound spec.")

        return self.expression(
            exp.PartitionBoundSpec,
            this=this,
            expression=expression,
            from_expressions=from_expressions,
            to_expressions=to_expressions,
        )

    # https://www.postgresql.org/docs/current/sql-createtable.html
    def _parse_partitioned_of(self) -> t.Optional[exp.PartitionedOfProperty]:
        """
        解析 PostgreSQL `PARTITION OF <parent> { DEFAULT | FOR VALUES ... }` 语法。

        关键逻辑：
        - 未命中 `OF` 时直接回退一格并返回 None，交由其它分支解析；
        - 命中后必须跟父表名；
        - 子句必须二选一：`DEFAULT` 或 `FOR VALUES ...`（否则报错）。
        """
        if not self._match_text_seq("OF"):
            self._retreat(self._index - 1)
            return None

        this = self._parse_table(schema=True)

        if self._match(TokenType.DEFAULT):
            expression: exp.Var | exp.PartitionBoundSpec = exp.var("DEFAULT")
        elif self._match_text_seq("FOR", "VALUES"):
            expression = self._parse_partition_bound_spec()
        else:
            self.raise_error("Expecting either DEFAULT or FOR VALUES clause.")

        return self.expression(exp.PartitionedOfProperty, this=this, expression=expression)

    def _parse_partitioned_by(self) -> exp.PartitionedByProperty:
        """
        解析方言中的 `PARTITIONED_BY = (...)` 或等价语法。

        关键逻辑：
        - 等号后优先解析成 `schema`（如复杂对象/函数型表达式），否则解析成括号包裹字段列表；
        - 用 `PartitionedByProperty.this` 承载原始结构，由生成端做具体序列化。
        """
        self._match(TokenType.EQ)
        return self.expression(
            exp.PartitionedByProperty,
            this=self._parse_schema() or self._parse_bracket(self._parse_field()),
        )
    def _parse_withdata(self, no: bool = False) -> exp.WithDataProperty:
        # 解析 WITH DATA / NO DATA 的伴随选项：AND [NO] STATISTICS
        # 注：不同方言在创建/物化对象（如表、MV）时允许指定是否生成统计信息；
        # 这里将有无 STATISTICS 映射为三态：True(AND STATISTICS)、False(AND NO STATISTICS)、None(未出现)
        if self._match_text_seq("AND", "STATISTICS"):
            statistics = True
        elif self._match_text_seq("AND", "NO", "STATISTICS"):
            statistics = False
        else:
            statistics = None

        # 返回 WithDataProperty，no 表示上层是否是 NO DATA（由调用者传入），statistics 表示统计信息策略
        return self.expression(exp.WithDataProperty, no=no, statistics=statistics)

    def _parse_contains_property(self) -> t.Optional[exp.SqlReadWriteProperty]:
        # 解析 CONTAINS SQL（SQL:2003 存储过程/函数语义，表明可能包含 SQL 但不读写数据）
        if self._match_text_seq("SQL"):
            return self.expression(exp.SqlReadWriteProperty, this="CONTAINS SQL")
        return None

    def _parse_modifies_property(self) -> t.Optional[exp.SqlReadWriteProperty]:
        # 解析 MODIFIES SQL DATA（SQL:2003，表明可能修改 SQL 数据）
        if self._match_text_seq("SQL", "DATA"):
            return self.expression(exp.SqlReadWriteProperty, this="MODIFIES SQL DATA")
        return None

    def _parse_no_property(self) -> t.Optional[exp.Expression]:
        # 解析 NO 开头的属性：
        # - PRIMARY INDEX：部分方言（如 Teradata）用于声明无主索引
        # - SQL：NO SQL（SQL:2003，声明不包含 SQL 语句）
        if self._match_text_seq("PRIMARY", "INDEX"):
            return exp.NoPrimaryIndexProperty()
        if self._match_text_seq("SQL"):
            return self.expression(exp.SqlReadWriteProperty, this="NO SQL")
        return None

    def _parse_on_property(self) -> t.Optional[exp.Expression]:
        # 解析 ON 提交策略（临时表/全局临时表常见）：
        # - COMMIT PRESERVE ROWS：提交时保留数据
        # - COMMIT DELETE ROWS：提交时删除数据
        if self._match_text_seq("COMMIT", "PRESERVE", "ROWS"):
            return exp.OnCommitProperty()
        if self._match_text_seq("COMMIT", "DELETE", "ROWS"):
            return exp.OnCommitProperty(delete=True)
        # 其他 ON xxx 情况：将 xxx 作为 schema/id 解析为 OnProperty 的目标
        return self.expression(exp.OnProperty, this=self._parse_schema(self._parse_id_var()))

    def _parse_reads_property(self) -> t.Optional[exp.SqlReadWriteProperty]:
        # 解析 READS SQL DATA（SQL:2003，声明只读 SQL 数据）
        if self._match_text_seq("SQL", "DATA"):
            return self.expression(exp.SqlReadWriteProperty, this="READS SQL DATA")
        return None

    def _parse_distkey(self) -> exp.DistKeyProperty:
        # 解析分布键（如 Redshift/某些 MPP 方言），通常为单个标识符；
        # 使用 _parse_wrapped 以支持括号包裹的写法，提高方言兼容性
        return self.expression(exp.DistKeyProperty, this=self._parse_wrapped(self._parse_id_var))

    def _parse_create_like(self) -> t.Optional[exp.LikeProperty]:
        # 解析 CREATE ... LIKE 语法：从现有表复制定义，并可带 INCLUDING/EXCLUDING 选项
        table = self._parse_table(schema=True)

        options = []
        # 收集 INCLUDING / EXCLUDING 属性列表；注意这里使用 _match_texts 支持二选一匹配
        while self._match_texts(("INCLUDING", "EXCLUDING")):
            this = self._prev.text.upper()  # 规范化关键字大小写，避免大小写差异影响后续生成

            id_var = self._parse_id_var()
            if not id_var:
                # 如果缺少属性名（例如 INCLUDING/EXCLUDING 后未跟标识符），则认为不是 LIKE 选项块
                return None

            options.append(
                # 将属性名提升为统一大写存储，减少不同方言/大小写带来的分支处理复杂度
                self.expression(exp.Property, this=this, value=exp.var(id_var.this.upper()))
            )

        # 返回 LikeProperty，this=被复制的表，expressions=选项集合
        return self.expression(exp.LikeProperty, this=table, expressions=options)

    def _parse_sortkey(self, compound: bool = False) -> exp.SortKeyProperty:
        # 解析排序键（常见于列式/MPP 数据库），支持括号包裹及多个列；
        # compound 标志用于区分单一/复合排序键在特定方言中的语义
        return self.expression(
            exp.SortKeyProperty, this=self._parse_wrapped_id_vars(), compound=compound
        )

    def _parse_character_set(self, default: bool = False) -> exp.CharacterSetProperty:
        # 解析 CHARACTER SET / DEFAULT CHARACTER SET:
        # - 可选等号（=）以兼容部分方言写法：CHARACTER SET = 'utf8'
        # - 支持标识符或字符串字面量作为字符集名称
        self._match(TokenType.EQ)
        return self.expression(
            exp.CharacterSetProperty, this=self._parse_var_or_string(), default=default
        )

    def _parse_remote_with_connection(self) -> exp.RemoteWithConnectionModelProperty:
        # 解析 WITH CONNECTION 子句：声明随后提供连接信息片段
        # 原因：部分方言/对象（如外部表/远端模型）允许通过 WITH CONNECTION 绑定连接资源
        self._match_text_seq("WITH", "CONNECTION")
        return self.expression(
            exp.RemoteWithConnectionModelProperty, this=self._parse_table_parts()
        )

    def _parse_returns(self) -> exp.ReturnsProperty:
        # 解析 RETURNS 子句：支持三种形式
        # 1) RETURNS TABLE 或 RETURNS TABLE<struct>（表形返回）
        # 2) RETURNS NULL ON NULL INPUT（入参含 NULL 时返回 NULL）
        # 3) RETURNS <type>（普通类型）
        value: t.Optional[exp.Expression]
        null = None
        is_table = self._match(TokenType.TABLE)  # 出现 TABLE 则进入表返回模式

        if is_table:
            if self._match(TokenType.LT):  # 形如 TABLE<col1 type, col2 type, ...>
                value = self.expression(
                    exp.Schema,
                    this="TABLE",
                    expressions=self._parse_csv(self._parse_struct_types),
                )
                if not self._match(TokenType.GT):
                    # 必须闭合 >，否则语法不完整
                    self.raise_error("Expecting >")
            else:
                # 不带 <> 的 TABLE：以 Schema + Var("TABLE") 标记表返回
                value = self._parse_schema(exp.var("TABLE"))
        elif self._match_text_seq("NULL", "ON", "NULL", "INPUT"):
            # 标准语义：入参包含 NULL 时返回 NULL；用 null=True+value=None 表达
            null = True
            value = None
        else:
            # 默认：解析为普通类型（标量/复合），由类型解析器处理
            value = self._parse_types()

        return self.expression(exp.ReturnsProperty, this=value, is_table=is_table, null=null)

    def _parse_describe(self) -> exp.Describe:
        # 解析 DESCRIBE/EXPLAIN 等描述类语句
        kind = self._match_set(self.CREATABLES) and self._prev.text  # 若描述对象是可创建对象（TABLE/VIEW 等），记录类型
        style = self._match_texts(self.DESCRIBE_STYLES) and self._prev.text.upper()  # 风格关键字统一大写，便于生成
        if self._match(TokenType.DOT):
            # 若紧跟点号，说明之前识别的 style 实为名称一部分（如 db.table），需回退以正确解析标识
            style = None
            self._retreat(self._index - 2)

        # 可选 FORMAT 属性；advance=False 仅窥探不前进，由 _parse_property 消耗
        format = self._parse_property() if self._match(TokenType.FORMAT, advance=False) else None

        # 若后续是语句起始（如 SELECT），解析为语句；否则解析为表
        if self._match_set(self.STATEMENT_PARSERS, advance=False):
            this = self._parse_statement()
        else:
            this = self._parse_table(schema=True)

        properties = self._parse_properties()
        expressions = properties.expressions if properties else None
        partition = self._parse_partition()  # 某些方言支持 DESCRIBE PARTITION 语义
        return self.expression(
            exp.Describe,
            this=this,
            style=style,
            kind=kind,
            expressions=expressions,
            partition=partition,
            format=format,
        )

    def _parse_multitable_inserts(self, comments: t.Optional[t.List[str]]) -> exp.MultitableInserts:
        # 解析多表插入（如 Oracle 的 INSERT ALL / INSERT FIRST）
        kind = self._prev.text.upper()  # 记录 ALL / FIRST，生成时据此决定行为
        expressions = []

        def parse_conditional_insert() -> t.Optional[exp.ConditionalInsert]:
            # 条件块：WHEN <cond> THEN INTO ... [ELSE]
            if self._match(TokenType.WHEN):
                expression = self._parse_disjunction()  # 使用析取优先级，符合多数方言 WHEN 语义
                self._match(TokenType.THEN)
            else:
                expression = None  # 无 WHEN 表示无条件分支或结束

            else_ = self._match(TokenType.ELSE)  # 标记是否 ELSE 分支（匹配剩余）

            if not self._match(TokenType.INTO):
                # 未出现 INTO，说明条件块结束
                return None

            return self.expression(
                exp.ConditionalInsert,
                this=self.expression(
                    exp.Insert,
                    this=self._parse_table(schema=True),  # 目标表
                    expression=self._parse_derived_table_values(),  # 来源（VALUES/SELECT）
                ),
                expression=expression,
                else_=else_,
            )

        # 收集多个条件插入项，直到无法再解析出新的小节
        expression = parse_conditional_insert()
        while expression is not None:
            expressions.append(expression)
            expression = parse_conditional_insert()

        return self.expression(
            exp.MultitableInserts,
            kind=kind,
            comments=comments,
            expressions=expressions,
            source=self._parse_table(),  # 多表插入的源（方言可选）
        )

    def _parse_insert(self) -> t.Union[exp.Insert, exp.MultitableInserts]:
        # 解析 INSERT 语句，兼容多种方言扩展（如 DIRECTORY、OR REPLACE、BY NAME、SETTINGS 等）
        comments = []
        hint = self._parse_hint()  # 方言 Hint（例如 /*+ APPEND */）
        overwrite = self._match(TokenType.OVERWRITE)  # INSERT OVERWRITE（Hive/Trino 等）
        ignore = self._match(TokenType.IGNORE)  # INSERT IGNORE（MySQL）
        local = self._match_text_seq("LOCAL")  # LOCAL DIRECTORY（Hive 等）
        alternative = None  # 记录 OR REPLACE / OR ABORT 等替代关键字
        is_function = None  # 标记是否 INSERT INTO FUNCTION（部分方言）

        if self._match_text_seq("DIRECTORY"):
            # INSERT DIRECTORY：将结果输出到目录而非表
            this: t.Optional[exp.Expression] = self.expression(
                exp.Directory,
                this=self._parse_var_or_string(),  # 目录路径既可为字符串也可为标识符
                local=local,
                row_format=self._parse_row_format(match_row=True),  # 可选行/文件格式
            )
        else:
            # 多表插入：INSERT {FIRST|ALL}
            if self._match_set((TokenType.FIRST, TokenType.ALL)):
                comments += ensure_list(self._prev_comments)  # 保留紧邻注释
                return self._parse_multitable_inserts(comments)

            # OR 分支（SQLite/Oracle 风格）：OR REPLACE / OR ABORT / OR IGNORE 等
            if self._match(TokenType.OR):
                alternative = self._match_texts(self.INSERT_ALTERNATIVES) and self._prev.text

            # INTO/TABLE 可选，不同方言对必选性不同
            self._match(TokenType.INTO)
            comments += ensure_list(self._prev_comments)
            self._match(TokenType.TABLE)
            is_function = self._match(TokenType.FUNCTION)

            # 目标可能是表或函数（如 ClickHouse 支持 INTO FUNCTION）
            this = (
                self._parse_table(schema=True, parse_partition=True)
                if not is_function
                else self._parse_function()
            )
            if isinstance(this, exp.Table) and self._match(TokenType.ALIAS, advance=False):
                # 允许在目标上使用别名（少见，但保留解析能力）
                this.set("alias", self._parse_table_alias())

        returning = self._parse_returning()  # 预读 RETURNING，以兼容不同出现位置

        return self.expression(
            exp.Insert,
            comments=comments,
            hint=hint,
            is_function=is_function,
            this=this,
            stored=self._match_text_seq("STORED") and self._parse_stored(),  # Delta/部分方言：STORED AS ...
            by_name=self._match_text_seq("BY", "NAME"),  # ClickHouse：按列名匹配插入
            exists=self._parse_exists(),  # INSERT ... EXISTS(...) 扩展
            where=self._match_pair(TokenType.REPLACE, TokenType.WHERE) and self._parse_assignment(),  # ClickHouse：REPLACE WHERE
            partition=self._match(TokenType.PARTITION_BY) and self._parse_partitioned_by(),  # 方言：PARTITION BY
            settings=self._match_text_seq("SETTINGS") and self._parse_settings_property(),  # ClickHouse：SETTINGS
            default=self._match_text_seq("DEFAULT", "VALUES"),
            expression=self._parse_derived_table_values() or self._parse_ddl_select(),  # 数据来源：VALUES/SELECT 或 DDL SELECT
            conflict=self._parse_on_conflict(),  # 冲突处理：ON CONFLICT
            returning=returning or self._parse_returning(),  # RETURNING 可能在后面再次出现
            overwrite=overwrite,
            alternative=alternative,
            ignore=ignore,
            source=self._match(TokenType.TABLE) and self._parse_table(),  # 扩展：INSERT SOURCE TABLE
        )

    def _parse_kill(self) -> exp.Kill:
        # 解析 KILL 语句：终止连接或查询
        # 原因：不同方言可指定 KILL CONNECTION / KILL QUERY，这里将其映射到 kind 以便生成端区分
        kind = exp.var(self._prev.text) if self._match_texts(("CONNECTION", "QUERY")) else None  # 指定要终止的对象类型

        return self.expression(
            exp.Kill,
            this=self._parse_primary(),  # 目标标识（如连接 id 或查询 id），用主表达式解析保证数值/变量均可
            kind=kind,
        )

    def _parse_on_conflict(self) -> t.Optional[exp.OnConflict]:
        # 解析冲突处理语义：ON CONFLICT（Postgres 等）或 ON DUPLICATE KEY（MySQL）
        # 设计原因：两类语法等价于“冲突时采取某种动作”，统一抽象为 OnConflict，并记录是 duplicate 还是 conflict
        conflict = self._match_text_seq("ON", "CONFLICT")
        duplicate = self._match_text_seq("ON", "DUPLICATE", "KEY")

        if not conflict and not duplicate:
            # 二者都未出现，说明当前并非冲突处理子句
            return None

        conflict_keys = None
        constraint = None

        if conflict:
            # Postgres 风格：ON CONFLICT 可跟 ON CONSTRAINT <name> 或者 (col_list)
            if self._match_text_seq("ON", "CONSTRAINT"):
                constraint = self._parse_id_var()  # 具名唯一/排他约束
            elif self._match(TokenType.L_PAREN):
                conflict_keys = self._parse_csv(self._parse_id_var)  # 冲突键列表
                self._match_r_paren()

        # 动作解析：DO NOTHING / DO UPDATE SET ... 等
        action = self._parse_var_from_options(self.CONFLICT_ACTIONS)
        if self._prev.token_type == TokenType.UPDATE:
            # DO UPDATE SET a=b, ...
            self._match(TokenType.SET)
            expressions = self._parse_csv(self._parse_equality)  # 统一解析成等式列表
        else:
            expressions = None

        return self.expression(
            exp.OnConflict,
            duplicate=duplicate,  # MySQL 语义：ON DUPLICATE KEY
            expressions=expressions,
            action=action,
            conflict_keys=conflict_keys,
            constraint=constraint,
            where=self._parse_where(),  # Postgres 允许 DO UPDATE ... WHERE 条件
        )

    def _parse_returning(self) -> t.Optional[exp.Returning]:
        # 解析 RETURNING 子句：部分方言允许在 INSERT/UPDATE/DELETE 后返回受影响数据
        if not self._match(TokenType.RETURNING):
            # 未出现 RETURNING 即不解析该子句
            return None
        return self.expression(
            exp.Returning,
            expressions=self._parse_csv(self._parse_expression),  # 返回的表达式列表
            into=self._match(TokenType.INTO) and self._parse_table_part(),  # 方言扩展：RETURNING INTO <var/table-part>
        )

    def _parse_row(self) -> t.Optional[exp.RowFormatSerdeProperty | exp.RowFormatDelimitedProperty]:
        # 解析 ROW FORMAT 子句（Hive/Trino 等）：用于控制序列化/分隔
        if not self._match(TokenType.FORMAT):
            # 未出现 FORMAT 则不认为是 ROW FORMAT 子句
            return None
        return self._parse_row_format()

    def _parse_serde_properties(self, with_: bool = False) -> t.Optional[exp.SerdeProperties]:
        # 解析 SERDEPROPERTIES 子句（Hive 等），可选 WITH 关键字
        index = self._index
        with_ = with_ or self._match_text_seq("WITH")  # 记录是否显式出现 WITH

        if not self._match(TokenType.SERDE_PROPERTIES):
            # 未匹配到 SERDEPROPERTIES，回退到进入该方法前的位置
            self._retreat(index)
            return None
        return self.expression(
            exp.SerdeProperties,
            **{  # type: ignore
                "expressions": self._parse_wrapped_properties(),  # 解析括号包裹的 K=V 属性列表
                "with": with_,
            },
        )

    def _parse_row_format(
        self, match_row: bool = False
    ) -> t.Optional[exp.RowFormatSerdeProperty | exp.RowFormatDelimitedProperty]:
        # 解析各类分隔/转义配置；
        # 原因：Hive/Trino 等的 ROW FORMAT DELIMITED 支持精细控制序列化分隔符与空值表示
        if match_row and not self._match_pair(TokenType.ROW, TokenType.FORMAT):
            return None

        if self._match_text_seq("SERDE"):
            this = self._parse_string()

            serde_properties = self._parse_serde_properties()

            return self.expression(
                exp.RowFormatSerdeProperty, this=this, serde_properties=serde_properties
            )

        self._match_text_seq("DELIMITED")

        kwargs = {}


        if self._match_text_seq("FIELDS", "TERMINATED", "BY"):
            # 字段分隔符，例如 '\t' 或 ','
            kwargs["fields"] = self._parse_string()
            if self._match_text_seq("ESCAPED", "BY"):
                # 字段转义字符，例如 '\\'；用于对分隔符等特殊字符进行转义
                kwargs["escaped"] = self._parse_string()
        if self._match_text_seq("COLLECTION", "ITEMS", "TERMINATED", "BY"):
            # 集合（数组）元素分隔符
            kwargs["collection_items"] = self._parse_string()
        if self._match_text_seq("MAP", "KEYS", "TERMINATED", "BY"):
            # Map 的 key 与 value 之间的分隔符
            kwargs["map_keys"] = self._parse_string()
        if self._match_text_seq("LINES", "TERMINATED", "BY"):
            # 行分隔符，常见为 '\n' 或 '\r\n'
            kwargs["lines"] = self._parse_string()
        if self._match_text_seq("NULL", "DEFINED", "AS"):
            # 空值的文本表示，例如 '\\N' 或 'NULL'
            kwargs["null"] = self._parse_string()

        return self.expression(exp.RowFormatDelimitedProperty, **kwargs)  # type: ignore

    def _parse_load(self) -> exp.LoadData | exp.Command:
        # 解析 LOAD DATA（Hive/Trino 等）：将外部数据加载到表
        if self._match_text_seq("DATA"):
            local = self._match_text_seq("LOCAL")  # LOCAL 表示从本地文件系统加载，否则从分布式存储
            self._match_text_seq("INPATH")
            inpath = self._parse_string()  # 数据路径
            overwrite = self._match(TokenType.OVERWRITE)  # 是否覆写（OVERWRITE）
            self._match_pair(TokenType.INTO, TokenType.TABLE)

            return self.expression(
                exp.LoadData,
                this=self._parse_table(schema=True),  # 目标表
                local=local,
                overwrite=overwrite,
                inpath=inpath,
                partition=self._parse_partition(),  # 可选：指定分区
                input_format=self._match_text_seq("INPUTFORMAT") and self._parse_string(),  # 可选输入格式类
                serde=self._match_text_seq("SERDE") and self._parse_string(),  # 可选 SerDe 类
            )
        # 若不是 LOAD DATA 语义，则按通用命令处理
        return self._parse_as_command(self._prev)

    def _parse_delete(self) -> exp.Delete:
        # 解析 DELETE 语句，兼容 MySQL 的多表删除语法
        # 参考：https://dev.mysql.com/doc/refman/8.0/en/delete.html
        tables = None
        if not self._match(TokenType.FROM, advance=False):
            # MySQL 多表语法：DELETE t1, t2 FROM t1 JOIN t2 ...
            # 当第一个 token 不是 FROM 时，前面可能是要删除的表列表
            tables = self._parse_csv(self._parse_table) or None

        returning = self._parse_returning()  # 部分方言支持 DELETE ... RETURNING

        return self.expression(
            exp.Delete,
            tables=tables,
            this=self._match(TokenType.FROM) and self._parse_table(joins=True),  # FROM 后的主表及联接
            using=self._match(TokenType.USING)
            and self._parse_csv(lambda: self._parse_table(joins=True)),  # MySQL：USING 子句
            cluster=self._match(TokenType.ON) and self._parse_on_property(),  # 方言扩展：ON CLUSTER/ON COMMIT 等
            where=self._parse_where(),
            returning=returning or self._parse_returning(),
            limit=self._parse_limit(),
        )

    def _parse_update(self) -> exp.Update:
        # 解析 UPDATE 语句，支持 FROM/RETURNING/ORDER/LIMIT 等扩展
        kwargs: t.Dict[str, t.Any] = {
            "this": self._parse_table(joins=True, alias_tokens=self.UPDATE_ALIAS_TOKENS),
        }
        while self._curr:
            if self._match(TokenType.SET):
                kwargs["expressions"] = self._parse_csv(self._parse_equality)
            elif self._match(TokenType.RETURNING, advance=False):
                kwargs["returning"] = self._parse_returning()
            elif self._match(TokenType.FROM, advance=False):
                kwargs["from"] = self._parse_from(joins=True)
            elif self._match(TokenType.WHERE, advance=False):
                kwargs["where"] = self._parse_where()
            elif self._match(TokenType.ORDER_BY, advance=False):
                kwargs["order"] = self._parse_order()
            elif self._match(TokenType.LIMIT, advance=False):
                kwargs["limit"] = self._parse_limit()
            else:
                break

        return self.expression(exp.Update, **kwargs)

    def _parse_use(self) -> exp.Use:
        # 解析 USE 语句：切换数据库/模式/目录等
        return self.expression(
            exp.Use,
            kind=self._parse_var_from_options(self.USABLES, raise_unmatched=False),  # 解析 USE 的对象类型（DATABASE/SCHEMA/CATALOG 等）
            this=self._parse_table(schema=False),  # 目标名称，允许多段标识
        )

    def _parse_uncache(self) -> exp.Uncache:
        # 解析 UNCACHE 语句：取消缓存的表
        if not self._match(TokenType.TABLE):
            # UNCACHE 必须跟 TABLE；若缺失立即报错，便于快速定位语法问题
            self.raise_error("Expecting TABLE after UNCACHE")

        return self.expression(
            exp.Uncache,
            exists=self._parse_exists(),  # 可选：IF EXISTS 之类的存在性判断
            this=self._parse_table(schema=True),  # 目标表，允许带 schema 前缀
        )

    def _parse_cache(self) -> exp.Cache:
        # 解析 CACHE TABLE 语句：将查询结果缓存为表
        lazy = self._match_text_seq("LAZY")  # LAZY 表示延迟缓存（由执行引擎决定具体策略）
        self._match(TokenType.TABLE)
        table = self._parse_table(schema=True)  # 目标缓存表

        options = []
        if self._match_text_seq("OPTIONS"):
            # OPTIONS(k = v) 形式；此处解析单个键值对，保持最小实现
            self._match_l_paren()
            k = self._parse_string()
            self._match(TokenType.EQ)
            v = self._parse_string()
            options = [k, v]
            self._match_r_paren()

        self._match(TokenType.ALIAS)  # 要求提供别名定义（即后续 SELECT 的输出别名）
        return self.expression(
            exp.Cache,
            this=table,
            lazy=lazy,
            options=options,
            expression=self._parse_select(nested=True),  # 缓存来源为嵌套 SELECT
        )

    def _parse_partition(self) -> t.Optional[exp.Partition]:
        # 解析 PARTITION / SUBPARTITION 子句
        if not self._match_texts(self.PARTITION_KEYWORDS):
            return None  # 未出现分区关键字则不进入该分支

        return self.expression(
            exp.Partition,
            subpartition=self._prev.text.upper() == "SUBPARTITION",  # 区分是否为二级分区
            expressions=self._parse_wrapped_csv(self._parse_assignment),  # 括号包裹的分区表达式列表
        )

    def _parse_value(self, values: bool = True) -> t.Optional[exp.Tuple]:
        # 解析 VALUES 子句中的值元组；支持 DEFAULT 与括号/非括号形式
        def _parse_value_expression() -> t.Optional[exp.Expression]:
            if self.dialect.SUPPORTS_VALUES_DEFAULT and self._match(TokenType.DEFAULT):
                # 支持 DEFAULT 关键字作为值（由方言能力控制）
                return exp.var(self._prev.text.upper())
            return self._parse_expression()

        if self._match(TokenType.L_PAREN):
            # 括号包裹的多列值：VALUES (a, b, ...)
            expressions = self._parse_csv(_parse_value_expression)
            self._match_r_paren()
            return self.expression(exp.Tuple, expressions=expressions)

        # 某些方言允许 VALUES 1, 2（单列多行）；此时每个表达式构成一个单元素 Tuple
        expression = self._parse_expression()
        if expression:
            return self.expression(exp.Tuple, expressions=[expression])
        return None

    def _parse_projections(self) -> t.List[exp.Expression]:
        # 解析 SELECT 投影列表；代理到通用表达式序列解析
        return self._parse_expressions()

    def _parse_wrapped_select(self, table: bool = False) -> t.Optional[exp.Expression]:
        # 解析括号包裹的 SELECT/表/值等片段，并在必要时补齐 FROM 或应用管道/集合操作
        if self._match_set((TokenType.PIVOT, TokenType.UNPIVOT)):
            # 简化 PIVOT/UNPIVOT 语法：根据当前 token 判断是否为 UNPIVOT
            this: t.Optional[exp.Expression] = self._parse_simplified_pivot(
                is_unpivot=self._prev.token_type == TokenType.UNPIVOT
            )
        elif self._match(TokenType.FROM):
            # duckdb 的 FROM-first 语法：允许形如 (FROM t SELECT ...)
            from_ = self._parse_from(skip_from_token=True, consume_pipe=True)
            # Support parentheses for duckdb FROM-first syntax
            select = self._parse_select(from_=from_)
            if select:
                if not select.args.get("from"):
                    select.set("from", from_)
                this = select
            else:
                # 否则退化为 SELECT * FROM <from_>
                this = exp.select("*").from_(t.cast(exp.From, from_))
                this = self._parse_query_modifiers(self._parse_set_operations(this))
        else:
            # 不是 FROM-first：根据是否处于 table 上下文，选择解析表或嵌套 SELECT
            this = (
                self._parse_table(consume_pipe=True)
                if table
                else self._parse_select(nested=True, parse_set_operation=False)
            )

            # 若 table=True 且解析结果为 Values（同时带别名），转为 Table 以便后续 JOIN 等修饰器处理
            # 原因：某些方言允许 (VALUES ...) AS 别名 后直接跟 JOIN，需要以表语义参与
            if table and isinstance(this, exp.Values) and this.alias:
                alias = this.args["alias"].pop()
                this = exp.Table(this=this, alias=alias)

            # 对当前片段先应用集合操作（UNION 等），再应用查询修饰器（WHERE/ORDER/JOIN 等）
            this = self._parse_query_modifiers(self._parse_set_operations(this))

        return this

    def _parse_select(
        self,
        nested: bool = False,
        table: bool = False,
        parse_subquery_alias: bool = True,
        parse_set_operation: bool = True,
        consume_pipe: bool = True,
        from_: t.Optional[exp.From] = None,
    ) -> t.Optional[exp.Expression]:
        # 解析 SELECT 入口：根据上下文（nested/table）与标志位控制别名解析、集合操作、管道语法等
        query = self._parse_select_query(
            nested=nested,
            table=table,
            parse_subquery_alias=parse_subquery_alias,
            parse_set_operation=parse_set_operation,
        )

        # ClickHouse/duckdb 等的管道语法：|> 后可接进一步操作
        if consume_pipe and self._match(TokenType.PIPE_GT, advance=False):
            if not query and from_:
                query = exp.select("*").from_(from_)
            if isinstance(query, exp.Query):
                query = self._parse_pipe_syntax_query(query)
                query = query.subquery(copy=False) if query and table else query

        return query

    def _parse_select_query(
        self,
        nested: bool = False,
        table: bool = False,
        parse_subquery_alias: bool = True,
        parse_set_operation: bool = True,
    ) -> t.Optional[exp.Expression]:
        # 解析 SELECT 的核心查询结构：处理 CTE、FROM-first、SELECT 主体、括号/VALUES 等分支
        cte = self._parse_with()

        if cte:
            # 若存在 WITH，则期望跟随一个可接收 WITH 的语句
            this = self._parse_statement()

            if not this:
                self.raise_error("Failed to parse any statement following CTE")
                return cte

            if "with" in this.arg_types:
                this.set("with", cte)  # 挂载 CTE 到后续语句
            else:
                # 跟随的语句不支持 WITH，则报错并退回 CTE（尽可能保留上下文）
                self.raise_error(f"{this.key} does not support CTE")
                this = cte

            return this

        # duckdb 支持以 FROM 开头（FROM-first）
        from_ = (
            self._parse_from(consume_pipe=True)
            if self._match(TokenType.FROM, advance=False)
            else None
        )

        if self._match(TokenType.SELECT):
            comments = self._prev_comments  # 保留紧邻注释

            hint = self._parse_hint()  # 方言 Hint

            # 若下一个 token 不是点（访问字段），则解析 ALL/DISTINCT 修饰
            if self._next and not self._next.token_type == TokenType.DOT:
                all_ = self._match(TokenType.ALL)
                distinct = self._match_set(self.DISTINCT_TOKENS)
            else:
                all_, distinct = None, None

            # BigQuery/某些方言：SELECT AS STRUCT / AS VALUE
            kind = (
                self._match(TokenType.ALIAS)
                and self._match_texts(("STRUCT", "VALUE"))
                and self._prev.text.upper()
            )

            # DISTINCT [ON (...)]
            if distinct:
                distinct = self.expression(
                    exp.Distinct,
                    on=self._parse_value(values=False) if self._match(TokenType.ON) else None,
                )

            if all_ and distinct:
                self.raise_error("Cannot specify both ALL and DISTINCT after SELECT")

            # 解析操作修饰符（如 SQL Server 的 WITH TIES/ALL ROWS 等，按项目具体定义）
            operation_modifiers = []
            while self._curr and self._match_texts(self.OPERATION_MODIFIERS):
                operation_modifiers.append(exp.var(self._prev.text.upper()))

            limit = self._parse_limit(top=True)  # TOP/LIMIT 等置顶限制
            projections = self._parse_projections()

            this = self.expression(
                exp.Select,
                kind=kind,
                hint=hint,
                distinct=distinct,
                expressions=projections,
                limit=limit,
                operation_modifiers=operation_modifiers or None,
            )
            this.comments = comments

            into = self._parse_into()  # INSERT INTO 风格的 SELECT INTO 目标（方言）
            if into:
                this.set("into", into)

            if not from_:
                from_ = self._parse_from()

            if from_:
                this.set("from", from_)

            # 处理 WHERE/GROUP/HAVING/WINDOW/ORDER 等查询修饰
            this = self._parse_query_modifiers(this)
        elif (table or nested) and self._match(TokenType.L_PAREN):
            # 嵌套/表上下文中的括号子句，委托给 _parse_wrapped_select
            this = self._parse_wrapped_select(table=table)

            # 提前返回，使得后续 UNION 不会附着在子查询上，而是成为父节点
            self._match_r_paren()
            return self._parse_subquery(this, parse_alias=parse_subquery_alias)
        elif self._match(TokenType.VALUES, advance=False):
            this = self._parse_derived_table_values()
        elif from_:
            # 仅有 FROM 时退化为 SELECT * FROM（duckdb FROM-first）
            this = exp.select("*").from_(from_.this, copy=False)
        elif self._match(TokenType.SUMMARIZE):
            table = self._match(TokenType.TABLE)
            this = self._parse_select() or self._parse_string() or self._parse_table()
            return self.expression(exp.Summarize, this=this, table=table)
        elif self._match(TokenType.DESCRIBE):
            this = self._parse_describe()
        elif self._match_text_seq("STREAM"):
            this = self._parse_function()
            if this:
                this = self.expression(exp.Stream, this=this)
            else:
                self._retreat(self._index - 1)
        else:
            this = None

        # 若允许集合操作，则在此阶段处理（UNION/INTERSECT/EXCEPT 等）
        return self._parse_set_operations(this) if parse_set_operation else this

    def _parse_recursive_with_search(self) -> t.Optional[exp.RecursiveWithSearch]:
        # 解析递归 CTE 的 SEARCH 子句（SQL:1999/2003），指定深度/广度等搜索策略
        self._match_text_seq("SEARCH")

        kind = self._match_texts(self.RECURSIVE_CTE_SEARCH_KIND) and self._prev.text.upper()  # 深度优先/广度优先等

        if not kind:
            # 未识别到合法搜索策略则不进入该子句
            return None

        self._match_text_seq("FIRST", "BY")  # SEARCH FIRST BY <column> [SET <name>] [USING <name>]

        return self.expression(
            exp.RecursiveWithSearch,
            kind=kind,
            this=self._parse_id_var(),  # 按某列/表达式排序以确定搜索顺序
            expression=self._match_text_seq("SET") and self._parse_id_var(),  # SET 目标列/别名
            using=self._match_text_seq("USING") and self._parse_id_var(),  # USING 子句（方言/扩展）
        )

    def _parse_with(self, skip_with_token: bool = False) -> t.Optional[exp.With]:
        # 解析 WITH 子句：可包含一个或多个 CTE，支持递归与 SEARCH 子句
        if not skip_with_token and not self._match(TokenType.WITH):
            return None

        comments = self._prev_comments  # WITH 前的注释
        recursive = self._match(TokenType.RECURSIVE)  # 可选：WITH RECURSIVE

        last_comments = None
        expressions = []
        while True:
            cte = self._parse_cte()
            if isinstance(cte, exp.CTE):
                expressions.append(cte)
                if last_comments:
                    cte.add_comments(last_comments)  # 将逗号后的注释挂到下一个 CTE

            # 允许 `WITH a AS (...), WITH b AS (...)` 两种写法（逗号或重复 WITH）
            if not self._match(TokenType.COMMA) and not self._match(TokenType.WITH):
                break
            else:
                self._match(TokenType.WITH)

            last_comments = self._prev_comments

        return self.expression(
            exp.With,
            comments=comments,
            expressions=expressions,
            recursive=recursive,
            search=self._parse_recursive_with_search(),  # 可选：解析 SEARCH FIRST BY ...
        )

    def _parse_cte(self) -> t.Optional[exp.CTE]:
        # 解析单个 CTE：别名、是否（NOT）MATERIALIZED、以及 CTE 体
        index = self._index

        alias = self._parse_table_alias(self.ID_VAR_TOKENS)  # CTE 名称及可选列列表
        if not alias or not alias.this:
            self.raise_error("Expected CTE to have alias")

        key_expressions = (
            self._parse_wrapped_id_vars() if self._match_text_seq("USING", "KEY") else None
        )

        # 某些方言要求显式 AS；若既不匹配 AS 又不允许省略，则回退并放弃该 CTE 分支
        if not self._match(TokenType.ALIAS) and not self.OPTIONAL_ALIAS_TOKEN_CTE:
            self._retreat(index)
            return None

        comments = self._prev_comments

        # PostgreSQL 等支持 MATERIALIZED/NOT MATERIALIZED 提示优化器物化策略
        if self._match_text_seq("NOT", "MATERIALIZED"):
            materialized = False
        elif self._match_text_seq("MATERIALIZED"):
            materialized = True
        else:
            materialized = None

        cte = self.expression(
            exp.CTE,
            this=self._parse_wrapped(self._parse_statement),  # CTE 体，通常为括号包裹的 SELECT/VALUES 等
            alias=alias,
            materialized=materialized,
            key_expressions=key_expressions,
            comments=comments,
        )

        # 若 CTE 体是 VALUES，则保证其以 FROM 方式接入，使后续引用更一致
        values = cte.this
        if isinstance(values, exp.Values):
            if values.alias:
                cte.set("this", exp.select("*").from_(values))
            else:
                cte.set("this", exp.select("*").from_(exp.alias_(values, "_values", table=True)))

        return cte

    def _parse_table_alias(
        self, alias_tokens: t.Optional[t.Collection[TokenType]] = None
    ) -> t.Optional[exp.TableAlias]:
        # 解析表别名：兼容 LIMIT/OFFSET 可作为标识符或子句的歧义
        # 若此时应解析为 LIMIT/OFFSET 子句（而非别名），则直接返回 None
        if self._can_parse_limit_or_offset():
            return None

        any_token = self._match(TokenType.ALIAS)  # 可选 AS
        alias = (
            self._parse_id_var(any_token=any_token, tokens=alias_tokens or self.TABLE_ALIAS_TOKENS)
            or self._parse_string_as_identifier()
        )

        index = self._index
        if self._match(TokenType.L_PAREN):
            # 列别名列表（如 table_alias(col1, col2)），使用函数参数解析器复用类型/默认值能力
            columns = self._parse_csv(self._parse_function_parameter)
            self._match_r_paren() if columns else self._retreat(index)
        else:
            columns = None

        if not alias and not columns:
            return None

        table_alias = self.expression(exp.TableAlias, this=alias, columns=columns)

        # 将标识符上的注释上移到 TableAlias，便于统一注释位置
        if isinstance(alias, exp.Identifier):
            table_alias.add_comments(alias.pop_comments())

        return table_alias

    def _parse_subquery(
        self, this: t.Optional[exp.Expression], parse_alias: bool = True
    ) -> t.Optional[exp.Subquery]:
        # 解析子查询包装：附加 PIVOT/SAMPLE/别名等表级修饰
        if not this:
            return None

        return self.expression(
            exp.Subquery,
            this=this,
            pivots=self._parse_pivots(),  # 某些方言允许在子查询层应用 PIVOT/UNPIVOT
            alias=self._parse_table_alias() if parse_alias else None,  # 子查询别名及可选列别名
            sample=self._parse_table_sample(),  # TABLESAMPLE 等
        )

    def _implicit_unnests_to_explicit(self, this: E) -> E:
        # 将隐式的 UNNEST 语义（如 JOIN 某个“看起来是表名”的字段）显式转换为 UNNEST 节点，统一 AST 形态
        # 原因：部分方言/写法里，FROM 之后直接 JOIN 一个名称，语义上其实是对上文表的嵌套字段做展开
        from sqlglot.optimizer.normalize_identifiers import normalize_identifiers as _norm

        # 记录 FROM 主表（经规范化后的）别名或名称，用于后续判断 JOIN 对象是否实际上引用了该表的某个“列”
        refs = {_norm(this.args["from"].this.copy(), dialect=self.dialect).alias_or_name}
        for i, join in enumerate(this.args.get("joins") or []):
            table = join.this
            normalized_table = table.copy()
            # 打上也许是“列”的标记，提示规范化过程：此标识符既可能是表，也可能是列
            normalized_table.meta["maybe_column"] = True
            normalized_table = _norm(normalized_table, dialect=self.dialect)

            # 仅当：JOIN 的是一个“表节点”，且没有 ON 条件（常见于隐式展开场景）
            if isinstance(table, exp.Table) and not join.args.get("on"):
                # 若该名称（规范化后）出现在已知引用集合中，说明它很可能是“上文表的列”，而非独立表
                if normalized_table.parts[0].name in refs:
                    # 将“表”视作“列引用”，并包裹成 UNNEST(expressions=[...])，显式表达展开
                    table_as_column = table.to_column()
                    unnest = exp.Unnest(expressions=[table_as_column])

                    # Table.to_column 会生成上层 Alias 节点；为与解析产物对齐，将其转成 TableAlias 并挂到 UNNEST 上
                    # 这样之后的生成/优化阶段可以基于一致的 AST 结构处理
                    if isinstance(table.args.get("alias"), exp.TableAlias):
                        table_as_column.replace(table_as_column.this)
                        exp.alias_(unnest, None, table=[table.args["alias"].this], copy=False)

                    # 用 Unnest 节点替换原先的“表”节点
                    table.replace(unnest)

            # 将当前 JOIN 对象（规范化后）的别名或名称加入集合，供后续 JOIN 判断使用
            refs.add(normalized_table.alias_or_name)

        return this

    @t.overload
    def _parse_query_modifiers(self, this: E) -> E: ...

    @t.overload
    def _parse_query_modifiers(self, this: None) -> None: ...

    def _parse_query_modifiers(self, this):
        # 解析可带修饰的查询体：累加 JOIN/LATERAL 以及其他尾部修饰（如 LIMIT/OFFSET 等）
        if isinstance(this, self.MODIFIABLES):
            for join in self._parse_joins():
                # 将解析到的 JOIN 附加到当前查询节点
                this.append("joins", join)
            for lateral in iter(self._parse_lateral, None):
                # 将解析到的 LATERAL 节点附加到当前查询节点
                this.append("laterals", lateral)

            while True:
                # 若当前位置匹配到某种查询修饰符，则调用对应解析器
                if self._match_set(self.QUERY_MODIFIER_PARSERS, advance=False):
                    modifier_token = self._curr
                    parser = self.QUERY_MODIFIER_PARSERS[modifier_token.token_type]
                    key, expression = parser(self)

                    if expression:
                        if this.args.get(key):
                            self.raise_error(
                                f"Found multiple '{modifier_token.text.upper()}' clauses",
                                token=modifier_token,
                            )

                        this.set(key, expression)
                        # 特殊处理 LIMIT：统一 AST 表达，抽出其中的 offset 以及 "BY" 等扩展字段
                        if key == "limit":
                            offset = expression.args.get("offset")
                            expression.set("offset", None)

                            if offset:
                                # 将 offset 从 limit 表达式中提取为独立 Offset 节点
                                offset = exp.Offset(expression=offset)
                                this.set("offset", offset)

                                # 某些方言（如 ClickHouse）存在 "LIMIT ... BY ..." 的扩展
                                # 这里将原先挂在 LIMIT 上的 expressions 转移到 OFFSET 上，统一后续处理
                                limit_by_expressions = expression.expressions
                                expression.set("expressions", None)
                                offset.set("expressions", limit_by_expressions)
                        continue
                break

        # 若方言支持“隐式 UNNEST”，且存在 FROM，则在解析修饰后，将隐式展开补为显式 UNNEST
        if self.SUPPORTS_IMPLICIT_UNNEST and this and this.args.get("from"):
            this = self._implicit_unnests_to_explicit(this)

        return this

    def _parse_hint_fallback_to_string(self) -> t.Optional[exp.Hint]:
        # 作为 Hint 的兜底解析：若结构化解析失败，则把后续原始 SQL 片段整体作为字符串存入 Hint
        start = self._curr
        while self._curr:
            # 吃掉后续所有 token，以便提取完整原始文本
            self._advance()

        end = self._tokens[self._index - 1]
        # 将原始 SQL 片段封装为 Hint(expressions=[raw_sql])
        return exp.Hint(expressions=[self._find_sql(start, end)])

    def _parse_hint_function_call(self) -> t.Optional[exp.Expression]:
        # Hint 中可能出现函数调用形式，直接复用通用的函数解析
        return self._parse_function_call()

    def _parse_hint_body(self) -> t.Optional[exp.Hint]:
        # 尝试以结构化的方式解析 Hint 列表，失败则回退到原文字符串
        start_index = self._index
        should_fallback_to_string = False

        hints = []
        try:
            for hint in iter(
                lambda: self._parse_csv(
                    # 单个 Hint 可是函数调用或大写标识符，按 CSV 解析多个
                    lambda: self._parse_hint_function_call() or self._parse_var(upper=True),
                ),
                [],
            ):
                hints.extend(hint)
        except ParseError:
            # 任一 Hint 结构化解析失败即触发回退
            should_fallback_to_string = True

        if should_fallback_to_string or self._curr:
            # 回退到解析开始前的位置，改走“原文字符串”兜底逻辑
            self._retreat(start_index)
            return self._parse_hint_fallback_to_string()

        # 结构化解析成功，构造 Hint 表达式
        return self.expression(exp.Hint, expressions=hints)

    def _parse_hint(self) -> t.Optional[exp.Hint]:
        # 当词法层面识别到 Hint，且有“前置注释”时，尝试将其作为 Hint 解析
        # 说明：诸如 Oracle 等方言常把 Hint 写在特殊注释中
        if self._match(TokenType.HINT) and self._prev_comments:
            return exp.maybe_parse(self._prev_comments[0], into=exp.Hint, dialect=self.dialect)

        return None

    def _parse_into(self) -> t.Optional[exp.Into]:
        # 解析 INTO 目标（可带 TEMPORARY/UNLOGGED/TABLE 修饰），统一成 Into 表达式
        if not self._match(TokenType.INTO):
            return None

        temp = self._match(TokenType.TEMPORARY)
        unlogged = self._match_text_seq("UNLOGGED")
        self._match(TokenType.TABLE)

        # 生成 Into 表达式：包含目标表、是否临时、是否 UNLOGGED 等属性
        return self.expression(
            exp.Into, this=self._parse_table(schema=True), temporary=temp, unlogged=unlogged
        )

    def _parse_from(
        self,
        joins: bool = False,
        skip_from_token: bool = False,
        consume_pipe: bool = False,
    ) -> t.Optional[exp.From]:
        # 当未跳过 FROM 关键字时，必须匹配 FROM，否则说明没有 FROM 子句
        if not skip_from_token and not self._match(TokenType.FROM):
            return None

        # 构造 From 表达式：
        # - comments: 保留 FROM 之前收集到的注释（有些方言里可能承载 Hint）
        # - this: 解析主表/子查询，允许同时解析 joins 或管道（|>）语法
        return self.expression(
            exp.From,
            comments=self._prev_comments,
            this=self._parse_table(joins=joins, consume_pipe=consume_pipe),
        )

    def _parse_match_recognize_measure(self) -> exp.MatchRecognizeMeasure:
        # 解析单个 MEASURE：
        # - window_frame: 可出现 FINAL/RUNNING 关键字，表示累计或最终窗口，取上一个匹配并标准化为大写
        # - this: 具体的度量表达式
        return self.expression(
            exp.MatchRecognizeMeasure,
            window_frame=self._match_texts(("FINAL", "RUNNING")) and self._prev.text.upper(),
            this=self._parse_expression(),
        )

    def _parse_match_recognize(self) -> t.Optional[exp.MatchRecognize]:
        # 解析 MATCH_RECOGNIZE 子句：负责解析窗口分区、排序、度量、行匹配策略、跳过策略、模式与定义等
        # 设计原因：该子句语法复杂且跨方言差异大，分步构造各子表达式以统一 AST
        if not self._match(TokenType.MATCH_RECOGNIZE):
            return None

        self._match_l_paren()

        partition = self._parse_partition_by()
        order = self._parse_order()

        # 可选的 MEASURES 子句：以 CSV 形式解析多个 measure，缺省为 None
        measures = (
            self._parse_csv(self._parse_match_recognize_measure)
            if self._match_text_seq("MEASURES")
            else None
        )

        # 解析行输出策略：ONE/ALL ROWS PER MATCH 以及可选的 SHOW/OMIT EMPTY、WITH UNMATCHED 等修饰
        if self._match_text_seq("ONE", "ROW", "PER", "MATCH"):
            # 每个匹配产出单行
            rows = exp.var("ONE ROW PER MATCH")
        elif self._match_text_seq("ALL", "ROWS", "PER", "MATCH"):
            text = "ALL ROWS PER MATCH"
            # 针对 ALL ROWS PER MATCH 的可选修饰，按出现顺序动态拼接，保留原始语义
            if self._match_text_seq("SHOW", "EMPTY", "MATCHES"):
                text += " SHOW EMPTY MATCHES"
            elif self._match_text_seq("OMIT", "EMPTY", "MATCHES"):
                text += " OMIT EMPTY MATCHES"
            elif self._match_text_seq("WITH", "UNMATCHED", "ROWS"):
                text += " WITH UNMATCHED ROWS"
            rows = exp.var(text)
        else:
            rows = None

        # 解析跳过策略 AFTER MATCH SKIP：指定下一次匹配起点
        if self._match_text_seq("AFTER", "MATCH", "SKIP"):
            text = "AFTER MATCH SKIP"
            if self._match_text_seq("PAST", "LAST", "ROW"):
                text += " PAST LAST ROW"
            elif self._match_text_seq("TO", "NEXT", "ROW"):
                text += " TO NEXT ROW"
            elif self._match_text_seq("TO", "FIRST"):
                # 需要读一个后续标识符（如子模式名），因此显式前进一个 token
                text += f" TO FIRST {self._advance_any().text}"  # type: ignore
            elif self._match_text_seq("TO", "LAST"):
                # 同理，LAST 也需要携带一个标识符
                text += f" TO LAST {self._advance_any().text}"  # type: ignore
            after = exp.var(text)
        else:
            after = None

        # 解析 PATTERN 子句：读取括号内的原始模式串，并保持括号平衡
        if self._match_text_seq("PATTERN"):
            self._match_l_paren()

            if not self._curr:
                self.raise_error("Expecting )", self._curr)

            # 通过括号计数追踪嵌套，直到配对完成；原因：模式内部可能包含括号，不能简单遇到一个 ) 就结束
            paren = 1
            start = self._curr

            while self._curr and paren > 0:
                if self._curr.token_type == TokenType.L_PAREN:
                    paren += 1
                if self._curr.token_type == TokenType.R_PAREN:
                    paren -= 1

                end = self._prev
                self._advance()

            if paren > 0:
                self.raise_error("Expecting )", self._curr)

            # 将括号内的原始文本切片作为模式内容保留，方便后续方言特定处理
            pattern = exp.var(self._find_sql(start, end))
        else:
            pattern = None

        define = (
            self._parse_csv(self._parse_name_as_expression)
            if self._match_text_seq("DEFINE")
            else None
        )

        self._match_r_paren()

        # 组装最终的 MatchRecognize 表达式：将前面解析到的各组成部分统一挂载
        return self.expression(
            exp.MatchRecognize,
            partition_by=partition,
            order=order,
            measures=measures,
            rows=rows,
            after=after,
            pattern=pattern,
            define=define,
            alias=self._parse_table_alias(),
        )

    def _parse_lateral(self) -> t.Optional[exp.Lateral]:
        # 支持多种侧向联接语法：
        # - CROSS APPLY / OUTER APPLY（T-SQL 等方言）
        # - LATERAL（Postgres 等方言）
        # 设计原因：不同方言语义等价但关键字不同，这里统一解析为 Lateral 表达式
        cross_apply = self._match_pair(TokenType.CROSS, TokenType.APPLY)
        # cross_apply 三态：
        # - True: 匹配到 CROSS APPLY
        # - False: 匹配到 OUTER APPLY
        # - None: 未匹配到 APPLY 语法
        if not cross_apply and self._match_pair(TokenType.OUTER, TokenType.APPLY):
            cross_apply = False

        if cross_apply is not None:
            # APPLY 语法后面通常跟一个“可当作表的 select”
            this = self._parse_select(table=True)
            view = None
            outer = None
        elif self._match(TokenType.LATERAL):
            # LATERAL 语法下允许 VIEW/OUTER 等修饰
            this = self._parse_select(table=True)
            view = self._match(TokenType.VIEW)
            outer = self._match(TokenType.OUTER)
        else:
            return None

        if not this:
            # 若 LATERAL/APPLY 后未直接出现子查询，尝试解析为 UNNEST/函数/标识符
            # 原因：很多场景中 LATERAL 用于展开函数返回的表或数组
            this = (
                self._parse_unnest()
                or self._parse_function()
                or self._parse_id_var(any_token=False)
            )

            # 支持点号链式访问，如 schema.func 或 obj.method 形式
            while self._match(TokenType.DOT):
                this = exp.Dot(
                    this=this,
                    expression=self._parse_function() or self._parse_id_var(any_token=False),
                )

        ordinality: t.Optional[bool] = None

        if view:
            # LATERAL VIEW col_list 形式：构造表别名及列别名列表
            table = self._parse_id_var(any_token=False)
            columns = self._parse_csv(self._parse_id_var) if self._match(TokenType.ALIAS) else []
            table_alias: t.Optional[exp.TableAlias] = self.expression(
                exp.TableAlias, this=table, columns=columns
            )
        elif isinstance(this, (exp.Subquery, exp.Unnest)) and this.alias:
            # 将子节点上的别名上移到 Lateral 本身，保持 AST 结构一致性，便于后续处理
            # We move the alias from the lateral's child node to the lateral itself
            table_alias = this.args["alias"].pop()
        else:
            # 支持 WITH ORDINALITY（如 Postgres），为行附加序号列
            ordinality = self._match_pair(TokenType.WITH, TokenType.ORDINALITY)
            table_alias = self._parse_table_alias()

        # 输出标准化的 Lateral 表达式，统一下游优化与生成逻辑
        return self.expression(
            exp.Lateral,
            this=this,
            view=view,
            outer=outer,
            alias=table_alias,
            cross_apply=cross_apply,
            ordinality=ordinality,
        )

    def _parse_join_parts(
        self,
    ) -> t.Tuple[t.Optional[Token], t.Optional[Token], t.Optional[Token]]:
        # 解析 JOIN 的三个可选部分：
        # - method（如 NATURAL、SEMI、ANTI 等）
        # - side（如 LEFT、RIGHT、FULL 等）
        # - kind（如 INNER、CROSS、STRAIGHT_JOIN、ARRAY 等）
        # 返回各自命中的 Token（若未命中为 None），供后续 _parse_join 组装使用
        return (
            self._match_set(self.JOIN_METHODS) and self._prev,
            self._match_set(self.JOIN_SIDES) and self._prev,
            self._match_set(self.JOIN_KINDS) and self._prev,
        )

    def _parse_using_identifiers(self) -> t.List[exp.Expression]:
        # 解析 USING (col1, col2, ...) 中的标识符列表
        # 设计原因：USING 列表有时写成列引用（table.col），但语义上只需要标识符名
        def _parse_column_as_identifier() -> t.Optional[exp.Expression]:
            this = self._parse_column()
            if isinstance(this, exp.Column):
                # 若解析到列引用，取其标识符部分（去除表前缀），与 USING 语义对齐
                return this.this
            return this

        # USING 的括号为可选（部分方言允许），此处以 CSV 解析返回表达式列表
        return self._parse_wrapped_csv(_parse_column_as_identifier, optional=True)

    def _parse_join(
        self, skip_join_token: bool = False, parse_bracket: bool = False
    ) -> t.Optional[exp.Join]:
        # 逗号作为 JOIN：等价于 CROSS JOIN（在某些方言/设置下）
        if self._match(TokenType.COMMA):
            table = self._try_parse(self._parse_table)
            cross_join = self.expression(exp.Join, this=table) if table else None

            # 当设置为“JOIN 同等优先级”时，将逗号 JOIN 归一为 CROSS
            if cross_join and self.JOINS_HAVE_EQUAL_PRECEDENCE:
                cross_join.set("kind", "CROSS")

            return cross_join

        # 捕获当前位置，若后续确认非 JOIN，将回退到此处
        index = self._index
        # method/side/kind 对应 NATURAL/SEMI/ANTI、LEFT/RIGHT/FULL、INNER/CROSS/ARRAY 等可选部分
        method, side, kind = self._parse_join_parts()
        # 方言特定的 JOIN hint，如 /*+ BROADCAST */ 等，保留其原文文本
        hint = self._prev.text if self._match_texts(self.JOIN_HINTS) else None
        # 判断是否真正出现 JOIN 关键字，或是 MySQL 的 STRAIGHT_JOIN 这类 kind
        join = self._match(TokenType.JOIN) or (kind and kind.token_type == TokenType.STRAIGHT_JOIN)
        join_comments = self._prev_comments

        # 若未命中 JOIN 且不能跳过，则回退并清空前面捕获的可选部分
        if not skip_join_token and not join:
            self._retreat(index)
            kind = None
            method = None
            side = None

        # 支持 OUTER/CROSS APPLY（T-SQL 等），与 JOIN 平级判断
        outer_apply = self._match_pair(TokenType.OUTER, TokenType.APPLY, False)
        cross_apply = self._match_pair(TokenType.CROSS, TokenType.APPLY, False)

        # 若既非 JOIN 也非 APPLY，则不是一个 join 片段
        if not skip_join_token and not join and not outer_apply and not cross_apply:
            return None

        # this 是被联接的目标，可能是表/子查询/函数/UNNEST 等
        kwargs: t.Dict[str, t.Any] = {"this": self._parse_table(parse_bracket=parse_bracket)}
        # 特殊：ARRAY JOIN（如某些方言），允许在 kind=ARRAY 后跟多个目标，以逗号分隔
        if kind and kind.token_type == TokenType.ARRAY and self._match(TokenType.COMMA):
            kwargs["expressions"] = self._parse_csv(
                lambda: self._parse_table(parse_bracket=parse_bracket)
            )

        # 将前面解析到的可选属性挂载到 join 节点
        if method:
            kwargs["method"] = method.text
        if side:
            kwargs["side"] = side.text
        if kind:
            kwargs["kind"] = kind.text
        if hint:
            kwargs["hint"] = hint

        # 方言扩展：MATCH_CONDITION（如 Oracle MATCH_RECOGNIZE 的 JOIN 变体）
        if self._match(TokenType.MATCH_CONDITION):
            kwargs["match_condition"] = self._parse_wrapped(self._parse_comparison)

        # 解析 ON / USING 条件
        if self._match(TokenType.ON):
            kwargs["on"] = self._parse_assignment()
        elif self._match(TokenType.USING):
            kwargs["using"] = self._parse_using_identifiers()
        # 尝试“延迟获取” join 条件：当没有显式 method/APPLY/UNNEST/CROSS/ARRAY 时
        elif (
            not method
            and not (outer_apply or cross_apply)
            and not isinstance(kwargs["this"], exp.Unnest)
            and not (kind and kind.token_type in (TokenType.CROSS, TokenType.ARRAY))
        ):
            # 先临时解析后续 joins 看看是否带条件，若失败再整体回退
            index = self._index
            joins: t.Optional[list] = list(self._parse_joins())

            if joins and self._match(TokenType.ON):
                kwargs["on"] = self._parse_assignment()
            elif joins and self._match(TokenType.USING):
                kwargs["using"] = self._parse_using_identifiers()
            else:
                joins = None
                self._retreat(index)

            # 若解析到了子 join，将其附加到当前被联接对象 this 上
            kwargs["this"].set("joins", joins if joins else None)

        # 解析 PIVOT（如某些方言），并统一挂载
        kwargs["pivots"] = self._parse_pivots()

        # 合并 join 关键字与修饰上的注释，便于后续生成保留这些信息
        comments = [c for token in (method, side, kind) if token for c in token.comments]
        comments = (join_comments or []) + comments

        if (
            self.ADD_JOIN_ON_TRUE
            and not kwargs.get("on")
            and not kwargs.get("using")
            and not kwargs.get("method")
            and kwargs.get("kind") in (None, "INNER", "OUTER")
        ):
            kwargs["on"] = exp.true()

        return self.expression(exp.Join, comments=comments, **kwargs)

    def _parse_opclass(self) -> t.Optional[exp.Expression]:
        # 解析 opclass（操作符类）：用于索引/排序规则等方言特性
        # 设计原因：当后续未出现特定“跟随关键字/符号”时，将当前 assignment 归一为 Opclass 节点
        this = self._parse_assignment()

        # 若后续出现被视为“紧随 opclass 之后”的关键字，则直接返回原表达式，不包装为 Opclass
        if self._match_texts(self.OPCLASS_FOLLOW_KEYWORDS, advance=False):
            return this

        # 若后续未出现“操作符类型跟随 token”，则将 this 与 table_parts 组装成 Opclass
        if not self._match_set(self.OPTYPE_FOLLOW_TOKENS, advance=False):
            return self.expression(exp.Opclass, this=this, expression=self._parse_table_parts())

        # 否则返回原表达式（说明它并非独立的 opclass 定义场景）
        return this

    def _parse_index_params(self) -> exp.IndexParameters:
        # 解析索引参数（广义）：USING/列列表/INCLUDE/PARTITION BY/WITH 存储选项/表空间/WHERE/ON
        # 设计原因：不同方言对索引定义的可选项差异较大，此处集中解析并统一成 IndexParameters
        using = self._parse_var(any_token=True) if self._match(TokenType.USING) else None

        # 可选的列列表（括号可选的情况下需提前窥探）
        if self._match(TokenType.L_PAREN, advance=False):
            columns = self._parse_wrapped_csv(self._parse_with_operator)
        else:
            columns = None

        include = self._parse_wrapped_id_vars() if self._match_text_seq("INCLUDE") else None
        partition_by = self._parse_partition_by()
        # WITH 后接存储属性（如 fillfactor 等），按属性表的形式封装
        with_storage = self._match(TokenType.WITH) and self._parse_wrapped_properties()
        # 某些方言：USING INDEX TABLESPACE <name>
        tablespace = (
            self._parse_var(any_token=True)
            if self._match_text_seq("USING", "INDEX", "TABLESPACE")
            else None
        )
        where = self._parse_where()

        # ON <field>：如对某个存储或表空间进一步限定
        on = self._parse_field() if self._match(TokenType.ON) else None

        # 统一输出 IndexParameters，便于下游生成或优化
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

    def _parse_index(
        self, index: t.Optional[exp.Expression] = None, anonymous: bool = False
    ) -> t.Optional[exp.Index]:
        # 解析 CREATE/ALTER 语句中的索引定义；支持匿名索引与显式索引名
        if index or anonymous:
            # 匿名或已给定索引名的分支：此时 unique/primary/amp 等属性不在前缀位置出现
            unique = None
            primary = None
            amp = None

            self._match(TokenType.ON)
            self._match(TokenType.TABLE)  # hive 方言可能出现 TABLE 关键字
            table = self._parse_table_parts(schema=True)
        else:
            # 显式索引名的分支：可能带 UNIQUE/PRIMARY/AMP 等前缀修饰
            unique = self._match(TokenType.UNIQUE)
            primary = self._match_text_seq("PRIMARY")
            amp = self._match_text_seq("AMP")

            if not self._match(TokenType.INDEX):
                return None

            index = self._parse_id_var()
            table = None

        # 解析索引参数（USING/列/INCLUDE/分区/存储/表空间/过滤/作用对象）
        params = self._parse_index_params()

        # 统一输出 Index 节点，便于生成器/优化器处理不同方言的索引定义
        return self.expression(
            exp.Index,
            this=index,
            table=table,
            unique=unique,
            primary=primary,
            amp=amp,
            params=params,
        )

    def _parse_table_hints(self) -> t.Optional[t.List[exp.Expression]]:
        # 解析表级 Hint：
        # - T-SQL: WITH (<hint_list>) 形式，解析为 WithTableHint，允许函数或标识符作为项
        # - MySQL: USE/FORCE/IGNORE {INDEX|KEY} (idx_list) 解析为 IndexTableHint
        hints: t.List[exp.Expression] = []
        if self._match_pair(TokenType.WITH, TokenType.L_PAREN):
            # https://learn.microsoft.com/en-us/sql/t-sql/queries/hints-transact-sql-table?view=sql-server-ver16
            hints.append(
                self.expression(
                    exp.WithTableHint,
                    expressions=self._parse_csv(
                        lambda: self._parse_function() or self._parse_var(any_token=True)
                    ),
                )
            )
            self._match_r_paren()
        else:
            # https://dev.mysql.com/doc/refman/8.0/en/index-hints.html
            while self._match_set(self.TABLE_INDEX_HINT_TOKENS):
                # 记录 Hint 的主类型（如 USE/FORCE/IGNORE），归一为大写文本
                hint = exp.IndexTableHint(this=self._prev.text.upper())

                # 可选的 {INDEX|KEY} 关键字（MySQL 同义），如果存在则消耗
                self._match_set((TokenType.INDEX, TokenType.KEY))
                if self._match(TokenType.FOR):
                    # FOR 子句指定目标（JOIN/ORDER BY/GROUP BY 等），保存为上层语义
                    hint.set("target", self._advance_any() and self._prev.text.upper())

                # 包裹的索引名列表
                hint.set("expressions", self._parse_wrapped_id_vars())
                hints.append(hint)

        return hints or None

    def _parse_table_part(self, schema: bool = False) -> t.Optional[exp.Expression]:
        # 解析“表名片段”：可能是函数、标识符、字符串标识符或占位符
        # 原因：FROM/引用路径中，表位可能由函数返回表、带引号的标识符或参数化占位符构成
        return (
            (not schema and self._parse_function(optional_parens=False))
            or self._parse_id_var(any_token=False)
            or self._parse_string_as_identifier()
            or self._parse_placeholder()
        )

    def _parse_table_parts(
        self, schema: bool = False, is_db_reference: bool = False, wildcard: bool = False
    ) -> exp.Table:
        # 解析一个“表引用”的完整路径：可能包含 catalog.db.table 或 db.table
        catalog = None
        db = None
        table: t.Optional[exp.Expression | str] = self._parse_table_part(schema=schema)

        while self._match(TokenType.DOT):
            if catalog:
                # 若 catalog 已经存在，说明之前已出现 catalog 和 db；此时把后续继续嵌套为 Dot 表达式
                # 原因：部分方言支持更深层级的命名或对象（如 schema.object.method）
                table = self.expression(
                    exp.Dot, this=table, expression=self._parse_table_part(schema=schema)
                )
            else:
                # 首次遇到点号，进行左移：catalog <- db, db <- table，再继续解析 table 片段
                catalog = db
                db = table
                # 兼容 T-SQL 的 a..b 写法，此时中间数据库名可为空串
                table = self._parse_table_part(schema=schema) or ""

        if (
            wildcard
            and self._is_connected()
            and (isinstance(table, exp.Identifier) or not table)
            and self._match(TokenType.STAR)
        ):
            # 支持通配符，形如 db.* 或 *，将 * 合并进标识符末尾或直接作为标识符
            if isinstance(table, exp.Identifier):
                table.args["this"] += "*"
            else:
                table = exp.Identifier(this="*")

        # 将标识符上的注释“上浮”到 Table 节点，便于统一保留输出
        comments = table.pop_comments() if isinstance(table, exp.Expression) else None

        if is_db_reference:
            # 仅引用到数据库层级，如 catalog.db 的形式：此时 table 置空，db 上移到 table 位
            catalog = db
            db = table
            table = None

        # 基本健壮性校验：确保在不同语义下拿到了必要层级
        if not table and not is_db_reference:
            self.raise_error(f"Expected table name but got {self._curr}")
        if not db and is_db_reference:
            self.raise_error(f"Expected database name but got {self._curr}")

        # 构造 Table 节点，挂载解析得到的层级信息
        table = self.expression(
            exp.Table,
            comments=comments,
            this=table,
            db=db,
            catalog=catalog,
        )

        # 表级变更语义（如系统时间版本表），与时间点查询（AS OF/BEFORE）
        changes = self._parse_changes()
        if changes:
            table.set("changes", changes)

        at_before = self._parse_historical_data()
        if at_before:
            table.set("when", at_before)

        # 支持表引用后的 PIVOT 语法，统一挂载
        pivots = self._parse_pivots()
        if pivots:
            table.set("pivots", pivots)

        return table

    def _parse_table(
        self,
        schema: bool = False,
        joins: bool = False,
        alias_tokens: t.Optional[t.Collection[TokenType]] = None,
        parse_bracket: bool = False,
        is_db_reference: bool = False,
        parse_partition: bool = False,
        consume_pipe: bool = False,
    ) -> t.Optional[exp.Expression]:
        # 解析一个“表位”单元：可能是 LATERAL/UNNEST/VALUES/子查询/括号包裹的对象/ROWS FROM 等多种形式
        # 设计原因：不同方言允许在 FROM 中出现多形态来源，将其统一抽象为表表达式
        lateral = self._parse_lateral()
        if lateral:
            return lateral

        unnest = self._parse_unnest()
        if unnest:
            return unnest

        values = self._parse_derived_table_values()
        if values:
            return values

        # 解析子查询作为表；若开启管道语法（consume_pipe），子查询里也会消费 |> 连接
        subquery = self._parse_select(table=True, consume_pipe=consume_pipe)
        if subquery:
            # 子查询若尚未设置 pivots，则在此阶段补齐，保持 AST 一致
            if not subquery.args.get("pivots"):
                subquery.set("pivots", self._parse_pivots())
            return subquery

        # 可选的括号包裹对象（如 (table) 或 (subquery)），并包装成 Table 节点
        bracket = parse_bracket and self._parse_bracket(None)
        bracket = self.expression(exp.Table, this=bracket) if bracket else None

        # ROWS FROM(...) 语法（Postgres），解析为包含多个表函数的组合
        rows_from = self._match_text_seq("ROWS", "FROM") and self._parse_wrapped_csv(
            self._parse_table
        )
        rows_from = self.expression(exp.Table, rows_from=rows_from) if rows_from else None

        only = self._match(TokenType.ONLY)

        # 三选一：括号对象、ROWS FROM、普通的 catalog.db.table（支持括号包裹）
        this = t.cast(
            exp.Expression,
            bracket
            or rows_from
            or self._parse_bracket(
                self._parse_table_parts(schema=schema, is_db_reference=is_db_reference)
            ),
        )

        if only:
            # ONLY(table) 语义（Postgres）：限制扫描不包含继承/分区子表
            this.set("only", only)

        # Postgres 支持 table 后缀 *（无副作用，此处仅消费以保持语法一致）
        self._match_text_seq("*")

        # 分区选择语法：当开启支持或调用方要求解析分区时，解析 PARTITION 子句
        parse_partition = parse_partition or self.SUPPORTS_PARTITION_SELECTION
        if parse_partition and self._match(TokenType.PARTITION, advance=False):
            this.set("partition", self._parse_partition())

        # DDL 场景：若在 schema 模式下，后续进入表结构解析分支
        if schema:
            return self._parse_schema(this=this)

        # 版本/快照（如 FOR SYSTEM_TIME AS OF 或 AT TIME ZONE 等），统一挂载到 version
        version = self._parse_version()

        if version:
            this.set("version", version)

        # 一些方言要求在表别名之后再写 TABLESAMPLE，这里根据方言控制 sample 的解析时机
        if self.dialect.ALIAS_POST_TABLESAMPLE:
            this.set("sample", self._parse_table_sample())

        # 表别名（以及列别名列表）的解析
        alias = self._parse_table_alias(alias_tokens=alias_tokens or self.TABLE_ALIAS_TOKENS)
        if alias:
            this.set("alias", alias)

        # 语法：table AT index_name（如某些方言的版本/索引指向语义）
        if isinstance(this, exp.Table) and self._match_text_seq("AT"):
            return self.expression(
                exp.AtIndex, this=this.to_column(copy=False), expression=self._parse_id_var()
            )

        # 表级 Hint（T-SQL/MySQL 等）
        this.set("hints", self._parse_table_hints())

        # 若此前未挂载 pivots，则在此阶段补齐，保证统一
        if not this.args.get("pivots"):
            this.set("pivots", self._parse_pivots())

        # 若方言要求在别名前写 sample，则这里在别名之后再解析 sample，以满足两种顺序
        if not self.dialect.ALIAS_POST_TABLESAMPLE:
            this.set("sample", self._parse_table_sample())

        # 可选：在解析表后立即解析 JOIN 列表并附加
        if joins:
            for join in self._parse_joins():
                this.append("joins", join)

        # WITH ORDINALITY：为 LATERAL/函数返回的表附加序号列
        if self._match_pair(TokenType.WITH, TokenType.ORDINALITY):
            this.set("ordinality", True)
            this.set("alias", self._parse_table_alias())

        return this

    def _parse_version(self) -> t.Optional[exp.Version]:
        # 解析版本/快照子句：TIMESTAMP/_VERSION + (FROM/BETWEEN/CONTAINED IN/ALL/AS OF)
        # 设计原因：不同方言支持多种时间旅行/快照语法，统一归一到 Version 表达式
        if self._match(TokenType.TIMESTAMP_SNAPSHOT):
            this = "TIMESTAMP"
        elif self._match(TokenType.VERSION_SNAPSHOT):
            this = "VERSION"
        else:
            return None

        # 区间版本：FROM/ BETWEEN start TO/AND end，归一为 Tuple(start, end)
        if self._match_set((TokenType.FROM, TokenType.BETWEEN)):
            kind = self._prev.text.upper()
            start = self._parse_bitwise()
            self._match_texts(("TO", "AND"))
            end = self._parse_bitwise()
            expression: t.Optional[exp.Expression] = self.expression(
                exp.Tuple, expressions=[start, end]
            )
        # 集合版本：CONTAINED IN (v1, v2, ...)
        elif self._match_text_seq("CONTAINED", "IN"):
            kind = "CONTAINED IN"
            expression = self.expression(
                exp.Tuple, expressions=self._parse_wrapped_csv(self._parse_bitwise)
            )
        # 全量：ALL 表示不限定版本
        elif self._match(TokenType.ALL):
            kind = "ALL"
            expression = None
        else:
            # 点时间：AS OF <timestamp|version>
            self._match_text_seq("AS", "OF")
            kind = "AS OF"
            expression = self._parse_type()

        return self.expression(exp.Version, this=this, expression=expression, kind=kind)

    def _parse_historical_data(self) -> t.Optional[exp.HistoricalData]:
        # 解析历史数据子句（如 Snowflake 的 AT/BEFORE）：支持可选的 (KIND) => 表达式
        # 设计原因：该结构可能半途失败（缺少表达式），需要在失败时整体回退，保证语法同步
        # https://docs.snowflake.com/en/sql-reference/constructs/at-before
        index = self._index
        historical_data = None
        if self._match_texts(self.HISTORICAL_DATA_PREFIX):
            this = self._prev.text.upper()
            # 可选的 (KIND) 部分，如 (TIMESTAMP) 或 (STATEMENT)
            kind = (
                self._match(TokenType.L_PAREN)
                and self._match_texts(self.HISTORICAL_DATA_KIND)
                and self._prev.text.upper()
            )
            # 使用 => 连接的右侧表达式（如具体时间戳/语句 ID）
            expression = self._match(TokenType.FARROW) and self._parse_bitwise()

            if expression:
                self._match_r_paren()
                historical_data = self.expression(
                    exp.HistoricalData, this=this, kind=kind, expression=expression
                )
            else:
                # 匹配开头但未拿到表达式，说明不是一个完整历史子句；回退避免误消耗 token
                self._retreat(index)

        return historical_data

    def _parse_changes(self) -> t.Optional[exp.Changes]:
        # 解析 CHANGES(information => ...) 结构，并允许附带 at_before/end 历史点
        # 设计原因：部分数据系统支持对变更流进行时间片段查询
        if not self._match_text_seq("CHANGES", "(", "INFORMATION", "=>"):
            return None

        information = self._parse_var(any_token=True)
        self._match_r_paren()

        return self.expression(
            exp.Changes,
            information=information,
            at_before=self._parse_historical_data(),
            end=self._parse_historical_data(),
        )

    def _parse_unnest(self, with_alias: bool = True) -> t.Optional[exp.Unnest]:
        # 解析 UNNEST：将数组/复杂类型展开为行
        # 设计原因：不同方言的 UNNEST 支持列别名、WITH ORDINALITY/OFFSET 等可选项，需统一 AST
        if not self._match_pair(TokenType.UNNEST, TokenType.L_PAREN, advance=False):
            return None

        self._advance()

        # UNNEST(expr1, expr2, ...) 形式的表达式列表
        expressions = self._parse_wrapped_csv(self._parse_equality)
        # WITH ORDINALITY：为每行附带序号列（方言特性）
        offset = self._match_pair(TokenType.WITH, TokenType.ORDINALITY)

        # 可选的表别名/列别名列表
        alias = self._parse_table_alias() if with_alias else None

        if alias:
            if self.dialect.UNNEST_COLUMN_ONLY:
                # 某些方言只允许给列起别名，不允许表别名；若用户提供了列别名列表，则报错
                if alias.args.get("columns"):
                    self.raise_error("Unexpected extra column alias in unnest.")

                # 将原本的表别名提升为列别名，清空表别名本体
                alias.set("columns", [alias.this])
                alias.set("this", None)

            # 若存在列别名列表，且 WITH ORDINALITY 被使用但表达式数量少于列数，
            # 则将最后一个列别名视为 ordinality 列名（与常见语义保持一致）
            columns = alias.args.get("columns") or []
            if offset and len(expressions) < len(columns):
                offset = columns.pop()

        # WITH OFFSET as <alias> 语法：指定序号列名；若未指定则默认 offset
        if not offset and self._match_pair(TokenType.WITH, TokenType.OFFSET):
            self._match(TokenType.ALIAS)
            offset = self._parse_id_var(
                any_token=False, tokens=self.UNNEST_OFFSET_ALIAS_TOKENS
            ) or exp.to_identifier("offset")

        return self.expression(exp.Unnest, expressions=expressions, alias=alias, offset=offset)

    def _parse_derived_table_values(self) -> t.Optional[exp.Values]:
        # 解析 VALUES 表达式作为“派生表”：
        # - (VALUES ...) 形式（派生）
        # - VALUES 或 FORMAT VALUES（ClickHouse）
        is_derived = self._match_pair(TokenType.L_PAREN, TokenType.VALUES)
        if not is_derived and not (
            # ClickHouse 的 `FORMAT Values` 等价于 `VALUES`
            self._match_text_seq("VALUES") or self._match_text_seq("FORMAT", "VALUES")
        ):
            return None

        # VALUES 后跟 CSV 的值列表，每一项可为一行或标量，交由 _parse_value 处理
        expressions = self._parse_csv(self._parse_value)
        alias = self._parse_table_alias()

        if is_derived:
            # 若以 (VALUES ...) 包裹，则需要补右括号
            self._match_r_paren()

        # 若未显式提供别名，部分方言仍允许后置别名；因此用已有别名或再尝试解析一次
        return self.expression(
            exp.Values, expressions=expressions, alias=alias or self._parse_table_alias()
        )

    def _parse_table_sample(self, as_modifier: bool = False) -> t.Optional[exp.TableSample]:
        # 解析 TABLESAMPLE/USING SAMPLE：支持百分比、行数、分桶与随机种子
        # 设计原因：不同方言在采样语法与单位（百分比/行数）上存在差异，需要统一抽象
        if not self._match(TokenType.TABLE_SAMPLE) and not (
            as_modifier and self._match_text_seq("USING", "SAMPLE")
        ):
            return None

        bucket_numerator = None
        bucket_denominator = None
        bucket_field = None
        percent = None
        size = None
        seed = None

        method = self._parse_var(tokens=(TokenType.ROW,), upper=True)
        matched_l_paren = self._match(TokenType.L_PAREN)

        if self.TABLESAMPLE_CSV:
            # 某些方言允许写成 TABLESAMPLE(method(expr1, expr2, ...)) 的 CSV 参数
            num = None
            expressions = self._parse_csv(self._parse_primary)
        else:
            expressions = None
            # 采样比例/大小可以是数字、表达式或占位符
            num = (
                self._parse_factor()
                if self._match(TokenType.NUMBER, advance=False)
                else self._parse_primary() or self._parse_placeholder()
            )

        # 分桶采样：BUCKET x OUT OF y ON <field>
        if self._match_text_seq("BUCKET"):
            bucket_numerator = self._parse_number()
            self._match_text_seq("OUT", "OF")
            bucket_denominator = bucket_denominator = self._parse_number()
            self._match(TokenType.ON)
            bucket_field = self._parse_field()
        # 百分比：PERCENT 或 %，将 num 解释为百分比
        elif self._match_set((TokenType.PERCENT, TokenType.MOD)):
            percent = num
        # 行数：ROWS 或当方言定义 size 非百分比时
        elif self._match(TokenType.ROWS) or not self.dialect.TABLESAMPLE_SIZE_IS_PERCENT:
            size = num
        else:
            # 默认按百分比处理
            percent = num

        if matched_l_paren:
            self._match_r_paren()

        # 可选 method(seed) 尾随参数，或 SEED/REPEATABLE 包裹种子
        if self._match(TokenType.L_PAREN):
            method = self._parse_var(upper=True)
            seed = self._match(TokenType.COMMA) and self._parse_number()
            self._match_r_paren()
        elif self._match_texts(("SEED", "REPEATABLE")):
            seed = self._parse_wrapped(self._parse_number)

        # 若未解析到 method，使用默认的采样方法（如 SYSTEM/BERNOULLI）
        if not method and self.DEFAULT_SAMPLING_METHOD:
            method = exp.var(self.DEFAULT_SAMPLING_METHOD)

        return self.expression(
            exp.TableSample,
            expressions=expressions,
            method=method,
            bucket_numerator=bucket_numerator,
            bucket_denominator=bucket_denominator,
            bucket_field=bucket_field,
            percent=percent,
            size=size,
            seed=seed,
        )

    def _parse_pivots(self) -> t.Optional[t.List[exp.Pivot]]:
        # 解析连续的 PIVOT/UNPIVOT 片段，直到不再匹配为止；若无则返回 None
        return list(iter(self._parse_pivot, None)) or None

    def _parse_joins(self) -> t.Iterator[exp.Join]:
        # 迭代解析多个 JOIN 片段，直至无法继续匹配
        return iter(self._parse_join, None)

    def _parse_unpivot_columns(self) -> t.Optional[exp.UnpivotColumns]:
        # 解析 UNPIVOT 的列目标描述：INTO NAME <col> VALUE (<col_list>)
        # 设计原因：UNPIVOT 需要明确行名称列与值列的映射
        if not self._match(TokenType.INTO):
            return None

        return self.expression(
            exp.UnpivotColumns,
            this=self._match_text_seq("NAME") and self._parse_column(),
            expressions=self._match_text_seq("VALUE") and self._parse_csv(self._parse_column),
        )

    # https://duckdb.org/docs/sql/statements/pivot
    def _parse_simplified_pivot(self, is_unpivot: t.Optional[bool] = None) -> exp.Pivot:
        # DuckDB 简化语法的 PIVOT/UNPIVOT 解析：
        # - PIVOT: FROM <table> ON <col or expr> IN (<row values...>) USING <aggs>
        # - UNPIVOT: FROM <table> ON (<cols...>) AS <row_val> USING <expr AS col>
        def _parse_on() -> t.Optional[exp.Expression]:
            this = self._parse_bitwise()

            if self._match(TokenType.IN):
                # PIVOT ... ON col IN (row_val1, row_val2)
                return self._parse_in(this)
            if self._match(TokenType.ALIAS, advance=False):
                # UNPIVOT ... ON (col1, col2, col3) AS row_val
                return self._parse_alias(this)

            return this

        this = self._parse_table()
        # ON 子句可选：PIVOT 时解析 IN 列表；UNPIVOT 时解析列列表 + 行名别名
        expressions = self._match(TokenType.ON) and self._parse_csv(_parse_on)
        into = self._parse_unpivot_columns()
        # USING 后通常接聚合或表达式，并允许为其取别名作为列名
        using = self._match(TokenType.USING) and self._parse_csv(
            lambda: self._parse_alias(self._parse_function())
        )
        group = self._parse_group()

        return self.expression(
            exp.Pivot,
            this=this,
            expressions=expressions,
            using=using,
            group=group,
            unpivot=is_unpivot,
            into=into,
        )

    def _parse_pivot_in(self) -> exp.In:
        # 解析 PIVOT 的 IN 子句：IN (expr [AS alias], ... | ANY ORDER BY ...)
        def _parse_aliased_expression() -> t.Optional[exp.Expression]:
            this = self._parse_select_or_expression()

            # 允许对枚举值取别名，作为生成列名使用
            self._match(TokenType.ALIAS)
            alias = self._parse_bitwise()
            if alias:
                # 若别名是列引用且无库前缀，仅取其标识符部分
                if isinstance(alias, exp.Column) and not alias.db:
                    alias = alias.this
                return self.expression(exp.PivotAlias, this=this, alias=alias)

            return this

        value = self._parse_column()

        if not self._match_pair(TokenType.IN, TokenType.L_PAREN):
            self.raise_error("Expecting IN (")

        # DuckDB 扩展：IN (ANY ORDER BY ...) 形式
        if self._match(TokenType.ANY):
            exprs: t.List[exp.Expression] = ensure_list(exp.PivotAny(this=self._parse_order()))
        else:
            exprs = self._parse_csv(_parse_aliased_expression)

        self._match_r_paren()
        return self.expression(exp.In, this=value, expressions=exprs)

    def _parse_pivot_aggregation(self) -> t.Optional[exp.Expression]:
        # 解析 PIVOT 的聚合函数，并允许为其取别名作为列名
        func = self._parse_function()
        if not func:
            if self._prev and self._prev.token_type == TokenType.COMMA:
                return None
            self.raise_error("Expecting an aggregation function in PIVOT")

        return self._parse_alias(func)

    def _parse_pivot(self) -> t.Optional[exp.Pivot]:
        # 解析 PIVOT/UNPIVOT 主体：支持 Databricks 的 INCLUDE/EXCLUDE NULLS 选项
        index = self._index
        include_nulls = None

        if self._match(TokenType.PIVOT):
            unpivot = False
        elif self._match(TokenType.UNPIVOT):
            unpivot = True

            # https://docs.databricks.com/en/sql/language-manual/sql-ref-syntax-qry-select-unpivot.html#syntax
            if self._match_text_seq("INCLUDE", "NULLS"):
                include_nulls = True
            elif self._match_text_seq("EXCLUDE", "NULLS"):
                include_nulls = False
        else:
            return None

        expressions = []

        # 需要紧随左括号，否则回退至起始点视为非 PIVOT 语法
        if not self._match(TokenType.L_PAREN):
            self._retreat(index)
            return None

        # UNPIVOT: 收集被反透视的列；PIVOT: 收集聚合（可带别名）
        if unpivot:
            expressions = self._parse_csv(self._parse_column)
        else:
            expressions = self._parse_csv(self._parse_pivot_aggregation)

        if not expressions:
            self.raise_error("Failed to parse PIVOT's aggregation list")

        if not self._match(TokenType.FOR):
            self.raise_error("Expecting FOR")

        # 支持多个 FOR ... IN(...) 片段，逐个尝试解析
        fields = []
        while True:
            field = self._try_parse(self._parse_pivot_in)
            if not field:
                break
            fields.append(field)

        # DEFAULT ON NULL (<expr>)：为 NULL 透视值提供默认值
        default_on_null = self._match_text_seq("DEFAULT", "ON", "NULL") and self._parse_wrapped(
            self._parse_bitwise
        )

        group = self._parse_group()

        self._match_r_paren()

        pivot = self.expression(
            exp.Pivot,
            expressions=expressions,
            fields=fields,
            unpivot=unpivot,
            include_nulls=include_nulls,
            default_on_null=default_on_null,
            group=group,
        )

        # 若后续不再连续出现 PIVOT/UNPIVOT，则此处允许设置别名
        if not self._match_set((TokenType.PIVOT, TokenType.UNPIVOT), advance=False):
            pivot.set("alias", self._parse_table_alias())

        # PIVOT 情况下，基于 fields 的每个取值与聚合名，生成列名的笛卡尔积
        if not unpivot:
            names = self._pivot_column_names(t.cast(t.List[exp.Expression], expressions))

            columns: t.List[exp.Expression] = []
            all_fields = []
            for pivot_field in pivot.fields:
                pivot_field_expressions = pivot_field.expressions

                # PivotAny 对应 ANY ORDER BY <column>，无法推断具体枚举，跳过
                if isinstance(seq_get(pivot_field_expressions, 0), exp.PivotAny):
                    continue

                all_fields.append(
                    [
                        # IDENTIFY_PIVOT_STRINGS 为 True 时保留原始 SQL 字符串，否则用 alias_or_name
                        fld.sql() if self.IDENTIFY_PIVOT_STRINGS else fld.alias_or_name
                        for fld in pivot_field_expressions
                    ]
                )

            if all_fields:
                if names:
                    all_fields.append(names)

                # 生成所有可能的列名组合，如 [[2000, 2010], ['NL', 'US'], ['total']]
                for fld_parts_tuple in itertools.product(*all_fields):
                    fld_parts = list(fld_parts_tuple)

                    if names and self.PREFIXED_PIVOT_COLUMNS:
                        # 当配置前缀列名时，将“聚合名”移动到最前
                        fld_parts.insert(0, fld_parts.pop(-1))

                    columns.append(exp.to_identifier("_".join(fld_parts)))

            pivot.set("columns", columns)

        return pivot

    def _pivot_column_names(self, aggregations: t.List[exp.Expression]) -> t.List[str]:
        # 提取聚合表达式的别名，用于参与生成透视后的列名
        return [agg.alias for agg in aggregations if agg.alias]

    def _parse_prewhere(self, skip_where_token: bool = False) -> t.Optional[exp.PreWhere]:
        # ClickHouse 方言特有的 PREWHERE：在 WHERE 前进行更早的过滤，优化 I/O
        if not skip_where_token and not self._match(TokenType.PREWHERE):
            return None

        return self.expression(
            exp.PreWhere, comments=self._prev_comments, this=self._parse_assignment()
        )

    def _parse_where(self, skip_where_token: bool = False) -> t.Optional[exp.Where]:
        # 常规 WHERE 子句，承载过滤条件；保留前置注释（可能包含 Hints）
        if not skip_where_token and not self._match(TokenType.WHERE):
            return None

        return self.expression(
            exp.Where, comments=self._prev_comments, this=self._parse_assignment()
        )

    def _parse_group(self, skip_group_by_token: bool = False) -> t.Optional[exp.Group]:
        # 解析 GROUP BY：支持 ALL/DISTINCT、ROLLUP、CUBE、GROUPING SETS、TOTALS（ClickHouse）
        if not skip_group_by_token and not self._match(TokenType.GROUP_BY):
            return None
        comments = self._prev_comments

        elements: t.Dict[str, t.Any] = defaultdict(list)

        # 规范化 ALL/DISTINCT 标记，供生成阶段决定是否保留/转译
        if self._match(TokenType.ALL):
            elements["all"] = True
        elif self._match(TokenType.DISTINCT):
            elements["all"] = False

        # 若后续直接出现查询修饰符（如 HAVING/ORDER 等），返回仅包含 flags 的 Group
        if self._match_set(self.QUERY_MODIFIER_TOKENS, advance=False):
            return self.expression(exp.Group, comments=comments, **elements)  # type: ignore

        # 循环累加 group 表达式与扩展子句
        while True:
            index = self._index

            # 先尽可能解析逗号分隔的 grouping 表达式
            elements["expressions"].extend(
                self._parse_csv(
                    lambda: None
                    if self._match_set((TokenType.CUBE, TokenType.ROLLUP), advance=False)
                    else self._parse_assignment()
                )
            )

            # WITH ROLLUP/CUBE/GROUPING SETS/TOTALS 扩展
            before_with_index = self._index
            with_prefix = self._match(TokenType.WITH)

            if cube_or_rollup := self._parse_cube_or_rollup(with_prefix=with_prefix):
                key = "rollup" if isinstance(cube_or_rollup, exp.Rollup) else "cube"
                elements[key].append(cube_or_rollup)
            elif grouping_sets := self._parse_grouping_sets():
                elements["grouping_sets"].append(grouping_sets)
            elif self._match_text_seq("TOTALS"):
                elements["totals"] = True  # type: ignore

            # 若 WITH 后未跟任何有效子句，回退避免误消费
            if before_with_index <= self._index <= before_with_index + 1:
                self._retreat(before_with_index)
                break

            # 防御：無前进则退出，避免死循环
            if index == self._index:
                break

        return self.expression(exp.Group, comments=comments, **elements)  # type: ignore

    def _parse_cube_or_rollup(self, with_prefix: bool = False) -> t.Optional[exp.Cube | exp.Rollup]:
        """解析 ROLLUP/CUBE：当带 WITH 前缀时，允许空表达式列表，否则必须为括号包裹的列列表"""
        if self._match(TokenType.CUBE):
            kind: t.Type[exp.Cube | exp.Rollup] = exp.Cube
        elif self._match(TokenType.ROLLUP):
            kind = exp.Rollup
        else:
            return None

        return self.expression(
            kind, expressions=[] if with_prefix else self._parse_wrapped_csv(self._parse_bitwise)
        )

    def _parse_grouping_sets(self) -> t.Optional[exp.GroupingSets]:
        """解析 GROUPING SETS 内部单元：
           - (a, b) 作为一个 Tuple 分组
           - 单列时直接解析为列表达式"""
        if self._match(TokenType.GROUPING_SETS):
            return self.expression(
                exp.GroupingSets, expressions=self._parse_wrapped_csv(self._parse_grouping_set)
            )
        return None

    def _parse_grouping_set(self) -> t.Optional[exp.Expression]:
        return self._parse_grouping_sets() or self._parse_cube_or_rollup() or self._parse_bitwise()

    def _parse_having(self, skip_having_token: bool = False) -> t.Optional[exp.Having]:
        # 解析 HAVING：基于分组结果的过滤条件，保留前置注释
        if not skip_having_token and not self._match(TokenType.HAVING):
            return None
        return self.expression(
            exp.Having, comments=self._prev_comments, this=self._parse_assignment()
        )

    def _parse_qualify(self) -> t.Optional[exp.Qualify]:
        # 解析 QUALIFY：对窗口函数结果进行过滤（如 BigQuery/Snowflake）
        if not self._match(TokenType.QUALIFY):
            return None
        return self.expression(exp.Qualify, this=self._parse_assignment())

    def _parse_connect_with_prior(self) -> t.Optional[exp.Expression]:
        # 解析 Oracle 层次查询中的 PRIOR：在 CONNECT BY 中表示父子关系
        # 通过暂时注册一个“无括号函数解析器”来处理 PRIOR expr 语法
        self.NO_PAREN_FUNCTION_PARSERS["PRIOR"] = lambda self: self.expression(
            exp.Prior, this=self._parse_bitwise()
        )
        connect = self._parse_assignment()
        self.NO_PAREN_FUNCTION_PARSERS.pop("PRIOR")
        return connect

    def _parse_connect(self, skip_start_token: bool = False) -> t.Optional[exp.Connect]:
        # 解析 Oracle 层次查询：START WITH ... CONNECT BY [NOCYCLE] PRIOR ...
        # 设计原因：START WITH 可前/后置（少见），因此需要两次条件检查
        if skip_start_token:
            start = None
        elif self._match(TokenType.START_WITH):
            start = self._parse_assignment()
        else:
            return None

        self._match(TokenType.CONNECT_BY)
        nocycle = self._match_text_seq("NOCYCLE")
        connect = self._parse_connect_with_prior()

        # 某些方言允许 START WITH 出现在 CONNECT BY 之后
        if not start and self._match(TokenType.START_WITH):
            start = self._parse_assignment()

        return self.expression(exp.Connect, start=start, connect=connect, nocycle=nocycle)

    def _parse_name_as_expression(self) -> t.Optional[exp.Expression]:
        # 解析 name AS expr：当存在别名关键字时将其包装为 Alias
        this = self._parse_id_var(any_token=True)
        if self._match(TokenType.ALIAS):
            this = self.expression(exp.Alias, alias=this, this=self._parse_assignment())
        return this

    def _parse_interpolate(self) -> t.Optional[t.List[exp.Expression]]:
        # ClickHouse 扩展：WITH FILL INTERPOLATE 支持线性插值/填充
        if self._match_text_seq("INTERPOLATE"):
            return self._parse_wrapped_csv(self._parse_name_as_expression)
        return None

    def _parse_order(
        self, this: t.Optional[exp.Expression] = None, skip_order_token: bool = False
    ) -> t.Optional[exp.Expression]:
        # 解析 ORDER BY / ORDER SIBLINGS BY（Oracle 层次查询同级排序）
        siblings = None
        if not skip_order_token and not self._match(TokenType.ORDER_BY):
            if not self._match(TokenType.ORDER_SIBLINGS_BY):
                return this

            siblings = True

        return self.expression(
            exp.Order,
            comments=self._prev_comments,
            this=this,
            expressions=self._parse_csv(self._parse_ordered),
            siblings=siblings,
        )

    def _parse_sort(self, exp_class: t.Type[E], token: TokenType) -> t.Optional[E]:
        # 通用排序解析器：用于 SORT BY / DISTRIBUTE BY 等方言扩展
        if not self._match(token):
            return None
        return self.expression(exp_class, expressions=self._parse_csv(self._parse_ordered))

    def _parse_ordered(
        self, parse_method: t.Optional[t.Callable] = None
    ) -> t.Optional[exp.Ordered]:
        # 解析单个排序元素：表达式 + 升降序 + NULLS FIRST/LAST + WITH FILL
        this = parse_method() if parse_method else self._parse_assignment()
        if not this:
            return None

        # 支持 ORDER BY ALL（方言可选），将其规范化为变量 ALL
        if this.name.upper() == "ALL" and self.dialect.SUPPORTS_ORDER_BY_ALL:
            this = exp.var("ALL")

        asc = self._match(TokenType.ASC)
        desc = self._match(TokenType.DESC) or (asc and False)

        # 显式 NULLS 排序优先级
        is_nulls_first = self._match_text_seq("NULLS", "FIRST")
        is_nulls_last = self._match_text_seq("NULLS", "LAST")

        nulls_first = is_nulls_first or False
        explicitly_null_ordered = is_nulls_first or is_nulls_last

        # 若未显式指定 NULLS FIRST/LAST，则根据方言默认行为推导
        if (
            not explicitly_null_ordered
            and (
                (not desc and self.dialect.NULL_ORDERING == "nulls_are_small")
                or (desc and self.dialect.NULL_ORDERING != "nulls_are_small")
            )
            and self.dialect.NULL_ORDERING != "nulls_are_last"
        ):
            nulls_first = True

        # ClickHouse: WITH FILL [FROM ...] [TO ...] [STEP ...] [INTERPOLATE (...)]
        if self._match_text_seq("WITH", "FILL"):
            with_fill = self.expression(
                exp.WithFill,
                **{  # type: ignore
                    "from": self._match(TokenType.FROM) and self._parse_bitwise(),
                    "to": self._match_text_seq("TO") and self._parse_bitwise(),
                    "step": self._match_text_seq("STEP") and self._parse_bitwise(),
                    "interpolate": self._parse_interpolate(),
                },
            )
        else:
            with_fill = None

        return self.expression(
            exp.Ordered, this=this, desc=desc, nulls_first=nulls_first, with_fill=with_fill
        )

    def _parse_limit_options(self) -> t.Optional[exp.LimitOptions]:
        """解析 LIMIT/TOP 的附加选项：PERCENT、ROW(S)、ONLY、WITH TIES 等"""
        percent = self._match_set((TokenType.PERCENT, TokenType.MOD))
        rows = self._match_set((TokenType.ROW, TokenType.ROWS))
        self._match_text_seq("ONLY")
        with_ties = self._match_text_seq("WITH", "TIES")

        if not (percent or rows or with_ties):
            return None

        return self.expression(exp.LimitOptions, percent=percent, rows=rows, with_ties=with_ties)

    def _parse_limit(
        self,
        this: t.Optional[exp.Expression] = None,
        top: bool = False,
        skip_limit_token: bool = False,
    ) -> t.Optional[exp.Expression]:
        # 解析 LIMIT/TOP：支持 T-SQL 的 TOP (...) 与标准 LIMIT 语法
        if skip_limit_token or self._match(TokenType.TOP if top else TokenType.LIMIT):
            comments = self._prev_comments
            if top:
                # TOP (expr) 或 TOP expr：括号存在与否影响解析方式
                limit_paren = self._match(TokenType.L_PAREN)
                expression = self._parse_term() if limit_paren else self._parse_number()

                if limit_paren:
                    self._match_r_paren()
            else:
                # Parsing LIMIT x% (i.e x PERCENT) as a term leads to an error, since
                # we try to build an exp.Mod expr. For that matter, we backtrack and instead
                # consume the factor plus parse the percentage separately
                index = self._index
                expression = self._try_parse(self._parse_term)
                if isinstance(expression, exp.Mod):
                    self._retreat(index)
                    expression = self._parse_factor()
                elif not expression:
                    expression = self._parse_factor()
            limit_options = self._parse_limit_options()

            # LIMIT offset, count 的逗号语法（MySQL 等）
            if self._match(TokenType.COMMA):
                offset = expression
                expression = self._parse_term()
            else:
                offset = None

            limit_exp = self.expression(
                exp.Limit,
                this=this,
                expression=expression,
                offset=offset,
                comments=comments,
                limit_options=limit_options,
                expressions=self._parse_limit_by(),
            )

            return limit_exp

        # FETCH FIRST/NEXT n ROW(S) ... 的等价语法
        if self._match(TokenType.FETCH):
            direction = self._match_set((TokenType.FIRST, TokenType.NEXT))
            direction = self._prev.text.upper() if direction else "FIRST"

            count = self._parse_field(tokens=self.FETCH_TOKENS)

            return self.expression(
                exp.Fetch,
                direction=direction,
                count=count,
                limit_options=self._parse_limit_options(),
            )

        return this

    def _parse_offset(self, this: t.Optional[exp.Expression] = None) -> t.Optional[exp.Expression]:
        # 解析 OFFSET：支持 OFFSET <n> ROW(S) [BY ...]
        if not self._match(TokenType.OFFSET):
            return this

        count = self._parse_term()
        # 兼容存在/不存在 ROW(S) 的写法
        self._match_set((TokenType.ROW, TokenType.ROWS))

        return self.expression(
            exp.Offset, this=this, expression=count, expressions=self._parse_limit_by()
        )

    def _can_parse_limit_or_offset(self) -> bool:
        # 在歧义上下文（如别名/关键字相邻）下，探测后续是否是 LIMIT/OFFSET
        # 设计原因：避免把 LIMIT/OFFSET 误解析为别名或其他标识符
        if not self._match_set(self.AMBIGUOUS_ALIAS_TOKENS, advance=False):
            return False

        index = self._index
        result = bool(
            self._try_parse(self._parse_limit, retreat=True)
            or self._try_parse(self._parse_offset, retreat=True)
        )
        self._retreat(index)
        return result

    def _parse_limit_by(self) -> t.Optional[t.List[exp.Expression]]:
        # ClickHouse 等方言：LIMIT ... BY <exprs>
        return self._match_text_seq("BY") and self._parse_csv(self._parse_bitwise)

    def _parse_locks(self) -> t.List[exp.Lock]:
        # 解析行级锁：FOR UPDATE/SHARE、FOR KEY SHARE、FOR NO KEY UPDATE 等（Postgres 等）
        # 设计原因：锁语义可能附带 OF <tables> 与等待策略（NOWAIT/WAIT n/SKIP LOCKED）
        locks = []
        while True:
            update, key = None, None
            if self._match_text_seq("FOR", "UPDATE"):
                update = True
            elif self._match_text_seq("FOR", "SHARE") or self._match_text_seq(
                "LOCK", "IN", "SHARE", "MODE"
            ):
                update = False
            elif self._match_text_seq("FOR", "KEY", "SHARE"):
                update, key = False, True
            elif self._match_text_seq("FOR", "NO", "KEY", "UPDATE"):
                update, key = True, True
            else:
                break

            expressions = None
            if self._match_text_seq("OF"):
                # 指定锁目标表/别名列表
                expressions = self._parse_csv(lambda: self._parse_table(schema=True))

            # 等待策略：NOWAIT（立即失败）、WAIT <n>、SKIP LOCKED（跳过被锁行）
            wait: t.Optional[bool | exp.Expression] = None
            if self._match_text_seq("NOWAIT"):
                wait = True
            elif self._match_text_seq("WAIT"):
                wait = self._parse_primary()
            elif self._match_text_seq("SKIP", "LOCKED"):
                wait = False

            locks.append(
                self.expression(
                    exp.Lock, update=update, expressions=expressions, wait=wait, key=key
                )
            )

        return locks

    def parse_set_operation(
        self, this: t.Optional[exp.Expression], consume_pipe: bool = False
    ) -> t.Optional[exp.Expression]:
        # 解析集合操作：UNION/INTERSECT/EXCEPT 及其 DISTINCT/ALL、BY NAME/STRICT CORRESPONDING 等变体
        start = self._index
        # 复用 join 的 side/kind 解析，用于 UNION [LEFT|RIGHT|FULL] [OUTER] 等非标准扩展
        _, side_token, kind_token = self._parse_join_parts()

        side = side_token.text if side_token else None
        kind = kind_token.text if kind_token else None

        if not self._match_set(self.SET_OPERATIONS):
            self._retreat(start)
            return None

        token_type = self._prev.token_type

        if token_type == TokenType.UNION:
            operation: t.Type[exp.SetOperation] = exp.Union
        elif token_type == TokenType.EXCEPT:
            operation = exp.Except
        else:
            operation = exp.Intersect

        comments = self._prev.comments

        # DISTINCT/ALL：若方言默认值为 None，则必须显式指定，否则抛错
        if self._match(TokenType.DISTINCT):
            distinct: t.Optional[bool] = True
        elif self._match(TokenType.ALL):
            distinct = False
        else:
            distinct = self.dialect.SET_OP_DISTINCT_BY_DEFAULT[operation]
            if distinct is None:
                self.raise_error(f"Expected DISTINCT or ALL for {operation.__name__}")

        # 列对齐方式：BY NAME | STRICT CORRESPONDING / CORRESPONDING [BY (...)]
        by_name = self._match_text_seq("BY", "NAME") or self._match_text_seq(
            "STRICT", "CORRESPONDING"
        )
        if self._match_text_seq("CORRESPONDING"):
            by_name = True
            # 若未显式 side/kind，默认 INNER
            if not side and not kind:
                kind = "INNER"

        on_column_list = None
        if by_name and self._match_texts(("ON", "BY")):
            on_column_list = self._parse_wrapped_csv(self._parse_column)

        # 集合操作右侧为子查询；禁用继续解析集合操作以避免递归歧义
        expression = self._parse_select(
            nested=True, parse_set_operation=False, consume_pipe=consume_pipe
        )

        return self.expression(
            operation,
            comments=comments,
            this=this,
            distinct=distinct,
            by_name=by_name,
            expression=expression,
            side=side,
            kind=kind,
            on=on_column_list,
        )

    def _parse_set_operations(self, this: t.Optional[exp.Expression]) -> t.Optional[exp.Expression]:
        # 连续解析集合操作链（如 a UNION b INTERSECT c），直到不再匹配
        while this:
            setop = self.parse_set_operation(this)
            if not setop:
                break
            this = setop

        # 某些方言将修饰符（ORDER/LIMIT 等）挂在集合操作内部的最后一个 SELECT 上，这里向上提升到集合节点
        if isinstance(this, exp.SetOperation) and self.MODIFIERS_ATTACHED_TO_SET_OP:
            expression = this.expression

            if expression:
                for arg in self.SET_OP_MODIFIERS:
                    expr = expression.args.get(arg)
                    if expr:
                        this.set(arg, expr.pop())

        return this

    def _parse_expression(self) -> t.Optional[exp.Expression]:
        # 表达式入口：先解析赋值/二元表达式，再尝试别名封装
        return self._parse_alias(self._parse_assignment())

    def _parse_assignment(self) -> t.Optional[exp.Expression]:
        # 解析赋值/绑定：支持 :=、= 等运算符链，且左侧可为非常规标记
        this = self._parse_disjunction()
        if not this and self._next and self._next.token_type in self.ASSIGNMENT:
            # 支持形如 <非标识 token> := <expr> 的写法：先推进一个 token 作为列名占位
            this = exp.column(
                t.cast(str, self._advance_any(ignore_reserved=True) and self._prev.text)
            )

        while self._match_set(self.ASSIGNMENT):
            # 若左侧是单段列标识（无 schema/table 前缀），提升为其标识符本体
            if isinstance(this, exp.Column) and len(this.parts) == 1:
                this = this.this

            # 构造对应赋值表达式节点（如 NamedArgument/Assignment 等），右侧递归解析
            this = self.expression(
                self.ASSIGNMENT[self._prev.token_type],
                this=this,
                comments=self._prev_comments,
                expression=self._parse_assignment(),
            )

        return this

    def _parse_disjunction(self) -> t.Optional[exp.Expression]:
        # 解析“或”逻辑（OR）：使用通用的 _parse_tokens，将子解析器设为“与”
        return self._parse_tokens(self._parse_conjunction, self.DISJUNCTION)

    def _parse_conjunction(self) -> t.Optional[exp.Expression]:
        # 解析“与”逻辑（AND）：子解析器为“相等/不等”层级
        return self._parse_tokens(self._parse_equality, self.CONJUNCTION)

    def _parse_equality(self) -> t.Optional[exp.Expression]:
        # 解析相等与等价关系（=、<>, !=、<=> 等）：子解析器为比较层级
        return self._parse_tokens(self._parse_comparison, self.EQUALITY)

    def _parse_comparison(self) -> t.Optional[exp.Expression]:
        # 解析比较（<、<=、>、>=、LIKE/ILIKE 等）：子解析器为区间层级
        return self._parse_tokens(self._parse_range, self.COMPARISON)

    def _parse_range(self, this: t.Optional[exp.Expression] = None) -> t.Optional[exp.Expression]:
        # 解析区间类谓词：BETWEEN/IN/LIKE/REGEXP 等，以及 ISNULL/NOTNULL
        this = this or self._parse_bitwise()
        negate = self._match(TokenType.NOT)

        # 优先尝试区间类专用解析器（由当前 token 决定具体函数）
        if self._match_set(self.RANGE_PARSERS):
            expression = self.RANGE_PARSERS[self._prev.token_type](self, this)
            if not expression:
                return this

            this = expression
        elif self._match(TokenType.ISNULL):
            # ISNULL 等价于 IS NULL
            this = self.expression(exp.Is, this=this, expression=exp.Null())

        # Postgres: 兼容 NOTNULL（相当于 IS NOT NULL）
        # https://blog.andreiavram.ro/postgresql-null-composite-type/
        if self._match(TokenType.NOTNULL):
            this = self.expression(exp.Is, this=this, expression=exp.Null())
            this = self.expression(exp.Not, this=this)

        # 若前缀带 NOT，则对区间谓词整体取反（如 NOT BETWEEN/NOT IN 等）
        if negate:
            this = self._negate_range(this)

        # 支持 IS/IS NOT <expr|JSON ...> 谓词
        if self._match(TokenType.IS):
            this = self._parse_is(this)

        return this

    def _negate_range(self, this: t.Optional[exp.Expression] = None) -> t.Optional[exp.Expression]:
        # 将已构造的区间谓词整体取反：包装为 NOT(<expr>)
        if not this:
            return this

        return self.expression(exp.Not, this=this)

    def _parse_is(self, this: t.Optional[exp.Expression]) -> t.Optional[exp.Expression]:
        # 解析 IS/IS NOT 谓词：
        # - IS [NOT] DISTINCT FROM：空安全比较（与 =/<> 不同，NULL 与 NULL 也可比较）
        # - IS [NOT] JSON [VALUE|SCALAR|ARRAY|OBJECT] [WITH|WITHOUT] UNIQUE KEYS
        # - IS [NOT] <primary|null>
        index = self._index - 1
        negate = self._match(TokenType.NOT)

        if self._match_text_seq("DISTINCT", "FROM"):
            klass = exp.NullSafeEQ if negate else exp.NullSafeNEQ
            return self.expression(klass, this=this, expression=self._parse_bitwise())

        if self._match(TokenType.JSON):
            kind = self._match_texts(self.IS_JSON_PREDICATE_KIND) and self._prev.text.upper()

            if self._match_text_seq("WITH"):
                _with = True
            elif self._match_text_seq("WITHOUT"):
                _with = False
            else:
                _with = None

            unique = self._match(TokenType.UNIQUE)
            self._match_text_seq("KEYS")
            expression: t.Optional[exp.Expression] = self.expression(
                exp.JSON, **{"this": kind, "with": _with, "unique": unique}
            )
        else:
            expression = self._parse_primary() or self._parse_null()
            if not expression:
                self._retreat(index)
                return None

        this = self.expression(exp.Is, this=this, expression=expression)
        this = self.expression(exp.Not, this=this) if negate else this
        return self._parse_column_ops(this)

    def _parse_in(self, this: t.Optional[exp.Expression], alias: bool = False) -> exp.In:
        # 解析 IN 谓词：支持 IN (子查询|列表) / IN [数组] / IN UNNEST(...)
        unnest = self._parse_unnest(with_alias=False)
        if unnest:
            this = self.expression(exp.In, this=this, unnest=unnest)
        elif self._match_set((TokenType.L_PAREN, TokenType.L_BRACKET)):
            matched_l_paren = self._prev.token_type == TokenType.L_PAREN
            expressions = self._parse_csv(lambda: self._parse_select_or_expression(alias=alias))

            if len(expressions) == 1 and isinstance(query := expressions[0], exp.Query):
                this = self.expression(
                    exp.In,
                    this=this,
                    query=self._parse_query_modifiers(query).subquery(copy=False),
                )
            else:
                this = self.expression(exp.In, this=this, expressions=expressions)

            # 根据开括号类型匹配闭合：) 或 ]
            if matched_l_paren:
                self._match_r_paren(this)
            elif not self._match(TokenType.R_BRACKET, expression=this):
                self.raise_error("Expecting ]")
        else:
            # 支持 IN <column> 的方言扩展
            this = self.expression(exp.In, this=this, field=self._parse_column())

        return this

    def _parse_between(self, this: t.Optional[exp.Expression]) -> exp.Between:
        # 解析 BETWEEN [SYMMETRIC|ASYMMETRIC] low AND high
        # SYMMETRIC: BETWEEN a AND b 等价于 (>= LEAST(a,b) AND <= GREATEST(a,b))
        symmetric = None
        if self._match_text_seq("SYMMETRIC"):
            symmetric = True
        elif self._match_text_seq("ASYMMETRIC"):
            symmetric = False

        low = self._parse_bitwise()
        self._match(TokenType.AND)
        high = self._parse_bitwise()

        return self.expression(
            exp.Between,
            this=this,
            low=low,
            high=high,
            symmetric=symmetric,
        )

    def _parse_escape(self, this: t.Optional[exp.Expression]) -> t.Optional[exp.Expression]:
        # 解析 LIKE ... ESCAPE 'x'：自定义转义字符
        if not self._match(TokenType.ESCAPE):
            return this
        return self.expression(
            exp.Escape, this=this, expression=self._parse_string() or self._parse_null()
        )

    def _parse_interval(self, match_interval: bool = True) -> t.Optional[exp.Add | exp.Interval]:
        # 解析 INTERVAL：将多方言写法归一为 INTERVAL 'value' unit 的规范形态，便于转译
        index = self._index

        # 当 match_interval=True 时，必须匹配到 INTERVAL 关键字；否则允许作为“+ 后续拼接”的递归分支
        if not self._match(TokenType.INTERVAL) and match_interval:
            return None

        # INTERVAL 后若紧跟字符串，优先按 primary 读取，否则读取一个 term（可能是数字/表达式）
        if self._match(TokenType.STRING, advance=False):
            this = self._parse_primary()
        else:
            this = self._parse_term()

        # 兜底：若取不到值，或被误识别为列名 IS（如把 IS 当作标识符），回退并放弃解析
        if not this or (
            isinstance(this, exp.Column)
            and not this.table
            and not this.this.quoted
            and self._curr
            and self._curr.text.upper() not in self.dialect.VALID_INTERVAL_UNITS
        ):
            self._retreat(index)
            return None

        # 单位解析：优先尝试函数（如 HOUR()），否则在未出现别名的情况下，读取一个变量作为单位（如 DAY）
        # handle day-time format interval span with omitted units:
        #   INTERVAL '<number days> hh[:][mm[:ss[.ff]]]' <maybe `unit TO unit`>
        interval_span_units_omitted = None
        if (
            this
            and this.is_string
            and self.SUPPORTS_OMITTED_INTERVAL_SPAN_UNIT
            and exp.INTERVAL_DAY_TIME_RE.match(this.name)
        ):
            index = self._index

            # Var "TO" Var
            first_unit = self._parse_var(any_token=True, upper=True)
            second_unit = None
            if first_unit and self._match_text_seq("TO"):
                second_unit = self._parse_var(any_token=True, upper=True)

            interval_span_units_omitted = not (first_unit and second_unit)

            self._retreat(index)

        unit = (
            None
            if interval_span_units_omitted
            else (
                self._parse_function()
                or (
                    not self._match(TokenType.ALIAS, advance=False)
                    and self._parse_var(any_token=True, upper=True)
                )
            )
        )

        # 规范化：将 INTERVAL 5 DAY 归一为 INTERVAL '5' DAY；便于跨方言生成
        # 同时支持 '5 day' 这种把单位写入字符串的形式
        if this and this.is_number:
            this = exp.Literal.string(this.to_py())
        elif this and this.is_string:
            parts = exp.INTERVAL_STRING_RE.findall(this.name)
            if parts and unit:
                # 若单位实际存在于字符串中，则撤销前面“贪婪解析”的单位读取
                unit = None
                self._retreat(self._index - 1)

            if len(parts) == 1:
                # 拆分 '5 day' → '5' 与 DAY 两部分
                this = exp.Literal.string(parts[0][0])
                unit = self.expression(exp.Var, this=parts[0][1].upper())

        # 支持复合跨度：INTERVAL '1' YEAR TO MONTH 等
        if self.INTERVAL_SPANS and self._match_text_seq("TO"):
            unit = self.expression(
                exp.IntervalSpan, this=unit, expression=self._parse_var(any_token=True, upper=True)
            )

        # 构造 Interval 节点
        interval = self.expression(exp.Interval, this=this, unit=unit)

        # 支持形如 INTERVAL '1' DAY + '2' HOUR 的相加表达
        index = self._index
        self._match(TokenType.PLUS)

        # Convert INTERVAL 'val_1' unit_1 [+] ... [+] 'val_n' unit_n into a sum of intervals
        if self._match_set((TokenType.STRING, TokenType.NUMBER), advance=False):
            return self.expression(
                exp.Add, this=interval, expression=self._parse_interval(match_interval=False)
            )

        # 若没有继续相加，则回退到 PLUS 前的位置
        self._retreat(index)
        return interval

    def _parse_bitwise(self) -> t.Optional[exp.Expression]:
        # 解析位运算/扩展运算：| & ^ ~ << >>，以及可选的字符串拼接 || 与空合并 ??
        this = self._parse_term()

        while True:
            if self._match_set(self.BITWISE):
                # 基于当前匹配到的位运算符，构造对应的二元表达式节点
                this = self.expression(
                    self.BITWISE[self._prev.token_type],
                    this=this,
                    expression=self._parse_term(),
                )
            elif self.dialect.DPIPE_IS_STRING_CONCAT and self._match(TokenType.DPIPE):
                # 一些方言使用 || 作为字符串拼接，safe 取决于是否要求严格拼接（空值行为）
                this = self.expression(
                    exp.DPipe,
                    this=this,
                    expression=self._parse_term(),
                    safe=not self.dialect.STRICT_STRING_CONCAT,
                )
            elif self._match(TokenType.DQMARK):
                # ?? 作为空合并运算符：x ?? y 等价于 COALESCE(x, y)
                this = self.expression(
                    exp.Coalesce, this=this, expressions=ensure_list(self._parse_term())
                )
            elif self._match_pair(TokenType.LT, TokenType.LT):
                this = self.expression(
                    exp.BitwiseLeftShift, this=this, expression=self._parse_term()
                )
            elif self._match_pair(TokenType.GT, TokenType.GT):
                this = self.expression(
                    exp.BitwiseRightShift, this=this, expression=self._parse_term()
                )
            else:
                break

        return this

    def _parse_term(self) -> t.Optional[exp.Expression]:
        # 解析“项”级二元运算：加/减/字符串拼接等（具体集合见 self.TERM）
        this = self._parse_factor()

        while self._match_set(self.TERM):
            klass = self.TERM[self._prev.token_type]
            comments = self._prev_comments
            expression = self._parse_factor()

            # 构造对应的二元表达式（如 Add/Sub/Concat），并保留运算符前的注释
            this = self.expression(klass, this=this, comments=comments, expression=expression)

            if isinstance(this, exp.Collate):
                expr = this.expression

                # 保留如 pg_catalog."default"（Postgres）的排序规则为列，否则回退为 Identifier/Var
                if isinstance(expr, exp.Column) and len(expr.parts) == 1:
                    ident = expr.this
                    if isinstance(ident, exp.Identifier):
                        this.set("expression", ident if ident.quoted else exp.var(ident.name))

        return this

    def _parse_factor(self) -> t.Optional[exp.Expression]:
        # 解析“因子”级二元运算：乘/除/取模/整数除等（具体集合见 self.FACTOR）
        # 说明：根据是否支持指数运算，选择“指数优先”或“直接一元”作为子解析器
        parse_method = self._parse_exponent if self.EXPONENT else self._parse_unary
        this = parse_method()

        while self._match_set(self.FACTOR):
            klass = self.FACTOR[self._prev.token_type]
            comments = self._prev_comments
            expression = parse_method()

            # 处理形如 `DIV` 作为标识符而非运算符的歧义：
            # 若紧随的是字母（可能是标识符开头），则回退并把 `DIV` 视为标识符
            if not expression and klass is exp.IntDiv and self._prev.text.isalpha():
                self._retreat(self._index - 1)
                return this

            # 构造乘/除/模/整除等表达式节点
            this = self.expression(klass, this=this, comments=comments, expression=expression)

            # 对除法附加方言策略：typed（类型化除法）/safe（空值安全）
            if isinstance(this, exp.Div):
                this.set("typed", self.dialect.TYPED_DIVISION)
                this.set("safe", self.dialect.SAFE_DIVISION)

        return this

    def _parse_exponent(self) -> t.Optional[exp.Expression]:
        # 解析指数运算（如 ^ 或 **，视方言配置）：子解析器是一元层级
        return self._parse_tokens(self._parse_unary, self.EXPONENT)

    def _parse_unary(self) -> t.Optional[exp.Expression]:
        # 解析一元运算：正负号、逻辑非、位非等，优先使用已注册的一元解析器
        if self._match_set(self.UNARY_PARSERS):
            return self.UNARY_PARSERS[self._prev.token_type](self)
        # 否则继续解析类型/时区修饰等后续层级，保持表达式链路连贯
        return self._parse_at_time_zone(self._parse_type())

    def _parse_type(
        self, parse_interval: bool = True, fallback_to_identifier: bool = False
    ) -> t.Optional[exp.Expression]:
        # 解析类型或类型构造器：优先将 INTERVAL 作为类型相关表达式处理
        interval = parse_interval and self._parse_interval()
        if interval:
            return interval

        index = self._index
        # 尝试解析数据类型（禁用把标识符当作类型，以免与列名冲突）；允许函数类型（如 GEOGRAPHY())
        data_type = self._parse_types(check_func=True, allow_identifiers=False)

        # parse_types() returns a Cast if we parsed BQ's inline constructor <type>(<values>) e.g.
        # STRUCT<a INT, b STRING>(1, 'foo'), which is canonicalized to CAST(<values> AS <type>)
        # 若解析到 BigQuery 的内联构造器（<type>(<values>）），已被规范化为 CAST(<values> AS <type>)
        if isinstance(data_type, exp.Cast):
            # This constructor can contain ops directly after it, for instance struct unnesting:
            # STRUCT<a INT, b STRING>(1, 'foo').* --> CAST(STRUCT(1, 'foo') AS STRUCT<a iNT, b STRING).*
            # 构造器后可能紧跟列操作（如 . * 进行展开），因此继续走列操作链
            return self._parse_column_ops(data_type)

        if data_type:
            index2 = self._index
            this = self._parse_primary()

            if isinstance(this, exp.Literal):
                literal = this.name
                this = self._parse_column_ops(this)

                # 某些类型字面量需要方言专用解析（如 ARRAY[...], STRUCT(...) 等）
                parser = self.TYPE_LITERAL_PARSERS.get(data_type.this)
                if parser:
                    return parser(self, this, data_type)

                # 带时区的时间字面量：若构造器中含时区，统一提升为 TIMESTAMPTZ
                if (
                    self.ZONE_AWARE_TIMESTAMP_CONSTRUCTOR
                    and data_type.is_type(exp.DataType.Type.TIMESTAMP)
                    and TIME_ZONE_RE.search(literal)
                ):
                    data_type = exp.DataType.build("TIMESTAMPTZ")

                # 常规：字面量 CAST 到解析出的 data_type
                return self.expression(exp.Cast, this=this, to=data_type)


            # The expressions arg gets set by the parser when we have something like DECIMAL(38, 0)
            # in the input SQL. In that case, we'll produce these tokens: DECIMAL ( 38 , 0 )
            #
            # If the index difference here is greater than 1, that means the parser itself must have
            # consumed additional tokens such as the DECIMAL scale and precision in the above example.
            #
            # If it's not greater than 1, then it must be 1, because we've consumed at least the type
            # keyword, meaning that the expressions arg of the DataType must have gotten set by a
            # callable in the TYPE_CONVERTERS mapping. For example, Snowflake converts DECIMAL to
            # DECIMAL(38, 0)) in order to facilitate the data type's transpilation.
            #
            # In these cases, we don't really want to return the converted type, but instead retreat
            # and try to parse a Column or Identifier in the section below.
            # 注意：当类型本身带参数（如 DECIMAL(38, 0)）时，解析会消费额外 token
            # 若确实消费了这些参数（index2 - index > 1），保持该类型返回；否则退回并当作列/标识符继续解析
            if data_type.expressions and index2 - index > 1:
                self._retreat(index2)
                return self._parse_column_ops(data_type)

            # 未形成完整类型构造，回到起始位置，尝试解析列/标识符
            self._retreat(index)

        if fallback_to_identifier:
            return self._parse_id_var()

        this = self._parse_column()
        return this and self._parse_column_ops(this)

    def _parse_type_size(self) -> t.Optional[exp.DataTypeParam]:
        # 解析类型尺寸/参数：如 VARCHAR(n) 或 DECIMAL(p, s) 中的 n / (p, s)
        this = self._parse_type()
        if not this:
            return None

        # 若解析到的是无前缀列名（如 varchar），提升为变量形式以参与 DataTypeParam 构造
        if isinstance(this, exp.Column) and not this.table:
            this = exp.var(this.name.upper())

        return self.expression(
            exp.DataTypeParam, this=this, expression=self._parse_var(any_token=True)
        )

    def _parse_user_defined_type(self, identifier: exp.Identifier) -> t.Optional[exp.Expression]:
        # 解析用户自定义类型（UDT）：支持多段命名（schema.package.type 或 db.schema.type）
        type_name = identifier.name

        while self._match(TokenType.DOT):
            type_name = f"{type_name}.{self._advance_any() and self._prev.text}"

        # 使用方言感知的 DataType 构造，并标记为 udt=True
        return exp.DataType.build(type_name, dialect=self.dialect, udt=True)

    def _parse_types(
        self, check_func: bool = False, schema: bool = False, allow_identifiers: bool = True
    ) -> t.Optional[exp.Expression]:
        index = self._index

        this: t.Optional[exp.Expression] = None
        prefix = self._match_text_seq("SYSUDTLIB", ".")

        # 若未直接匹配到内建类型关键字，则在允许的情况下回退到“标识符作为类型”的路径：
        # - 单 token 的标识符若能再次被词法器识别为类型，就等价于类型关键字
        # - 若方言支持 UDT，尝试解析用户自定义类型
        # - 否则回退并放弃类型解析
        if self._match_set(self.TYPE_TOKENS):
            type_token = self._prev.token_type
        else:
            type_token = None
            identifier = allow_identifiers and self._parse_id_var(
                any_token=False, tokens=(TokenType.VAR,)
            )
            if isinstance(identifier, exp.Identifier):
                try:
                    tokens = self.dialect.tokenize(identifier.name)
                except TokenError:
                    tokens = None

                if tokens and len(tokens) == 1 and tokens[0].token_type in self.TYPE_TOKENS:
                    type_token = tokens[0].token_type
                elif self.dialect.SUPPORTS_USER_DEFINED_TYPES:
                    this = self._parse_user_defined_type(identifier)
                else:
                    self._retreat(self._index - 1)
                    return None
            else:
                return None

        if type_token == TokenType.PSEUDO_TYPE:
            return self.expression(exp.PseudoType, this=self._prev.text.upper())

        if type_token == TokenType.OBJECT_IDENTIFIER:
            return self.expression(exp.ObjectIdentifier, this=self._prev.text.upper())

        # https://materialize.com/docs/sql/types/map/
        if type_token == TokenType.MAP and self._match(TokenType.L_BRACKET):
            # 解析 MAP[key_type => value_type]（Materialize 等）：要求箭头与右括号成对匹配
            key_type = self._parse_types(
                check_func=check_func, schema=schema, allow_identifiers=allow_identifiers
            )
            if not self._match(TokenType.FARROW):
                self._retreat(index)
                return None

            value_type = self._parse_types(
                check_func=check_func, schema=schema, allow_identifiers=allow_identifiers
            )
            if not self._match(TokenType.R_BRACKET):
                self._retreat(index)
                return None

            return exp.DataType(
                this=exp.DataType.Type.MAP,
                expressions=[key_type, value_type],
                nested=True,
                prefix=prefix,
            )

        nested = type_token in self.NESTED_TYPE_TOKENS
        is_struct = type_token in self.STRUCT_TYPE_TOKENS
        is_aggregate = type_token in self.AGGREGATE_TYPE_TOKENS
        expressions = None
        maybe_func = False

        if self._match(TokenType.L_PAREN):
            if is_struct:
                # 结构体：括号内以“字段名+类型”的 CSV 形式解析
                expressions = self._parse_csv(lambda: self._parse_struct_types(type_required=True))
            elif nested:
                # 嵌套类型（ARRAY/MAP/STRUCT 等）：递归解析内部类型列表
                expressions = self._parse_csv(
                    lambda: self._parse_types(
                        check_func=check_func, schema=schema, allow_identifiers=allow_identifiers
                    )
                )
                # NULLABLE(T) → 在 DataType 上标注 nullable=True
                if type_token == TokenType.NULLABLE and len(expressions) == 1:
                    this = expressions[0]
                    this.set("nullable", True)
                    self._match_r_paren()
                    return this
            elif type_token in self.ENUM_TYPE_TOKENS:
                # 枚举：括号内是常量列表
                expressions = self._parse_csv(self._parse_equality)
            elif is_aggregate:
                # 聚合类型：括号内第一个参数可能是函数或标识符，后续可能跟类型列表
                func_or_ident = self._parse_function(anonymous=True) or self._parse_id_var(
                    any_token=False, tokens=(TokenType.VAR, TokenType.ANY)
                )
                if not func_or_ident:
                    return None
                expressions = [func_or_ident]
                if self._match(TokenType.COMMA):
                    expressions.extend(
                        self._parse_csv(
                            lambda: self._parse_types(
                                check_func=check_func,
                                schema=schema,
                                allow_identifiers=allow_identifiers,
                            )
                        )
                    )
            else:
                # 通用类型参数：如 DECIMAL(38, 0)、VARCHAR(n)
                expressions = self._parse_csv(self._parse_type_size)

                # Snowflake VECTOR 可能传形如 VECTOR(FLOAT, 3)，需将第一个参数提升为 DataType
                # https://docs.snowflake.com/en/sql-reference/data-types-vector
                if type_token == TokenType.VECTOR and len(expressions) == 2:
                    expressions = self._parse_vector_expressions(expressions)

            # 括号必须闭合；否则整体回退
            if not self._match(TokenType.R_PAREN):
                self._retreat(index)
                return None

            maybe_func = True

        values: t.Optional[t.List[exp.Expression]] = None

        if nested and self._match(TokenType.LT):
            # 解析尖括号 <...> 内的嵌套类型参数：
            # - STRUCT<...>：字段列表
            # - 其他嵌套类型：递归解析为类型列表
            if is_struct:
                expressions = self._parse_csv(lambda: self._parse_struct_types(type_required=True))
            else:
                expressions = self._parse_csv(
                    lambda: self._parse_types(
                        check_func=check_func, schema=schema, allow_identifiers=allow_identifiers
                    )
                )

            if not self._match(TokenType.GT):
                self.raise_error("Expecting >")

            # 支持尖括号后的值列表（结构体/数组构造器），用于内联构造
            if self._match_set((TokenType.L_BRACKET, TokenType.L_PAREN)):
                values = self._parse_csv(self._parse_assignment)
                if not values and is_struct:
                    values = None
                    self._retreat(self._index - 1)
                else:
                    self._match_set((TokenType.R_BRACKET, TokenType.R_PAREN))

        # 时间类型修饰：WITH/WITHOUT TIME ZONE 影响是否产出带时区的类型；解析到此则不再视为“函数调用”
        if type_token in self.TIMESTAMPS:
            if self._match_text_seq("WITH", "TIME", "ZONE"):
                maybe_func = False
                tz_type = (
                    exp.DataType.Type.TIMETZ
                    if type_token in self.TIMES
                    else exp.DataType.Type.TIMESTAMPTZ
                )
                this = exp.DataType(this=tz_type, expressions=expressions)
            elif self._match_text_seq("WITH", "LOCAL", "TIME", "ZONE"):
                maybe_func = False
                this = exp.DataType(this=exp.DataType.Type.TIMESTAMPLTZ, expressions=expressions)
            elif self._match_text_seq("WITHOUT", "TIME", "ZONE"):
                maybe_func = False
        elif type_token == TokenType.INTERVAL:
            # INTERVAL 类型：可携带单位（如 DAY/HOUR），并支持跨度 TO（如 YEAR TO MONTH）
            unit = self._parse_var(upper=True)
            if unit:
                if self._match_text_seq("TO"):
                    unit = exp.IntervalSpan(this=unit, expression=self._parse_var(upper=True))

                this = self.expression(exp.DataType, this=self.expression(exp.Interval, unit=unit))
            else:
                this = self.expression(exp.DataType, this=exp.DataType.Type.INTERVAL)
        elif type_token == TokenType.VOID:
            # VOID 在不少方言中等价于 NULL 类型
            this = exp.DataType(this=exp.DataType.Type.NULL)

        # 当且仅当允许将类型当作函数检查时，尝试窥探字符串参数以判定是否为函数调用
        if maybe_func and check_func:
            index2 = self._index
            peek = self._parse_string()

            if not peek:
                # 非函数调用，回退到进入类型分支前的位置，让上层改走其他解析路径
                self._retreat(index)
                return None

            # 是函数调用：回退到窥探前的位置，后续正常按函数路径处理
            self._retreat(index2)

        if not this:
            # MySQL 等：UNSIGNED 修饰，需将有符号类型映射为无符号类型 token
            if self._match_text_seq("UNSIGNED"):
                unsigned_type_token = self.SIGNED_TO_UNSIGNED_TYPE_TOKEN.get(type_token)
                if not unsigned_type_token:
                    self.raise_error(f"Cannot convert {type_token.value} to unsigned.")

                type_token = unsigned_type_token or type_token

            # 构造最终 DataType，并记录是否为嵌套类型及前缀（如 SYSUDTLIB.）
            # NULLABLE without parentheses can be a column (Presto/Trino)
            if type_token == TokenType.NULLABLE and not expressions:
                self._retreat(index)
                return None
            this = exp.DataType(
                this=exp.DataType.Type[type_token.value],
                expressions=expressions,
                nested=nested,
                prefix=prefix,
            )

            # 允许空的数组/结构体构造：将 values 封装为 Struct/Array 再 cast 为指定类型
            if values is not None:
                cls = exp.Struct if is_struct else exp.Array
                this = exp.cast(cls(expressions=values), this, copy=False)

        elif expressions:
            # 若此前已构造 DataType，则在此补上参数表达式
            this.set("expressions", expressions)

        # 列表类型语法糖：<type> LIST → LIST<<type>>
        # https://materialize.com/docs/sql/types/list/#type-name
        while self._match(TokenType.LIST):
            this = exp.DataType(this=exp.DataType.Type.LIST, expressions=[this], nested=True)

        index = self._index

        # Postgres 语法：INT ARRAY[3] 等价于 INT[3]
        matched_array = self._match(TokenType.ARRAY)

        while self._curr:
            datatype_token = self._prev.token_type
            matched_l_bracket = self._match(TokenType.L_BRACKET)

            if (not matched_l_bracket and not matched_array) or (
                datatype_token == TokenType.ARRAY and self._match(TokenType.R_BRACKET)
            ):
                # 注意：Postgres 允许 CAST 空数组，如 ARRAY[]::INT[]，不要与固定长度数组混淆
                break

            matched_array = False
            values = self._parse_csv(self._parse_assignment) or None
            if (
                values
                and not schema
                and (
                    not self.dialect.SUPPORTS_FIXED_SIZE_ARRAYS or datatype_token == TokenType.ARRAY
                )
            ):
                # 在部分方言（如 DuckDB）中，ARRAY[1] 应被解析为值构造而非固定大小数组类型，因此回退
                self._retreat(index)
                break

            this = exp.DataType(
                this=exp.DataType.Type.ARRAY, expressions=[this], values=values, nested=True
            )
            self._match(TokenType.R_BRACKET)

        # 最后一步：若配置了类型转换器，则对 DataType 做方言特定的收尾转换
        if self.TYPE_CONVERTERS and isinstance(this.this, exp.DataType.Type):
            converter = self.TYPE_CONVERTERS.get(this.this)
            if converter:
                this = converter(t.cast(exp.DataType, this))

        return this

    def _parse_vector_expressions(
        self, expressions: t.List[exp.Expression]
    ) -> t.List[exp.Expression]:
        return [exp.DataType.build(expressions[0].name, dialect=self.dialect), *expressions[1:]]

    def _parse_struct_types(self, type_required: bool = False) -> t.Optional[exp.Expression]:
        index = self._index
        # 解析 STRUCT/ROW 等复合类型字段：处理标识符与类型标记冲突的场景
        if (
            self._curr
            and self._next
            and self._curr.token_type in self.TYPE_TOKENS
            and self._next.token_type in self.TYPE_TOKENS
        ):
            # Takes care of special cases like `STRUCT<list ARRAY<...>>` where the identifier is also a
            # type token. Without this, the list will be parsed as a type and we'll eventually crash
            # 处理诸如 STRUCT<list ARRAY<...>> 这类特殊情况：
            # 当前与下一个 token 都是类型标记时，将当前视为标识符，避免被误解析为类型导致后续崩溃
            this = self._parse_id_var()
        else:
            # 常规路径：优先解析类型；若不成，则回退为标识符
            this = (
                self._parse_type(parse_interval=False, fallback_to_identifier=True)
                or self._parse_id_var()
            )

        # 支持 name: type 形式
        self._match(TokenType.COLON)

        if (
            type_required
            and not isinstance(this, exp.DataType)
            and not self._match_set(self.TYPE_TOKENS, advance=False)
        ):
            # 当明确要求出现类型，而当前并非类型且后续也不似类型时：
            # 回退到进入本函数前的位置，整体改用通用类型解析，提高健壮性
            self._retreat(index)
            return self._parse_types()

        # 构造列定义（可能包含默认值/约束等）
        return self._parse_column_def(this)

    def _parse_at_time_zone(self, this: t.Optional[exp.Expression]) -> t.Optional[exp.Expression]:
        # 解析 AT TIME ZONE 子句；若不存在则原样返回
        if not self._match_text_seq("AT", "TIME", "ZONE"):
            return this
        # 规整为统一的 AtTimeZone 表达式，便于各方言之间转换
        return self.expression(exp.AtTimeZone, this=this, zone=self._parse_unary())

    def _parse_column(self) -> t.Optional[exp.Expression]:
        # 先解析一个“列引用”，再尝试附加列级操作（.、::、-> 等）
        this = self._parse_column_reference()
        column = self._parse_column_ops(this) if this else self._parse_bracket(this)

        # 某些方言支持列级 join 标记（如 ? 表示可选连接）
        if self.dialect.SUPPORTS_COLUMN_JOIN_MARKS and column:
            column.set("join_mark", self._match(TokenType.JOIN_MARKER))

        return column

    def _parse_column_reference(self) -> t.Optional[exp.Expression]:
        # 常规尝试解析一个字段/标识符/函数等
        this = self._parse_field()
        if (
            not this
            and self._match(TokenType.VALUES, advance=False)
            and self.VALUES_FOLLOWED_BY_PAREN
            and (not self._next or self._next.token_type != TokenType.L_PAREN)
        ):
            # 在 VALUES 可作关键字也可作标识符的歧义场景下：
            # 若未跟随 '('，则将 VALUES 解析为标识符以提高兼容性
            this = self._parse_id_var()

        if isinstance(this, exp.Identifier):
            # We bubble up comments from the Identifier to the Column
            # 将 Identifier 冒泡的注释提升至 Column 节点，避免注释丢失
            this = self.expression(exp.Column, comments=this.pop_comments(), this=this)

        return this

    def _parse_colon_as_variant_extract(
        self, this: t.Optional[exp.Expression]
    ) -> t.Optional[exp.Expression]:
        # 解析 VARIANT/JSON 的冒号提取语法（如 Snowflake/Databricks：col:"a".b:c）
        # 目标：将链式的 :/./[] 访问规整为 JSONExtract，并处理与 :: 强制类型转换的优先级
        casts = []
        json_path = []
        escape = None

        while self._match(TokenType.COLON):
            start_index = self._index

            # Snowflake 允许保留关键字作为 JSON 键名；
            # 这里允许 SELECT 出现在键名中以提升兼容性
            # Snowflake allows reserved keywords as json keys but advance_any() excludes TokenType.SELECT from any_tokens=True
            path = self._parse_column_ops(
                self._parse_field(any_token=True, tokens=(TokenType.SELECT,))
            )

            # The cast :: operator has a lower precedence than the extraction operator :, so
            # we rearrange the AST appropriately to avoid casting the JSON path
            # :: 转换的优先级低于 : 提取；
            # 若 path 被提前解析为 Cast，则把目标类型暂存起来，稍后在最外层 JSONExtract 上再应用
            while isinstance(path, exp.Cast):
                casts.append(path.to)
                path = path.this

            # 为了构造 JSON path 字符串，需要知道当前 : 访问的结束 token
            if casts:
                dcolon_offset = next(
                    i
                    for i, t in enumerate(self._tokens[start_index:])
                    if t.token_type == TokenType.DCOLON
                )
                end_token = self._tokens[start_index + dcolon_offset - 1]
            else:
                end_token = self._prev

            if path:
                # Escape single quotes from Snowflake's colon extraction (e.g. col:"a'b") as
                # it'll roundtrip to a string literal in GET_PATH
                # 若键名为带引号的标识符（如 "a'b"），Snowflake 的冒号语法会保留单引号；
                # 这里记录 escape 标志，便于在转为 GET_PATH/JSON 路径时进行转义，保证可逆
                if isinstance(path, exp.Identifier) and path.quoted:
                    escape = True

                # 收集当前片段对应的原始 SQL 作为 JSON 路径的一部分，保持 round-trip
                json_path.append(self._find_sql(self._tokens[start_index], end_token))

        # The VARIANT extract in Snowflake/Databricks is parsed as a JSONExtract; Snowflake uses the json_path in GET_PATH() while
        # Databricks transforms it back to the colon/dot notation
        # 将冒号链统一封装为 JSONExtract；Snowflake 最终会转为 GET_PATH 形式，
        # 而 Databricks 会在 to_sql 时恢复为冒号/点符号
        if json_path:
            json_path_expr = self.dialect.to_json_path(exp.Literal.string(".".join(json_path)))

            if json_path_expr:
                json_path_expr.set("escape", escape)

            this = self.expression(
                exp.JSONExtract,
                this=this,
                expression=json_path_expr,
                variant_extract=True,
                requires_json=self.JSON_EXTRACT_REQUIRES_JSON_EXPRESSION,
            )

            # 重新把之前推迟的 :: 强制类型转换应用到最外层结果上
            while casts:
                this = self.expression(exp.Cast, this=this, to=casts.pop())

        return this

    def _parse_dcolon(self) -> t.Optional[exp.Expression]:
        # 解析 :: 类型转换操作；复用通用类型解析逻辑
        return self._parse_types()

    def _parse_column_ops(self, this: t.Optional[exp.Expression]) -> t.Optional[exp.Expression]:
        # 解析列上的后缀/中缀操作（., ::, -> 等）；按顺序折叠为 AST 链
        this = self._parse_bracket(this)  # 先处理可能存在的括号访问（如 arr[1] 或函数调用）

        while self._match_set(self.COLUMN_OPERATORS):  # 连续匹配所有可用的列操作符
            op_token = self._prev.token_type
            op = self.COLUMN_OPERATORS.get(op_token)

            if op_token in self.CAST_COLUMN_OPERATORS:
                # 处理强制类型转换（如 :: 或 .:）；此时期待紧随一个类型
                field = self._parse_dcolon()
                if not field:
                    self.raise_error("Expected type")
            elif op and self._curr:
                # 对常规列操作，优先解析为“列引用”或括号表达式
                field = self._parse_column_reference() or self._parse_bitwise()
                if isinstance(field, exp.Column) and self._match(TokenType.DOT, advance=False):
                    # 若后面紧跟点号，说明是链式访问，如 a.b.c；递归继续解析
                    field = self._parse_column_ops(field)
            else:
                # 兜底：尽可能宽松地解析任意字段（支持匿名函数等情况）
                field = self._parse_field(any_token=True, anonymous_func=True)

            # 函数调用可带限定前缀（如 x.y.FOO()）；
            # 为了让前缀路径按点号连接到函数上，需要把此前的 Column 链转换为 Dot 链
            # 参考 BigQuery 函数限定规则
            if isinstance(field, (exp.Func, exp.Window)) and this:
                this = this.transform(
                    lambda n: n.to_dot(include_dots=False) if isinstance(n, exp.Column) else n
                )

            if op:
                # 若为已注册的列操作（如 ->、->>、.: 等），调用其对应的处理器构造表达式
                this = op(self, this, field)
            elif isinstance(this, exp.Column) and not this.args.get("catalog"):
                # 否则，若当前为 Column 且无最上层 catalog，则将 field 作为更深一层路径：
                # 维持 catalog -> db -> table -> column 的层级映射
                this = self.expression(
                    exp.Column,
                    comments=this.comments,
                    this=field,
                    table=this.this,
                    db=this.args.get("table"),
                    catalog=this.args.get("db"),
                )
            elif isinstance(field, exp.Window):
                # 特殊情况：当 field 是窗口表达式时，把已有的路径（Dot）应用到窗口函数本身
                # 等价于把前缀移动到 window 的函数节点上
                window_func = self.expression(exp.Dot, this=this, expression=field.this)
                field.set("this", window_func)
                this = field
            else:
                # 常规点号访问：构造 Dot(this.field)
                this = self.expression(exp.Dot, this=this, expression=field)

            if field and field.comments:
                # 合并子节点上的注释，避免注释在转换中丢失
                t.cast(exp.Expression, this).add_comments(field.pop_comments())

            # 操作符后仍可能紧跟括号（如函数调用/下标访问），继续解析
            this = self._parse_bracket(this)

        # Snowflake/Databricks 可将 ':' 视作 VARIANT/JSON 提取；
        # 若启用，则把点号/冒号链统一转换为 JSONExtract 以便不同方言间转译
        return self._parse_colon_as_variant_extract(this) if self.COLON_IS_VARIANT_EXTRACT else this

    def _parse_paren(self) -> t.Optional[exp.Expression]:
        # 解析以括号开头的片段：可能是 (SELECT ...)、(a, b, c) 或 (expr)
        if not self._match(TokenType.L_PAREN):
            return None

        comments = self._prev_comments  # 记录左括号之前的注释，稍后附回
        query = self._parse_select()

        if query:
            # 若括号内是查询，先作为单元素列表处理，便于统一逻辑
            expressions = [query]
        else:
            # 否则按通用规则解析可能的表达式列表（逗号分隔）
            expressions = self._parse_expressions()

        # 对第一个元素应用可能的查询修饰符（ORDER/LIMIT 等）
        this = seq_get(expressions, 0)

        if not this and self._match(TokenType.R_PAREN, advance=False):
            # 空括号：() 归一为 Tuple()
            this = self.expression(exp.Tuple)
        elif isinstance(this, exp.UNWRAPPED_QUERIES):
            # 某些查询在此处需包裹为子查询，以便后续可挂载别名/修饰
            this = self._parse_subquery(this=this, parse_alias=False)
        elif isinstance(this, (exp.Subquery, exp.Values)):
            # 若已经是子查询，先解析可能的集合操作（UNION/INTERSECT/...）再继续
            this = self._parse_subquery(
                this=self._parse_query_modifiers(self._parse_set_operations(this)),
                parse_alias=False,
            )
        elif len(expressions) > 1 or self._prev.token_type == TokenType.COMMA:
            # 多个表达式或以逗号为结尾：视为 Tuple(a, b, ...)
            this = self.expression(exp.Tuple, expressions=expressions)
        else:
            # 单一表达式且非查询：保留括号信息（如影响函数/运算优先级）
            this = self.expression(exp.Paren, this=this)

        if this:
            # 将进入括号前采集到的注释补回当前节点，保证注释位置语义不丢失
            this.add_comments(comments)

        self._match_r_paren(expression=this)

        if isinstance(this, exp.Paren) and isinstance(this.this, exp.AggFunc):
            return self._parse_window(this)

        return this

    def _parse_primary(self) -> t.Optional[exp.Expression]:
        # 解析“原子”级别的主表达式：数字/字符串/NULL/TRUE 等，或内置主解析器
        if self._match_set(self.PRIMARY_PARSERS):
            token_type = self._prev.token_type
            primary = self.PRIMARY_PARSERS[token_type](self, self._prev)

            if token_type == TokenType.STRING:
                # 处理相邻字符串字面量自动合并（SQL 标准允许 'a' 'b' => 'ab'）
                expressions = [primary]
                while self._match(TokenType.STRING):
                    expressions.append(exp.Literal.string(self._prev.text))

                if len(expressions) > 1:
                    return self.expression(exp.Concat, expressions=expressions)

            return primary

        if self._match_pair(TokenType.DOT, TokenType.NUMBER):
            # 处理形如 .123 的数字写法，规范化为 0.123
            return exp.Literal.number(f"0.{self._prev.text}")

        # 否则尝试解析括号表达式（可能展开为 Tuple/子查询/Paren）
        return self._parse_paren()

    def _parse_field(
        self,
        any_token: bool = False,
        tokens: t.Optional[t.Collection[TokenType]] = None,
        anonymous_func: bool = False,
    ) -> t.Optional[exp.Expression]:
        """解析一个字段单元，按模式优先级在函数/primary/标识符之间选择。

        设计动机：
        - 匿名函数模式（anonymous_func=True）优先尝试函数解析，因部分方言匿名函数可能不以典型函数名开头；
        - 常规模式先尝试 primary，避免把普通标识符或字面量误判为函数调用；
        - 若都不匹配，回退到变量/标识符解析，并可通过 any_token/tokens 控制匹配范围。
        """
        if anonymous_func:
            # 匿名函数优先：尽早捕获可能的匿名函数形态
            field = (
                self._parse_function(anonymous=anonymous_func, any_token=any_token)
                or self._parse_primary()
            )
        else:
            field = self._parse_primary() or self._parse_function(
                anonymous=anonymous_func, any_token=any_token
            )
        return field or self._parse_id_var(any_token=any_token, tokens=tokens)

    def _parse_function(
        self,
        functions: t.Optional[t.Dict[str, t.Callable]] = None,
        anonymous: bool = False,
        optional_parens: bool = True,
        any_token: bool = False,
    ) -> t.Optional[exp.Expression]:
        """解析函数入口，兼容 {fn <function>} 逃逸语法并委派给具体函数调用解析。

        说明：
        - 部分方言（Snowflake/MySQL）支持 `{fn ...}` 语法；这里负责检测并成对消费大括号；
        - 具体函数体与参数/窗口等细节解析在 `_parse_function_call` 中完成。
        """
        # This allows us to also parse {fn <function>} syntax (Snowflake, MySQL support this)
        # See: https://community.snowflake.com/s/article/SQL-Escape-Sequences
        fn_syntax = False
        if (
            self._match(TokenType.L_BRACE, advance=False)
            and self._next
            and self._next.text.upper() == "FN"
        ):
            # 进入 `{ fn` 语法：消费 "{" 与 "fn"
            self._advance(2)
            fn_syntax = True

        func = self._parse_function_call(
            functions=functions,
            anonymous=anonymous,
            optional_parens=optional_parens,
            any_token=any_token,
        )

        if fn_syntax:
            # 结束 `{fn ...}` 语法体：匹配右大括号
            self._match(TokenType.R_BRACE)

        return func

    def _parse_function_args(self, alias: bool = False) -> t.List[exp.Expression]:
        return self._parse_csv(lambda: self._parse_lambda(alias=alias))

    def _parse_function_call(
        self,
        functions: t.Optional[t.Dict[str, t.Callable]] = None,
        anonymous: bool = False,
        optional_parens: bool = True,
        any_token: bool = False,
    ) -> t.Optional[exp.Expression]:
        """解析函数调用的主体（含无括号函数、参数、子查询谓词与窗口）。

        关键点：
        - 无括号函数优先：当允许省略括号且找到匹配解析器时直接处理；
        - 非 `(` 开头时，仍可能是“无括号函数”，否则即非函数返回 None；
        - any_token 放宽函数名匹配，但仍拒绝保留字；
        - 对已知函数按签名构造表达式；若被注释标记为匿名则转为 Anonymous；
        - 对 IN/EXISTS 等子查询谓词，`(...)` 内遇到 SELECT/WITH 则整体作为子查询；
        - 统一在末尾匹配右括号并解析窗口（OVER ...）。
        """
        if not self._curr:
            return None

        comments = self._curr.comments
        prev = self._prev
        token = self._curr
        token_type = self._curr.token_type
        this = self._curr.text
        upper = this.upper()

        # 无括号函数：存在专用解析器且当前 token 合法时，直接解析并尝试窗口
        parser = self.NO_PAREN_FUNCTION_PARSERS.get(upper)
        if optional_parens and parser and token_type not in self.INVALID_FUNC_NAME_TOKENS:
            self._advance()
            return self._parse_window(parser(self))

        # 非 `(` 开头：若是已知的无括号函数则构造表达式；否则不是函数，返回 None
        if not self._next or self._next.token_type != TokenType.L_PAREN:
            if optional_parens and token_type in self.NO_PAREN_FUNCTIONS:
                self._advance()
                return self.expression(self.NO_PAREN_FUNCTIONS[token_type])

            return None

        # any_token 模式放宽函数名，但保留字仍不可用作函数名
        if any_token:
            if token_type in self.RESERVED_TOKENS:
                return None
        elif token_type not in self.FUNC_TOKENS:
            return None

        # 消费函数名与左括号，进入参数解析
        self._advance(2)

        # 若函数有专用解析器则优先之；否则进入通用流程
        parser = self.FUNCTION_PARSERS.get(upper)
        if parser and not anonymous:
            this = parser(self)
        else:
            # 子查询谓词：IN/EXISTS 等需要将括号内容整体视作子查询或元组
            subquery_predicate = self.SUBQUERY_PREDICATES.get(token_type)

            if subquery_predicate:
                expr = None
                if self._curr.token_type in (TokenType.SELECT, TokenType.WITH):
                    # 形如 IN (SELECT ...)、EXISTS (WITH ... SELECT ...)
                    expr = self._parse_select()
                    self._match_r_paren()
                elif prev and prev.token_type in (TokenType.LIKE, TokenType.ILIKE):
                    # Backtrack one token since we've consumed the L_PAREN here. Instead, we'd like
                    # to parse "LIKE [ANY | ALL] (...)" as a whole into an exp.Tuple or exp.Paren
                    # 在 LIKE/ILIKE 上下文中回退一格，交由更高层解析为整体元组/括号表达式
                    self._advance(-1)
                    expr = self._parse_bitwise()

                if expr:
                    return self.expression(subquery_predicate, comments=comments, this=expr)

            if functions is None:
                functions = self.FUNCTIONS

            function = functions.get(upper)
            known_function = function and not anonymous

            # 未知函数或需要命名参数的函数：以 KV 形式解析参数，稍后可能转为属性等价表达
            alias = not known_function or upper in self.FUNCTIONS_WITH_ALIASED_ARGS
            args = self._parse_function_args(alias)

            post_func_comments = self._curr and self._curr.comments
            if known_function and post_func_comments:
                # If the user-inputted comment "/* sqlglot.anonymous */" is following the function
                # call we'll construct it as exp.Anonymous, even if it's "known"
                # 支持通过跟随的特殊注释将“已知函数”强制视为匿名函数，便于生成保留原貌
                if any(
                    comment.lstrip().startswith(exp.SQLGLOT_ANONYMOUS)
                    for comment in post_func_comments
                ):
                    known_function = False

            if alias and known_function:
                # 已知函数且使用命名参数：把 KV 参数转化为属性等价表达（PropEq）
                args = self._kv_to_prop_eq(args)

            if known_function:
                func_builder = t.cast(t.Callable, function)

                if "dialect" in func_builder.__code__.co_varnames:
                    # 某些函数构造器依赖方言特性
                    func = func_builder(args, dialect=self.dialect)
                else:
                    func = func_builder(args)

                func = self.validate_expression(func, args)
                if self.dialect.PRESERVE_ORIGINAL_NAMES:
                    # 可选：保留用户输入的原始函数名，避免在生成阶段被规范化
                    func.meta["name"] = this

                this = func
            else:
                if token_type == TokenType.IDENTIFIER:
                    # 将函数名标识为带引号的 Identifier，以保留大小写/特殊字符信息
                    this = exp.Identifier(this=this, quoted=True).update_positions(token)

                # 构造匿名函数表达式，携带参数列表以便后续生成还原
                this = self.expression(exp.Anonymous, this=this, expressions=args)
                this = this.update_positions(token)

        if isinstance(this, exp.Expression):
            # 将靠近函数名的前置注释合并到表达式，保证注释信息不丢
            this.add_comments(comments)

        # 匹配右括号并尝试解析窗口（OVER 子句）
        self._match_r_paren(this)
        return self._parse_window(this)

    def _to_prop_eq(self, expression: exp.Expression, index: int) -> exp.Expression:
        """将非 KV 形态的参数在需要时转为属性等价表达（占位，默认直返）。

        说明：
        - 基类默认直接返回原表达式；子类可按方言需要覆盖以支持位置参数转命名参数。
        - 这里保留 index 便于子类根据参数位置做映射。
        """
        return expression

    def _kv_to_prop_eq(
        self, expressions: t.List[exp.Expression], parse_map: bool = False
    ) -> t.List[exp.Expression]:
        """将参数列表中的 KV 形态统一转换为 PropertyEQ，方便下游统一处理。

        设计原因：
        - 某些函数允许以 `key => value`、`key: value` 或 `alias := value` 等形态传参；
        - 统一转换为 `exp.PropertyEQ(this=<key>, expression=<value>)`，以便生成与优化阶段可统一处理；
        - `parse_map=True` 时保留 map 风格的 key 表达式；否则将 key 规范化为 Identifier。
        """
        transformed = []

        for index, e in enumerate(expressions):
            if isinstance(e, self.KEY_VALUE_DEFINITIONS):
                # 形如 `alias := value` 的别名赋值，归一化为 PropertyEQ
                if isinstance(e, exp.Alias):
                    e = self.expression(exp.PropertyEQ, this=e.args.get("alias"), expression=e.this)

                if not isinstance(e, exp.PropertyEQ):
                    # 将通用的 KeyValue 形态转为 PropertyEQ；注意 map 形态保留原始 this
                    e = self.expression(
                        exp.PropertyEQ,
                        this=e.this if parse_map else exp.to_identifier(e.this.name),
                        expression=e.expression,
                    )

                if isinstance(e.this, exp.Column):
                    # 若 key 被解析成列（如 a.b），仅保留末级标识符（b）作为属性名
                    e.this.replace(e.this.this)
            else:
                # 非 KV 形态交由 _to_prop_eq 钩子按需处理（默认直返）
                e = self._to_prop_eq(e, index)

            transformed.append(e)

        return transformed

    def _parse_user_defined_function_expression(self) -> t.Optional[exp.Expression]:
        """解析用户自定义函数（UDF）体的表达式部分。

        说明：
        - 此处直接委派到通用语句解析 `_parse_statement`，以兼容方言对 UDF 体的多样语法。
        """
        return self._parse_statement()

    def _parse_function_parameter(self) -> t.Optional[exp.Expression]:
        """解析 UDF 的单个参数定义。"""
        return self._parse_column_def(this=self._parse_id_var(), computed_column=False)

    def _parse_user_defined_function(
        self, kind: t.Optional[TokenType] = None
    ) -> t.Optional[exp.Expression]:
        """解析用户自定义函数（UDF）声明。

        关键逻辑：
        - 先解析函数的限定名（可能包含模式/schema）；
        - 若后续没有参数括号，直接返回函数名表达式；
        - 若存在括号，解析逗号分隔的参数列表并包装为 `exp.UserDefinedFunction`。
        """
        this = self._parse_table_parts(schema=True)

        if not self._match(TokenType.L_PAREN):
            # 无参数形态：直接返回解析出的函数名表达式
            return this

        expressions = self._parse_csv(self._parse_function_parameter)
        self._match_r_paren()
        return self.expression(
            exp.UserDefinedFunction, this=this, expressions=expressions, wrapped=True
        )

    def _parse_introducer(self, token: Token) -> exp.Introducer | exp.Identifier:
        """解析引入器（如 `_utf8'...'`），或回退为标识符。

        说明：
        - 若能解析到后续 primary 字面量，则构造 `exp.Introducer` 绑定引入器文本与字面量；
        - 否则，回退为常规的标识符解析，保留原始 token。
        """
        literal = self._parse_primary()
        if literal:
            return self.expression(exp.Introducer, this=token.text, expression=literal)

        return self._identifier_expression(token)

    def _parse_session_parameter(self) -> exp.SessionParameter:
        """解析会话参数（可选包含命名空间/种类）。

        关键逻辑：
        - 先解析一个标识符或主表达式作为参数名；
        - 若匹配到点号，点号前部分作为 `kind`（命名空间/类别）；
        - 返回 `exp.SessionParameter`，其中 `kind` 记录类别信息。
        """
        kind = None
        this = self._parse_id_var() or self._parse_primary()

        if this and self._match(TokenType.DOT):
            kind = this.name
            this = self._parse_var() or self._parse_primary()

        return self.expression(exp.SessionParameter, this=this, kind=kind)

    def _parse_lambda_arg(self) -> t.Optional[exp.Expression]:
        """解析 lambda 形参：默认解析为一个标识符/变量。"""
        return self._parse_id_var()

    def _parse_lambda(self, alias: bool = False) -> t.Optional[exp.Expression]:
        """解析 lambda 表达式（形参列表与主体）。

        关键逻辑：
        - 先尝试解析括号包裹的形参列表；若右括号缺失则整体回退，避免误吞；
        - 若无括号，则按单参数形态解析一个形参；
        - 若随后匹配到 lambda 箭头/关键词（在 `LAMBDAS` 集中），则用已收集的形参构造 lambda；
        - 否则回退到最初位置，将其视为普通表达式或 SELECT；
        - 对于非 lambda 情况，继续解析 DISTINCT/ORDER/LIMIT 等后续子句，保持与函数参数语境一致。
        """
        index = self._index

        if self._match(TokenType.L_PAREN):
            expressions = t.cast(
                t.List[t.Optional[exp.Expression]], self._parse_csv(self._parse_lambda_arg)
            )

            if not self._match(TokenType.R_PAREN):
                # 缺少右括号：认为并非 lambda 形参列表，回退到进入前的索引
                self._retreat(index)
        else:
            # 单参数 lambda 形态：无括号，直接解析一个形参
            expressions = [self._parse_lambda_arg()]

        if self._match_set(self.LAMBDAS):
            # 命中 lambda 箭头/关键字：据以构造 lambda 表达式
            return self.LAMBDAS[self._prev.token_type](self, expressions)

        # 非 lambda：全部回退，随后按普通表达式/SELECT 处理
        self._retreat(index)

        this: t.Optional[exp.Expression]

        if self._match(TokenType.DISTINCT):
            this = self.expression(
                exp.Distinct, expressions=self._parse_csv(self._parse_assignment)
            )
        else:
            this = self._parse_select_or_expression(alias=alias)

        # 继续解析 RESPECT/IGNORE NULLS、HAVING MAX、ORDER、LIMIT 等链式子句
        return self._parse_limit(
            self._parse_order(self._parse_having_max(self._parse_respect_or_ignore_nulls(this)))
        )

    def _parse_schema(self, this: t.Optional[exp.Expression] = None) -> t.Optional[exp.Expression]:
        """解析列定义 schema（区别于子查询/CTE 的括号部分）。

        关键逻辑：
        - 先尝试匹配左括号；若没有则直接返回传入的 `this`；
        - 通过尝试匹配 SELECT 起始 token 来判定是否其实是子查询/CTE，上述情况需回退并返回；
        - 否则解析逗号分隔的约束或字段定义，封装为 `exp.Schema`。
        """
        index = self._index
        if not self._match(TokenType.L_PAREN):
            return this

        # Disambiguate between schema and subquery/CTE, e.g. in INSERT INTO table (<expr>),
        # expr can be of both types
        if self._match_set(self.SELECT_START_TOKENS):
            # 命中 SELECT 起点：说明这是子查询/CTE 的括号，不是 schema，回退
            self._retreat(index)
            return this
        args = self._parse_csv(lambda: self._parse_constraint() or self._parse_field_def())
        self._match_r_paren()
        return self.expression(exp.Schema, this=this, expressions=args)

    def _parse_field_def(self) -> t.Optional[exp.Expression]:
        """解析字段定义：把一个字段/表达式包装为列定义单元。"""
        return self._parse_column_def(self._parse_field(any_token=True))

    def _parse_column_def(
        self, this: t.Optional[exp.Expression], computed_column: bool = True
    ) -> t.Optional[exp.Expression]:
        """解析列定义（ColumnDef），支持计算列与多种约束。

        关键逻辑：
        - 若传入的是 `exp.Column`，列定义语义上应为标识符，因此取其 `this`；
        - `computed_column=False` 时允许存在别名关键字，需先尝试匹配 `ALIAS`；
        - 先解析类型；若匹配到 `FOR ORDINALITY` 则构造序号列定义；
        - 当无类型但遇到 `ALIAS`/`MATERIALIZED`，解析为“计算列”约束；
        - 否则若既有类型又探测到 `ALIAS`（并满足包裹条件），则解析“包裹的计算列”约束；
        - 随后持续解析其它列级约束；若既无类型也无约束，则视为裸标识符返回；
        - 最终返回 ColumnDef，包含名称、类型与全部约束。
        """
        # column defs are not really columns, they're identifiers
        if isinstance(this, exp.Column):
            # 列定义处不应保留表.列结构，取末级标识符作为列名
            this = this.this

        if not computed_column:
            # 对于非计算列形态，允许紧随一个 ALIAS 关键字（方言兼容）
            self._match(TokenType.ALIAS)

        # 解析类型信息（schema=True 允许更宽松的类型上下文，如 DDL）
        kind = self._parse_types(schema=True)

        # 形如 FOR ORDINALITY 的序号列
        if self._match_text_seq("FOR", "ORDINALITY"):
            return self.expression(exp.ColumnDef, this=this, ordinality=True)

        constraints: t.List[exp.Expression] = []

        # 情况一：未显式给出类型，且命中 ALIAS/MATERIALIZED，解析为计算列约束
        if (not kind and self._match(TokenType.ALIAS)) or self._match_texts(
            ("ALIAS", "MATERIALIZED")
        ):
            persisted = self._prev.text.upper() == "MATERIALIZED"
            constraint_kind = exp.ComputedColumnConstraint(
                this=self._parse_assignment(),
                persisted=persisted or self._match_text_seq("PERSISTED"),
                data_type=exp.Var(this="AUTO")
                if self._match_text_seq("AUTO")
                else self._parse_types(),
                not_null=self._match_pair(TokenType.NOT, TokenType.NULL),
            )
            constraints.append(self.expression(exp.ColumnConstraint, kind=constraint_kind))
        # 情况二：已有类型，同时探测到 ALIAS；若要求包裹（某些方言），则需后续为左括号
        elif (
            kind
            and self._match(TokenType.ALIAS, advance=False)
            and (
                not self.WRAPPED_TRANSFORM_COLUMN_CONSTRAINT
                or (self._next and self._next.token_type == TokenType.L_PAREN)
            )
        ):
            self._advance()
            constraints.append(
                self.expression(
                    exp.ColumnConstraint,
                    kind=exp.ComputedColumnConstraint(
                        this=self._parse_disjunction(),
                        persisted=self._match_texts(("STORED", "VIRTUAL"))
                        and self._prev.text.upper() == "STORED",
                    ),
                )
            )

        # 继续解析其它列级约束，直到不再匹配
        while True:
            constraint = self._parse_column_constraint()
            if not constraint:
                break
            constraints.append(constraint)

        # 若既无类型也无任何约束，维持为裸标识符表达式
        if not kind and not constraints:
            return this

        # 构造完整 ColumnDef，包含列名/类型/约束集合
        return self.expression(exp.ColumnDef, this=this, kind=kind, constraints=constraints)

    def _parse_auto_increment(
        self,
    ) -> exp.GeneratedAsIdentityColumnConstraint | exp.AutoIncrementColumnConstraint:
        """解析自增/标识列约束（两种语法路径）。

        关键逻辑：
        - 括号形态：`(start, increment)`，按位置获取起始与步长；
        - 关键字形态：`START <expr> INCREMENT <expr> [ORDER|NOORDER]`，可额外指示顺序；
        - 若同时解析到 `start` 与 `increment`，则构造 GeneratedAsIdentity（含可选 order）；
        - 否则退化为简单的 AutoIncrement 约束。
        """
        start = None
        increment = None
        order = None

        if self._match(TokenType.L_PAREN, advance=False):
            # 形如 (start, increment) 的位置参数写法
            args = self._parse_wrapped_csv(self._parse_bitwise)
            start = seq_get(args, 0)
            increment = seq_get(args, 1)
        elif self._match_text_seq("START"):
            # 关键字写法：START x INCREMENT y [ORDER|NOORDER]
            start = self._parse_bitwise()
            self._match_text_seq("INCREMENT")
            increment = self._parse_bitwise()
            if self._match_text_seq("ORDER"):
                order = True
            elif self._match_text_seq("NOORDER"):
                order = False

        if start and increment:
            # 同时具备 start/increment 时按“标识列”语义建模，order 表示顺序性（None/True/False）
            return exp.GeneratedAsIdentityColumnConstraint(
                start=start, increment=increment, this=False, order=order
            )

        # 否则回退到简单的自增列约束
        return exp.AutoIncrementColumnConstraint()

    def _parse_auto_property(self) -> t.Optional[exp.AutoRefreshProperty]:
        """解析自动属性（当前支持 REFRESH <IDENT> 形式）。"""
        if not self._match_text_seq("REFRESH"):
            # 未命中关键字：回退一步以避免吞掉上文 token，返回 None 表示不解析此属性
            self._retreat(self._index - 1)
            return None
        return self.expression(exp.AutoRefreshProperty, this=self._parse_var(upper=True))

    def _parse_compress(self) -> exp.CompressColumnConstraint:
        """解析压缩列约束：支持括号内多参数或单表达式。"""
        if self._match(TokenType.L_PAREN, advance=False):
            # 括号形态：解析逗号分隔参数列表作为压缩选项集合
            return self.expression(
                exp.CompressColumnConstraint, this=self._parse_wrapped_csv(self._parse_bitwise)
            )

        # 非括号形态：单一表达式作为压缩选项
        return self.expression(exp.CompressColumnConstraint, this=self._parse_bitwise())

    def _parse_generated_as_identity(
        self,
    ) -> (
        exp.GeneratedAsIdentityColumnConstraint
        | exp.ComputedColumnConstraint
        | exp.GeneratedAsRowColumnConstraint
    ):
        """解析 GENERATED AS IDENTITY/ROW 等生成列约束。

        关键逻辑：
        - `BY DEFAULT` 与 `ALWAYS` 两种模式：分别对应 `this=False/True`；
        - 支持 `ROW START/END [HIDDEN]`，返回 Row 级生成列约束；
        - 若后随括号，解析 START WITH / INCREMENT BY / MINVALUE / MAXVALUE / [NO] CYCLE 等参数；
        - 非 IDENTITY 形态时，括号里可直接携带表达式范围；
        - 若无显式 START，但接着是数字序列 `(n, m)`，按位置补全 start/increment。
        """
        if self._match_text_seq("BY", "DEFAULT"):
            on_null = self._match_pair(TokenType.ON, TokenType.NULL)
            this = self.expression(
                exp.GeneratedAsIdentityColumnConstraint, this=False, on_null=on_null
            )
        else:
            self._match_text_seq("ALWAYS")
            this = self.expression(exp.GeneratedAsIdentityColumnConstraint, this=True)

        # 规范语法中 GENERATED ALWAYS/DEFAULT AS ...，此处匹配 AS（在 tokens 中别名为 ALIAS）
        self._match(TokenType.ALIAS)

        if self._match_text_seq("ROW"):
            # 生成行号/系统行信息：ROW START|END [HIDDEN]
            start = self._match_text_seq("START")
            if not start:
                self._match(TokenType.END)
            hidden = self._match_text_seq("HIDDEN")
            return self.expression(exp.GeneratedAsRowColumnConstraint, start=start, hidden=hidden)

        identity = self._match_text_seq("IDENTITY")

        if self._match(TokenType.L_PAREN):
            # 解析 IDENTITY 参数或表达式范围
            if self._match(TokenType.START_WITH):
                this.set("start", self._parse_bitwise())
            if self._match_text_seq("INCREMENT", "BY"):
                this.set("increment", self._parse_bitwise())
            if self._match_text_seq("MINVALUE"):
                this.set("minvalue", self._parse_bitwise())
            if self._match_text_seq("MAXVALUE"):
                this.set("maxvalue", self._parse_bitwise())

            if self._match_text_seq("CYCLE"):
                this.set("cycle", True)
            elif self._match_text_seq("NO", "CYCLE"):
                this.set("cycle", False)

            if not identity:
                # 非 IDENTITY 形态：括号内给的是一般表达式区间
                this.set("expression", self._parse_range())
            elif not this.args.get("start") and self._match(TokenType.NUMBER, advance=False):
                # 兼容形如 (n, m) 的位置参数，补齐 start/increment
                args = self._parse_csv(self._parse_bitwise)
                this.set("start", seq_get(args, 0))
                this.set("increment", seq_get(args, 1))

            self._match_r_paren()

        return this

    def _parse_inline(self) -> exp.InlineLengthColumnConstraint:
        """解析内联列长度约束（INLINE LENGTH）。

        说明：
        - 需匹配关键字 LENGTH 后，使用一个表达式作为长度值；
        - 返回 `exp.InlineLengthColumnConstraint`，其 `this` 为解析出的表达式。
        """
        self._match_text_seq("LENGTH")
        return self.expression(exp.InlineLengthColumnConstraint, this=self._parse_bitwise())

    def _parse_not_constraint(self) -> t.Optional[exp.Expression]:
        """解析以 NOT 起始的列级约束（如 NOT NULL、NOT CASESPECIFIC 等）。

        逻辑说明：
        - 依次尝试已支持的 NOT 后接关键字；
        - 若均不匹配，则回退一步“吐回”已消费的 NOT，交由上层继续处理。
        """
        if self._match_text_seq("NULL"):
            return self.expression(exp.NotNullColumnConstraint)
        if self._match_text_seq("CASESPECIFIC"):
            return self.expression(exp.CaseSpecificColumnConstraint, not_=True)
        if self._match_text_seq("FOR", "REPLICATION"):
            return self.expression(exp.NotForReplicationColumnConstraint)

        # 未识别的 NOT* 形态：回退以免误吞 NOT
        self._retreat(self._index - 1)
        return None

    def _parse_column_constraint(self) -> t.Optional[exp.Expression]:
        """解析列级约束（可带可选的 CONSTRAINT 名称）。

        关键点：
        - 若出现 `CONSTRAINT <name>`，捕获其名称到 `this`；
        - 为避免与过程选项冲突，若后续跟随 `WITH <PROCEDURE_OPTION>`，则不解析列约束；
        - 若匹配到已知的约束关键字，构造 `exp.ColumnConstraint`，其 `kind` 为对应解析结果；
        - 否则返回可能捕获到的 `this`（仅名称，无 kind）。
        """
        this = self._match(TokenType.CONSTRAINT) and self._parse_id_var()

        procedure_option_follows = (
            self._match(TokenType.WITH, advance=False)
            and self._next
            and self._next.text.upper() in self.PROCEDURE_OPTIONS
        )

        if not procedure_option_follows and self._match_texts(self.CONSTRAINT_PARSERS):
            return self.expression(
                exp.ColumnConstraint,
                this=this,
                kind=self.CONSTRAINT_PARSERS[self._prev.text.upper()](self),
            )

        return this

    def _parse_constraint(self) -> t.Optional[exp.Expression]:
        """解析命名/未命名的 schema 级约束。

        - 若未出现 `CONSTRAINT` 关键字，则尝试解析未命名约束；
        - 否则解析 `CONSTRAINT <name> <constraints...>` 并构造 `exp.Constraint`。
        """
        if not self._match(TokenType.CONSTRAINT):
            return self._parse_unnamed_constraint(constraints=self.SCHEMA_UNNAMED_CONSTRAINTS)

        return self.expression(
            exp.Constraint,
            this=self._parse_id_var(),
            expressions=self._parse_unnamed_constraints(),
        )

    def _parse_unnamed_constraints(self) -> t.List[exp.Expression]:
        """解析一组未命名约束，直到不再匹配。

        说明：
        - 每轮尝试 `_parse_unnamed_constraint()`，若失败再尝试函数解析（兼容部分方言把函数作为约束体）；
        - 无匹配则结束循环，返回收集到的约束列表。
        """
        constraints = []
        while True:
            constraint = self._parse_unnamed_constraint() or self._parse_function()
            if not constraint:
                break
            constraints.append(constraint)

        return constraints

    def _parse_unnamed_constraint(
        self, constraints: t.Optional[t.Collection[str]] = None
    ) -> t.Optional[exp.Expression]:
        """解析单个未命名约束。

        逻辑说明：
        - 若遇到标识符起始（容易是列名等）或不在允许集合中，则认为不是约束并返回 None；
        - 否则根据关键字选择对应解析器，若无解析器则抛出错误提示；
        - 成功时返回具体约束表达式。
        """
        if self._match(TokenType.IDENTIFIER, advance=False) or not self._match_texts(
            constraints or self.CONSTRAINT_PARSERS
        ):
            return None

        constraint = self._prev.text.upper()
        if constraint not in self.CONSTRAINT_PARSERS:
            self.raise_error(f"No parser found for schema constraint {constraint}.")

        return self.CONSTRAINT_PARSERS[constraint](self)

    def _parse_unique_key(self) -> t.Optional[exp.Expression]:
        """解析 UNIQUE 约束的键名（仅标识符）。"""
        return self._parse_id_var(any_token=False)

    def _parse_unique(self) -> exp.UniqueColumnConstraint:
        """解析 UNIQUE 列级约束（可带 KEY/INDEX、NULLS NOT DISTINCT、USING 等修饰）。"""
        self._match_texts(("KEY", "INDEX"))
        return self.expression(
            exp.UniqueColumnConstraint,
            # 某些方言允许声明 NULLS NOT DISTINCT 影响 NULL 的唯一性语义
            nulls=self._match_text_seq("NULLS", "NOT", "DISTINCT"),
            # 支持 `UNIQUE KEY (col1, col2)` 或 `UNIQUE (col)` 等，内部用 schema 封装列列表
            this=self._parse_schema(self._parse_unique_key()),
            # 允许 `USING <type>` 指定索引类型；通过先匹配 USING，再前进一步读类型标识
            index_type=self._match(TokenType.USING) and self._advance_any() and self._prev.text,
            # 解析 ON CONFLICT/ON DUPLICATE KEY 等冲突处理段
            on_conflict=self._parse_on_conflict(),
            # 解析额外的键级选项（如 ON UPDATE/DELETE、存储参数等，方言相关）
            options=self._parse_key_constraint_options(),
        )

    def _parse_key_constraint_options(self) -> t.List[str]:
        """解析键级约束的可选子句集合（ON ... ACTION/SET .../方言特定变量）。"""
        options = []
        while True:
            if not self._curr:
                break

            if self._match(TokenType.ON):
                # 解析形如 ON DELETE/UPDATE NO ACTION|CASCADE|RESTRICT|SET NULL|SET DEFAULT
                action = None
                on = self._advance_any() and self._prev.text

                if self._match_text_seq("NO", "ACTION"):
                    action = "NO ACTION"
                elif self._match_text_seq("CASCADE"):
                    action = "CASCADE"
                elif self._match_text_seq("RESTRICT"):
                    action = "RESTRICT"
                elif self._match_pair(TokenType.SET, TokenType.NULL):
                    action = "SET NULL"
                elif self._match_pair(TokenType.SET, TokenType.DEFAULT):
                    action = "SET DEFAULT"
                else:
                    self.raise_error("Invalid key constraint")

                options.append(f"ON {on} {action}")
            else:
                # 解析其它方言选项（在 KEY_CONSTRAINT_OPTIONS 列表中），不匹配即结束
                var = self._parse_var_from_options(
                    self.KEY_CONSTRAINT_OPTIONS, raise_unmatched=False
                )
                if not var:
                    break
                options.append(var.name)

        return options

    def _parse_references(self, match: bool = True) -> t.Optional[exp.Reference]:
        """解析 REFERENCES 子句（引用表与其键级选项）。

        - 当 `match=True` 且未匹配到 `REFERENCES` 时返回 None；
        - 解析被引用的表名（schema=True 以兼容限定名）；
        - 解析额外键级选项并封装为 `exp.Reference`。
        """
        if match and not self._match(TokenType.REFERENCES):
            return None

        expressions = None
        this = self._parse_table(schema=True)
        options = self._parse_key_constraint_options()
        return self.expression(exp.Reference, this=this, expressions=expressions, options=options)

    def _parse_foreign_key(self) -> exp.ForeignKey:
        """解析 FOREIGN KEY 约束，包括列列表、REFERENCES 与 ON 子句动作。"""
        # 若未立刻出现 REFERENCES，则解析被约束的本地列列表；否则列列表可能出现在稍后或省略
        expressions = (
            self._parse_wrapped_id_vars()
            if not self._match(TokenType.REFERENCES, advance=False)
            else None
        )
        reference = self._parse_references()
        on_options = {}

        # 解析 ON DELETE/UPDATE 等动作
        while self._match(TokenType.ON):
            if not self._match_set((TokenType.DELETE, TokenType.UPDATE)):
                self.raise_error("Expected DELETE or UPDATE")

            kind = self._prev.text.lower()

            if self._match_text_seq("NO", "ACTION"):
                action = "NO ACTION"
            elif self._match(TokenType.SET):
                self._match_set((TokenType.NULL, TokenType.DEFAULT))
                action = "SET " + self._prev.text.upper()
            else:
                self._advance()
                action = self._prev.text.upper()

            on_options[kind] = action

        return self.expression(
            exp.ForeignKey,
            expressions=expressions,
            reference=reference,
            options=self._parse_key_constraint_options(),
            **on_options,  # type: ignore
        )

    def _parse_primary_key_part(self) -> t.Optional[exp.Expression]:
        """解析主键的一部分：优先解析带排序信息的列，否则解析通用字段。"""
        return self._parse_ordered() or self._parse_field()

    def _parse_period_for_system_time(self) -> t.Optional[exp.PeriodForSystemTimeConstraint]:
        """解析 PERIOD FOR SYSTEM_TIME 约束。

        逻辑说明：
        - 需要匹配 `TIMESTAMP_SNAPSHOT`（方言 token），否则回退一步并返回 None；
        - 解析括号内两个标识（起止列），封装为 `exp.PeriodForSystemTimeConstraint`。
        """
        if not self._match(TokenType.TIMESTAMP_SNAPSHOT):
            # 未命中关键 token：回退以避免吞掉令牌
            self._retreat(self._index - 1)
            return None

        id_vars = self._parse_wrapped_id_vars()
        return self.expression(
            exp.PeriodForSystemTimeConstraint,
            this=seq_get(id_vars, 0),
            expression=seq_get(id_vars, 1),
        )

    def _parse_primary_key(
        self, wrapped_optional: bool = False, in_props: bool = False
    ) -> exp.PrimaryKeyColumnConstraint | exp.PrimaryKey:
        """解析 PRIMARY KEY，支持列级与表级两种形态。

        关键逻辑：
        - 先尝试读取紧随其后的 ASC/DESC 标记，仅当 DESC 才认为需要降序；
        - 若不在 props 中且未见 `(`，按列级主键返回（可附带 key 级选项）；
        - 否则解析括号内的主键列列表（允许 optional 表示括号可省略的方言），并构造表级主键；
        - 表级主键可携带 `INCLUDE`/索引参数以及其它键级选项。
        """
        desc = (
            self._match_set((TokenType.ASC, TokenType.DESC))
            and self._prev.token_type == TokenType.DESC
        )

        if not in_props and not self._match(TokenType.L_PAREN, advance=False):
            return self.expression(
                exp.PrimaryKeyColumnConstraint,
                desc=desc,
                options=self._parse_key_constraint_options(),
            )

        expressions = self._parse_wrapped_csv(
            self._parse_primary_key_part, optional=wrapped_optional
        )

        return self.expression(
            exp.PrimaryKey,
            expressions=expressions,
            include=self._parse_index_params(),
            options=self._parse_key_constraint_options(),
        )

    def _parse_bracket_key_value(self, is_map: bool = False) -> t.Optional[exp.Expression]:
        """解析中括号形态的 key/value（或切片）表达式。"""
        return self._parse_slice(self._parse_alias(self._parse_assignment(), explicit=True))

    def _parse_odbc_datetime_literal(self) -> exp.Expression:
        """解析 ODBC 风格的日期时间字面量。

        原说明：
        Parses a datetime column in ODBC format. We parse the column into the corresponding
        types, for example `{d'yyyy-mm-dd'}` will be parsed as a `Date` column, exactly the
        same as we did for `DATE('yyyy-mm-dd')`.

        参考：
        https://learn.microsoft.com/en-us/sql/odbc/reference/develop-app/date-time-and-timestamp-literals
        """
        self._match(TokenType.VAR)
        exp_class = self.ODBC_DATETIME_LITERALS[self._prev.text.lower()]
        expression = self.expression(exp_class=exp_class, this=self._parse_string())
        if not self._match(TokenType.R_BRACE):
            self.raise_error("Expected }")
        return expression

    def _parse_bracket(self, this: t.Optional[exp.Expression] = None) -> t.Optional[exp.Expression]:
        """解析方括号/花括号引导的构造：数组、结构体、下标/切片、以及 ODBC 字面量等。

        关键逻辑：
        - 首先匹配 `[` 或 `{`；`{` 并紧随 ODBC VAR 时转入 ODBC 日期时间字面量解析；
        - 根据 `MAP` 上下文决定是否将 `{k:v}` 的 key 保留为任意表达式（parse_map=True）；
        - 逗号分隔地解析内部元素，随后严格匹配对应右括号；
        - `{...}` 优先构造 `exp.Struct`，否则：
          - 若 `this` 为空则构造数组；
          - 若 `this` 命中已知构造器（如数组构造函数），使用专有构造逻辑；
          - 否则视为下标/切片，应用方言索引偏移并构造 `exp.Bracket`；
        - 最后合并注释并支持链式继续解析（连续括号）。
        """
        if not self._match_set((TokenType.L_BRACKET, TokenType.L_BRACE)):
            return this

        if self.MAP_KEYS_ARE_ARBITRARY_EXPRESSIONS:
            # 为 MAP 上下文启用任意表达式作为 key 的解析模式
            map_token = seq_get(self._tokens, self._index - 2)
            parse_map = map_token is not None and map_token.text.upper() == "MAP"
        else:
            parse_map = False

        bracket_kind = self._prev.token_type
        if (
            bracket_kind == TokenType.L_BRACE
            and self._curr
            and self._curr.token_type == TokenType.VAR
            and self._curr.text.lower() in self.ODBC_DATETIME_LITERALS
        ):
            # 命中 ODBC 风格字面量，例如 {d'2020-01-01'}
            return self._parse_odbc_datetime_literal()

        expressions = self._parse_csv(
            lambda: self._parse_bracket_key_value(is_map=bracket_kind == TokenType.L_BRACE)
        )

        # 严格匹配对应右括号，缺失则抛错
        if bracket_kind == TokenType.L_BRACKET and not self._match(TokenType.R_BRACKET):
            self.raise_error("Expected ]")
        elif bracket_kind == TokenType.L_BRACE and not self._match(TokenType.R_BRACE):
            self.raise_error("Expected }")

        # https://duckdb.org/docs/sql/data_types/struct.html#creating-structs
        if bracket_kind == TokenType.L_BRACE:
            # `{...}` 归一化为 Struct，并按 parse_map 决定是否保留原始 key 表达式
            this = self.expression(
                exp.Struct,
                expressions=self._kv_to_prop_eq(expressions=expressions, parse_map=parse_map),
            )
        elif not this:
            # 无前置对象：构造数组常量/列表
            this = build_array_constructor(
                exp.Array, args=expressions, bracket_kind=bracket_kind, dialect=self.dialect
            )
        else:
            # 存在前置对象：优先尝试已注册的数组构造器（如 ARRAY[...] 变体）
            constructor_type = self.ARRAY_CONSTRUCTORS.get(this.name.upper())
            if constructor_type:
                return build_array_constructor(
                    constructor_type,
                    args=expressions,
                    bracket_kind=bracket_kind,
                    dialect=self.dialect,
                )

            # 否则作为下标/切片处理，并根据方言设置校正索引偏移
            expressions = apply_index_offset(
                this, expressions, -self.dialect.INDEX_OFFSET, dialect=self.dialect
            )
            this = self.expression(
                exp.Bracket,
                this=this,
                expressions=expressions,
                comments=this.pop_comments(),
            )

        # 合并注释并支持连续括号链式解析（如 a[1][2]{k:v}）
        self._add_comments(this)
        return self._parse_bracket(this)

    def _parse_slice(self, this: t.Optional[exp.Expression]) -> t.Optional[exp.Expression]:
        """解析切片语法：`a[b:c]` 的 `b:c` 部分。"""
        if self._match(TokenType.COLON):
            return self.expression(exp.Slice, this=this, expression=self._parse_assignment())
        return this

    def _parse_case(self) -> t.Optional[exp.Expression]:
        """解析 CASE 表达式，兼容简单 CASE 与搜索型 CASE。

        逻辑说明：
        - 初始 `expression` 捕获简单 CASE 的比较基准（如 CASE x WHEN ...）；
        - 循环解析 `WHEN <cond> THEN <expr>` 子句列表，构造一组 If 表达式；
        - 可选解析 `ELSE <expr>`；
        - 严格要求 `END` 结尾：特殊处理 `ELSE INTERVAL END` 被误读为 Interval 的边缘情况；
        - 返回 `exp.Case`，携带注释、基准表达式、条件分支与默认值。
        """
        if self._match(TokenType.DOT, advance=False):
            # Avoid raising on valid expressions like case.*, supported by, e.g., spark & snowflake
            self._retreat(self._index - 1)
            return None
        ifs = []
        default = None

        comments = self._prev_comments
        expression = self._parse_assignment()

        while self._match(TokenType.WHEN):
            this = self._parse_assignment()
            self._match(TokenType.THEN)
            then = self._parse_assignment()
            ifs.append(self.expression(exp.If, this=this, true=then))

        if self._match(TokenType.ELSE):
            default = self._parse_assignment()

        if not self._match(TokenType.END):
            # 边界：当 ELSE 后解析为 Interval 且其 literal 为 "END"，应将 default 纠正为列名 interval
            if isinstance(default, exp.Interval) and default.this.sql().upper() == "END":
                default = exp.column("interval")
            else:
                self.raise_error("Expected END after CASE", self._prev)

        return self.expression(
            exp.Case, comments=comments, this=expression, ifs=ifs, default=default
        )

    def _parse_if(self) -> t.Optional[exp.Expression]:
        """解析 IF 表达式，支持带括号参数列与关键字分隔两种形态。

        关键逻辑：
        - 形参列表形态：`IF(<cond>, <then>, <else?>)`，以 CSV 方式解析并校验；
        - 关键字形态：`IF <cond> THEN <then> [ELSE <else>] END`；
        - 对于不满足条件的分支，注意回退索引以避免误吞 tokens；
        - 若启用 `NO_PAREN_IF_COMMANDS` 且在首位，解析为命令而非表达式。
        """
        if self._match(TokenType.L_PAREN):
            args = self._parse_csv(
                lambda: self._parse_alias(self._parse_assignment(), explicit=True)
            )
            this = self.validate_expression(exp.If.from_arg_list(args), args)
            self._match_r_paren()
        else:
            index = self._index - 1

            if self.NO_PAREN_IF_COMMANDS and index == 0:
                return self._parse_as_command(self._prev)

            condition = self._parse_assignment()

            if not condition:
                self._retreat(index)
                return None

            self._match(TokenType.THEN)
            true = self._parse_assignment()
            false = self._parse_assignment() if self._match(TokenType.ELSE) else None
            self._match(TokenType.END)
            this = self.expression(exp.If, this=condition, true=true, false=false)

        return this

    def _parse_next_value_for(self) -> t.Optional[exp.Expression]:
        """解析 `NEXT VALUE FOR` 序列取值表达式，支持可选的窗口 `OVER(...)`。"""
        if not self._match_text_seq("VALUE", "FOR"):
            # 未命中关键字序列：回退一步避免误吞上文 token
            self._retreat(self._index - 1)
            return None

        return self.expression(
            exp.NextValueFor,
            this=self._parse_column(),
            # 兼容部分方言在 NEXT VALUE FOR 上使用 OVER 窗口语法
            order=self._match(TokenType.OVER) and self._parse_wrapped(self._parse_order),
        )

    def _parse_extract(self) -> exp.Extract | exp.Anonymous:
        """解析 EXTRACT 表达式，支持 `EXTRACT(part FROM expr)` 与 `EXTRACT(part, expr)` 两形态。"""
        this = self._parse_function() or self._parse_var_or_string(upper=True)

        if self._match(TokenType.FROM):
            # 常见形态：EXTRACT(part FROM expr)
            return self.expression(exp.Extract, this=this, expression=self._parse_bitwise())

        if not self._match(TokenType.COMMA):
            self.raise_error("Expected FROM or comma after EXTRACT", self._prev)

        # 兼容形态：EXTRACT(part, expr)
        return self.expression(exp.Extract, this=this, expression=self._parse_bitwise())

    def _parse_gap_fill(self) -> exp.GapFill:
        """解析 GAP_FILL 语法：`GAP_FILL TABLE <table>, <lambda1>, <lambda2>, ...`。"""
        self._match(TokenType.TABLE)
        this = self._parse_table()

        self._match(TokenType.COMMA)
        # 将 TABLE 与后续一系列 lambda 一起作为参数列表供 from_arg_list 构造
        args = [this, *self._parse_csv(self._parse_lambda)]

        gap_fill = exp.GapFill.from_arg_list(args)
        return self.validate_expression(gap_fill, args)

    def _parse_cast(self, strict: bool, safe: t.Optional[bool] = None) -> exp.Expression:
        """解析 CAST/TRY_CAST 样式的类型转换，支持格式与默认值等扩展。

        关键逻辑：
        - 形如 `CAST(expr AS type)` 或 `CAST(expr, 'fmt')`（字符串目标类型）；
        - 若缺少 AS 而出现逗号，解析为 `CastToStrType`（目标类型为字符串字面量）；
        - 解析 `DEFAULT <expr> ON CONVERSION ERROR` 作为失败时默认值；
        - 支持 `FORMAT` 或逗号引导的格式字串，并在目标为时间类型时转为 `StrToDate/StrToTime`；
        - 当格式附带 `AT TIME ZONE` 时，若结果为 `StrToTime`，补充时区 `zone`；
        - 目标类型缺失时报错；标识符类型会按方言构造 UDT；CHAR 可携带 `CHARACTER SET`；
        - 最终调用 `build_cast` 统一构建表达式。
        """
        this = self._parse_assignment()

        if not self._match(TokenType.ALIAS):
            if self._match(TokenType.COMMA):
                return self.expression(exp.CastToStrType, this=this, to=self._parse_string())

            self.raise_error("Expected AS after CAST")

        fmt = None
        to = self._parse_types()

        default = self._match(TokenType.DEFAULT)
        if default:
            default = self._parse_bitwise()
            self._match_text_seq("ON", "CONVERSION", "ERROR")

        if self._match_set((TokenType.FORMAT, TokenType.COMMA)):
            fmt_string = self._parse_string()
            fmt = self._parse_at_time_zone(fmt_string)

            if not to:
                to = exp.DataType.build(exp.DataType.Type.UNKNOWN)
            if to.this in exp.DataType.TEMPORAL_TYPES:
                this = self.expression(
                    exp.StrToDate if to.this == exp.DataType.Type.DATE else exp.StrToTime,
                    this=this,
                    format=exp.Literal.string(
                        format_time(
                            fmt_string.this if fmt_string else "",
                            self.dialect.FORMAT_MAPPING or self.dialect.TIME_MAPPING,
                            self.dialect.FORMAT_TRIE or self.dialect.TIME_TRIE,
                        )
                    ),
                    safe=safe,
                )

                if isinstance(fmt, exp.AtTimeZone) and isinstance(this, exp.StrToTime):
                    this.set("zone", fmt.args["zone"])
                return this
        elif not to:
            self.raise_error("Expected TYPE after CAST")
        elif isinstance(to, exp.Identifier):
            to = exp.DataType.build(to.name, dialect=self.dialect, udt=True)
        elif to.this == exp.DataType.Type.CHAR:
            if self._match(TokenType.CHARACTER_SET):
                to = self.expression(exp.CharacterSet, this=self._parse_var_or_string())

        return self.build_cast(
            strict=strict,
            this=this,
            to=to,
            format=fmt,
            safe=safe,
            action=self._parse_var_from_options(self.CAST_ACTIONS, raise_unmatched=False),
            default=default,
        )

    def _parse_string_agg(self) -> exp.GroupConcat:
        """解析 STRING_AGG/LISTAGG：支持 DISTINCT、分隔符、ON OVERFLOW、ORDER/LIMIT、WITHIN GROUP 等。

        说明：
        - DISTINCT 形态：将第一个表达式包裹成 Distinct；
        - ON OVERFLOW：Trino 语法，ERROR 或 TRUNCATE '...' [WITH|WITHOUT COUNT]；
        - 若未立即闭合 `)`，解析 ORDER/LIMIT 并以 `this` 传递顺序，统一语义；
        - WITHIN GROUP(ORDER BY ...) 分支单独解析，以便与 MySQL/SQLite 的语义对齐便于转译。
        """
        if self._match(TokenType.DISTINCT):
            args: t.List[t.Optional[exp.Expression]] = [
                self.expression(exp.Distinct, expressions=[self._parse_assignment()])
            ]
            if self._match(TokenType.COMMA):
                args.extend(self._parse_csv(self._parse_assignment))
        else:
            args = self._parse_csv(self._parse_assignment)  # type: ignore

        if self._match_text_seq("ON", "OVERFLOW"):
            # trino: LISTAGG(expression [, separator] [ON OVERFLOW overflow_behavior])
            if self._match_text_seq("ERROR"):
                on_overflow: t.Optional[exp.Expression] = exp.var("ERROR")
            else:
                self._match_text_seq("TRUNCATE")
                on_overflow = self.expression(
                    exp.OverflowTruncateBehavior,
                    this=self._parse_string(),
                    with_count=(
                        self._match_text_seq("WITH", "COUNT")
                        or not self._match_text_seq("WITHOUT", "COUNT")
                    ),
                )
        else:
            on_overflow = None

        index = self._index
        if not self._match(TokenType.R_PAREN) and args:
            # postgres: STRING_AGG([DISTINCT] expression, separator [ORDER BY expression1 {ASC | DESC} [, ...]])
            # bigquery: STRING_AGG([DISTINCT] expression [, separator] [ORDER BY key [{ASC | DESC}] [, ... ]] [LIMIT n])
            # The order is parsed through `this` as a canonicalization for WITHIN GROUPs
            args[0] = self._parse_limit(this=self._parse_order(this=args[0]))
            return self.expression(exp.GroupConcat, this=args[0], separator=seq_get(args, 1))

        # Checks if we can parse an order clause: WITHIN GROUP (ORDER BY <order_by_expression_list> [ASC | DESC]).
        # This is done "manually", instead of letting _parse_window parse it into an exp.WithinGroup node, so that
        # the STRING_AGG call is parsed like in MySQL / SQLite and can thus be transpiled more easily to them.
        if not self._match_text_seq("WITHIN", "GROUP"):
            self._retreat(index)
            return self.validate_expression(exp.GroupConcat.from_arg_list(args), args)

        # The corresponding match_r_paren will be called in parse_function (caller)
        self._match_l_paren()

        return self.expression(
            exp.GroupConcat,
            this=self._parse_order(this=seq_get(args, 0)),
            separator=seq_get(args, 1),
            on_overflow=on_overflow,
        )

    def _parse_convert(
        self, strict: bool, safe: t.Optional[bool] = None
    ) -> t.Optional[exp.Expression]:
        """解析 CONVERT：`CONVERT(expr USING charset)` 或 `CONVERT(expr, type)`。

        说明：
        - USING 分支用于字符集转换，封装为 `exp.CharacterSet`；
        - 逗号分支解析目标类型；
        - 最终复用 `build_cast` 构造表达式，兼容严格/安全模式。
        """
        this = self._parse_bitwise()

        if self._match(TokenType.USING):
            to: t.Optional[exp.Expression] = self.expression(
                exp.CharacterSet, this=self._parse_var()
            )
        elif self._match(TokenType.COMMA):
            to = self._parse_types()
        else:
            to = None

        return self.build_cast(strict=strict, this=this, to=to, safe=safe)

    def _parse_xml_table(self) -> exp.XMLTable:
        """解析 XMLTABLE 调用：支持 XMLNAMESPACES、PASSING、RETURNING SEQUENCE BY REF、COLUMNS 等子句。"""
        namespaces = None
        passing = None
        columns = None

        if self._match_text_seq("XMLNAMESPACES", "("):
            namespaces = self._parse_xml_namespace()
            self._match_text_seq(")", ",")

        this = self._parse_string()

        if self._match_text_seq("PASSING"):
            # BY VALUE 关键字可选，仅用于语义清晰
            self._match_text_seq("BY", "VALUE")
            passing = self._parse_csv(self._parse_column)

        by_ref = self._match_text_seq("RETURNING", "SEQUENCE", "BY", "REF")

        if self._match_text_seq("COLUMNS"):
            columns = self._parse_csv(self._parse_field_def)

        return self.expression(
            exp.XMLTable,
            this=this,
            namespaces=namespaces,
            passing=passing,
            columns=columns,
            by_ref=by_ref,
        )

    def _parse_xml_namespace(self) -> t.List[exp.XMLNamespace]:
        """解析 XMLNAMESPACES 列表，支持 DEFAULT 与命名前缀两种形式。"""
        namespaces = []

        while True:
            if self._match(TokenType.DEFAULT):
                uri = self._parse_string()
            else:
                uri = self._parse_alias(self._parse_string())
            namespaces.append(self.expression(exp.XMLNamespace, this=uri))
            if not self._match(TokenType.COMMA):
                break

        return namespaces

    def _parse_decode(self) -> t.Optional[exp.Decode | exp.DecodeCase]:
        """解析 DECODE：当参数不足 3 个时视为字符集 DECODE，否则视为 CASE 风格。"""
        args = self._parse_csv(self._parse_assignment)

        if len(args) < 3:
            return self.expression(exp.Decode, this=seq_get(args, 0), charset=seq_get(args, 1))

        return self.expression(exp.DecodeCase, expressions=args)

    def _parse_json_key_value(self) -> t.Optional[exp.JSONKeyValue]:
        """解析 JSON 键值对：`KEY <col> <sep> VALUE <expr>`。

        说明：
        - 支持多种分隔符（见 `JSON_KEY_VALUE_SEPARATOR_TOKENS`）；
        - 若 `key` 与 `value` 同时缺失，返回 None，避免构造空对。
        """
        self._match_text_seq("KEY")
        key = self._parse_column()
        self._match_set(self.JSON_KEY_VALUE_SEPARATOR_TOKENS)
        self._match_text_seq("VALUE")
        value = self._parse_bitwise()

        if not key and not value:
            return None
        return self.expression(exp.JSONKeyValue, this=key, expression=value)

    def _parse_format_json(self, this: t.Optional[exp.Expression]) -> t.Optional[exp.Expression]:
        """解析 `FORMAT JSON` 修饰：将已有表达式标记为 JSON 格式。"""
        if not this or not self._match_text_seq("FORMAT", "JSON"):
            return this

        return self.expression(exp.FormatJson, this=this)

    def _parse_on_condition(self) -> t.Optional[exp.OnCondition]:
        """解析 JSON 系列的 ON 条件（EMPTY/ERROR/NULL 的处理策略）。

        说明：
        - 不同方言顺序不同：MySQL 通常 "X ON EMPTY Y ON ERROR"，Oracle 相反；
        - 分别解析 EMPTY/ERROR/NULL 的处理（如 RETURN NULL/ERROR/DEFAULT <expr>）。
        """
        # MySQL uses "X ON EMPTY Y ON ERROR" (e.g. JSON_VALUE) while Oracle uses the opposite (e.g. JSON_EXISTS)
        if self.dialect.ON_CONDITION_EMPTY_BEFORE_ERROR:
            empty = self._parse_on_handling("EMPTY", *self.ON_CONDITION_TOKENS)
            error = self._parse_on_handling("ERROR", *self.ON_CONDITION_TOKENS)
        else:
            error = self._parse_on_handling("ERROR", *self.ON_CONDITION_TOKENS)
            empty = self._parse_on_handling("EMPTY", *self.ON_CONDITION_TOKENS)

        null = self._parse_on_handling("NULL", *self.ON_CONDITION_TOKENS)

        if not empty and not error and not null:
            return None

        return self.expression(
            exp.OnCondition,
            empty=empty,
            error=error,
            null=null,
        )

    def _parse_on_handling(
        self, on: str, *values: str
    ) -> t.Optional[str] | t.Optional[exp.Expression]:
        """解析 "X ON <on>" 或 "DEFAULT <expr> ON <on>" 语法。

        逻辑说明：
        - 优先匹配固定子句，如 `NULL ON NULL`、`ERROR ON ERROR`；命中则返回字符串标记；
        - 否则尝试 `DEFAULT <expr> ON <on>`：成功则返回表达式；失败回退，避免误吞 tokens。
        """
        # Parses the "X ON Y" or "DEFAULT <expr> ON Y syntax, e.g. NULL ON NULL (Oracle, T-SQL, MySQL)
        for value in values:
            if self._match_text_seq(value, "ON", on):
                return f"{value} ON {on}"

        index = self._index
        if self._match(TokenType.DEFAULT):
            default_value = self._parse_bitwise()
            if self._match_text_seq("ON", on):
                return default_value

            self._retreat(index)

        return None

    @t.overload
    def _parse_json_object(self, agg: Lit[False]) -> exp.JSONObject: ...

    @t.overload
    def _parse_json_object(self, agg: Lit[True]) -> exp.JSONObjectAgg: ...

    def _parse_json_object(self, agg=False):
        """解析 JSON_OBJECT/JSON_OBJECTAGG。

        关键逻辑：
        - 支持 `*` 星号形式或逗号分隔的 `KEY ... VALUE ...` 列表（可带 FORMAT JSON 修饰）；
        - 解析 `NULL ON NULL`/`ABSENT ON NULL` 等 NULL 处理策略；
        - `WITH|WITHOUT UNIQUE KEYS` 控制键重复策略；
        - 可选的 `RETURNING <type>`（可带 FORMAT JSON）与 `ENCODING <var>`；
        - 根据 `agg` 选择构造 `JSONObject` 或 `JSONObjectAgg`。
        """
        star = self._parse_star()
        expressions = (
            [star]
            if star
            else self._parse_csv(lambda: self._parse_format_json(self._parse_json_key_value()))
        )
        null_handling = self._parse_on_handling("NULL", "NULL", "ABSENT")

        unique_keys = None
        if self._match_text_seq("WITH", "UNIQUE"):
            unique_keys = True
        elif self._match_text_seq("WITHOUT", "UNIQUE"):
            unique_keys = False

        self._match_text_seq("KEYS")

        return_type = self._match_text_seq("RETURNING") and self._parse_format_json(
            self._parse_type()
        )
        encoding = self._match_text_seq("ENCODING") and self._parse_var()

        return self.expression(
            exp.JSONObjectAgg if agg else exp.JSONObject,
            expressions=expressions,
            null_handling=null_handling,
            unique_keys=unique_keys,
            return_type=return_type,
            encoding=encoding,
        )

    # Note: this is currently incomplete; it only implements the "JSON_value_column" part
    def _parse_json_column_def(self) -> exp.JSONColumnDef:
        """解析 JSON 表列定义的一项（当前只实现 JSON_value_column 部分）。

        说明：
        - 支持 `NESTED` 前缀表示嵌套列；
        - 解析列名与类型（禁止标识符型类型，避免与列名冲突）；
        - 可选 `PATH 'jsonpath'`；
        - 若为嵌套列，则继续解析嵌套 schema。
        """
        if not self._match_text_seq("NESTED"):
            this = self._parse_id_var()
            ordinality = self._match_pair(TokenType.FOR, TokenType.ORDINALITY)
            kind = self._parse_types(allow_identifiers=False)
            nested = None
        else:
            this = None
            ordinality = None
            kind = None
            nested = True

        path = self._match_text_seq("PATH") and self._parse_string()
        nested_schema = nested and self._parse_json_schema()

        return self.expression(
            exp.JSONColumnDef,
            this=this,
            kind=kind,
            path=path,
            nested_schema=nested_schema,
            ordinality=ordinality,
        )

    def _parse_json_schema(self) -> exp.JSONSchema:
        """解析 JSON TABLE 的 COLUMNS 子句为 JSONSchema。"""
        self._match_text_seq("COLUMNS")
        return self.expression(
            exp.JSONSchema,
            expressions=self._parse_wrapped_csv(self._parse_json_column_def, optional=True),
        )

    def _parse_json_table(self) -> exp.JSONTable:
        """解析 JSON_TABLE：支持可选路径、错误与空值处理策略、以及 COLUMNS schema。"""
        this = self._parse_format_json(self._parse_bitwise())
        path = self._match(TokenType.COMMA) and self._parse_string()
        error_handling = self._parse_on_handling("ERROR", "ERROR", "NULL")
        empty_handling = self._parse_on_handling("EMPTY", "ERROR", "NULL")
        schema = self._parse_json_schema()

        return exp.JSONTable(
            this=this,
            schema=schema,
            path=path,
            error_handling=error_handling,
            empty_handling=empty_handling,
        )

    def _parse_match_against(self) -> exp.MatchAgainst:
        """解析 MySQL 风格 `MATCH(col1, col2) AGAINST('query' <modifier>)`。"""
        if self._match_text_seq("TABLE"):
            # parse SingleStore MATCH(TABLE ...) syntax
            # https://docs.singlestore.com/cloud/reference/sql-reference/full-text-search-functions/match/
            expressions = []
            table = self._parse_table()
            if table:
                expressions = [table]
        else:
            expressions = self._parse_csv(self._parse_column)

        self._match_text_seq(")", "AGAINST", "(")

        this = self._parse_string()

        if self._match_text_seq("IN", "NATURAL", "LANGUAGE", "MODE"):
            modifier = "IN NATURAL LANGUAGE MODE"
            if self._match_text_seq("WITH", "QUERY", "EXPANSION"):
                modifier = f"{modifier} WITH QUERY EXPANSION"
        elif self._match_text_seq("IN", "BOOLEAN", "MODE"):
            modifier = "IN BOOLEAN MODE"
        elif self._match_text_seq("WITH", "QUERY", "EXPANSION"):
            modifier = "WITH QUERY EXPANSION"
        else:
            modifier = None

        return self.expression(
            exp.MatchAgainst, this=this, expressions=expressions, modifier=modifier
        )

    # https://learn.microsoft.com/en-us/sql/t-sql/functions/openjson-transact-sql?view=sql-server-ver16
    def _parse_open_json(self) -> exp.OpenJSON:
        """解析 SQL Server 的 OPENJSON(expr [, path]) WITH (...) 语法。"""
        this = self._parse_bitwise()
        path = self._match(TokenType.COMMA) and self._parse_string()

        def _parse_open_json_column_def() -> exp.OpenJSONColumnDef:
            this = self._parse_field(any_token=True)
            kind = self._parse_types()
            path = self._parse_string()
            as_json = self._match_pair(TokenType.ALIAS, TokenType.JSON)

            return self.expression(
                exp.OpenJSONColumnDef, this=this, kind=kind, path=path, as_json=as_json
            )

        expressions = None
        if self._match_pair(TokenType.R_PAREN, TokenType.WITH):
            self._match_l_paren()
            expressions = self._parse_csv(_parse_open_json_column_def)

        return self.expression(exp.OpenJSON, this=this, path=path, expressions=expressions)

    def _parse_position(self, haystack_first: bool = False) -> exp.StrPosition:
        """解析 POSITION/STRPOS 等：确定子串 needle 在 haystack 中的位置。

        说明：
        - 支持 `POSITION(substr IN expr)` 语形；
        - 对于无 IN 的形态，根据方言/参数 `haystack_first` 决定参数顺序；
        - 可选第三个参数作为起始位置/偏移，存入 `position`。
        """
        args = self._parse_csv(self._parse_bitwise)

        if self._match(TokenType.IN):
            return self.expression(
                exp.StrPosition, this=self._parse_bitwise(), substr=seq_get(args, 0)
            )

        if haystack_first:
            haystack = seq_get(args, 0)
            needle = seq_get(args, 1)
        else:
            haystack = seq_get(args, 1)
            needle = seq_get(args, 0)

        return self.expression(
            exp.StrPosition, this=haystack, substr=needle, position=seq_get(args, 2)
        )

    def _parse_predict(self) -> exp.Predict:
        """解析 PREDICT MODEL 语法：`PREDICT MODEL <model_table>, TABLE <input_table> [, params]`。"""
        self._match_text_seq("MODEL")
        this = self._parse_table()

        self._match(TokenType.COMMA)
        self._match_text_seq("TABLE")

        return self.expression(
            exp.Predict,
            this=this,
            expression=self._parse_table(),
            params_struct=self._match(TokenType.COMMA) and self._parse_bitwise(),
        )

    def _parse_join_hint(self, func_name: str) -> exp.JoinHint:
        """解析 JOIN 提示函数（如 BROADCAST/SHUFFLE_JOIN 等），把名称标准化为大写。"""
        args = self._parse_csv(self._parse_table)
        return exp.JoinHint(this=func_name.upper(), expressions=args)

    def _parse_substring(self) -> exp.Substring:
        """解析 SUBSTRING，兼容 Postgres 形态：substring(string [from int] [for int])。"""
        # Postgres supports the form: substring(string [from int] [for int])
        # (despite being undocumented, the reverse order also works)
        # https://www.postgresql.org/docs/9.1/functions-string.html @ Table 9-6

        args = t.cast(t.List[t.Optional[exp.Expression]], self._parse_csv(self._parse_bitwise))

        start, length = None, None

        while self._curr:
            if self._match(TokenType.FROM):
                start = self._parse_bitwise()
            elif self._match(TokenType.FOR):
                if not start:
                    start = exp.Literal.number(1)
                length = self._parse_bitwise()
            else:
                break

        if start:
            args.append(start)
        if length:
            args.append(length)

        return self.validate_expression(exp.Substring.from_arg_list(args), args)

    def _parse_trim(self) -> exp.Trim:
        """解析 TRIM：支持 [LEADING|TRAILING|BOTH]、FROM/逗号顺序以及 COLLATE。"""
        # https://www.w3resource.com/sql/character-functions/trim.php
        # https://docs.oracle.com/javadb/10.8.3.0/ref/rreftrimfunc.html

        position = None
        collation = None
        expression = None

        if self._match_texts(self.TRIM_TYPES):
            # 捕获 TRIM 方向（LEADING/TRAILING/BOTH）
            position = self._prev.text.upper()

        this = self._parse_bitwise()
        if self._match_set((TokenType.FROM, TokenType.COMMA)):
            # 一些方言参数顺序为 TRIM(<pattern> FROM <string>)，另一些为 TRIM(<string>, <pattern>)
            invert_order = self._prev.token_type == TokenType.FROM or self.TRIM_PATTERN_FIRST
            expression = self._parse_bitwise()

            if invert_order:
                # 需要交换主串与模式/字符的顺序
                this, expression = expression, this

        if self._match(TokenType.COLLATE):
            # 可选的排序规则（校对规则）
            collation = self._parse_bitwise()

        return self.expression(
            exp.Trim, this=this, position=position, expression=expression, collation=collation
        )

    def _parse_window_clause(self) -> t.Optional[t.List[exp.Expression]]:
        """解析 WINDOW 子句，返回命名窗口定义列表或 None。"""
        return self._match(TokenType.WINDOW) and self._parse_csv(self._parse_named_window)

    def _parse_named_window(self) -> t.Optional[exp.Expression]:
        """解析命名窗口定义：`WINDOW x AS (...)` 中的单个窗口。"""
        return self._parse_window(self._parse_id_var(), alias=True)

    def _parse_respect_or_ignore_nulls(
        self, this: t.Optional[exp.Expression]
    ) -> t.Optional[exp.Expression]:
        """解析窗口函数上的 IGNORE/RESPECT NULLS 修饰。"""
        if self._match_text_seq("IGNORE", "NULLS"):
            return self.expression(exp.IgnoreNulls, this=this)
        if self._match_text_seq("RESPECT", "NULLS"):
            return self.expression(exp.RespectNulls, this=this)
        return this

    def _parse_having_max(self, this: t.Optional[exp.Expression]) -> t.Optional[exp.Expression]:
        """解析 HAVING MAX/MIN 修饰，用于某些窗口/聚合函数的简写变体。"""
        if self._match(TokenType.HAVING):
            self._match_texts(("MAX", "MIN"))
            max = self._prev.text.upper() != "MIN"
            return self.expression(
                exp.HavingMax, this=this, expression=self._parse_column(), max=max
            )

        return this

    def _parse_window(
        self, this: t.Optional[exp.Expression], alias: bool = False
    ) -> t.Optional[exp.Expression]:
        """解析窗口修饰（WITHIN GROUP/FILTER/IGNORE|RESPECT NULLS/OVER 以及窗口规范）。

        关键逻辑：
        - 允许在 WITHIN GROUP 后紧随 OVER（T-SQL 扩展）；
        - 支持 FILTER (WHERE ...) 子句，并在闭括号后继续；
        - 规范支持可选的 IGNORE/RESPECT NULLS，部分方言实现；
        - 对 FIRST_VALUE 等聚合函数，若已经在内部包含 IGNORE/RESPECT NULLS，则提升为外层修饰；
        - 解析 OVER：别名模式（BigQuery 的 WINDOW x AS (...)）或常规模式；
        - 解析窗口体：可选 FIRST/LAST、PARTITION BY/ORDER BY、ROWS/RANGE BETWEEN ... AND ...、EXCLUDE 子句；
        - 若后续仍有窗口起始 token，递归解析以覆盖 Oracle KEEP(...) OVER(...) 的链式结构。
        """
        func = this
        comments = func.comments if isinstance(func, exp.Expression) else None

        # T-SQL allows the OVER (...) syntax after WITHIN GROUP.
        # https://learn.microsoft.com/en-us/sql/t-sql/functions/percentile-disc-transact-sql?view=sql-server-ver16
        if self._match_text_seq("WITHIN", "GROUP"):
            order = self._parse_wrapped(self._parse_order)
            this = self.expression(exp.WithinGroup, this=this, expression=order)

        if self._match_pair(TokenType.FILTER, TokenType.L_PAREN):
            self._match(TokenType.WHERE)
            this = self.expression(
                exp.Filter, this=this, expression=self._parse_where(skip_where_token=True)
            )
            self._match_r_paren()

        # SQL spec defines an optional [ { IGNORE | RESPECT } NULLS ] OVER
        # Some dialects choose to implement and some do not.
        # https://dev.mysql.com/doc/refman/8.0/en/window-function-descriptions.html

        # There is some code above in _parse_lambda that handles
        #   SELECT FIRST_VALUE(TABLE.COLUMN IGNORE|RESPECT NULLS) OVER ...

        # The below changes handle
        #   SELECT FIRST_VALUE(TABLE.COLUMN) IGNORE|RESPECT NULLS OVER ...

        # Oracle allows both formats
        #   (https://docs.oracle.com/en/database/oracle/oracle-database/19/sqlrf/img_text/first_value.html)
        #   and Snowflake chose to do the same for familiarity
        #   https://docs.snowflake.com/en/sql-reference/functions/first_value.html#usage-notes
        if isinstance(this, exp.AggFunc):
            ignore_respect = this.find(exp.IgnoreNulls, exp.RespectNulls)

            if ignore_respect and ignore_respect is not this:
                ignore_respect.replace(ignore_respect.this)
                this = self.expression(ignore_respect.__class__, this=this)

        this = self._parse_respect_or_ignore_nulls(this)

        # bigquery select from window x AS (partition by ...)
        if alias:
            over = None
            self._match(TokenType.ALIAS)
        elif not self._match_set(self.WINDOW_BEFORE_PAREN_TOKENS):
            return this
        else:
            over = self._prev.text.upper()

        if comments and isinstance(func, exp.Expression):
            func.pop_comments()

        if not self._match(TokenType.L_PAREN):
            return self.expression(
                exp.Window,
                comments=comments,
                this=this,
                alias=self._parse_id_var(False),
                over=over,
            )

        window_alias = self._parse_id_var(any_token=False, tokens=self.WINDOW_ALIAS_TOKENS)

        first = self._match(TokenType.FIRST)
        if self._match_text_seq("LAST"):
            first = False

        partition, order = self._parse_partition_and_order()
        kind = self._match_set((TokenType.ROWS, TokenType.RANGE)) and self._prev.text

        if kind:
            self._match(TokenType.BETWEEN)
            start = self._parse_window_spec()

            end = self._parse_window_spec() if self._match(TokenType.AND) else {}
            exclude = (
                self._parse_var_from_options(self.WINDOW_EXCLUDE_OPTIONS)
                if self._match_text_seq("EXCLUDE")
                else None
            )

            spec = self.expression(
                exp.WindowSpec,
                kind=kind,
                start=start["value"],
                start_side=start["side"],
                end=end.get("value"),
                end_side=end.get("side"),
                exclude=exclude,
            )
        else:
            spec = None

        self._match_r_paren()

        window = self.expression(
            exp.Window,
            comments=comments,
            this=this,
            partition_by=partition,
            order=order,
            spec=spec,
            alias=window_alias,
            over=over,
            first=first,
        )

        # This covers Oracle's FIRST/LAST syntax: aggregate KEEP (...) OVER (...)
        if self._match_set(self.WINDOW_BEFORE_PAREN_TOKENS, advance=False):
            return self._parse_window(window, alias=alias)

        return window

    def _parse_partition_and_order(
        self,
    ) -> t.Tuple[t.List[exp.Expression], t.Optional[exp.Expression]]:
        """解析窗口的 PARTITION BY 与 ORDER BY 并返回二元组。"""
        return self._parse_partition_by(), self._parse_order()

    def _parse_window_spec(self) -> t.Dict[str, t.Optional[str | exp.Expression]]:
        """解析窗口的 BETWEEN 边界：返回边界值与方向（PRECEDING/FOLLOWING）。"""
        self._match(TokenType.BETWEEN)

        return {
            "value": (
                (self._match_text_seq("UNBOUNDED") and "UNBOUNDED")
                or (self._match_text_seq("CURRENT", "ROW") and "CURRENT ROW")
                or self._parse_type()
            ),
            # 方向：PRECEDING/FOLLOWING，若未匹配则为 None
            "side": self._match_texts(self.WINDOW_SIDES) and self._prev.text,
        }

    def _parse_alias(
        self, this: t.Optional[exp.Expression], explicit: bool = False
    ) -> t.Optional[exp.Expression]:
        """解析别名（AS/隐式），兼容别名列表与字符串别名。

        关键逻辑：
        - 某些方言中 LIMIT/OFFSET 可作关键字与标识符，此处先尝试按子句解析，否则回退为别名；
        - explicit=True 时必须存在 AS，否则直接返回；
        - 支持 `(a, b, ...)` 别名列表形式，解析为 `exp.Aliases`；
        - 支持字符串字面量作为别名（在 `STRING_ALIASES` 开启时）；
        - 将注释从原表达式移动到别名节点，保持注释靠近别名。
        """
        # In some dialects, LIMIT and OFFSET can act as both identifiers and keywords (clauses)
        # so this section tries to parse the clause version and if it fails, it treats the token
        # as an identifier (alias)
        if self._can_parse_limit_or_offset():
            return this

        any_token = self._match(TokenType.ALIAS)
        comments = self._prev_comments or []

        if explicit and not any_token:
            return this

        if self._match(TokenType.L_PAREN):
            aliases = self.expression(
                exp.Aliases,
                comments=comments,
                this=this,
                expressions=self._parse_csv(lambda: self._parse_id_var(any_token)),
            )
            self._match_r_paren(aliases)
            return aliases

        alias = self._parse_id_var(any_token, tokens=self.ALIAS_TOKENS) or (
            self.STRING_ALIASES and self._parse_string_as_identifier()
        )

        if alias:
            comments.extend(alias.pop_comments())
            this = self.expression(exp.Alias, comments=comments, this=this, alias=alias)
            column = this.this

            # Moves the comment next to the alias in `expr /* comment */ AS alias`
            if not this.comments and column and column.comments:
                this.comments = column.pop_comments()

        return this

    def _parse_id_var(
        self,
        any_token: bool = True,
        tokens: t.Optional[t.Collection[TokenType]] = None,
    ) -> t.Optional[exp.Expression]:
        """解析标识符/变量名：优先尝试标准标识符，否则按 token/任意 token 兜底。

        说明：
        - 先尝试 `_parse_identifier()`；
        - 若未命中，允许在 `any_token=True` 时推进一个 token 作为候选，或匹配指定 token 集；
        - 若上一步命中字符串 token，视为带引号的标识符处理（quoted=True）。
        """
        expression = self._parse_identifier()
        if not expression and (
            (any_token and self._advance_any()) or self._match_set(tokens or self.ID_VAR_TOKENS)
        ):
            quoted = self._prev.token_type == TokenType.STRING
            expression = self._identifier_expression(quoted=quoted)

        return expression

    def _parse_string(self) -> t.Optional[exp.Expression]:
        """解析字符串字面量：按注册解析器优先，否则回退占位符。"""
        if self._match_set(self.STRING_PARSERS):
            return self.STRING_PARSERS[self._prev.token_type](self, self._prev)
        return self._parse_placeholder()

    def _parse_string_as_identifier(self) -> t.Optional[exp.Identifier]:
        """将字符串字面量提升为带引号的标识符，用于支持字符串别名。"""
        output = exp.to_identifier(self._match(TokenType.STRING) and self._prev.text, quoted=True)
        if output:
            output.update_positions(self._prev)
        return output

    def _parse_number(self) -> t.Optional[exp.Expression]:
        """解析数字字面量：按注册解析器优先，否则回退占位符。"""
        if self._match_set(self.NUMERIC_PARSERS):
            return self.NUMERIC_PARSERS[self._prev.token_type](self, self._prev)
        return self._parse_placeholder()

    def _parse_identifier(self) -> t.Optional[exp.Expression]:
        """解析标识符：匹配 IDENTIFIER 并构造带引号的标识符表达。"""
        if self._match(TokenType.IDENTIFIER):
            return self._identifier_expression(quoted=True)
        return self._parse_placeholder()

    def _parse_var(
        self,
        any_token: bool = False,
        tokens: t.Optional[t.Collection[TokenType]] = None,
        upper: bool = False,
    ) -> t.Optional[exp.Expression]:
        """解析变量/标识类 token：支持任意 token 推进与大小写规整。"""
        if (
            (any_token and self._advance_any())
            or self._match(TokenType.VAR)
            or (self._match_set(tokens) if tokens else False)
        ):
            return self.expression(
                exp.Var, this=self._prev.text.upper() if upper else self._prev.text
            )
        return self._parse_placeholder()

    def _advance_any(self, ignore_reserved: bool = False) -> t.Optional[Token]:
        """推进一个非保留关键字 token；在 ignore_reserved=True 时放宽限制。"""
        if self._curr and (ignore_reserved or self._curr.token_type not in self.RESERVED_TOKENS):
            self._advance()
            return self._prev
        return None

    def _parse_var_or_string(self, upper: bool = False) -> t.Optional[exp.Expression]:
        """优先解析字符串，否则回退解析变量（可选大写化）。"""
        return self._parse_string() or self._parse_var(any_token=True, upper=upper)

    def _parse_primary_or_var(self) -> t.Optional[exp.Expression]:
        """优先解析 primary（更具体的原子表达式），否则回退为变量解析。"""
        return self._parse_primary() or self._parse_var(any_token=True)

    def _parse_null(self) -> t.Optional[exp.Expression]:

        """解析 NULL 字面量：按注册解析器处理，否则回退占位符。"""

        if self._match_set((TokenType.NULL, TokenType.UNKNOWN)):
            return self.PRIMARY_PARSERS[TokenType.NULL](self, self._prev)
        return self._parse_placeholder()

    def _parse_boolean(self) -> t.Optional[exp.Expression]:
        """解析布尔字面量 TRUE/FALSE。"""
        if self._match(TokenType.TRUE):
            return self.PRIMARY_PARSERS[TokenType.TRUE](self, self._prev)
        if self._match(TokenType.FALSE):
            return self.PRIMARY_PARSERS[TokenType.FALSE](self, self._prev)
        return self._parse_placeholder()

    def _parse_star(self) -> t.Optional[exp.Expression]:
        """解析 `*` 星号（通配列/所有列）。"""
        if self._match(TokenType.STAR):
            return self.PRIMARY_PARSERS[TokenType.STAR](self, self._prev)
        return self._parse_placeholder()

    def _parse_parameter(self) -> exp.Parameter:
        """解析参数节点：优先标识符，否则使用 primary/var。"""
        this = self._parse_identifier() or self._parse_primary_or_var()
        return self.expression(exp.Parameter, this=this)

    def _parse_placeholder(self) -> t.Optional[exp.Expression]:
        """按占位符解析器尝试解析，否则回退一位并返回 None。"""
        if self._match_set(self.PLACEHOLDER_PARSERS):
            placeholder = self.PLACEHOLDER_PARSERS[self._prev.token_type](self)
            if placeholder:
                return placeholder
            self._advance(-1)
        return None

    def _parse_star_op(self, *keywords: str) -> t.Optional[t.List[exp.Expression]]:
        """解析以关键字引导的操作：支持括号参数或单表达式。"""
        if not self._match_texts(keywords):
            return None
        if self._match(TokenType.L_PAREN, advance=False):
            return self._parse_wrapped_csv(self._parse_expression)

        expression = self._parse_alias(self._parse_assignment(), explicit=True)
        return [expression] if expression else None

    def _parse_csv(
        self, parse_method: t.Callable, sep: TokenType = TokenType.COMMA
    ) -> t.List[exp.Expression]:
        """以分隔符解析一串表达式，保持注释紧随前一项。

        说明：
        - 首先解析首项，若存在则纳入列表；
        - 循环匹配分隔符，每次把分隔符前的注释归并到上一项，再解析下一项；
        - 仅在解析方法非空时追加，避免产生 None 占位。
        """
        parse_result = parse_method()
        items = [parse_result] if parse_result is not None else []

        while self._match(sep):
            self._add_comments(parse_result)
            parse_result = parse_method()
            if parse_result is not None:
                items.append(parse_result)

        return items

    def _parse_tokens(
        self, parse_method: t.Callable, expressions: t.Dict
    ) -> t.Optional[exp.Expression]:
        """按给定 token→表达式映射，迭代构造链式表达。

        说明：
        - 先解析首个子表达式 `this`；
        - 只要后续 token 在映射表中出现，就将其包装为对应表达式类型，并继续解析右侧表达式；
        - 通过 `comments=self._prev_comments` 传递紧邻操作符处的注释。
        """
        this = parse_method()

        while self._match_set(expressions):
            this = self.expression(
                expressions[self._prev.token_type],
                this=this,
                comments=self._prev_comments,
                expression=parse_method(),
            )

        return this

    def _parse_wrapped_id_vars(self, optional: bool = False) -> t.List[exp.Expression]:
        """解析括号包裹的 id/var 列表：等价于 `(_parse_id_var, sep=,)`。"""
        return self._parse_wrapped_csv(self._parse_id_var, optional=optional)

    def _parse_wrapped_csv(
        self, parse_method: t.Callable, sep: TokenType = TokenType.COMMA, optional: bool = False
    ) -> t.List[exp.Expression]:
        """解析括号内以分隔符分隔的表达式列表，支持可选括号。"""
        return self._parse_wrapped(
            lambda: self._parse_csv(parse_method, sep=sep), optional=optional
        )

    def _parse_wrapped(self, parse_method: t.Callable, optional: bool = False) -> t.Any:
        """通用括号解析：若匹配到左括号则调用方法并匹配右括号；否则按 optional 决定是否报错。"""
        wrapped = self._match(TokenType.L_PAREN)
        if not wrapped and not optional:
            self.raise_error("Expecting (")
        parse_result = parse_method()
        if wrapped:
            self._match_r_paren()
        return parse_result

    def _parse_expressions(self) -> t.List[exp.Expression]:
        """解析通用表达式序列，逗号分隔。"""
        return self._parse_csv(self._parse_expression)

    def _parse_select_or_expression(self, alias: bool = False) -> t.Optional[exp.Expression]:
        """SELECT 优先，否则解析赋值表达式（可强制带别名）。"""

        return (
            self._parse_set_operations(
                self._parse_alias(self._parse_assignment(), explicit=True)
                if alias
                else self._parse_assignment()
            )
            or self._parse_select()
        )

    def _parse_ddl_select(self) -> t.Optional[exp.Expression]:
        """解析 DDL 语境下的 SELECT：开启 nested，禁止子查询别名解析，随后应用集合与修饰。"""
        return self._parse_query_modifiers(
            self._parse_set_operations(self._parse_select(nested=True, parse_subquery_alias=False))
        )

    def _parse_transaction(self) -> exp.Transaction | exp.Command:
        """解析事务语句：可选事务类型 + TRANSACTION/WORK + 模式列表。

        说明：
        - 事务类型来自 `TRANSACTION_KIND`（如 BEGIN/START）；
        - 支持 `TRANSACTION` 或 `WORK`；
        - 后续解析以逗号分隔的模式，每个模式由若干 VAR 组成并以空格连接。
        """
        this = None
        if self._match_texts(self.TRANSACTION_KIND):
            this = self._prev.text

        self._match_texts(("TRANSACTION", "WORK"))

        modes = []
        while True:
            mode = []
            while self._match(TokenType.VAR) or self._match(TokenType.NOT):
                mode.append(self._prev.text)

            if mode:
                modes.append(" ".join(mode))
            if not self._match(TokenType.COMMA):
                break

        return self.expression(exp.Transaction, this=this, modes=modes)

    def _parse_commit_or_rollback(self) -> exp.Commit | exp.Rollback:
        """解析 COMMIT/ROLLBACK 语句：支持 WORK/TRANSACTION、TO SAVEPOINT 与 [AND NO] CHAIN。"""
        chain = None
        savepoint = None
        is_rollback = self._prev.token_type == TokenType.ROLLBACK

        self._match_texts(("TRANSACTION", "WORK"))

        if self._match_text_seq("TO"):
            self._match_text_seq("SAVEPOINT")
            savepoint = self._parse_id_var()

        if self._match(TokenType.AND):
            chain = not self._match_text_seq("NO")
            self._match_text_seq("CHAIN")

        if is_rollback:
            return self.expression(exp.Rollback, savepoint=savepoint)

        return self.expression(exp.Commit, chain=chain)

    # 解析 REFRESH 语句：兼容 "REFRESH TABLE x"，TABLE 为可选关键字
    def _parse_refresh(self) -> exp.Refresh:
        # 若存在 TABLE 关键字则消费；用于兼容不同方言
        self._match(TokenType.TABLE)
        # 优先解析字符串（部分方言允许字符串目标），否则解析为表对象
        return self.expression(exp.Refresh, this=self._parse_string() or self._parse_table())

    # 解析带 [NOT] EXISTS 的列定义（常见于 ALTER TABLE ADD COLUMN ... IF NOT EXISTS）
    def _parse_column_def_with_exists(self):
        # 记录当前位置：如失败需回退，避免错误消费后续 token
        start = self._index
        # COLUMN 关键字在部分方言可选，存在则消费以提高兼容性
        self._match(TokenType.COLUMN)

        # 解析 [NOT] EXISTS 条件，用于条件性添加/变更
        exists_column = self._parse_exists(not_=True)
        # 解析字段定义，期望返回 ColumnDef
        expression = self._parse_field_def()

        # 若非 ColumnDef，说明当前分支不匹配：回退并返回 None
        if not isinstance(expression, exp.ColumnDef):
            self._retreat(start)
            return None

        # 将 EXISTS 条件挂载到列定义，便于后续生成器/优化器处理
        expression.set("exists", exists_column)

        return expression

    # 解析 ADD COLUMN 分支（要求前一个 token 已识别为 ADD）
    def _parse_add_column(self) -> t.Optional[exp.ColumnDef]:
        # 仅当紧随 ADD 后才尝试解析，避免与其他 ADD 分支混淆
        if not self._prev.text.upper() == "ADD":
            return None

        # 复用带 EXISTS 的列定义解析逻辑
        expression = self._parse_column_def_with_exists()
        if not expression:
            return None

        # Databricks Delta: 支持列位置（FIRST/AFTER 某列）
        # 若匹配到位置关键字，则构造列位置节点并附着到列定义
        # 生成 SQL 时可还原方言化列位置语义
        # https://docs.databricks.com/delta/update-schema.html#explicitly-update-schema-to-add-columns
        if self._match_texts(("FIRST", "AFTER")):
            position = self._prev.text
            column_position = self.expression(
                exp.ColumnPosition, this=self._parse_column(), position=position
            )
            expression.set("position", column_position)

        return expression

    # 解析 DROP COLUMN（或兼容其他 DROP 变体），统一为 kind=COLUMN
    def _parse_drop_column(self) -> t.Optional[exp.Drop | exp.Command]:
        # 匹配到 DROP 后再解析具体对象；否则返回 None
        drop = self._match(TokenType.DROP) and self._parse_drop()
        # 非通用 Command 时补齐 kind，保证 AST 语义明确
        if drop and not isinstance(drop, exp.Command):
            drop.set("kind", drop.args.get("kind", "COLUMN"))
        return drop

    # https://docs.aws.amazon.com/athena/latest/ug/alter-table-drop-partition.html
    # 解析 DROP PARTITION：支持 IF EXISTS 与多个分区的 CSV 形式
    def _parse_drop_partition(self, exists: t.Optional[bool] = None) -> exp.DropPartition:
        return self.expression(
            exp.DropPartition, expressions=self._parse_csv(self._parse_partition), exists=exists
        )

    # 解析 ALTER TABLE ... ADD ...（列/约束/分区等添加操作）
    def _parse_alter_table_add(self) -> t.List[exp.Expression]:
        def _parse_add_alteration() -> t.Optional[exp.Expression]:
            # 消费 ADD 关键字，随后分支解析具体对象
            self._match_text_seq("ADD")
            # 优先解析约束（避免与列定义产生歧义）
            if self._match_set(self.ADD_CONSTRAINT_TOKENS, advance=False):
                return self.expression(
                    exp.AddConstraint, expressions=self._parse_csv(self._parse_constraint)
                )

            # 其次尝试解析列定义（支持 [NOT] EXISTS）
            column_def = self._parse_add_column()
            if isinstance(column_def, exp.ColumnDef):
                return column_def

            # 解析 IF [NOT] EXISTS，用于分区添加
            exists = self._parse_exists(not_=True)
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

    # 解析 ALTER TABLE ... ALTER ...：优先分派至专用解析器，否则按 ALTER [COLUMN] 处理
    def _parse_alter_table_alter(self) -> t.Optional[exp.Expression]:
        # 若 ALTER 后紧跟已注册的关键字（如 RENAME/OWNER 等），使用对应解析器处理
        if self._match_texts(self.ALTER_ALTER_PARSERS):
            return self.ALTER_ALTER_PARSERS[self._prev.text.upper()](self)

        # 许多方言支持 ALTER [COLUMN] 语法；若无专用关键字，则默认解析列级变更
        # 尝试可选 COLUMN 关键字并解析列目标
        self._match(TokenType.COLUMN)
        column = self._parse_field(any_token=True)

        # DROP DEFAULT：删除列默认值
        if self._match_pair(TokenType.DROP, TokenType.DEFAULT):
            return self.expression(exp.AlterColumn, this=column, drop=True)
        # SET DEFAULT：为列设置默认值
        if self._match_pair(TokenType.SET, TokenType.DEFAULT):
            return self.expression(exp.AlterColumn, this=column, default=self._parse_assignment())
        # COMMENT：设置列注释
        if self._match(TokenType.COMMENT):
            return self.expression(exp.AlterColumn, this=column, comment=self._parse_string())
        # DROP NOT NULL：允许空值（以 drop+allow_null=True 表达）
        if self._match_text_seq("DROP", "NOT", "NULL"):
            return self.expression(
                exp.AlterColumn,
                this=column,
                drop=True,
                allow_null=True,
            )
        # SET NOT NULL：不允许空值
        if self._match_text_seq("SET", "NOT", "NULL"):
            return self.expression(
                exp.AlterColumn,
                this=column,
                allow_null=False,
            )

        # MySQL 方言：列可见性（VISIBLE/INVISIBLE）
        if self._match_text_seq("SET", "VISIBLE"):
            return self.expression(exp.AlterColumn, this=column, visible="VISIBLE")
        if self._match_text_seq("SET", "INVISIBLE"):
            return self.expression(exp.AlterColumn, this=column, visible="INVISIBLE")

        # SET DATA TYPE：修改列类型；可选 COLLATE 与 USING 子句用于附加语义
        self._match_text_seq("SET", "DATA")
        self._match_text_seq("TYPE")
        return self.expression(
            exp.AlterColumn,
            this=column,
            dtype=self._parse_types(),
            collate=self._match(TokenType.COLLATE) and self._parse_term(),
            using=self._match(TokenType.USING) and self._parse_assignment(),
        )

    # 解析 ALTER TABLE ... ALTER DISTSTYLE：Redshift 分布风格调整
    def _parse_alter_diststyle(self) -> exp.AlterDistStyle:
        # 简单枚举：ALL/EVEN/AUTO 直接作为模式，统一转为变量表达式
        if self._match_texts(("ALL", "EVEN", "AUTO")):
            return self.expression(exp.AlterDistStyle, this=exp.var(self._prev.text.upper()))

        # KEY DISTKEY <column>：键分布，需要指定分布键列
        self._match_text_seq("KEY", "DISTKEY")
        return self.expression(exp.AlterDistStyle, this=self._parse_column())

    # 解析 ALTER TABLE ... ALTER [COMPOUND] SORTKEY ...：排序键调整
    def _parse_alter_sortkey(self, compound: t.Optional[bool] = None) -> exp.AlterSortKey:
        # 当传入 compound=True 时，需要显式匹配 SORTKEY 关键字（方言差异）
        if compound:
            self._match_text_seq("SORTKEY")

        # 括号列表形式：解析括号内的列/变量列表
        if self._match(TokenType.L_PAREN, advance=False):
            return self.expression(
                exp.AlterSortKey, expressions=self._parse_wrapped_id_vars(), compound=compound
            )

        # 非列表形式：仅允许 AUTO/NONE 两种模式
        self._match_texts(("AUTO", "NONE"))
        return self.expression(
            exp.AlterSortKey, this=exp.var(self._prev.text.upper()), compound=compound
        )

    # 解析 ALTER TABLE ... DROP ...：根据是否为分区 DROP 或列 DROP 进行分支
    def _parse_alter_table_drop(self) -> t.List[exp.Expression]:
        # 记录当前位置：若不是分区分支需回退，避免消费多余 token
        index = self._index - 1

        # 解析可选的 IF EXISTS 语义（适用于 DROP PARTITION）
        partition_exists = self._parse_exists()
        # 若后续为 PARTITION，则按分区删除解析（可一次 CSV 删除多个分区）
        if self._match(TokenType.PARTITION, advance=False):
            return self._parse_csv(lambda: self._parse_drop_partition(exists=partition_exists))

        # 否则回退到 DROP 之前位置，按列删除解析（默认 kind=COLUMN 由下游逻辑处理）
        self._retreat(index)
        return self._parse_csv(self._parse_drop_column)

    # 解析 ALTER TABLE ... RENAME ...：支持改列名或改表名
    def _parse_alter_table_rename(self) -> t.Optional[exp.AlterRename | exp.RenameColumn]:
        # 若方言允许省略 COLUMN，或显式写了 COLUMN，则进入重命名列逻辑
        if self._match(TokenType.COLUMN) or not self.ALTER_RENAME_REQUIRES_COLUMN:
            exists = self._parse_exists()
            old_column = self._parse_column()
            to = self._match_text_seq("TO")
            new_column = self._parse_column()

            # 三要素（旧列、TO 关键字、新列）任一缺失即返回 None，交由上层处理
            if old_column is None or to is None or new_column is None:
                return None

            return self.expression(exp.RenameColumn, this=old_column, to=new_column, exists=exists)

        # 否则为重命名表：ALTER TABLE ... RENAME TO <new_table>
        self._match_text_seq("TO")
        return self.expression(exp.AlterRename, this=self._parse_table(schema=True))

    # 解析 ALTER TABLE ... SET ...：统一用 AlterSet 承载不同方言选项
    def _parse_alter_table_set(self) -> exp.AlterSet:
        alter_set = self.expression(exp.AlterSet)

        # ( ... ) 或 TABLE PROPERTIES (...)：通用属性赋值列表
        if self._match(TokenType.L_PAREN, advance=False) or self._match_text_seq(
            "TABLE", "PROPERTIES"
        ):
            alter_set.set("expressions", self._parse_wrapped_csv(self._parse_assignment))
        # FILESTREAM_ON：特定方言单项赋值
        elif self._match_text_seq("FILESTREAM_ON", advance=False):
            alter_set.set("expressions", [self._parse_assignment()])
        # LOGGED / UNLOGGED：开关类选项，直接提升为变量
        elif self._match_texts(("LOGGED", "UNLOGGED")):
            alter_set.set("option", exp.var(self._prev.text.upper()))
        # WITHOUT CLUSTER / WITHOUT OIDS：两词组合，保留 WITHOUT 前缀以表达完整语义
        elif self._match_text_seq("WITHOUT") and self._match_texts(("CLUSTER", "OIDS")):
            alter_set.set("option", exp.var(f"WITHOUT {self._prev.text.upper()}"))
        # LOCATION/ACCESS METHOD/TABLESPACE：存储位置/访问方法/表空间
        elif self._match_text_seq("LOCATION"):
            alter_set.set("location", self._parse_field())
        elif self._match_text_seq("ACCESS", "METHOD"):
            alter_set.set("access_method", self._parse_field())
        elif self._match_text_seq("TABLESPACE"):
            alter_set.set("tablespace", self._parse_field())
        # FILE FORMAT：文件格式可为单字段或包装选项（Snowflake STAGE_*）
        elif self._match_text_seq("FILE", "FORMAT") or self._match_text_seq("FILEFORMAT"):
            alter_set.set("file_format", [self._parse_field()])
        elif self._match_text_seq("STAGE_FILE_FORMAT"):
            alter_set.set("file_format", self._parse_wrapped_options())
        # 复制选项（Snowflake）：包装选项列表
        elif self._match_text_seq("STAGE_COPY_OPTIONS"):
            alter_set.set("copy_options", self._parse_wrapped_options())
        # TAG/TAGS：标签赋值列表
        elif self._match_text_seq("TAG") or self._match_text_seq("TAGS"):
            alter_set.set("tag", self._parse_csv(self._parse_assignment))
        else:
            # Hive SERDE：自定义序列化/反序列化库
            if self._match_text_seq("SERDE"):
                alter_set.set("serde", self._parse_field())

            # 其他 PROPERTIES 形式：包一层 expressions 以统一结构
            properties = self._parse_wrapped(self._parse_properties, optional=True)
            alter_set.set("expressions", [properties])

        return alter_set


    def _parse_alter_session(self) -> exp.AlterSession:
        """Parse ALTER SESSION SET/UNSET statements."""
        if self._match(TokenType.SET):
            expressions = self._parse_csv(lambda: self._parse_set_item_assignment())
            return self.expression(exp.AlterSession, expressions=expressions, unset=False)

        self._match_text_seq("UNSET")
        expressions = self._parse_csv(
            lambda: self.expression(exp.SetItem, this=self._parse_id_var(any_token=True))
        )
        return self.expression(exp.AlterSession, expressions=expressions, unset=True)

    # 解析 ALTER 顶层：匹配可被 ALTER 的对象并派发到具体动作解析器
    def _parse_alter(self) -> exp.Alter | exp.Command:
        start = self._prev

        # 先匹配 ALTER 的对象类型（如 TABLE/INDEX 等）；匹配失败则按通用命令处理
        alter_token = self._match_set(self.ALTERABLES) and self._prev
        if not alter_token:
            return self._parse_as_command(start)

        # 常见修饰：IF EXISTS、ONLY、表/模式名、可选 ON <cluster>（特定方言）
        exists = self._parse_exists()
        only = self._match_text_seq("ONLY")

        # 若存在下一个 token，前进一位以读取动作关键字（如 ADD/DROP/RENAME/SET）
        if alter_token.token_type == TokenType.SESSION:
            this = None
            check = None
            cluster = None
        else:
            this = self._parse_table(schema=True, parse_partition=self.ALTER_TABLE_PARTITIONS)
            check = self._match_text_seq("WITH", "CHECK")
            cluster = self._parse_on_property() if self._match(TokenType.ON) else None

            if self._next:
                self._advance()

        # 根据动作关键字选择解析器
        parser = self.ALTER_PARSERS.get(self._prev.text.upper()) if self._prev else None
        if parser:
            # 统一将单个或多个动作包装为列表，便于上层消费
            actions = ensure_list(parser(self))
            # 可选修饰：NOT VALID（如约束验证行为）
            not_valid = self._match_text_seq("NOT", "VALID")
            # 额外选项列表：以逗号分隔的属性
            options = self._parse_csv(self._parse_property)
            cascade = self.dialect.ALTER_TABLE_SUPPORTS_CASCADE and self._match_text_seq("CASCADE")

            # 只有当语句已到末尾（not self._curr）且成功解析出动作时才构建 Alter AST
            # 这样可避免将部分解析的 ALTER 误当作完整语句
            if not self._curr and actions:
                return self.expression(
                    exp.Alter,
                    this=this,
                    kind=alter_token.text.upper(),
                    exists=exists,
                    actions=actions,
                    only=only,
                    options=options,
                    cluster=cluster,
                    not_valid=not_valid,
                    check=check,
                    cascade=cascade,
                )

        # 若未找到对应解析器或未满足完整性条件，则回退到通用命令处理
        return self._parse_as_command(start)

    # 解析 ANALYZE 语句：覆盖多方言（DuckDB/Presto/StarRocks 等）的语法分支
    def _parse_analyze(self) -> exp.Analyze | exp.Command:
        start = self._prev  # 记录起始 token，便于必要时回退为通用命令解析
        # https://duckdb.org/docs/sql/statements/analyze
        # DuckDB：允许裸 ANALYZE（无后续 token），直接构建空 Analyze 节点
        if not self._curr:
            return self.expression(exp.Analyze)

        # 解析可选的 ANALYZE 风格/选项；BUFFER_USAGE_LIMIT 后需紧跟数值
        options = []
        while self._match_texts(self.ANALYZE_STYLES):
            if self._prev.text.upper() == "BUFFER_USAGE_LIMIT":
                options.append(f"BUFFER_USAGE_LIMIT {self._parse_number()}")
            else:
                options.append(self._prev.text.upper())

        # this：Analyze 的目标对象（表/索引/库等）；inner_expression：特定表达式子句
        this: t.Optional[exp.Expression] = None
        inner_expression: t.Optional[exp.Expression] = None

        # kind：关键字类别（如 TABLE / TABLES FROM / DATABASE 等），用于还原具体语义
        kind = self._curr and self._curr.text.upper()

        # 按优先级匹配不同目标：TABLE/INDEX/TABLES FROM|IN/DATABASE/CLUSTER
        if self._match(TokenType.TABLE) or self._match(TokenType.INDEX):
            this = self._parse_table_parts()
        elif self._match_text_seq("TABLES"):
            # TABLES 后常跟 FROM/IN 指定库；需要把 FROM/IN 拼接到 kind 以保留语义
            if self._match_set((TokenType.FROM, TokenType.IN)):
                kind = f"{kind} {self._prev.text.upper()}"
                this = self._parse_table(schema=True, is_db_reference=True)
        elif self._match_text_seq("DATABASE"):
            this = self._parse_table(schema=True, is_db_reference=True)
        elif self._match_text_seq("CLUSTER"):
            this = self._parse_table()
        # 在回退到解析表之前，优先尝试匹配内层表达式（如特定统计项）
        elif self._match_texts(self.ANALYZE_EXPRESSION_PARSERS):
            kind = None
            inner_expression = self.ANALYZE_EXPRESSION_PARSERS[self._prev.text.upper()](self)
        else:
            # Presto 允许空 kind（ANALYZE <table>）；此时仅解析表名
            # https://prestodb.io/docs/current/sql/analyze.html
            kind = None
            this = self._parse_table_parts()

        # 尝试解析分区；若失败但出现了分区关键字，说明应走方言命令分支，回退处理
        partition = self._try_parse(self._parse_partition)
        if not partition and self._match_texts(self.PARTITION_KEYWORDS):
            return self._parse_as_command(start)

        # StarRocks：支持 WITH SYNC/ASYNC MODE，需保留到 mode 字段
        # https://docs.starrocks.io/docs/sql-reference/sql-statements/cbo_stats/ANALYZE_TABLE/
        if self._match_text_seq("WITH", "SYNC", "MODE") or self._match_text_seq(
            "WITH", "ASYNC", "MODE"
        ):
            mode = f"WITH {self._tokens[self._index - 2].text.upper()} MODE"
        else:
            mode = None

        # 再次尝试匹配可能出现于末尾的内层表达式（与前面的优先级保持一致）
        if self._match_texts(self.ANALYZE_EXPRESSION_PARSERS):
            inner_expression = self.ANALYZE_EXPRESSION_PARSERS[self._prev.text.upper()](self)

        # 解析 WITH(...) / 其他属性键值对
        properties = self._parse_properties()
        return self.expression(
            exp.Analyze,
            kind=kind,
            this=this,
            mode=mode,
            partition=partition,
            properties=properties,
            expression=inner_expression,
            options=options,
        )

    # https://spark.apache.org/docs/3.5.1/sql-ref-syntax-aux-analyze-table.html
    # 解析 Spark 风格的 ANALYZE ... STATISTICS：支持 NOSCAN / FOR COLUMNS / SAMPLE 等
    def _parse_analyze_statistics(self) -> exp.AnalyzeStatistics:
        this = None
        # kind 记录当前关键字（如 ANALYZE TABLE/VIEW），用于还原语义
        kind = self._prev.text.upper()
        # 可选 DELTA 选项（Delta Lake），需要保留到 option 字段
        option = self._prev.text.upper() if self._match_text_seq("DELTA") else None
        expressions = []

        # Spark 语法必须带 STATISTICS；若没有则报错，避免错误吞掉 token
        if not self._match_text_seq("STATISTICS"):
            self.raise_error("Expecting token STATISTICS")

        # NOSCAN：不扫描数据，仅基于元数据刷新统计
        if self._match_text_seq("NOSCAN"):
            this = "NOSCAN"
        # FOR [ALL] COLUMNS / FOR COLUMNS (<col, ...>)：对列级统计
        elif self._match(TokenType.FOR):
            if self._match_text_seq("ALL", "COLUMNS"):
                this = "FOR ALL COLUMNS"
            if self._match_texts("COLUMNS"):
                this = "FOR COLUMNS"
                # 允许解析列列表，逗号分隔
                expressions = self._parse_csv(self._parse_column_reference)
        # SAMPLE <n> [PERCENT]：采样统计；若跟随 PERCENT，标记为百分比
        elif self._match_text_seq("SAMPLE"):
            sample = self._parse_number()
            expressions = [
                self.expression(
                    exp.AnalyzeSample,
                    sample=sample,
                    kind=self._prev.text.upper() if self._match(TokenType.PERCENT) else None,
                )
            ]

        return self.expression(
            exp.AnalyzeStatistics, kind=kind, option=option, this=this, expressions=expressions
        )

    # https://docs.oracle.com/en/database/oracle/oracle-database/21/sqlrf/ANALYZE.html
    # 解析 Oracle 风格的 VALIDATE 分支：REF UPDATE / STRUCTURE ...
    def _parse_analyze_validate(self) -> exp.AnalyzeValidate:
        kind = None
        this = None
        expression: t.Optional[exp.Expression] = None
        # REF UPDATE [SET DANGLING TO NULL]
        if self._match_text_seq("REF", "UPDATE"):
            kind = "REF"
            this = "UPDATE"
            # 可选：将悬空引用置为空
            if self._match_text_seq("SET", "DANGLING", "TO", "NULL"):
                this = "UPDATE SET DANGLING TO NULL"
        # STRUCTURE [CASCADE FAST | CASCADE COMPLETE (ONLINE|OFFLINE)] [INTO ...]
        elif self._match_text_seq("STRUCTURE"):
            kind = "STRUCTURE"
            if self._match_text_seq("CASCADE", "FAST"):
                this = "CASCADE FAST"
            elif self._match_text_seq("CASCADE", "COMPLETE") and self._match_texts(
                ("ONLINE", "OFFLINE")
            ):
                # 记录 ONLINE/OFFLINE 模式，并允许跟随 INTO 目标
                this = f"CASCADE COMPLETE {self._prev.text.upper()}"
                expression = self._parse_into()

        return self.expression(exp.AnalyzeValidate, kind=kind, this=this, expression=expression)

    # 解析列相关的附加关键字：如 "FOR COLUMNS" 的变体拼接
    def _parse_analyze_columns(self) -> t.Optional[exp.AnalyzeColumns]:
        this = self._prev.text.upper()
        if self._match_text_seq("COLUMNS"):
            # 将前一关键字与 COLUMNS 拼接保留原始语义
            return self.expression(exp.AnalyzeColumns, this=f"{this} {self._prev.text.upper()}")
        return None

    # 解析删除统计：如 ANALYZE DELETE [SYSTEM] STATISTICS
    def _parse_analyze_delete(self) -> t.Optional[exp.AnalyzeDelete]:
        # 可选 SYSTEM 修饰
        kind = self._prev.text.upper() if self._match_text_seq("SYSTEM") else None
        if self._match_text_seq("STATISTICS"):
            return self.expression(exp.AnalyzeDelete, kind=kind)
        return None

    # 解析链式行列表：ANALYZE LIST CHAINED ROWS [INTO ...]
    def _parse_analyze_list(self) -> t.Optional[exp.AnalyzeListChainedRows]:
        if self._match_text_seq("CHAINED", "ROWS"):
            # INTO 子句指定导出目标
            return self.expression(exp.AnalyzeListChainedRows, expression=self._parse_into())
        return None

    # https://dev.mysql.com/doc/refman/8.4/en/analyze-table.html
    # 解析直方图统计：MySQL 及兼容方言（支持 WITH ... MODE / BUCKETS / USING DATA）
    def _parse_analyze_histogram(self) -> exp.AnalyzeHistogram:
        this = self._prev.text.upper()
        expression: t.Optional[exp.Expression] = None
        expressions = []
        update_options = None

        if self._match_text_seq("HISTOGRAM", "ON"):
            # 解析列列表：HISTOGRAM ON <col, ...>
            expressions = self._parse_csv(self._parse_column_reference)
            with_expressions = []
            while self._match(TokenType.WITH):
                # StarRocks 扩展：WITH SYNC/ASYNC MODE
                # https://docs.starrocks.io/docs/sql-reference/sql-statements/cbo_stats/ANALYZE_TABLE/
                if self._match_texts(("SYNC", "ASYNC")):
                    if self._match_text_seq("MODE", advance=False):
                        with_expressions.append(f"{self._prev.text.upper()} MODE")
                        self._advance()
                else:
                    # WITH <n> BUCKETS：指定直方图桶数
                    buckets = self._parse_number()
                    if self._match_text_seq("BUCKETS"):
                        with_expressions.append(f"{buckets} BUCKETS")
            if with_expressions:
                expression = self.expression(exp.AnalyzeWith, expressions=with_expressions)

            # UPDATE MANUAL/AUTO：更新策略；需要在看到 UPDATE 后读取前一项
            if self._match_texts(("MANUAL", "AUTO")) and self._match(
                TokenType.UPDATE, advance=False
            ):
                update_options = self._prev.text.upper()
                self._advance()
            # USING DATA 'path'：从数据样本构建直方图
            elif self._match_text_seq("USING", "DATA"):
                expression = self.expression(exp.UsingData, this=self._parse_string())

        return self.expression(
            exp.AnalyzeHistogram,
            this=this,
            expressions=expressions,
            expression=expression,
            update_options=update_options,
        )

    # 解析 MERGE 语句：MERGE INTO <target> USING <source> ON <condition> WHEN ...
    def _parse_merge(self) -> exp.Merge:
        # MERGE INTO：匹配 INTO 后解析目标表名
        self._match(TokenType.INTO)
        target = self._parse_table()

        # 目标表可带别名：有助于后续 WHEN 子句中引用目标/源字段
        if target and self._match(TokenType.ALIAS, advance=False):
            target.set("alias", self._parse_table_alias())

        # USING <source>：解析源表/子查询
        self._match(TokenType.USING)
        using = self._parse_table()

        # ON <condition>：连接条件；此处用通用 assignment 解析，兼容复杂表达式
        # 收集 WHEN 分支与可选 RETURNING 子句，构造 Merge AST
        return self.expression(
            exp.Merge,
            this=target,
            using=using,
            on=self._match(TokenType.ON) and self._parse_assignment(),
            using_cond=self._match(TokenType.USING) and self._parse_using_identifiers(),
            whens=self._parse_when_matched(),
            returning=self._parse_returning(),
        )

    # 解析 WHEN 分支：支持 WHEN [NOT] MATCHED [BY TARGET|BY SOURCE] [AND <cond>] THEN <action>
    def _parse_when_matched(self) -> exp.Whens:
        whens = []

        while self._match(TokenType.WHEN):
            # matched=True 表示 MATCHED；若出现 NOT 则为未匹配分支
            matched = not self._match(TokenType.NOT)
            self._match_text_seq("MATCHED")
            # BY TARGET / BY SOURCE：标记触发来源（目标侧或源侧），缺省由 else 分支处理 SOURCE
            source = (
                False
                if self._match_text_seq("BY", "TARGET")
                else self._match_text_seq("BY", "SOURCE")
            )
            # 可选 AND <condition>：额外过滤条件，仅在出现 AND 时解析
            condition = self._parse_assignment() if self._match(TokenType.AND) else None

            self._match(TokenType.THEN)

            # THEN INSERT ...
            if self._match(TokenType.INSERT):
                this = self._parse_star()
                if this:
                    # INSERT *：星号场景直接封装 Insert 节点
                    then: t.Optional[exp.Expression] = self.expression(exp.Insert, this=this)
                else:
                    # INSERT ROW 或 INSERT (<cols>) VALUES (...)
                    insert_columns: t.Optional[exp.Expression] = None
                    if self._match_text_seq("ROW"):
                        insert_columns = exp.var("ROW")
                    elif self._curr and self._curr.token_type == TokenType.L_PAREN:
                        # Only treat leading parenthesis as column tuple; avoid
                        # consuming VALUES/SELECT as part of the column list.
                        insert_columns = self._parse_value(values=False)
                    overriding: t.Optional[str] = None
                    if self.MERGE_INSERT_OVERRIDING_SUPPORTED and self._match_text_seq("OVERRIDING"):
                        if self._match_texts(("SYSTEM", "USER")):
                            overriding = self._prev.text.upper()
                            self._match_text_seq("VALUE")
                        else:
                            # If OVERRIDING is present but malformed, treat as unsupported by falling through
                            overriding = None

                    # DEFAULT VALUES
                    if self.MERGE_INSERT_DEFAULT_VALUES_SUPPORTED and self._match_text_seq("DEFAULT", "VALUES"):
                        then = self.expression(
                            exp.Insert,
                            this=insert_columns,
                            default=True,
                            **({"overriding": exp.var(overriding)} if overriding else {}),
                        )
                    else:
                        insert_expression: t.Optional[exp.Expression] = None
                        if self._match_text_seq("VALUES"):
                            insert_expression = self._parse_value()
                        else:
                            # Allow dialects that support SELECT/CTE sources to be parsed as a Subquery.
                            select_expr: t.Optional[exp.Expression] = None
                            if self._curr and self._curr.token_type == TokenType.L_PAREN:
                                select_expr = self._parse_wrapped(self._parse_ddl_select)
                            elif self._curr and self._curr.token_type in (TokenType.SELECT, TokenType.WITH):
                                select_expr = self._parse_ddl_select()
                            if select_expr is not None:
                                insert_expression = self.expression(exp.Subquery, this=select_expr)

                        then = self.expression(
                            exp.Insert,
                            this=insert_columns,
                            expression=insert_expression,
                            **({"overriding": exp.var(overriding)} if overriding else {}),
                        )

                    if self.MERGE_INSERT_WHERE_SUPPORTED and isinstance(then, exp.Insert):
                        where = self._parse_where()
                        if where is not None:
                            then.set("where", where)
            # THEN UPDATE ...
            elif self._match(TokenType.UPDATE):
                expressions = self._parse_star()
                if expressions:
                    # UPDATE *：允许使用星号批量指定
                    then = self.expression(exp.Update, expressions=expressions)
                else:
                    # UPDATE SET a=b, c=d ...：仅在出现 SET 时解析等式列表
                    then = self.expression(
                        exp.Update,
                        expressions=self._match(TokenType.SET)
                        and self._parse_csv(self._parse_equality),
                    )
            # THEN DELETE
            elif self._match(TokenType.DELETE):
                # DELETE 无子结构，使用 Var 表达指令本身
                then = self.expression(exp.Var, this=self._prev.text)
            else:
                # 其他冲突动作（方言扩展）：从允许集合中解析
                then = self._parse_var_from_options(self.CONFLICT_ACTIONS)

            whens.append(
                self.expression(
                    exp.When,
                    matched=matched,
                    source=source,
                    condition=condition,
                    then=then,
                )
            )
        # 汇总多个 WHEN 子句
        return self.expression(exp.Whens, expressions=whens)

    # 解析 SHOW：基于 trie 分派到具体 SHOW_* 解析器，失败则退化为通用命令
    def _parse_show(self) -> t.Optional[exp.Expression]:
        parser = self._find_parser(self.SHOW_PARSERS, self.SHOW_TRIE)
        if parser:
            return parser(self)
        return self._parse_as_command(self._prev)

    # 解析 SET 单项赋值：支持 GLOBAL/SESSION TRANSACTION，及 name = value / name TO value 等方言
    def _parse_set_item_assignment(
        self, kind: t.Optional[str] = None
    ) -> t.Optional[exp.Expression]:
        # 记录当前位置：若后续判断失败需回退，避免错误消费 token
        index = self._index

        # 当为 GLOBAL/SESSION 前缀且跟随 TRANSACTION 时，进入事务特性解析
        if kind in ("GLOBAL", "SESSION") and self._match_text_seq("TRANSACTION"):
            return self._parse_set_transaction(global_=kind == "GLOBAL")

        # 左侧解析：优先解析通用主表达式，否则降级为列（标识）解析
        left = self._parse_primary() or self._parse_column()
        # 赋值分隔符：兼容不同方言，既支持 '=' 也支持 'TO'
        assignment_delimiter = self._match_texts(("=", "TO"))

        # 若左侧缺失，或方言要求显式分隔符但未出现，则回退并返回 None
        if not left or (self.SET_REQUIRES_ASSIGNMENT_DELIMITER and not assignment_delimiter):
            self._retreat(index)
            return None

        # 右侧解析：可为一般语句或标识；若为列/标识，则归一化为变量表达（避免被当作列引用）
        right = self._parse_statement() or self._parse_id_var()
        if isinstance(right, (exp.Column, exp.Identifier)):
            right = exp.var(right.name)

        # 组装等式表达式，再封装为 SetItem；kind 表示作用域或类别
        this = self.expression(exp.EQ, this=left, expression=right)
        return self.expression(exp.SetItem, this=this, kind=kind)

    # 解析 SET TRANSACTION：事务特性设置，支持 GLOBAL 标志
    def _parse_set_transaction(self, global_: bool = False) -> exp.Expression:
        self._match_text_seq("TRANSACTION")
        # 解析事务特性列表（CSV），每一项为受支持选项（如 ISOLATION LEVEL 等）
        characteristics = self._parse_csv(
            lambda: self._parse_var_from_options(self.TRANSACTION_CHARACTERISTICS)
        )
        # 返回统一的 SetItem 表达式，并设置 kind="TRANSACTION" 与 global 标志
        return self.expression(
            exp.SetItem,
            expressions=characteristics,
            kind="TRANSACTION",
            **{"global": global_},  # type: ignore
        )

    # 解析 SET 子项：优先从 trie/表驱动解析器中寻找特定分支，否则回退到通用赋值解析
    def _parse_set_item(self) -> t.Optional[exp.Expression]:
        parser = self._find_parser(self.SET_PARSERS, self.SET_TRIE)
        return parser(self) if parser else self._parse_set_item_assignment(kind=None)

    # 解析 SET 语句：收集多个 SET 子项（CSV）；若行尾仍有 token，则回退并按通用命令处理
    def _parse_set(self, unset: bool = False, tag: bool = False) -> exp.Set | exp.Command:
        index = self._index
        set_ = self.expression(
            exp.Set, expressions=self._parse_csv(self._parse_set_item), unset=unset, tag=tag
        )

        # 若还有剩余 token，说明该 SET 分支不独立成立；回退并交给命令解析，避免半解析
        if self._curr:
            self._retreat(index)
            return self._parse_as_command(self._prev)

        return set_

    # 从受支持的选项集合中解析变量（含多词续接），并在不匹配时选择报错或回退
    def _parse_var_from_options(
        self, options: OPTIONS_TYPE, raise_unmatched: bool = True
    ) -> t.Optional[exp.Var]:
        start = self._curr
        if not start:
            return None

        # 取首词作为起始选项（统一大写便于匹配）
        option = start.text.upper()
        continuations = options.get(option)

        index = self._index
        self._advance()
        # 匹配可能的后续关键字序列（可为单词或多词组合）以组成完整选项
        for keywords in continuations or []:
            if isinstance(keywords, str):
                keywords = (keywords,)

            if self._match_text_seq(*keywords):
                option = f"{option} {' '.join(keywords)}"
                break
        else:
            # continuations 存在（或显式为 None 表示必须精确匹配）却未匹配成功
            if continuations or continuations is None:
                if raise_unmatched:
                    # 严格模式下直接抛错，帮助用户定位未支持/拼写错误的选项
                    self.raise_error(f"Unknown option {option}")

                # 宽松模式下回退至起始位置并返回 None，交由上层分支处理
                self._retreat(index)
                return None

        # 返回规范化后的变量表达（统一为 Var）
        return exp.var(option)

    # 将剩余 token 吞并恢复原始 SQL 片段，回退为通用命令表达（用于不支持/未显式解析的语句）
    def _parse_as_command(self, start: Token) -> exp.Command:
        while self._curr:
            self._advance()
        text = self._find_sql(start, self._prev)
        size = len(start.text)
        # 触发一次“不支持”告警，提示调用方该语句仅以原文形式保留
        self._warn_unsupported()
        return exp.Command(this=text[:size], expression=text[size:])

    # 解析字典属性：形如 NAME(KIND( key value, ... )) 的层级结构
    def _parse_dict_property(self, this: str) -> exp.DictProperty:
        settings = []

        self._match_l_paren()
        # KIND 可选，统一解析为 id/var 用于保留原始标识
        kind = self._parse_id_var()

        # 若紧随左括号，则解析子属性列表（支持多组键值对）
        if self._match(TokenType.L_PAREN):
            while True:
                key = self._parse_id_var()
                value = self._parse_primary()
                # key 与 value 同时缺失视为终止
                if not key and value is None:
                    break
                settings.append(self.expression(exp.DictSubProperty, this=key, value=value))
            self._match(TokenType.R_PAREN)

        self._match_r_paren()

        return self.expression(
            exp.DictProperty,
            this=this,
            kind=kind.this if kind else None,
            settings=settings,
        )

    # 解析区间属性：支持 MIN ... MAX ... 或仅 MAX ...（默认最小值为 0）
    def _parse_dict_range(self, this: str) -> exp.DictRange:
        self._match_l_paren()
        has_min = self._match_text_seq("MIN")
        if has_min:
            # MIN 优先：按 MIN 再 MAX 顺序解析
            min = self._parse_var() or self._parse_primary()
            self._match_text_seq("MAX")
            max = self._parse_var() or self._parse_primary()
        else:
            # 仅给定 MAX 时，最小值回退为 0（与部分方言默认一致）
            max = self._parse_var() or self._parse_primary()
            min = exp.Literal.number(0)
        self._match_r_paren()
        return self.expression(exp.DictRange, this=this, min=min, max=max)

    # 解析推导式（comprehension）: <expr> IN <iterator> [IF <condition>]
    def _parse_comprehension(
        self, this: t.Optional[exp.Expression]
    ) -> t.Optional[exp.Comprehension]:
        # 记录当前位置：若不匹配 IN，则需要回退以避免多读一个 token
        index = self._index
        # 先解析被映射的表达式部分（如 Python 风格推导式中的前项）
        expression = self._parse_column()
        position = self._match(TokenType.COMMA) and self._parse_column()
        # 必须匹配 IN，否则不是推导式；回退到 index-1（把多 advance 的一步撤回）
        if not self._match(TokenType.IN):
            self._retreat(index - 1)
            return None
        # 解析迭代器标识（集合/表/列）
        iterator = self._parse_column()
        # 可选的过滤条件：存在 IF 时才解析，避免误吞 token
        condition = self._parse_assignment() if self._match_text_seq("IF") else None
        return self.expression(
            exp.Comprehension,
            this=this,
            expression=expression,
            position=position,
            iterator=iterator,
            condition=condition,
        )

    # 解析 heredoc 字符串：支持 $TAG$...$TAG$ 或 $$...$$ 的标签包裹形式
    def _parse_heredoc(self) -> t.Optional[exp.Heredoc]:
        # 直接匹配到整块 heredoc 字符串时，复用已有 token 的文本
        if self._match(TokenType.HEREDOC_STRING):
            return self.expression(exp.Heredoc, this=self._prev.text)

        # 否则按 $...$ 标签手动扫描：首个字符必须是 '$'
        if not self._match_text_seq("$"):
            return None

        tags = ["$"]
        tag_text = None

        # 若 $ 后紧跟标识，则形成 $TAG 的起始；否则无法闭合，报错提示
        if self._is_connected():
            self._advance()
            tags.append(self._prev.text.upper())
        else:
            self.raise_error("No closing $ found")

        # 若 TAG 存在，则要求后续必须再匹配一个 '$' 以构成 $TAG$
        if tags[-1] != "$":
            if self._is_connected() and self._match_text_seq("$"):
                tag_text = tags[-1]
                tags.append("$")
            else:
                self.raise_error("No closing $ found")

        heredoc_start = self._curr

        # 扫描直到出现闭合标签序列（不前进指针试探），命中后截取文本并跳过整个标签长度
        while self._curr:
            if self._match_text_seq(*tags, advance=False):
                this = self._find_sql(heredoc_start, self._prev)
                self._advance(len(tags))
                return self.expression(exp.Heredoc, this=this, tag=tag_text)

            self._advance()

        # 扫描到末尾仍未闭合：报错，指明期望的闭合标签
        self.raise_error(f"No closing {''.join(tags)} found")
        return None

    # 在多词关键字前缀树（trie）中查找解析器：支持多词 token 的逐步匹配
    def _find_parser(
        self, parsers: t.Dict[str, t.Callable], trie: t.Dict
    ) -> t.Optional[t.Callable]:
        if not self._curr:
            return None

        index = self._index
        this = []
        while True:
            # 当前 token 可能包含空格（多词），先按空格拆分以兼容 trie 的键
            curr = self._curr.text.upper()
            key = curr.split(" ")
            this.append(curr)

            self._advance()
            result, trie = in_trie(trie, key)
            if result == TrieResult.FAILED:
                break

            # 命中完整关键字序列：返回对应子解析器
            if result == TrieResult.EXISTS:
                subparser = parsers[" ".join(this)]
                return subparser

        # 未匹配成功则回退到初始位置，交由其他分支处理
        self._retreat(index)
        return None

    # 关键函数
    # 尝试匹配当前 token 的类型；可选地前进指针，并将紧邻的注释附着到表达式上
    def _match(self, token_type, advance=True, expression=None):
        if not self._curr:
            return None

        # 精确匹配 token 类型；若要求前进，则消费该 token
        if self._curr.token_type == token_type:
            if advance:
                self._advance()
            # 把本 token 前后的注释挂到目标表达式上，便于保留注释语义
            self._add_comments(expression)
            return True

        return None

    # 尝试匹配当前 token 是否属于给定类型集合；匹配成功可选择前进
    def _match_set(self, types, advance=True):
        if not self._curr:
            return None

        if self._curr.token_type in types:
            if advance:
                self._advance()
            return True

        return None

    # 尝试匹配连续的两个 token 类型对（常用于识别关键字对，如 DROP DEFAULT）
    def _match_pair(self, token_type_a, token_type_b, advance=True):
        if not self._curr or not self._next:
            return None

        if self._curr.token_type == token_type_a and self._next.token_type == token_type_b:
            if advance:
                # 同时前进两个 token，保持解析游标与消耗的长度一致
                self._advance(2)
            return True

        return None

    # 必须匹配左括号，否则报错（用于强约束处的括号）
    def _match_l_paren(self, expression: t.Optional[exp.Expression] = None) -> None:
        if not self._match(TokenType.L_PAREN, expression=expression):
            self.raise_error("Expecting (")

    # 必须匹配右括号，否则报错
    def _match_r_paren(self, expression: t.Optional[exp.Expression] = None) -> None:
        if not self._match(TokenType.R_PAREN, expression=expression):
            self.raise_error("Expecting )")

    # 尝试匹配当前 token 文本是否属于给定集合；忽略字符串字面量以避免误匹配
    def _match_texts(self, texts, advance=True):
        if (
            self._curr
            and self._curr.token_type != TokenType.STRING
            and self._curr.text.upper() in texts
        ):
            if advance:
                self._advance()
            return True
        return None

    # 尝试按顺序匹配一串关键字文本；若中途失败，回退到进入函数时的位置
    def _match_text_seq(self, *texts, advance=True):
        index = self._index
        for text in texts:
            if (
                self._curr
                and self._curr.token_type != TokenType.STRING
                and self._curr.text.upper() == text
            ):
                self._advance()
            else:
                # 任一关键字不匹配则整体失败，回溯至初始位置，保证幂等
                self._retreat(index)
                return None

        # 若仅做前瞻（advance=False），在成功匹配后回退，不实际消耗 token
        if not advance:
            self._retreat(index)

        return True

    # 将表达式中的列引用按给定映射替换/强制类型转换
    # expressions 形如 [Lambda(name=col, to=datatype or False), ...]
    # - 若映射值为 False：仅用列标识（或点式标识）替换
    # - 若映射值为类型：构造 CAST(col AS type)
    def _replace_lambda(
        self, node: t.Optional[exp.Expression], expressions: t.List[exp.Expression]
    ) -> t.Optional[exp.Expression]:
        if not node:
            return node

        # 预构造列名到目标类型/标记的映射
        lambda_types = {e.name: e.args.get("to") or False for e in expressions}

        for column in node.find_all(exp.Column):
            # 取列的第一段名做匹配键（保留表前缀不影响匹配）
            typ = lambda_types.get(column.parts[0].name)
            if typ is not None:
                # 若列带表前缀，优先取点式表达（a.b）；否则仅取列标识
                dot_or_id = column.to_dot() if column.table else column.this

                # 需要类型转换时，将列包装为 CAST(col AS typ)
                if typ:
                    dot_or_id = self.expression(
                        exp.Cast,
                        this=dot_or_id,
                        to=typ,
                    )

                parent = column.parent

                # 若列处于点访问链条中（a.b.c），需要替换最外层非 Dot 的父节点
                while isinstance(parent, exp.Dot):
                    if not isinstance(parent.parent, exp.Dot):
                        parent.replace(dot_or_id)
                        break
                    parent = parent.parent
                else:
                    # 列正好是根节点：直接替换根；否则仅替换该列节点
                    if column is node:
                        node = dot_or_id
                    else:
                        column.replace(dot_or_id)
        return node

    # 解析 TRUNCATE TABLE/DB 语句：支持 ClickHouse 的 TRUNCATE DATABASE 与多表、可选修饰
    def _parse_truncate_table(self) -> t.Optional[exp.TruncateTable] | exp.Expression:
        start = self._prev

        # 防止与函数调用 TRUNCATE(number, decimals) 混淆：遇到 '(' 则回溯并按函数解析
        if self._match(TokenType.L_PAREN):
            self._retreat(self._index - 2)
            return self._parse_function()

        # ClickHouse 支持 TRUNCATE DATABASE：若匹配到 DATABASE，后续表解析以库引用方式处理
        is_database = self._match(TokenType.DATABASE)

        # 兼容存在 TABLE 关键字（可选）
        self._match(TokenType.TABLE)

        # IF EXISTS / EXISTS 语义：此处 not_=False 表示只匹配 EXISTS，不匹配 NOT EXISTS
        exists = self._parse_exists(not_=False)

        # 解析目标（支持 CSV）：当 is_database 时，按库级引用解析
        expressions = self._parse_csv(
            lambda: self._parse_table(schema=True, is_db_reference=is_database)
        )

        # 可选 ON <cluster> 修饰（方言扩展）
        cluster = self._parse_on_property() if self._match(TokenType.ON) else None

        # 可选身份策略：RESTART/CONTINUE IDENTITY（如 Postgres）
        if self._match_text_seq("RESTART", "IDENTITY"):
            identity = "RESTART"
        elif self._match_text_seq("CONTINUE", "IDENTITY"):
            identity = "CONTINUE"
        else:
            identity = None

        # 级联/限制删除：CASCADE/RESTRICT，保留原始文本
        if self._match_text_seq("CASCADE") or self._match_text_seq("RESTRICT"):
            option = self._prev.text
        else:
            option = None

        # 可选分区：TRUNCATE ... PARTITION(...)
        partition = self._parse_partition()

        # 若仍有残余 token，说明非标准 TRUNCATE 分支，回退为通用命令处理，避免半解析
        if self._curr:
            return self._parse_as_command(start)

        return self.expression(
            exp.TruncateTable,
            expressions=expressions,
            is_database=is_database,
            exists=exists,
            cluster=cluster,
            identity=identity,
            option=option,
            partition=partition,
        )

    # 解析 WITH OPERATOR：在已解析的操作符类别（opclass）基础上，处理可选 WITH <op>
    def _parse_with_operator(self) -> t.Optional[exp.Expression]:
        # 先解析排序后的 opclass 列表（可能为空），作为基底表达式
        this = self._parse_ordered(self._parse_opclass)

        # 若未出现 WITH，则直接返回之前解析的基底
        if not self._match(TokenType.WITH):
            return this

        # WITH 之后允许任意标识/关键字作为运算符名
        op = self._parse_var(any_token=True)

        return self.expression(exp.WithOperator, this=this, op=op)

    # 解析包装选项：形如 = ( key=value, ... )，并处理 FORMAT_NAME 特例
    def _parse_wrapped_options(self) -> t.List[t.Optional[exp.Expression]]:
        self._match(TokenType.EQ)
        self._match(TokenType.L_PAREN)

        opts: t.List[t.Optional[exp.Expression]] = []
        option: exp.Expression | None
        while self._curr and not self._match(TokenType.R_PAREN):
            if self._match_text_seq("FORMAT_NAME", "="):
                # FORMAT_NAME 在 Snowflake/T-SQL 可设为标识符，需特殊解析
                option = self._parse_format_name()
            else:
                option = self._parse_property()

            # 若某项无法解析，抛错并中断，以便调用方获取明确错误信息
            if option is None:
                self.raise_error("Unable to parse option")
                break

            opts.append(option)

        return opts

    # 解析 COPY 参数：根据方言决定是否按 CSV（逗号）分隔选项
    def _parse_copy_parameters(self) -> t.List[exp.CopyParameter]:
        # 某些方言（如 Snowflake/Databricks）允许以逗号分隔多个参数
        sep = TokenType.COMMA if self.dialect.COPY_PARAMS_ARE_CSV else None

        options = []
        while self._curr and not self._match(TokenType.R_PAREN, advance=False):
            # 选项名可为任意标识/关键字（如 FILE_FORMAT、FORMAT、CREDENTIALS 等）
            option = self._parse_var(any_token=True)
            prev = self._prev.text.upper()

            # 不同方言在选项与值之间可能使用空格、'=' 或 'AS' 作为分隔
            self._match(TokenType.EQ)
            self._match(TokenType.ALIAS)

            param = self.expression(exp.CopyParameter, this=option)

            if prev in self.COPY_INTO_VARLEN_OPTIONS and self._match(
                TokenType.L_PAREN, advance=False
            ):
                # 可变长度选项：如 Snowflake 的 FILE_FORMAT，Databricks 的 COPY/FORMAT
                param.set("expressions", self._parse_wrapped_options())
            elif prev == "FILE_FORMAT":
                # T-SQL 外部文件格式：值是一个标识符（格式对象名）
                param.set("expression", self._parse_field())
            else:
                # 普通键值：解析未加引号的字段，兼容裸值/标识
                param.set("expression", self._parse_unquoted_field() or self._parse_bracket())

            options.append(param)
            # 若启用 CSV 模式，则尝试消费逗号分隔，否者不分隔
            self._match(sep)

        return options

    # 解析跨方言的凭据段：支持 Snowflake/Redshift 等不同写法
    def _parse_credentials(self) -> t.Optional[exp.Credentials]:
        expr = self.expression(exp.Credentials)

        # Snowflake：STORAGE_INTEGRATION=...
        if self._match_text_seq("STORAGE_INTEGRATION", "="):
            expr.set("storage", self._parse_field())
        # Snowflake：CREDENTIALS=(...)；Redshift：CREDENTIALS '...'
        if self._match_text_seq("CREDENTIALS"):
            creds = (
                self._parse_wrapped_options() if self._match(TokenType.EQ) else self._parse_field()
            )
            expr.set("credentials", creds)
        # Snowflake：ENCRYPTION=(...)
        if self._match_text_seq("ENCRYPTION"):
            expr.set("encryption", self._parse_wrapped_options())
        # Redshift/IAM：IAM_ROLE <arn>
        if self._match_text_seq("IAM_ROLE"):
            expr.set("iam_role", self._parse_field())
        # 区域：REGION <name>
        if self._match_text_seq("REGION"):
            expr.set("region", self._parse_field())

        return expr

    # 解析文件位置：此处复用通用字段解析，兼容路径/标识/字符串
    def _parse_file_location(self) -> t.Optional[exp.Expression]:
        return self._parse_field()

    # 解析 COPY 语句：COPY INTO <target> FROM/TO <files...> [WITH (...)]
    def _parse_copy(self) -> exp.Copy | exp.Command:
        start = self._prev

        self._match(TokenType.INTO)

        # 目标可为子查询：COPY INTO (<select>) ...；否则为表
        this = (
            self._parse_select(nested=True, parse_subquery_alias=False)
            if self._match(TokenType.L_PAREN, advance=False)
            else self._parse_table(schema=True)
        )

        # kind 标记方向：优先匹配 FROM，否则若没有 TO 则视为 FROM；用于 into/from 变体
        kind = self._match(TokenType.FROM) or not self._match_text_seq("TO")

        # 文件列表与凭据
        files = self._parse_csv(self._parse_file_location)
        if self._match(TokenType.EQ, advance=False):
            # Backtrack one token since we've consumed the lhs of a parameter assignment here.
            # This can happen for Snowflake dialect. Instead, we'd like to parse the parameter
            # list via `_parse_wrapped(..)` below.
            self._advance(-1)
            files = []

        credentials = self._parse_credentials()

        # 可选 WITH 子句：携带 COPY 参数列表
        self._match_text_seq("WITH")

        params = self._parse_wrapped(self._parse_copy_parameters, optional=True)

        # 若仍有残余 token，回退为通用命令，避免半解析
        if self._curr:
            return self._parse_as_command(start)

        return self.expression(
            exp.Copy,
            this=this,
            kind=kind,
            credentials=credentials,
            files=files,
            params=params,
        )

    # 解析 NORMALIZE(expr[, form])：form 可选，通过逗号方式出现
    def _parse_normalize(self) -> exp.Normalize:
        return self.expression(
            exp.Normalize,
            this=self._parse_bitwise(),
            form=self._match(TokenType.COMMA) and self._parse_var(),
        )

    # 解析 CEIL/FLOOR：支持 CEIL(expr[, decimals]) [TO unit] 语义
    def _parse_ceil_floor(self, expr_type: t.Type[TCeilFloor]) -> TCeilFloor:
        args = self._parse_csv(lambda: self._parse_lambda())

        this = seq_get(args, 0)
        decimals = seq_get(args, 1)

        return expr_type(
            this=this, decimals=decimals, to=self._match_text_seq("TO") and self._parse_var()
        )

    # 解析星号扩展：支持 COLUMNS(...) 解包、EXCEPT/REPLACE/RENAME 修饰
    def _parse_star_ops(self) -> t.Optional[exp.Expression]:
        star_token = self._prev

        # 预读 COLUMNS( ... )：若真为函数形式，交由通用函数解析；并标记解包
        if self._match_text_seq("COLUMNS", "(", advance=False):
            this = self._parse_function()
            if isinstance(this, exp.Columns):
                this.set("unpack", True)
            return this

        # 普通星号：可带 EXCEPT/REPLACE/RENAME 变体
        return self.expression(
            exp.Star,
            **{  # type: ignore
                "except": self._parse_star_op("EXCEPT", "EXCLUDE"),
                "replace": self._parse_star_op("REPLACE"),
                "rename": self._parse_star_op("RENAME"),
            },
        ).update_positions(star_token)

    # 解析 GRANT 权限项：收集连续关键字为一个权限，支持列级权限列表
    def _parse_grant_privilege(self) -> t.Optional[exp.GrantPrivilege]:
        privilege_parts = []

        # 连续消费关键字，直至遇到逗号（当前权限结束）、ON（权限列表结束）或左括号（列列表开始）
        while self._curr and not self._match_set(self.PRIVILEGE_FOLLOW_TOKENS, advance=False):
            privilege_parts.append(self._curr.text.upper())
            self._advance()

        # 将收集的关键字拼成单个权限名（如 SELECT、INSERT、REFERENCES 等）
        this = exp.var(" ".join(privilege_parts))
        # 若随后出现列列表（(col, ...)），则解析并挂到 expressions 上
        expressions = (
            self._parse_wrapped_csv(self._parse_column)
            if self._match(TokenType.L_PAREN, advance=False)
            else None
        )

        return self.expression(exp.GrantPrivilege, this=this, expressions=expressions)

    # 解析 GRANT 主体（被授予者）：支持 ROLE/GROUP 前缀
    def _parse_grant_principal(self) -> t.Optional[exp.GrantPrincipal]:
        kind = self._match_texts(("ROLE", "GROUP")) and self._prev.text.upper()
        principal = self._parse_id_var()

        if not principal:
            return None

        return self.expression(exp.GrantPrincipal, this=principal, kind=kind)


    # 解析 GRANT：GRANT <privileges> ON <kind> <securable> TO <principals> [WITH GRANT OPTION]
    def _parse_grant_revoke_common(
        self,
    ) -> t.Tuple[t.Optional[t.List], t.Optional[str], t.Optional[exp.Expression]]:
        privileges = self._parse_csv(self._parse_grant_privilege)

        # ON <对象类型> <可保护对象>
        self._match(TokenType.ON)
        kind = self._match_set(self.CREATABLES) and self._prev.text.upper()

        # securable 可能包含 MySQL 的特殊形式（如 foo.*、*.*），用 try_parse 宽松处理
        securable = self._try_parse(self._parse_table_parts)

        # 若对象未解析或缺少 TO 关键字，视为非常见方言用法：回退到通用命令
        return privileges, kind, securable

    def _parse_grant(self) -> exp.Grant | exp.Command:
        start = self._prev

        privileges, kind, securable = self._parse_grant_revoke_common()
        if not securable or not self._match_text_seq("TO"):
            return self._parse_as_command(start)

        # 被授予者列表：支持 ROLE/GROUP 前缀
        principals = self._parse_csv(self._parse_grant_principal)

        # 可选 WITH GRANT OPTION
        grant_option = self._match_text_seq("WITH", "GRANT", "OPTION")

        # 若还有残余 token，回退为通用命令，避免半解析
        if self._curr:
            return self._parse_as_command(start)

        return self.expression(
            exp.Grant,
            privileges=privileges,
            kind=kind,
            securable=securable,
            principals=principals,
            grant_option=grant_option,
        )




    def _parse_revoke(self) -> exp.Revoke | exp.Command:
        start = self._prev

        grant_option = self._match_text_seq("GRANT", "OPTION", "FOR")

        privileges, kind, securable = self._parse_grant_revoke_common()

        if not securable or not self._match_text_seq("FROM"):
            return self._parse_as_command(start)

        principals = self._parse_csv(self._parse_grant_principal)

        cascade = None
        if self._match_texts(("CASCADE", "RESTRICT")):
            cascade = self._prev.text.upper()

        if self._curr:
            return self._parse_as_command(start)

        return self.expression(
            exp.Revoke,
            privileges=privileges,
            kind=kind,
            securable=securable,
            principals=principals,
            grant_option=grant_option,
            cascade=cascade,
        )

    # 解析 OVERLAY：映射 PLACING/FROM/FOR 子句到对应字段
    def _parse_overlay(self) -> exp.Overlay:
        return self.expression(
            exp.Overlay,
            **{  # type: ignore
                "this": self._parse_bitwise(),
                "expression": self._match_text_seq("PLACING") and self._parse_bitwise(),
                "from": self._match_text_seq("FROM") and self._parse_bitwise(),
                "for": self._match_text_seq("FOR") and self._parse_bitwise(),
            },
        )

    # 解析 FILE_FORMAT 名称：Snowflake 允许字符串或标识
    def _parse_format_name(self) -> exp.Property:
        # Snowflake 文档外行为：FILE_FORMAT = <string|identifier>
        return self.expression(
            exp.Property,
            this=exp.var("FORMAT_NAME"),
            value=self._parse_string() or self._parse_table_parts(),
        )

    # 解析 MAX_BY/MIN_BY：形如 MAX_BY(x, y, [count])，可带 DISTINCT x, y 形式
    def _parse_max_min_by(self, expr_type: t.Type[exp.AggFunc]) -> exp.AggFunc:
        args: t.List[exp.Expression] = []

        # DISTINCT x, y：先包装 DISTINCT(x)，再继续解析其余参数
        if self._match(TokenType.DISTINCT):
            args.append(self.expression(exp.Distinct, expressions=[self._parse_lambda()]))
            self._match(TokenType.COMMA)

        # 继续解析剩余参数（x, y[, count]），均按赋值表达式处理
        args.extend(self._parse_function_args())

        return self.expression(
            expr_type, this=seq_get(args, 0), expression=seq_get(args, 1), count=seq_get(args, 2)
        )

    # 构造标识符表达式：保留原始 token 文本与位置信息，便于后续生成与定位
    def _identifier_expression(
        self, token: t.Optional[Token] = None, **kwargs: t.Any
    ) -> exp.Identifier:
        token = token or self._prev
        expression = self.expression(exp.Identifier, this=token.text, **kwargs)
        # 将 token 的位置信息同步到表达式，确保误差最小化
        expression.update_positions(token)
        return expression

    # 将当前查询包装为 WITH/CTE 的一环，用于管道语法每一步产出临时结果
    def _build_pipe_cte(
        self,
        query: exp.Query,
        expressions: t.List[exp.Expression],
        alias_cte: t.Optional[exp.TableAlias] = None,
    ) -> exp.Select:
        new_cte: t.Optional[t.Union[str, exp.TableAlias]]
        if alias_cte:
            new_cte = alias_cte
        else:
            # 未指定别名时自动生成 __tmpN，确保每步管道均可引用
            self._pipe_cte_counter += 1
            new_cte = f"__tmp{self._pipe_cte_counter}"

        # 若原查询已有 WITH，则把 CTE 链挂到新的 SELECT 上，避免丢失
        with_ = query.args.get("with")
        ctes = with_.pop() if with_ else None

        new_select = exp.select(*expressions, copy=False).from_(new_cte, copy=False)
        if ctes:
            new_select.set("with", ctes)

        # 形如: WITH new_cte AS (query) SELECT ... FROM new_cte
        return new_select.with_(new_cte, as_=query, copy=False)

    # 管道语法：处理 SELECT 片段，合并进现有 SELECT 并产出新的 CTE
    def _parse_pipe_syntax_select(self, query: exp.Select) -> exp.Select:
        select = self._parse_select(consume_pipe=False)
        if not select:
            return query

        return self._build_pipe_cte(
            query=query.select(*select.expressions, append=False), expressions=[exp.Star()]
        )

    # 管道语法：处理 LIMIT/OFFSET。保留更严格的行数限制，并对偏移量做累加
    def _parse_pipe_syntax_limit(self, query: exp.Select) -> exp.Select:
        limit = self._parse_limit()
        offset = self._parse_offset()
        if limit:
            curr_limit = query.args.get("limit", limit)
            # 若已有更小/相等的限制，则使用当前 limit（避免放宽限制）
            if curr_limit.expression.to_py() >= limit.expression.to_py():
                query.limit(limit, copy=False)
        if offset:
            curr_offset = query.args.get("offset")
            curr_offset = curr_offset.expression.to_py() if curr_offset else 0
            # 多次管道步骤的 offset 需要累加
            query.offset(exp.Literal.number(curr_offset + offset.expression.to_py()), copy=False)

        return query

    # 管道语法：解析聚合字段（含别名/排序）。若即将进入 GROUP AND 则直接返回表达式
    def _parse_pipe_syntax_aggregate_fields(self) -> t.Optional[exp.Expression]:
        this = self._parse_assignment()
        if self._match_text_seq("GROUP", "AND", advance=False):
            return this

        # 允许对字段进行别名设置，后续 GROUP/ORDER 可引用别名
        this = self._parse_alias(this)

        # 支持 ASC/DESC 排序修饰，封装为 Ordered 表达式
        if self._match_set((TokenType.ASC, TokenType.DESC), advance=False):
            return self._parse_ordered(lambda: this)

        return this

    # 管道语法：聚合 +（可选）分组/排序的组合解析
    def _parse_pipe_syntax_aggregate_group_order_by(
        self, query: exp.Select, group_by_exists: bool = True
    ) -> exp.Select:
        # 解析逗号分隔的聚合/分组/排序字段序列
        expr = self._parse_csv(self._parse_pipe_syntax_aggregate_fields)
        aggregates_or_groups, orders = [], []
        for element in expr:
            if isinstance(element, exp.Ordered):
                this = element.this
                # 若排序对象是别名（Alias），将排序对象替换为别名标识，保证生成 SQL 时引用一致
                if isinstance(this, exp.Alias):
                    element.set("this", this.args["alias"])
                orders.append(element)
            else:
                this = element
            aggregates_or_groups.append(this)

        if group_by_exists:
            # 将这些字段投影到 SELECT，并按其别名或本体进行 GROUP BY
            query.select(*aggregates_or_groups, copy=False).group_by(
                *[projection.args.get("alias", projection) for projection in aggregates_or_groups],
                copy=False,
            )
        else:
            # 仅替换 SELECT 投影，不新增到末尾（append=False）
            query.select(*aggregates_or_groups, append=False, copy=False)

        if orders:
            # 若存在排序字段，则追加 ORDER BY 并覆盖先前的排序（append=False）
            return query.order_by(*orders, append=False, copy=False)

        return query

    # 管道语法：AGGREGATE 起始，随后可跟 GROUP BY 或 GROUP AND ORDER BY 的组合
    def _parse_pipe_syntax_aggregate(self, query: exp.Select) -> exp.Select:
        self._match_text_seq("AGGREGATE")
        # 第一步仅做聚合投影（不立即 group by），便于后续可选的 group/order 组合
        query = self._parse_pipe_syntax_aggregate_group_order_by(query, group_by_exists=False)

        # 支持两种跟随形式：
        # 1) GROUP BY ...
        # 2) GROUP AND ORDER BY ...
        if self._match(TokenType.GROUP_BY) or (
            self._match_text_seq("GROUP", "AND") and self._match(TokenType.ORDER_BY)
        ):
            query = self._parse_pipe_syntax_aggregate_group_order_by(query)

        # 产出新的 CTE，以便后续管道继续使用
        return self._build_pipe_cte(query=query, expressions=[exp.Star()])

    # 管道语法：集合算子（UNION/EXCEPT/INTERSECT），自动解包子查询并保持 CTE 链
    def _parse_pipe_syntax_set_operator(self, query: exp.Query) -> t.Optional[exp.Query]:
        first_setop = self.parse_set_operation(this=query)
        if not first_setop:
            return None

        def _parse_and_unwrap_query() -> t.Optional[exp.Select]:
            # 尝试解析括号内的子查询，并解包为 Select
            expr = self._parse_paren()
            return expr.assert_is(exp.Subquery).unnest() if expr else None

        # 移除最左侧输入，后续将其作为 CTE 来源
        first_setop.this.pop()

        # 解析后续集合项，并统一解包子查询
        setops = [
            first_setop.expression.pop().assert_is(exp.Subquery).unnest(),
            *self._parse_csv(_parse_and_unwrap_query),
        ]

        # 先把当前输入变成 CTE，避免 FROM 来源缺失
        query = self._build_pipe_cte(query=query, expressions=[exp.Star()])
        with_ = query.args.get("with")
        ctes = with_.pop() if with_ else None

        # 保留 first_setop 的参数（如 ALL/ DISTINCT）合并集合
        if isinstance(first_setop, exp.Union):
            query = query.union(*setops, copy=False, **first_setop.args)
        elif isinstance(first_setop, exp.Except):
            query = query.except_(*setops, copy=False, **first_setop.args)
        else:
            query = query.intersect(*setops, copy=False, **first_setop.args)

        # 还原先前 WITH 链
        query.set("with", ctes)

        return self._build_pipe_cte(query=query, expressions=[exp.Star()])

    # 管道语法：JOIN 操作。若当前查询为 Select，则直接在其上追加 join
    def _parse_pipe_syntax_join(self, query: exp.Query) -> t.Optional[exp.Query]:
        join = self._parse_join()
        if not join:
            return None

        if isinstance(query, exp.Select):
            return query.join(join, copy=False)

        return query

    # 管道语法：PIVOT 操作。若存在 FROM 子句，将透传到表项；否则设置在查询上
    def _parse_pipe_syntax_pivot(self, query: exp.Select) -> exp.Select:
        pivots = self._parse_pivots()
        if not pivots:
            return query

        from_ = query.args.get("from")
        if from_:
            from_.this.set("pivots", pivots)
        else:
            query.set("pivots", pivots)

        return self._build_pipe_cte(query=query, expressions=[exp.Star()])

    # 管道语法：EXTEND，保留所有列并追加新的表达式列
    def _parse_pipe_syntax_extend(self, query: exp.Select) -> exp.Select:
        self._match_text_seq("EXTEND")
        # 保留 * 并在其后追加表达式；append=False 覆盖而非追加，保证列顺序按当前步定义
        query.select(*[exp.Star(), *self._parse_expressions()], append=False, copy=False)
        return self._build_pipe_cte(query=query, expressions=[exp.Star()])

    # 管道语法：TABLESAMPLE，将采样设置到最后一个 with/表项；若不存在则设置到查询上
    def _parse_pipe_syntax_tablesample(self, query: exp.Select) -> exp.Select:
        sample = self._parse_table_sample()

        with_ = query.args.get("with")
        if with_:
            # 若有 with，说明 FROM 可能通过 CTE 提供；将 sample 赋予最后一个 CTE 的主表
            with_.expressions[-1].this.set("sample", sample)
        else:
            # 无 with 时，直接挂在查询上，保持语义一致
            query.set("sample", sample)

        return query

    # 管道语法主循环：逐个消费 '|' 后的操作名并分派；集合算子与 JOIN 采用回退-再试策略消歧
    def _parse_pipe_syntax_query(self, query: exp.Query) -> t.Optional[exp.Query]:
        if isinstance(query, exp.Subquery):
            # 子查询作为起点时，先升格为 SELECT * FROM (<subquery>)
            query = exp.select("*").from_(query, copy=False)

        if not query.args.get("from"):
            # 若还没有 FROM，则将自身包成子查询并 FROM 之，以便后续操作有输入
            query = exp.select("*").from_(query.subquery(copy=False), copy=False)

        while self._match(TokenType.PIPE_GT):
            start = self._curr
            parser = self.PIPE_SYNTAX_TRANSFORM_PARSERS.get(self._curr.text.upper())
            if not parser:
                # The set operators (UNION, etc) and the JOIN operator have a few common starting
                # keywords, making it tricky to disambiguate them without lookahead. The approach
                # here is to try and parse a set operation and if that fails, then try to parse a
                # join operator. If that fails as well, then the operator is not supported.
                # 集合算子（UNION/…）与 JOIN 的起始关键字相似：先尝试集合算子，失败再尝试 JOIN
                parsed_query = self._parse_pipe_syntax_set_operator(query)
                parsed_query = parsed_query or self._parse_pipe_syntax_join(query)
                if not parsed_query:
                    # 二者都失败则回退并报错，提示不支持的操作
                    self._retreat(start)
                    self.raise_error(f"Unsupported pipe syntax operator: '{start.text.upper()}'.")
                    break
                query = parsed_query
            else:
                query = parser(self, query)

        return query

    # 解析 DECLARE 的单个声明项：支持多个变量名、类型与可选默认值
    def _parse_declareitem(self) -> t.Optional[exp.DeclareItem]:
        # 变量名列表（CSV），如 a, b, c
        vars = self._parse_csv(self._parse_id_var)
        if not vars:
            # 无变量名则当前项无效，返回 None 交由上层处理
            return None

        # 组合声明项：类型使用 _parse_types，DEFAULT 后允许任意表达式（用 bitwise 覆盖广）
        return self.expression(
            exp.DeclareItem,
            this=vars,
            kind=self._parse_types(),
            default=self._match(TokenType.DEFAULT) and self._parse_bitwise(),
        )

    # 解析 DECLARE 语句：由若干声明项组成；不完整或含剩余 token 时回退为通用命令
    def _parse_declare(self) -> exp.Declare | exp.Command:
        start = self._prev
        # 宽松尝试解析：若解析失败不抛错，避免误吞后续 token
        expressions = self._try_parse(lambda: self._parse_csv(self._parse_declareitem))

        # 条件1：未解析出任何声明项；条件2：仍有残余 token → 视为非标准 DECLARE，回退
        if not expressions or self._curr:
            return self._parse_as_command(start)

        return self.expression(exp.Declare, expressions=expressions)

    # 构造 CAST/TRY_CAST 表达式：strict=True 用 CAST，否则用 TRY_CAST（方言可控行为）
    def build_cast(self, strict: bool, **kwargs) -> exp.Cast:
        exp_class = exp.Cast if strict else exp.TryCast

        if exp_class == exp.TryCast:
            # 某些方言要求 TRY_CAST 的输入为字符串，透传方言开关到节点属性
            kwargs["requires_string"] = self.dialect.TRY_CAST_REQUIRES_STRING

        return self.expression(exp_class, **kwargs)


    def _parse_tablespace_property(self) -> exp.TablespaceProperty:
        """解析TABLESPACE属性"""
        # 解析表空间名称
        tablespace_name = self._parse_field()
        return self.expression(exp.TablespaceProperty, this=tablespace_name)

    def _parse_server_property(self) -> exp.TablespaceProperty:
        """解析外表的SERVER属性"""
        # 解析SERVER名称
        server_name = self._parse_field()
        return self.expression(exp.ServerProperty, this=server_name)
    # def _parse_index_name(self) -> t.Optional[exp.Expression]:
    #     """解析可能包含schema的索引名称"""
    #     # 先尝试解析单个标识符
    #     index_parts=[]
    #     index_full_name = ""
    #     while True:
    #         index_part_name = self._parse_id_var()
    #         # 如果后面有点号，说明是schema限定的索引名称
    #         if isinstance(index_part_name,exp.Identifier) :
    #             index_full_name += index_part_name.this
    #         else:
    #             break
    #         if self._match(TokenType.DOT):
    #             index_full_name += '.'
    #         else:
    #             break
    #     if index_full_name:
    #         return self.expression(
    #             exp.Identifier,
    #             this=index_full_name,
    #             quoted = False,
    #         )
    #     return index_part_name


    def _parse_json_value(self) -> exp.JSONValue:
        this = self._parse_bitwise()
        self._match(TokenType.COMMA)
        path = self._parse_bitwise()

        returning = self._match(TokenType.RETURNING) and self._parse_type()

        return self.expression(
            exp.JSONValue,
            this=this,
            path=self.dialect.to_json_path(path),
            returning=returning,
            on_condition=self._parse_on_condition(),
        )

    def _parse_group_concat(self) -> t.Optional[exp.Expression]:
        def concat_exprs(
            node: t.Optional[exp.Expression], exprs: t.List[exp.Expression]
        ) -> exp.Expression:
            if isinstance(node, exp.Distinct) and len(node.expressions) > 1:
                concat_exprs = [
                    self.expression(exp.Concat, expressions=node.expressions, safe=True)
                ]
                node.set("expressions", concat_exprs)
                return node
            if len(exprs) == 1:
                return exprs[0]
            return self.expression(exp.Concat, expressions=args, safe=True)

        args = self._parse_csv(self._parse_lambda)

        if args:
            order = args[-1] if isinstance(args[-1], exp.Order) else None

            if order:
                # Order By is the last (or only) expression in the list and has consumed the 'expr' before it,
                # remove 'expr' from exp.Order and add it back to args
                args[-1] = order.this
                order.set("this", concat_exprs(order.this, args))

            this = order or concat_exprs(args[0], args)
        else:
            this = None

        separator = self._parse_field() if self._match(TokenType.SEPARATOR) else None

        return self.expression(exp.GroupConcat, this=this, separator=separator)
