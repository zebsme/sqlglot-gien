from __future__ import annotations

import os
import typing as t
from enum import auto

from sqlglot.errors import SqlglotError, TokenError
from sqlglot.helper import AutoName
from sqlglot.trie import TrieResult, in_trie, new_trie

if t.TYPE_CHECKING:
    from sqlglot.dialects.dialect import DialectType


try:
    from sqlglotrs import (  # type: ignore
        Tokenizer as RsTokenizer,
        TokenizerDialectSettings as RsTokenizerDialectSettings,
        TokenizerSettings as RsTokenizerSettings,
        TokenTypeSettings as RsTokenTypeSettings,
    )

    USE_RS_TOKENIZER = os.environ.get("SQLGLOTRS_TOKENIZER", "1") == "1"
except ImportError:
    USE_RS_TOKENIZER = False


class TokenType(AutoName):
    L_PAREN = auto()
    R_PAREN = auto()
    L_BRACKET = auto()
    R_BRACKET = auto()
    L_BRACE = auto()
    R_BRACE = auto()
    COMMA = auto()
    DOT = auto()
    DASH = auto()
    PLUS = auto()
    COLON = auto()
    DOTCOLON = auto()
    DCOLON = auto()
    DCOLONDOLLAR = auto()
    DCOLONPERCENT = auto()
    DQMARK = auto()
    SEMICOLON = auto()
    STAR = auto()
    BACKSLASH = auto()
    SLASH = auto()
    LT = auto()
    LTE = auto()
    GT = auto()
    GTE = auto()
    NOT = auto()
    EQ = auto()
    NEQ = auto()
    NULLSAFE_EQ = auto()
    COLON_EQ = auto()
    COLON_GT = auto()
    NCOLON_GT = auto()
    AND = auto()
    OR = auto()
    AMP = auto()
    DPIPE = auto()
    PIPE_GT = auto()
    PIPE = auto()
    PIPE_SLASH = auto()
    DPIPE_SLASH = auto()
    CARET = auto()
    CARET_AT = auto()
    TILDA = auto()
    ARROW = auto()
    DARROW = auto()
    FARROW = auto()
    HASH = auto()
    HASH_ARROW = auto()
    DHASH_ARROW = auto()
    LR_ARROW = auto()
    DAT = auto()
    LT_AT = auto()
    AT_GT = auto()
    DOLLAR = auto()
    PARAMETER = auto()
    SESSION = auto()
    SESSION_PARAMETER = auto()
    DAMP = auto()
    XOR = auto()
    DSTAR = auto()
    QMARK_AMP = auto()
    QMARK_PIPE = auto()
    HASH_DASH = auto()
    EXCLAMATION = auto()

    URI_START = auto()

    BLOCK_START = auto()
    BLOCK_END = auto()

    SPACE = auto()
    BREAK = auto()

    STRING = auto()
    NUMBER = auto()
    IDENTIFIER = auto()
    DATABASE = auto()
    COLUMN = auto()
    COLUMN_DEF = auto()
    SCHEMA = auto()
    TABLE = auto()
    WAREHOUSE = auto()
    STAGE = auto()
    STREAMLIT = auto()
    VAR = auto()
    BIT_STRING = auto()
    HEX_STRING = auto()
    BYTE_STRING = auto()
    NATIONAL_STRING = auto()
    RAW_STRING = auto()
    HEREDOC_STRING = auto()
    UNICODE_STRING = auto()

    # types
    BIT = auto()
    BOOLEAN = auto()
    TINYINT = auto()
    UTINYINT = auto()
    SMALLINT = auto()
    USMALLINT = auto()
    MEDIUMINT = auto()
    UMEDIUMINT = auto()
    INT = auto()
    UINT = auto()
    BIGINT = auto()
    UBIGINT = auto()
    INT128 = auto()
    UINT128 = auto()
    INT256 = auto()
    UINT256 = auto()
    FLOAT = auto()
    DOUBLE = auto()
    UDOUBLE = auto()
    DECIMAL = auto()
    DECIMAL32 = auto()
    DECIMAL64 = auto()
    DECIMAL128 = auto()
    DECIMAL256 = auto()
    UDECIMAL = auto()
    BIGDECIMAL = auto()
    CHAR = auto()
    NCHAR = auto()
    VARCHAR = auto()
    NVARCHAR = auto()
    BPCHAR = auto()
    TEXT = auto()
    MEDIUMTEXT = auto()
    LONGTEXT = auto()
    BLOB = auto()
    MEDIUMBLOB = auto()
    LONGBLOB = auto()
    TINYBLOB = auto()
    TINYTEXT = auto()
    NAME = auto()
    BINARY = auto()
    VARBINARY = auto()
    JSON = auto()
    JSONB = auto()
    TIME = auto()
    TIMETZ = auto()
    TIMESTAMP = auto()
    TIMESTAMPTZ = auto()
    TIMESTAMPLTZ = auto()
    TIMESTAMPNTZ = auto()
    TIMESTAMP_S = auto()
    TIMESTAMP_MS = auto()
    TIMESTAMP_NS = auto()
    DATETIME = auto()
    DATETIME2 = auto()
    DATETIME64 = auto()
    SMALLDATETIME = auto()
    DATE = auto()
    DATE32 = auto()
    INT4RANGE = auto()
    INT4MULTIRANGE = auto()
    INT8RANGE = auto()
    INT8MULTIRANGE = auto()
    NUMRANGE = auto()
    NUMMULTIRANGE = auto()
    TSRANGE = auto()
    TSMULTIRANGE = auto()
    TSTZRANGE = auto()
    TSTZMULTIRANGE = auto()
    DATERANGE = auto()
    DATEMULTIRANGE = auto()
    UUID = auto()
    GEOGRAPHY = auto()
    GEOGRAPHYPOINT = auto()
    NULLABLE = auto()
    GEOMETRY = auto()
    POINT = auto()
    RING = auto()
    LINESTRING = auto()
    MULTILINESTRING = auto()
    POLYGON = auto()
    MULTIPOLYGON = auto()
    HLLSKETCH = auto()
    HSTORE = auto()
    SUPER = auto()
    SERIAL = auto()
    SMALLSERIAL = auto()
    BIGSERIAL = auto()
    XML = auto()
    YEAR = auto()
    USERDEFINED = auto()
    MONEY = auto()
    SMALLMONEY = auto()
    ROWVERSION = auto()
    IMAGE = auto()
    VARIANT = auto()
    OBJECT = auto()
    INET = auto()
    IPADDRESS = auto()
    IPPREFIX = auto()
    IPV4 = auto()
    IPV6 = auto()
    ENUM = auto()
    ENUM8 = auto()
    ENUM16 = auto()
    FIXEDSTRING = auto()
    LOWCARDINALITY = auto()
    NESTED = auto()
    AGGREGATEFUNCTION = auto()
    SIMPLEAGGREGATEFUNCTION = auto()
    TDIGEST = auto()
    UNKNOWN = auto()
    VECTOR = auto()
    DYNAMIC = auto()
    VOID = auto()

    # DB2特有数据类型
    LONG_VARCHAR = auto()
    DECFLOAT = auto()
    LONG_VARGRAPHIC = auto()
    
    # TERADATA特有数据类型
    BYTEINT = auto()
    # INTEGER = auto()

    # keywords
    ALIAS = auto()
    ALTER = auto()
    ALL = auto()
    ANTI = auto()
    ANY = auto()
    APPLY = auto()
    ARRAY = auto()
    ASC = auto()
    ASOF = auto()
    ATTACH = auto()
    AUTO_INCREMENT = auto()
    BEGIN = auto()
    BETWEEN = auto()
    BULK_COLLECT_INTO = auto()
    CACHE = auto()
    CASE = auto()
    CHARACTER_SET = auto()
    CLUSTER_BY = auto()
    COLLATE = auto()
    COMMAND = auto()
    COMMENT = auto()
    COMMIT = auto()
    CONNECT_BY = auto()
    CONSTRAINT = auto()
    COPY = auto()
    CREATE = auto()
    CROSS = auto()
    CUBE = auto()
    CURRENT_DATE = auto()
    CURRENT_DATETIME = auto()
    CURRENT_SCHEMA = auto()
    CURRENT_TIME = auto()
    CURRENT_TIMESTAMP = auto()
    CURRENT_USER = auto()
    DECLARE = auto()
    DEFAULT = auto()
    DELETE = auto()
    DESC = auto()
    DESCRIBE = auto()
    DETACH = auto()
    DICTIONARY = auto()
    DISTINCT = auto()
    DISTRIBUTE_BY = auto()
    DIV = auto()
    DROP = auto()
    ELSE = auto()
    END = auto()
    ESCAPE = auto()
    EXCEPT = auto()
    EXECUTE = auto()
    EXISTS = auto()
    FALSE = auto()
    FETCH = auto()
    FILE_FORMAT = auto()
    FILTER = auto()
    FINAL = auto()
    FIRST = auto()
    FOR = auto()
    FORCE = auto()
    FOREIGN_KEY = auto()
    FORMAT = auto()
    FROM = auto()
    FULL = auto()
    FUNCTION = auto()
    GET = auto()
    GLOB = auto()
    GLOBAL = auto()
    GRANT = auto()
    GROUP_BY = auto()
    GROUPING_SETS = auto()
    HAVING = auto()
    HINT = auto()
    IGNORE = auto()
    ILIKE = auto()
    IN = auto()
    INDEX = auto()
    INNER = auto()
    INSERT = auto()
    INSTALL = auto()
    INTERSECT = auto()
    INTERVAL = auto()
    INTO = auto()
    INTRODUCER = auto()
    IRLIKE = auto()
    IS = auto()
    ISNULL = auto()
    JOIN = auto()
    JOIN_MARKER = auto()
    KEEP = auto()
    KEY = auto()
    KILL = auto()
    LANGUAGE = auto()
    LATERAL = auto()
    LEFT = auto()
    LIKE = auto()
    LIMIT = auto()
    LIST = auto()
    LOAD = auto()
    LOCK = auto()
    MAP = auto()
    MATCH_CONDITION = auto()
    MATCH_RECOGNIZE = auto()
    MEMBER_OF = auto()
    MERGE = auto()
    MOD = auto()
    MODEL = auto()
    NATURAL = auto()
    NEXT = auto()
    NOTHING = auto()
    NOTNULL = auto()
    NULL = auto()
    OBJECT_IDENTIFIER = auto()
    OFFSET = auto()
    ON = auto()
    ONLY = auto()
    OPERATOR = auto()
    ORDER_BY = auto()
    ORDER_SIBLINGS_BY = auto()
    ORDERED = auto()
    ORDINALITY = auto()
    OUTER = auto()
    OVER = auto()
    OVERLAPS = auto()
    OVERWRITE = auto()
    PARTITION = auto()
    PARTITION_BY = auto()
    PERCENT = auto()
    PIVOT = auto()
    PLACEHOLDER = auto()
    POSITIONAL = auto()
    PRAGMA = auto()
    PREWHERE = auto()
    PRIMARY_KEY = auto()
    PROCEDURE = auto()
    PROPERTIES = auto()
    PSEUDO_TYPE = auto()
    PUT = auto()
    QUALIFY = auto()
    QUOTE = auto()
    RANGE = auto()
    RECURSIVE = auto()
    REFRESH = auto()
    RENAME = auto()
    REPLACE = auto()
    RETURNING = auto()
    REVOKE = auto()
    REFERENCES = auto()
    RIGHT = auto()
    RLIKE = auto()
    ROLLBACK = auto()
    ROLLUP = auto()
    ROW = auto()
    ROWS = auto()
    SELECT = auto()
    SEMI = auto()
    SEPARATOR = auto()
    SEQUENCE = auto()
    XMLAGG = auto()
    SERDE_PROPERTIES = auto()
    SET = auto()
    SETTINGS = auto()
    SHOW = auto()
    SIMILAR_TO = auto()
    SOME = auto()
    SORT_BY = auto()
    SOUNDS_LIKE = auto()
    START_WITH = auto()
    STORAGE_INTEGRATION = auto()
    STRAIGHT_JOIN = auto()
    STRUCT = auto()
    SUMMARIZE = auto()
    TABLE_SAMPLE = auto()
    TAG = auto()
    TEMPORARY = auto()
    TOP = auto()
    THEN = auto()
    TRUE = auto()
    TRUNCATE = auto()
    UNCACHE = auto()
    UNION = auto()
    UNNEST = auto()
    UNPIVOT = auto()
    UPDATE = auto()
    USE = auto()
    USING = auto()
    VALUES = auto()
    VIEW = auto()
    SEMANTIC_VIEW = auto()
    VOLATILE = auto()
    WHEN = auto()
    WHERE = auto()
    WINDOW = auto()
    WITH = auto()
    UNIQUE = auto()
    ENABLE = auto()
    DISABLE = auto()
    NOVALIDATE = auto()
    DISTRIBUTED = auto()
    UTC_DATE = auto()
    UTC_TIME = auto()
    UTC_TIMESTAMP = auto()
    VERSION_SNAPSHOT = auto()
    TIMESTAMP_SNAPSHOT = auto()
    OPTION = auto()
    SINK = auto()
    SOURCE = auto()
    ANALYZE = auto()
    NAMESPACE = auto()
    EXPORT = auto()
    SERVER = auto()
    EXTERNAL = auto()
    LOG_INTO = auto()
    REJECT_LIMIT = auto()

    # sentinel
    HIVE_TOKEN_STREAM = auto()


_ALL_TOKEN_TYPES = list(TokenType)
_TOKEN_TYPE_TO_INDEX = {token_type: i for i, token_type in enumerate(_ALL_TOKEN_TYPES)}


class Token:
    __slots__ = ("token_type", "text", "line", "col", "start", "end", "comments")

    @classmethod
    def number(cls, number: int) -> Token:
        """Returns a NUMBER token with `number` as its text."""
        return cls(TokenType.NUMBER, str(number))

    @classmethod
    def string(cls, string: str) -> Token:
        """Returns a STRING token with `string` as its text."""
        return cls(TokenType.STRING, string)

    @classmethod
    def identifier(cls, identifier: str) -> Token:
        """Returns an IDENTIFIER token with `identifier` as its text."""
        return cls(TokenType.IDENTIFIER, identifier)

    @classmethod
    def var(cls, var: str) -> Token:
        """Returns an VAR token with `var` as its text."""
        return cls(TokenType.VAR, var)

    def __init__(
        self,
        token_type: TokenType,
        text: str,
        line: int = 1,
        col: int = 1,
        start: int = 0,
        end: int = 0,
        comments: t.Optional[t.List[str]] = None,
    ) -> None:
        """Token initializer.

        Args:
            token_type: The TokenType Enum.
            text: The text of the token.
            line: The line that the token ends on.
            col: The column that the token ends on.
            start: The start index of the token.
            end: The ending index of the token.
            comments: The comments to attach to the token.
        """
        self.token_type = token_type
        self.text = text
        self.line = line
        self.col = col
        self.start = start
        self.end = end
        self.comments = [] if comments is None else comments

    def __repr__(self) -> str:
        attributes = ", ".join(f"{k}: {getattr(self, k)}" for k in self.__slots__)
        return f"<Token {attributes}>"


class _Tokenizer(type):
    def __new__(cls, clsname, bases, attrs):
        klass = super().__new__(cls, clsname, bases, attrs)

        def _convert_quotes(arr: t.List[str | t.Tuple[str, str]]) -> t.Dict[str, str]:
            return dict(
                (item, item) if isinstance(item, str) else (item[0], item[1]) for item in arr
            )

        def _quotes_to_format(
            token_type: TokenType, arr: t.List[str | t.Tuple[str, str]]
        ) -> t.Dict[str, t.Tuple[str, TokenType]]:
            return {k: (v, token_type) for k, v in _convert_quotes(arr).items()}

        klass._QUOTES = _convert_quotes(klass.QUOTES)
        klass._IDENTIFIERS = _convert_quotes(klass.IDENTIFIERS)

        klass._FORMAT_STRINGS = {
            **{
                p + s: (e, TokenType.NATIONAL_STRING)
                for s, e in klass._QUOTES.items()
                for p in ("n", "N")
            },
            **_quotes_to_format(TokenType.BIT_STRING, klass.BIT_STRINGS),
            **_quotes_to_format(TokenType.BYTE_STRING, klass.BYTE_STRINGS),
            **_quotes_to_format(TokenType.HEX_STRING, klass.HEX_STRINGS),
            **_quotes_to_format(TokenType.RAW_STRING, klass.RAW_STRINGS),
            **_quotes_to_format(TokenType.HEREDOC_STRING, klass.HEREDOC_STRINGS),
            **_quotes_to_format(TokenType.UNICODE_STRING, klass.UNICODE_STRINGS),
        }

        klass._STRING_ESCAPES = set(klass.STRING_ESCAPES)
        klass._IDENTIFIER_ESCAPES = set(klass.IDENTIFIER_ESCAPES)
        klass._COMMENTS = {
            **dict(
                (comment, None) if isinstance(comment, str) else (comment[0], comment[1])
                for comment in klass.COMMENTS
            ),
            "{#": "#}",  # Ensure Jinja comments are tokenized correctly in all dialects
        }
        if klass.HINT_START in klass.KEYWORDS:
            klass._COMMENTS[klass.HINT_START] = "*/"

        klass._KEYWORD_TRIE = new_trie(
            key.upper()
            for key in (
                *klass.KEYWORDS,
                *klass._COMMENTS,
                *klass._QUOTES,
                *klass._FORMAT_STRINGS,
            )
            if " " in key or any(single in key for single in klass.SINGLE_TOKENS)
        )

        if USE_RS_TOKENIZER:
            settings = RsTokenizerSettings(
                white_space={k: _TOKEN_TYPE_TO_INDEX[v] for k, v in klass.WHITE_SPACE.items()},
                single_tokens={k: _TOKEN_TYPE_TO_INDEX[v] for k, v in klass.SINGLE_TOKENS.items()},
                keywords={k: _TOKEN_TYPE_TO_INDEX[v] for k, v in klass.KEYWORDS.items()},
                numeric_literals=klass.NUMERIC_LITERALS,
                identifiers=klass._IDENTIFIERS,
                identifier_escapes=klass._IDENTIFIER_ESCAPES,
                string_escapes=klass._STRING_ESCAPES,
                quotes=klass._QUOTES,
                format_strings={
                    k: (v1, _TOKEN_TYPE_TO_INDEX[v2])
                    for k, (v1, v2) in klass._FORMAT_STRINGS.items()
                },
                has_bit_strings=bool(klass.BIT_STRINGS),
                has_hex_strings=bool(klass.HEX_STRINGS),
                comments=klass._COMMENTS,
                var_single_tokens=klass.VAR_SINGLE_TOKENS,
                commands={_TOKEN_TYPE_TO_INDEX[v] for v in klass.COMMANDS},
                command_prefix_tokens={
                    _TOKEN_TYPE_TO_INDEX[v] for v in klass.COMMAND_PREFIX_TOKENS
                },
                heredoc_tag_is_identifier=klass.HEREDOC_TAG_IS_IDENTIFIER,
                string_escapes_allowed_in_raw_strings=klass.STRING_ESCAPES_ALLOWED_IN_RAW_STRINGS,
                nested_comments=klass.NESTED_COMMENTS,
                hint_start=klass.HINT_START,
                tokens_preceding_hint={
                    _TOKEN_TYPE_TO_INDEX[v] for v in klass.TOKENS_PRECEDING_HINT
                },
            )
            token_types = RsTokenTypeSettings(
                bit_string=_TOKEN_TYPE_TO_INDEX[TokenType.BIT_STRING],
                break_=_TOKEN_TYPE_TO_INDEX[TokenType.BREAK],
                dcolon=_TOKEN_TYPE_TO_INDEX[TokenType.DCOLON],
                heredoc_string=_TOKEN_TYPE_TO_INDEX[TokenType.HEREDOC_STRING],
                raw_string=_TOKEN_TYPE_TO_INDEX[TokenType.RAW_STRING],
                hex_string=_TOKEN_TYPE_TO_INDEX[TokenType.HEX_STRING],
                identifier=_TOKEN_TYPE_TO_INDEX[TokenType.IDENTIFIER],
                number=_TOKEN_TYPE_TO_INDEX[TokenType.NUMBER],
                parameter=_TOKEN_TYPE_TO_INDEX[TokenType.PARAMETER],
                semicolon=_TOKEN_TYPE_TO_INDEX[TokenType.SEMICOLON],
                string=_TOKEN_TYPE_TO_INDEX[TokenType.STRING],
                var=_TOKEN_TYPE_TO_INDEX[TokenType.VAR],
                heredoc_string_alternative=_TOKEN_TYPE_TO_INDEX[klass.HEREDOC_STRING_ALTERNATIVE],
                hint=_TOKEN_TYPE_TO_INDEX[TokenType.HINT],
            )
            klass._RS_TOKENIZER = RsTokenizer(settings, token_types)
        else:
            klass._RS_TOKENIZER = None

        return klass


class Tokenizer(metaclass=_Tokenizer):
    SINGLE_TOKENS = {
        "(": TokenType.L_PAREN,
        ")": TokenType.R_PAREN,
        "[": TokenType.L_BRACKET,
        "]": TokenType.R_BRACKET,
        "{": TokenType.L_BRACE,
        "}": TokenType.R_BRACE,
        "&": TokenType.AMP,
        "^": TokenType.CARET,
        ":": TokenType.COLON,
        ",": TokenType.COMMA,
        ".": TokenType.DOT,
        "-": TokenType.DASH,
        "=": TokenType.EQ,
        ">": TokenType.GT,
        "<": TokenType.LT,
        "%": TokenType.MOD,
        "!": TokenType.NOT,
        "|": TokenType.PIPE,
        "+": TokenType.PLUS,
        ";": TokenType.SEMICOLON,
        "/": TokenType.SLASH,
        "\\": TokenType.BACKSLASH,
        "*": TokenType.STAR,
        "~": TokenType.TILDA,
        "?": TokenType.PLACEHOLDER,
        "@": TokenType.PARAMETER,
        "#": TokenType.HASH,
        # Used for breaking a var like x'y' but nothing else the token type doesn't matter
        "'": TokenType.UNKNOWN,
        "`": TokenType.UNKNOWN,
        '"': TokenType.UNKNOWN,
    }

    BIT_STRINGS: t.List[str | t.Tuple[str, str]] = []
    BYTE_STRINGS: t.List[str | t.Tuple[str, str]] = []
    HEX_STRINGS: t.List[str | t.Tuple[str, str]] = []
    RAW_STRINGS: t.List[str | t.Tuple[str, str]] = []
    HEREDOC_STRINGS: t.List[str | t.Tuple[str, str]] = []
    UNICODE_STRINGS: t.List[str | t.Tuple[str, str]] = []
    IDENTIFIERS: t.List[str | t.Tuple[str, str]] = ['"']
    QUOTES: t.List[t.Tuple[str, str] | str] = ["'"]
    STRING_ESCAPES = ["'"]
    VAR_SINGLE_TOKENS: t.Set[str] = set()

    # The strings in this list can always be used as escapes, regardless of the surrounding
    # identifier delimiters. By default, the closing delimiter is assumed to also act as an
    # identifier escape, e.g. if we use double-quotes, then they also act as escapes: "x"""
    IDENTIFIER_ESCAPES: t.List[str] = []

    # Whether the heredoc tags follow the same lexical rules as unquoted identifiers
    HEREDOC_TAG_IS_IDENTIFIER = False

    # Token that we'll generate as a fallback if the heredoc prefix doesn't correspond to a heredoc
    HEREDOC_STRING_ALTERNATIVE = TokenType.VAR

    # Whether string escape characters function as such when placed within raw strings
    STRING_ESCAPES_ALLOWED_IN_RAW_STRINGS = True

    NESTED_COMMENTS = True

    HINT_START = "/*+"

    TOKENS_PRECEDING_HINT = {TokenType.SELECT, TokenType.INSERT, TokenType.UPDATE, TokenType.DELETE}

    # Autofilled
    _COMMENTS: t.Dict[str, str] = {}
    _FORMAT_STRINGS: t.Dict[str, t.Tuple[str, TokenType]] = {}
    _IDENTIFIERS: t.Dict[str, str] = {}
    _IDENTIFIER_ESCAPES: t.Set[str] = set()
    _QUOTES: t.Dict[str, str] = {}
    _STRING_ESCAPES: t.Set[str] = set()
    _KEYWORD_TRIE: t.Dict = {}
    _RS_TOKENIZER: t.Optional[t.Any] = None

    KEYWORDS: t.Dict[str, TokenType] = {
        **{f"{{%{postfix}": TokenType.BLOCK_START for postfix in ("", "+", "-")},
        **{f"{prefix}%}}": TokenType.BLOCK_END for prefix in ("", "+", "-")},
        **{f"{{{{{postfix}": TokenType.BLOCK_START for postfix in ("+", "-")},
        **{f"{prefix}}}}}": TokenType.BLOCK_END for prefix in ("+", "-")},
        HINT_START: TokenType.HINT,
        "==": TokenType.EQ,
        "::": TokenType.DCOLON,
        "||": TokenType.DPIPE,
        "|>": TokenType.PIPE_GT,
        ">=": TokenType.GTE,
        "<=": TokenType.LTE,
        "<>": TokenType.NEQ,
        "!=": TokenType.NEQ,
        ":=": TokenType.COLON_EQ,
        "<=>": TokenType.NULLSAFE_EQ,
        "->": TokenType.ARROW,
        "->>": TokenType.DARROW,
        "=>": TokenType.FARROW,
        "#>": TokenType.HASH_ARROW,
        "#>>": TokenType.DHASH_ARROW,
        "<->": TokenType.LR_ARROW,
        "&&": TokenType.DAMP,
        "??": TokenType.DQMARK,
        "~~~": TokenType.GLOB,
        "~~": TokenType.LIKE,
        "~~*": TokenType.ILIKE,
        "~*": TokenType.IRLIKE,
        "ALL": TokenType.ALL,
        "AND": TokenType.AND,
        "ANTI": TokenType.ANTI,
        "ANY": TokenType.ANY,
        "ASC": TokenType.ASC,
        "AS": TokenType.ALIAS,
        "ASOF": TokenType.ASOF,
        "AUTOINCREMENT": TokenType.AUTO_INCREMENT,
        "AUTO_INCREMENT": TokenType.AUTO_INCREMENT,
        "BEGIN": TokenType.BEGIN,
        "BETWEEN": TokenType.BETWEEN,
        "CACHE": TokenType.CACHE,
        "UNCACHE": TokenType.UNCACHE,
        "CASE": TokenType.CASE,
        "CHARACTER SET": TokenType.CHARACTER_SET,
        "CLUSTER BY": TokenType.CLUSTER_BY,
        "COLLATE": TokenType.COLLATE,
        "COLUMN": TokenType.COLUMN,
        "COMMIT": TokenType.COMMIT,
        "CONNECT BY": TokenType.CONNECT_BY,
        "CONSTRAINT": TokenType.CONSTRAINT,
        "COPY": TokenType.COPY,
        "CREATE": TokenType.CREATE,
        "CROSS": TokenType.CROSS,
        "CUBE": TokenType.CUBE,
        "CURRENT_DATE": TokenType.CURRENT_DATE,
        "CURRENT_SCHEMA": TokenType.CURRENT_SCHEMA,
        "CURRENT_TIME": TokenType.CURRENT_TIME,
        "CURRENT_TIMESTAMP": TokenType.CURRENT_TIMESTAMP,
        "CURRENT_USER": TokenType.CURRENT_USER,
        "DATABASE": TokenType.DATABASE,
        "DEFAULT": TokenType.DEFAULT,
        "DELETE": TokenType.DELETE,
        "DESC": TokenType.DESC,
        "DESCRIBE": TokenType.DESCRIBE,
        "DISTINCT": TokenType.DISTINCT,
        "DISTRIBUTE BY": TokenType.DISTRIBUTE_BY,
        "DIV": TokenType.DIV,
        "DROP": TokenType.DROP,
        "ELSE": TokenType.ELSE,
        "END": TokenType.END,
        "ENUM": TokenType.ENUM,
        "ESCAPE": TokenType.ESCAPE,
        "EXCEPT": TokenType.EXCEPT,
        "EXECUTE": TokenType.EXECUTE,
        "EXISTS": TokenType.EXISTS,
        "FALSE": TokenType.FALSE,
        "FETCH": TokenType.FETCH,
        "FILTER": TokenType.FILTER,
        "FIRST": TokenType.FIRST,
        "FULL": TokenType.FULL,
        "FUNCTION": TokenType.FUNCTION,
        "FOR": TokenType.FOR,
        "FOREIGN KEY": TokenType.FOREIGN_KEY,
        "FORMAT": TokenType.FORMAT,
        "FROM": TokenType.FROM,
        "GEOGRAPHY": TokenType.GEOGRAPHY,
        "GEOMETRY": TokenType.GEOMETRY,
        "GLOB": TokenType.GLOB,
        "GROUP BY": TokenType.GROUP_BY,
        "GROUPING SETS": TokenType.GROUPING_SETS,
        "HAVING": TokenType.HAVING,
        "ILIKE": TokenType.ILIKE,
        "IN": TokenType.IN,
        "INDEX": TokenType.INDEX,
        "INET": TokenType.INET,
        "INNER": TokenType.INNER,
        "INSERT": TokenType.INSERT,
        "INTERVAL": TokenType.INTERVAL,
        "INTERSECT": TokenType.INTERSECT,
        "INTO": TokenType.INTO,
        "IS": TokenType.IS,
        "ISNULL": TokenType.ISNULL,
        "JOIN": TokenType.JOIN,
        "KEEP": TokenType.KEEP,
        "KILL": TokenType.KILL,
        "LATERAL": TokenType.LATERAL,
        "LEFT": TokenType.LEFT,
        "LIKE": TokenType.LIKE,
        "LIMIT": TokenType.LIMIT,
        "LOAD": TokenType.LOAD,
        "LOCK": TokenType.LOCK,
        "MERGE": TokenType.MERGE,
        "NAMESPACE": TokenType.NAMESPACE,
        "NATURAL": TokenType.NATURAL,
        "NEXT": TokenType.NEXT,
        "NOT": TokenType.NOT,
        "NOTNULL": TokenType.NOTNULL,
        "NULL": TokenType.NULL,
        "OBJECT": TokenType.OBJECT,
        "OFFSET": TokenType.OFFSET,
        "ON": TokenType.ON,
        "OR": TokenType.OR,
        "XOR": TokenType.XOR,
        "ORDER BY": TokenType.ORDER_BY,
        "ORDINALITY": TokenType.ORDINALITY,
        "OUTER": TokenType.OUTER,
        "OVER": TokenType.OVER,
        "OVERLAPS": TokenType.OVERLAPS,
        "OVERWRITE": TokenType.OVERWRITE,
        "PARTITION": TokenType.PARTITION,
        "PARTITION BY": TokenType.PARTITION_BY,
        "PARTITIONED BY": TokenType.PARTITION_BY,
        "PARTITIONED_BY": TokenType.PARTITION_BY,
        "PERCENT": TokenType.PERCENT,
        "PIVOT": TokenType.PIVOT,
        "PRAGMA": TokenType.PRAGMA,
        "PRIMARY KEY": TokenType.PRIMARY_KEY,
        "PROCEDURE": TokenType.PROCEDURE,
        "QUALIFY": TokenType.QUALIFY,
        "RANGE": TokenType.RANGE,
        "RECURSIVE": TokenType.RECURSIVE,
        "REGEXP": TokenType.RLIKE,
        "RENAME": TokenType.RENAME,
        "REPLACE": TokenType.REPLACE,
        "RETURNING": TokenType.RETURNING,
        "REFERENCES": TokenType.REFERENCES,
        "RIGHT": TokenType.RIGHT,
        "RLIKE": TokenType.RLIKE,
        "ROLLBACK": TokenType.ROLLBACK,
        "ROLLUP": TokenType.ROLLUP,
        "ROW": TokenType.ROW,
        "ROWS": TokenType.ROWS,
        "SCHEMA": TokenType.SCHEMA,
        "SELECT": TokenType.SELECT,
        "SEMI": TokenType.SEMI,
        "SESSION": TokenType.SESSION,
        "SET": TokenType.SET,
        "SETTINGS": TokenType.SETTINGS,
        "SHOW": TokenType.SHOW,
        "SIMILAR TO": TokenType.SIMILAR_TO,
        "SOME": TokenType.SOME,
        "SORT BY": TokenType.SORT_BY,
        "START WITH": TokenType.START_WITH,
        "STRAIGHT_JOIN": TokenType.STRAIGHT_JOIN,
        "TABLE": TokenType.TABLE,
        "TABLESAMPLE": TokenType.TABLE_SAMPLE,
        "TEMP": TokenType.TEMPORARY,
        "TEMPORARY": TokenType.TEMPORARY,
        "THEN": TokenType.THEN,
        "TRUE": TokenType.TRUE,
        "TRUNCATE": TokenType.TRUNCATE,
        "UNION": TokenType.UNION,
        "UNKNOWN": TokenType.UNKNOWN,
        "UNNEST": TokenType.UNNEST,
        "UNPIVOT": TokenType.UNPIVOT,
        "UPDATE": TokenType.UPDATE,
        "USE": TokenType.USE,
        "USING": TokenType.USING,
        "UUID": TokenType.UUID,
        "VALUES": TokenType.VALUES,
        "VIEW": TokenType.VIEW,
        "VOLATILE": TokenType.VOLATILE,
        "WHEN": TokenType.WHEN,
        "WHERE": TokenType.WHERE,
        "WINDOW": TokenType.WINDOW,
        "WITH": TokenType.WITH,
        "APPLY": TokenType.APPLY,
        "ARRAY": TokenType.ARRAY,
        "BIT": TokenType.BIT,
        "BOOL": TokenType.BOOLEAN,
        "BOOLEAN": TokenType.BOOLEAN,
        "BYTE": TokenType.TINYINT,
        "MEDIUMINT": TokenType.MEDIUMINT,
        "INT1": TokenType.TINYINT,
        "TINYINT": TokenType.TINYINT,
        "INT16": TokenType.SMALLINT,
        "SHORT": TokenType.SMALLINT,
        "SMALLINT": TokenType.SMALLINT,
        "HUGEINT": TokenType.INT128,
        "UHUGEINT": TokenType.UINT128,
        "INT2": TokenType.SMALLINT,
        "INTEGER": TokenType.INT,
        "INT": TokenType.INT,
        "INT4": TokenType.INT,
        "INT32": TokenType.INT,
        "INT64": TokenType.BIGINT,
        "INT128": TokenType.INT128,
        "INT256": TokenType.INT256,
        "LONG": TokenType.BIGINT,
        "BIGINT": TokenType.BIGINT,
        "INT8": TokenType.TINYINT,
        "UINT": TokenType.UINT,
        "UINT128": TokenType.UINT128,
        "UINT256": TokenType.UINT256,
        "DEC": TokenType.DECIMAL,
        "DECIMAL": TokenType.DECIMAL,
        "DECIMAL32": TokenType.DECIMAL32,
        "DECIMAL64": TokenType.DECIMAL64,
        "DECIMAL128": TokenType.DECIMAL128,
        "DECIMAL256": TokenType.DECIMAL256,
        "BIGDECIMAL": TokenType.BIGDECIMAL,
        "BIGNUMERIC": TokenType.BIGDECIMAL,
        "LIST": TokenType.LIST,
        "MAP": TokenType.MAP,
        "NULLABLE": TokenType.NULLABLE,
        "NUMBER": TokenType.DECIMAL,
        "NUMERIC": TokenType.DECIMAL,
        "FIXED": TokenType.DECIMAL,
        "REAL": TokenType.FLOAT,
        "FLOAT": TokenType.FLOAT,
        "FLOAT4": TokenType.FLOAT,
        "FLOAT8": TokenType.DOUBLE,
        "DOUBLE": TokenType.DOUBLE,
        "DOUBLE PRECISION": TokenType.DOUBLE,
        "JSON": TokenType.JSON,
        "JSONB": TokenType.JSONB,
        "CHAR": TokenType.CHAR,
        "CHARACTER": TokenType.CHAR,
        "CHAR VARYING": TokenType.VARCHAR,
        "CHARACTER VARYING": TokenType.VARCHAR,
        "NCHAR": TokenType.NCHAR,
        "VARCHAR": TokenType.VARCHAR,
        "VARCHAR2": TokenType.VARCHAR,
        "NVARCHAR": TokenType.NVARCHAR,
        "NVARCHAR2": TokenType.NVARCHAR,
        "BPCHAR": TokenType.BPCHAR,
        "STR": TokenType.TEXT,
        "STRING": TokenType.TEXT,
        "TEXT": TokenType.TEXT,
        "LONGTEXT": TokenType.LONGTEXT,
        "MEDIUMTEXT": TokenType.MEDIUMTEXT,
        "TINYTEXT": TokenType.TINYTEXT,
        "CLOB": TokenType.TEXT,
        "LONGVARCHAR": TokenType.TEXT,
        "LONG VARCHAR": TokenType.LONG_VARCHAR,
        "DECFLOAT": TokenType.DECFLOAT,
        "LONG VARGRAPHIC": TokenType.LONG_VARGRAPHIC,
        "BINARY": TokenType.BINARY,
        "BLOB": TokenType.VARBINARY,
        "LONGBLOB": TokenType.LONGBLOB,
        "MEDIUMBLOB": TokenType.MEDIUMBLOB,
        "TINYBLOB": TokenType.TINYBLOB,
        "BYTEA": TokenType.VARBINARY,
        "VARBINARY": TokenType.VARBINARY,
        "TIME": TokenType.TIME,
        "TIMETZ": TokenType.TIMETZ,
        "TIMESTAMP": TokenType.TIMESTAMP,
        "TIMESTAMPTZ": TokenType.TIMESTAMPTZ,
        "TIMESTAMPLTZ": TokenType.TIMESTAMPLTZ,
        "TIMESTAMP_LTZ": TokenType.TIMESTAMPLTZ,
        "TIMESTAMPNTZ": TokenType.TIMESTAMPNTZ,
        "TIMESTAMP_NTZ": TokenType.TIMESTAMPNTZ,
        "DATE": TokenType.DATE,
        "DATETIME": TokenType.DATETIME,
        "INT4RANGE": TokenType.INT4RANGE,
        "INT4MULTIRANGE": TokenType.INT4MULTIRANGE,
        "INT8RANGE": TokenType.INT8RANGE,
        "INT8MULTIRANGE": TokenType.INT8MULTIRANGE,
        "NUMRANGE": TokenType.NUMRANGE,
        "NUMMULTIRANGE": TokenType.NUMMULTIRANGE,
        "TSRANGE": TokenType.TSRANGE,
        "TSMULTIRANGE": TokenType.TSMULTIRANGE,
        "TSTZRANGE": TokenType.TSTZRANGE,
        "TSTZMULTIRANGE": TokenType.TSTZMULTIRANGE,
        "DATERANGE": TokenType.DATERANGE,
        "DATEMULTIRANGE": TokenType.DATEMULTIRANGE,
        "UNIQUE": TokenType.UNIQUE,
        "VECTOR": TokenType.VECTOR,
        "STRUCT": TokenType.STRUCT,
        "SEQUENCE": TokenType.SEQUENCE,
        "XMLAGG": TokenType.XMLAGG,
        "VARIANT": TokenType.VARIANT,
        "ALTER": TokenType.ALTER,
        "ANALYZE": TokenType.ANALYZE,
        "CALL": TokenType.COMMAND,
        "COMMENT": TokenType.COMMENT,
        "EXPLAIN": TokenType.COMMAND,
        "GRANT": TokenType.GRANT,
        "REVOKE": TokenType.REVOKE,
        "OPTIMIZE": TokenType.COMMAND,
        "PREPARE": TokenType.COMMAND,
        "VACUUM": TokenType.COMMAND,
        "USER-DEFINED": TokenType.USERDEFINED,
        "FOR VERSION": TokenType.VERSION_SNAPSHOT,
        "FOR TIMESTAMP": TokenType.TIMESTAMP_SNAPSHOT,
    }

    WHITE_SPACE: t.Dict[t.Optional[str], TokenType] = {
        " ": TokenType.SPACE,
        "\t": TokenType.SPACE,
        "\n": TokenType.BREAK,
        "\r": TokenType.BREAK,
    }

    COMMANDS = {
        TokenType.COMMAND,
        TokenType.EXECUTE,
        TokenType.FETCH,
        TokenType.SHOW,
        TokenType.RENAME,
    }

    COMMAND_PREFIX_TOKENS = {TokenType.SEMICOLON, TokenType.BEGIN}

    # Handle numeric literals like in hive (3L = BIGINT)
    NUMERIC_LITERALS: t.Dict[str, str] = {}

    COMMENTS = ["--", ("/*", "*/")]

    __slots__ = (
        "sql",
        "size",
        "tokens",
        "dialect",
        "use_rs_tokenizer",
        "_start",
        "_current",
        "_line",
        "_col",
        "_comments",
        "_char",
        "_end",
        "_peek",
        "_prev_token_line",
        "_rs_dialect_settings",
    )

    def __init__(
        self,
        dialect: DialectType = None,
        use_rs_tokenizer: t.Optional[bool] = None,
        **opts: t.Any,
    ) -> None:
        from sqlglot.dialects import Dialect

        self.dialect = Dialect.get_or_raise(dialect)

        # initialize `use_rs_tokenizer`, and allow it to be overwritten per Tokenizer instance
        self.use_rs_tokenizer = (
            use_rs_tokenizer if use_rs_tokenizer is not None else USE_RS_TOKENIZER
        )

        if self.use_rs_tokenizer:
            self._rs_dialect_settings = RsTokenizerDialectSettings(
                unescaped_sequences=self.dialect.UNESCAPED_SEQUENCES,
                identifiers_can_start_with_digit=self.dialect.IDENTIFIERS_CAN_START_WITH_DIGIT,
                numbers_can_be_underscore_separated=self.dialect.NUMBERS_CAN_BE_UNDERSCORE_SEPARATED,
            )

        self.reset()

    def reset(self) -> None:
        self.sql = ""
        self.size = 0
        self.tokens: t.List[Token] = []
        self._start = 0
        self._current = 0
        self._line = 1
        self._col = 0
        self._comments: t.List[str] = []

        self._char = ""
        self._end = False
        self._peek = ""
        self._prev_token_line = -1

    def tokenize(self, sql: str) -> t.List[Token]:
        """Returns a list of tokens corresponding to the SQL string `sql`."""
        if self.use_rs_tokenizer:
            return self.tokenize_rs(sql)

        self.reset()
        self.sql = sql
        self.size = len(sql)

        try:
            self._scan()
        except Exception as e:
            start = max(self._current - 50, 0)
            end = min(self._current + 50, self.size - 1)
            context = self.sql[start:end]
            raise TokenError(f"Error tokenizing '{context}'") from e

        return self.tokens

    def _scan(self, until: t.Optional[t.Callable] = None) -> None:
        """
        主扫描循环：从 `self.sql` 中按字符推进，构造 `self.tokens`。

        参数:
            until: 可选的回调。当该回调返回 True 时提前结束扫描（用于扫描到特定边界，如遇到分号等）。

        流程概览:
            - 批量跳过空格或制表符（不包含换行），减少多次调用 _advance 的开销。
            - 设置当前 token 的起始位置 `_start`，并调用 `_advance` 前进，刷新当前字符与行列位置。
            - 根据当前字符类型分派到对应的扫描函数：数字、带引号标识符、或关键字/运算符/变量。
            - 若提供 `until` 并满足条件，则提前终止主循环。
            - 结束时若仍有挂起的注释，将其附加到最后一个 token 上。
        """
        while self.size and not self._end:
            # 记录当前位置，后续用于"批量跳过空白"
            current = self._current

            # Skip spaces here rather than iteratively calling advance() for performance reasons
            # 为提升性能，在这里一次性跳过连续的空格或制表符（不处理换行符）
            while current < self.size:
                char = self.sql[current]

                if char.isspace() and (char == " " or char == "\t"):
                    current += 1
                else:
                    break

            # 若确实跳过了空白，则偏移量为跳过的字符数；否则至少前进 1 个字符
            offset = current - self._current if current > self._current else 1

            # 设置当前 token 的起始位置到第一个非空白字符
            self._start = current
            # 前进 offset 个字符，同时更新 _char/_peek/_line/_col/_end 等内部指针与状态
            self._advance(offset)

            # 非空白字符：根据类型分派到相应扫描器
            if not self._char.isspace():
                if self._char.isdigit():
                    # 处理数字字面量（含小数点、科学计数、类型后缀等）
                    self._scan_number()
                elif self._char in self._IDENTIFIERS:
                    # 处理带定界符的标识符（如 "..."、`...` 等）
                    self._scan_identifier(self._IDENTIFIERS[self._char])
                else:
                    # 处理关键字、多字符/单字符运算符、注释、或未引号标识符/变量
                    self._scan_keywords()

            # 若设置了提前终止条件且已满足，则跳出主循环
            if until and until():
                break

        # 扫描完成后，如果还有挂起的注释，把它们附加到最后一个 token 上
        if self.tokens and self._comments:
            self.tokens[-1].comments.extend(self._comments)

    def _chars(self, size: int) -> str:
        """
        获取当前位置开始的指定长度的字符序列
        
        用于检查多字符token（如操作符 '>=', '<=', '!=' 等）或字符串分隔符
        
        Args:
            size: 要获取的字符数量
            
        Returns:
            str: 指定长度的字符序列，如果超出字符串边界则返回空字符串
        """
        if size == 1:
            # 单字符情况，直接返回当前字符
            return self._char

        # 计算字符序列的起始和结束位置
        start = self._current - 1  # 当前位置（_current是下一个要读取的位置）
        end = start + size

        # 返回指定范围的字符，如果超出边界则返回空字符串
        return self.sql[start:end] if end <= self.size else ""

    def _advance(self, i: int = 1, alnum: bool = False) -> None:
        """
        向前移动扫描位置
        
        这是词法分析器的核心移动方法，负责更新当前位置、行号、列号等状态信息。
        支持单步移动和连续字母数字字符的快速跳过。
        
        Args:
            i: 要前进的字符数，默认为1
            alnum: 是否连续跳过字母数字字符，用于性能优化
        """
        # 处理换行符的行号更新
        if self.WHITE_SPACE.get(self._char) is TokenType.BREAK:
            # 确保 \r\n 序列不会导致行号重复计算
            # 这是Windows风格换行符的特殊处理
            if not (self._char == "\r" and self._peek == "\n"):
                self._col = i  # 重置列号为前进的字符数
                self._line += 1  # 增加行号
        else:
            # 非换行符，正常增加列号
            self._col += i

        # 更新当前位置和结束状态
        self._current += i
        self._end = self._current >= self.size
        # 更新当前字符和下一个字符
        self._char = self.sql[self._current - 1]
        self._peek = "" if self._end else self.sql[self._current]

        # 性能优化：连续跳过字母数字字符
        if alnum and self._char.isalnum():
            # 使用局部变量而不是实例属性，避免重复的属性访问开销
            _col = self._col
            _current = self._current
            _end = self._end
            _peek = self._peek

            # 连续跳过字母数字字符
            while _peek.isalnum():
                _col += 1
                _current += 1
                _end = _current >= self.size
                _peek = "" if _end else self.sql[_current]

            # 将局部变量的值更新回实例属性
            self._col = _col
            self._current = _current
            self._end = _end
            self._peek = _peek
            self._char = self.sql[_current - 1]

    @property
    def _text(self) -> str:
        """
        获取当前token的文本内容
        
        返回从token开始位置到当前位置的所有字符，用于构建token的文本值
        
        Returns:
            str: 当前token的文本内容
        """
        return self.sql[self._start : self._current]

    def _add(self, token_type: TokenType, text: t.Optional[str] = None) -> None:
        """
        添加新的token到token列表
        
        这是词法分析器的核心方法，负责创建和添加token，处理注释关联，
        以及处理特殊命令token的后续字符串解析。
        
        Args:
            token_type: token的类型
            text: token的文本内容，如果为None则使用当前位置的文本
        """
        # 记录前一个token的行号，用于注释关联判断
        self._prev_token_line = self._line

        # 特殊处理：分号前的注释附加到前一个token
        # 这确保了语句结束时的注释能正确关联
        if self._comments and token_type == TokenType.SEMICOLON and self.tokens:
            self.tokens[-1].comments.extend(self._comments)
            self._comments = []

        # 创建并添加新的token
        self.tokens.append(
            Token(
                token_type,
                text=self._text if text is None else text,  # 使用指定文本或当前位置文本
                line=self._line,      # 行号
                col=self._col,        # 列号
                start=self._start,    # 开始位置
                end=self._current - 1,  # 结束位置
                comments=self._comments,  # 关联的注释
            )
        )
        # 清空当前注释列表，为下一个token准备
        self._comments = []

        # 特殊处理：命令token后的字符串解析
        # 某些SQL命令（如EXECUTE, PREPARE等）后面可能跟着字符串参数
        if (
            token_type in self.COMMANDS  # 当前token是命令类型
            and self._peek != ";"        # 下一个字符不是分号
            and (len(self.tokens) == 1 or self.tokens[-2].token_type in self.COMMAND_PREFIX_TOKENS)  # 是第一个token或前一个token是命令前缀
        ):
            # 记录开始位置和当前token数量
            start = self._current
            tokens = len(self.tokens)
            # 扫描直到遇到分号，但不添加token
            self._scan(lambda: self._peek == ";")
            # 恢复token列表到命令token之后的状态
            self.tokens = self.tokens[:tokens]
            # 提取命令后的字符串内容
            text = self.sql[start : self._current].strip()
            if text:
                # 将命令后的内容作为字符串token添加
                self._add(TokenType.STRING, text)

    def _scan_keywords(self) -> None:
        """
        关键字/操作符/注释/字符串/变量 的分派扫描器。

        工作原理:
            - 利用 `_KEYWORD_TRIE` 在当前位置基于大小写不敏感匹配尝试最长可用的关键字/多字符操作符/注释起始符等。
            - 将连续空白折叠为单个空格，以便正确匹配包含空格的多词关键字（例如 "GROUP BY"、"ORDER BY"）。
            - 若命中候选 `word`，优先尝试将其解释为字符串起始或注释起始；若成立则进入对应扫描分支。
            - 否则在边界条件满足时（后继为空白、下一个是单字符 token、或已到末尾）将 `word` 视为关键字并产出 token。
            - 若没有匹配到 `word`，尝试识别单字符 token；都没有则回退到 `_scan_var` 扫描普通标识符/变量。
        """
        size = 0
        word = None
        chars = self._text
        char = chars
        prev_space = False
        skip = False
        trie = self._KEYWORD_TRIE
        single_token = char in self.SINGLE_TOKENS

        # 逐字符扩展 `chars`（把连续空白折叠为单个空格），并在 Trie 中推进匹配状态
        while chars:
            if skip:
                # 连续空白的第二个及之后的空白不参与 Trie 匹配，但保留"仍可能是前缀"的状态
                result = TrieResult.PREFIX
            else:
                # 大小写不敏感匹配，统一为大写推进 Trie
                result, trie = in_trie(trie, char.upper())

            if result == TrieResult.FAILED:
                # 无法继续匹配
                break
            if result == TrieResult.EXISTS:
                # 当前 `chars` 是一个有效的词条（可能还有更长的），记录为候选
                word = chars

            end = self._current + size
            size += 1

            if end < self.size:
                char = self.sql[end]
                # 如果后继字符是单字符 token（如 , ( ) 等），需记录以便后续判断边界
                single_token = single_token or char in self.SINGLE_TOKENS
                is_space = char.isspace()

                if not is_space or not prev_space:
                    # 首次遇到空白，折叠为空格加入；否则直接加入字符
                    if is_space:
                        char = " "
                    chars += char
                    prev_space = is_space
                    skip = False
                else:
                    # 避免将第二个及之后的连续空白并入匹配序列
                    skip = True
            else:
                # 已到输入末尾
                char = ""
                break

        if word:
            # 优先判断是否为字符串/格式化字符串起始
            if self._scan_string(word):
                return
            # 其次判断是否为注释起始
            if self._scan_comment(word):
                return
            # 到达边界（后继是空白、单字符 token、或输入末尾），确认为关键字并产出 token
            if prev_space or single_token or not char:
                self._advance(size - 1)
                word = word.upper()
                self._add(self.KEYWORDS[word], text=word)
                return

        # 若当前字符本身就是单字符 token，直接产出
        if self._char in self.SINGLE_TOKENS:
            self._add(self.SINGLE_TOKENS[self._char], text=self._char)
            return

        # 否则将其视为普通变量/未引号标识符继续扫描
        self._scan_var()

    def _scan_comment(self, comment_start: str) -> bool:
        """
        扫描并处理SQL注释
        
        Args:
            comment_start: 注释开始标记（如 '--', '/*', '#' 等）
            
        Returns:
            bool: 如果成功识别并处理了注释则返回True，否则返回False
        """
        # 检查注释开始标记是否在当前方言支持的注释类型中
        if comment_start not in self._COMMENTS:
            return False

        # 记录注释开始的行号和长度，用于后续判断注释位置
        comment_start_line = self._line
        comment_start_size = len(comment_start)
        comment_end = self._COMMENTS[comment_start]  # 获取对应的注释结束标记

        if comment_end:
            # 处理块注释（如 /* ... */）
            # 跳过注释开始标记
            self._advance(comment_start_size)

            # 使用计数器处理嵌套注释，某些方言支持嵌套注释
            comment_count = 1
            comment_end_size = len(comment_end)

            # 扫描直到找到匹配的注释结束标记
            while not self._end:
                # 检查是否遇到注释结束标记
                if self._chars(comment_end_size) == comment_end:
                    comment_count -= 1
                    if not comment_count:  # 当计数器归零时，找到最外层的注释结束
                        break

                self._advance(alnum=True)

                # 处理嵌套注释：某些方言（如databricks, duckdb, postgres）支持嵌套注释
                # 当遇到新的注释开始标记时，增加嵌套计数
                if (
                    self.NESTED_COMMENTS
                    and not self._end
                    and self._chars(comment_end_size) == comment_start
                ):
                    self._advance(comment_start_size)
                    comment_count += 1

            # 提取注释内容（去掉开始和结束标记）
            self._comments.append(self._text[comment_start_size : -comment_end_size + 1])
            self._advance(comment_end_size - 1)
        else:
            # 处理行注释（如 -- 或 #），直到行尾或遇到换行符
            while not self._end and self.WHITE_SPACE.get(self._peek) is not TokenType.BREAK:
                self._advance(alnum=True)
            # 提取注释内容（去掉开始标记）
            self._comments.append(self._text[comment_start_size:])

        # 检查是否为查询提示注释（如 /*+ ... */）
        # 如果注释开始标记是提示标记，且前一个token是允许提示的SQL关键字
        if (
            comment_start == self.HINT_START
            and self.tokens
            and self.tokens[-1].token_type in self.TOKENS_PRECEDING_HINT
        ):
            self._add(TokenType.HINT)

        # 处理注释与token的关联关系：
        # - 前导注释（leading comment）附加到后续的token上
        # - 尾随注释（trailing comment）附加到前一个token上
        # - 多个连续注释通过追加到当前注释列表来保留
        if comment_start_line == self._prev_token_line:
            # 如果注释与前一个token在同一行，则附加到前一个token
            self.tokens[-1].comments.extend(self._comments)
            self._comments = []
            self._prev_token_line = self._line

        return True

    def _scan_number(self) -> None:
        """
        扫描并处理数字字面量
        
        支持多种数字格式：
        - 普通数字：123, 123.45
        - 科学计数法：1.23e-4
        - 二进制：0b1010
        - 十六进制：0x1A
        - 带类型后缀：123::INTEGER
        - 下划线分隔：1_000_000
        """
        # 处理以0开头的特殊数字格式
        if self._char == "0":
            peek = self._peek.upper()
            if peek == "B":
                # 二进制数字（如 0b1010）
                return self._scan_bits() if self.BIT_STRINGS else self._add(TokenType.NUMBER)
            elif peek == "X":
                # 十六进制数字（如 0x1A）
                return self._scan_hex() if self.HEX_STRINGS else self._add(TokenType.NUMBER)

        # 初始化数字扫描状态
        decimal = False  # 是否已遇到小数点
        scientific = 0   # 科学计数法状态：0=未开始，1=遇到E，2=遇到符号

        # 循环扫描数字的各个部分
        while True:
            if self._peek.isdigit():
                # 遇到数字字符，继续扫描
                self._advance()
            elif self._peek == "." and not decimal:
                # 遇到小数点且之前没有小数点
                # 特殊处理：如果前一个token是参数标记，则停止扫描
                if self.tokens and self.tokens[-1].token_type == TokenType.PARAMETER:
                    return self._add(TokenType.NUMBER)
                decimal = True
                self._advance()
            elif self._peek in ("-", "+") and scientific == 1:
                # 科学计数法的符号部分（如 1.23e-4 中的 -）
                scientific += 1
                self._advance()
            elif self._peek.upper() == "E" and not scientific:
                # 科学计数法的指数标记（如 1.23e4 中的 e）
                scientific += 1
                self._advance()
            elif self._peek.isidentifier():
                # 遇到标识符字符，可能是类型后缀或下划线分隔符
                number_text = self._text  # 保存当前数字部分
                literal = ""

                # 收集后续的标识符字符
                while self._peek.strip() and self._peek not in self.SINGLE_TOKENS:
                    literal += self._peek
                    self._advance()

                # 检查是否为数字类型后缀（如 INTEGER, BIGINT 等）
                token_type = self.KEYWORDS.get(self.NUMERIC_LITERALS.get(literal.upper(), ""))

                if token_type:
                    # 发现类型后缀，生成三个token：数字、双冒号、类型
                    # 例如：123::INTEGER
                    self._add(TokenType.NUMBER, number_text)
                    self._add(TokenType.DCOLON, "::")
                    return self._add(token_type, literal)
                else:
                    # 没有找到类型后缀，检查其他可能性
                    replaced = literal.replace("_", "")
                    # 检查是否为下划线分隔的数字（如 1_000_000）
                    if self.dialect.NUMBERS_CAN_BE_UNDERSCORE_SEPARATED and replaced.isdigit():
                        return self._add(TokenType.NUMBER, number_text + replaced)
                    # 检查方言是否允许标识符以数字开头
                    if self.dialect.IDENTIFIERS_CAN_START_WITH_DIGIT:
                        return self._add(TokenType.VAR)

                # 回退扫描位置，将数字部分作为普通数字处理
                self._advance(-len(literal))
                return self._add(TokenType.NUMBER, number_text)
            else:
                # 遇到其他字符，数字扫描结束
                return self._add(TokenType.NUMBER)

    def _scan_bits(self) -> None:
        """
        扫描二进制数字字面量（如 0b1010）
        
        处理以 '0b' 开头的二进制数字，验证其有效性并生成相应的token
        """
        # 跳过 'b' 字符，因为已经在 _scan_number 中处理了 '0'
        self._advance()
        # 提取完整的二进制数字字符串（包括 '0b' 前缀）
        value = self._extract_value()
        try:
            # 尝试将字符串转换为二进制数字，验证其有效性
            # 如果转换失败，说明不是有效的二进制数字
            int(value, 2)
            # 生成 BIT_STRING token，去掉 '0b' 前缀，只保留二进制数字部分
            self._add(TokenType.BIT_STRING, value[2:])  # Drop the 0b
        except ValueError:
            # 如果无法转换为二进制数字，则将其作为普通标识符处理
            # 这处理了像 '0bxyz' 这样的无效二进制格式
            self._add(TokenType.IDENTIFIER)

    def _scan_hex(self) -> None:
        """
        扫描十六进制数字字面量（如 0x1A）
        
        处理以 '0x' 开头的十六进制数字，验证其有效性并生成相应的token
        """
        # 跳过 'x' 字符，因为已经在 _scan_number 中处理了 '0'
        self._advance()
        # 提取完整的十六进制数字字符串（包括 '0x' 前缀）
        value = self._extract_value()
        try:
            # 尝试将字符串转换为十六进制数字，验证其有效性
            # 如果转换失败，说明不是有效的十六进制数字
            int(value, 16)
            # 生成 HEX_STRING token，去掉 '0x' 前缀，只保留十六进制数字部分
            self._add(TokenType.HEX_STRING, value[2:])  # Drop the 0x
        except ValueError:
            # 如果无法转换为十六进制数字，则将其作为普通标识符处理
            # 这处理了像 '0xghz' 这样的无效十六进制格式
            self._add(TokenType.IDENTIFIER)

    def _extract_value(self) -> str:
        """
        提取当前扫描位置的值
        
        从当前位置开始，收集连续的字符直到遇到分隔符或特殊token
        
        Returns:
            str: 提取的字符串值
        """
        # 循环收集字符，直到遇到分隔符或特殊token
        while True:
            char = self._peek.strip()
            # 如果字符存在且不是单字符token，则继续收集
            if char and char not in self.SINGLE_TOKENS:
                self._advance(alnum=True)
            else:
                # 遇到分隔符或特殊token，停止收集
                break

        # 返回从开始位置到当前位置的所有文本
        return self._text

    def _scan_string(self, start: str) -> bool:
        """
        扫描字符串字面量
        
        支持多种字符串格式：
        - 普通字符串：'text', "text"
        - 原始字符串：r'text'
        - 十六进制字符串：x'1A2B'
        - 二进制字符串：b'1010'
        - Here文档字符串：$tag$text$tag$
        
        Args:
            start: 字符串开始标记
            
        Returns:
            bool: 如果成功识别并处理了字符串则返回True，否则返回False
        """
        base = None  # 数字进制（用于十六进制和二进制字符串）
        token_type = TokenType.STRING  # 默认字符串类型

        # 根据开始标记确定字符串类型和结束标记
        if start in self._QUOTES:
            # 普通引号字符串（单引号或双引号）
            end = self._QUOTES[start]
        elif start in self._FORMAT_STRINGS:
            # 格式化字符串（如十六进制、二进制、原始字符串等）
            end, token_type = self._FORMAT_STRINGS[start]

            # 根据字符串类型设置相应的数字进制
            if token_type == TokenType.HEX_STRING:
                base = 16  # 十六进制
            elif token_type == TokenType.BIT_STRING:
                base = 2   # 二进制
            elif token_type == TokenType.HEREDOC_STRING:
                # 处理Here文档字符串（如PostgreSQL的 $tag$...$tag$ 格式）
                self._advance()

                # 检查是否立即遇到结束标记（空标签）
                if self._char == end:
                    tag = ""
                else:
                    # 提取标签内容
                    tag = self._extract_string(
                        end,
                        raw_string=True,
                        raise_unmatched=not self.HEREDOC_TAG_IS_IDENTIFIER,
                    )

                # 验证标签的有效性
                # 如果标签必须是标识符但实际不是，则回退处理
                if (
                    tag
                    and self.HEREDOC_TAG_IS_IDENTIFIER
                    and (self._end or tag.isdigit() or any(c.isspace() for c in tag))
                ):
                    if not self._end:
                        self._advance(-1)

                    # 回退到标签开始位置
                    self._advance(-len(tag))
                    # 使用替代token类型处理
                    self._add(self.HEREDOC_STRING_ALTERNATIVE)
                    return True

                # 构造完整的结束标记（开始标记 + 标签 + 结束标记）
                end = f"{start}{tag}{end}"
        else:
            # 不支持的字符串格式
            return False

        # 跳过开始标记
        self._advance(len(start))
        # 提取字符串内容，根据类型决定是否为原始字符串
        text = self._extract_string(end, raw_string=token_type == TokenType.RAW_STRING)

        # 对于数字字符串（十六进制、二进制），验证其有效性
        if base and text:
            try:
                # 尝试将字符串转换为指定进制的数字
                int(text, base)
            except Exception:
                # 如果转换失败，抛出错误
                raise TokenError(
                    f"Numeric string contains invalid characters from {self._line}:{self._start}"
                )

        # 生成相应的token
        self._add(token_type, text)
        return True

    def _scan_identifier(self, identifier_end: str) -> None:
        self._advance()
        text = self._extract_string(
            identifier_end, escapes=self._IDENTIFIER_ESCAPES | {identifier_end}
        )
        self._add(TokenType.IDENTIFIER, text)

    def _scan_var(self) -> None:
        """
        扫描未引号标识符/变量。

        规则:
            - 连续读取下一个非空白字符（`_peek.strip()`），只要它不是"硬边界"的单字符 token，
              或者该字符被允许出现在变量中（`VAR_SINGLE_TOKENS`，由方言定制）。
            - 使用 `alnum=True` 加速前进，尽可能一次性跨过连续的字母数字片段以提升性能。
            - 扫描结束后：
                * 如果前一个 token 是参数标记（如 `@`，`TokenType.PARAMETER`），则强制将本片段作为变量 `VAR`；
                * 否则将本片段尝试按关键字表（不区分大小写）解析为关键字；若非关键字，则作为 `VAR`。
        """
        while True:
            char = self._peek.strip()
            if char and (char in self.VAR_SINGLE_TOKENS or char not in self.SINGLE_TOKENS):
                # 允许的单字符（方言自定义）或非单字符分隔符：继续前进
                self._advance(alnum=True)
            else:
                # 遇到硬边界（如逗号、括号、运算符等）或输入结束：停止
                break

        # 若前一个 token 是参数起始（如 @x），则无条件视为变量；
        # 否则先尝试关键字映射（大写匹配），失败则退化为变量。
        self._add(
            TokenType.VAR
            if self.tokens and self.tokens[-1].token_type == TokenType.PARAMETER
            else self.KEYWORDS.get(self._text.upper(), TokenType.VAR)
        )

    def _extract_string(
        self,
        delimiter: str,
        escapes: t.Optional[t.Set[str]] = None,
        raw_string: bool = False,
        raise_unmatched: bool = True,
    ) -> str:
        """
        提取字符串内容，处理转义字符和特殊序列
        
        这是字符串解析的核心方法，负责从当前位置提取字符串内容直到遇到结束分隔符。
        支持转义字符处理、原始字符串、未转义序列等高级特性。
        
        Args:
            delimiter: 字符串结束分隔符
            escapes: 转义字符集合，如果为None则使用默认的STRING_ESCAPES
            raw_string: 是否为原始字符串（不处理转义）
            raise_unmatched: 是否在找不到匹配分隔符时抛出异常
            
        Returns:
            str: 提取的字符串内容
        """
        text = ""
        delim_size = len(delimiter)
        # 如果没有指定转义字符，使用默认的字符串转义字符集合
        escapes = self._STRING_ESCAPES if escapes is None else escapes

        # 主循环：逐字符处理字符串内容
        while True:
            # 处理未转义序列（某些方言支持的特殊序列，如PostgreSQL的$$）
            if (
                not raw_string  # 原始字符串不处理未转义序列
                and self.dialect.UNESCAPED_SEQUENCES  # 方言支持未转义序列
                and self._peek  # 还有下一个字符
                and self._char in self.STRING_ESCAPES  # 当前字符是转义字符
            ):
                # 检查当前字符+下一个字符是否构成未转义序列
                unescaped_sequence = self.dialect.UNESCAPED_SEQUENCES.get(self._char + self._peek)
                if unescaped_sequence:
                    # 找到未转义序列，跳过两个字符并添加对应的内容
                    self._advance(2)
                    text += unescaped_sequence
                    continue
            
            # 处理转义字符
            if (
                (self.STRING_ESCAPES_ALLOWED_IN_RAW_STRINGS or not raw_string)  # 原始字符串是否允许转义
                and self._char in escapes  # 当前字符是转义字符
                and (self._peek == delimiter or self._peek in escapes)  # 下一个字符是分隔符或转义字符
                and (self._char not in self._QUOTES or self._char == self._peek)  # 引号字符的特殊处理
            ):
                # 处理转义序列
                if self._peek == delimiter:
                    # 转义分隔符：将分隔符作为普通字符添加到字符串中
                    text += self._peek
                else:
                    # 转义转义字符：将转义字符本身添加到字符串中
                    text += self._char + self._peek

                # 检查是否还有更多字符可读
                if self._current + 1 < self.size:
                    self._advance(2)  # 跳过转义字符和下一个字符
                else:
                    # 到达字符串末尾但转义序列不完整，抛出错误
                    raise TokenError(f"Missing {delimiter} from {self._line}:{self._current}")
            else:
                # 检查是否遇到字符串结束分隔符
                if self._chars(delim_size) == delimiter:
                    # 找到结束分隔符
                    if delim_size > 1:
                        # 多字符分隔符需要额外前进
                        self._advance(delim_size - 1)
                    break  # 字符串提取完成

                # 检查是否到达输入末尾
                if self._end:
                    if not raise_unmatched:
                        # 不抛出异常，返回当前已提取的内容加上当前字符
                        return text + self._char

                    # 抛出未匹配分隔符错误
                    raise TokenError(f"Missing {delimiter} from {self._line}:{self._start}")

                # 记录当前位置，用于提取字符范围
                current = self._current - 1
                # 前进到下一个字符
                self._advance(alnum=True)
                # 将当前位置到新位置之间的字符添加到结果中
                text += self.sql[current : self._current - 1]

        return text

    def tokenize_rs(self, sql: str) -> t.List[Token]:
        """
        使用Rust实现的词法分析器进行token化
        
        这是对Rust版本词法分析器的Python包装，提供更高的性能。
        当Rust tokenizer可用时，优先使用它来提升解析速度。
        
        Args:
            sql: 要解析的SQL字符串
            
        Returns:
            t.List[Token]: 解析得到的token列表
            
        Raises:
            SqlglotError: 当Rust tokenizer不可用时
            TokenError: 当解析过程中出现错误时
        """
        # 检查Rust tokenizer是否可用
        if not self._RS_TOKENIZER:
            raise SqlglotError("Rust tokenizer is not available")

        # 调用Rust实现的tokenizer进行解析
        # 返回tokens列表和可能的错误信息
        tokens, error_msg = self._RS_TOKENIZER.tokenize(sql, self._rs_dialect_settings)
        
        # 将Rust返回的token类型索引转换为Python的TokenType枚举
        for token in tokens:
            token.token_type = _ALL_TOKEN_TYPES[token.token_type_index]

        # 设置tokens属性，即使解析失败也能检查部分结果
        # 这对于调试和错误处理很有用
        self.tokens = tokens

        # 如果Rust tokenizer报告了错误，抛出Python异常
        if error_msg is not None:
            raise TokenError(error_msg)

        return tokens
