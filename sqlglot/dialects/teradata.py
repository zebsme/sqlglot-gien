from __future__ import annotations

import typing as t

from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
    Dialect,
    max_or_greatest,
    min_or_least,
    rename_func,
    strposition_sql,
    to_number_with_nls_param,
)
from sqlglot.helper import seq_get
from sqlglot.tokens import TokenType


def _date_add_sql(
    kind: t.Literal["+", "-"],
) -> t.Callable[[Teradata.Generator, exp.DateAdd | exp.DateSub], str]:
    """
    生成 Teradata 日期加减运算的 SQL 函数
    
    Args:
        kind: 运算类型，"+" 表示加法，"-" 表示减法
        
    Returns:
        返回一个函数，用于生成 Teradata 格式的日期运算 SQL
        
    功能说明:
        - 处理 DateAdd 和 DateSub 表达式
        - 将日期运算转换为 Teradata 特定的语法格式
        - 支持负数值的处理，会自动调整运算符号
    """
    def func(self: Teradata.Generator, expression: exp.DateAdd | exp.DateSub) -> str:
        # 获取日期表达式的基础部分
        this = self.sql(expression, "this")
        # 获取时间单位（如 DAY, MONTH, YEAR 等）
        unit = expression.args.get("unit")
        # 简化表达式，除非是字面量
        value = self._simplify_unless_literal(expression.expression)

        # 检查值是否为字面量，Teradata 不支持非字面量的日期运算
        if not isinstance(value, exp.Literal):
            self.unsupported("Cannot add non literal")

        # 处理负数值的情况
        if isinstance(value, exp.Neg):
            # 如果值是负数，需要反转运算符号
            kind_to_op = {"+": "-", "-": "+"}
            value = exp.Literal.string(value.this.to_py())
        else:
            # 正常情况下保持原运算符号
            kind_to_op = {"+": "+", "-": "-"}
            value.set("is_string", True)

        # 生成最终的 SQL：日期 +/- INTERVAL 值 单位
        return f"{this} {kind_to_op[kind]} {self.sql(exp.Interval(this=value, unit=unit))}"

    return func


class Teradata(Dialect):
    """
    Teradata 数据库方言类
    
    功能说明:
        - 定义 Teradata 数据库的特定语法和行为
        - 继承自基础 Dialect 类，实现 Teradata 特有的 SQL 转换
        - 包含词法分析器、语法分析器和代码生成器
    """
    # Teradata 不支持半连接和反连接
    SUPPORTS_SEMI_ANTI_JOIN = False
    # Teradata 支持类型化除法（返回精确的数值类型）
    TYPED_DIVISION = True

    # Teradata 时间格式映射表，将 Teradata 格式转换为 Python strftime 格式
    TIME_MAPPING = {
        "YY": "%y",      # 两位年份
        "Y4": "%Y",      # 四位年份
        "YYYY": "%Y",    # 四位年份
        "M4": "%B",      # 完整月份名称
        "M3": "%b",      # 缩写月份名称
        "M": "%-M",      # 月份数字（不补零）
        "MI": "%M",      # 分钟
        "MM": "%m",      # 月份数字（补零）
        "MMM": "%b",     # 缩写月份名称
        "MMMM": "%B",    # 完整月份名称
        "D": "%-d",      # 日期（不补零）
        "DD": "%d",      # 日期（补零）
        "D3": "%j",      # 一年中的第几天（3位）
        "DDD": "%j",     # 一年中的第几天
        "H": "%-H",      # 小时（不补零）
        "HH": "%H",      # 小时（补零）
        "HH24": "%H",    # 24小时制小时
        "S": "%-S",      # 秒（不补零）
        "SS": "%S",      # 秒（补零）
        "SSSSSS": "%f",  # 微秒
        "E": "%a",       # 缩写星期名称
        "EE": "%a",      # 缩写星期名称
        "E3": "%a",      # 缩写星期名称
        "E4": "%A",      # 完整星期名称
        "EEE": "%a",     # 缩写星期名称
        "EEEE": "%A",    # 完整星期名称
    }

    class Tokenizer(tokens.Tokenizer):
        """
        Teradata 词法分析器
        
        功能说明:
            - 继承自基础 Tokenizer 类
            - 定义 Teradata 特有的关键字和语法规则
            - 处理十六进制字符串和特殊操作符
        """
        # 十六进制字符串格式，虽然 Teradata 文档中没有明确提及，但经测试可用
        HEX_STRINGS = [("X'", "'"), ("x'", "'"), ("0x", "")]
        
        # Teradata 关键字映射表
        # 参考文档：
        # https://docs.teradata.com/r/Teradata-Database-SQL-Functions-Operators-Expressions-and-Predicates/March-2017/Comparison-Operators-and-Functions/Comparison-Operators/ANSI-Compliance
        # https://docs.teradata.com/r/SQL-Functions-Operators-Expressions-and-Predicates/June-2017/Arithmetic-Trigonometric-Hyperbolic-Operators/Functions
        KEYWORDS = {
            **tokens.Tokenizer.KEYWORDS,
            "**": TokenType.DSTAR,           # 幂运算符
            "^=": TokenType.NEQ,             # 不等于操作符
            "BYTEINT": TokenType.BYTEINT,   # BYTEINT 数据类型映射为 BYTEINT
            "COLLECT": TokenType.COMMAND,    # COLLECT 统计命令
            # "CONTENT": TokenType.CONTENT,    # RETURNING CONTENT 关键字
            "DEL": TokenType.DELETE,         # DELETE 的缩写形式
            "EQ": TokenType.EQ,              # 等于操作符
            "GE": TokenType.GTE,             # 大于等于操作符
            "GT": TokenType.GT,              # 大于操作符
            "HELP": TokenType.COMMAND,       # HELP 命令
            "INS": TokenType.INSERT,         # INSERT 的缩写形式
            "LE": TokenType.LTE,             # 小于等于操作符
            "LOCKING": TokenType.LOCK,       # 锁定语句关键字
            "LT": TokenType.LT,              # 小于操作符
            "MINUS": TokenType.EXCEPT,       # MINUS 操作符（等同于 EXCEPT）
            "MOD": TokenType.MOD,            # MOD 操作符
            "NE": TokenType.NEQ,             # 不等于操作符
            "NOT=": TokenType.NEQ,           # 不等于操作符的另一种形式
            "SAMPLE": TokenType.TABLE_SAMPLE, # 表采样关键字
            "SEL": TokenType.SELECT,         # SELECT 的缩写形式
            "ST_GEOMETRY": TokenType.GEOMETRY, # 空间几何数据类型
            "TOP": TokenType.TOP,            # TOP 限制关键字
            "UPD": TokenType.UPDATE,         # UPDATE 的缩写形式
            "XMLAGG": TokenType.XMLAGG,      # XMLAGG 函数关键字
            "MOD": TokenType.MOD,            # MOD 操作符
        }
        # 移除 Oracle 风格的提示注释支持
        KEYWORDS.pop("/*+")

        # Teradata 支持 % 作为取模运算符
        SINGLE_TOKENS = {**tokens.Tokenizer.SINGLE_TOKENS}

    class Parser(parser.Parser):
        """
        Teradata 语法分析器
        
        功能说明:
            - 继承自基础 Parser 类
            - 实现 Teradata 特有的 SQL 语法解析
            - 支持 Teradata 特殊语句和函数的解析
        """
        # 表采样支持 CSV 格式
        TABLESAMPLE_CSV = True
        # VALUES 子句后不需要括号
        VALUES_FOLLOWED_BY_PAREN = False

        # Teradata 字符集转换器列表
        # 用于 TRANSLATE 函数中的字符集转换
        CHARSET_TRANSLATORS = {
            "GRAPHIC_TO_KANJISJIS",
            "GRAPHIC_TO_LATIN",
            "GRAPHIC_TO_UNICODE",
            "GRAPHIC_TO_UNICODE_PadSpace",
            "KANJI1_KanjiEBCDIC_TO_UNICODE",
            "KANJI1_KanjiEUC_TO_UNICODE",
            "KANJI1_KANJISJIS_TO_UNICODE",
            "KANJI1_SBC_TO_UNICODE",
            "KANJISJIS_TO_GRAPHIC",
            "KANJISJIS_TO_LATIN",
            "KANJISJIS_TO_UNICODE",
            "LATIN_TO_GRAPHIC",
            "LATIN_TO_KANJISJIS",
            "LATIN_TO_UNICODE",
            "LOCALE_TO_UNICODE",
            "UNICODE_TO_GRAPHIC",
            "UNICODE_TO_GRAPHIC_PadGraphic",
            "UNICODE_TO_GRAPHIC_VarGraphic",
            "UNICODE_TO_KANJI1_KanjiEBCDIC",
            "UNICODE_TO_KANJI1_KanjiEUC",
            "UNICODE_TO_KANJI1_KANJISJIS",
            "UNICODE_TO_KANJI1_SBC",
            "UNICODE_TO_KANJISJIS",
            "UNICODE_TO_LATIN",
            "UNICODE_TO_LOCALE",
            "UNICODE_TO_UNICODE_FoldSpace",
            "UNICODE_TO_UNICODE_Fullwidth",
            "UNICODE_TO_UNICODE_Halfwidth",
            "UNICODE_TO_UNICODE_NFC",
            "UNICODE_TO_UNICODE_NFD",
            "UNICODE_TO_UNICODE_NFKC",
            "UNICODE_TO_UNICODE_NFKD",
        }

        # Teradata 数据类型关键字列表
        # 用于识别 expr(datatype) 语法中的数据类型
        DATA_TYPE_KEYWORDS = {
            # ARRAY 
            "ARRAY", "VARRAY",
            # BYTE
            "BLOB", "BYTE", "VARBYTE",
            # NUMERIC
            "BIGINT", "BYTEINT", "DECIMAL",
            "DOUBLE PRECISION", "FLOAT", 
            "INTEGER", "NUMBER", "NUMERIC", "REAL", "SMALLINT", 
            # DATETIME
            "DATE", "TIME", "TIMESTAMP", 
            # INTERVAL
            "INTERVAL",
            # CHARACTER
            "CHAR", "CHARACTER", "CHARACTER SET",
            "CLOB", "CHAR VARYING", "LONG VARCHAR", "VARCHAR",
            # PERIOD
            "PERIOD", 
            # Complex Data Types
            "GEOMETRY", "ST_GEOMETRY", "GRAPHIC", "VARGRAPHIC",
            "JSON", "XML", "MULTISET", "VARIANT", "SYSUDTLIB",
        }

        # 函数标记集合，移除 REPLACE 因为在 Teradata 中它是语句关键字
        FUNC_TOKENS = {*parser.Parser.FUNC_TOKENS}
        FUNC_TOKENS.remove(TokenType.REPLACE)
        FUNC_TOKENS.add(TokenType.XMLAGG)  # 添加 XMLAGG 函数标记

        # Teradata 特有的语句解析器映射
        STATEMENT_PARSERS = {
            **parser.Parser.STATEMENT_PARSERS,
            # DATABASE 语句解析为 USE 表达式
            TokenType.DATABASE: lambda self: self.expression(
                exp.Use, this=self._parse_table(schema=False)
            ),
            # REPLACE 语句解析为 CREATE 语句
            TokenType.REPLACE: lambda self: self._parse_create(),
            # LOCKING 语句的专门解析器
            TokenType.LOCK: lambda self: self._parse_locking_statement(),
        }

        def _parse_locking_statement(self) -> exp.LockingStatement:
            """
            解析 Teradata LOCKING 语句
            
            功能说明:
                - 解析 LOCKING 子句和后续的 SELECT 语句
                - 创建 LockingStatement 表达式节点
                - 确保 LOCKING 子句后必须跟 SELECT 语句
            """
            # 复用 LockingProperty 解析器来处理锁类型等信息
            locking_property = self._parse_locking()
            # 解析被包装的查询语句
            wrapped_query = self._parse_select()

            # 验证 LOCKING 子句后必须有 SELECT 语句
            if not wrapped_query:
                self.raise_error("Expected SELECT statement after LOCKING clause")

            # 创建 LockingStatement 表达式
            return self.expression(
                exp.LockingStatement,
                this=locking_property,
                expression=wrapped_query,
            )

        # SET 语句解析器，添加 QUERY_BAND 支持
        SET_PARSERS = {
            **parser.Parser.SET_PARSERS,
            "QUERY_BAND": lambda self: self._parse_query_band(),
        }

        # Teradata 特有函数解析器
        FUNCTION_PARSERS = {
            **parser.Parser.FUNCTION_PARSERS,
            # TRYCAST 函数映射到 TRY_CAST 解析器
            # 参考：https://docs.teradata.com/r/SQL-Functions-Operators-Expressions-and-Predicates/June-2017/Data-Type-Conversions/TRYCAST
            "TRYCAST": parser.Parser.FUNCTION_PARSERS["TRY_CAST"],
            # RANGE_N 函数的专门解析器
            "RANGE_N": lambda self: self._parse_rangen(),
            # TRANSLATE 函数的专门解析器（字符集转换）
            "TRANSLATE": lambda self: self._parse_translate(),
            # XMLAGG 函数的专门解析器
            "XMLAGG": lambda self: self._parse_xmlagg(),
        }

        # Teradata 函数映射表
        FUNCTIONS = {
            **parser.Parser.FUNCTIONS,
            # CARDINALITY 函数映射为数组大小函数
            "CARDINALITY": exp.ArraySize.from_arg_list,
            # RANDOM 函数映射为 Rand 表达式，支持上下界参数
            "RANDOM": lambda args: exp.Rand(lower=seq_get(args, 0), upper=seq_get(args, 1)),
        }

        # 指数运算符映射
        EXPONENT = {
            TokenType.DSTAR: exp.Pow,  # ** 操作符映射为幂运算
        }

        def _parse_translate(self) -> exp.TranslateCharacters:
            """
            解析 Teradata TRANSLATE 函数
            
            功能说明:
                - 解析字符集转换函数
                - 支持 USING 子句指定转换器
                - 支持 WITH ERROR 选项
            
            语法: TRANSLATE(expression USING charset_translator [WITH ERROR])
            """
            # 解析要转换的表达式
            this = self._parse_assignment()
            # 匹配 USING 关键字
            self._match(TokenType.USING)
            # 匹配字符集转换器名称
            self._match_texts(self.CHARSET_TRANSLATORS)

            # 创建 TranslateCharacters 表达式
            return self.expression(
                exp.TranslateCharacters,
                this=this,
                expression=self._prev.text.upper(),  # 转换器名称
                with_error=self._match_text_seq("WITH", "ERROR"),  # 错误处理选项
            )

        # FROM before SET in Teradata UPDATE syntax
        # https://docs.teradata.com/r/Enterprise_IntelliFlex_VMware/Teradata-VantageTM-SQL-Data-Manipulation-Language-17.20/Statement-Syntax/UPDATE/UPDATE-Syntax-Basic-Form-FROM-Clause
        def _parse_update(self) -> exp.Update:
            """
            解析 Teradata UPDATE 语句
            
            功能说明:
                - Teradata 的 UPDATE 语法中 FROM 子句在 SET 子句之前
                - 支持连接操作在 FROM 子句中
                - 与标准 SQL 的语法顺序不同
            
            语法: UPDATE table FROM ... SET ... WHERE ...
            参考: https://docs.teradata.com/r/Enterprise_IntelliFlex_VMware/Teradata-VantageTM-SQL-Data-Manipulation-Language-17.20/Statement-Syntax/UPDATE/UPDATE-Syntax-Basic-Form-FROM-Clause
            """
            return self.expression(
                exp.Update,
                **{  # type: ignore
                    # 解析目标表
                    "this": self._parse_table(alias_tokens=self.UPDATE_ALIAS_TOKENS),
                    # 解析 FROM 子句（支持连接）
                    "from": self._parse_from(joins=True),
                    # 解析 SET 子句中的赋值表达式
                    "expressions": self._match(TokenType.SET)
                    and self._parse_csv(self._parse_equality),
                    # 解析 WHERE 条件
                    "where": self._parse_where(),
                },
            )

        def _parse_rangen(self):
            """
            解析 Teradata RANGE_N 函数
            
            功能说明:
                - 解析范围分区函数，用于数据分布
                - 支持 BETWEEN 子句指定范围
                - 支持 EACH 子句指定步长
            
            语法: RANGE_N(column BETWEEN value1 AND value2 [EACH step])
            """
            # 解析列名或变量
            this = self._parse_id_var()
            # 匹配 BETWEEN 关键字
            self._match(TokenType.BETWEEN)

            # 解析范围值列表（通常是两个值：起始值和结束值）
            expressions = self._parse_csv(self._parse_assignment)
            # 解析可选的 EACH 子句（步长）
            each = self._match_text_seq("EACH") and self._parse_assignment()

            # 创建 RangeN 表达式
            return self.expression(exp.RangeN, this=this, expressions=expressions, each=each)

        def _parse_query_band(self) -> exp.QueryBand:
            """
            解析 Teradata QUERY_BAND 语句
            
            功能说明:
                - 解析查询标签设置语句，用于查询分类和监控
                - 支持字符串格式和 NONE 关键字
                - 支持不同的作用域（SESSION、TRANSACTION 等）
                - 支持 UPDATE 选项
            
            支持的语法:
                - SET QUERY_BAND = 'key=value;key2=value2;' FOR SESSION|TRANSACTION
                - SET QUERY_BAND = 'key=value;' UPDATE FOR SESSION|TRANSACTION  
                - SET QUERY_BAND = NONE FOR SESSION|TRANSACTION
            """
            # 匹配等号
            self._match(TokenType.EQ)

            # 处理字符串字面量和 NONE 关键字两种情况
            if self._match_text_seq("NONE"):
                query_band_string: t.Optional[exp.Expression] = exp.Var(this="NONE")
            else:
                query_band_string = self._parse_string()

            # 检查是否有 UPDATE 选项
            update = self._match_text_seq("UPDATE")
            # 匹配 FOR 关键字
            self._match_text_seq("FOR")

            # 处理作用域 - 可以是 SESSION、TRANSACTION、VOLATILE 或 SESSION VOLATILE
            if self._match_text_seq("SESSION", "VOLATILE"):
                scope = "SESSION VOLATILE"
            elif self._match_texts(("SESSION", "TRANSACTION")):
                scope = self._prev.text.upper()
            else:
                scope = None

            # 创建 QueryBand 表达式
            return self.expression(
                exp.QueryBand,
                this=query_band_string,
                scope=scope,
                update=update,
            )

        def _parse_xmlagg(self) -> exp.XMLAgg:
            """
            解析 Teradata XMLAGG 函数
            
            功能说明:
                - 解析XML聚合函数，将多行XML数据聚合为单个XML文档
                - 支持ORDER BY子句指定聚合顺序
                - 支持RETURNING子句指定返回类型（CONTENT或SEQUENCE）
            
            语法: XMLAGG(XML_value_expr [ORDER BY order_by_spec [,...]] [RETURNING {CONTENT | SEQUENCE}])
            """
            # 解析XML表达式
            this = self._parse_assignment()
            
            # 检查是否有ORDER BY子句（类似STRING_AGG的处理方式）
            if self._match(TokenType.ORDER_BY):
                # 将ORDER BY包装到this参数中
                this = self._parse_order(this=this,skip_order_token=True)
            
            # 解析可选的RETURNING子句
            returning = None
            if self._match_text_seq("RETURNING"):
                if self._match_text_seq("CONTENT"):
                    returning = "CONTENT"
                elif self._match_text_seq("SEQUENCE"):
                    returning = "SEQUENCE"
                else:
                    # 如果没有指定CONTENT或SEQUENCE，默认为SEQUENCE
                    returning = "SEQUENCE"
            
            # 创建XMLAgg表达式
            return self.expression(exp.XMLAgg, this=this, returning=returning)


        def _parse_primary(self) -> t.Optional[exp.Expression]:
            """
            重写主表达式解析以支持Teradata类型转换语法
            """
            # 调用父类方法解析基础表达式
            primary = super()._parse_primary()
            
            # 处理Teradata类型转换语法：expr(data_type)
            return self._parse_teradata_cast_postfix(primary)
        
        def _is_teradata_cast_pattern(self) -> bool:
            """
            检查当前位置是否匹配Teradata类型转换模式：identifier(data_type_keyword)
            
            这是预检查方法，不移动解析器位置，不产生副作用
            
            注意：只有未引用的标识符才可能是类型转换，引用的标识符应该是函数调用
            
            Returns:
                bool: 如果匹配类型转换模式则返回True
            """
            # 检查基本结构：当前token + 下一个是( + 再下一个是数据类型关键字
            if (self._curr and 
                self._next and 
                self._next.token_type == TokenType.L_PAREN and  # 下一个是(
                self._index + 2 < len(self._tokens)):  # 还有更多token
                
                # 检查当前token是否可能是类型转换的源表达式
                # 在Teradata中：
                # - 引用的标识符（如"INT"）可以是列名，"INT"(INTEGER) 是类型转换
                # - 普通变量名（如col）可以是列名，col(INTEGER) 是类型转换  
                # - 未引用的数据类型关键字（如INT）在此上下文中通常是函数调用
                is_potential_cast_source = False
                
                if self._curr.token_type == TokenType.IDENTIFIER:
                    # IDENTIFIER token：引用的标识符，可能是列名，支持类型转换
                    is_potential_cast_source = True
                elif self._curr.token_type == TokenType.VAR:
                    # VAR token：普通变量名，支持类型转换
                    is_potential_cast_source = True
                elif self._curr.token_type in self.TYPE_TOKENS:
                    # 数据类型关键字token：在此上下文中通常是函数调用，不是类型转换
                    is_potential_cast_source = False
                
                if is_potential_cast_source:
                    # 检查括号后的token是否为数据类型关键字
                    potential_type_token = self._tokens[self._index + 2]
                    
                    # 检查是否为未引用的数据类型关键字
                    if (potential_type_token.text.upper() in self.DATA_TYPE_KEYWORDS and
                        not getattr(potential_type_token, 'quoted', False)):
                        
                        # 排除ANSI字面量语法的情况
                        # ANSI字面量：DATE'2023-08-15', TIME'14:30:00', TIMESTAMP'...'
                        # 这种情况下，数据类型关键字后面紧跟字符串字面量
                        if (self._index + 3 < len(self._tokens) and
                            self._tokens[self._index + 3].token_type == TokenType.STRING):
                            # 这是ANSI字面量，不是类型转换
                            return False
                        
                        return True
            
            return False

        def _parse_teradata_cast_as_function(self) -> t.Optional[exp.Expression]:
            """
            将Teradata类型转换语法解析为Cast表达式
            
            假设已经通过_is_teradata_cast_pattern()确认了模式匹配
            
            Returns:
                exp.Cast: 类型转换表达式
            """
            # 获取源表达式（当前标识符）
            # TokenType.IDENTIFIER 表示引用的标识符，需要设置 quoted=True
            is_quoted = self._curr.token_type == TokenType.IDENTIFIER
            source_expr = exp.Column(this=exp.Identifier(this=self._curr.text, quoted=is_quoted))
            
            # 跳过标识符和左括号
            self._advance(2)
            
            # 解析数据类型
            data_type = self._parse_types()
            
            # 匹配右括号
            self._match_r_paren()
            
            # 创建Cast表达式
            cast_expr = self.expression(exp.Cast, this=source_expr, to=data_type)
            
            # 处理可能的嵌套类型转换
            return self._parse_teradata_cast_postfix(cast_expr)

        def _parse_teradata_cast_postfix(self, expr: t.Optional[exp.Expression]) -> t.Optional[exp.Expression]:
            """
            解析Teradata类型转换后缀语法：expr(data_type)
            
            支持嵌套转换：expr(int)(varchar(10))
            
            Args:
                expr: 要进行类型转换的表达式
                
            Returns:
                转换后的表达式，如果不是类型转换则返回原表达式
            """
            if not expr:
                return expr
                
            # 处理连续的类型转换
            while (self._match(TokenType.L_PAREN) and 
                   self._curr and 
                   self._curr.text.upper() in self.DATA_TYPE_KEYWORDS and
                   not getattr(self._curr, 'quoted', False)):
                
                # 解析数据类型
                data_type = self._parse_types()
                
                # 匹配右括号
                self._match_r_paren()
                
                # 创建Cast表达式
                expr = self.expression(exp.Cast, this=expr, to=data_type)
            
            return expr

        def _is_data_type_expression(self, expr: t.Optional[exp.Expression]) -> bool:
            """
            检查表达式是否为数据类型定义
            
            功能说明:
                - 检查简单数据类型关键字（如INT, VARCHAR等）
                - 检查带参数的数据类型（如VARCHAR(100), DECIMAL(10,2)等）
                - 用于区分函数调用和类型转换
            """
            if not expr:
                return False
            
            # 1. 检查简单数据类型关键字（如INT, VARCHAR等）
            if isinstance(expr, exp.Column) and expr.this:
                type_name = expr.this.this.upper() if hasattr(expr.this, 'this') else str(expr.this).upper()
                return type_name in self.DATA_TYPE_KEYWORDS
            
            # 2. 检查带参数的数据类型（如VARCHAR(100), DECIMAL(10,2)等）
            if isinstance(expr, exp.Anonymous) and expr.this:
                func_name = str(expr.this).upper()
                return func_name in self.DATA_TYPE_KEYWORDS
            
            # 3. 检查直接的标识符（如果被解析为Identifier）
            if isinstance(expr, exp.Identifier):
                return expr.this.upper() in self.DATA_TYPE_KEYWORDS
            
            return False



        def _parse_index_params(self) -> exp.IndexParameters:
            """
            解析索引参数
            
            功能说明:
                - 调用父类方法解析基础索引参数
                - 处理 Teradata 特有的索引语法差异
                - 移除 ON 子句并回退解析位置
            """
            # 调用父类方法获取基础索引参数
            this = super()._parse_index_params()

            # 如果存在 ON 子句，需要移除并回退解析位置
            # 这是因为 Teradata 的索引语法与标准 SQL 有所不同
            if this.args.get("on"):
                this.set("on", None)
                self._retreat(self._index - 2)
            return this

        def _parse_function(
            self,
            functions: t.Optional[t.Dict[str, t.Callable]] = None,
            anonymous: bool = False,
            optional_parens: bool = True,
            any_token: bool = False,
        ) -> t.Optional[exp.Expression]:
            """
            解析函数调用
            
            功能说明:
                - 重写父类的函数解析方法
                - 处理 Teradata 特有的 FORMAT 子句
                - 避免将 FORMAT 子句误解析为函数调用
            
            特殊处理:
                - 检测 (FORMAT <format_string>) 模式
                - 这种模式用于列引用后的输出格式覆盖
                - 参考: https://docs.teradata.com/r/Enterprise_IntelliFlex_VMware/SQL-Data-Types-and-Literals/Data-Type-Formats-and-Format-Phrases/FORMAT
            """
            # 检查是否是 Teradata 的 FORMAT 子句模式
            # 格式: column_name (FORMAT 'format_string')
            if (
                self._next
                and self._next.token_type == TokenType.L_PAREN
                and self._index + 2 < len(self._tokens)
                and self._tokens[self._index + 2].token_type == TokenType.FORMAT
            ):
                # 不将此模式解析为函数调用
                return None

            # 先检查是否为Teradata类型转换语法模式
            if self._is_teradata_cast_pattern():
                return self._parse_teradata_cast_as_function()
            
            # 否则按标准函数解析
            return super()._parse_function(
                functions=functions,
                anonymous=anonymous,
                optional_parens=optional_parens,
                any_token=any_token,
            )

        def _parse_column_ops(self, this: t.Optional[exp.Expression]) -> t.Optional[exp.Expression]:
            """
            解析列操作
            
            功能说明:
                - 调用父类方法解析基础列操作
                - 处理 Teradata 特有的 FORMAT 短语
                - 创建 FormatPhrase 表达式节点
            """
            # 调用父类方法处理基础列操作
            this = super()._parse_column_ops(this)

            # 检查后缀语法：FORMAT 或 类型转换
            # 处理三种情况：
            # 1. FORMAT语法：col (FORMAT'...')  -> L_PAREN + FORMAT
            # 2. FORMAT语法（ANSI字面量）：DATE'...' (FORMAT'...')  -> 直接FORMAT
            # 3. 类型转换语法：expr(data_type)  -> L_PAREN + 数据类型关键字
            
            # 检查后缀语法：FORMAT 或 类型转换
            if self._match(TokenType.FORMAT):
                # ANSI字面量的FORMAT语法
                fmt_string = self._parse_string()
                self._match_r_paren()
                this = self.expression(exp.FormatPhrase, this=this, format=fmt_string)
            elif self._match(TokenType.L_PAREN):
                # 检查括号内容
                if self._match(TokenType.FORMAT):
                    # 普通列的FORMAT语法
                    fmt_string = self._parse_string()
                    self._match_r_paren()
                    this = self.expression(exp.FormatPhrase, this=this, format=fmt_string)
                elif (self._curr and self._curr.text.upper() in self.DATA_TYPE_KEYWORDS):
                    # 类型转换语法
                    data_type = self._parse_types()
                    self._match_r_paren()
                    this = self.expression(exp.Cast, this=this, to=data_type)
                    # 处理可能的嵌套类型转换
                    this = self._parse_teradata_cast_postfix(this)
                else:
                    # 不是我们要处理的模式，回退
                    self._retreat(self._index - 1)
                    this = self._parse_teradata_cast_postfix(this)
            else:
                # 其他情况：调用原有的类型转换后缀处理
                this = self._parse_teradata_cast_postfix(this)

            return this

    class Generator(generator.Generator):
        """
        Teradata SQL 代码生成器
        
        功能说明:
            - 继承自基础 Generator 类
            - 将抽象语法树转换为 Teradata 特定的 SQL 代码
            - 处理 Teradata 特有的语法和函数
        """
        # LIMIT 子句使用 TOP 关键字
        LIMIT_IS_TOP = True
        # 不支持连接提示
        JOIN_HINTS = False
        # 不支持表提示
        TABLE_HINTS = False
        # 不支持查询提示
        QUERY_HINTS = False
        # 表采样使用 SAMPLE 关键字
        TABLESAMPLE_KEYWORDS = "SAMPLE"
        # LAST_DAY 函数不支持日期部分参数
        LAST_DAY_SUPPORTS_DATE_PART = False
        # 可以实现 ARRAY_ANY 功能
        CAN_IMPLEMENT_ARRAY_ANY = True
        # 时区转换为 WITH TIME ZONE 格式
        TZ_TO_WITH_TIME_ZONE = True
        # 数组大小函数名称
        ARRAY_SIZE_NAME = "CARDINALITY"

        # Teradata 数据类型映射表
        TYPE_MAPPING = {
            **generator.Generator.TYPE_MAPPING,
            exp.DataType.Type.GEOMETRY: "ST_GEOMETRY",      # 几何类型映射
            exp.DataType.Type.DOUBLE: "DOUBLE PRECISION",   # 双精度浮点数
            exp.DataType.Type.TIMESTAMPTZ: "TIMESTAMP",     # 带时区的时间戳
            exp.DataType.Type.BYTEINT: "BYTEINT",           # BYTEINT 数据类型映射
            exp.DataType.Type.INT: "INTEGER",           # INTEGER 数据类型映射
        }

        # 属性位置映射表，定义各种属性在 SQL 语句中的位置
        PROPERTIES_LOCATION = {
            **generator.Generator.PROPERTIES_LOCATION,
            # ON COMMIT 属性位置在索引之后
            exp.OnCommitProperty: exp.Properties.Location.POST_INDEX,
            # PARTITION BY 属性位置在表达式之后
            exp.PartitionedByProperty: exp.Properties.Location.POST_EXPRESSION,
            # STABILITY 属性位置在 CREATE 之后
            exp.StabilityProperty: exp.Properties.Location.POST_CREATE,
        }

        # Teradata 表达式转换映射表
        # 将标准 SQL 表达式转换为 Teradata 特定的 SQL 语法
        TRANSFORMS = {
            **generator.Generator.TRANSFORMS,
            # ArgMax 函数重命名为 MAX_BY
            exp.ArgMax: rename_func("MAX_BY"),
            # ArgMin 函数重命名为 MIN_BY
            exp.ArgMin: rename_func("MIN_BY"),
            # Max 函数转换为 GREATEST 或保持 MAX
            exp.Max: max_or_greatest,
            # Min 函数转换为 LEAST 或保持 MIN
            exp.Min: min_or_least,
            # 幂运算使用 ** 操作符
            exp.Pow: lambda self, e: self.binary(e, "**"),
            # Rand 函数转换为 RANDOM 函数，支持上下界参数
            exp.Rand: lambda self, e: self.func("RANDOM", e.args.get("lower"), e.args.get("upper")),
            # Select 语句预处理：消除 DISTINCT ON 和半连接/反连接
            exp.Select: transforms.preprocess(
                [transforms.eliminate_distinct_on, transforms.eliminate_semi_and_anti_joins]
            ),
            # 字符串位置函数转换为 INSTR，支持位置和出现次数参数
            exp.StrPosition: lambda self, e: (
                strposition_sql(
                    self, e, func_name="INSTR", supports_position=True, supports_occurrence=True
                )
            ),
            # 字符串转日期函数，使用 CAST ... AS DATE FORMAT 语法
            exp.StrToDate: lambda self,
            e: f"CAST({self.sql(e, 'this')} AS DATE FORMAT {self.format_time(e)})",
            # ToChar 函数使用回退 SQL 生成
            exp.ToChar: lambda self, e: self.function_fallback_sql(e),
            # ToNumber 函数支持 NLS 参数
            exp.ToNumber: to_number_with_nls_param,
            # Use 语句转换为 DATABASE 语句
            exp.Use: lambda self, e: f"DATABASE {self.sql(e, 'this')}",
            # 日期加法使用自定义函数
            exp.DateAdd: _date_add_sql("+"),
            # 日期减法使用自定义函数
            exp.DateSub: _date_add_sql("-"),
            # Quarter 函数转换为 EXTRACT(QUARTER FROM ...)
            exp.Quarter: lambda self, e: self.sql(exp.Extract(this="QUARTER", expression=e.this)),
            # XMLAgg 函数转换为 Teradata 特定的 XMLAGG 语法
            exp.XMLAgg: lambda self, e: self.xmlagg_sql(e),
        }

        def currenttimestamp_sql(self, expression: exp.CurrentTimestamp) -> str:
            """
            生成 CURRENT_TIMESTAMP 函数的 SQL
            
            功能说明:
                - 处理 CurrentTimestamp 表达式
                - 根据是否有精度参数决定是否添加括号
                - 生成 Teradata 格式的当前时间戳函数
            """
            # 如果有精度参数，添加括号；否则不添加
            prefix, suffix = ("(", ")") if expression.this else ("", "")
            return self.func("CURRENT_TIMESTAMP", expression.this, prefix=prefix, suffix=suffix)

        def cast_sql(self, expression: exp.Cast, safe_prefix: t.Optional[str] = None) -> str:
            """
            生成 CAST 函数的 SQL
            
            功能说明:
                - 处理类型转换表达式
                - 特殊处理 FORMAT 子句的情况
                - 移除 UNKNOWN 类型以支持 FORMAT 语法
            """
            # 处理 CAST(<value> AS FORMAT <format>) 语法
            # 在这种情况下，我们不需要打印 UNKNOWN 类型
            if expression.to.this == exp.DataType.Type.UNKNOWN and expression.args.get("format"):
                expression.to.pop()

            # 调用父类方法生成标准 CAST SQL
            return super().cast_sql(expression, safe_prefix=safe_prefix)

        def trycast_sql(self, expression: exp.TryCast) -> str:
            """
            生成 TRYCAST 函数的 SQL
            
            功能说明:
                - 处理 TryCast 表达式
                - 使用 TRY 前缀调用 cast_sql 方法
                - 生成 Teradata 的 TRYCAST 语法
            """
            return self.cast_sql(expression, safe_prefix="TRY")

        def tablesample_sql(
            self,
            expression: exp.TableSample,
            tablesample_keyword: t.Optional[str] = None,
        ) -> str:
            """
            生成表采样的 SQL
            
            功能说明:
                - 处理 TableSample 表达式
                - 使用 Teradata 的 SAMPLE 关键字
                - 生成表采样语法
            """
            return f"{self.sql(expression, 'this')} SAMPLE {self.expressions(expression)}"

        def partitionedbyproperty_sql(self, expression: exp.PartitionedByProperty) -> str:
            """
            生成分区属性的 SQL
            
            功能说明:
                - 处理 PartitionedByProperty 表达式
                - 生成 PARTITION BY 子句
            """
            return f"PARTITION BY {self.sql(expression, 'this')}"

        def update_sql(self, expression: exp.Update) -> str:
            """
            生成 UPDATE 语句的 SQL
            
            功能说明:
                - 处理 Teradata 特有的 UPDATE 语法
                - FROM 子句在 SET 子句之前
                - 与标准 SQL 的语法顺序不同
            
            语法: UPDATE table FROM ... SET ... WHERE ...
            参考: https://docs.teradata.com/r/Enterprise_IntelliFlex_VMware/Teradata-VantageTM-SQL-Data-Manipulation-Language-17.20/Statement-Syntax/UPDATE/UPDATE-Syntax-Basic-Form-FROM-Clause
            """
            # 生成各个部分的 SQL
            this = self.sql(expression, "this")
            from_sql = self.sql(expression, "from")
            set_sql = self.expressions(expression, flat=True)
            where_sql = self.sql(expression, "where")
            # 组装 UPDATE 语句：UPDATE table FROM ... SET ... WHERE ...
            sql = f"UPDATE {this}{from_sql} SET {set_sql}{where_sql}"
            return self.prepend_ctes(expression, sql)

        def mod_sql(self, expression: exp.Mod) -> str:
            """
            生成取模运算的 SQL
            
            功能说明:
                - 处理 Mod 表达式
                - 使用 % 操作符进行取模运算
            """
            return self.binary(expression, "%")

        def datatype_sql(self, expression: exp.DataType) -> str:
            """
            生成数据类型的 SQL
            
            功能说明:
                - 处理 DataType 表达式
                - 支持 SYSUDTLIB 前缀的用户定义类型
                - 生成 Teradata 格式的数据类型
            """
            # 调用父类方法生成基础类型 SQL
            type_sql = super().datatype_sql(expression)
            # 检查是否有前缀（用于用户定义类型）
            prefix_sql = expression.args.get("prefix")
            return f"SYSUDTLIB.{type_sql}" if prefix_sql else type_sql

        def rangen_sql(self, expression: exp.RangeN) -> str:
            """
            生成 RANGE_N 函数的 SQL
            
            功能说明:
                - 处理 RangeN 表达式
                - 生成 RANGE_N 函数调用
                - 支持 BETWEEN 和 EACH 子句
            
            语法: RANGE_N(column BETWEEN value1 AND value2 [EACH step])
            """
            # 生成各个部分的 SQL
            this = self.sql(expression, "this")
            expressions_sql = self.expressions(expression)
            each_sql = self.sql(expression, "each")
            # 如果有 EACH 子句，添加到 SQL 中
            each_sql = f" EACH {each_sql}" if each_sql else ""

            return f"RANGE_N({this} BETWEEN {expressions_sql}{each_sql})"

        def lockingstatement_sql(self, expression: exp.LockingStatement) -> str:
            """
            生成 LOCKING 语句的 SQL
            
            功能说明:
                - 处理 LockingStatement 表达式
                - 生成 LOCKING 子句和查询语句
                - 组装完整的 LOCKING 语句
            """
            # 生成锁定子句
            locking_clause = self.sql(expression, "this")
            # 生成查询语句
            query_sql = self.sql(expression, "expression")

            return f"{locking_clause} {query_sql}"

        def createable_sql(self, expression: exp.Create, locations: t.DefaultDict) -> str:
            """
            生成可创建对象的 SQL
            
            功能说明:
                - 处理 Create 表达式
                - 特殊处理 TABLE 类型的创建语句
                - 处理表名后的属性和模式列
            """
            # 获取创建对象的类型
            kind = self.sql(expression, "kind").upper()
            # 对于 TABLE 类型且有 POST_NAME 属性的情况
            if kind == "TABLE" and locations.get(exp.Properties.Location.POST_NAME):
                # 生成表名
                this_name = self.sql(expression.this, "this")
                # 生成表属性
                this_properties = self.properties(
                    exp.Properties(expressions=locations[exp.Properties.Location.POST_NAME]),
                    wrapped=False,
                    prefix=",",
                )
                # 生成模式列
                this_schema = self.schema_columns_sql(expression.this)
                return f"{this_name}{this_properties}{self.sep()}{this_schema}"

            # 其他情况调用父类方法
            return super().createable_sql(expression, locations)

        def extract_sql(self, expression: exp.Extract) -> str:
            """
            生成 EXTRACT 函数的 SQL
            
            功能说明:
                - 处理 Extract 表达式
                - 特殊处理 QUARTER 提取
                - 对于 QUARTER，使用 TO_CHAR 函数转换
            """
            # 获取提取的部分
            this = self.sql(expression, "this")
            # 对于非 QUARTER 的情况，使用父类方法
            if this.upper() != "QUARTER":
                return super().extract_sql(expression)

            # 对于 QUARTER，使用 TO_CHAR 函数转换为季度数字
            to_char = exp.func("to_char", expression.expression, exp.Literal.string("Q"))
            return self.sql(exp.cast(to_char, exp.DataType.Type.INT))

        def interval_sql(self, expression: exp.Interval) -> str:
            """
            生成 INTERVAL 表达式的 SQL
            
            功能说明:
                - 处理 Interval 表达式
                - 将 WEEK 和 QUARTER 转换为 DAY 的倍数
                - Teradata 不直接支持 WEEK 和 QUARTER 间隔
            """
            multiplier = 0
            unit = expression.text("unit")

            # 将 WEEK 转换为 7 天
            if unit.startswith("WEEK"):
                multiplier = 7
            # 将 QUARTER 转换为 90 天（近似值）
            elif unit.startswith("QUARTER"):
                multiplier = 90

            # 如果需要转换，生成乘法表达式
            if multiplier:
                return f"({multiplier} * {super().interval_sql(exp.Interval(this=expression.this, unit=exp.var('DAY')))})"

            # 其他情况使用父类方法
            return super().interval_sql(expression)

        def xmlagg_sql(self, expression: exp.XMLAgg) -> str:
            """
            生成 XMLAGG 函数的 SQL
            
            功能说明:
                - 处理 XMLAgg 表达式
                - 生成 Teradata 格式的 XMLAGG 函数调用
                - 支持 ORDER BY 和 RETURNING 子句
            
            语法: XMLAGG(XML_value_expr [ORDER BY order_by_spec [,...]] [RETURNING {CONTENT | SEQUENCE}])
            """
            # 检查this参数是否包含ORDER BY
            this_expr = expression.this
            if isinstance(this_expr, exp.Order):
                # 如果this是Order表达式，分别处理XML表达式和ORDER BY
                xml_expr = self.sql(this_expr, "this")
                order_sql = f" {self.sql(this_expr).replace(self.sql(this_expr, 'this'), '').strip()}"
            else:
                # 普通的XML表达式
                xml_expr = self.sql(this_expr)
                order_sql = ""
            
            # 生成RETURNING子句
            returning = expression.args.get("returning")
            returning_sql = f" RETURNING {returning}" if returning else ""
            
            return f"XMLAGG({xml_expr}{order_sql}{returning_sql})"
