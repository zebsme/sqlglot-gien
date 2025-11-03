from __future__ import annotations

import typing as t
from sqlglot import exp, tokens ,parser
from sqlglot.dialects.athena import Athena
from sqlglot.tokens import Tokenizer, TokenType



class DB2(Athena):  # 继承自 Athena 方言
    class Tokenizer(Athena.Tokenizer):
        IDENTIFIERS = ['"']
        QUOTES = ["'"]

        # 扩展关键字映射
        KEYWORDS = {
            **Athena.Tokenizer.KEYWORDS,
        }

    class Parser(Athena.Parser):
        """DB2特定的解析器"""
        
        # 扩展类型token集合
        TYPE_TOKENS = {
            *parser.Parser.TYPE_TOKENS,
            TokenType.LONG_VARCHAR,
            TokenType.DECFLOAT,
            TokenType.LONG_VARGRAPHIC,
        }

    class Generator(Athena.Generator):
        """DB2特定的生成器"""
        
        # 覆盖类型映射
        TYPE_MAPPING = {
            **Athena.Generator.TYPE_MAPPING,  # 继承原有映射
            exp.DataType.Type.VARBINARY: "BLOB",  # VARBINARY → BLOB
            
            # 添加DB2特有数据类型的映射
            exp.DataType.Type.LONG_VARCHAR: "LONG VARCHAR",
            exp.DataType.Type.DECFLOAT: "DECFLOAT", 
            exp.DataType.Type.LONG_VARGRAPHIC: "LONG VARGRAPHIC",
        }
