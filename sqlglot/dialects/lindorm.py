from __future__ import annotations

import typing as t

from sqlglot import exp, generator, parser
from sqlglot.dialects.dialect import Dialect
from sqlglot.tokens import TokenType


class Lindorm(Dialect):
    class Parser(parser.Parser):
        PROPERTY_PARSERS = {
            **parser.Parser.PROPERTY_PARSERS,
            "WITH": lambda self: self._parse_lindorm_with_property(),
        }

        def _parse_lindorm_property(self) -> t.Optional[exp.Property]:
            index = self._index
            key = self._parse_column()

            if not key or not self._match(TokenType.EQ):
                self._retreat(index)
                return None

            if isinstance(key, exp.Column):
                key = key.to_dot() if len(key.parts) > 1 else exp.var(key.name)

            value = self._parse_bitwise() or self._parse_var(any_token=True)

            if isinstance(value, exp.Column):
                value = exp.var(value.name)

            return self.expression(exp.Property, this=key, value=value)

        def _parse_lindorm_with_property(self) -> t.Optional[exp.Expression] | t.List[exp.Expression]:
            if self._match(TokenType.L_PAREN, advance=False):
                return self._parse_wrapped_csv(self._parse_lindorm_property)

            return super()._parse_with_property()

        def _parse_index(
            self, index: t.Optional[exp.Expression] = None, anonymous: bool = False
        ) -> t.Optional[exp.Index]:
            if not (index or anonymous):
                return super()._parse_index(index=index, anonymous=anonymous)

            using = self._parse_var(any_token=True) if self._match(TokenType.USING) else None

            self._match(TokenType.ON)
            self._match(TokenType.TABLE)
            table = self._parse_table_parts(schema=True)
            params = self._parse_index_params()

            if using:
                if params.args.get("using"):
                    self.raise_error("Expected a single USING clause in index definition.")
                params.set("using", using)

            return self.expression(
                exp.Index,
                this=index,
                table=table,
                unique=None,
                primary=None,
                amp=None,
                params=params,
            )

    class Generator(generator.Generator):
        TYPE_MAPPING = {
            **generator.Generator.TYPE_MAPPING,
            exp.DataType.Type.INT: "INTEGER",
            exp.DataType.Type.TEXT: "STRING",
        }

        def index_sql(self, expression: exp.Index) -> str:
            params = expression.args.get("params")
            table = self.sql(expression, "table")

            if table and params and params.args.get("using"):
                unique = "UNIQUE " if expression.args.get("unique") else ""
                primary = "PRIMARY " if expression.args.get("primary") else ""
                amp = "AMP " if expression.args.get("amp") else ""
                name = self.sql(expression, "this")
                name = f"{name} " if name else ""
                using = self.sql(params, "using")
                using = f"USING {using} " if using else ""

                params_without_using = params.copy()
                params_without_using.set("using", None)
                params_sql = self.sql(params_without_using)

                return f"{unique}{primary}{amp}{name}{using}{self.INDEX_ON} {table}{params_sql}"

            return super().index_sql(expression)
