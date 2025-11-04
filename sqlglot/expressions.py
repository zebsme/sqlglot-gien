"""
## Expressions

Every AST node in SQLGlot is represented by a subclass of `Expression`.

This module contains the implementation of all supported `Expression` types. Additionally,
it exposes a number of helper functions, which are mainly used to programmatically build
SQL expressions, such as `sqlglot.expressions.select`.

----
"""

from __future__ import annotations

import datetime
import math
import numbers
import re
import textwrap
import typing as t
from collections import deque
from copy import deepcopy
from decimal import Decimal
from enum import auto
from functools import reduce

from sqlglot.errors import ErrorLevel, ParseError
from sqlglot.helper import (
    AutoName,
    camel_to_snake_case,
    ensure_collection,
    ensure_list,
    seq_get,
    split_num_words,
    subclasses,
    to_bool,
)
from sqlglot.tokens import Token, TokenError

if t.TYPE_CHECKING:
    from typing_extensions import Self

    from sqlglot._typing import E, Lit
    from sqlglot.dialects.dialect import DialectType

    Q = t.TypeVar("Q", bound="Query")
    S = t.TypeVar("S", bound="SetOperation")


class _Expression(type):
    def __new__(cls, clsname, bases, attrs):
        klass = super().__new__(cls, clsname, bases, attrs)

        # When an Expression class is created, its key is automatically set
        # to be the lowercase version of the class' name.
        klass.key = clsname.lower()

        # This is so that docstrings are not inherited in pdoc
        klass.__doc__ = klass.__doc__ or ""

        return klass


SQLGLOT_META = "sqlglot.meta"
SQLGLOT_ANONYMOUS = "sqlglot.anonymous"
TABLE_PARTS = ("this", "db", "catalog")
COLUMN_PARTS = ("this", "table", "db", "catalog")
POSITION_META_KEYS = ("line", "col", "start", "end")


class Expression(metaclass=_Expression):
    """
    The base class for all expressions in a syntax tree. Each Expression encapsulates any necessary
    context, such as its child expressions, their names (arg keys), and whether a given child expression
    is optional or not.

    Attributes:
        key: a unique key for each class in the Expression hierarchy. This is useful for hashing
            and representing expressions as strings.
        arg_types: determines the arguments (child nodes) supported by an expression. It maps
            arg keys to booleans that indicate whether the corresponding args are optional.
        parent: a reference to the parent expression (or None, in case of root expressions).
        arg_key: the arg key an expression is associated with, i.e. the name its parent expression
            uses to refer to it.
        index: the index of an expression if it is inside of a list argument in its parent.
        comments: a list of comments that are associated with a given expression. This is used in
            order to preserve comments when transpiling SQL code.
        type: the `sqlglot.expressions.DataType` type of an expression. This is inferred by the
            optimizer, in order to enable some transformations that require type information.
        meta: a dictionary that can be used to store useful metadata for a given expression.

    Example:
        >>> class Foo(Expression):
        ...     arg_types = {"this": True, "expression": False}

        The above definition informs us that Foo is an Expression that requires an argument called
        "this" and may also optionally receive an argument called "expression".

    Args:
        args: a mapping used for retrieving the arguments of an expression, given their arg keys.
    """

    key = "expression"
    arg_types = {"this": True}
    __slots__ = ("args", "parent", "arg_key", "index", "comments", "_type", "_meta", "_hash")

    def __init__(self, **args: t.Any):
        self.args: t.Dict[str, t.Any] = args
        self.parent: t.Optional[Expression] = None
        self.arg_key: t.Optional[str] = None
        self.index: t.Optional[int] = None
        self.comments: t.Optional[t.List[str]] = None
        self._type: t.Optional[DataType] = None
        self._meta: t.Optional[t.Dict[str, t.Any]] = None
        self._hash: t.Optional[int] = None

        for arg_key, value in self.args.items():
            self._set_parent(arg_key, value)

    def __eq__(self, other) -> bool:
        """
        判断两个表达式是否相等
        
        使用类型和哈希值进行快速比较，确保相同类型的表达式具有相同的结构
        才被认为是相等的。
        
        Args:
            other: 要比较的另一个对象
            
        Returns:
            bool: 如果两个表达式相等则返回True，否则返回False
        """
        # 首先检查类型是否完全相同，然后比较哈希值
        # 使用 type(self) is type(other) 而不是 isinstance 确保严格的类型匹配
        return self is other or (type(self) is type(other) and hash(self) == hash(other))

    def __hash__(self) -> int:
        """
        计算表达式的哈希值
        
        使用缓存机制避免重复计算，基于表达式类型和参数生成哈希值。
        这确保了具有相同结构和内容的表达式具有相同的哈希值。
        
        Returns:
            int: 表达式的哈希值
        """
        if self._hash is None:
            nodes = []
            queue = deque([self])

            while queue:
                node = queue.popleft()
                nodes.append(node)

                for v in node.iter_expressions():
                    if v._hash is None:
                        queue.append(v)

            for node in reversed(nodes):
                hash_ = hash(node.key)
                t = type(node)

                if t is Literal or t is Identifier:
                    for k, v in sorted(node.args.items()):
                        if v:
                            hash_ = hash((hash_, k, v))
                else:
                    for k, v in sorted(node.args.items()):
                        t = type(v)

                        if t is list:
                            for x in v:
                                if x is not None and x is not False:
                                    hash_ = hash((hash_, k, x.lower() if type(x) is str else x))
                                else:
                                    hash_ = hash((hash_, k))
                        elif v is not None and v is not False:
                            hash_ = hash((hash_, k, v.lower() if t is str else v))

                node._hash = hash_
        assert self._hash
        return self._hash

    def __reduce__(self) -> t.Tuple[t.Callable, t.Tuple[t.List[t.Dict[str, t.Any]]]]:
        from sqlglot.serde import dump, load

        return (load, (dump(self),))

    @property
    def this(self) -> t.Any:
        """
        获取名为"this"的参数
        
        这是表达式中最常用的参数，通常表示表达式的主要操作对象。
        例如：在 Column("name") 中，"this" 就是 "name"。
        
        Returns:
            参数值，如果不存在则返回None
        """
        return self.args.get("this")

    @property
    def expression(self) -> t.Any:
        """
        获取名为"expression"的参数
        
        通常用于表示子表达式或嵌套的表达式结构。
        
        Returns:
            参数值，如果不存在则返回None
        """
        return self.args.get("expression")

    @property
    def expressions(self) -> t.List[t.Any]:
        """
        获取名为"expressions"的参数列表
        
        用于表示多个子表达式的集合，如函数参数列表、SELECT列表等。
        如果参数不存在，返回空列表而不是None。
        
        Returns:
            list: 表达式列表，如果不存在则返回空列表
        """
        return self.args.get("expressions") or []

    def text(self, key) -> str:
        """
        获取指定参数的文本表示
        
        只能用于字符串或叶子表达式（如标识符、字面量）的参数。
        对于复杂表达式，返回空字符串。
        
        Args:
            key: 参数名称
            
        Returns:
            str: 参数的文本表示
        """
        field = self.args.get(key)
        if isinstance(field, str):
            # 直接返回字符串参数
            return field
        if isinstance(field, (Identifier, Literal, Var)):
            # 对于标识符、字面量和变量，返回其内部值
            return field.this
        if isinstance(field, (Star, Null)):
            # 对于星号和NULL，返回其名称
            return field.name
        # 对于其他复杂表达式，返回空字符串
        return ""

    @property
    def is_string(self) -> bool:
        """
        检查是否为字符串字面量
        
        判断当前表达式是否为字符串类型的字面量。
        
        Returns:
            bool: 如果是字符串字面量则返回True，否则返回False
        """
        return isinstance(self, Literal) and self.args["is_string"]

    @property
    def is_number(self) -> bool:
        """
        检查是否为数字表达式
        
        判断当前表达式是否为数字类型，包括字面量数字和负数表达式。
        
        Returns:
            bool: 如果是数字表达式则返回True，否则返回False
        """
        # 检查是否为数字字面量，或者是否为负数的数字表达式
        return (isinstance(self, Literal) and not self.args["is_string"]) or (
            isinstance(self, Neg) and self.this.is_number
        )

    def to_py(self) -> t.Any:
        """
        将SQL表达式转换为对应的Python对象
        
        将SQL字面量转换为Python原生类型，如字符串、数字等。
        对于无法转换的复杂表达式，抛出异常。
        
        Returns:
            对应的Python对象
            
        Raises:
            ValueError: 当表达式无法转换为Python对象时
        """
        raise ValueError(f"{self} cannot be converted to a Python object.")

    @property
    def is_int(self) -> bool:
        """
        检查是否为整数表达式
        
        判断当前表达式是否为整数类型，包括字面量整数和负整数。
        
        Returns:
            bool: 如果是整数表达式则返回True，否则返回False
        """
        # 首先检查是否为数字，然后检查转换为Python对象后是否为整数
        return self.is_number and isinstance(self.to_py(), int)

    @property
    def is_star(self) -> bool:
        """
        检查是否为星号表达式
        
        判断当前表达式是否为星号（*），包括直接的Star表达式
        和Column中包含Star的情况。
        
        Returns:
            bool: 如果是星号表达式则返回True，否则返回False
        """
        # 检查是否为Star表达式，或者是否为包含Star的Column表达式
        return isinstance(self, Star) or (isinstance(self, Column) and isinstance(self.this, Star))
    
    @property
    def alias(self) -> str:
        """
        获取表达式的别名
        
        返回表达式的别名，如果没有别名则返回空字符串。
        支持TableAlias对象和普通字符串别名。
        
        Returns:
            str: 表达式的别名，如果没有则返回空字符串
        """
        # 检查别名是否为TableAlias对象
        if isinstance(self.args.get("alias"), TableAlias):
            # 如果是TableAlias，返回其名称
            return self.args["alias"].name
        # 否则使用text方法获取别名的文本表示
        return self.text("alias")

    @property
    def alias_column_names(self) -> t.List[str]:
        """
        获取表别名的列名列表
        
        如果表达式有表别名，返回该别名对应的列名列表。
        主要用于处理表别名的列重命名功能。
        
        Returns:
            list[str]: 列名列表，如果没有表别名则返回空列表
        """
        table_alias = self.args.get("alias")
        if not table_alias:
            # 没有表别名，返回空列表
            return []
        # 从表别名中提取列名列表
        return [c.name for c in table_alias.args.get("columns") or []]

    @property
    def name(self) -> str:
        """
        获取表达式的主要名称
        
        返回表达式的"this"参数的文本表示，通常是表达式的主要标识符。
        
        Returns:
            str: 表达式的主要名称
        """
        return self.text("this")

    @property
    def alias_or_name(self) -> str:
        """
        获取别名或名称
        
        优先返回别名，如果没有别名则返回名称。
        这是一个便捷方法，用于获取表达式的显示名称。
        
        Returns:
            str: 别名或名称
        """
        return self.alias or self.name

    @property
    def output_name(self) -> str:
        """
        获取输出列的名称
        
        如果表达式是SELECT语句中的选择项，返回其输出列的名称。
        如果表达式没有输出名称，返回空字符串。
        
        示例:
            >>> from sqlglot import parse_one
            >>> parse_one("SELECT a").expressions[0].output_name
            'a'
            >>> parse_one("SELECT b AS c").expressions[0].output_name
            'c'
            >>> parse_one("SELECT 1 + 2").expressions[0].output_name
            ''
            
        Returns:
            str: 输出列的名称，如果没有则返回空字符串
        """
        return ""

    @property
    def type(self) -> t.Optional[DataType]:
        """
        获取表达式的数据类型
        
        Returns:
            DataType: 表达式的数据类型，如果没有设置则返回None
        """
        return self._type

    @type.setter
    def type(self, dtype: t.Optional[DataType | DataType.Type | str]) -> None:
        """
        设置表达式的数据类型
        
        支持多种数据类型格式：DataType对象、DataType.Type枚举或字符串。
        自动将非DataType对象转换为DataType。
        
        Args:
            dtype: 要设置的数据类型
        """
        # 如果提供了数据类型但不是DataType对象，则构建DataType对象
        if dtype and not isinstance(dtype, DataType):
            dtype = DataType.build(dtype)
        self._type = dtype  # type: ignore

    def is_type(self, *dtypes) -> bool:
        """
        检查表达式是否为指定的数据类型
        
        检查当前表达式的数据类型是否匹配任何一个指定的数据类型。
        
        Args:
            *dtypes: 要检查的数据类型列表
            
        Returns:
            bool: 如果表达式类型匹配任何一个指定类型则返回True，否则返回False
        """
        # 确保表达式有类型信息，然后调用DataType的is_type方法
        return self.type is not None and self.type.is_type(*dtypes)

    def is_leaf(self) -> bool:
        """
        检查表达式是否为叶子节点
        
        叶子节点是指不包含子表达式的表达式，即没有嵌套的Expression对象或列表。
        
        Returns:
            bool: 如果是叶子节点则返回True，否则返回False
        """
        # 检查所有参数值，如果没有任何Expression对象或列表，则为叶子节点
        return not any(isinstance(v, (Expression, list)) and v for v in self.args.values())

    @property
    def meta(self) -> t.Dict[str, t.Any]:
        """
        获取表达式的元数据字典
        
        元数据用于存储表达式的额外信息，如优化器信息、统计信息等。
        如果元数据不存在，会自动创建空字典。
        
        Returns:
            dict: 元数据字典
        """
        # 懒加载：如果元数据不存在，创建空字典
        if self._meta is None:
            self._meta = {}
        return self._meta

    def __deepcopy__(self, memo):
        """
        深拷贝表达式对象
        
        创建表达式的完整深拷贝，包括所有子表达式、注释、类型信息和元数据。
        使用迭代方式而不是递归，避免深度嵌套时的栈溢出问题。
        
        Args:
            memo: 深拷贝的备忘录，用于避免循环引用
            
        Returns:
            Expression: 深拷贝后的表达式对象
        """
        # 创建根节点
        root = self.__class__()
        # 使用栈来迭代处理所有节点，避免递归深度问题
        stack = [(self, root)]

        while stack:
            node, copy = stack.pop()

            # 深拷贝注释
            if node.comments is not None:
                copy.comments = deepcopy(node.comments)
            # 深拷贝类型信息
            if node._type is not None:
                copy._type = deepcopy(node._type)
            # 深拷贝元数据
            if node._meta is not None:
                copy._meta = deepcopy(node._meta)
            # 复制哈希值（哈希值不需要深拷贝）
            if node._hash is not None:
                copy._hash = node._hash

            # 处理所有参数
            for k, vs in node.args.items():
                if hasattr(vs, "parent"):
                    # 如果参数是表达式对象，创建新实例并加入栈中处理
                    stack.append((vs, vs.__class__()))
                    copy.set(k, stack[-1][-1])
                elif type(vs) is list:
                    # 如果参数是列表，创建新列表
                    copy.args[k] = []

                    for v in vs:
                        if hasattr(v, "parent"):
                            # 列表中的表达式对象也需要深拷贝
                            stack.append((v, v.__class__()))
                            copy.append(k, stack[-1][-1])
                        else:
                            # 非表达式对象直接添加
                            copy.append(k, v)
                else:
                    # 其他类型的参数直接复制
                    copy.args[k] = vs

        return root

    def copy(self) -> Self:
        """
        返回表达式的深拷贝
        
        这是一个便捷方法，使用deepcopy创建表达式的完整副本。
        
        Returns:
            Self: 深拷贝后的表达式对象
        """
        return deepcopy(self)

    def add_comments(self, comments: t.Optional[t.List[str]] = None, prepend: bool = False) -> None:
        """
        添加注释到表达式
        
        支持添加普通注释和包含元数据的特殊注释。
        特殊注释格式：-- comment /* SQLGLOT_META key=value,key2=value2 */
        
        Args:
            comments: 要添加的注释列表
            prepend: 是否将注释添加到列表开头，默认为False（添加到末尾）
        """
        # 如果表达式还没有注释列表，创建一个空列表
        if self.comments is None:
            self.comments = []

        if comments:
            for comment in comments:
                # 解析特殊注释中的元数据
                # 格式：comment /* SQLGLOT_META key=value,key2=value2 */
                _, *meta = comment.split(SQLGLOT_META)
                if meta:
                    # 处理元数据部分
                    for kv in "".join(meta).split(","):
                        # 解析键值对：key=value
                        k, *v = kv.split("=")
                        # 如果没有值，默认为True
                        value = v[0].strip() if v else True
                        # 将元数据存储到表达式的meta字典中
                        self.meta[k.strip()] = to_bool(value)

                # 根据prepend参数决定注释的添加位置
                if not prepend:
                    # 添加到注释列表末尾
                    self.comments.append(comment)

            if prepend:
                # 添加到注释列表开头
                self.comments = comments + self.comments

    def pop_comments(self) -> t.List[str]:
        """
        移除并返回表达式的所有注释
        
        清空表达式的注释列表，并返回之前的注释内容。
        用于注释的转移或清理操作。
        
        Returns:
            list[str]: 被移除的注释列表
        """
        # 获取当前注释列表，如果不存在则返回空列表
        comments = self.comments or []
        # 清空表达式的注释列表
        self.comments = None
        return comments

    def append(self, arg_key: str, value: t.Any) -> None:
        """
        向指定参数添加值
        
        如果参数是列表，将值添加到列表末尾；如果不是列表，创建新列表。
        自动处理父子关系的设置。
        
        Args:
            arg_key: 参数名称
            value: 要添加的值
        """
        # 如果参数不是列表，初始化为空列表
        if type(self.args.get(arg_key)) is not list:
            self.args[arg_key] = []
        
        # 设置父子关系
        self._set_parent(arg_key, value)
        values = self.args[arg_key]
        
        # 如果添加的是表达式对象，设置其索引
        if hasattr(value, "parent"):
            value.index = len(values)
        
        # 将值添加到列表末尾
        values.append(value)

    def set(
        self,
        arg_key: str,
        value: t.Any,
        index: t.Optional[int] = None,
        overwrite: bool = True,
    ) -> None:
        """
        设置参数的值
        
        支持直接设置、按索引设置列表元素、插入或覆盖操作。
        
        Args:
            arg_key: 参数名称
            value: 要设置的值
            index: 如果参数是列表，指定插入/覆盖的位置
            overwrite: 当指定索引时，是否覆盖现有值（True）还是插入新值（False）
        """
        expression: t.Optional[Expression] = self

        while expression and expression._hash is not None:
            expression._hash = None
            expression = expression.parent

        if index is not None:
            # 按索引操作列表参数
            expressions = self.args.get(arg_key) or []

            # 检查索引是否有效
            if seq_get(expressions, index) is None:
                return
            
            # 如果值为None，删除指定索引的元素
            if value is None:
                expressions.pop(index)
                # 更新后续元素的索引
                for v in expressions[index:]:
                    v.index = v.index - 1
                return

            # 如果值是列表，替换指定位置的元素
            if isinstance(value, list):
                expressions.pop(index)
                expressions[index:index] = value
            elif overwrite:
                # 覆盖指定位置的元素
                expressions[index] = value
            else:
                # 在指定位置插入新元素
                expressions.insert(index, value)

            # 将修改后的列表作为新值
            value = expressions
        elif value is None:
            # 如果值为None且没有指定索引，删除整个参数
            self.args.pop(arg_key, None)
            return

        # 设置参数值
        self.args[arg_key] = value
        # 设置父子关系
        self._set_parent(arg_key, value, index)

    def _set_parent(self, arg_key: str, value: t.Any, index: t.Optional[int] = None) -> None:
        """
        设置父子关系
        
        为表达式对象设置父节点、参数键和索引信息。
        这是维护表达式树结构的关键方法。
        
        Args:
            arg_key: 参数名称
            value: 要设置父子关系的值
            index: 在列表中的索引位置
        """
        if hasattr(value, "parent"):
            # 如果值是表达式对象，设置其父子关系
            value.parent = self      # 设置父节点
            value.arg_key = arg_key  # 设置参数键
            value.index = index      # 设置索引
        elif type(value) is list:
            # 如果值是列表，为列表中的每个表达式对象设置父子关系
            for index, v in enumerate(value):
                if hasattr(v, "parent"):
                    v.parent = self      # 设置父节点
                    v.arg_key = arg_key  # 设置参数键
                    v.index = index      # 设置索引

    @property
    def depth(self) -> int:
        """
        获取表达式树中当前节点的深度
        
        深度是指从根节点到当前节点的路径长度。
        根节点的深度为0，每向下一层深度加1。
        
        Returns:
            int: 当前节点在树中的深度
        """
        # 如果有父节点，返回父节点深度+1
        if self.parent:
            return self.parent.depth + 1
        # 根节点的深度为0
        return 0

    def iter_expressions(self, reverse: bool = False) -> t.Iterator[Expression]:
        """
        迭代遍历当前表达式的所有子表达式
        
        展开列表参数，返回所有子表达式的迭代器。
        支持正向和反向遍历。
        
        Args:
            reverse: 是否反向遍历，默认为False
            
        Yields:
            Expression: 子表达式对象
        """
        # 根据reverse参数决定遍历方向
        for vs in reversed(self.args.values()) if reverse else self.args.values():  # type: ignore
            if type(vs) is list:
                # 如果参数是列表，遍历列表中的每个元素
                for v in reversed(vs) if reverse else vs:  # type: ignore
                    if hasattr(v, "parent"):
                        # 只返回表达式对象
                        yield v
            else:
                # 如果参数不是列表，直接检查是否为表达式对象
                if hasattr(vs, "parent"):
                    yield vs

    def find(self, *expression_types: t.Type[E], bfs: bool = True) -> t.Optional[E]:
        """
        查找第一个匹配指定类型的表达式节点
        
        在表达式树中搜索第一个匹配指定类型的节点。
        支持广度优先搜索(BFS)和深度优先搜索(DFS)。
        
        Args:
            *expression_types: 要匹配的表达式类型
            bfs: 是否使用广度优先搜索，默认为True（False则使用深度优先搜索）
            
        Returns:
            匹配的节点，如果没有找到则返回None
        """
        # 使用find_all方法获取第一个匹配的节点
        return next(self.find_all(*expression_types, bfs=bfs), None)

    def find_all(self, *expression_types: t.Type[E], bfs: bool = True) -> t.Iterator[E]:
        """
        查找所有匹配指定类型的表达式节点
        
        在表达式树中搜索所有匹配指定类型的节点，返回生成器。
        支持广度优先搜索(BFS)和深度优先搜索(DFS)。
        
        Args:
            *expression_types: 要匹配的表达式类型
            bfs: 是否使用广度优先搜索，默认为True（False则使用深度优先搜索）
            
        Yields:
            匹配的表达式节点
        """
        # 遍历整个表达式树
        for expression in self.walk(bfs=bfs):
            # 检查当前表达式是否匹配任何一个指定类型
            if isinstance(expression, expression_types):
                yield expression

    def find_ancestor(self, *expression_types: t.Type[E]) -> t.Optional[E]:
        """
        查找最近的匹配指定类型的祖先节点
        
        从当前节点开始，向上遍历父节点链，返回第一个匹配指定类型的祖先节点。
        
        Args:
            *expression_types: 要匹配的表达式类型
            
        Returns:
            匹配的祖先节点，如果没有找到则返回None
        """
        # 从父节点开始向上遍历
        ancestor = self.parent
        # 继续向上查找，直到找到匹配的类型或到达根节点
        while ancestor and not isinstance(ancestor, expression_types):
            ancestor = ancestor.parent
        return ancestor  # type: ignore

    @property
    def parent_select(self) -> t.Optional[Select]:
        """
        获取父级SELECT语句
        
        查找当前表达式所在的SELECT语句节点。
        这是SQL解析中常用的导航方法。
        
        Returns:
            Select: 父级SELECT语句，如果没有找到则返回None
        """
        # 使用find_ancestor方法查找Select类型的祖先节点
        return self.find_ancestor(Select)

    @property
    def same_parent(self) -> bool:
        """
        检查父节点是否与当前节点类型相同
        
        判断父节点是否为相同的表达式类型。
        用于检测表达式树中的重复结构。
        
        Returns:
            bool: 如果父节点类型相同则返回True，否则返回False
        """
        # 使用type()而不是isinstance()确保严格的类型匹配
        return type(self.parent) is self.__class__

    def root(self) -> Expression:
        """
        获取表达式树的根节点
        
        从当前节点开始，向上遍历直到找到没有父节点的根节点。
        
        Returns:
            Expression: 表达式树的根节点
        """
        expression = self
        # 向上遍历父节点链，直到找到根节点
        while expression.parent:
            expression = expression.parent
        return expression

    def walk(
        self, bfs: bool = True, prune: t.Optional[t.Callable[[Expression], bool]] = None
    ) -> t.Iterator[Expression]:
        """
        遍历表达式树中的所有节点
        
        提供统一的遍历接口，支持广度优先搜索(BFS)和深度优先搜索(DFS)。
        可以指定剪枝函数来跳过某些分支的遍历。
        
        Args:
            bfs: 是否使用广度优先搜索，默认为True（False则使用深度优先搜索）
            prune: 剪枝函数，返回True时停止遍历当前分支
            
        Returns:
            Iterator[Expression]: 遍历所有节点的生成器
        """
        # 根据bfs参数选择遍历算法
        if bfs:
            yield from self.bfs(prune=prune)
        else:
            yield from self.dfs(prune=prune)

    def dfs(
        self, prune: t.Optional[t.Callable[[Expression], bool]] = None
    ) -> t.Iterator[Expression]:
        """
        深度优先搜索遍历表达式树
        
        使用栈实现深度优先搜索，先访问父节点，再访问子节点。
        适合需要深入分析表达式结构的场景。
        
        Args:
            prune: 剪枝函数，返回True时停止遍历当前分支
            
        Returns:
            Iterator[Expression]: 深度优先遍历的生成器
        """
        # 使用栈实现深度优先搜索
        stack = [self]

        while stack:
            # 从栈顶取出节点
            node = stack.pop()

            # 先访问当前节点
            yield node

            # 如果剪枝函数返回True，跳过当前分支的遍历
            if prune and prune(node):
                continue

            # 将子节点按反向顺序压入栈中
            # 使用reverse=True确保正向遍历顺序
            for v in node.iter_expressions(reverse=True):
                stack.append(v)

    def bfs(
        self, prune: t.Optional[t.Callable[[Expression], bool]] = None
    ) -> t.Iterator[Expression]:
        """
        广度优先搜索遍历表达式树
        
        使用队列实现广度优先搜索，按层级访问节点。
        适合需要按层级分析表达式结构的场景。
        
        Args:
            prune: 剪枝函数，返回True时停止遍历当前分支
            
        Returns:
            Iterator[Expression]: 广度优先遍历的生成器
        """
        # 使用双端队列实现广度优先搜索
        queue = deque([self])

        while queue:
            # 从队列头部取出节点
            node = queue.popleft()

            # 访问当前节点
            yield node

            # 如果剪枝函数返回True，跳过当前分支的遍历
            if prune and prune(node):
                continue

            # 将子节点按正向顺序加入队列尾部
            for v in node.iter_expressions():
                queue.append(v)

    def unnest(self):
        """
        去除括号包装，返回第一个非括号子表达式或自身
        
        递归去除表达式外层的括号，直到找到非括号的表达式。
        用于简化表达式的结构表示。
        
        Returns:
            Expression: 去除括号后的表达式
        """
        expression = self
        # 循环去除括号，直到不是Paren类型
        while type(expression) is Paren:
            expression = expression.this
        return expression

    def unalias(self):
        """
        去除别名包装，返回内部表达式或自身
        
        如果当前表达式是别名，返回其内部表达式；
        否则返回自身。
        
        Returns:
            Expression: 去除别名后的表达式
        """
        # 如果是别名表达式，返回其内部表达式
        if isinstance(self, Alias):
            return self.this
        # 否则返回自身
        return self

    def unnest_operands(self):
        """
        返回去除括号后的操作数元组
        
        对表达式的所有操作数执行unnest操作，返回处理后的操作数元组。
        用于获取表达式的核心操作数，忽略括号包装。
        
        Returns:
            tuple: 去除括号后的操作数元组
        """
        # 对所有子表达式执行unnest操作，返回元组
        return tuple(arg.unnest() for arg in self.iter_expressions())

    def flatten(self, unnest=True):
        """
        Returns a generator which yields child nodes whose parents are the same class.

        A AND B AND C -> [A, B, C]
        """
        for node in self.dfs(prune=lambda n: n.parent and type(n) is not self.__class__):
            if type(node) is not self.__class__:
                yield node.unnest() if unnest and not isinstance(node, Subquery) else node

    def __str__(self) -> str:
        return self.sql()

    def __repr__(self) -> str:
        return _to_s(self)

    def to_s(self) -> str:
        """
        Same as __repr__, but includes additional information which can be useful
        for debugging, like empty or missing args and the AST nodes' object IDs.
        """
        return _to_s(self, verbose=True)

    def sql(self, dialect: DialectType = None, **opts) -> str:
        """
        Returns SQL string representation of this tree.

        Args:
            dialect: the dialect of the output SQL string (eg. "spark", "hive", "presto", "mysql").
            opts: other `sqlglot.generator.Generator` options.

        Returns:
            The SQL string.
        """
        from sqlglot.dialects import Dialect

        return Dialect.get_or_raise(dialect).generate(self, **opts)

    def transform(self, fun: t.Callable, *args: t.Any, copy: bool = True, **kwargs) -> Expression:
        """
        Visits all tree nodes (excluding already transformed ones)
        and applies the given transformation function to each node.

        Args:
            fun: a function which takes a node as an argument and returns a
                new transformed node or the same node without modifications. If the function
                returns None, then the corresponding node will be removed from the syntax tree.
            copy: if set to True a new tree instance is constructed, otherwise the tree is
                modified in place.

        Returns:
            The transformed tree.
        """
        root = None
        new_node = None

        for node in (self.copy() if copy else self).dfs(prune=lambda n: n is not new_node):
            parent, arg_key, index = node.parent, node.arg_key, node.index
            new_node = fun(node, *args, **kwargs)

            if not root:
                root = new_node
            elif parent and arg_key and new_node is not node:
                parent.set(arg_key, new_node, index)

        assert root
        return root.assert_is(Expression)

    @t.overload
    def replace(self, expression: E) -> E: ...

    @t.overload
    def replace(self, expression: None) -> None: ...

    def replace(self, expression):
        """
        Swap out this expression with a new expression.

        For example::

            >>> tree = Select().select("x").from_("tbl")
            >>> tree.find(Column).replace(column("y"))
            Column(
              this=Identifier(this=y, quoted=False))
            >>> tree.sql()
            'SELECT y FROM tbl'

        Args:
            expression: new node

        Returns:
            The new expression or expressions.
        """
        parent = self.parent

        if not parent or parent is expression:
            return expression

        key = self.arg_key
        value = parent.args.get(key)

        if type(expression) is list and isinstance(value, Expression):
            # We are trying to replace an Expression with a list, so it's assumed that
            # the intention was to really replace the parent of this expression.
            value.parent.replace(expression)
        else:
            parent.set(key, expression, self.index)

        if expression is not self:
            self.parent = None
            self.arg_key = None
            self.index = None

        return expression

    def pop(self: E) -> E:
        """
        Remove this expression from its AST.

        Returns:
            The popped expression.
        """
        self.replace(None)
        return self

    def assert_is(self, type_: t.Type[E]) -> E:
        """
        Assert that this `Expression` is an instance of `type_`.

        If it is NOT an instance of `type_`, this raises an assertion error.
        Otherwise, this returns this expression.

        Examples:
            This is useful for type security in chained expressions:

            >>> import sqlglot
            >>> sqlglot.parse_one("SELECT x from y").assert_is(Select).select("z").sql()
            'SELECT x, z FROM y'
        """
        if not isinstance(self, type_):
            raise AssertionError(f"{self} is not {type_}.")
        return self

    def error_messages(self, args: t.Optional[t.Sequence] = None) -> t.List[str]:
        """
        Checks if this expression is valid (e.g. all mandatory args are set).

        Args:
            args: a sequence of values that were used to instantiate a Func expression. This is used
                to check that the provided arguments don't exceed the function argument limit.

        Returns:
            A list of error messages for all possible errors that were found.
        """
        errors: t.List[str] = []

        for k in self.args:
            if k not in self.arg_types:
                errors.append(f"Unexpected keyword: '{k}' for {self.__class__}")
        for k, mandatory in self.arg_types.items():
            v = self.args.get(k)
            if mandatory and (v is None or (isinstance(v, list) and not v)):
                errors.append(f"Required keyword: '{k}' missing for {self.__class__}")

        if (
            args
            and isinstance(self, Func)
            and len(args) > len(self.arg_types)
            and not self.is_var_len_args
        ):
            errors.append(
                f"The number of provided arguments ({len(args)}) is greater than "
                f"the maximum number of supported arguments ({len(self.arg_types)})"
            )

        return errors

    def dump(self):
        """
        Dump this Expression to a JSON-serializable dict.
        """
        from sqlglot.serde import dump

        return dump(self)

    @classmethod
    def load(cls, obj):
        """
        Load a dict (as returned by `Expression.dump`) into an Expression instance.
        """
        from sqlglot.serde import load

        return load(obj)

    def and_(
        self,
        *expressions: t.Optional[ExpOrStr],
        dialect: DialectType = None,
        copy: bool = True,
        wrap: bool = True,
        **opts,
    ) -> Condition:
        """
        AND this condition with one or multiple expressions.

        Example:
            >>> condition("x=1").and_("y=1").sql()
            'x = 1 AND y = 1'

        Args:
            *expressions: the SQL code strings to parse.
                If an `Expression` instance is passed, it will be used as-is.
            dialect: the dialect used to parse the input expression.
            copy: whether to copy the involved expressions (only applies to Expressions).
            wrap: whether to wrap the operands in `Paren`s. This is true by default to avoid
                precedence issues, but can be turned off when the produced AST is too deep and
                causes recursion-related issues.
            opts: other options to use to parse the input expressions.

        Returns:
            The new And condition.
        """
        """
        将当前条件与一个或多个表达式进行AND逻辑运算。
        
        此方法允许链式调用，用于构建复杂的布尔条件表达式。

        Example:
            >>> condition("x=1").and_("y=1").sql()
            'x = 1 AND y = 1'

        Args:
            *expressions: 要解析的SQL代码字符串。
                如果传入的是`Expression`实例，将直接使用而不进行解析。
            dialect: 用于解析输入表达式的SQL方言。
            copy: 是否复制涉及的表达式（仅适用于Expression对象）。
            wrap: 是否将操作数包装在`Paren`中。默认为True以避免
                优先级问题，但当生成的AST过深导致递归相关问题时可以关闭。
            opts: 用于解析输入表达式的其他选项。

        Returns:
            新的And条件表达式。
        """
        # 调用全局and_函数，将当前表达式与传入的表达式进行AND运算
        # 这种设计模式允许方法链式调用，提高代码可读性

        return and_(self, *expressions, dialect=dialect, copy=copy, wrap=wrap, **opts)

    def or_(
        self,
        *expressions: t.Optional[ExpOrStr],
        dialect: DialectType = None,
        copy: bool = True,
        wrap: bool = True,
        **opts,
    ) -> Condition:
        """
        OR this condition with one or multiple expressions.

        Example:
            >>> condition("x=1").or_("y=1").sql()
            'x = 1 OR y = 1'

        Args:
            *expressions: the SQL code strings to parse.
                If an `Expression` instance is passed, it will be used as-is.
            dialect: the dialect used to parse the input expression.
            copy: whether to copy the involved expressions (only applies to Expressions).
            wrap: whether to wrap the operands in `Paren`s. This is true by default to avoid
                precedence issues, but can be turned off when the produced AST is too deep and
                causes recursion-related issues.
            opts: other options to use to parse the input expressions.

        Returns:
            The new Or condition.
        """
        """
        将当前条件与一个或多个表达式进行OR逻辑运算。
        
        此方法允许链式调用，用于构建复杂的布尔条件表达式。

        Example:
            >>> condition("x=1").or_("y=1").sql()
            'x = 1 OR y = 1'

        Args:
            *expressions: 要解析的SQL代码字符串。
                如果传入的是`Expression`实例，将直接使用而不进行解析。
            dialect: 用于解析输入表达式的SQL方言。
            copy: 是否复制涉及的表达式（仅适用于Expression对象）。
            wrap: 是否将操作数包装在`Paren`中。默认为True以避免
                优先级问题，但当生成的AST过深导致递归相关问题时可以关闭。
            opts: 用于解析输入表达式的其他选项。

        Returns:
            新的Or条件表达式。
        """
        # 调用全局or_函数，将当前表达式与传入的表达式进行OR运算
        # 与and_方法类似，支持链式调用和灵活的表达式组合

        return or_(self, *expressions, dialect=dialect, copy=copy, wrap=wrap, **opts)

    def not_(self, copy: bool = True):
        """
        Wrap this condition with NOT.

        Example:
            >>> condition("x=1").not_().sql()
            'NOT x = 1'

        Args:
            copy: whether to copy this object.

        Returns:
            The new Not instance.
        """
        """
        将当前条件包装在NOT逻辑运算符中。
        
        用于对布尔表达式进行逻辑取反操作。

        Example:
            >>> condition("x=1").not_().sql()
            'NOT x = 1'

        Args:
            copy: 是否复制当前对象。

        Returns:
            新的Not表达式实例。
        """
        # 调用全局not_函数，对当前表达式进行逻辑取反
        # copy参数控制是否创建新对象，避免修改原始表达式

        return not_(self, copy=copy)

    def update_positions(
        self: E, other: t.Optional[Token | Expression] = None, **kwargs: t.Any
    ) -> E:
        """
        Update this expression with positions from a token or other expression.

        Args:
            other: a token or expression to update this expression with.

        Returns:
            The updated expression.
        """
        """
        使用来自token或其他表达式的位置信息更新当前表达式。
        
        位置信息包括行号、列号、起始和结束位置，用于错误报告和调试。

        Args:
            other: 用于更新当前表达式的token或表达式。

        Returns:
            更新后的表达式。
        """
        # 如果other是Expression类型，从其他表达式的元数据中提取位置信息
        # 只提取位置相关的元数据键，避免复制不相关的信息

        if isinstance(other, Expression):
            self.meta.update({k: v for k, v in other.meta.items() if k in POSITION_META_KEYS})
        
        # 如果other是Token类型且不为None，直接从token对象获取位置属性        
        elif other is not None:
            self.meta.update(
                {
                    "line": other.line,
                    "col": other.col,
                    "start": other.start,
                    "end": other.end,
                }
            )
        # 最后处理通过kwargs传入的位置信息，允许手动指定位置
        # 这样可以覆盖从other对象获取的位置信息
        self.meta.update({k: v for k, v in kwargs.items() if k in POSITION_META_KEYS})
        return self

    def as_(
        self,
        alias: str | Identifier,
        quoted: t.Optional[bool] = None,
        dialect: DialectType = None,
        copy: bool = True,
        **opts,
    ) -> Alias:
        """
        为当前表达式创建别名。
        
        在SQL中，别名用于为表、列或子查询提供简短的名称，提高可读性。

        Args:
            alias: 别名，可以是字符串或Identifier对象。
            quoted: 是否对别名加引号，None表示使用默认行为。
            dialect: 用于生成别名的SQL方言。
            copy: 是否复制当前表达式。
            opts: 其他选项。

        Returns:
            包含别名的Alias表达式。
        """
        # 调用全局alias_函数创建别名表达式
        # 这种设计保持了API的一致性，所有表达式操作都通过全局函数实现
        return alias_(self, alias, quoted=quoted, dialect=dialect, copy=copy, **opts)

    def _binop(self, klass: t.Type[E], other: t.Any, reverse: bool = False) -> E:
        this = self.copy()
        other = convert(other, copy=True)
        if not isinstance(this, klass) and not isinstance(other, klass):
            this = _wrap(this, Binary)
            other = _wrap(other, Binary)
        if reverse:
            return klass(this=other, expression=this)
        return klass(this=this, expression=other)

    def __getitem__(self, other: ExpOrStr | t.Tuple[ExpOrStr]) -> Bracket:
        return Bracket(
            this=self.copy(), expressions=[convert(e, copy=True) for e in ensure_list(other)]
        )

    def __iter__(self) -> t.Iterator:
        if "expressions" in self.arg_types:
            return iter(self.args.get("expressions") or [])
        # We define this because __getitem__ converts Expression into an iterable, which is
        # problematic because one can hit infinite loops if they do "for x in some_expr: ..."
        # See: https://peps.python.org/pep-0234/
        raise TypeError(f"'{self.__class__.__name__}' object is not iterable")

    def isin(
        self,
        *expressions: t.Any,
        query: t.Optional[ExpOrStr] = None,
        unnest: t.Optional[ExpOrStr] | t.Collection[ExpOrStr] = None,
        copy: bool = True,
        **opts,
    ) -> In:
        subquery = maybe_parse(query, copy=copy, **opts) if query else None
        if subquery and not isinstance(subquery, Subquery):
            subquery = subquery.subquery(copy=False)

        return In(
            this=maybe_copy(self, copy),
            expressions=[convert(e, copy=copy) for e in expressions],
            query=subquery,
            unnest=(
                Unnest(
                    expressions=[
                        maybe_parse(t.cast(ExpOrStr, e), copy=copy, **opts)
                        for e in ensure_list(unnest)
                    ]
                )
                if unnest
                else None
            ),
        )

    def between(
        self,
        low: t.Any,
        high: t.Any,
        copy: bool = True,
        symmetric: t.Optional[bool] = None,
        **opts,
    ) -> Between:
        """
        创建BETWEEN条件表达式，用于检查值是否在指定范围内。
        
        BETWEEN操作符在SQL中用于范围查询，等价于 value >= low AND value <= high。

        Args:
            low: 范围的下界值。
            high: 范围的上界值。
            copy: 是否复制当前表达式。
            symmetric: 是否使用对称BETWEEN（某些数据库支持）。
            opts: 其他选项。

        Returns:
            Between表达式对象。
        """
        # 创建Between表达式，包含当前表达式、下界和上界
        # maybe_copy确保根据copy参数决定是否复制当前表达式
        # convert函数将输入值转换为适当的表达式类型
        between = Between(
            this=maybe_copy(self, copy),           # 当前表达式作为被比较的值
            low=convert(low, copy=copy, **opts),   # 下界值，转换为表达式
            high=convert(high, copy=copy, **opts), # 上界值，转换为表达式
        )
        # 如果指定了symmetric参数，设置对称BETWEEN标志
        # 对称BETWEEN在某些数据库中是特殊语法，表示范围是双向的
        if symmetric is not None:
            between.set("symmetric", symmetric)

        return between

    def is_(self, other: ExpOrStr) -> Is:
        """
        创建IS条件表达式，用于NULL值比较。
        
        IS操作符专门用于与NULL值进行比较，是SQL中的特殊比较操作符。

        Args:
            other: 要比较的表达式或字符串。

        Returns:
            Is表达式对象。
        """
        # 使用_binop方法创建二元操作符表达式
        # IS操作符主要用于NULL值检查，如 column IS NULL
        return self._binop(Is, other)

    def like(self, other: ExpOrStr) -> Like:
        """
        创建LIKE条件表达式，用于模式匹配。
        
        LIKE操作符用于字符串模式匹配，支持通配符%和_。

        Args:
            other: 模式字符串或表达式。

        Returns:
            Like表达式对象。
        """
        # LIKE操作符用于字符串模式匹配，区分大小写
        # 支持%匹配任意字符序列，_匹配单个字符
        return self._binop(Like, other)

    def ilike(self, other: ExpOrStr) -> ILike:
        """
        创建ILIKE条件表达式，用于不区分大小写的模式匹配。
        
        ILIKE是PostgreSQL等数据库中的不区分大小写LIKE操作符。

        Args:
            other: 模式字符串或表达式。

        Returns:
            ILike表达式对象。
        """
        # ILIKE是LIKE的不区分大小写版本
        # 主要用于PostgreSQL等支持此语法的数据库
        return self._binop(ILike, other)

    def eq(self, other: t.Any) -> EQ:
        """
        创建等于比较表达式。
        
        用于检查两个值是否相等。

        Args:
            other: 要比较的值或表达式。

        Returns:
            EQ表达式对象。
        """
        # 创建等于比较操作符
        # 这是最常用的比较操作符，用于相等性检查
        return self._binop(EQ, other)

    def neq(self, other: t.Any) -> NEQ:
        """
        创建不等于比较表达式。
        
        用于检查两个值是否不相等。

        Args:
            other: 要比较的值或表达式。

        Returns:
            NEQ表达式对象。
        """
        # 创建不等于比较操作符
        # 等价于 != 或 <> 操作符
        return self._binop(NEQ, other)

    def rlike(self, other: ExpOrStr) -> RegexpLike:
        """
        创建正则表达式匹配条件。
        
        用于基于正则表达式的字符串匹配。

        Args:
            other: 正则表达式模式字符串。

        Returns:
            RegexpLike表达式对象。
        """
        # 创建正则表达式匹配操作符
        # 支持复杂的字符串模式匹配，比LIKE更强大但性能较低
        return self._binop(RegexpLike, other)

    def div(self, other: ExpOrStr, typed: bool = False, safe: bool = False) -> Div:
        """
        创建除法运算表达式。
        
        支持整数除法和安全除法（避免除零错误）。

        Args:
            other: 除数表达式或值。
            typed: 是否进行整数除法（截断小数部分）。
            safe: 是否使用安全除法（除零时返回NULL而不是报错）。

        Returns:
            Div表达式对象。
        """
        # 创建除法表达式
        div = self._binop(Div, other)
        # 设置除法类型标志：typed表示整数除法
        # 整数除法会截断小数部分，如 5/2 = 2
        div.set("typed", typed)
        # 设置安全除法标志：safe表示除零时返回NULL
        # 避免除零错误，提高SQL的健壮性
        div.set("safe", safe)
        return div

    def asc(self, nulls_first: bool = True) -> Ordered:
        """
        创建升序排序表达式。
        
        用于ORDER BY子句中指定升序排序。

        Args:
            nulls_first: NULL值是否排在前面。

        Returns:
            Ordered表达式对象。
        """
        # 创建升序排序表达式
        # 默认情况下NULL值排在前面，符合SQL标准
        # 使用copy()确保不修改原始表达式
        return Ordered(this=self.copy(), nulls_first=nulls_first)

    def desc(self, nulls_first: bool = False) -> Ordered:
        """
        创建降序排序表达式。
        
        用于ORDER BY子句中指定降序排序。

        Args:
            nulls_first: NULL值是否排在前面。

        Returns:
            Ordered表达式对象。
        """
        # 创建降序排序表达式
        # desc=True表示降序排序
        # 默认情况下NULL值排在后面，与升序相反
        # 使用copy()确保不修改原始表达式
        return Ordered(this=self.copy(), desc=True, nulls_first=nulls_first)


    def __lt__(self, other: t.Any) -> LT:
        """
        实现小于比较操作符 (<)。
        
        允许使用Python的 < 操作符来创建SQL的LT表达式。

        Args:
            other: 要比较的值或表达式。

        Returns:
            LT表达式对象。
        """
        # 使用_binop方法创建二元操作符表达式
        # 这样可以使用Python语法 expr1 < expr2 来构建SQL比较条件
        return self._binop(LT, other)

    def __le__(self, other: t.Any) -> LTE:
        """
        实现小于等于比较操作符 (<=)。
        
        允许使用Python的 <= 操作符来创建SQL的LTE表达式。

        Args:
            other: 要比较的值或表达式。

        Returns:
            LTE表达式对象。
        """
        # 创建小于等于比较表达式
        # 等价于SQL中的 <= 操作符
        return self._binop(LTE, other)

    def __gt__(self, other: t.Any) -> GT:
        """
        实现大于比较操作符 (>)。
        
        允许使用Python的 > 操作符来创建SQL的GT表达式。

        Args:
            other: 要比较的值或表达式。

        Returns:
            GT表达式对象。
        """
        # 创建大于比较表达式
        # 等价于SQL中的 > 操作符
        return self._binop(GT, other)

    def __ge__(self, other: t.Any) -> GTE:
        """
        实现大于等于比较操作符 (>=)。
        
        允许使用Python的 >= 操作符来创建SQL的GTE表达式。

        Args:
            other: 要比较的值或表达式。

        Returns:
            GTE表达式对象。
        """
        # 创建大于等于比较表达式
        # 等价于SQL中的 >= 操作符
        return self._binop(GTE, other)

    def __add__(self, other: t.Any) -> Add:
        """
        实现加法操作符 (+)。
        
        允许使用Python的 + 操作符来创建SQL的Add表达式。

        Args:
            other: 要相加的值或表达式。

        Returns:
            Add表达式对象。
        """
        # 创建加法表达式
        # 支持数值相加和字符串连接（取决于数据库方言）
        return self._binop(Add, other)

    def __radd__(self, other: t.Any) -> Add:
        """
        实现右加法操作符 (+)，用于处理左操作数不支持加法的情况。
        
        当左操作数没有实现__add__方法时，Python会尝试调用右操作数的__radd__。

        Args:
            other: 左操作数。

        Returns:
            Add表达式对象。
        """
        # 使用reverse=True参数，表示操作数顺序被反转
        # 例如：5 + column 会调用 column.__radd__(5)
        return self._binop(Add, other, reverse=True)

    def __sub__(self, other: t.Any) -> Sub:
        """
        实现减法操作符 (-)。
        
        允许使用Python的 - 操作符来创建SQL的Sub表达式。

        Args:
            other: 要相减的值或表达式。

        Returns:
            Sub表达式对象。
        """
        # 创建减法表达式
        # 用于数值相减运算
        return self._binop(Sub, other)

    def __rsub__(self, other: t.Any) -> Sub:
        """
        实现右减法操作符 (-)，用于处理左操作数不支持减法的情况。

        Args:
            other: 左操作数。

        Returns:
            Sub表达式对象。
        """
        # 处理右减法，如 10 - column 的情况
        # reverse=True确保操作数顺序正确
        return self._binop(Sub, other, reverse=True)

    def __mul__(self, other: t.Any) -> Mul:
        """
        实现乘法操作符 (*)。
        
        允许使用Python的 * 操作符来创建SQL的Mul表达式。

        Args:
            other: 要相乘的值或表达式。

        Returns:
            Mul表达式对象。
        """
        # 创建乘法表达式
        # 用于数值相乘运算
        return self._binop(Mul, other)

    def __rmul__(self, other: t.Any) -> Mul:
        """
        实现右乘法操作符 (*)，用于处理左操作数不支持乘法的情况。

        Args:
            other: 左操作数。

        Returns:
            Mul表达式对象。
        """
        # 处理右乘法，如 3 * column 的情况
        return self._binop(Mul, other, reverse=True)

    def __truediv__(self, other: t.Any) -> Div:
        """
        实现真除法操作符 (/)。
        
        允许使用Python的 / 操作符来创建SQL的Div表达式。

        Args:
            other: 除数。

        Returns:
            Div表达式对象。
        """
        # 创建真除法表达式（浮点除法）
        # 与__floordiv__不同，这里保留小数部分
        return self._binop(Div, other)

    def __rtruediv__(self, other: t.Any) -> Div:
        """
        实现右真除法操作符 (/)，用于处理左操作数不支持除法的情况。

        Args:
            other: 左操作数。

        Returns:
            Div表达式对象。
        """
        # 处理右除法，如 10 / column 的情况
        return self._binop(Div, other, reverse=True)

    def __floordiv__(self, other: t.Any) -> IntDiv:
        """
        实现整数除法操作符 (//)。
        
        允许使用Python的 // 操作符来创建SQL的IntDiv表达式。

        Args:
            other: 除数。

        Returns:
            IntDiv表达式对象。
        """
        # 创建整数除法表达式
        # 截断小数部分，只保留整数结果
        return self._binop(IntDiv, other)

    def __rfloordiv__(self, other: t.Any) -> IntDiv:
        """
        实现右整数除法操作符 (//)，用于处理左操作数不支持整数除法的情况。

        Args:
            other: 左操作数。

        Returns:
            IntDiv表达式对象。
        """
        # 处理右整数除法，如 10 // column 的情况
        return self._binop(IntDiv, other, reverse=True)

    def __mod__(self, other: t.Any) -> Mod:
        """
        实现取模操作符 (%)。
        
        允许使用Python的 % 操作符来创建SQL的Mod表达式。

        Args:
            other: 模数。

        Returns:
            Mod表达式对象。
        """
        # 创建取模表达式
        # 用于计算余数运算
        return self._binop(Mod, other)

    def __rmod__(self, other: t.Any) -> Mod:
        """
        实现右取模操作符 (%)，用于处理左操作数不支持取模的情况。

        Args:
            other: 左操作数。

        Returns:
            Mod表达式对象。
        """
        # 处理右取模，如 10 % column 的情况
        return self._binop(Mod, other, reverse=True)

    def __pow__(self, other: t.Any) -> Pow:
        """
        实现幂运算操作符 (**)。
        
        允许使用Python的 ** 操作符来创建SQL的Pow表达式。

        Args:
            other: 指数。

        Returns:
            Pow表达式对象。
        """
        # 创建幂运算表达式
        # 用于计算乘方运算
        return self._binop(Pow, other)

    def __rpow__(self, other: t.Any) -> Pow:
        """
        实现右幂运算操作符 (**)，用于处理左操作数不支持幂运算的情况。

        Args:
            other: 左操作数（底数）。

        Returns:
            Pow表达式对象。
        """
        # 处理右幂运算，如 2 ** column 的情况
        return self._binop(Pow, other, reverse=True)

    def __and__(self, other: t.Any) -> And:
        """
        实现逻辑与操作符 (&)。
        
        允许使用Python的 & 操作符来创建SQL的And表达式。

        Args:
            other: 要进行逻辑与运算的表达式。

        Returns:
            And表达式对象。
        """
        # 创建逻辑与表达式
        # 注意：这里使用&而不是and，因为and是Python关键字
        return self._binop(And, other)

    def __rand__(self, other: t.Any) -> And:
        """
        实现右逻辑与操作符 (&)，用于处理左操作数不支持逻辑与的情况。

        Args:
            other: 左操作数。

        Returns:
            And表达式对象。
        """
        # 处理右逻辑与，如 True & column 的情况
        return self._binop(And, other, reverse=True)

    def __or__(self, other: t.Any) -> Or:
        """
        实现逻辑或操作符 (|)。
        
        允许使用Python的 | 操作符来创建SQL的Or表达式。

        Args:
            other: 要进行逻辑或运算的表达式。

        Returns:
            Or表达式对象。
        """
        # 创建逻辑或表达式
        # 注意：这里使用|而不是or，因为or是Python关键字
        return self._binop(Or, other)

    def __ror__(self, other: t.Any) -> Or:
        """
        实现右逻辑或操作符 (|)，用于处理左操作数不支持逻辑或的情况。

        Args:
            other: 左操作数。

        Returns:
            Or表达式对象。
        """
        # 处理右逻辑或，如 False | column 的情况
        return self._binop(Or, other, reverse=True)

    def __neg__(self) -> Neg:
        """
        实现一元负号操作符 (-)。
        
        允许使用Python的 - 操作符来创建SQL的Neg表达式。

        Returns:
            Neg表达式对象。
        """
        # 创建一元负号表达式
        # _wrap函数确保表达式被正确包装，特别是对于Binary类型的表达式
        # 一元操作符需要特殊处理，因为它们的操作数结构不同于二元操作符
        return Neg(this=_wrap(self.copy(), Binary))

    def __invert__(self) -> Not:
        """
        实现按位取反操作符 (~)。
        
        允许使用Python的 ~ 操作符来创建SQL的Not表达式。

        Returns:
            Not表达式对象。
        """
        # 创建逻辑非表达式
        # 使用copy()确保不修改原始表达式
        # 这里使用~操作符来表示逻辑非，因为not是Python关键字
        return not_(self.copy())


IntoType = t.Union[
    str,
    t.Type[Expression],
    t.Collection[t.Union[str, t.Type[Expression]]],
]
ExpOrStr = t.Union[str, Expression]


class Condition(Expression):
    """
    逻辑条件表达式基类。
    
    表示SQL中的逻辑条件，如 x AND y 这样的复合条件，或者简单的条件表达式。
    这是所有布尔条件表达式的基类，包括比较操作、逻辑运算等。
    """
    # 继承自Expression，提供逻辑条件的基础功能
    # 所有需要返回布尔值的SQL表达式都应该继承此类


class Predicate(Condition):
    """
    谓词表达式类。
    
    表示SQL中的关系表达式，如 x = y、x > 1、x >= y 等比较操作。
    谓词是条件的一种特殊形式，专门用于表示两个值之间的关系比较。
    """
    # 继承自Condition，专门用于关系比较
    # 谓词表达式总是返回布尔值（真或假）
    # 包括等于、不等于、大于、小于、大于等于、小于等于等比较操作


class DerivedTable(Expression):
    """
    派生表表达式类。
    
    表示SQL中的派生表（子查询作为表使用），如 FROM (SELECT ...) AS alias。
    派生表允许将子查询的结果作为临时表在外部查询中使用。
    """
    
    @property
    def selects(self) -> t.List[Expression]:
        """
        获取派生表中的选择表达式列表。
        
        如果当前表达式包含一个查询（Query），则返回该查询的选择列表；
        否则返回空列表。

        Returns:
            选择表达式列表。
        """
        # 检查this属性是否为Query类型
        # 如果是查询，则返回查询的选择列表；否则返回空列表
        # 这种设计允许派生表访问其内部查询的列信息
        return self.this.selects if isinstance(self.this, Query) else []

    @property
    def named_selects(self) -> t.List[str]:
        """
        获取派生表中命名选择表达式的名称列表。
        
        返回所有选择表达式的输出名称，这些名称可以用作列名。

        Returns:
            选择表达式的名称列表。
        """
        # 从selects属性中提取每个选择表达式的输出名称
        # output_name属性包含表达式的别名或默认名称
        # 这些名称用于标识派生表的列，便于外部查询引用
        return [select.output_name for select in self.selects]



class Query(Expression):
    """
    查询表达式基类。
    
    表示SQL查询的基础类，包括SELECT、INSERT、UPDATE、DELETE等所有查询类型。
    提供了查询构建的通用方法，如子查询、限制、排序、条件过滤等。
    """
    
    def subquery(self, alias: t.Optional[ExpOrStr] = None, copy: bool = True) -> Subquery:
        """
        将当前查询包装为子查询。
        
        创建一个Subquery对象，将当前查询作为子查询使用，通常用于FROM子句或WHERE子句中。

        Example:
            >>> subquery = Select().select("x").from_("tbl").subquery()
            >>> Select().select("x").from_(subquery).sql()
            'SELECT x FROM (SELECT x FROM tbl)'

        Args:
            alias: an optional alias for the subquery.
            copy: if `False`, modify this expression instance in-place.
            alias: 子查询的可选别名。
            copy: 如果为False，则就地修改此表达式实例。
        Returns:
            Subquery表达式对象
        """
        # 根据copy参数决定是否复制当前查询实例
        # 这确保了原始查询不被意外修改
        instance = maybe_copy(self, copy)
        
        # 处理别名参数：如果不是Expression类型，则转换为TableAlias
        # 如果alias为None，则保持None（表示无别名）
        if not isinstance(alias, Expression):
            alias = TableAlias(this=to_identifier(alias)) if alias else None

        # 创建Subquery对象，包含查询实例和别名
        return Subquery(this=instance, alias=alias)

    def limit(
        self: Q, expression: ExpOrStr | int, dialect: DialectType = None, copy: bool = True, **opts
    ) -> Q:
        """
        Adds a LIMIT clause to this query.
        为查询添加LIMIT子句。
        限制查询返回的行数，用于分页或限制结果集大小。

        Example:
            >>> select("1").union(select("1")).limit(1).sql()
            'SELECT 1 UNION SELECT 1 LIMIT 1'

        Args:
            expression: the SQL code string to parse.
                This can also be an integer.
                If a `Limit` instance is passed, it will be used as-is.
                If another `Expression` instance is passed, it will be wrapped in a `Limit`.
            dialect: the dialect used to parse the input expression.
            copy: if `False`, modify this expression instance in-place.
            opts: other options to use to parse the input expressions.
            expression: 要解析的SQL代码字符串，也可以是整数。
                如果传入Limit实例，将直接使用。
                如果传入其他Expression实例，将被包装在Limit中。
            dialect: 用于解析输入表达式的SQL方言。
            copy: 如果为False，则就地修改此表达式实例。
            opts: 用于解析输入表达式的其他选项。
        Returns:
            A limited Select expression.
            带LIMIT限制的查询表达式。
        """
        # 使用_apply_builder辅助函数来构建LIMIT子句
        # 这种模式统一了各种查询子句的构建逻辑
        return _apply_builder(
            expression=expression,
            instance=self,
            arg="limit",                    # 在查询对象中设置limit参数
            into=Limit,                     # 创建Limit类型的表达式
            prefix="LIMIT",                 # SQL关键字前缀
            dialect=dialect,
            copy=copy,
            into_arg="expression",          # 传递给Limit构造函数的参数名
            **opts,
        )

    def offset(
        self: Q, expression: ExpOrStr | int, dialect: DialectType = None, copy: bool = True, **opts
    ) -> Q:
        """
        设置OFFSET表达式。
        
        指定查询结果跳过的行数，通常与LIMIT一起使用实现分页。

        Example:
            >>> Select().from_("tbl").select("x").offset(10).sql()
            'SELECT x FROM tbl OFFSET 10'

        Args:
            expression: 要解析的SQL代码字符串，也可以是整数。
                如果传入Offset实例，将直接使用。
                如果传入其他Expression实例，将被包装在Offset中。
            dialect: 用于解析输入表达式的SQL方言。
            copy: 如果为False，则就地修改此表达式实例。
            opts: 用于解析输入表达式的其他选项。

        Returns:
            修改后的查询表达式。
        """
        # 使用相同的_apply_builder模式构建OFFSET子句
        return _apply_builder(
            expression=expression,
            instance=self,
            arg="offset",                   # 在查询对象中设置offset参数
            into=Offset,                    # 创建Offset类型的表达式
            prefix="OFFSET",                # SQL关键字前缀
            dialect=dialect,
            copy=copy,
            into_arg="expression",          # 传递给Offset构造函数的参数名
            **opts,
        )

    def order_by(
        self: Q,
        *expressions: t.Optional[ExpOrStr],
        append: bool = True,
        dialect: DialectType = None,
        copy: bool = True,
        **opts,
    ) -> Q:
        """
        设置ORDER BY表达式。
        
        指定查询结果的排序规则，可以按多个列进行排序。

        Example:
            >>> Select().from_("tbl").select("x").order_by("x DESC").sql()
            'SELECT x FROM tbl ORDER BY x DESC'

        Args:
            *expressions: 要解析的SQL代码字符串。
                如果传入Group实例，将直接使用。
                如果传入其他Expression实例，将被包装在Order中。
            append: 如果为True，添加到现有表达式；否则扁平化所有Order表达式。
            dialect: 用于解析输入表达式的SQL方言。
            copy: 如果为False，则就地修改此表达式实例。
            opts: 用于解析输入表达式的其他选项。

        Returns:
            修改后的查询表达式。
        """
        # 使用_apply_child_list_builder处理多个排序表达式
        # 与limit/offset不同，ORDER BY可以包含多个表达式
        return _apply_child_list_builder(
            *expressions,
            instance=self,
            arg="order",                    # 在查询对象中设置order参数
            append=append,                  # 控制是否追加到现有排序
            copy=copy,
            prefix="ORDER BY",              # SQL关键字前缀
            into=Order,                     # 创建Order类型的表达式
            dialect=dialect,
            **opts,
        )

    @property
    def ctes(self) -> t.List[CTE]:
        """
        返回附加到此查询的所有CTE（公共表表达式）列表。
        
        CTE是WITH子句中定义的临时命名结果集。

        Returns:
            CTE表达式列表。
        """
        # 从查询的args中获取with子句
        with_ = self.args.get("with")
        # 如果存在with子句，返回其表达式列表；否则返回空列表
        return with_.expressions if with_ else []

    @property
    def selects(self) -> t.List[Expression]:
        """
        返回查询的投影表达式列表。
        
        投影是指SELECT子句中指定的列或表达式。

        Returns:
            选择表达式列表。
        """
        # 抽象方法，必须由子类实现
        # 不同的查询类型（SELECT、INSERT等）有不同的投影实现
        raise NotImplementedError("Query objects must implement `selects`")

    @property
    def named_selects(self) -> t.List[str]:
        """
        返回查询投影的输出名称列表。
        
        这些名称是列或表达式的别名，用于标识结果集中的列。

        Returns:
            选择表达式的名称列表。
        """
        # 抽象方法，必须由子类实现
        # 用于获取结果集中每列的名称
        raise NotImplementedError("Query objects must implement `named_selects`")

    def select(
        self: Q,
        *expressions: t.Optional[ExpOrStr],
        append: bool = True,
        dialect: DialectType = None,
        copy: bool = True,
        **opts,
    ) -> Q:
        """
        添加或设置SELECT表达式。
        
        指定查询要选择的列或表达式。

        Example:
            >>> Select().select("x", "y").sql()
            'SELECT x, y'

        Args:
            *expressions: 要解析的SQL代码字符串。
                如果传入Expression实例，将直接使用。
            append: 如果为True，添加到现有表达式；否则重置表达式。
            dialect: 用于解析输入表达式的SQL方言。
            copy: 如果为False，则就地修改此表达式实例。
            opts: 用于解析输入表达式的其他选项。

        Returns:
            修改后的查询表达式。
        """
        # 抽象方法，必须由子类实现
        # 不同的查询类型有不同的select实现
        raise NotImplementedError("Query objects must implement `select`")

    def where(
        self: Q,
        *expressions: t.Optional[ExpOrStr],
        append: bool = True,
        dialect: DialectType = None,
        copy: bool = True,
        **opts,
    ) -> Q:
        """
        添加或设置WHERE表达式。
        
        指定查询的过滤条件，用于限制返回的行。

        Examples:
            >>> Select().select("x").from_("tbl").where("x = 'a' OR x < 'b'").sql()
            "SELECT x FROM tbl WHERE x = 'a' OR x < 'b'"

        Args:
            *expressions: 要解析的SQL代码字符串。
                如果传入Expression实例，将直接使用。
                多个表达式将使用AND操作符组合。
            append: 如果为True，将新表达式与现有表达式进行AND操作。
                否则重置表达式。
            dialect: 用于解析输入表达式的SQL方言。
            copy: 如果为False，则就地修改此表达式实例。
            opts: 用于解析输入表达式的其他选项。

        Returns:
            修改后的表达式。
        """
        # 使用_apply_conjunction_builder处理WHERE条件
        # 特殊处理：如果表达式已经是Where类型，提取其this属性
        # 这避免了嵌套Where表达式的问题
        return _apply_conjunction_builder(
            *[expr.this if isinstance(expr, Where) else expr for expr in expressions],
            instance=self,
            arg="where",                    # 在查询对象中设置where参数
            append=append,                  # 控制是否追加到现有条件
            into=Where,                     # 创建Where类型的表达式
            dialect=dialect,
            copy=copy,
            **opts,
        )

    def with_(
        self: Q,
        alias: ExpOrStr,
        as_: ExpOrStr,
        recursive: t.Optional[bool] = None,
        materialized: t.Optional[bool] = None,
        append: bool = True,
        dialect: DialectType = None,
        copy: bool = True,
        scalar: bool = False,
        **opts,
    ) -> Q:
        """
        添加或设置公共表表达式（CTE）。
        
        CTE允许在查询中定义临时的命名结果集，提高查询的可读性和复用性。

        Example:
            >>> Select().with_("tbl2", as_="SELECT * FROM tbl").select("x").from_("tbl2").sql()
            'WITH tbl2 AS (SELECT * FROM tbl) SELECT x FROM tbl2'

        Args:
            alias: 作为表名的SQL代码字符串。
                如果传入Expression实例，将直接使用。
            as_: 作为表表达式的SQL代码字符串。
                如果传入Expression实例，将直接使用。
            recursive: 设置表达式的RECURSIVE部分，默认为False。
            materialized: 设置表达式的MATERIALIZED部分。
            append: 如果为True，添加到现有表达式；否则重置表达式。
            dialect: 用于解析输入表达式的SQL方言。
            copy: 如果为False，则就地修改此表达式实例。
            scalar: 如果为True，这是标量公共表表达式。
            opts: 用于解析输入表达式的其他选项。

        Returns:
            修改后的表达式。
        """
        # 使用专门的CTE构建器处理复杂的CTE逻辑
        # CTE需要特殊处理，因为它涉及别名和查询的绑定
        return _apply_cte_builder(
            self,
            alias,
            as_,
            recursive=recursive,            # 递归CTE标志
            materialized=materialized,      # 物化CTE标志
            append=append,                  # 控制是否追加到现有CTE
            dialect=dialect,
            copy=copy,
            scalar=scalar,                  # 标量CTE标志
            **opts,
        )

    def union(
        self, *expressions: ExpOrStr, distinct: bool = True, dialect: DialectType = None, **opts
    ) -> Union:
        """
        构建UNION表达式。
        
        将当前查询与其他查询的结果集进行并集操作。

        Example:
            >>> import sqlglot
            >>> sqlglot.parse_one("SELECT * FROM foo").union("SELECT * FROM bla").sql()
            'SELECT * FROM foo UNION SELECT * FROM bla'

        Args:
            expressions: SQL代码字符串。
                如果传入Expression实例，将直接使用。
            distinct: 如果为True，设置DISTINCT标志。
            dialect: 用于解析输入表达式的SQL方言。
            opts: 用于解析输入表达式的其他选项。

        Returns:
            新的Union表达式。
        """
        # 调用全局union函数创建UNION表达式
        # 这种设计保持了API的一致性
        return union(self, *expressions, distinct=distinct, dialect=dialect, **opts)

    def intersect(
        self, *expressions: ExpOrStr, distinct: bool = True, dialect: DialectType = None, **opts
    ) -> Intersect:
        """
        构建INTERSECT表达式。
        
        将当前查询与其他查询的结果集进行交集操作。

        Example:
            >>> import sqlglot
            >>> sqlglot.parse_one("SELECT * FROM foo").intersect("SELECT * FROM bla").sql()
            'SELECT * FROM foo INTERSECT SELECT * FROM bla'

        Args:
            expressions: SQL代码字符串。
                如果传入Expression实例，将直接使用。
            distinct: 如果为True，设置DISTINCT标志。
            dialect: 用于解析输入表达式的SQL方言。
            opts: 用于解析输入表达式的其他选项。

        Returns:
            新的Intersect表达式。
        """
        # 调用全局intersect函数创建INTERSECT表达式
        return intersect(self, *expressions, distinct=distinct, dialect=dialect, **opts)

    def except_(
        self, *expressions: ExpOrStr, distinct: bool = True, dialect: DialectType = None, **opts
    ) -> Except:
        """
        构建EXCEPT表达式。
        
        从当前查询的结果集中排除其他查询的结果集。

        Example:
            >>> import sqlglot
            >>> sqlglot.parse_one("SELECT * FROM foo").except_("SELECT * FROM bla").sql()
            'SELECT * FROM foo EXCEPT SELECT * FROM bla'

        Args:
            expressions: SQL代码字符串。
                如果传入Expression实例，将直接使用。
            distinct: 如果为True，设置DISTINCT标志。
            dialect: 用于解析输入表达式的SQL方言。
            opts: 用于解析输入表达式的其他选项。

        Returns:
            新的Except表达式。
        """
        # 调用全局except_函数创建EXCEPT表达式
        # 注意：方法名使用except_是因为except是Python关键字
        return except_(self, *expressions, distinct=distinct, dialect=dialect, **opts)


class UDTF(DerivedTable):
    """
    用户定义表函数（User-Defined Table Function）表达式类。
    
    UDTF是返回表结果集的用户定义函数，可以在FROM子句中使用。
    继承自DerivedTable，因为UDTF的行为类似于派生表。
    """
    @property
    def selects(self) -> t.List[Expression]:
        """
        获取UDTF的选择表达式列表。
        
        对于UDTF，选择表达式来自其别名定义的列，而不是内部查询。

        Returns:
            选择表达式列表。
        """
        # 从UDTF的参数中获取别名对象
        alias = self.args.get("alias")
        # 如果存在别名且别名定义了列，返回这些列；否则返回空列表
        # UDTF的列结构由其别名中的columns属性定义
        return alias.columns if alias else []


class Cache(Expression):
    """
    缓存表达式类。
    
    表示SQL中的CACHE TABLE语句，用于将表或查询结果缓存到内存中。
    主要用于大数据处理系统如Spark SQL。
    """
    arg_types = {
        "this": True,        # 必需参数：要缓存的表或查询
        "lazy": False,       # 可选参数：是否延迟缓存（lazy evaluation）
        "options": False,    # 可选参数：缓存选项，如存储级别
        "expression": False, # 可选参数：缓存的表达式或查询
    }


class Uncache(Expression):
    """
    取消缓存表达式类。
    
    表示SQL中的UNCACHE TABLE语句，用于从内存中移除表的缓存。
    与Cache语句配对使用，用于释放内存资源。
    """
    arg_types = {
        "this": True,      # 必需参数：要取消缓存的表名
        "exists": False    # 可选参数：IF EXISTS标志，用于避免表不存在时的错误
    }


class Refresh(Expression):
    """
    刷新表达式类。
    
    表示SQL中的REFRESH语句，用于刷新表的元数据或重新加载数据。
    常用于数据湖或外部表的元数据同步。
    """
    # 简单的表达式类，没有特定的参数类型定义
    # 具体的参数处理由父类Expression提供
    pass


class DDL(Expression):
    """
    数据定义语言（Data Definition Language）表达式基类。
    
    表示SQL中的DDL语句，如CREATE、ALTER、DROP等用于定义数据库结构的语句。
    DDL语句可能包含CTE和查询部分（如CTAS - CREATE TABLE AS SELECT）。
    """
    
    @property
    def ctes(self) -> t.List[CTE]:
        """
        返回附加到此DDL语句的所有CTE列表。
        
        某些DDL语句（如CTAS）可能包含WITH子句定义的CTE。

        Returns:
            CTE表达式列表。
        """
        # 从DDL语句的参数中获取with子句
        with_ = self.args.get("with")
        # 如果存在with子句，返回其中的表达式列表；否则返回空列表
        # DDL语句中的CTE通常用于复杂的表创建场景
        return with_.expressions if with_ else []

    @property
    def selects(self) -> t.List[Expression]:
        """
        如果DDL语句包含查询（例如CTAS），返回查询的投影表达式。
        
        某些DDL语句如CREATE TABLE AS SELECT包含查询部分。

        Returns:
            选择表达式列表。
        """
        # 检查DDL语句的expression属性是否为Query类型
        # 如果是查询，返回其选择表达式；否则返回空列表
        # 这主要用于CTAS、CVAS等包含查询的DDL语句
        return self.expression.selects if isinstance(self.expression, Query) else []

    @property
    def named_selects(self) -> t.List[str]:
        """
        如果DDL语句包含查询（例如CTAS），返回查询投影的输出名称。
        
        这些名称通常成为新创建表的列名。

        Returns:
            选择表达式的名称列表。
        """
        # 从DDL语句中的查询获取命名的选择表达式
        # 这些名称在CTAS等语句中用作新表的列名
        return self.expression.named_selects if isinstance(self.expression, Query) else []


# https://docs.teradata.com/r/Enterprise_IntelliFlex_VMware/SQL-Data-Manipulation-Language/Statement-Syntax/LOCKING-Request-Modifier/LOCKING-Request-Modifier-Syntax
class LockingStatement(Expression):
    """
    锁定语句表达式类。
    
    表示Teradata等数据库中的LOCKING语句，用于在事务中锁定表或行。
    主要用于控制并发访问和数据一致性。
    
    参考：Teradata SQL文档中的LOCKING Request Modifier语法。
    """
    arg_types = {
        "this": True,        # 必需参数：锁定的目标（表名等）
        "expression": True   # 必需参数：锁定的具体表达式或条件
    }


class DML(Expression):
    """
    数据操作语言（Data Manipulation Language）表达式基类。
    
    表示SQL中的DML语句，如INSERT、UPDATE、DELETE等用于操作数据的语句。
    提供了DML语句的通用功能，如RETURNING子句。
    """
    
    def returning(
        self,
        expression: ExpOrStr,
        dialect: DialectType = None,
        copy: bool = True,
        **opts,
    ) -> "Self":
        """
        设置RETURNING表达式。并非所有SQL方言都支持此功能。
        
        RETURNING子句允许DML语句返回受影响行的数据，常用于获取自动生成的ID。

        Example:
            >>> delete("tbl").returning("*", dialect="postgres").sql()
            'DELETE FROM tbl RETURNING *'

        Args:
            expression: 要解析的SQL代码字符串。
                如果传入Expression实例，将直接使用。
            dialect: 用于解析输入表达式的SQL方言。
            copy: 如果为False，则就地修改此表达式实例。
            opts: 用于解析输入表达式的其他选项。

        Returns:
            修改后的DML表达式。
        """
        # 使用_apply_builder模式构建RETURNING子句
        # RETURNING子句在PostgreSQL、Oracle等数据库中支持
        # 它允许DML操作返回受影响行的特定列值
        return _apply_builder(
            expression=expression,
            instance=self,
            arg="returning",               # 在DML对象中设置returning参数
            prefix="RETURNING",            # SQL关键字前缀
            dialect=dialect,
            copy=copy,
            into=Returning,                # 创建Returning类型的表达式
            **opts,
        )


class Create(DDL):
    """
    CREATE语句表达式类。
    
    表示SQL中的CREATE语句，用于创建各种数据库对象，如表、视图、索引、函数等。
    继承自DDL，因为CREATE是数据定义语言的核心语句。
    """
    arg_types = {
        "with": False,           # 可选参数：WITH子句，用于CTE或其他选项
        "this": True,            # 必需参数：要创建的对象名称
        "kind": True,            # 必需参数：创建的对象类型（TABLE、VIEW、INDEX等）
        "expression": False,     # 可选参数：创建表达式，如CTAS中的SELECT语句
        "exists": False,         # 可选参数：IF NOT EXISTS标志
        "properties": False,     # 可选参数：对象属性，如表的存储属性
        "replace": False,        # 可选参数：OR REPLACE标志，用于替换现有对象
        "refresh": False,        # 可选参数：REFRESH标志，某些系统支持
        "unique": False,         # 可选参数：UNIQUE标志，用于索引创建
        "indexes": False,        # 可选参数：索引定义列表
        "no_schema_binding": False,  # 可选参数：无模式绑定标志
        "begin": False,          # 可选参数：BEGIN标志，用于函数或过程
        "end": False,            # 可选参数：END标志，用于函数或过程
        "clone": False,          # 可选参数：CLONE子句，用于复制表结构
        "concurrently": False,   # 可选参数：CONCURRENTLY标志，PostgreSQL并发创建
        "clustered": False,      # 可选参数：CLUSTERED标志，SQL Server聚簇索引
    }

    @property
    def kind(self) -> t.Optional[str]:
        """
        获取CREATE语句的对象类型。
        
        返回要创建的对象类型，如TABLE、VIEW、INDEX等。

        Returns:
            对象类型的大写字符串，如果未指定则返回None。
        """
        # 从参数中获取kind值
        kind = self.args.get("kind")
        # 如果存在kind值，转换为大写并返回；否则返回None
        # 转换为大写是为了标准化SQL关键字的表示
        return kind and kind.upper()


class SequenceProperties(Expression):
    """
    序列属性表达式类。
    
    表示CREATE SEQUENCE语句中的序列属性定义。
    序列是数据库中用于生成唯一数值的对象，常用于主键生成。
    """
    arg_types = {
        "increment": False,      # 可选参数：INCREMENT BY值，序列的增量步长
        "minvalue": False,       # 可选参数：MINVALUE，序列的最小值
        "maxvalue": False,       # 可选参数：MAXVALUE，序列的最大值
        "cache": False,          # 可选参数：CACHE值，缓存的序列值数量
        "start": False,          # 可选参数：START WITH值，序列的起始值
        "owned": False,          # 可选参数：OWNED BY子句，序列所属的列
        "options": False,        # 可选参数：其他序列选项
    }


class TruncateTable(Expression):
    """
    TRUNCATE TABLE语句表达式类。
    
    表示SQL中的TRUNCATE TABLE语句，用于快速删除表中的所有数据。
    TRUNCATE比DELETE更快，但不能使用WHERE条件。
    """
    arg_types = {
        "expressions": True,     # 必需参数：要清空的表名列表
        "is_database": False,    # 可选参数：是否操作数据库级别
        "exists": False,         # 可选参数：IF EXISTS标志
        "only": False,           # 可选参数：ONLY标志，仅操作指定表
        "cluster": False,        # 可选参数：CLUSTER相关选项
        "identity": False,       # 可选参数：IDENTITY列处理选项
        "option": False,         # 可选参数：其他TRUNCATE选项
        "partition": False,      # 可选参数：分区相关选项
    }


# https://docs.snowflake.com/en/sql-reference/sql/create-clone
# https://cloud.google.com/bigquery/docs/reference/standard-sql/data-definition-language#create_table_clone_statement
# https://cloud.google.com/bigquery/docs/reference/standard-sql/data-definition-language#create_table_copy
class Clone(Expression):
    """
    克隆表达式类。
    
    表示现代数据仓库系统中的表克隆功能，如Snowflake的CREATE CLONE和BigQuery的CLONE。
    克隆操作可以快速复制表的结构和数据，支持浅克隆和深克隆。
    
    参考：
    - Snowflake CREATE CLONE语法
    - BigQuery CREATE TABLE CLONE语法
    - BigQuery CREATE TABLE COPY语法
    """
    arg_types = {
        "this": True,           # 必需参数：要克隆的源表
        "shallow": False,       # 可选参数：是否浅克隆（仅复制元数据）
        "copy": False,          # 可选参数：是否深拷贝（完整复制数据）
    }


class Describe(Expression):
    """
    DESCRIBE语句表达式类。
    
    表示SQL中的DESCRIBE或DESC语句，用于查看表结构、列信息等元数据。
    不同数据库系统的DESCRIBE语法略有差异。
    """
    arg_types = {
        "this": True,           # 必需参数：要描述的对象名称
        "style": False,         # 可选参数：描述的样式或格式
        "kind": False,          # 可选参数：描述的对象类型
        "expressions": False,   # 可选参数：要描述的特定列或表达式
        "partition": False,     # 可选参数：分区相关信息
        "format": False,        # 可选参数：输出格式选项
    }


# https://duckdb.org/docs/sql/statements/attach.html#attach
class Attach(Expression):
    """
    ATTACH语句表达式类。
    
    表示DuckDB等系统中的ATTACH语句，用于附加外部数据库或数据源。
    允许在单个会话中访问多个数据库。
    
    参考：DuckDB ATTACH语法文档
    """
    arg_types = {
        "this": True,           # 必需参数：要附加的数据库路径或连接字符串
        "exists": False,        # 可选参数：IF NOT EXISTS标志
        "expressions": False,   # 可选参数：附加选项或参数
    }


# https://duckdb.org/docs/sql/statements/attach.html#detach
class Detach(Expression):
    """
    DETACH语句表达式类。
    
    表示DuckDB等系统中的DETACH语句，用于分离已附加的数据库。
    与ATTACH语句配对使用，管理数据库连接。
    
    参考：DuckDB DETACH语法文档
    """
    arg_types = {
        "this": True,           # 必需参数：要分离的数据库名称
        "exists": False,        # 可选参数：IF EXISTS标志
    }


# https://duckdb.org/docs/sql/statements/load_and_install.html
class Install(Expression):
    arg_types = {"this": True, "from": False, "force": False}


# https://duckdb.org/docs/guides/meta/summarize.html
class Summarize(Expression):
    """
    SUMMARIZE语句表达式类。
    
    表示DuckDB中的SUMMARIZE语句，用于快速生成表的统计摘要。
    提供表的基本统计信息，如行数、列类型、数据分布等。
    
    参考：DuckDB SUMMARIZE功能文档
    """
    arg_types = {
        "this": True,           # 必需参数：要汇总的表或查询
        "table": False,         # 可选参数：表格格式输出选项
    }


class Kill(Expression):
    """
    KILL语句表达式类。
    
    表示SQL中的KILL语句，用于终止正在运行的查询或连接。
    主要用于MySQL、SQL Server等数据库系统。
    """
    arg_types = {
        "this": True,           # 必需参数：要终止的进程ID或连接ID
        "kind": False,          # 可选参数：终止类型（CONNECTION、QUERY等）
    }


class Pragma(Expression):
    """
    PRAGMA语句表达式类。
    
    表示SQLite等系统中的PRAGMA语句，用于查询或设置数据库的各种参数和选项。
    PRAGMA是SQLite特有的命令，用于控制数据库行为。
    """
    # 简单的表达式类，没有特定的参数类型定义
    # PRAGMA语句的参数处理较为灵活，由父类Expression处理
    pass


class Declare(Expression):
    """
    DECLARE语句表达式类。
    
    表示SQL中的DECLARE语句，用于声明变量、游标或其他对象。
    主要用于存储过程、函数和批处理脚本中。
    """
    arg_types = {
        "expressions": True,    # 必需参数：声明的项目列表
    }


class DeclareItem(Expression):
    """
    声明项表达式类。
    
    表示DECLARE语句中的单个声明项，如变量声明、游标声明等。
    包含声明的名称、类型和默认值。
    """
    arg_types = {
        "this": True,           # 必需参数：声明项的名称
        "kind": False,          # 可选参数：声明项的类型
        "default": False,       # 可选参数：默认值
    }


class Set(Expression):
    """
    SET语句表达式类。
    
    表示SQL中的SET语句，用于设置会话变量、系统参数或其他配置选项。
    不同数据库系统的SET语法有所差异。
    """
    arg_types = {
        "expressions": False,   # 可选参数：设置项列表
        "unset": False,         # 可选参数：是否为UNSET操作
        "tag": False,           # 可选参数：标签或分类
    }


class Heredoc(Expression):
    """
    Here文档表达式类。
    
    表示某些SQL方言中的Here文档语法，用于定义多行字符串文字。
    常见于支持复杂字符串定义的数据库系统。
    """
    arg_types = {
        "this": True,           # 必需参数：Here文档的内容
        "tag": False,           # 可选参数：Here文档的标识符
    }


class SetItem(Expression):
    """
    SET项表达式类。
    
    表示SET语句中的单个设置项，包含变量名、值和其他选项。
    支持各种SET变体，如全局设置、字符集设置等。
    """
    arg_types = {
        "this": False,          # 可选参数：设置项的名称
        "expressions": False,   # 可选参数：设置值列表
        "kind": False,          # 可选参数：设置项的类型
        "collate": False,       # 可选参数：MySQL SET NAMES语句的排序规则
        "global": False,        # 可选参数：是否为全局设置
    }



class QueryBand(Expression):
    """
    查询带宽表达式类。
    
    表示Teradata等数据库中的QUERYBAND语句，用于设置查询的优先级、资源限制等属性。
    QueryBand可以用于工作负载管理，帮助数据库优化器分配资源。
    
    参考：Teradata QueryBand功能文档
    """
    # this: 查询带宽的设置内容，scope: 作用范围，update: 是否更新现有设置
    arg_types = {"this": True, "scope": False, "update": False}


class Show(Expression):
    """
    SHOW语句表达式类。
    
    表示SQL中的SHOW语句，用于显示数据库、表、列、索引等元数据信息。
    不同数据库系统支持的SHOW语句选项差异很大，这里包含了主要数据库的常见选项。
    
    支持的数据库包括：MySQL、PostgreSQL、SQL Server、ClickHouse等
    """
    arg_types = {
        "this": True,        # 要显示的对象类型（如TABLES、DATABASES等）
        "history": False,    # 是否显示历史信息
        "terse": False,      # 是否使用简洁格式显示
        "target": False,     # 目标对象
        "offset": False,     # 偏移量，用于分页
        "starts_with": False, # 以指定字符串开头的过滤条件
        "limit": False,      # 限制返回的记录数
        "from": False,       # 从指定数据库/表中显示
        "like": False,       # LIKE模式匹配条件
        "where": False,      # WHERE过滤条件
        "db": False,         # 指定数据库名
        "scope": False,      # 作用域限制
        "scope_kind": False, # 作用域类型
        "full": False,       # 是否显示完整信息
        "mutex": False,      # 互斥锁相关信息
        "query": False,      # 查询相关信息
        "channel": False,    # 通道信息（主要用于MySQL复制）
        "global": False,     # 是否显示全局信息
        "log": False,        # 日志相关信息
        "position": False,   # 位置信息
        "types": False,      # 类型信息
        "privileges": False, # 权限信息
        "for_table": False,
        "for_group": False,
        "for_user": False,
        "for_role": False,
        "into_outfile": False,
        "json": False,
    }


class UserDefinedFunction(Expression):
    """
    用户定义函数表达式类。
    
    表示用户自定义的函数调用，包括存储函数、标量函数等。
    用于支持各种数据库中的自定义函数语法。
    """
    # this: 函数名，expressions: 函数参数列表，wrapped: 是否用括号包装
    arg_types = {"this": True, "expressions": False, "wrapped": False}


class CharacterSet(Expression):
    """
    字符集表达式类。
    
    表示SQL中的字符集定义，用于指定文本数据的编码方式。
    主要用于CREATE TABLE、ALTER TABLE等语句中的字符集设置。
    """
    # this: 字符集名称，default: 是否为默认字符集
    arg_types = {"this": True, "default": False}


class RecursiveWithSearch(Expression):
    """
    递归WITH搜索表达式类。
    
    表示SQL标准中的递归CTE（公共表表达式）的SEARCH子句。
    用于控制递归查询的搜索顺序（深度优先或广度优先）。
    
    参考：SQL:1999标准中的递归查询语法
    """
    # kind: 搜索类型（DEPTH FIRST或BREADTH FIRST），this: 搜索列，expression: 搜索表达式，using: 使用的列
    arg_types = {"kind": True, "this": True, "expression": True, "using": False}


class With(Expression):
    """
    WITH表达式类。
    
    表示SQL中的WITH子句（公共表表达式/CTE），用于定义临时的命名查询结果集。
    支持递归和非递归CTE，是现代SQL的重要特性。
    """
    # expressions: CTE表达式列表，recursive: 是否为递归CTE，search: 搜索子句
    arg_types = {"expressions": True, "recursive": False, "search": False}

    @property
    def recursive(self) -> bool:
        """
        检查是否为递归CTE。
        
        递归CTE可以引用自身，用于处理树形结构、图遍历等场景。
        通过检查recursive参数来判断是否需要生成RECURSIVE关键字。
        """
        return bool(self.args.get("recursive"))


class WithinGroup(Expression):
    """
    WITHIN GROUP表达式类。
    
    表示SQL中的WITHIN GROUP子句，主要用于有序集聚合函数。
    例如：PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY salary)
    
    参考：SQL标准中的有序集函数语法
    """
    # this: 聚合函数，expression: ORDER BY表达式
    arg_types = {"this": True, "expression": False}


# ClickHouse支持标量CTE功能
# 参考：https://clickhouse.com/docs/en/sql-reference/statements/select/with
class CTE(DerivedTable):
    """
    公共表表达式（CTE）类。
    
    继承自DerivedTable，表示WITH子句中定义的临时表。
    CTE可以被同一查询中的其他部分引用，提高查询的可读性和可维护性。
    
    ClickHouse等系统还支持标量CTE，即返回单个值的CTE。
    """
    arg_types = {
        "this": True,         # CTE的查询表达式
        "alias": True,        # CTE的别名（必需）
        "scalar": False,      # 是否为标量CTE（ClickHouse特性）
        "materialized": False, # 是否物化CTE（某些数据库支持）
        "key_expressions": False,
    }


class ProjectionDef(Expression):
    """
    投影定义表达式类。
    
    表示CREATE TABLE语句中的投影定义，主要用于列式数据库。
    投影是预计算的查询结果，可以加速特定查询模式。
    
    主要用于ClickHouse等支持投影的数据库系统。
    """
    # this: 投影名称，expression: 投影的查询表达式
    arg_types = {"this": True, "expression": True}


class TableAlias(Expression):
    """
    表别名表达式类。
    
    表示表的别名定义，包括表别名本身和可选的列别名列表。
    例如：FROM users AS u(id, name) 中的 "u(id, name)" 部分。
    """
    # this: 表别名，columns: 列别名列表（可选）
    arg_types = {"this": False, "columns": False}

    @property
    def columns(self):
        """
        获取列别名列表。
        
        返回表别名中定义的列别名，如果没有定义则返回空列表。
        用于支持 AS alias(col1, col2, ...) 这种语法。
        """
        return self.args.get("columns") or []


class BitString(Condition):
    """
    位字符串表达式类。
    
    表示SQL中的位字符串字面量，如 B'101010'。
    位字符串是由0和1组成的二进制数据表示方式。
    
    主要用于需要精确位操作的场景。
    """
    pass


class HexString(Condition):
    """
    十六进制字符串表达式类。
    
    表示SQL中的十六进制字符串字面量，如 X'41424320' 或 0x41424320。
    用于表示二进制数据的十六进制形式。
    """
    # this: 十六进制字符串内容，is_integer: 是否为整数形式（如0x123）
    arg_types = {"this": True, "is_integer": False}


class ByteString(Condition):
    """
    字节字符串表达式类。
    
    表示原始字节数据的字符串表示，主要用于存储二进制数据。
    不同数据库对字节字符串的语法支持略有差异。
    """
    pass


class RawString(Condition):
    """
    原始字符串表达式类。
    
    表示原始字符串字面量，通常不进行转义处理。
    用于存储包含特殊字符的字符串数据。
    """
    pass


class UnicodeString(Condition):
    """
    Unicode字符串表达式类。
    
    表示Unicode字符串字面量，如 U&'Hello\0041' 或 N'Hello'。
    支持Unicode转义序列，用于处理国际化文本数据。
    """
    # this: Unicode字符串内容，escape: 转义字符（用于自定义转义序列）
    arg_types = {"this": True, "escape": False}


class Column(Condition):
    """
    列表达式类。
    
    表示SQL中的列引用，支持完全限定的列名格式：catalog.database.table.column。
    这是SQL查询中最基础的表达式之一，用于引用表中的具体字段。
    
    支持的格式：
    - column_name
    - table.column_name  
    - database.table.column_name
    - catalog.database.table.column_name
    """
    # this: 列名，table: 表名，db: 数据库名，catalog: 目录名，join_mark: 连接标记
    arg_types = {"this": True, "table": False, "db": False, "catalog": False, "join_mark": False}

    @property
    def table(self) -> str:
        """
        获取列所属的表名。
        
        从参数中提取表名部分，用于构建完整的列引用路径。
        """
        return self.text("table")

    @property
    def db(self) -> str:
        """
        获取列所属的数据库名。
        
        从参数中提取数据库名部分，支持跨数据库的列引用。
        """
        return self.text("db")

    @property
    def catalog(self) -> str:
        """
        获取列所属的目录名。
        
        从参数中提取目录名部分，支持跨目录的列引用（主要用于企业级数据库系统）。
        """
        return self.text("catalog")

    @property
    def output_name(self) -> str:
        """
        获取列的输出名称。
        
        返回列在查询结果中显示的名称，通常就是列名本身。
        """
        return self.name

    @property
    def parts(self) -> t.List[Identifier]:
        """
        按顺序返回列的各个组成部分：catalog, db, table, name。
        
        这个方法构建完整的列标识符层次结构，用于：
        1. 生成完全限定的列名
        2. 进行列名解析和验证
        3. 支持跨数据库/目录的列引用
        """
        return [
            # 将每个存在的部分转换为Identifier对象
            # 按照SQL标准的层次顺序：catalog -> db -> table -> column
            t.cast(Identifier, self.args[part])
            for part in ("catalog", "db", "table", "this")
            if self.args.get(part)  # 只包含实际存在的部分
        ]

    def to_dot(self, include_dots: bool = True) -> Dot | Identifier:
        """
        将列转换为点表达式。
        
        这个方法处理列名的层次结构表示，将多部分的列名转换为嵌套的Dot表达式。
        例如：database.table.column -> Dot(Dot(database, table), column)
        
        Args:
            include_dots: 是否包含父级的Dot表达式，用于处理嵌套的点表达式
        """
        parts = self.parts
        parent = self.parent

        if include_dots:
            # 向上遍历父节点，收集所有的Dot表达式部分
            # 这样可以正确处理复杂的嵌套结构
            while isinstance(parent, Dot):
                parts.append(parent.expression)
                parent = parent.parent

        # 如果有多个部分，构建Dot表达式；否则返回单个标识符
        return Dot.build(deepcopy(parts)) if len(parts) > 1 else parts[0]


class ColumnPosition(Expression):
    """
    列位置表达式类。
    
    表示ALTER TABLE语句中列的位置信息，如FIRST、AFTER column_name等。
    主要用于MySQL等支持列位置操作的数据库系统。
    
    例如：ALTER TABLE t ADD COLUMN c INT AFTER existing_column
    """
    # this: 列标识符（可选），position: 位置关键字（如FIRST、AFTER）
    arg_types = {"this": False, "position": True}


class ColumnDef(Expression):
    """
    列定义表达式类。
    
    表示CREATE TABLE或ALTER TABLE语句中的列定义，包含列的所有属性：
    名称、数据类型、约束、默认值等。这是DDL语句中的核心组件。
    
    例如：id INT PRIMARY KEY AUTO_INCREMENT DEFAULT 0
    """
    arg_types = {
        "this": True,        # 列名（必需）
        "kind": False,       # 数据类型（如INT、VARCHAR等）
        "constraints": False, # 列约束列表（如NOT NULL、PRIMARY KEY等）
        "exists": False,     # IF EXISTS子句
        "position": False,   # 列位置信息（MySQL等支持）
        "default": False,    # 默认值表达式
        "output": False,     # 输出相关属性
    }

    @property
    def constraints(self) -> t.List[ColumnConstraint]:
        """
        获取列的约束列表。
        
        返回应用于此列的所有约束，如NOT NULL、PRIMARY KEY、CHECK等。
        如果没有约束则返回空列表，确保调用方可以安全地迭代。
        """
        return self.args.get("constraints") or []

    @property
    def kind(self) -> t.Optional[DataType]:
        """
        获取列的数据类型。
        
        返回列的数据类型定义，如INT、VARCHAR(255)、DECIMAL(10,2)等。
        数据类型是列定义的核心属性，用于确定存储格式和操作规则。
        """
        return self.args.get("kind")


class AlterColumn(Expression):
    """
    修改列表达式类。
    
    表示ALTER TABLE语句中的列修改操作，支持各种列级别的变更：
    - 修改数据类型
    - 添加/删除约束
    - 修改默认值
    - 添加/修改注释
    
    不同数据库系统的ALTER COLUMN语法略有差异，这个类统一了各种变更操作。
    """
    arg_types = {
        "this": True,        # 要修改的列名
        "dtype": False,      # 新的数据类型
        "collate": False,    # 排序规则（主要用于字符串类型）
        "using": False,      # USING子句（PostgreSQL等支持，用于类型转换）
        "default": False,    # 新的默认值
        "drop": False,       # 要删除的属性（如DROP DEFAULT、DROP NOT NULL）
        "comment": False,    # 列注释
        "allow_null": False, # 是否允许NULL值
        "visible": False,    # 列可见性（Oracle等支持隐藏列）
        "rename_to": False,
    }


# https://dev.mysql.com/doc/refman/8.0/en/invisible-indexes.html
class AlterIndex(Expression):
    """
    修改索引表达式类。
    
    表示ALTER INDEX语句，用于修改索引的属性，如可见性等。
    主要用于支持Oracle、MySQL等数据库的索引管理功能。
    """
    # this: 索引名称，visible: 索引可见性（True表示可见，False表示不可见）
    arg_types = {"this": True, "visible": True}


# 参考：https://docs.aws.amazon.com/redshift/latest/dg/r_ALTER_TABLE.html
class AlterDistStyle(Expression):
    """
    修改分布样式表达式类。
    
    表示Amazon Redshift中的ALTER TABLE DISTSTYLE语句，用于修改表的数据分布策略。
    分布样式影响数据在集群节点间的分布方式，是Redshift性能优化的关键。
    
    支持的分布样式：
    - EVEN: 数据均匀分布
    - KEY: 按指定列分布
    - ALL: 每个节点都有完整数据副本
    """
    pass


class AlterSortKey(Expression):
    """
    修改排序键表达式类。
    
    表示Amazon Redshift中的ALTER TABLE SORTKEY语句，用于修改表的排序键。
    排序键决定了数据在磁盘上的物理排序方式，影响查询性能。
    """
    arg_types = {
        "this": False,         # 排序键名称（可选，用于单列排序键）
        "expressions": False,  # 排序键列表（用于复合排序键）
        "compound": False,     # 是否为复合排序键（vs 交错排序键）
    }


class AlterSet(Expression):
    """
    ALTER SET表达式类。
    
    表示各种ALTER TABLE SET语句，用于设置表的各种属性和选项。
    不同数据库系统支持的SET选项差异很大，这里统一了常见选项。
    """
    arg_types = {
        "expressions": False,    # 设置表达式列表
        "option": False,         # 设置选项
        "tablespace": False,     # 表空间设置
        "access_method": False,  # 访问方法（PostgreSQL等）
        "file_format": False,    # 文件格式（用于外部表）
        "copy_options": False,   # 复制选项
        "tag": False,           # 标签设置（Snowflake等）
        "location": False,      # 位置设置（用于外部表）
        "serde": False,         # 序列化/反序列化设置（Hive等）
    }


class RenameColumn(Expression):
    """
    重命名列表达式类。
    
    表示ALTER TABLE RENAME COLUMN语句，用于重命名表中的列。
    支持条件重命名（IF EXISTS），避免重命名不存在的列时出错。
    """
    # this: 原列名，to: 新列名，exists: 是否使用IF EXISTS条件
    arg_types = {"this": True, "to": True, "exists": False}


class AlterRename(Expression):
    """
    重命名表达式类。
    
    表示ALTER TABLE RENAME语句的通用形式，用于重命名表或其他数据库对象。
    """
    pass

class AlterToGroup(Expression):
    """
    ALTER TABLE TO GROUP表达式类。
    
    表示ALTER TABLE table_name TO GROUP groupname语句，
    用于将表重新分配到指定的节点组。
    """
    arg_types = {
        "this": True,    # 目标组名（必需）
    }

class AlterToNode(Expression):
    """
    ALTER TABLE TO NODE表达式类。
    
    表示ALTER TABLE table_name TO NODE (nodename [, ...])语句，
    用于将表重新分配到指定的节点。
    """
    arg_types = {
        "expressions": True,  # 节点名列表（必需）
    }

class SwapTable(Expression):
    """
    交换表表达式类。
    
    表示某些数据库系统中的表交换操作，如Snowflake的SWAP WITH语句。
    用于原子性地交换两个表的内容和元数据。
    """
    pass


class Comment(Expression):
    """
    注释表达式类。
    
    表示COMMENT ON语句，用于为数据库对象添加或修改注释。
    注释是重要的文档功能，帮助开发者理解数据结构的业务含义。
    """
    arg_types = {
        "this": True,         # 注释内容
        "kind": True,         # 对象类型（TABLE、COLUMN、INDEX等）
        "expression": True,   # 目标对象表达式
        "exists": False,      # IF EXISTS条件
        "materialized": False, # 是否为物化视图注释
    }


class Comprehension(Expression):
    """
    列表推导式表达式类。
    
    表示类似Python列表推导式的SQL语法，主要用于支持某些现代SQL引擎的高级语法。
    例如：[x * 2 for x in array_col if x > 0]
    """
    arg_types = {
        "this": True,        # 输出表达式
        "expression": True,  # 源表达式/数组
        "position": False,
        "iterator": True,    # 迭代变量
        "condition": False,  # 过滤条件（可选）
    }


# 参考：https://clickhouse.com/docs/en/engines/table-engines/mergetree-family/mergetree#mergetree-table-ttl
class MergeTreeTTLAction(Expression):
    """
    ClickHouse MergeTree TTL动作表达式类。
    
    表示ClickHouse中MergeTree表引擎的TTL（生存时间）动作配置。
    TTL用于自动管理数据的生命周期，可以删除过期数据或将其迁移到不同存储。
    """
    arg_types = {
        "this": True,        # TTL表达式
        "delete": False,     # 删除动作
        "recompress": False, # 重新压缩动作
        "to_disk": False,    # 迁移到指定磁盘
        "to_volume": False,  # 迁移到指定卷
    }


# 参考：https://clickhouse.com/docs/en/engines/table-engines/mergetree-family/mergetree#mergetree-table-ttl
class MergeTreeTTL(Expression):
    """
    ClickHouse MergeTree TTL表达式类。
    
    表示ClickHouse中完整的TTL配置，包括TTL表达式和相关的分组、聚合操作。
    支持行级TTL和列级TTL，用于数据生命周期管理。
    """
    arg_types = {
        "expressions": True,  # TTL表达式列表
        "where": False,      # 条件表达式
        "group": False,      # 分组表达式
        "aggregates": False, # 聚合表达式列表
    }


# 参考：https://dev.mysql.com/doc/refman/8.0/en/create-table.html
class IndexConstraintOption(Expression):
    """
    索引约束选项表达式类。
    
    表示MySQL等数据库中创建索引时的各种选项配置。
    这些选项用于优化索引性能和存储特性。
    """
    arg_types = {
        "key_block_size": False,       # 键块大小（影响索引页大小）
        "using": False,                # 索引算法（BTREE、HASH等）
        "parser": False,               # 全文索引解析器
        "comment": False,              # 索引注释
        "visible": False,              # 索引可见性
        "engine_attr": False,          # 存储引擎属性
        "secondary_engine_attr": False, # 辅助存储引擎属性
    }


class ColumnConstraint(Expression):
    """
    列约束表达式类。
    
    表示应用于列的约束，是列定义中的重要组成部分。
    列约束确保数据的完整性和一致性。
    """
    # this: 约束名称（可选），kind: 约束类型（必需）
    arg_types = {"this": False, "kind": True}

    @property
    def kind(self) -> ColumnConstraintKind:
        """
        获取约束类型。
        
        返回具体的约束类型对象，如NotNullColumnConstraint、PrimaryKeyColumnConstraint等。
        约束类型决定了数据库如何验证和处理该列的数据。
        """
        return self.args["kind"]


class ColumnConstraintKind(Expression):
    """
    列约束类型基类。
    
    所有具体列约束类型的基类，提供统一的接口。
    不同的约束类型实现不同的数据完整性规则。
    """
    pass


class AutoIncrementColumnConstraint(ColumnConstraintKind):
    """
    自增列约束类。
    
    表示AUTO_INCREMENT约束，用于自动生成递增的数值。
    主要用于MySQL、SQL Server等数据库的主键列。
    """
    pass


class PeriodForSystemTimeConstraint(ColumnConstraintKind):
    """
    系统时间周期约束类。
    
    表示SQL Server等数据库中的PERIOD FOR SYSTEM_TIME约束，
    用于时态表（temporal table）的系统版本控制。
    """
    # this: 开始时间列，expression: 结束时间列
    arg_types = {"this": True, "expression": True}


class CaseSpecificColumnConstraint(ColumnConstraintKind):
    """
    大小写敏感约束类。
    
    表示某些数据库中的大小写敏感性控制约束。
    用于指定字符串比较时是否区分大小写。
    """
    # not_: True表示CASE INSENSITIVE，False表示CASE SENSITIVE
    arg_types = {"not_": True}


class CharacterSetColumnConstraint(ColumnConstraintKind):
    """
    字符集约束类。
    
    表示列级别的字符集设置，指定该列使用的字符编码。
    主要用于MySQL等支持多字符集的数据库。
    """
    # this: 字符集名称
    arg_types = {"this": True}


class CheckColumnConstraint(ColumnConstraintKind):
    """
    检查约束类。
    
    表示CHECK约束，用于限制列值必须满足指定的条件。
    是实现复杂业务规则的重要机制。
    """
    # this: 检查条件表达式，enforced: 是否强制执行（某些数据库支持延迟检查）
    arg_types = {"this": True, "enforced": False}


class ClusteredColumnConstraint(ColumnConstraintKind):
    """
    聚集约束类。
    
    表示聚集索引约束，主要用于SQL Server等数据库。
    聚集索引决定了数据在磁盘上的物理存储顺序。
    """
    pass


class CollateColumnConstraint(ColumnConstraintKind):
    """
    排序规则约束类。
    
    表示列级别的排序规则设置，影响字符串的比较和排序行为。
    用于控制语言特定的排序规则。
    """
    pass


class CommentColumnConstraint(ColumnConstraintKind):
    """
    注释约束类。
    
    表示列级别的注释，用于文档化列的业务含义。
    虽然不影响数据存储，但对维护和理解数据结构很重要。
    """
    pass


class CompressColumnConstraint(ColumnConstraintKind):
    """
    压缩约束类。
    
    表示列级别的压缩设置，用于节省存储空间。
    主要用于支持列级压缩的数据库系统。
    """
    # this: 压缩算法或选项（可选）
    arg_types = {"this": False}


class DateFormatColumnConstraint(ColumnConstraintKind):
    """
    日期格式约束类。
    
    表示日期/时间列的格式约束，指定日期数据的显示和解析格式。
    主要用于Teradata等数据库的日期列定义。
    """
    # this: 日期格式字符串
    arg_types = {"this": True}


class DefaultColumnConstraint(ColumnConstraintKind):
    """
    默认值约束类。
    
    表示列的默认值设置，当插入数据时未指定该列值时使用。
    是数据完整性和用户体验的重要功能。
    """
    pass


class EncodeColumnConstraint(ColumnConstraintKind):
    """
    编码约束类。
    
    表示列的编码方式设置，主要用于Amazon Redshift等列式数据库。
    编码方式影响数据压缩和查询性能。
    """
    pass


# 参考：https://www.postgresql.org/docs/current/sql-createtable.html#SQL-CREATETABLE-EXCLUDE
class ExcludeColumnConstraint(ColumnConstraintKind):
    """
    排除约束类。
    
    表示PostgreSQL中的EXCLUDE约束，用于防止重叠或冲突的数据。
    常用于时间范围、几何对象等需要避免重叠的场景。
    """
    pass


class EphemeralColumnConstraint(ColumnConstraintKind):
    """
    临时列约束类。
    
    表示临时或短暂的列属性，数据不会持久化存储。
    主要用于某些专门的数据处理场景。
    """
    # this: 临时属性配置（可选）
    arg_types = {"this": False}


class WithOperator(Expression):
    """
    WITH操作符表达式类。
    
    表示PostgreSQL等数据库中的WITH操作符语法，
    用于在约束中指定操作符类或其他选项。
    """
    # this: 操作数，op: 操作符
    arg_types = {"this": True, "op": True}


class GeneratedAsIdentityColumnConstraint(ColumnConstraintKind):
    """
    生成标识列约束类。
    
    表示SQL标准中的GENERATED AS IDENTITY约束，用于自动生成唯一标识值。
    类似于AUTO_INCREMENT，但提供更多控制选项。
    """
    # this: True表示ALWAYS，False表示BY DEFAULT
    # 这决定了是否允许用户显式插入值
    arg_types = {
        "this": False,        # 生成策略（ALWAYS vs BY DEFAULT）
        "expression": False,  # 生成表达式
        "on_null": False,     # 空值处理策略
        "start": False,       # 起始值
        "increment": False,   # 增量值
        "minvalue": False,    # 最小值
        "maxvalue": False,    # 最大值
        "cycle": False,       # 是否循环
        "order": False,       # 是否保证顺序
    }


class GeneratedAsRowColumnConstraint(ColumnConstraintKind):
    """
    行生成约束类。
    
    表示SQL Server等数据库中的行版本生成约束，
    用于自动跟踪行的版本或变更。
    """
    # start: 起始版本值，hidden: 是否为隐藏列
    arg_types = {"start": False, "hidden": False}


# 参考：https://dev.mysql.com/doc/refman/8.0/en/create-table.html
# 参考：https://github.com/ClickHouse/ClickHouse/blob/master/src/Parsers/ParserCreateQuery.h#L646
class IndexColumnConstraint(ColumnConstraintKind):
    """
    索引列约束类。
    
    表示列级别的索引定义，支持MySQL和ClickHouse等数据库的语法。
    允许在列定义时直接创建索引。
    """
    arg_types = {
        "this": False,        # 索引名称（可选）
        "expressions": False, # 索引表达式列表
        "kind": False,        # 索引类型（PRIMARY、UNIQUE等）
        "index_type": False,  # 索引算法（BTREE、HASH等）
        "options": False,     # 索引选项
        "expression": False,  # ClickHouse索引表达式
        "granularity": False, # ClickHouse索引粒度
    }


class InlineLengthColumnConstraint(ColumnConstraintKind):
    """
    内联长度约束类。
    
    表示某些数据库中的内联长度限制，
    用于控制变长数据的内联存储阈值。
    """
    pass


class NonClusteredColumnConstraint(ColumnConstraintKind):
    """
    非聚集约束类。
    
    表示非聚集索引约束，主要用于SQL Server等数据库。
    与聚集索引相对，不改变数据的物理存储顺序。
    """
    pass


class NotForReplicationColumnConstraint(ColumnConstraintKind):
    """
    非复制约束类。
    
    表示SQL Server中的NOT FOR REPLICATION约束，
    指示该约束在复制过程中不会被检查。
    """
    arg_types = {}


# 参考：https://docs.snowflake.com/en/sql-reference/sql/create-table
class MaskingPolicyColumnConstraint(ColumnConstraintKind):
    """
    数据掩码策略约束类。
    
    表示Snowflake等数据库中的数据掩码策略，
    用于保护敏感数据，根据用户权限显示不同的数据视图。
    """
    # this: 掩码策略名称，expressions: 策略参数
    arg_types = {"this": True, "expressions": False}


class NotNullColumnConstraint(ColumnConstraintKind):
    """
    非空约束类。
    
    表示NOT NULL约束，是最常用的数据完整性约束之一。
    确保列值不能为空。
    """
    # allow_null: 是否允许NULL（用于某些特殊情况）
    arg_types = {"allow_null": False}


# 参考：https://dev.mysql.com/doc/refman/5.7/en/timestamp-initialization.html
class OnUpdateColumnConstraint(ColumnConstraintKind):
    """
    更新时约束类。
    
    表示MySQL中的ON UPDATE约束，主要用于TIMESTAMP列。
    当行被更新时自动更新该列的值。
    """
    pass


class PrimaryKeyColumnConstraint(ColumnConstraintKind):
    """
    主键约束类。
    
    表示列级别的主键约束，确保列值唯一且非空。
    主键是关系数据库中最重要的约束之一。
    """
    # desc: 是否降序，options: 主键选项
    arg_types = {"desc": False, "options": False}


class TitleColumnConstraint(ColumnConstraintKind):
    """
    标题约束类。
    
    表示某些数据库中的列标题或显示名称设置。
    用于改变列在查询结果中的显示名称。
    """
    pass


class UniqueColumnConstraint(ColumnConstraintKind):
    """
    唯一约束类。
    
    表示UNIQUE约束，确保列值在表中唯一。
    与主键类似，但允许NULL值且一个表可以有多个唯一约束。
    """
    arg_types = {
        "this": False,        # 约束名称（可选）
        "index_type": False,  # 索引类型
        "on_conflict": False, # 冲突处理策略
        "nulls": False,       # NULL值处理方式
        "options": False,     # 约束选项
    }


class UppercaseColumnConstraint(ColumnConstraintKind):
    """
    大写约束类。
    
    表示某些数据库中的自动大写转换约束。
    插入或更新时自动将字符串值转换为大写。
    """
    arg_types: t.Dict[str, t.Any] = {}


# 参考：https://docs.risingwave.com/processing/watermarks#syntax
class WatermarkColumnConstraint(Expression):
    """
    水印约束类。
    
    表示RisingWave等流处理数据库中的水印约束。
    用于流数据处理中的时间窗口和延迟容忍设置。
    """
    # this: 时间列，expression: 水印表达式
    arg_types = {"this": True, "expression": True}


class PathColumnConstraint(ColumnConstraintKind):
    """
    路径约束类。
    
    表示某些数据库中的路径或层次结构约束。
    用于树形数据结构的路径管理。
    """
    pass


# 参考：https://docs.snowflake.com/en/sql-reference/sql/create-table
class ProjectionPolicyColumnConstraint(ColumnConstraintKind):
    """
    投影策略约束类。
    
    表示Snowflake等数据库中的投影策略约束。
    用于控制列在不同查询中的投影和访问权限。
    """
    pass


# 计算列表达式
# 参考：https://learn.microsoft.com/en-us/sql/t-sql/statements/create-table-transact-sql?view=sql-server-ver16
class ComputedColumnConstraint(ColumnConstraintKind):
    """
    计算列约束类。
    
    表示SQL Server等数据库中的计算列定义。
    计算列的值由表达式计算得出，不直接存储数据。
    """
    arg_types = {
        "this": True,        # 计算表达式
        "persisted": False,  # 是否持久化计算结果
        "not_null": False,   # 是否非空
        "data_type": False
    }



class Constraint(Expression):
    """
    约束表达式基类。
    
    表示表级别的约束，如外键、检查约束等。
    与列约束不同，表约束可以涉及多个列。
    """
    # this: 约束名称，expressions: 约束定义表达式
    arg_types = {"this": True, "expressions": True}


class Delete(DML):
    """
    DELETE语句表达式类。
    
    继承自DML（数据操作语言），表示SQL中的DELETE语句。
    支持各种DELETE语法变体，包括多表删除、条件删除、限制删除等。
    """
    arg_types = {
        "with": False,       # WITH子句（CTE）
        "this": False,       # 目标表
        "using": False,      # USING子句（PostgreSQL等支持）
        "where": False,      # WHERE条件
        "returning": False,  # RETURNING子句（返回被删除的行）
        "limit": False,      # LIMIT子句（MySQL等支持）
        "tables": False,     # 多表语法（MySQL多表删除）
        "cluster": False,    # 集群相关选项（ClickHouse）
    }

    def delete(
        self,
        table: ExpOrStr,
        dialect: DialectType = None,
        copy: bool = True,
        **opts,
    ) -> Delete:
        """
        创建DELETE表达式或替换现有DELETE表达式的目标表。
        
        这个方法实现了流式API，允许链式调用来构建复杂的DELETE语句。
        通过指定目标表来设置DELETE操作的对象。

        示例:
            >>> delete("tbl").sql()
            'DELETE FROM tbl'

        参数:
            table: 要删除数据的目标表名或表表达式
            dialect: 用于解析输入表达式的SQL方言
            copy: 如果为False，则就地修改当前表达式实例
            opts: 解析输入表达式的其他选项

        返回:
            Delete: 修改后的DELETE表达式对象
        """
        # 使用_apply_builder工具函数来设置目标表
        # 将table表达式转换为Table对象并赋值给this参数
        return _apply_builder(
            expression=table,    # 输入的表表达式
            instance=self,       # 当前DELETE实例
            arg="this",         # 目标参数名
            dialect=dialect,    # SQL方言
            into=Table,         # 目标类型为Table
            copy=copy,          # 是否复制
            **opts,
        )

    def where(
        self,
        *expressions: t.Optional[ExpOrStr],
        append: bool = True,
        dialect: DialectType = None,
        copy: bool = True,
        **opts,
    ) -> Delete:
        """
        添加或设置WHERE条件表达式。
        
        这个方法支持构建复杂的WHERE条件，多个条件会用AND连接。
        通过append参数控制是追加还是替换现有条件。

        示例:
            >>> delete("tbl").where("x = 'a' OR x < 'b'").sql()
            "DELETE FROM tbl WHERE x = 'a' OR x < 'b'"

        参数:
            *expressions: 要解析的SQL代码字符串。
                如果传入Expression实例，将直接使用。
                多个表达式会用AND操作符组合。
            append: 如果为True，将新表达式与现有表达式用AND连接。
                否则，重置现有表达式。
            dialect: 用于解析输入表达式的SQL方言
            copy: 如果为False，则就地修改当前表达式实例
            opts: 解析输入表达式的其他选项

        返回:
            Delete: 修改后的DELETE表达式对象
        """
        # 使用_apply_conjunction_builder来处理WHERE条件的逻辑组合
        # 多个条件会自动用AND连接，符合SQL语义
        return _apply_conjunction_builder(
            *expressions,
            instance=self,      # 当前DELETE实例
            arg="where",        # 目标参数名为where
            append=append,      # 是否追加到现有条件
            into=Where,         # 目标类型为Where表达式
            dialect=dialect,
            copy=copy,
            **opts,
        )


class Drop(Expression):
    """
    DROP语句表达式类。
    
    表示SQL中的DROP语句，用于删除数据库对象（表、视图、索引、函数等）。
    支持各种DROP选项，如CASCADE、IF EXISTS、CONCURRENTLY等。
    """
    arg_types = {
        "this": False,        # 要删除的对象名称
        "kind": False,        # 对象类型（TABLE、VIEW、INDEX等）
        "expressions": False, # 对象表达式列表（支持删除多个对象）
        "exists": False,      # IF EXISTS选项，避免删除不存在对象时出错
        "temporary": False,   # 是否删除临时对象
        "external": False,    # 是否删除外部表
        "materialized": False, # 是否为物化视图（PostgreSQL等）
        "cascade": False,     # CASCADE选项，级联删除依赖对象
        "constraints": False, # 约束相关选项
        "purge": False,       # PURGE选项（Oracle等，永久删除）
        "cluster": False,     # 集群相关选项
        "concurrently": False, # CONCURRENTLY选项（PostgreSQL，并发删除索引）
    }

    @property
    def kind(self) -> t.Optional[str]:
        """
        获取DROP操作的对象类型。
        
        返回要删除的数据库对象类型，如TABLE、VIEW、INDEX等。
        自动转换为大写形式，符合SQL标准惯例。
        """
        kind = self.args.get("kind")
        # 如果kind存在，转换为大写；否则返回None
        # 大写转换确保SQL关键字的标准化表示
        return kind and kind.upper()


# 参考：https://cloud.google.com/bigquery/docs/reference/standard-sql/export-statements
class Export(Expression):
    """
    EXPORT语句表达式类。
    
    表示BigQuery等数据库中的EXPORT语句，用于将查询结果导出到外部存储。
    主要用于数据导出和ETL流程。
    """
    # this: 导出的查询或数据，connection: 连接配置，options: 导出选项
    arg_types = {"this": True, "connection": False, "options": True}


class Filter(Expression):
    """
    FILTER表达式类。
    
    表示聚合函数中的FILTER子句，用于条件聚合。
    例如：COUNT(*) FILTER (WHERE condition)
    这是SQL标准中的高级聚合功能。
    """
    # this: 聚合函数，expression: 过滤条件
    arg_types = {"this": True, "expression": True}


class Check(Expression):
    """
    CHECK表达式类。
    
    表示各种检查操作的通用表达式，可用于数据验证、约束检查等。
    具体含义依赖于使用上下文。
    """
    pass


class Changes(Expression):
    """
    CHANGES表达式类。
    
    表示某些数据库中的变更跟踪功能，用于查询数据变更历史。
    主要用于审计和变更监控场景。
    """
    arg_types = {
        "information": True,  # 变更信息类型
        "at_before": False,   # 时间点或版本（查询某个时间点之前的变更）
        "end": False,         # 结束时间点或版本
    }


# 参考：https://docs.snowflake.com/en/sql-reference/constructs/connect-by
class Connect(Expression):
    """
    CONNECT BY表达式类。
    
    表示Oracle、Snowflake等数据库中的CONNECT BY语句，用于层次查询。
    主要用于处理树形结构数据，如组织架构、分类体系等。
    """
    arg_types = {
        "start": False,    # START WITH子句，指定层次查询的起始点
        "connect": True,   # CONNECT BY子句，定义父子关系（必需）
        "nocycle": False,  # NOCYCLE选项，防止循环引用
    }


class CopyParameter(Expression):
    """
    COPY参数表达式类。
    
    表示COPY语句中的单个参数配置，用于控制数据复制的行为。
    不同数据库的COPY语句参数格式略有差异。
    """
    arg_types = {
        "this": True,        # 参数名称
        "expression": False, # 参数值表达式
        "expressions": False, # 参数值列表（用于复杂参数）
    }


class Copy(DML):
    """
    COPY语句表达式类。
    
    继承自DML，表示数据复制语句，主要用于：
    - 从文件批量导入数据
    - 将查询结果导出到文件
    - 在不同存储系统间复制数据
    
    支持多种数据格式和认证方式。
    """
    arg_types = {
        "this": True,         # 目标表或查询
        "kind": True,         # 复制类型（FROM/TO等）
        "files": False,        # 文件路径列表
        "credentials": False, # 认证信息
        "format": False,      # 数据格式配置
        "params": False,      # 复制参数列表
    }


class Credentials(Expression):
    """
    认证信息表达式类。
    
    表示访问外部资源时的认证配置，如云存储访问密钥、IAM角色等。
    用于COPY、EXPORT等需要访问外部系统的操作。
    """
    arg_types = {
        "credentials": False, # 基本认证信息（用户名密码等）
        "encryption": False,  # 加密配置
        "storage": False,     # 存储类型配置
        "iam_role": False,    # IAM角色（AWS等云平台）
        "region": False,      # 地理区域配置
    }


class Prior(Expression):
    """
    PRIOR表达式类。
    
    表示Oracle等数据库中的PRIOR操作符，主要用于层次查询。
    在CONNECT BY子句中用于引用父节点的值。
    例如：CONNECT BY PRIOR parent_id = child_id
    """
    pass


class Directory(Expression):
    """
    DIRECTORY表达式类。
    
    表示Spark、Hive等大数据系统中的目录操作，
    主要用于INSERT OVERWRITE DIRECTORY语句，将数据写入HDFS目录。
    """
    # 参考：https://spark.apache.org/docs/3.0.0-preview/sql-ref-syntax-dml-insert-overwrite-directory-hive.html
    arg_types = {
        "this": True,        # 目录路径
        "local": False,      # 是否为本地文件系统（vs HDFS）
        "row_format": False, # 行格式配置（Hive格式）
    }


# https://docs.snowflake.com/en/user-guide/data-load-dirtables-query
class DirectoryStage(Expression):
    pass


class ForeignKey(Expression):
    """
    外键表达式类。
    
    表示表级别的外键约束定义，用于维护引用完整性。
    外键确保引用表中的值在被引用表中存在。
    """
    arg_types = {
        "expressions": False, # 外键列列表
        "reference": False,   # 引用的表和列
        "delete": False,      # ON DELETE动作（CASCADE、SET NULL等）
        "update": False,      # ON UPDATE动作
        "options": False,     # 外键选项（如DEFERRABLE等）
    }


class ColumnPrefix(Expression):
    """
    列前缀表达式类。
    
    表示某些数据库中的列前缀功能，用于批量操作具有相同前缀的列。
    主要用于宽表场景下的列集合操作。
    """
    # this: 前缀字符串，expression: 关联表达式
    arg_types = {"this": True, "expression": True}


class PrimaryKey(Expression):
    """
    主键表达式类。
    
    表示表级别的主键约束定义。主键确保行的唯一性，是关系数据库的核心概念。
    一个表只能有一个主键，但主键可以包含多个列（复合主键）。
    """
    arg_types = {
        "expressions": True, # 主键列列表（支持复合主键）
        "options": False,    # 主键选项（如聚集/非聚集等）
        "include": False,    # INCLUDE列（SQL Server等支持）
    }


# 参考：https://www.postgresql.org/docs/9.1/sql-selectinto.html
# 参考：https://docs.aws.amazon.com/redshift/latest/dg/r_SELECT_INTO.html#r_SELECT_INTO-examples
class Into(Expression):
    """
    INTO表达式类。
    
    表示SELECT INTO语句，用于将查询结果插入到新表中。
    支持多种表类型（临时表、非日志表等），主要用于数据复制和临时分析。
    """
    arg_types = {
        "this": False,        # 目标表名
        "temporary": False,   # 是否创建临时表
        "unlogged": False,    # 是否为非日志表（PostgreSQL）
        "bulk_collect": False, # 批量收集选项（Oracle等）
        "expressions": False, # 目标列表达式
    }


class From(Expression):
    """
    FROM表达式类。
    
    表示SQL查询中的FROM子句，指定查询的数据源。
    FROM子句是SELECT语句的核心组成部分，定义了数据的来源。
    """
    
    @property
    def name(self) -> str:
        """
        获取FROM子句中数据源的名称。
        
        直接返回底层表或子查询的名称，用于标识数据源。
        这是FROM子句最基本的标识属性。
        """
        return self.this.name

    @property
    def alias_or_name(self) -> str:
        """
        获取FROM子句中数据源的别名或名称。
        
        优先返回别名，如果没有别名则返回原始名称。
        这个方法在查询分析和代码生成中经常使用。
        """
        return self.this.alias_or_name


class Having(Expression):
    """
    HAVING表达式类。
    
    表示SQL中的HAVING子句，用于对分组后的结果进行过滤。
    HAVING与WHERE的区别在于：WHERE在分组前过滤，HAVING在分组后过滤。
    主要用于聚合查询中的条件筛选。
    """
    pass


class Hint(Expression):
    """
    查询提示表达式类。
    
    表示SQL中的查询提示（Hint），用于向优化器提供执行建议。
    不同数据库的提示语法差异很大，这里提供统一的抽象。
    
    常见用途：
    - 强制使用特定索引
    - 控制连接算法
    - 设置并行度
    """
    # expressions: 提示表达式列表，支持多个提示
    arg_types = {"expressions": True}


class JoinHint(Expression):
    """
    连接提示表达式类。
    
    表示针对特定表连接的提示，用于优化JOIN操作的执行策略。
    比通用Hint更精确，专门针对连接操作。
    
    例如：/*+ USE_NL(t1 t2) */ 强制使用嵌套循环连接
    """
    arg_types = {
        "this": True,        # 目标表或连接
        "expressions": True, # 提示表达式列表
    }


class Identifier(Expression):
    """
    标识符表达式类。
    
    表示SQL中的标识符（如表名、列名、函数名等）。
    标识符是SQL语言的基本构建块，需要处理引用、大小写等复杂性。
    """
    arg_types = {
        "this": True,       # 标识符名称（必需）
        "quoted": False,    # 是否被引号包围
        "global": False,    # 是否为全局标识符
        "temporary": False, # 是否为临时标识符
    }

    @property
    def quoted(self) -> bool:
        """
        检查标识符是否被引号包围。
        
        引用的标识符可以包含特殊字符和关键字，不受SQL标准命名规则限制。
        这在处理包含空格、特殊字符或保留字的标识符时很重要。
        """
        return bool(self.args.get("quoted"))

    @property
    def output_name(self) -> str:
        """
        获取标识符的输出名称。
        
        返回在查询结果中显示的名称，通常就是标识符本身。
        用于结果集的列名生成和显示。
        """
        return self.name


# 参考：https://www.postgresql.org/docs/current/indexes-opclass.html
class Opclass(Expression):
    """
    操作符类表达式类。
    
    表示PostgreSQL等数据库中的操作符类（Operator Class）定义。
    操作符类定义了数据类型在特定索引方法中的行为方式。
    
    主要用于：
    - 自定义索引行为
    - 支持新的数据类型索引
    - 优化特定查询模式
    """
    # this: 操作符类名称，expression: 相关表达式
    arg_types = {"this": True, "expression": True}


class Index(Expression):
    """
    索引表达式类。
    
    表示CREATE INDEX语句，用于创建数据库索引。
    索引是提高查询性能的关键数据结构。
    """
    arg_types = {
        "this": False,     # 索引名称（可选，某些数据库支持匿名索引）
        "table": False,    # 目标表名
        "unique": False,   # 是否为唯一索引
        "primary": False,  # 是否为主键索引
        "amp": False,      # AMP索引（Teradata特有）
        "params": False,   # 索引参数配置
    }


class IndexParameters(Expression):
    """
    索引参数表达式类。
    
    表示创建索引时的各种参数和选项配置。
    不同数据库支持的索引参数差异很大，这里统一了常见选项。
    """
    arg_types = {
        "using": False,         # 索引方法（BTREE、HASH、GIN等）
        "include": False,       # INCLUDE列（覆盖索引）
        "columns": False,       # 索引列定义
        "with_storage": False,  # 存储参数
        "partition_by": False,  # 分区方式
        "tablespace": False,    # 表空间
        "where": False,         # 部分索引条件（WHERE子句）
        "on": False,           # 索引目标（表或表达式）
    }


class Insert(DDL, DML):
    """
    INSERT语句表达式类。
    
    继承自DDL和DML，表示SQL中的INSERT语句。
    INSERT是最复杂的DML语句之一，支持多种插入模式和选项。
    
    支持的插入类型：
    - 简单值插入：INSERT INTO table VALUES (...)
    - 查询插入：INSERT INTO table SELECT ...
    - 批量插入：INSERT INTO table VALUES (...), (...)
    - 冲突处理：INSERT ... ON CONFLICT ...
    """
    arg_types = {
        "hint": False,        # 查询提示
        "with": False,        # WITH子句（CTE）
        "is_function": False, # 是否为函数式插入
        "this": False,        # 目标表
        "expression": False,  # 插入的数据表达式（VALUES或SELECT）
        "conflict": False,    # 冲突处理策略（ON CONFLICT/ON DUPLICATE KEY）
        "returning": False,   # RETURNING子句（返回插入的数据）
        "overwrite": False,   # 是否覆盖模式（INSERT OVERWRITE）
        "exists": False,      # IF EXISTS条件
        "alternative": False, # 替代操作（如REPLACE）
        "where": False,       # WHERE条件（某些数据库支持）
        "ignore": False,      # IGNORE选项（MySQL等）
        "by_name": False,     # 按名称插入（某些列式数据库）
        "stored": False,      # 存储相关选项
        "partition": False,   # 分区信息
        "settings": False,    # 插入设置
        "source": False,      # 数据源表达式
        "default": False,
    }

    def with_(
        self,
        alias: ExpOrStr,
        as_: ExpOrStr,
        recursive: t.Optional[bool] = None,
        materialized: t.Optional[bool] = None,
        append: bool = True,
        dialect: DialectType = None,
        copy: bool = True,
        **opts,
    ) -> Insert:
        """
        添加或设置公共表表达式（CTE）。
        
        这个方法实现了流式API，允许在INSERT语句中使用WITH子句。
        CTE可以简化复杂的插入逻辑，提高查询的可读性。

        示例:
            >>> insert("SELECT x FROM cte", "t").with_("cte", as_="SELECT * FROM tbl").sql()
            'WITH cte AS (SELECT * FROM tbl) INSERT INTO t SELECT x FROM cte'

        参数:
            alias: CTE的别名，用作临时表名
            as_: CTE的查询定义，可以是字符串或Expression实例
            recursive: 是否为递归CTE，默认为False
            materialized: 是否物化CTE（某些数据库支持）
            append: 如果为True，添加到现有CTE；否则替换现有CTE
            dialect: 用于解析输入表达式的SQL方言
            copy: 如果为False，就地修改当前表达式实例
            opts: 解析输入表达式的其他选项

        返回:
            Insert: 修改后的INSERT表达式对象
        """
        # 使用专门的CTE构建器来处理WITH子句的复杂逻辑
        # 这确保了CTE的正确解析、验证和集成
        return _apply_cte_builder(
            self,                    # 当前INSERT实例
            alias,                   # CTE别名
            as_,                     # CTE定义
            recursive=recursive,     # 递归标志
            materialized=materialized, # 物化标志
            append=append,           # 是否追加
            dialect=dialect,         # SQL方言
            copy=copy,              # 是否复制
            **opts,
        )


class ConditionalInsert(Expression):
    """
    条件插入表达式类。
    
    表示某些数据库中的条件插入语法，如Oracle的多表INSERT语句。
    允许根据条件决定插入到不同的表或执行不同的插入逻辑。
    
    例如：INSERT ALL WHEN condition THEN INTO table1 ELSE INTO table2
    """
    arg_types = {
        "this": True,        # 主插入条件或表达式
        "expression": False, # 条件表达式
        "else_": False,      # ELSE分支（可选）
    }


class MultitableInserts(Expression):
    """
    多表插入表达式类。
    
    表示Oracle等数据库中的多表INSERT语句，允许一次性向多个表插入数据。
    这是Oracle的高级特性，可以提高批量数据处理的效率。
    
    支持的模式：
    - INSERT ALL: 向所有匹配的表插入
    - INSERT FIRST: 只向第一个匹配的表插入
    """
    arg_types = {
        "expressions": True, # 插入表达式列表（每个表的插入定义）
        "kind": True,        # 插入类型（ALL或FIRST）
        "source": True,      # 数据源查询
    }


class OnConflict(Expression):
    """
    冲突处理表达式类。
    
    表示INSERT语句中的冲突处理逻辑，如PostgreSQL的ON CONFLICT和MySQL的ON DUPLICATE KEY。
    用于处理插入数据时遇到主键或唯一约束冲突的情况。
    
    支持的处理策略：
    - DO NOTHING: 忽略冲突
    - DO UPDATE: 更新现有记录
    - REPLACE: 替换现有记录
    """
    arg_types = {
        "duplicate": False,    # 是否为重复键冲突（MySQL语法）
        "expressions": False,  # 冲突时要更新的列表达式
        "action": False,       # 冲突处理动作（NOTHING、UPDATE等）
        "conflict_keys": False, # 冲突检测的键列
        "constraint": False,   # 约束名称（用于指定特定约束）
        "where": False,        # 条件表达式（PostgreSQL支持）
    }


class OnCondition(Expression):
    """
    条件处理表达式类。
    
    表示某些数据库中对特殊条件的处理方式，如空值、错误等。
    主要用于数据转换和错误处理场景。
    """
    arg_types = {
        "error": False, # 错误处理选项
        "empty": False, # 空值处理选项
        "null": False,  # NULL值处理选项
    }


class Returning(Expression):
    """
    RETURNING表达式类。
    
    表示DML语句中的RETURNING子句，用于返回被影响的行数据。
    主要用于INSERT、UPDATE、DELETE语句，获取操作后的数据。
    
    常见用途：
    - 获取自动生成的主键值
    - 返回计算列的值
    - 审计和日志记录
    """
    arg_types = {
        "expressions": True, # 要返回的列表达式列表
        "into": False,       # INTO子句（某些数据库支持）
    }


# 参考：https://dev.mysql.com/doc/refman/8.0/en/charset-introducer.html
class Introducer(Expression):
    """
    字符集引导符表达式类。
    
    表示MySQL中的字符集引导符语法，用于指定字符串字面量的字符集。
    例如：_utf8'Hello' 或 _latin1'text'
    
    主要用于多字符集环境下的字符串处理。
    """
    arg_types = {
        "this": True,        # 字符集名称
        "expression": True,  # 字符串表达式
    }


# 国家字符集，如 N'utf8'
class National(Expression):
    """
    国家字符集表达式类。
    
    表示SQL标准中的国家字符集字面量，如N'string'。
    主要用于支持Unicode字符串，确保国际化文本的正确处理。
    
    例如：N'Hello 世界' 表示Unicode字符串
    """
    pass


class LoadData(Expression):
    """
    LOAD DATA表达式类。
    
    表示MySQL、Hive等数据库中的LOAD DATA语句，用于从文件批量导入数据。
    这是高效的批量数据加载机制，比逐行INSERT快得多。
    
    支持的数据源：
    - 本地文件（LOCAL）
    - HDFS文件（Hive/Spark）
    - 云存储文件
    """
    arg_types = {
        "this": True,         # 目标表名
        "local": False,       # 是否为本地文件（LOCAL选项）
        "overwrite": False,   # 是否覆盖现有数据
        "inpath": True,       # 输入文件路径
        "partition": False,   # 分区信息（分区表）
        "input_format": False, # 输入文件格式
        "serde": False,       # 序列化/反序列化器（Hive）
    }


class Partition(Expression):
    """
    分区表达式类。
    
    表示表分区的定义，包括分区列和子分区。
    分区是大型表的重要优化技术，提高查询性能和数据管理效率。
    """
    arg_types = {
        "expressions": True,  # 分区表达式列表（分区列和值）
        "subpartition": False, # 子分区定义（某些数据库支持二级分区）
    }


class PartitionRange(Expression):
    """
    分区范围表达式类。
    
    表示范围分区的定义，指定分区的值范围。
    范围分区是最常用的分区类型，按照列值范围划分数据。
    
    例如：PARTITION p1 VALUES LESS THAN (100)
    """
    arg_types = {
        "this": True,        # 分区名称
        "expression": False, # 单个范围表达式
        "expressions": False, # 多个范围表达式（用于复合分区键）
    }


# 参考：https://clickhouse.com/docs/en/sql-reference/statements/alter/partition#how-to-set-partition-expression
class PartitionId(Expression):
    """
    分区ID表达式类。
    
    表示ClickHouse等数据库中的分区标识符。
    用于在ALTER语句中精确指定要操作的分区。
    
    ClickHouse支持通过分区ID进行精确的分区管理操作。
    """
    pass





class Fetch(Expression):
    """
    FETCH表达式类。
    
    表示SQL标准中的FETCH子句，用于限制查询返回的行数。
    类似于LIMIT，但提供更标准化的语法和更多选项。
    
    FETCH比LIMIT更符合SQL标准，支持百分比和WITH TIES等高级选项。
    """
    arg_types = {
        "direction": False,    # 方向（FIRST、NEXT等）
        "count": False,        # 行数或百分比
        "limit_options": False, # 限制选项（PERCENT、WITH TIES等）
    }


class Grant(Expression):
    """
    GRANT语句表达式类。
    
    表示SQL中的权限授予语句，用于数据库安全管理。
    GRANT是数据库访问控制的核心机制。
    
    支持的权限类型：
    - 对象权限（SELECT、INSERT、UPDATE等）
    - 系统权限（CREATE、DROP等）
    - 角色权限
    """
    arg_types = {
        "privileges": True,   # 权限列表（必需）
        "kind": False,        # 权限类型（对象权限、系统权限等）
        "securable": True,    # 安全对象（表、视图、数据库等）
        "principals": True,   # 被授权主体（用户、角色）
        "grant_option": False, # 是否包含WITH GRANT OPTION
    }


class Revoke(Expression):
    arg_types = {**Grant.arg_types, "cascade": False}


class Group(Expression):
    """
    GROUP BY表达式类。
    
    表示SQL中的GROUP BY子句，用于数据分组和聚合。
    支持多种分组方式，包括简单分组和高级分组功能。
    
    高级分组功能：
    - GROUPING SETS: 多维分组
    - CUBE: 立方体分组（所有可能的组合）
    - ROLLUP: 层次分组（逐级汇总）
    """
    arg_types = {
        "expressions": False,  # 分组表达式列表
        "grouping_sets": False, # GROUPING SETS定义
        "cube": False,         # CUBE分组列
        "rollup": False,       # ROLLUP分组列  
        "totals": False,       # WITH TOTALS（ClickHouse等）
        "all": False,          # GROUP BY ALL选项
    }


class Cube(Expression):
    """
    CUBE表达式类。
    
    表示SQL中的CUBE分组操作，生成所有可能的分组组合。
    CUBE是OLAP分析中的重要功能，用于多维数据分析。
    
    例如：CUBE(a, b, c) 生成8种分组组合：
    - ()、(a)、(b)、(c)、(a,b)、(a,c)、(b,c)、(a,b,c)
    """
    # expressions: 参与CUBE操作的列表达式
    arg_types = {"expressions": False}


class Rollup(Expression):
    """
    ROLLUP表达式类。
    
    表示SQL中的ROLLUP分组操作，生成层次化的分组汇总。
    ROLLUP按照列的层次顺序逐级汇总，常用于报表生成。
    
    例如：ROLLUP(a, b, c) 生成4种分组：
    - (a,b,c)、(a,b)、(a)、()
    """
    # expressions: 参与ROLLUP操作的列表达式（按层次顺序）
    arg_types = {"expressions": False}


class GroupingSets(Expression):
    """
    分组集表达式类。
    
    表示SQL中的GROUPING SETS操作，允许在单个查询中定义多个分组。
    提供比CUBE和ROLLUP更灵活的分组控制。
    
    例如：GROUPING SETS ((a), (b), (a,b)) 只生成指定的分组组合
    """
    # expressions: 分组集定义列表，每个元素是一个分组
    arg_types = {"expressions": True}


class Lambda(Expression):
    """
    Lambda表达式类。
    
    表示SQL中的Lambda函数，主要用于高阶函数和数组操作。
    Lambda表达式允许在SQL中定义匿名函数。
    
    常见用途：
    - 数组变换：transform(array, x -> x * 2)
    - 数组过滤：filter(array, x -> x > 0)
    - 聚合操作：reduce(array, 0, (acc, x) -> acc + x)
    """
    arg_types = {
        "this": True,        # Lambda函数体表达式
        "expressions": True, # 参数列表
        "colon": False,      # 是否使用冒号语法（某些方言）
    }


class Limit(Expression):
    """
    LIMIT表达式类。
    
    表示SQL中的LIMIT子句，用于限制查询返回的行数。
    LIMIT是最常用的分页和结果集限制机制。
    
    支持的功能：
    - 简单限制：LIMIT 10
    - 偏移限制：LIMIT 10 OFFSET 20
    - 百分比限制：LIMIT 50 PERCENT（某些数据库）
    """
    arg_types = {
        "this": False,         # LIMIT表达式本身（可选）
        "expression": True,    # 限制数量表达式（必需）
        "offset": False,       # 偏移量（OFFSET子句）
        "limit_options": False, # 限制选项（如PERCENT、WITH TIES）
        "expressions": False,  # 额外表达式列表
    }


class LimitOptions(Expression):
    """
    LIMIT选项表达式类。
    
    表示LIMIT子句的高级选项，提供更精细的结果集控制。
    这些选项扩展了基本LIMIT功能，支持更复杂的分页需求。
    """
    arg_types = {
        "percent": False,   # 是否使用百分比限制（如LIMIT 50 PERCENT）
        "rows": False,      # 行数限制选项
        "with_ties": False, # WITH TIES选项（包含并列记录）
    }


class Literal(Condition):
    """
    字面量表达式类。
    
    表示SQL中的字面量常数（数字、字符串、布尔值等）。
    字面量是SQL表达式的基础构建块，需要精确处理类型和值。
    
    支持的字面量类型：
    - 数字：123、123.45、1.23E+4
    - 字符串：'hello'、"world"
    - 布尔值：TRUE、FALSE
    - 特殊值：NULL
    """
    arg_types = {
        "this": True,      # 字面量的值
        "is_string": True, # 是否为字符串类型
    }

    @classmethod
    def number(cls, number) -> Literal:
        """
        创建数字字面量的类方法。
        
        将输入的数字转换为字符串存储，但标记为非字符串类型。
        这种设计统一了内部存储格式，同时保持类型信息。
        """
        return cls(this=str(number), is_string=False)

    @classmethod
    def string(cls, string) -> Literal:
        """
        创建字符串字面量的类方法。
        
        将输入转换为字符串并标记为字符串类型。
        提供明确的字符串字面量创建接口。
        """
        return cls(this=str(string), is_string=True)

    @property
    def output_name(self) -> str:
        """
        获取字面量的输出名称。
        
        返回在查询结果中显示的列名，通常就是字面量值本身。
        """
        return self.name

    def to_py(self) -> int | str | Decimal:
        """
        将字面量转换为Python原生类型。
        
        这个方法实现了SQL字面量到Python值的类型转换：
        - 数字字面量：尝试转换为int，失败则转换为Decimal
        - 字符串字面量：保持为字符串
        
        使用Decimal而不是float避免精度丢失，这在财务计算中很重要。
        """
        if self.is_number:
            try:
                # 优先尝试转换为整数，保持精确性
                return int(self.this)
            except ValueError:
                # 如果不是整数，使用Decimal保持精度
                return Decimal(self.this)
        # 字符串类型直接返回
        return self.this


class Join(Expression):
    """
    JOIN表达式类。
    
    表示SQL中的JOIN操作，是关系数据库查询的核心功能。
    支持多种连接类型、连接条件和优化提示。
    
    连接类型：
    - INNER JOIN: 内连接
    - LEFT/RIGHT/FULL OUTER JOIN: 外连接
    - CROSS JOIN: 笛卡尔积
    - SEMI/ANTI JOIN: 半连接/反连接（用于EXISTS/NOT EXISTS优化）
    """
    arg_types = {
        "this": True,               # 要连接的表或表达式
        "on": False,                # ON条件表达式
        "side": False,              # 连接方向（LEFT、RIGHT、FULL）
        "kind": False,              # 连接类型（INNER、OUTER、CROSS等）
        "using": False,             # USING子句的列列表
        "method": False,            # 连接方法提示（如HASH、MERGE、NESTED_LOOP）
        "global": False,            # 全局连接选项（ClickHouse等）
        "hint": False,              # 连接提示
        "match_condition": False,   # 匹配条件（Snowflake特性）
        "expressions": False,       # 额外表达式
        "pivots": False,            # 透视操作
    }

    @property
    def method(self) -> str:
        """
        获取连接方法。
        
        返回连接算法的提示，如HASH、MERGE、NESTED_LOOP等。
        自动转换为大写形式，符合SQL标准惯例。
        """
        return self.text("method").upper()

    @property
    def kind(self) -> str:
        """
        获取连接类型。
        
        返回连接的类型，如INNER、OUTER、CROSS等。
        大写转换确保SQL关键字的标准化表示。
        """
        return self.text("kind").upper()

    @property
    def side(self) -> str:
        """
        获取连接方向。
        
        返回连接的方向，如LEFT、RIGHT、FULL等。
        用于确定外连接的保留方向。
        """
        return self.text("side").upper()

    @property
    def hint(self) -> str:
        """
        获取连接提示。
        
        返回数据库优化器的连接提示信息。
        用于影响查询执行计划的生成。
        """
        return self.text("hint").upper()

    @property
    def alias_or_name(self) -> str:
        """
        获取连接表的别名或名称。
        
        优先返回表的别名，如果没有别名则返回原始表名。
        在查询分析和优化中经常需要标识连接的表。
        """
        return self.this.alias_or_name

    @property
    def is_semi_or_anti_join(self) -> bool:
        """
        检查是否为半连接或反连接。
        
        半连接和反连接是特殊的连接类型，通常用于优化EXISTS/NOT EXISTS子查询。
        这些连接不返回右表的列，只影响左表的行过滤。
        """
        return self.kind in ("SEMI", "ANTI")

    def on(
        self,
        *expressions: t.Optional[ExpOrStr],
        append: bool = True,
        dialect: DialectType = None,
        copy: bool = True,
        **opts,
    ) -> Join:
        """
        添加或设置ON条件表达式。
        
        ON条件定义了表之间的连接关系，是JOIN操作的核心逻辑。
        支持复杂的连接条件，多个条件会用AND连接。

        示例:
            >>> import sqlglot
            >>> sqlglot.parse_one("JOIN x", into=Join).on("y = 1").sql()
            'JOIN x ON y = 1'

        参数:
            *expressions: 要解析的SQL条件字符串。
                如果传入Expression实例，将直接使用。
                多个表达式会用AND操作符组合。
            append: 如果为True，将新条件与现有条件用AND连接。
                否则，重置现有条件。
            dialect: 用于解析输入表达式的SQL方言
            copy: 如果为False，就地修改当前表达式实例
            opts: 解析输入表达式的其他选项

        返回:
            Join: 修改后的JOIN表达式对象
        """
        # 使用连接条件构建器处理ON子句的逻辑组合
        join = _apply_conjunction_builder(
            *expressions,
            instance=self,
            arg="on",
            append=append,
            dialect=dialect,
            copy=copy,
            **opts,
        )

        # 特殊处理：如果原来是CROSS JOIN，添加ON条件后变为INNER JOIN
        # CROSS JOIN不应该有ON条件，有ON条件的是INNER JOIN
        if join.kind == "CROSS":
            join.set("kind", None)

        return join

    def using(
        self,
        *expressions: t.Optional[ExpOrStr],
        append: bool = True,
        dialect: DialectType = None,
        copy: bool = True,
        **opts,
    ) -> Join:
        """
        添加或设置USING子句表达式。
        
        USING是ON条件的简化形式，用于连接具有相同列名的表。
        USING(col1, col2)等价于ON t1.col1 = t2.col1 AND t1.col2 = t2.col2

        示例:
            >>> import sqlglot
            >>> sqlglot.parse_one("JOIN x", into=Join).using("foo", "bla").sql()
            'JOIN x USING (foo, bla)'

        参数:
            *expressions: 要解析的SQL列名字符串。
                如果传入Expression实例，将直接使用。
            append: 如果为True，将新列添加到现有USING列表。
                否则，重置现有列表。
            dialect: 用于解析输入表达式的SQL方言
            copy: 如果为False，就地修改当前表达式实例
            opts: 解析输入表达式的其他选项

        返回:
            Join: 修改后的JOIN表达式对象
        """
        # 使用列表构建器处理USING子句的列列表
        join = _apply_list_builder(
            *expressions,
            instance=self,
            arg="using",
            append=append,
            dialect=dialect,
            copy=copy,
            **opts,
        )

        # 特殊处理：如果原来是CROSS JOIN，添加USING条件后变为INNER JOIN
        # 原理同ON条件处理
        if join.kind == "CROSS":
            join.set("kind", None)

        return join


class Lateral(UDTF):
    """
    LATERAL表达式类。
    
    继承自UDTF（用户定义表函数），表示LATERAL子查询或表函数。
    LATERAL允许子查询引用前面表的列，是SQL的高级特性。
    
    用途：
    - 相关子查询的表形式
    - 表函数调用
    - SQL Server的CROSS/OUTER APPLY
    """
    arg_types = {
        "this": True,         # LATERAL的表达式或函数
        "view": False,        # 是否为视图形式
        "outer": False,       # 是否为OUTER LATERAL（保留左表的所有行）
        "alias": False,       # 别名定义
        "cross_apply": False, # True表示CROSS APPLY，False表示OUTER APPLY（SQL Server）
        "ordinality": False,  # 是否包含序号列（PostgreSQL特性）
    }


# 参考：https://docs.snowflake.com/sql-reference/literals-table
# 参考：https://docs.snowflake.com/en/sql-reference/functions-table#using-a-table-function
class TableFromRows(UDTF):
    """
    行转表表达式类。
    
    继承自UDTF，表示从行数据创建表的功能。
    主要用于Snowflake等现代数据仓库的表字面量和表函数功能。
    
    例如：将VALUES子句或函数结果转换为可查询的表形式。
    """
    arg_types = {
        "this": True,     # 行数据表达式或表函数
        "alias": False,   # 表别名
        "joins": False,   # 连接操作
        "pivots": False,  # 透视操作
        "sample": False,  # 采样配置
    }


class MatchRecognizeMeasure(Expression):
    """
    模式识别度量表达式类。
    
    表示MATCH_RECOGNIZE子句中的度量定义。
    度量用于计算模式匹配过程中的聚合值。
    
    主要用于复杂事件处理和时间序列分析。
    """
    arg_types = {
        "this": True,         # 度量表达式
        "window_frame": False, # 窗口框架定义
    }


class MatchRecognize(Expression):
    """
    模式识别表达式类。
    
    表示SQL标准中的MATCH_RECOGNIZE子句，用于在行序列中识别模式。
    这是SQL的高级分析功能，主要用于：
    - 复杂事件处理（CEP）
    - 时间序列模式分析
    - 业务流程监控
    
    例如：识别股票价格的上升趋势、检测系统异常模式等。
    """
    arg_types = {
        "partition_by": False, # 分区列（类似窗口函数）
        "order": False,        # 排序规则（定义行序列顺序）
        "measures": False,     # 度量定义（计算输出列）
        "rows": False,         # 行匹配选项
        "after": False,        # 匹配后的行处理
        "pattern": False,      # 模式定义（正则表达式风格）
        "define": False,       # 模式变量定义
        "alias": False,        # 别名
    }


# ClickHouse FROM FINAL修饰符
# 参考：https://clickhouse.com/docs/en/sql-reference/statements/select/from/#final-modifier
class Final(Expression):
    """
    FINAL表达式类。
    
    表示ClickHouse中的FROM FINAL修饰符。
    FINAL强制ClickHouse在查询时合并所有数据部分，确保数据一致性。
    
    用途：
    - 获取最新的数据状态
    - 在ReplacingMergeTree等引擎中去重
    - 确保强一致性读取（但会影响性能）
    """
    pass


class Offset(Expression):
    """
    OFFSET表达式类。
    
    表示SQL中的OFFSET子句，用于跳过指定数量的行。
    通常与LIMIT配合使用实现分页功能。
    
    OFFSET是标准SQL的分页机制，比MySQL的LIMIT offset, count更通用。
    """
    arg_types = {
        "this": False,        # OFFSET表达式本身（可选）
        "expression": True,   # 偏移量表达式（必需）
        "expressions": False, # 额外表达式列表
    }


class Order(Expression):
    """
    ORDER BY表达式类。
    
    表示SQL中的ORDER BY子句，用于指定查询结果的排序规则。
    排序是SQL查询的重要功能，影响结果的呈现顺序。
    """
    arg_types = {
        "this": False,        # ORDER表达式本身（可选）
        "expressions": True,  # 排序表达式列表（列名和排序方向）
        "siblings": False,    # SIBLINGS选项（Oracle层次查询）
    }


# 参考：https://clickhouse.com/docs/en/sql-reference/statements/select/order-by#order-by-expr-with-fill-modifier
class WithFill(Expression):
    """
    WITH FILL表达式类。
    
    表示ClickHouse中的WITH FILL修饰符，用于在ORDER BY中填充缺失值。
    这是时间序列分析的有用功能，可以生成连续的时间点数据。
    
    例如：填充缺失的日期，生成完整的时间序列数据。
    """
    arg_types = {
        "from": False,        # 填充起始值
        "to": False,          # 填充结束值
        "step": False,        # 填充步长
        "interpolate": False, # 插值表达式
    }


# Hive特定的排序方式
# 参考：https://cwiki.apache.org/confluence/display/Hive/LanguageManual+SortBy
class Cluster(Order):
    """
    CLUSTER BY表达式类。
    
    继承自Order，表示Hive中的CLUSTER BY子句。
    CLUSTER BY确保相同键值的行在同一个reducer中处理，用于优化分布式计算。
    
    在MapReduce环境中，CLUSTER BY影响数据的分布和排序策略。
    """
    pass


class Distribute(Order):
    """
    DISTRIBUTE BY表达式类。
    
    继承自Order，表示Hive中的DISTRIBUTE BY子句。
    DISTRIBUTE BY控制行在reducer之间的分布，但不保证reducer内部的排序。
    
    通常与SORT BY配合使用，实现分布式环境下的数据分区。
    """
    pass


class Sort(Order):
    """
    SORT BY表达式类。
    
    继承自Order，表示Hive中的SORT BY子句。
    SORT BY只在每个reducer内部排序，不保证全局排序。
    
    与ORDER BY的区别：ORDER BY保证全局排序，SORT BY只保证局部排序。
    """
    pass


class Ordered(Expression):
    """
    排序项表达式类。
    
    表示ORDER BY子句中的单个排序项，包含列、排序方向和NULL值处理。
    这是排序规则的基本单元。
    """
    arg_types = {
        "this": True,        # 排序表达式（列名或表达式）
        "desc": False,       # 是否降序排列（False表示升序）
        "nulls_first": True, # NULL值是否排在前面
        "with_fill": False,  # 是否使用WITH FILL（ClickHouse）
    }

    @property
    def name(self) -> str:
        """
        获取排序项的名称。
        
        返回排序表达式的名称，通常是列名。
        用于排序规则的标识和显示。
        """
        return self.this.name


class Property(Expression):
    """
    属性表达式基类。
    
    表示数据库对象的属性定义，如表属性、视图属性、索引属性等。
    属性是DDL语句中的重要组成部分，控制对象的行为和特性。
    """
    # this: 属性名称，value: 属性值
    arg_types = {"this": True, "value": True}


class GrantPrivilege(Expression):
    """
    授权权限表达式类。
    
    表示GRANT语句中的权限定义，如SELECT、INSERT、UPDATE等。
    用于数据库安全管理和访问控制。
    """
    # this: 权限名称，expressions: 权限相关的表达式
    arg_types = {"this": True, "expressions": False}


class GrantPrincipal(Expression):
    """
    授权主体表达式类。
    
    表示GRANT语句中的被授权主体，如用户、角色、组等。
    定义权限的接收者。
    """
    # this: 主体名称，kind: 主体类型（USER、ROLE等）
    arg_types = {"this": True, "kind": False}


class AllowedValuesProperty(Expression):
    """
    允许值属性表达式类。
    
    表示某些数据库中的允许值限制，用于约束列的可能取值。
    类似于CHECK约束，但更明确地列出允许的值。
    """
    # expressions: 允许的值列表
    arg_types = {"expressions": True}


class AlgorithmProperty(Property):
    """
    算法属性类。
    
    表示索引或表的算法选择，如BTREE、HASH等。
    主要用于MySQL等数据库的索引算法指定。
    """
    # this: 算法名称
    arg_types = {"this": True}


class AutoIncrementProperty(Property):
    """
    自增属性类。
    
    表示表的AUTO_INCREMENT起始值设置。
    用于指定自增列的起始值和增量。
    """
    # this: 自增起始值
    arg_types = {"this": True}


# 参考：https://docs.aws.amazon.com/prescriptive-guidance/latest/materialized-views-redshift/refreshing-materialized-views.html
class AutoRefreshProperty(Property):
    """
    自动刷新属性类。
    
    表示Amazon Redshift等数据库中的物化视图自动刷新配置。
    控制物化视图是否自动更新以保持数据最新。
    """
    # this: 是否启用自动刷新
    arg_types = {"this": True}


class BackupProperty(Property):
    """
    备份属性类。
    
    表示表或数据库的备份相关配置。
    用于控制数据的备份策略和恢复选项。
    """
    # this: 备份配置
    arg_types = {"this": True}


# https://doris.apache.org/docs/sql-manual/sql-statements/table-and-view/async-materialized-view/CREATE-ASYNC-MATERIALIZED-VIEW/
class BuildProperty(Property):
    arg_types = {"this": True}


class BlockCompressionProperty(Property):
    """
    块压缩属性类。
    
    表示Teradata等数据库中的块级压缩配置。
    压缩可以节省存储空间，但可能影响查询性能。
    """
    arg_types = {
        "autotemp": False,  # 自动临时压缩
        "always": False,    # 始终压缩
        "default": False,   # 默认压缩策略
        "manual": False,    # 手动压缩
        "never": False,     # 从不压缩
    }


class CharacterSetProperty(Property):
    """
    字符集属性类。
    
    表示表或列的字符集设置，如UTF8、GBK等。
    字符集影响文本数据的存储和比较规则。
    """
    # this: 字符集名称，default: 是否为默认字符集
    arg_types = {"this": True, "default": True}


class ChecksumProperty(Property):
    """
    校验和属性类。
    
    表示表的校验和配置，用于数据完整性验证。
    校验和可以检测数据损坏，但会增加存储开销。
    """
    # on: 是否启用校验和，default: 是否使用默认设置
    arg_types = {"on": False, "default": False}


class CollateProperty(Property):
    """
    排序规则属性类。
    
    表示表或列的排序规则设置，影响字符串比较和排序。
    不同的排序规则支持不同的语言和地区特性。
    """
    # this: 排序规则名称，default: 是否为默认排序规则
    arg_types = {"this": True, "default": False}


class CopyGrantsProperty(Property):
    """
    复制权限属性类。
    
    表示Snowflake等数据库中的权限复制功能。
    在创建视图或表时自动复制源对象的权限。
    """
    arg_types = {}


class DataBlocksizeProperty(Property):
    """
    数据块大小属性类。
    
    表示Teradata等数据库中的数据块大小配置。
    块大小影响I/O性能和存储效率。
    """
    arg_types = {
        "size": False,     # 块大小值
        "units": False,    # 大小单位（KB、MB等）
        "minimum": False,  # 最小块大小
        "maximum": False,  # 最大块大小
        "default": False,  # 默认块大小
    }


class DataDeletionProperty(Property):
    """
    数据删除属性类。
    
    表示自动数据删除策略，如TTL（生存时间）设置。
    用于自动清理过期数据，常见于时间序列数据管理。
    """
    arg_types = {
        "on": True,             # 是否启用自动删除
        "filter_col": False,    # 过滤列（通常是时间列）
        "retention_period": False, # 保留期限
    }


class DefinerProperty(Property):
    """
    定义者属性类。
    
    表示视图或存储过程的定义者（创建者）。
    影响权限检查和执行上下文。
    """
    # this: 定义者用户名
    arg_types = {"this": True}


class DistKeyProperty(Property):
    """
    分布键属性类。
    
    表示Amazon Redshift中的分布键设置。
    分布键决定数据在集群节点间的分布方式，影响查询性能。
    """
    # this: 分布键列名
    arg_types = {"this": True}


# 参考：https://docs.starrocks.io/docs/sql-reference/sql-statements/data-definition/CREATE_TABLE/#distribution_desc
# 参考：https://doris.apache.org/docs/sql-manual/sql-statements/Data-Definition-Statements/Create/CREATE-TABLE?_highlight=create&_highlight=table#distribution_desc
class DistributedByProperty(Property):
    """
    分布式属性类。
    
    表示StarRocks、Doris等MPP数据库中的数据分布策略。
    控制数据在分布式集群中的分布方式。
    """
    arg_types = {
        "expressions": False, # 分布列表达式
        "kind": True,         # 分布类型（HASH、RANDOM等）
        "buckets": False,     # 分桶数量
        "order": False,       # 排序规则
    }


class DistStyleProperty(Property):
    """
    分布样式属性类。
    
    表示Amazon Redshift中的分布样式（EVEN、KEY、ALL）。
    分布样式是Redshift性能优化的关键配置。
    """
    # this: 分布样式类型
    arg_types = {"this": True}


class DuplicateKeyProperty(Property):
    """
    重复键属性类。
    
    表示DorisDB等数据库中的重复键模型配置。
    重复键模型允许主键重复，适用于明细数据存储。
    """
    # expressions: 重复键列列表
    arg_types = {"expressions": True}


class EngineProperty(Property):
    """
    存储引擎属性类。
    
    表示表的存储引擎，如InnoDB、MyISAM、ClickHouse的MergeTree等。
    存储引擎决定了数据的存储格式和查询特性。
    """
    # this: 存储引擎名称
    arg_types = {"this": True}


class HeapProperty(Property):
    """
    堆属性类。
    
    表示PostgreSQL等数据库中的堆表属性。
    堆是最基本的表存储结构。
    """
    arg_types = {}


class ToTableProperty(Property):
    """
    目标表属性类。
    
    表示某些操作的目标表配置。
    用于指定数据迁移或转换的目标。
    """
    # this: 目标表名
    arg_types = {"this": True}


class ExecuteAsProperty(Property):
    """
    执行身份属性类。
    
    表示SQL Server等数据库中的EXECUTE AS设置。
    控制存储过程或函数的执行上下文。
    """
    # this: 执行身份（CALLER、OWNER等）
    arg_types = {"this": True}


class ExternalProperty(Property):
    """
    外部表属性类。
    
    表示外部表的配置，用于访问外部数据源。
    外部表不存储数据，只提供数据访问接口。
    """
    # this: 是否为外部表
    arg_types = {"this": False}


class FallbackProperty(Property):
    """
    回退属性类。
    
    表示Teradata等数据库中的回退保护设置。
    回退保护在节点故障时提供数据可用性。
    """
    # no: 是否禁用回退，protection: 保护级别
    arg_types = {"no": True, "protection": False}


# 参考：https://docs.databricks.com/aws/en/sql/language-manual/sql-ref-syntax-ddl-create-table-hiveformat
class FileFormatProperty(Property):
    """
    文件格式属性类。
    
    表示外部表或数据湖表的文件格式配置。
    支持Parquet、ORC、JSON、CSV等多种格式。
    """
    arg_types = {
        "this": False,         # 格式名称
        "expressions": False,  # 格式参数
        "hive_format": False,  # 是否为Hive格式
    }


class CredentialsProperty(Property):
    """
    凭证属性类。
    
    表示访问外部资源时的认证凭证配置。
    用于云存储、远程数据库等的安全访问。
    """
    # expressions: 凭证配置列表
    arg_types = {"expressions": True}


class FreespaceProperty(Property):
    """
    空闲空间属性类。
    
    表示Teradata等数据库中的空闲空间管理配置。
    控制表的空间分配和管理策略。
    """
    # this: 空闲空间大小，percent: 是否为百分比
    arg_types = {"this": True, "percent": False}


class GlobalProperty(Property):
    """
    全局属性类。
    
    表示全局性的表属性配置。
    用于标识全局临时表等特殊表类型。
    """
    arg_types = {}


class IcebergProperty(Property):
    """
    Iceberg属性类。
    
    表示Apache Iceberg表格式的相关属性。
    Iceberg是现代数据湖的表格式标准。
    """
    arg_types = {}


class InheritsProperty(Property):
    """
    继承属性类。
    
    表示PostgreSQL等数据库中的表继承关系。
    子表可以继承父表的结构和数据。
    """
    # expressions: 父表列表
    arg_types = {"expressions": True}


class InputModelProperty(Property):
    """
    输入模型属性类。
    
    表示机器学习场景中的输入模型配置。
    用于AI/ML数据库的模型管理。
    """
    # this: 输入模型名称
    arg_types = {"this": True}


class OutputModelProperty(Property):
    """
    输出模型属性类。
    
    表示机器学习场景中的输出模型配置。
    用于AI/ML数据库的模型管理。
    """
    # this: 输出模型名称
    arg_types = {"this": True}


class IsolatedLoadingProperty(Property):
    """
    隔离加载属性类。
    
    表示Teradata等数据库中的隔离加载配置。
    控制数据加载过程中的隔离级别和并发性。
    """
    arg_types = {
        "no": False,         # 是否禁用隔离加载
        "concurrent": False, # 是否支持并发
        "target": False,     # 目标配置
    }


class JournalProperty(Property):
    """
    日志属性类。
    
    表示Teradata等数据库中的事务日志配置。
    日志用于数据恢复和事务管理。
    """
    arg_types = {
        "no": False,     # 是否禁用日志
        "dual": False,   # 是否双重日志
        "before": False, # 前置日志
        "local": False,  # 本地日志
        "after": False,  # 后置日志
    }


class LanguageProperty(Property):
    """
    语言属性类。
    
    表示存储过程或函数的编程语言。
    如SQL、PL/SQL、Python、Java等。
    """
    # this: 编程语言名称
    arg_types = {"this": True}


class EnviromentProperty(Property):
    """
    环境属性类。
    
    表示执行环境的配置参数。
    用于函数或存储过程的环境设置。
    """
    # expressions: 环境变量列表
    arg_types = {"expressions": True}


# Spark DDL
class ClusteredByProperty(Property):
    """
    聚集属性类。
    
    表示Spark等大数据系统中的数据聚集配置。
    聚集影响数据在存储中的物理分布。
    """
    arg_types = {
        "expressions": True, # 聚集列列表
        "sorted_by": False,  # 排序列
        "buckets": True,     # 分桶数量
    }


class DictProperty(Property):
    """
    字典属性类。
    
    表示ClickHouse等数据库中的字典配置。
    字典用于高效的键值查找和数据关联。
    """
    arg_types = {
        "this": True,     # 字典名称
        "kind": True,     # 字典类型
        "settings": False, # 字典设置
    }


class DictSubProperty(Property):
    """
    字典子属性类。
    
    表示字典的子配置项。
    用于复杂字典结构的层次化配置。
    """
    pass


class DictRange(Property):
    """
    字典范围属性类。
    
    表示ClickHouse字典中的范围配置。
    用于范围字典的键值范围定义。
    """
    # this: 字典名称，min: 最小值，max: 最大值
    arg_types = {"this": True, "min": True, "max": True}


class DynamicProperty(Property):
    """
    动态属性类。
    
    表示动态配置的属性，运行时可变。
    用于支持动态调整的表配置。
    """
    arg_types = {}


# ClickHouse CREATE ... ON CLUSTER修饰符
# 参考：https://clickhouse.com/docs/en/sql-reference/distributed-ddl
class OnCluster(Property):
    """
    集群属性类。
    
    表示ClickHouse中的ON CLUSTER语法，用于集群范围的DDL操作。
    在所有集群节点上同步执行DDL语句。
    """
    # this: 集群名称
    arg_types = {"this": True}


# ClickHouse EMPTY表"属性"
class EmptyProperty(Property):
    """
    空表属性类。
    
    表示ClickHouse中的EMPTY表选项，创建空的表结构。
    只复制表结构，不复制数据。
    """
    arg_types = {}


class LikeProperty(Property):
    """
    相似属性类。
    
    表示CREATE TABLE LIKE语法，复制现有表的结构。
    LIKE是快速创建相同结构表的便捷方式。
    """
    # this: 源表名称，expressions: 额外选项
    arg_types = {"this": True, "expressions": False}


class LocationProperty(Property):
    """
    位置属性类。
    
    表示外部表或数据湖表的数据位置。
    通常是HDFS路径、S3路径等存储位置。
    """
    # this: 数据位置路径
    arg_types = {"this": True}


class LockProperty(Property):
    """
    锁属性类。
    
    表示表的锁定配置。
    用于控制表的并发访问和数据一致性。
    """
    # this: 锁定类型
    arg_types = {"this": True}


class LockingProperty(Property):
    """
    锁定属性类。
    
    表示Teradata等数据库中的详细锁定配置。
    控制事务中的锁定行为和级别。
    """
    arg_types = {
        "this": False,      # 锁定目标
        "kind": True,       # 锁定类型（必需）
        "for_or_in": False, # FOR/IN关键字
        "lock_type": True,  # 具体锁类型（必需）
        "override": False,  # 是否覆盖默认锁定
    }


class LogProperty(Property):
    """
    日志属性类。
    
    表示是否启用表的操作日志。
    日志记录对表的修改操作，用于审计和恢复。
    """
    # no: 是否禁用日志（True表示NO LOG）
    arg_types = {"no": True}


class MaterializedProperty(Property):
    """
    物化属性类。
    
    表示视图是否为物化视图。
    物化视图存储查询结果，提高查询性能。
    """
    # this: 是否物化
    arg_types = {"this": False}


class MergeBlockRatioProperty(Property):
    """
    合并块比率属性类。
    
    表示Teradata等数据库中的块合并配置。
    控制数据块的合并策略和比率。
    """
    arg_types = {
        "this": False,    # 比率值
        "no": False,      # 是否禁用合并
        "default": False, # 是否使用默认值
        "percent": False, # 是否为百分比
    }


class NoPrimaryIndexProperty(Property):
    """
    无主索引属性类。
    
    表示Teradata等数据库中的无主索引表配置。
    无主索引表适用于特定的数据分布场景。
    """
    arg_types = {}


class OnProperty(Property):
    """
    基于属性类。
    
    表示基于某个表或视图的配置。
    用于指定依赖关系或基础对象。
    """
    # this: 基础对象名称
    arg_types = {"this": True}


class OnCommitProperty(Property):
    """
    提交时属性类。
    
    表示临时表在事务提交时的行为。
    如DELETE ROWS（清空数据）或PRESERVE ROWS（保留数据）。
    """
    # delete: 是否在提交时删除行
    arg_types = {"delete": False}


class PartitionedByProperty(Property):
    """
    分区属性类。
    
    表示表的分区配置，分区是大表优化的重要技术。
    按照指定列对数据进行物理分割。
    """
    # this: 分区表达式
    arg_types = {"this": True}


class PartitionedByBucket(Property):
    """
    分桶分区属性类。
    
    表示基于分桶的分区配置。
    分桶是对分区的进一步细分。
    """
    # this: 分桶列，expression: 分桶表达式
    arg_types = {"this": True, "expression": True}


class PartitionByTruncate(Property):
    """
    截断分区属性类。
    
    表示基于截断函数的分区配置。
    如按年、月、日截断时间戳进行分区。
    """
    # this: 分区列，expression: 截断表达式
    arg_types = {"this": True, "expression": True}


# 参考：https://docs.starrocks.io/docs/sql-reference/sql-statements/table_bucket_part_index/CREATE_TABLE/
class PartitionByRangeProperty(Property):
    """
    范围分区属性类。
    
    表示StarRocks等数据库中的范围分区配置。
    按照值范围对数据进行分区。
    """
    arg_types = {
        "partition_expressions": True, # 分区表达式列表
        "create_expressions": True,    # 创建表达式列表
    }


# 参考：https://docs.starrocks.io/docs/table_design/data_distribution/#range-partitioning
class PartitionByRangePropertyDynamic(Expression):
    """
    动态范围分区属性类。
    
    表示StarRocks等数据库中的动态范围分区。
    自动根据数据范围创建分区。
    """
    arg_types = {
        "this": False,  # 分区名称模式
        "start": True,  # 起始值
        "end": True,    # 结束值
        "every": True,  # 间隔值
    }

# https://doris.apache.org/docs/table-design/data-partitioning/manual-partitioning
class PartitionByListProperty(Property):
    arg_types = {"partition_expressions": True, "create_expressions": True}


# https://doris.apache.org/docs/table-design/data-partitioning/manual-partitioning
class PartitionList(Expression):
    arg_types = {"this": True, "expressions": True}


# https://doris.apache.org/docs/sql-manual/sql-statements/table-and-view/async-materialized-view/CREATE-ASYNC-MATERIALIZED-VIEW
class RefreshTriggerProperty(Property):
    arg_types = {
        "method": True,
        "kind": False,
        "every": False,
        "unit": False,
        "starts": False,
    }


# 参考：https://docs.starrocks.io/docs/sql-reference/sql-statements/table_bucket_part_index/CREATE_TABLE/
class UniqueKeyProperty(Property):
    """
    唯一键属性类。
    
    表示StarRocks等数据库中的唯一键模型配置。
    唯一键模型适用于有更新需求的数据场景。
    """
    # expressions: 唯一键列列表
    arg_types = {"expressions": True}


# 参考：https://www.postgresql.org/docs/current/sql-createtable.html
class PartitionBoundSpec(Expression):
    """
    分区边界规范表达式类。
    
    表示PostgreSQL等数据库中的分区边界定义。
    用于声明式分区的边界值配置。
    """
    # this -> IN/MODULUS, expression -> REMAINDER, from_expressions -> FROM (...), to_expressions -> TO (...)
    arg_types = {
        "this": False,              # 分区类型（IN、MODULUS等）
        "expression": False,        # 余数或表达式
        "from_expressions": False,  # FROM值列表
        "to_expressions": False,    # TO值列表
    }


class PartitionedOfProperty(Property):
    """
    分区所属属性类。
    
    表示PostgreSQL中的分区表所属关系。
    子分区表指向其父表的配置。
    """
    # this -> 父表（schema），expression -> FOR VALUES ... / DEFAULT
    arg_types = {"this": True, "expression": True}


class StreamingTableProperty(Property):
    """
    流表属性类。
    
    表示Databricks等平台中的流表配置。
    流表用于实时数据处理和流式计算。
    """
    arg_types = {}


class RemoteWithConnectionModelProperty(Property):
    """
    远程连接模型属性类。
    
    表示远程数据访问的连接模型配置。
    用于跨数据库或跨系统的数据访问。
    """
    # this: 连接模型名称
    arg_types = {"this": True}


class ReturnsProperty(Property):
    """
    返回属性类。
    
    表示函数或存储过程的返回类型配置。
    定义返回值的数据类型和结构。
    """
    arg_types = {
        "this": False,     # 返回类型
        "is_table": False, # 是否返回表
        "table": False,    # 表定义
        "null": False,     # 是否可以返回NULL
    }


class StrictProperty(Property):
    """
    严格属性类。
    
    表示函数的严格性配置。
    严格函数不接受NULL参数。
    """
    arg_types = {}


class RowFormatProperty(Property):
    """
    行格式属性类。
    
    表示Hive等大数据系统中的行格式配置。
    定义数据在文件中的存储格式。
    """
    # this: 行格式类型
    arg_types = {"this": True}


class RowFormatDelimitedProperty(Property):
    """
    分隔行格式属性类。
    
    表示Hive中的分隔符行格式配置。
    定义字段、集合、映射等的分隔符。
    """
    # 参考：https://cwiki.apache.org/confluence/display/hive/languagemanual+dml
    arg_types = {
        "fields": False,         # 字段分隔符
        "escaped": False,        # 转义字符
        "collection_items": False, # 集合项分隔符
        "map_keys": False,       # 映射键分隔符
        "lines": False,          # 行分隔符
        "null": False,           # NULL值表示
        "serde": False,          # 序列化器
    }


class RowFormatSerdeProperty(Property):
    """
    SerDe行格式属性类。
    
    表示Hive中的SerDe（序列化/反序列化器）配置。
    SerDe定义了数据的读写格式。
    """
    # this: SerDe类名，serde_properties: SerDe属性
    arg_types = {"this": True, "serde_properties": False}


# 参考：https://spark.apache.org/docs/3.1.2/sql-ref-syntax-qry-select-transform.html
class QueryTransform(Expression):
    """
    查询转换表达式类。
    
    表示Spark等系统中的TRANSFORM语法，用于数据转换。
    允许使用外部脚本处理查询数据。
    """
    arg_types = {
        "expressions": True,       # 输入表达式
        "command_script": True,    # 转换脚本命令
        "schema": False,           # 输出模式
        "row_format_before": False, # 输入行格式
        "record_writer": False,    # 记录写入器
        "row_format_after": False, # 输出行格式
        "record_reader": False,    # 记录读取器
    }


class SampleProperty(Property):
    """
    采样属性类。
    
    表示表的数据采样配置。
    用于创建表的样本数据子集。
    """
    # this: 采样配置
    arg_types = {"this": True}


# 参考：https://prestodb.io/docs/current/sql/create-view.html#synopsis
class SecurityProperty(Property):
    """
    安全属性类。
    
    表示Presto等数据库中的安全级别配置。
    控制视图或表的安全访问策略。
    """
    # this: 安全级别
    arg_types = {"this": True}


class SchemaCommentProperty(Property):
    """
    模式注释属性类。
    
    表示数据库对象的注释信息。
    注释用于文档化对象的用途和说明。
    """
    # this: 注释内容
    arg_types = {"this": True}


class SemanticView(Expression):
    """
    语义视图表达式类。
    
    表示现代数据平台中的语义层视图。
    语义视图封装业务逻辑和指标定义。
    """
    arg_types = {
        "this": True,        # 视图名称
        "metrics": False,    # 指标定义
        "dimensions": False, # 维度定义
        "where": False,      # 过滤条件
    }


class SerdeProperties(Property):
    """
    SerDe属性类。
    
    表示Hive SerDe的详细属性配置。
    控制序列化和反序列化的具体行为。
    """
    # expressions: 属性列表，with: WITH关键字
    arg_types = {"expressions": True, "with": False}


class SetProperty(Property):
    """
    设置属性类。
    
    表示各种SET配置的属性。
    用于数据库或表的参数设置。
    """
    # multi: 是否支持多值设置
    arg_types = {"multi": True}


class SharingProperty(Property):
    """
    共享属性类。
    
    表示Snowflake等云数据库中的数据共享配置。
    控制数据在账户间的共享策略。
    """
    # this: 共享配置
    arg_types = {"this": False}


class SetConfigProperty(Property):
    """
    配置设置属性类。
    
    表示系统配置参数的设置。
    用于调整数据库行为和性能参数。
    """
    # this: 配置参数名
    arg_types = {"this": True}


class SettingsProperty(Property):
    """
    设置集属性类。
    
    表示多个设置项的集合配置。
    用于批量配置数据库参数。
    """
    # expressions: 设置表达式列表
    arg_types = {"expressions": True}


class SortKeyProperty(Property):
    """
    排序键属性类。
    
    表示Amazon Redshift中的排序键配置。
    排序键影响数据的物理存储顺序和查询性能。
    """
    # this: 排序键列，compound: 是否为复合排序键
    arg_types = {"this": True, "compound": False}


class SqlReadWriteProperty(Property):
    """
    SQL读写属性类。
    
    表示函数的SQL数据访问特性。
    定义函数是否读取或修改SQL数据。
    """
    # this: 读写类型（READS SQL DATA、MODIFIES SQL DATA等）
    arg_types = {"this": True}

class TableReadWriteProperty(Property):
    """
    Table读写属性类。
    
    表示外表数据访问特性。
    定义外表是否允许读取或修改数据。
    """
    # this: 读写类型（READ ONLY、WRITE ONLY, READ WRITE等）
    arg_types = {"this": True}

class SqlSecurityProperty(Property):
    """
    SQL安全属性类。
    
    表示函数或视图的SQL安全模式。
    控制执行时的权限检查方式。
    """
    # definer: 是否使用定义者权限
    arg_types = {"definer": True}


class StabilityProperty(Property):
    """
    稳定性属性类。
    
    表示函数的稳定性级别。
    如IMMUTABLE（不变）、STABLE（稳定）、VOLATILE（易变）。
    """
    # this: 稳定性级别
    arg_types = {"this": True}


class StorageHandlerProperty(Property):
    """
    存储处理器属性类。
    
    表示Hive等系统中的存储处理器配置。
    存储处理器定义了数据的存储和访问方式。
    """
    # this: 存储处理器类名
    arg_types = {"this": True}


class TemporaryProperty(Property):
    """
    临时属性类。
    
    表示表是否为临时表。
    临时表在会话结束时自动删除。
    """
    # this: 是否为临时表
    arg_types = {"this": False}


class SecureProperty(Property):
    """
    安全属性类。
    
    表示安全相关的表属性配置。
    用于启用额外的安全特性。
    """
    arg_types = {}


# 参考：https://docs.snowflake.com/en/sql-reference/sql/create-table
class Tags(ColumnConstraintKind, Property):
    """
    标签属性类。
    
    多重继承自ColumnConstraintKind和Property，表示Snowflake中的标签功能。
    标签用于数据治理、分类和元数据管理。
    """
    # expressions: 标签表达式列表
    arg_types = {"expressions": True}


class TransformModelProperty(Property):
    """
    转换模型属性类。
    
    表示机器学习场景中的数据转换模型配置。
    用于AI/ML数据库的模型管道管理。
    """
    # expressions: 转换模型表达式
    arg_types = {"expressions": True}


class TransientProperty(Property):
    """
    瞬态属性类。
    
    表示Snowflake等数据库中的瞬态表配置。
    瞬态表不参与Time Travel，节省存储成本。
    """
    # this: 是否为瞬态表
    arg_types = {"this": False}


class UnloggedProperty(Property):
    """
    非日志属性类。
    
    表示PostgreSQL等数据库中的非日志表配置。
    非日志表不写WAL，性能更高但不支持复制。
    """
    arg_types = {}


# 参考：https://docs.snowflake.com/en/sql-reference/sql/create-table#create-table-using-template
class UsingTemplateProperty(Property):
    """
    使用模板属性类。
    
    表示Snowflake中的模板表创建功能。
    使用现有表作为模板创建新表。
    """
    # this: 模板表名
    arg_types = {"this": True}


# 参考：https://learn.microsoft.com/en-us/sql/t-sql/statements/create-view-transact-sql?view=sql-server-ver16
class ViewAttributeProperty(Property):
    """
    视图属性类。
    
    表示SQL Server等数据库中的视图属性配置。
    如ENCRYPTION、SCHEMABINDING等视图选项。
    """
    # this: 视图属性类型
    arg_types = {"this": True}


class VolatileProperty(Property):
    """
    易变属性类。
    
    表示Teradata等数据库中的易变表配置。
    易变表用于临时数据存储，会话结束时删除。
    """
    # this: 是否为易变表
    arg_types = {"this": False}


class WithDataProperty(Property):
    """
    带数据属性类。
    
    表示创建物化视图时是否包含数据。
    WITH DATA表示立即填充数据，WITH NO DATA表示只创建结构。
    """
    # no: 是否不包含数据，statistics: 是否包含统计信息
    arg_types = {"no": True, "statistics": False}


class WithJournalTableProperty(Property):
    """
    带日志表属性类。
    
    表示Teradata等数据库中的日志表配置。
    日志表记录对主表的所有更改操作。
    """
    # this: 日志表名
    arg_types = {"this": True}


class WithSchemaBindingProperty(Property):
    """
    带模式绑定属性类。
    
    表示SQL Server中的SCHEMABINDING选项。
    模式绑定防止修改被视图引用的对象结构。
    """
    # this: 是否启用模式绑定
    arg_types = {"this": True}


class WithSystemVersioningProperty(Property):
    """
    带系统版本属性类。
    
    表示SQL Server中的系统版本时态表配置。
    时态表自动跟踪数据的历史版本。
    """
    arg_types = {
        "on": False,              # 是否启用系统版本
        "this": False,            # 版本配置
        "data_consistency": False, # 数据一致性检查
        "retention_period": False, # 历史数据保留期
        "with": True,             # WITH选项（必需）
    }


class WithProcedureOptions(Property):
    """
    带过程选项属性类。
    
    表示存储过程的执行选项配置。
    控制过程的执行行为和特性。
    """
    # expressions: 过程选项表达式列表
    arg_types = {"expressions": True}


class EncodeProperty(Property):
    """
    编码属性类。
    
    表示Amazon Redshift等数据库中的列编码配置。
    编码影响数据压缩和查询性能。
    """
    arg_types = {
        "this": True,        # 编码类型
        "properties": False, # 编码属性
        "key": False,        # 编码键
    }


class IncludeProperty(Property):
    """
    包含属性类。
    
    表示SQL Server等数据库中的INCLUDE列配置。
    INCLUDE列包含在索引中但不参与排序。
    """
    arg_types = {
        "this": True,        # 包含列名
        "alias": False,      # 列别名
        "column_def": False, # 列定义
    }


class ForceProperty(Property):
    """
    强制属性类。
    
    表示强制执行某种行为的属性配置。
    用于覆盖默认行为或安全检查。
    """
    arg_types = {}

class TablespaceProperty(Property):
    """
    表空间属性类。
    
    表示PostgreSQL、Oracle等数据库中的表空间配置。
    表空间控制数据的物理存储位置。
    """
    # this: 表空间名称
    arg_types = {"this": True}

class ServerProperty(Property):
    """
    服务器属性类。
    
    表示GaussDB中的服务器配置。
    服务器控制数据的物理存储位置。
    """
    # this: 服务器名称
    arg_types = {"this": True}

class Properties(Expression):
    """
    属性集合表达式类。
    
    表示数据库对象的属性集合，统一管理所有属性配置。
    提供属性名称到属性类的映射，支持动态属性解析。
    """
    # expressions: 属性表达式列表
    arg_types = {"expressions": True}

    # 属性名称到属性类的映射表
    # 这个映射表实现了字符串属性名到具体属性类的转换
    NAME_TO_PROPERTY = {
        "ALGORITHM": AlgorithmProperty,
        "AUTO_INCREMENT": AutoIncrementProperty,
        "CHARACTER SET": CharacterSetProperty,
        "CLUSTERED_BY": ClusteredByProperty,
        "COLLATE": CollateProperty,
        "COMMENT": SchemaCommentProperty,
        "CREDENTIALS": CredentialsProperty,
        "DEFINER": DefinerProperty,
        "DISTKEY": DistKeyProperty,
        "DISTRIBUTED_BY": DistributedByProperty,
        "DISTSTYLE": DistStyleProperty,
        "ENGINE": EngineProperty,
        "EXECUTE AS": ExecuteAsProperty,
        "FORMAT": FileFormatProperty,
        "LANGUAGE": LanguageProperty,
        "LOCATION": LocationProperty,
        "LOCK": LockProperty,
        "PARTITIONED_BY": PartitionedByProperty,
        "RETURNS": ReturnsProperty,
        "ROW_FORMAT": RowFormatProperty,
        "SORTKEY": SortKeyProperty,
        "ENCODE": EncodeProperty,
        "INCLUDE": IncludeProperty,
        "TABLESPACE": TablespaceProperty,
        "SERVER": ServerProperty,
    }

    # 反向映射：属性类到属性名称
    # 使用字典推导式自动生成反向映射，保持数据一致性
    PROPERTY_TO_NAME = {v: k for k, v in NAME_TO_PROPERTY.items()}

    # CREATE语句中属性的位置枚举
    # 定义了属性在CREATE语句中可以出现的位置
    # 
    # 形式1：指定模式
    #   create [POST_CREATE]
    #     table a [POST_NAME]
    #     (b int) [POST_SCHEMA]
    #     with ([POST_WITH])
    #     index (b) [POST_INDEX]
    #
    # 形式2：别名选择
    #   create [POST_CREATE]
    #     table a [POST_NAME]
    #     as [POST_ALIAS] (select * from b) [POST_EXPRESSION]
    #     index (c) [POST_INDEX]
    class Location(AutoName):
        POST_CREATE = auto()     # CREATE关键字后
        POST_NAME = auto()       # 对象名称后
        POST_SCHEMA = auto()     # 模式定义后
        POST_WITH = auto()       # WITH关键字后
        POST_ALIAS = auto()      # AS关键字后
        POST_EXPRESSION = auto() # 表达式后
        POST_INDEX = auto()      # 索引定义后
        UNSUPPORTED = auto()     # 不支持的位置

    @classmethod
    def from_dict(cls, properties_dict: t.Dict) -> Properties:
        """
        从字典创建Properties对象的类方法。
        
        这个方法实现了从简单字典到复杂属性对象的转换：
        1. 遍历字典中的每个键值对
        2. 查找预定义的属性类映射
        3. 创建相应的属性对象
        4. 对于未知属性，创建通用Property对象
        
        这种设计既支持已知属性的强类型处理，又保持了对未知属性的兼容性。
        """
        expressions = []
        for key, value in properties_dict.items():
            # 尝试查找预定义的属性类（大小写不敏感）
            property_cls = cls.NAME_TO_PROPERTY.get(key.upper())
            if property_cls:
                # 使用特定的属性类创建对象
                expressions.append(property_cls(this=convert(value)))
            else:
                # 对于未知属性，创建通用Property对象
                # 使用字符串字面量作为属性名，保持原始信息
                expressions.append(Property(this=Literal.string(key), value=convert(value)))

        return cls(expressions=expressions)


class Qualify(Expression):
    """
    QUALIFY表达式类。
    
    表示SQL中的QUALIFY子句，用于过滤窗口函数的结果。
    QUALIFY是窗口函数的专用过滤器，类似于WHERE对普通列的过滤。
    
    例如：QUALIFY ROW_NUMBER() OVER (ORDER BY date) = 1
    用于获取每组中的第一行数据。
    """
    pass


class InputOutputFormat(Expression):
    """
    输入输出格式表达式类。
    
    表示数据处理中的输入和输出格式配置。
    主要用于ETL、数据导入导出等场景的格式控制。
    
    支持指定不同的输入格式和输出格式，实现数据格式转换。
    """
    arg_types = {
        "input_format": False,  # 输入数据格式
        "output_format": False, # 输出数据格式
    }


# 参考：https://www.ibm.com/docs/en/ias?topic=procedures-return-statement-in-sql
class Return(Expression):
    """
    RETURN表达式类。
    
    表示存储过程或函数中的RETURN语句。
    用于从存储过程或函数中返回值并终止执行。
    
    IBM DB2等数据库中的存储过程控制流语句。
    """
    pass


class Reference(Expression):
    """
    引用表达式类。
    
    表示对其他数据库对象的引用，如外键引用、约束引用等。
    用于建立对象之间的关联关系和依赖关系。
    
    引用是数据库完整性约束的基础机制。
    """
    arg_types = {
        "this": True,        # 引用目标（必需）
        "expressions": False, # 引用表达式列表
        "options": False,    # 引用选项（如ON DELETE CASCADE）
    }


class Tuple(Expression):
    """
    元组表达式类。
    
    表示SQL中的元组结构，如(a, b, c)。
    元组常用于多列比较、IN子句、EXISTS子句等场景。
    
    支持复杂的元组操作，如元组与集合的比较。
    """
    arg_types = {
        "expressions": False, # 元组中的表达式列表
    }

    def isin(
        self,
        *expressions: t.Any,
        query: t.Optional[ExpOrStr] = None,
        unnest: t.Optional[ExpOrStr] | t.Collection[ExpOrStr] = None,
        copy: bool = True,
        **opts,
    ) -> In:
        """
        创建元组的IN表达式。
        
        这个方法支持多种形式的IN操作：
        1. 元组与值列表的比较：(a,b) IN ((1,2), (3,4))
        2. 元组与子查询的比较：(a,b) IN (SELECT x,y FROM table)
        3. 元组与UNNEST的比较：(a,b) IN UNNEST(array)
        
        参数:
            *expressions: 要比较的值表达式
            query: 子查询表达式
            unnest: UNNEST表达式或表达式集合
            copy: 是否复制当前元组
            **opts: 解析选项
        
        返回:
            In: IN表达式对象
        """
        return In(
            # 复制当前元组作为IN的左操作数
            this=maybe_copy(self, copy),
            # 转换右侧的值表达式列表
            expressions=[convert(e, copy=copy) for e in expressions],
            # 处理子查询（如果提供）
            query=maybe_parse(query, copy=copy, **opts) if query else None,
            # 处理UNNEST表达式
            unnest=(
                Unnest(
                    expressions=[
                        # 确保unnest是列表，然后解析每个表达式
                        maybe_parse(t.cast(ExpOrStr, e), copy=copy, **opts)
                        for e in ensure_list(unnest)
                    ]
                )
                if unnest
                else None
            ),
        )


# 查询修饰符配置字典
# 定义了所有可能的查询修饰符及其默认状态
# 这个字典是查询构建和验证的重要配置，确保查询结构的完整性
QUERY_MODIFIERS = {
    "match": False,      # MATCH子句（模式匹配）
    "laterals": False,   # LATERAL子查询
    "joins": False,      # JOIN连接
    "connect": False,    # CONNECT BY子句（Oracle层次查询）
    "pivots": False,     # PIVOT/UNPIVOT转换
    "prewhere": False,   # PREWHERE子句（ClickHouse预过滤）
    "where": False,      # WHERE过滤条件
    "group": False,      # GROUP BY分组
    "having": False,     # HAVING分组过滤
    "qualify": False,    # QUALIFY窗口函数过滤
    "windows": False,    # WINDOW窗口定义
    "distribute": False, # DISTRIBUTE BY分布（Hive）
    "sort": False,       # SORT BY排序（Hive）
    "cluster": False,    # CLUSTER BY聚集（Hive）
    "order": False,      # ORDER BY排序
    "limit": False,      # LIMIT行数限制
    "offset": False,     # OFFSET偏移量
    "locks": False,      # 锁定提示
    "sample": False,     # SAMPLE采样
    "settings": False,   # 设置选项
    "format": False,     # 格式化选项
    "options": False,    # 其他选项
}


# 参考：https://learn.microsoft.com/en-us/sql/t-sql/queries/option-clause-transact-sql?view=sql-server-ver16
# 参考：https://learn.microsoft.com/en-us/sql/t-sql/queries/hints-transact-sql-query?view=sql-server-ver16
class QueryOption(Expression):
    """
    查询选项表达式类。
    
    表示SQL Server中的OPTION子句和查询提示。
    用于控制查询优化器的行为和执行计划的生成。
    
    常见选项：
    - OPTION (RECOMPILE): 强制重新编译
    - OPTION (HASH JOIN): 强制使用哈希连接
    - OPTION (MAXDOP 4): 限制并行度
    """
    arg_types = {
        "this": True,        # 选项名称（必需）
        "expression": False, # 选项值表达式
    }


# 参考：https://learn.microsoft.com/en-us/sql/t-sql/queries/hints-transact-sql-table?view=sql-server-ver16
class WithTableHint(Expression):
    """
    表提示表达式类。
    
    表示SQL Server中的WITH表提示语法。
    表提示影响单个表的访问方式和锁定行为。
    
    常见提示：
    - WITH (NOLOCK): 读取时不加锁
    - WITH (INDEX(ix_name)): 强制使用指定索引
    - WITH (UPDLOCK): 更新锁定
    """
    arg_types = {
        "expressions": True, # 提示表达式列表
    }


# 参考：https://dev.mysql.com/doc/refman/8.0/en/index-hints.html
class IndexTableHint(Expression):
    """
    索引表提示表达式类。
    
    表示MySQL中的索引提示语法。
    用于指导MySQL优化器选择或避免特定的索引。
    
    提示类型：
    - USE INDEX: 建议使用指定索引
    - IGNORE INDEX: 忽略指定索引
    - FORCE INDEX: 强制使用指定索引
    """
    arg_types = {
        "this": True,        # 提示类型（USE/IGNORE/FORCE）
        "expressions": False, # 索引名称列表
        "target": False,     # 目标操作（JOIN/ORDER BY/GROUP BY）
    }


# 参考：https://docs.snowflake.com/en/sql-reference/constructs/at-before
class HistoricalData(Expression):
    """
    历史数据表达式类。
    
    表示Snowflake中的AT|BEFORE时间旅行语法。
    用于查询表或流的历史状态数据。
    
    时间旅行功能：
    - AT (TIMESTAMP => '2023-01-01'): 查询指定时间点的数据
    - BEFORE (STATEMENT => 'statement_id'): 查询语句执行前的数据
    - AT (OFFSET => -3600): 查询相对时间的数据
    """
    arg_types = {
        "this": True,       # 时间旅行目标（表名等）
        "kind": True,       # 时间类型（AT/BEFORE）
        "expression": True, # 时间表达式（TIMESTAMP/STATEMENT/OFFSET）
    }


# 参考：https://docs.snowflake.com/en/sql-reference/sql/put
class Put(Expression):
    """
    PUT表达式类。
    
    表示Snowflake中的PUT命令，用于将文件上传到内部阶段。
    PUT是Snowflake数据加载流程的第一步。
    
    典型流程：
    1. PUT: 上传文件到阶段
    2. COPY INTO: 从阶段加载数据到表
    
    例如：PUT file://data.csv @my_stage
    """
    arg_types = {
        "this": True,       # 源文件路径（必需）
        "target": True,     # 目标阶段（必需）
        "properties": False, # PUT选项（如AUTO_COMPRESS、PARALLEL等）
    }


# 参考：https://docs.snowflake.com/en/sql-reference/sql/get
class Get(Expression):
    """
    GET表达式类。
    
    表示Snowflake中的GET命令，用于从内部阶段下载文件。
    GET是PUT操作的逆向操作，用于文件的导出。
    
    典型用途：
    - 下载查询结果文件
    - 备份阶段中的文件
    - 将处理后的数据导出到本地
    
    例如：GET @my_stage/results.csv file://local_path/
    """
    arg_types = {
        "this": True,       # 源阶段路径（必需）
        "target": True,     # 目标本地路径（必需）
        "properties": False, # GET选项（如PATTERN等）
    }


class Table(Expression):
    """
    表表达式类。
    
    表示SQL中的表引用，包括表名、别名、数据库、模式等完整信息。
    这是SQL查询中最基础的数据源表示，支持复杂的表引用场景。
    
    支持的表引用形式：
    - 简单表名：table_name
    - 带数据库：db.table_name
    - 带模式：catalog.db.table_name
    - 带别名：table_name AS alias
    - 带提示：table_name WITH (NOLOCK)
    """
    arg_types = {
        "this": False,        # 表名（可选，可能是函数或子查询）
        "alias": False,       # 表别名
        "db": False,          # 数据库名
        "catalog": False,     # 目录/模式名
        "laterals": False,    # LATERAL连接列表
        "joins": False,       # JOIN连接列表
        "pivots": False,      # PIVOT操作列表
        "hints": False,       # 查询提示
        "system_time": False, # 系统时间（时间旅行）
        "version": False,     # 版本信息
        "format": False,      # 格式设置
        "pattern": False,     # 模式匹配
        "ordinality": False,  # 序数列（PostgreSQL等）
        "when": False,        # 条件子句
        "only": False,        # ONLY关键字（PostgreSQL继承）
        "partition": False,   # 分区信息
        "changes": False,     # 变更跟踪（SQL Server）
        "rows_from": False,   # ROWS FROM子句（PostgreSQL）
        "sample": False,      # 采样设置
    }

    @property
    def name(self) -> str:
        """
        获取表名。
        
        返回表的基本名称，不包括数据库和模式前缀。
        如果表引用是函数调用，则返回空字符串。
        """
        # 如果没有表名或者是函数调用，返回空字符串
        if not self.this or isinstance(self.this, Func):
            return ""
        return self.this.name

    @property
    def db(self) -> str:
        """获取数据库名。"""
        return self.text("db")

    @property
    def catalog(self) -> str:
        """获取目录/模式名。"""
        return self.text("catalog")

    @property
    def selects(self) -> t.List[Expression]:
        """
        获取选择列表。
        
        对于Table对象，返回空列表，因为表本身不包含选择列。
        子类（如子查询）可能重写此方法返回实际的选择列。
        """
        return []

    @property
    def named_selects(self) -> t.List[str]:
        """
        获取命名选择列表。
        
        返回选择列的名称列表，用于列名解析和验证。
        """
        return []

    @property
    def parts(self) -> t.List[Expression]:
        """
        按顺序返回表的组成部分：catalog.db.table。
        
        这个方法解析完整的表引用，将其分解为独立的组件。
        处理点号分隔的复杂表名，如 catalog.schema.table。
        """
        parts: t.List[Expression] = []

        # 按照catalog -> db -> this的顺序处理各部分
        for arg in ("catalog", "db", "this"):
            part = self.args.get(arg)

            # 如果是点号表达式，需要展平处理
            if isinstance(part, Dot):
                # flatten()方法将嵌套的点号表达式展开为平坦列表
                # 例如：a.b.c 展开为 [a, b, c]
                parts.extend(part.flatten())
            elif isinstance(part, Expression):
                # 普通表达式直接添加
                parts.append(part)

        return parts

    def to_column(self, copy: bool = True) -> Expression:
        """
        将表引用转换为列引用。
        
        这个方法用于将表名转换为对应的列引用，常用于：
        1. SELECT * 展开时需要获取表的所有列
        2. 列名解析时需要确定列所属的表
        3. 别名处理时需要应用表别名到列名
        
        参数:
            copy: 是否创建新的表达式副本
            
        返回:
            Expression: 对应的列表达式
        """
        parts = self.parts
        last_part = parts[-1]

        # 判断最后一部分是否为标识符（正常的表名）
        if isinstance(last_part, Identifier):
            # 构建列引用：反转parts顺序，因为column()函数期望从表名到catalog的顺序
            # parts[0:4]：取前4个部分（catalog, db, table, column）
            # parts[4:]：剩余部分作为字段路径（用于嵌套结构）
            col: Expression = column(*reversed(parts[0:4]), fields=parts[4:], copy=copy)  # type: ignore
        else:
            # 如果最后一部分是函数或数组，直接使用
            # 这种情况下表引用实际上是一个表达式（如函数调用）
            col = last_part

        # 处理表别名：如果表有别名，将别名应用到列上
        alias = self.args.get("alias")
        if alias:
            # 使用alias_()函数创建带别名的列表达式
            col = alias_(col, alias.this, copy=copy)

        return col


class SetOperation(Query):
    """
    集合操作表达式基类。
    
    表示SQL中的集合操作，如UNION、EXCEPT、INTERSECT。
    集合操作用于组合多个查询的结果集，是SQL中重要的数据组合机制。
    
    所有集合操作都有类似的结构：
    - 左查询 OPERATION 右查询
    - 可选的DISTINCT/ALL修饰符
    - 可选的ORDER BY等查询修饰符
    """
    arg_types = {
        "with": False,        # WITH子句（CTE）
        "this": True,         # 左查询（必需）
        "expression": True,   # 右查询（必需）
        "distinct": False,    # DISTINCT修饰符
        "by_name": False,     # BY NAME选项（某些数据库）
        "side": False,        # 侧向信息
        "kind": False,        # 操作类型信息
        "on": False,          # ON条件（某些特殊情况）
        **QUERY_MODIFIERS,    # 继承所有查询修饰符
    }

    def select(
        self: S,
        *expressions: t.Optional[ExpOrStr],
        append: bool = True,
        dialect: DialectType = None,
        copy: bool = True,
        **opts,
    ) -> S:
        """
        为集合操作的两个查询同时添加选择列。
        
        这个方法确保集合操作的左右两个查询具有相同的列结构，
        这是SQL集合操作的基本要求：参与操作的查询必须有相同数量和兼容类型的列。
        
        参数:
            *expressions: 要添加的选择表达式
            append: 是否追加到现有选择列（True）还是替换（False）
            dialect: SQL方言
            copy: 是否创建副本
            **opts: 其他选项
            
        返回:
            S: 修改后的集合操作对象
        """
        # 创建副本以避免修改原对象
        this = maybe_copy(self, copy)
        
        # 为左查询添加选择列
        # unnest()确保获取实际的查询对象（而不是嵌套的集合操作）
        this.this.unnest().select(*expressions, append=append, dialect=dialect, copy=False, **opts)
        
        # 为右查询添加相同的选择列，保持列结构一致
        this.expression.unnest().select(
            *expressions, append=append, dialect=dialect, copy=False, **opts
        )
        return this

    @property
    def named_selects(self) -> t.List[str]:
        """
        获取命名选择列列表。
        
        集合操作的列名由左查询决定，这符合SQL标准。
        右查询的列会自动对应到左查询的列位置。
        """
        return self.this.unnest().named_selects

    @property
    def is_star(self) -> bool:
        """
        检查是否包含星号选择。
        
        如果左查询或右查询任一包含SELECT *，则认为整个集合操作包含星号选择。
        这个信息用于查询分析和优化。
        """
        return self.this.is_star or self.expression.is_star

    @property
    def selects(self) -> t.List[Expression]:
        """
        获取选择表达式列表。
        
        返回左查询的选择列，因为集合操作的结果结构由左查询定义。
        """
        return self.this.unnest().selects

    @property
    def left(self) -> Query:
        """获取左查询。"""
        return self.this

    @property
    def right(self) -> Query:
        """获取右查询。"""
        return self.expression

    @property
    def kind(self) -> str:
        """
        获取操作类型。
        
        返回集合操作的具体类型（UNION/EXCEPT/INTERSECT），
        转换为大写以保持一致性。
        """
        return self.text("kind").upper()

    @property
    def side(self) -> str:
        """
        获取侧向信息。
        
        某些数据库可能有额外的侧向信息，转换为大写处理。
        """
        return self.text("side").upper()


class Union(SetOperation):
    """
    UNION表达式类。
    
    表示SQL中的UNION集合操作，用于合并两个查询的结果集。
    UNION会去除重复行，UNION ALL保留所有行。
    
    语法示例：
    - SELECT a FROM t1 UNION SELECT b FROM t2
    - SELECT a FROM t1 UNION ALL SELECT b FROM t2
    
    UNION是最常用的集合操作，用于数据合并和去重。
    """
    pass


class Except(SetOperation):
    """
    EXCEPT表达式类。
    
    表示SQL中的EXCEPT集合操作，用于从左查询结果中排除右查询的结果。
    也称为MINUS（在Oracle中）。
    
    语法示例：
    - SELECT a FROM t1 EXCEPT SELECT b FROM t2
    
    EXCEPT用于找出左集合中不在右集合中的元素。
    """
    pass


class Intersect(SetOperation):
    """
    INTERSECT表达式类。
    
    表示SQL中的INTERSECT集合操作，用于获取两个查询结果的交集。
    只返回同时存在于左右查询结果中的行。
    
    语法示例：
    - SELECT a FROM t1 INTERSECT SELECT b FROM t2
    
    INTERSECT用于找出两个集合的共同元素。
    """
    pass


class Update(DML):
    """
    UPDATE表达式类。
    
    表示SQL中的UPDATE语句，用于修改表中的现有数据。
    UPDATE是DML（数据操作语言）的核心操作之一，支持复杂的更新场景。
    
    支持的UPDATE语法特性：
    - 基本更新：UPDATE table SET col = value
    - 条件更新：UPDATE table SET col = value WHERE condition
    - 多表更新：UPDATE table SET col = value FROM other_table
    - CTE更新：WITH cte AS (...) UPDATE table SET col = value
    - 返回结果：UPDATE table SET col = value RETURNING *
    """
    arg_types = {
        "with": False,        # WITH子句（CTE公共表表达式）
        "this": False,        # 要更新的目标表
        "expressions": True,  # SET表达式列表（必需）
        "from": False,        # FROM子句（多表更新）
        "where": False,       # WHERE条件子句
        "returning": False,   # RETURNING子句
        "order": False,       # ORDER BY子句（MySQL等支持）
        "limit": False,       # LIMIT子句（MySQL等支持）
        "options": False,
    }

    def table(
        self, expression: ExpOrStr, dialect: DialectType = None, copy: bool = True, **opts
    ) -> Update:
        """
        设置要更新的目标表。
        
        这是UPDATE语句构建的第一步，指定数据修改的目标。
        支持简单表名、完全限定名、别名等各种表引用形式。

        示例:
            >>> Update().table("my_table").set_("x = 1").sql()
            'UPDATE my_table SET x = 1'

        参数:
            expression: 要解析的SQL代码字符串
                如果传入Table实例，则直接使用
                如果传入其他Expression实例，则包装为Table
            dialect: 用于解析输入表达式的SQL方言
            copy: 如果为False，则就地修改此表达式实例
            opts: 用于解析输入表达式的其他选项

        返回:
            Update: 修改后的Update表达式
        """
        # 使用_apply_builder统一处理表达式构建
        # arg="this"指定将表达式存储在this参数中
        # into=Table确保结果被包装为Table对象
        return _apply_builder(
            expression=expression,
            instance=self,
            arg="this",           # 存储在this参数中
            into=Table,           # 包装为Table类型
            prefix=None,          # 无前缀
            dialect=dialect,
            copy=copy,
            **opts,
        )

    def set_(
        self,
        *expressions: ExpOrStr,
        append: bool = True,
        dialect: DialectType = None,
        copy: bool = True,
        **opts,
    ) -> Update:
        """
        追加或设置SET表达式。
        
        SET子句是UPDATE语句的核心，定义了要修改的列和新值。
        支持多个赋值表达式，用逗号分隔。

        示例:
            >>> Update().table("my_table").set_("x = 1").sql()
            'UPDATE my_table SET x = 1'

        参数:
            *expressions: 要解析的SQL代码字符串
                如果传入Expression实例，则直接使用
                多个表达式用逗号组合
            append: 如果为True，将新表达式添加到现有SET表达式
                否则，重置表达式
            dialect: 用于解析输入表达式的SQL方言
            copy: 如果为False，则就地修改此表达式实例
            opts: 用于解析输入表达式的其他选项
        """
        # 使用_apply_list_builder处理表达式列表
        # 这确保多个SET表达式被正确组合
        return _apply_list_builder(
            *expressions,
            instance=self,
            arg="expressions",    # 存储在expressions参数中
            append=append,        # 控制是追加还是替换
            into=Expression,      # 保持为Expression类型
            prefix=None,          # 无前缀
            dialect=dialect,
            copy=copy,
            **opts,
        )

    def where(
        self,
        *expressions: t.Optional[ExpOrStr],
        append: bool = True,
        dialect: DialectType = None,
        copy: bool = True,
        **opts,
    ) -> Select:
        """
        追加或设置WHERE表达式。
        
        WHERE子句定义了更新操作的过滤条件，只有满足条件的行才会被更新。
        多个条件表达式会用AND操作符组合。

        示例:
            >>> Update().table("tbl").set_("x = 1").where("x = 'a' OR x < 'b'").sql()
            "UPDATE tbl SET x = 1 WHERE x = 'a' OR x < 'b'"

        参数:
            *expressions: 要解析的SQL代码字符串
                如果传入Expression实例，则直接使用
                多个表达式用AND操作符组合
            append: 如果为True，将新表达式AND到现有表达式
                否则，重置表达式
            dialect: 用于解析输入表达式的SQL方言
            copy: 如果为False，则就地修改此表达式实例
            opts: 用于解析输入表达式的其他选项

        返回:
            Select: 修改后的表达式（注：这里的返回类型标注可能有误）
        """
        # 使用_apply_conjunction_builder处理条件表达式
        # conjunction意味着多个条件用AND连接
        return _apply_conjunction_builder(
            *expressions,
            instance=self,
            arg="where",          # 存储在where参数中
            append=append,        # 控制是AND追加还是替换
            into=Where,           # 包装为Where对象
            dialect=dialect,
            copy=copy,
            **opts,
        )

    def from_(
        self,
        expression: t.Optional[ExpOrStr] = None,
        dialect: DialectType = None,
        copy: bool = True,
        **opts,
    ) -> Update:
        """
        设置FROM表达式。
        
        FROM子句允许UPDATE语句引用其他表，实现多表更新。
        这是某些数据库（如PostgreSQL、SQL Server）支持的高级特性。

        示例:
            >>> Update().table("my_table").set_("x = 1").from_("baz").sql()
            'UPDATE my_table SET x = 1 FROM baz'

        参数:
            expression: 要解析的SQL代码字符串
                如果传入From实例，则直接使用
                如果传入其他Expression实例，则包装为From
                如果不传入任何值，则不应用from到表达式
            dialect: 用于解析输入表达式的SQL方言
            copy: 如果为False，则就地修改此表达式实例
            opts: 用于解析输入表达式的其他选项

        返回:
            Update: 修改后的Update表达式
        """
        # 特殊处理：如果没有表达式，直接返回
        # 这允许条件性地添加FROM子句
        if not expression:
            return maybe_copy(self, copy)

        # 使用_apply_builder构建FROM子句
        return _apply_builder(
            expression=expression,
            instance=self,
            arg="from",           # 存储在from参数中
            into=From,            # 包装为From对象
            prefix="FROM",        # 添加FROM前缀
            dialect=dialect,
            copy=copy,
            **opts,
        )

    def with_(
        self,
        alias: ExpOrStr,
        as_: ExpOrStr,
        recursive: t.Optional[bool] = None,
        materialized: t.Optional[bool] = None,
        append: bool = True,
        dialect: DialectType = None,
        copy: bool = True,
        **opts,
    ) -> Update:
        """
        追加或设置公共表表达式（CTE）。
        
        WITH子句允许在UPDATE语句中定义临时命名的结果集，
        这些结果集可以在主UPDATE语句中引用，提高查询的可读性和复用性。

        示例:
            >>> Update().table("my_table").set_("x = 1").from_("baz").with_("baz", "SELECT id FROM foo").sql()
            'WITH baz AS (SELECT id FROM foo) UPDATE my_table SET x = 1 FROM baz'

        参数:
            alias: 要解析为表名的SQL代码字符串
                如果传入Expression实例，则直接使用
            as_: 要解析为表表达式的SQL代码字符串
                如果传入Expression实例，则直接使用
            recursive: 设置表达式的RECURSIVE部分，默认为False
            materialized: 设置表达式的MATERIALIZED部分
            append: 如果为True，添加到任何现有表达式
                否则，重置表达式
            dialect: 用于解析输入表达式的SQL方言
            copy: 如果为False，则就地修改此表达式实例
            opts: 用于解析输入表达式的其他选项

        返回:
            Update: 修改后的表达式
        """
        # 使用专用的CTE构建器处理WITH子句
        # CTE具有特殊的语法结构（alias AS (query)）
        return _apply_cte_builder(
            self,
            alias,                # CTE别名
            as_,                  # CTE查询定义
            recursive=recursive,  # 是否递归CTE
            materialized=materialized,  # 是否物化CTE
            append=append,        # 是否追加到现有CTE
            dialect=dialect,
            copy=copy,
            **opts,
        )


class Values(UDTF):
    """
    VALUES表达式类。
    
    表示SQL中的VALUES子句，用于构造表值构造器。
    VALUES继承自UDTF（用户定义表函数），因为它产生表格式结果。
    
    支持的VALUES用法：
    - 基本值列表：VALUES (1, 'a'), (2, 'b')
    - 插入语句：INSERT INTO table VALUES (1, 'a')
    - 子查询：SELECT * FROM (VALUES (1, 'a')) AS t(id, name)
    - 带别名：VALUES (1, 'a'), (2, 'b') AS t(id, name)
    """
    arg_types = {
        "expressions": True,  # 值表达式列表（必需）
        "alias": False,       # 表别名（可选）
        "order": False,
        "limit": False,
        "offset": False,
    }


class Var(Expression):
    """
    变量表达式类。
    
    表示SQL中的变量引用，如用户定义变量、会话变量等。
    不同数据库有不同的变量语法：
    - MySQL: @var_name, @@session.var_name
    - SQL Server: @var_name, @@GLOBAL.var_name
    - PostgreSQL: :var_name（在预处理语句中）
    
    变量用于存储临时值、参数传递、动态SQL构建等场景。
    """
    pass


class Version(Expression):
    """
    版本表达式类。
    
    表示时间旅行和版本控制功能，支持查询历史数据状态。
    这是现代云数据平台的重要特性，允许访问数据的历史快照。
    
    支持的时间旅行场景：
    - Iceberg: SELECT * FROM table FOR VERSION AS OF version_id
    - Delta Lake: SELECT * FROM table VERSION AS OF version_number
    - BigQuery: SELECT * FROM table FOR SYSTEM_TIME AS OF timestamp
    - SQL Server: SELECT * FROM table FOR SYSTEM_TIME AS OF datetime
    
    参考文档：
    - Trino Iceberg: https://trino.io/docs/current/connector/iceberg.html?highlight=snapshot#using-snapshots
    - Databricks Delta: https://www.databricks.com/blog/2019/02/04/introducing-delta-time-travel-for-large-scale-data-lakes.html
    - BigQuery: https://cloud.google.com/bigquery/docs/reference/standard-sql/query-syntax#for_system_time_as_of
    - SQL Server: https://learn.microsoft.com/en-us/sql/relational-databases/tables/querying-data-in-a-system-versioned-temporal-table?view=sql-server-ver16
    
    this参数：TIMESTAMP或VERSION标识符
    kind参数：时间旅行类型（"AS OF"、"BETWEEN"等）
    expression参数：具体的时间或版本值
    """
    arg_types = {
        "this": True,         # 时间戳或版本标识符（必需）
        "kind": True,         # 时间旅行类型："AS OF"或"BETWEEN"（必需）
        "expression": False,  # 具体的时间/版本值（可选，某些语法中内置在this中）
    }


class Schema(Expression):
    """
    模式表达式类。
    
    表示数据库模式定义，用于描述表结构、列定义等元数据信息。
    Schema在DDL语句、函数定义、存储过程等场景中用于指定数据结构。
    
    常见用途：
    - 表结构定义：CREATE TABLE时的列定义
    - 函数参数：存储过程的参数列表
    - 返回类型：表值函数的返回结构
    """
    arg_types = {
        "this": False,        # 模式名称或主体
        "expressions": False, # 模式表达式列表（如列定义）
    }


# 参考：https://dev.mysql.com/doc/refman/8.0/en/select.html
# 参考：https://docs.oracle.com/en/database/oracle/oracle-database/19/sqlrf/SELECT.html
class Lock(Expression):
    """
    锁定表达式类。
    
    表示SQL中的行锁定语法，如FOR UPDATE、FOR SHARE等。
    锁定语句用于在事务中显式控制行级锁定行为，确保数据一致性。
    
    支持的锁定类型：
    - FOR UPDATE：排他锁，阻止其他事务读取或修改
    - FOR SHARE：共享锁，允许其他事务读取但不能修改
    - FOR UPDATE NOWAIT：立即获取锁或失败
    - FOR UPDATE SKIP LOCKED：跳过已锁定的行
    
    MySQL和Oracle等数据库都支持这种锁定语法。
    """
    arg_types = {
        "update": True,       # 锁定类型：True为UPDATE，False为SHARE（必需）
        "expressions": False, # 要锁定的表或列列表
        "wait": False,        # 等待策略：WAIT/NOWAIT/SKIP LOCKED
        "key": False,         # 锁定键（某些数据库的特殊语法）
    }


class Select(Query):
    """
    SELECT查询表达式类。
    
    表示SQL中的SELECT语句，是最核心和最复杂的查询类型。
    SELECT继承自Query，具有完整的查询构建和修饰能力。
    
    支持的SELECT特性：
    - 基本查询：SELECT col FROM table
    - 复杂查询：WITH、JOIN、WHERE、GROUP BY、HAVING、ORDER BY
    - 高级特性：窗口函数、CTE、子查询、集合操作
    - 数据库特定：提示、锁定、分布式操作
    
    这是SQLGlot中最重要的表达式类，体现了SQL查询的完整能力。
    """
    arg_types = {
        "with": False,               # WITH子句（CTE公共表表达式）
        "kind": False,               # 查询类型（如RECURSIVE等）
        "expressions": False,        # SELECT列表（选择的字段）
        "hint": False,               # 查询提示（如/*+ HINT */）
        "distinct": False,           # DISTINCT去重
        "into": False,               # INTO子句（结果输出位置）
        "from": False,               # FROM子句（数据源）
        "operation_modifiers": False, # 操作修饰符
        **QUERY_MODIFIERS,           # 继承所有查询修饰符（23个）
    }

    def from_(
        self, expression: ExpOrStr, dialect: DialectType = None, copy: bool = True, **opts
    ) -> Select:
        """
        设置FROM表达式。
        
        FROM子句定义查询的数据源，是SELECT语句的基础组成部分。
        支持表、视图、子查询、表值函数等各种数据源类型。

        示例:
            >>> Select().from_("tbl").select("x").sql()
            'SELECT x FROM tbl'

        参数:
            expression: 要解析的SQL代码字符串
                如果传入From实例，则直接使用
                如果传入其他Expression实例，则包装为From
            dialect: 用于解析输入表达式的SQL方言
            copy: 如果为False，则就地修改此表达式实例
            opts: 用于解析输入表达式的其他选项

        返回:
            Select: 修改后的Select表达式
        """
        # 使用标准构建器模式设置FROM子句
        # prefix="FROM"确保生成正确的SQL语法
        return _apply_builder(
            expression=expression,
            instance=self,
            arg="from",           # 存储在from参数中
            into=From,            # 包装为From对象
            prefix="FROM",        # 添加FROM关键字
            dialect=dialect,
            copy=copy,
            **opts,
        )

    def group_by(
        self,
        *expressions: t.Optional[ExpOrStr],
        append: bool = True,
        dialect: DialectType = None,
        copy: bool = True,
        **opts,
    ) -> Select:
        """
        设置GROUP BY表达式。
        
        GROUP BY子句用于对查询结果进行分组，通常与聚合函数配合使用。
        分组是数据分析和报表生成的核心功能。

        示例:
            >>> Select().from_("tbl").select("x", "COUNT(1)").group_by("x").sql()
            'SELECT x, COUNT(1) FROM tbl GROUP BY x'

        参数:
            *expressions: 要解析的SQL代码字符串
                如果传入Group实例，则直接使用
                如果传入其他Expression实例，则包装为Group
                如果不传入任何参数，则不应用group by到表达式
            append: 如果为True，添加到任何现有表达式
                否则，将所有Group表达式展平为单个表达式
            dialect: 用于解析输入表达式的SQL方言
            copy: 如果为False，则就地修改此表达式实例
            opts: 用于解析输入表达式的其他选项

        返回:
            Select: 修改后的Select表达式
        """
        # 特殊处理：如果没有表达式，直接返回
        # 这允许条件性地添加GROUP BY子句
        if not expressions:
            return self if not copy else self.copy()

        # 使用子列表构建器处理GROUP BY表达式
        # 这允许多个分组条件的组合和展平
        return _apply_child_list_builder(
            *expressions,
            instance=self,
            arg="group",          # 存储在group参数中
            append=append,        # 控制是追加还是替换
            copy=copy,
            prefix="GROUP BY",    # 添加GROUP BY关键字
            into=Group,           # 包装为Group对象
            dialect=dialect,
            **opts,
        )

    def sort_by(
        self,
        *expressions: t.Optional[ExpOrStr],
        append: bool = True,
        dialect: DialectType = None,
        copy: bool = True,
        **opts,
    ) -> Select:
        """
        设置SORT BY表达式。
        
        SORT BY是Hive等大数据系统中的特殊排序语法，与ORDER BY不同。
        SORT BY只保证每个reducer内部的数据有序，而不保证全局有序，
        这在分布式计算中可以提供更好的性能。

        示例:
            >>> Select().from_("tbl").select("x").sort_by("x DESC").sql(dialect="hive")
            'SELECT x FROM tbl SORT BY x DESC'

        参数:
            *expressions: 要解析的SQL代码字符串
                如果传入Group实例，则直接使用
                如果传入其他Expression实例，则包装为SORT
            append: 如果为True，添加到任何现有表达式
                否则，将所有Order表达式展平为单个表达式
            dialect: 用于解析输入表达式的SQL方言
            copy: 如果为False，则就地修改此表达式实例
            opts: 用于解析输入表达式的其他选项

        返回:
            Select: 修改后的Select表达式
        """
        # 使用子列表构建器处理SORT BY表达式
        # 这是Hive等分布式系统的局部排序功能
        return _apply_child_list_builder(
            *expressions,
            instance=self,
            arg="sort",           # 存储在sort参数中
            append=append,        # 控制追加还是替换
            copy=copy,
            prefix="SORT BY",     # 添加SORT BY关键字
            into=Sort,            # 包装为Sort对象
            dialect=dialect,
            **opts,
        )

    def cluster_by(
        self,
        *expressions: t.Optional[ExpOrStr],
        append: bool = True,
        dialect: DialectType = None,
        copy: bool = True,
        **opts,
    ) -> Select:
        """
        设置CLUSTER BY表达式。
        
        CLUSTER BY是Hive等大数据系统中的数据聚集语法。
        它相当于DISTRIBUTE BY + SORT BY的组合，既控制数据分发又保证局部排序。
        数据会根据指定列的哈希值分发到不同的reducer，每个reducer内部按该列排序。

        示例:
            >>> Select().from_("tbl").select("x").cluster_by("x DESC").sql(dialect="hive")
            'SELECT x FROM tbl CLUSTER BY x DESC'

        参数:
            *expressions: 要解析的SQL代码字符串
                如果传入Group实例，则直接使用
                如果传入其他Expression实例，则包装为Cluster
            append: 如果为True，添加到任何现有表达式
                否则，将所有Order表达式展平为单个表达式
            dialect: 用于解析输入表达式的SQL方言
            copy: 如果为False，则就地修改此表达式实例
            opts: 用于解析输入表达式的其他选项

        返回:
            Select: 修改后的Select表达式
        """
        # 使用子列表构建器处理CLUSTER BY表达式
        # 这是Hive分布式计算中的数据聚集和排序功能
        return _apply_child_list_builder(
            *expressions,
            instance=self,
            arg="cluster",        # 存储在cluster参数中
            append=append,        # 控制追加还是替换
            copy=copy,
            prefix="CLUSTER BY",  # 添加CLUSTER BY关键字
            into=Cluster,         # 包装为Cluster对象
            dialect=dialect,
            **opts,
        )

    def select(
        self,
        *expressions: t.Optional[ExpOrStr],
        append: bool = True,
        dialect: DialectType = None,
        copy: bool = True,
        **opts,
    ) -> Select:
        """
        设置或追加SELECT列表表达式。
        
        SELECT列表定义查询要返回的列，是查询的核心输出定义。
        支持列名、表达式、函数、别名等各种选择类型。
        
        示例:
            >>> Select().select("x", "y", "COUNT(*) as cnt").from_("tbl").sql()
            'SELECT x, y, COUNT(*) AS cnt FROM tbl'
        
        参数:
            *expressions: 要选择的列表达式
            append: 是否追加到现有选择列（True）还是替换（False）
            dialect: SQL方言
            copy: 是否创建副本
            **opts: 其他解析选项
            
        返回:
            Select: 修改后的Select对象
        """
        # 使用列表构建器处理SELECT表达式
        # 这是查询构建的核心方法，处理所有输出列定义
        return _apply_list_builder(
            *expressions,
            instance=self,
            arg="expressions",    # 存储在expressions参数中
            append=append,        # 控制追加还是替换
            dialect=dialect,
            into=Expression,      # 保持Expression类型的灵活性
            copy=copy,
            **opts,
        )

    def lateral(
        self,
        *expressions: t.Optional[ExpOrStr],
        append: bool = True,
        dialect: DialectType = None,
        copy: bool = True,
        **opts,
    ) -> Select:
        """
        追加或设置LATERAL表达式。
        
        LATERAL是高级SQL特性，允许在FROM子句中引用前面表的列。
        在Hive中表现为LATERAL VIEW，用于展开数组或Map类型的数据。
        这是处理嵌套数据结构的重要工具。

        示例:
            >>> Select().select("x").lateral("OUTER explode(y) tbl2 AS z").from_("tbl").sql()
            'SELECT x FROM tbl LATERAL VIEW OUTER EXPLODE(y) tbl2 AS z'

        参数:
            *expressions: 要解析的SQL代码字符串
                如果传入Expression实例，则直接使用
            append: 如果为True，添加到任何现有表达式
                否则，重置表达式
            dialect: 用于解析输入表达式的SQL方言
            copy: 如果为False，则就地修改此表达式实例
            opts: 用于解析输入表达式的其他选项

        返回:
            Select: 修改后的Select表达式
        """
        # 使用列表构建器处理LATERAL表达式
        # LATERAL VIEW是Hive处理嵌套数据的重要功能
        return _apply_list_builder(
            *expressions,
            instance=self,
            arg="laterals",       # 存储在laterals参数中
            append=append,        # 控制追加还是替换
            into=Lateral,         # 包装为Lateral对象
            prefix="LATERAL VIEW", # 添加LATERAL VIEW关键字
            dialect=dialect,
            copy=copy,
            **opts,
        )

    def join(
        self,
        expression: ExpOrStr,
        on: t.Optional[ExpOrStr] = None,
        using: t.Optional[ExpOrStr | t.Collection[ExpOrStr]] = None,
        append: bool = True,
        join_type: t.Optional[str] = None,
        join_alias: t.Optional[Identifier | str] = None,
        dialect: DialectType = None,
        copy: bool = True,
        **opts,
    ) -> Select:
        """
        追加或设置JOIN表达式。
        
        JOIN是SQL中最重要的操作之一，用于连接多个表的数据。
        支持所有标准JOIN类型和连接条件语法。

        示例:
            >>> Select().select("*").from_("tbl").join("tbl2", on="tbl1.y = tbl2.y").sql()
            'SELECT * FROM tbl JOIN tbl2 ON tbl1.y = tbl2.y'

            >>> Select().select("1").from_("a").join("b", using=["x", "y", "z"]).sql()
            'SELECT 1 FROM a JOIN b USING (x, y, z)'

            使用 `join_type` 改变连接类型:

            >>> Select().select("*").from_("tbl").join("tbl2", on="tbl1.y = tbl2.y", join_type="left outer").sql()
            'SELECT * FROM tbl LEFT OUTER JOIN tbl2 ON tbl1.y = tbl2.y'

        参数:
            expression: 要解析的SQL代码字符串
                如果传入Expression实例，则直接使用
            on: 可选的连接"on"条件，作为SQL字符串
                如果传入Expression实例，则直接使用
            using: 可选的连接"using"条件，作为SQL字符串
                如果传入Expression实例，则直接使用
            append: 如果为True，添加到任何现有表达式
                否则，重置表达式
            join_type: 如果设置，改变解析的连接类型
            join_alias: 连接源的可选别名
            dialect: 用于解析输入表达式的SQL方言
            copy: 如果为False，则就地修改此表达式实例
            opts: 用于解析输入表达式的其他选项

        返回:
            Select: 修改后的表达式
        """
        # 准备解析参数，包含方言和其他选项
        parse_args: t.Dict[str, t.Any] = {"dialect": dialect, **opts}

        # 尝试解析表达式为JOIN对象
        # 首先尝试直接解析为JOIN，失败则尝试更宽松的解析
        try:
            expression = maybe_parse(expression, into=Join, prefix="JOIN", **parse_args)
        except ParseError:
            # 解析失败时，允许解析为Join或普通Expression
            expression = maybe_parse(expression, into=(Join, Expression), **parse_args)

        # 确保我们有一个Join对象
        join = expression if isinstance(expression, Join) else Join(this=expression)

        # 特殊处理：如果连接的是子查询，需要包装为subquery
        # 这确保生成正确的SQL语法：JOIN (SELECT ...) 而不是 JOIN SELECT ...
        if isinstance(join.this, Select):
            join.this.replace(join.this.subquery())

        # 处理连接类型：解析join_type字符串为具体的类型组件
        if join_type:
            method: t.Optional[Token]  # 连接方法（HASH, NESTED LOOP等）
            side: t.Optional[Token]    # 连接侧（LEFT, RIGHT, FULL）
            kind: t.Optional[Token]    # 连接种类（INNER, OUTER）

            # 解析连接类型字符串，获取各个组件
            method, side, kind = maybe_parse(join_type, into="JOIN_TYPE", **parse_args)  # type: ignore

            # 设置解析得到的连接类型组件
            if method:
                join.set("method", method.text)
            if side:
                join.set("side", side.text)
            if kind:
                join.set("kind", kind.text)

        # 处理ON条件：多个条件用AND连接
        if on:
            # ensure_list确保on是列表，and_将多个条件用AND连接
            on = and_(*ensure_list(on), dialect=dialect, copy=copy, **opts)
            join.set("on", on)

        # 处理USING条件：指定连接的列名列表
        if using:
            # 使用列表构建器处理USING列表
            join = _apply_list_builder(
                *ensure_list(using),
                instance=join,
                arg="using",          # 存储在using参数中
                append=append,
                copy=copy,
                into=Identifier,      # 转换为标识符
                **opts,
            )

        # 处理连接别名：为连接的表设置别名
        if join_alias:
            # 使用alias_函数为连接的表添加别名
            join.set("this", alias_(join.this, join_alias, table=True))

        # 将构建好的JOIN添加到SELECT的joins列表中
        return _apply_list_builder(
            join,
            instance=self,
            arg="joins",          # 存储在joins参数中
            append=append,        # 控制是追加还是替换
            copy=copy,
            **opts,
        )

    def having(
        self,
        *expressions: t.Optional[ExpOrStr],
        append: bool = True,
        dialect: DialectType = None,
        copy: bool = True,
        **opts,
    ) -> Select:
        """
        追加或设置HAVING表达式。
        
        HAVING子句用于对GROUP BY的结果进行过滤，类似于WHERE但作用于分组后的数据。
        HAVING中可以使用聚合函数，这是它与WHERE的主要区别。
        HAVING子句在SQL执行顺序中位于GROUP BY之后，ORDER BY之前。

        示例:
            >>> Select().select("x", "COUNT(y)").from_("tbl").group_by("x").having("COUNT(y) > 3").sql()
            'SELECT x, COUNT(y) FROM tbl GROUP BY x HAVING COUNT(y) > 3'

        参数:
            *expressions: 要解析的SQL代码字符串
                如果传入Expression实例，则直接使用
                多个表达式用AND操作符组合
            append: 如果为True，将新表达式AND到现有表达式
                否则，重置表达式
            dialect: 用于解析输入表达式的SQL方言
            copy: 如果为False，则就地修改此表达式实例
            opts: 用于解析输入表达式的其他选项

        返回:
            Select: 修改后的Select表达式
        """
        # 使用连接构建器处理HAVING条件
        # conjunction意味着多个条件用AND连接，这是HAVING的标准行为
        return _apply_conjunction_builder(
            *expressions,
            instance=self,
            arg="having",         # 存储在having参数中
            append=append,        # 控制AND追加还是替换
            into=Having,          # 包装为Having对象
            dialect=dialect,
            copy=copy,
            **opts,
        )

    def window(
        self,
        *expressions: t.Optional[ExpOrStr],
        append: bool = True,
        dialect: DialectType = None,
        copy: bool = True,
        **opts,
    ) -> Select:
        """
        设置或追加WINDOW表达式。
        
        WINDOW子句用于定义命名窗口，这些窗口可以在窗口函数中重复使用。
        这是SQL的高级特性，允许复杂的分析查询和窗口函数优化。
        
        示例:
            >>> Select().select("x", "ROW_NUMBER() OVER w").from_("tbl").window("w AS (ORDER BY x)").sql()
            'SELECT x, ROW_NUMBER() OVER w FROM tbl WINDOW w AS (ORDER BY x)'
        
        参数:
            *expressions: 窗口定义表达式
            append: 是否追加到现有窗口定义
            dialect: SQL方言
            copy: 是否创建副本
            **opts: 其他解析选项
            
        返回:
            Select: 修改后的Select对象
        """
        # 使用列表构建器处理WINDOW定义
        # 窗口定义允许复用复杂的窗口规范，提高查询性能和可读性
        return _apply_list_builder(
            *expressions,
            instance=self,
            arg="windows",        # 存储在windows参数中
            append=append,        # 控制追加还是替换
            into=Window,          # 包装为Window对象
            dialect=dialect,
            copy=copy,
            **opts,
        )

    def qualify(
        self,
        *expressions: t.Optional[ExpOrStr],
        append: bool = True,
        dialect: DialectType = None,
        copy: bool = True,
        **opts,
    ) -> Select:
        """
        设置或追加QUALIFY表达式。
        
        QUALIFY子句是窗口函数的专用过滤器，用于过滤窗口函数的结果。
        它类似于HAVING对聚合函数的作用，但专门针对窗口函数。
        QUALIFY在SQL执行顺序中位于窗口函数计算之后。
        
        示例:
            >>> Select().select("x", "ROW_NUMBER() OVER (ORDER BY x) as rn").from_("tbl").qualify("rn = 1").sql()
            'SELECT x, ROW_NUMBER() OVER (ORDER BY x) AS rn FROM tbl QUALIFY rn = 1'
        
        参数:
            *expressions: QUALIFY条件表达式
            append: 是否AND追加到现有条件
            dialect: SQL方言
            copy: 是否创建副本
            **opts: 其他解析选项
            
        返回:
            Select: 修改后的Select对象
        """
        # 使用连接构建器处理QUALIFY条件
        # QUALIFY是现代SQL的高级特性，专门用于窗口函数结果过滤
        return _apply_conjunction_builder(
            *expressions,
            instance=self,
            arg="qualify",        # 存储在qualify参数中
            append=append,        # 控制AND追加还是替换
            into=Qualify,         # 包装为Qualify对象
            dialect=dialect,
            copy=copy,
            **opts,
        )

    def distinct(
        self, *ons: t.Optional[ExpOrStr], distinct: bool = True, copy: bool = True
    ) -> Select:
        """
        设置DISTINCT表达式。
        
        DISTINCT用于去除查询结果中的重复行。
        可以是简单的DISTINCT（去除所有重复）或DISTINCT ON（基于特定列去重）。
        DISTINCT ON是PostgreSQL等数据库的扩展语法。

        示例:
            >>> Select().from_("tbl").select("x").distinct().sql()
            'SELECT DISTINCT x FROM tbl'

        参数:
            ons: 要去重的表达式（DISTINCT ON语法）
            distinct: 是否应该去重
            copy: 如果为False，则就地修改此表达式实例

        返回:
            Select: 修改后的表达式
        """
        # 创建实例副本（如果需要）
        instance = maybe_copy(self, copy)
        
        # 处理DISTINCT ON语法：如果有ons参数，创建元组包装
        # DISTINCT ON (col1, col2) 需要将列表达式包装在元组中
        on = Tuple(expressions=[maybe_parse(on, copy=copy) for on in ons if on]) if ons else None
        
        # 设置distinct属性：如果distinct为True则创建Distinct对象，否则为None
        instance.set("distinct", Distinct(on=on) if distinct else None)
        return instance

    def ctas(
        self,
        table: ExpOrStr,
        properties: t.Optional[t.Dict] = None,
        dialect: DialectType = None,
        copy: bool = True,
        **opts,
    ) -> Create:
        """
        将此表达式转换为CREATE TABLE AS语句。
        
        CTAS（Create Table As Select）是基于查询结果创建新表的便捷方法。
        新表的结构和数据都来自SELECT查询的结果。
        这是数据仓库和ETL流程中常用的操作。

        示例:
            >>> Select().select("*").from_("tbl").ctas("x").sql()
            'CREATE TABLE x AS SELECT * FROM tbl'

        参数:
            table: 要解析为表名的SQL代码字符串
                如果传入其他Expression实例，则直接使用
            properties: 可选的表属性映射字典
            dialect: 用于解析输入表的SQL方言
            copy: 如果为False，则就地修改此表达式实例
            opts: 用于解析输入表的其他选项

        返回:
            Create: 新的Create表达式
        """
        # 创建查询副本（如果需要）
        instance = maybe_copy(self, copy)
        
        # 解析表名表达式，确保包装为Table对象
        table_expression = maybe_parse(table, into=Table, dialect=dialect, **opts)

        # 处理表属性：如果提供了属性字典，转换为Properties对象
        properties_expression = None
        if properties:
            # 使用Properties.from_dict将字典转换为Properties表达式
            properties_expression = Properties.from_dict(properties)

        # 构建CREATE TABLE AS表达式
        return Create(
            this=table_expression,      # 目标表名
            kind="TABLE",               # 创建类型为TABLE
            expression=instance,        # SELECT查询作为数据源
            properties=properties_expression,  # 表属性
        )

    def lock(self, update: bool = True, copy: bool = True) -> Select:
        """
        为此表达式设置锁定读取模式。
        
        锁定读取用于在事务中显式控制行级锁定，确保数据一致性。
        FOR UPDATE提供排他锁，FOR SHARE提供共享锁。
        这在高并发环境中防止数据竞争条件非常重要。

        示例:
            >>> Select().select("x").from_("tbl").where("x = 'a'").lock().sql("mysql")
            "SELECT x FROM tbl WHERE x = 'a' FOR UPDATE"

            >>> Select().select("x").from_("tbl").where("x = 'a'").lock(update=False).sql("mysql")
            "SELECT x FROM tbl WHERE x = 'a' FOR SHARE"

        参数:
            update: 如果为True，锁定类型为FOR UPDATE，否则为FOR SHARE
            copy: 如果为False，则就地修改此表达式实例

        返回:
            Select: 修改后的表达式
        """
        # 创建实例副本（如果需要）
        inst = maybe_copy(self, copy)
        
        # 设置锁定：创建Lock对象并添加到locks列表
        # Lock对象的update参数控制是排他锁还是共享锁
        inst.set("locks", [Lock(update=update)])

        return inst

    def hint(self, *hints: ExpOrStr, dialect: DialectType = None, copy: bool = True) -> Select:
        """
        为此表达式设置查询提示。
        
        查询提示用于指导数据库优化器选择特定的执行计划。
        不同数据库有不同的提示语法，如Spark的BROADCAST、Oracle的INDEX等。
        提示可以显著影响查询性能，但需要谨慎使用。

        示例:
            >>> Select().select("x").from_("tbl").hint("BROADCAST(y)").sql(dialect="spark")
            'SELECT /*+ BROADCAST(y) */ x FROM tbl'

        参数:
            hints: 要解析为提示的SQL代码字符串
                如果传入Expression实例，则直接使用
            dialect: 用于解析提示的SQL方言
            copy: 如果为False，则就地修改此表达式实例

        返回:
            Select: 修改后的表达式
        """
        # 创建实例副本（如果需要）
        inst = maybe_copy(self, copy)
        
        # 设置提示：解析所有提示表达式并包装为Hint对象
        # 提示通常以 /*+ hint */ 形式出现在SELECT关键字后
        inst.set(
            "hint", Hint(expressions=[maybe_parse(h, copy=copy, dialect=dialect) for h in hints])
        )

        return inst

    @property
    def named_selects(self) -> t.List[str]:
        """
        获取命名选择列列表。
        
        返回所有有名称的选择列的输出名称列表。
        这些名称包括列别名、函数别名或原始列名。
        用于列名解析、结果集元数据生成等场景。
        
        返回:
            List[str]: 命名选择列的名称列表
        """
        # 遍历所有选择表达式，获取有名称的列的输出名称
        # output_name会返回别名或原始名称，alias_or_name确保列有名称
        selects = []

        for e in self.expressions:
            if e.alias_or_name:
                selects.append(e.output_name)
            elif isinstance(e, Aliases):
                selects.extend([a.name for a in e.aliases])
        return selects

    @property
    def is_star(self) -> bool:
        """
        检查是否包含星号选择。
        
        判断SELECT列表中是否包含 * 或 table.* 等星号表达式。
        星号选择意味着选择所有列，这在查询分析和优化中很重要。
        
        返回:
            bool: 如果包含星号选择则返回True
        """
        # 检查所有选择表达式中是否有星号表达式
        # any()函数只要有一个表达式是星号就返回True
        return any(expression.is_star for expression in self.expressions)

    @property
    def selects(self) -> t.List[Expression]:
        """
        获取选择表达式列表。
        
        返回SELECT子句中的所有表达式。
        这是查询输出定义的核心，包含列、函数、计算表达式等。
        
        返回:
            List[Expression]: 选择表达式列表
        """
        # 直接返回表达式列表，这是SELECT的核心内容
        return self.expressions


UNWRAPPED_QUERIES = (Select, SetOperation)


class Subquery(DerivedTable, Query):
    """
    子查询表达式类。
    
    表示SQL中的子查询，即括号包围的SELECT语句。
    子查询继承自DerivedTable（派生表）和Query（查询），具有完整的查询能力。
    
    支持的子查询场景：
    - FROM子查询：SELECT * FROM (SELECT * FROM t) AS sub
    - WHERE子查询：WHERE col IN (SELECT col FROM t)
    - SELECT子查询：SELECT (SELECT count FROM t2) FROM t1
    - 嵌套子查询：((SELECT * FROM t))
    
    子查询是SQL中最重要的复合结构之一，支持复杂的数据查询逻辑。
    """
    arg_types = {
        "this": True,         # 子查询的实际查询表达式（必需）
        "alias": False,       # 子查询别名
        "with": False,        # WITH子句（CTE）
        **QUERY_MODIFIERS,    # 继承所有查询修饰符
    }

    def unnest(self):
        """
        返回第一个非子查询表达式。
        
        这个方法用于"展开"嵌套的子查询结构，找到实际的查询内容。
        当有多层嵌套子查询时，递归向下查找直到找到实际的查询表达式。
        
        例如：((SELECT * FROM t)) -> SELECT * FROM t
        
        返回:
            Expression: 第一个非Subquery的表达式
        """
        expression = self
        # 循环展开嵌套的子查询，直到找到实际的查询内容
        # 这处理了形如 (((SELECT...))) 的多层嵌套情况
        while isinstance(expression, Subquery):
            expression = expression.this
        return expression

    def unwrap(self) -> Subquery:
        """
        展开包装器子查询。
        
        当子查询仅作为简单包装器时，向上查找到实际有意义的子查询。
        包装器子查询是指那些除了包含另一个表达式外没有其他内容的子查询。
        
        这个方法沿着父节点向上查找，直到找到非包装器的子查询。
        
        返回:
            Subquery: 展开后的实际子查询
        """
        expression = self
        # 向上遍历父节点，跳过简单的包装器子查询
        # same_parent确保在同一个父节点下，is_wrapper确保是包装器
        while expression.same_parent and expression.is_wrapper:
            # 向上移动到父节点，继续查找
            expression = t.cast(Subquery, expression.parent)
        return expression

    def select(
        self,
        *expressions: t.Optional[ExpOrStr],
        append: bool = True,
        dialect: DialectType = None,
        copy: bool = True,
        **opts,
    ) -> Subquery:
        """
        为子查询添加SELECT表达式。
        
        这个方法将SELECT操作代理到子查询的实际查询内容上。
        通过unnest()找到实际的查询，然后对其应用select操作。
        
        参数:
            *expressions: 要添加的SELECT表达式
            append: 是否追加到现有表达式
            dialect: SQL方言
            copy: 是否创建副本
            **opts: 其他选项
            
        返回:
            Subquery: 修改后的子查询
        """
        # 创建子查询副本（如果需要）
        this = maybe_copy(self, copy)
        
        # 找到实际的查询内容并应用select操作
        # copy=False避免二次复制，因为已经在上面处理了复制
        this.unnest().select(*expressions, append=append, dialect=dialect, copy=False, **opts)
        return this

    @property
    def is_wrapper(self) -> bool:
        """
        判断此子查询是否为简单包装器。
        
        包装器子查询是指除了包含另一个表达式外，没有任何其他属性的子查询。
        例如：SELECT * FROM (((SELECT * FROM t))) 中的内层括号就是包装器。
        
        示例:
            SELECT * FROM (((SELECT * FROM t)))
                          ^
                          这对应一个"包装器"子查询节点
                          
        返回:
            bool: 如果是包装器则返回True
        """
        # 检查除了"this"之外的所有参数是否都为None
        # 如果都为None，说明这个子查询只是简单包装了另一个表达式
        return all(v is None for k, v in self.args.items() if k != "this")

    @property
    def is_star(self) -> bool:
        """
        检查子查询是否包含星号选择。
        
        将星号检查代理到子查询的实际内容上。
        
        返回:
            bool: 如果包含星号选择则返回True
        """
        # 星号检查委托给子查询的实际内容
        return self.this.is_star

    @property
    def output_name(self) -> str:
        """
        获取子查询的输出名称。
        
        子查询的输出名称就是其别名，用于在外层查询中引用。
        
        返回:
            str: 子查询的别名
        """
        # 子查询的输出名称就是其别名
        return self.alias


class TableSample(Expression):
    """
    表采样表达式类。
    
    表示SQL中的表采样语法，用于从大表中随机采样数据子集。
    采样是大数据分析中的重要技术，可以在保持数据特征的同时大幅减少处理量。
    
    支持的采样方法：
    - 百分比采样：TABLESAMPLE(10 PERCENT)
    - 行数采样：TABLESAMPLE(1000 ROWS)
    - 大小采样：TABLESAMPLE(100M)
    - 桶采样：TABLESAMPLE BUCKET(1 OUT OF 10 ON col)
    - 种子采样：TABLESAMPLE(10 PERCENT) REPEATABLE(123)
    
    不同数据库有不同的采样语法，但核心概念相同。
    """
    arg_types = {
        "expressions": False,       # 采样表达式列表
        "method": False,            # 采样方法（BERNOULLI、SYSTEM等）
        "bucket_numerator": False,  # 桶采样的分子
        "bucket_denominator": False, # 桶采样的分母
        "bucket_field": False,      # 桶采样的字段
        "percent": False,           # 百分比采样值
        "rows": False,              # 行数采样值
        "size": False,              # 大小采样值
        "seed": False,              # 随机种子（REPEATABLE）
    }


class Tag(Expression):
    """
    标签表达式类。
    
    标签用于生成任意的SQL标记，如SELECT <span>x</span>。
    这是一个特殊的表达式类型，主要用于在SQL中嵌入HTML标签或其他标记。
    
    主要用途：
    - HTML报表生成：在SQL结果中嵌入HTML标签
    - 文档生成：为SQL元素添加标记信息
    - 调试标记：在复杂查询中添加标识
    """
    arg_types = {
        "this": False,    # 标签内容
        "prefix": False,  # 前缀标签
        "postfix": False, # 后缀标签
    }


# 参考：https://duckdb.org/docs/sql/statements/pivot
# 表示标准SQL PIVOT操作符和DuckDB的"简化"PIVOT语法
class Pivot(Expression):
    """
    PIVOT表达式类。
    
    表示SQL中的PIVOT和UNPIVOT操作，用于行列转换。
    PIVOT将行数据转换为列，UNPIVOT则相反。
    这是数据分析和报表生成中的重要功能。
    
    支持的PIVOT操作：
    - 标准PIVOT：将行转为列
    - UNPIVOT：将列转为行
    - DuckDB简化语法：更简洁的PIVOT语法
    - 包含空值处理：INCLUDE NULLS选项
    - 默认值设置：DEFAULT ON NULL选项
    """
    arg_types = {
        "this": False,           # PIVOT的目标表达式
        "alias": False,          # PIVOT结果的别名
        "expressions": False,    # PIVOT表达式列表
        "fields": False,         # PIVOT字段列表
        "unpivot": False,        # 是否为UNPIVOT操作
        "using": False,          # USING子句
        "group": False,          # GROUP BY子句
        "columns": False,        # 列定义
        "include_nulls": False,  # 是否包含空值
        "default_on_null": False, # 空值的默认处理
        "into": False,           # INTO子句（目标列）
    }

    @property
    def unpivot(self) -> bool:
        """
        检查是否为UNPIVOT操作。
        
        UNPIVOT是PIVOT的逆操作，将列数据转换为行。
        
        返回:
            bool: 如果是UNPIVOT操作则返回True
        """
        # 检查unpivot参数是否存在且为真值
        # bool()确保返回布尔值而不是其他类型
        return bool(self.args.get("unpivot"))

    @property
    def fields(self) -> t.List[Expression]:
        """
        获取PIVOT字段列表。
        
        字段列表定义了PIVOT操作中涉及的列。
        这些字段决定了数据转换的结构。
        
        返回:
            List[Expression]: PIVOT字段表达式列表
        """
        # 获取fields参数，如果不存在则返回空列表
        # 这确保总是返回列表类型，避免None值问题
        return self.args.get("fields", [])


# 参考：https://duckdb.org/docs/sql/statements/unpivot#simplified-unpivot-syntax
# UNPIVOT ... INTO [NAME <col_name> VALUE <col_value>][...,]
class UnpivotColumns(Expression):
    """
    UNPIVOT列表达式类。
    
    表示DuckDB简化UNPIVOT语法中的列定义。
    用于指定UNPIVOT操作的目标列名和值列名。
    
    语法格式：UNPIVOT ... INTO [NAME col_name VALUE col_value]
    
    这种简化语法使UNPIVOT操作更加直观和易用。
    """
    arg_types = {
        "this": True,        # 列名表达式（必需）
        "expressions": True, # 列定义表达式列表（必需）
    }


class Window(Condition):
    """
    窗口表达式类。
    
    表示SQL中的窗口函数定义，包括分区、排序和帧规范。
    窗口函数是现代SQL分析功能的核心，支持复杂的数据分析。
    
    窗口函数组成部分：
    - PARTITION BY：数据分区
    - ORDER BY：排序规则
    - 帧规范：ROWS/RANGE BETWEEN
    - 窗口别名：命名窗口重用
    
    继承自Condition，因为窗口可以在条件上下文中使用。
    """
    arg_types = {
        "this": True,        # 窗口名称或定义（必需）
        "partition_by": False, # PARTITION BY子句
        "order": False,      # ORDER BY子句
        "spec": False,       # 窗口帧规范
        "alias": False,      # 窗口别名
        "over": False,       # OVER关键字标识
        "first": False,      # FIRST_VALUE/LAST_VALUE标识
    }


class WindowSpec(Expression):
    """
    窗口规范表达式类。
    
    表示窗口函数的帧规范，定义计算窗口的范围。
    帧规范控制窗口函数在每一行计算时考虑哪些行。
    
    帧规范类型：
    - ROWS：基于行数的帧
    - RANGE：基于值范围的帧
    - GROUPS：基于分组的帧
    
    帧边界：
    - UNBOUNDED PRECEDING：无界前向
    - CURRENT ROW：当前行
    - UNBOUNDED FOLLOWING：无界后向
    - n PRECEDING/FOLLOWING：相对偏移
    """
    arg_types = {
        "kind": False,      # 帧类型（ROWS/RANGE/GROUPS）
        "start": False,     # 帧开始位置
        "start_side": False, # 开始边界方向（PRECEDING/FOLLOWING）
        "end": False,       # 帧结束位置
        "end_side": False,  # 结束边界方向（PRECEDING/FOLLOWING）
        "exclude": False,   # 排除选项（EXCLUDE CURRENT ROW等）
    }


class PreWhere(Expression):
    """
    PreWhere表达式类。
    
    表示ClickHouse中的PREWHERE子句，这是ClickHouse特有的优化功能。
    PREWHERE在WHERE之前执行，用于早期过滤数据，提高查询性能。
    
    PREWHERE与WHERE的区别：
    - 执行时机：PREWHERE在数据读取阶段执行
    - 性能优化：减少需要处理的数据量
    - 使用场景：适合高选择性的过滤条件
    
    这是ClickHouse等列式数据库的重要优化特性。
    """
    pass


class Where(Expression):
    """
    WHERE表达式类。
    
    表示SQL中的WHERE子句，用于过滤查询结果。
    WHERE是SQL查询的核心组成部分，定义数据筛选条件。
    
    WHERE子句特点：
    - 行级过滤：在GROUP BY之前执行
    - 条件表达式：支持复杂的布尔逻辑
    - 索引优化：数据库可以利用索引加速过滤
    - 早期执行：在查询处理的早期阶段执行
    
    WHERE是最常用的SQL子句之一。
    """
    pass


class Star(Expression):
    """
    星号表达式类。
    
    表示SQL中的星号(*)选择，用于选择所有列。
    星号是SQL中最常用的选择符，支持多种变体。
    
    星号变体：
    - 简单星号：SELECT *
    - 排除列：SELECT * EXCEPT (col1, col2)
    - 替换列：SELECT * REPLACE (expr AS col)
    - 重命名列：SELECT * RENAME (old AS new)
    
    这些变体在BigQuery、DuckDB等现代数据库中得到支持。
    """
    arg_types = {
        "except": False,  # EXCEPT子句（排除特定列）
        "replace": False, # REPLACE子句（替换特定列）
        "rename": False,  # RENAME子句（重命名特定列）
    }

    @property
    def name(self) -> str:
        """
        获取星号的名称。
        
        星号表达式的名称总是"*"，这是其标准表示。
        
        返回:
            str: 固定返回"*"
        """
        # 星号的名称总是"*"
        return "*"

    @property
    def output_name(self) -> str:
        """
        获取星号的输出名称。
        
        星号的输出名称与其名称相同，都是"*"。
        
        返回:
            str: 星号的输出名称
        """
        # 输出名称与名称相同
        return self.name


class Parameter(Condition):
    """
    参数表达式类。
    
    表示SQL中的命名参数，用于参数化查询和预处理语句。
    参数化查询是防止SQL注入和提高性能的重要技术。
    
    参数类型：
    - 命名参数：:param_name、$param_name
    - 位置参数：$1、$2、$3
    - 会话参数：@@session_var
    
    继承自Condition，因为参数可以在条件表达式中使用。
    """
    arg_types = {
        "this": True,        # 参数名称（必需）
        "expression": False, # 参数表达式（如默认值）
    }


class SessionParameter(Condition):
    """
    会话参数表达式类。
    
    表示数据库会话级别的参数，如系统变量和配置选项。
    会话参数控制查询行为和数据库配置。
    
    会话参数示例：
    - MySQL：@@session.sql_mode、@@global.max_connections
    - PostgreSQL：current_setting('timezone')
    - SQL Server：@@VERSION、@@SERVERNAME
    
    会话参数在动态SQL和数据库管理中很重要。
    """
    arg_types = {
        "this": True,   # 参数名称（必需）
        "kind": False,  # 参数类型（SESSION/GLOBAL等）
    }


# 参考：https://www.databricks.com/blog/parameterized-queries-pyspark
# 参考：https://jdbc.postgresql.org/documentation/query/#using-the-statement-or-preparedstatement-interface
class Placeholder(Condition):
    """
    占位符表达式类。
    
    表示参数化查询中的占位符，用于安全的动态SQL执行。
    占位符是现代数据库应用中防止SQL注入的标准方法。
    
    占位符类型：
    - JDBC风格：? 占位符
    - 命名占位符：:name、$name
    - 位置占位符：$1、$2、$3
    - 小组件占位符：用于BI工具的参数
    
    占位符在ETL、BI工具和应用程序中广泛使用。
    """
    arg_types = {
        "this": False,   # 占位符名称（可选）
        "kind": False,   # 占位符类型
        "widget": False, # 小组件类型（BI工具）
        "jdbc": False,   # JDBC标识符
    }

    @property
    def name(self) -> str:
        """
        获取占位符的名称。
        
        如果占位符有名称则返回名称，否则返回默认的"?"。
        这适用于不同类型的占位符语法。
        
        返回:
            str: 占位符名称或"?"
        """
        # 如果有名称则使用名称，否则使用默认的"?"
        # 这支持命名占位符和匿名占位符两种情况
        return self.this or "?"


class Null(Condition):
    """
    NULL值表达式类。
    
    表示SQL中的NULL值，这是SQL中表示缺失或未知数据的特殊值。
    NULL在SQL中有特殊的语义：任何与NULL的比较都返回UNKNOWN。
    
    NULL的特点：
    - 三值逻辑：TRUE、FALSE、UNKNOWN
    - 比较规则：NULL = NULL 返回UNKNOWN
    - 聚合函数：COUNT(*)包含NULL，COUNT(column)排除NULL
    - 排序规则：NULL在ORDER BY中的位置因数据库而异
    """
    arg_types: t.Dict[str, t.Any] = {}  # NULL不需要任何参数

    @property
    def name(self) -> str:
        """
        获取NULL的名称。
        
        NULL的名称总是"NULL"，这是其标准表示。
        
        返回:
            str: 固定返回"NULL"
        """
        # NULL的名称总是"NULL"
        return "NULL"

    def to_py(self) -> Lit[None]:
        """
        将NULL转换为Python的None值。
        
        这是SQL NULL到Python None的标准映射。
        
        返回:
            None: Python的None值
        """
        # SQL的NULL对应Python的None
        return None


class Boolean(Condition):
    """
    布尔值表达式类。
    
    表示SQL中的布尔值（TRUE/FALSE）。
    布尔值在条件表达式、WHERE子句、CASE语句等场景中广泛使用。
    
    布尔值特点：
    - 三值逻辑：TRUE、FALSE、NULL（UNKNOWN）
    - 逻辑运算：AND、OR、NOT
    - 比较结果：比较操作返回布尔值
    """
    def to_py(self) -> bool:
        """
        将布尔值转换为Python的bool类型。
        
        返回:
            bool: Python的布尔值
        """
        # 直接返回this属性，它应该包含布尔值
        return self.this


class DataTypeParam(Expression):
    """
    数据类型参数表达式类。
    
    表示数据类型定义中的参数，如精度、标度等。
    例如：DECIMAL(10,2)中的10和2就是数据类型参数。
    
    常见的数据类型参数：
    - 精度：DECIMAL(p,s)中的p
    - 标度：DECIMAL(p,s)中的s  
    - 长度：VARCHAR(n)中的n
    - 枚举值：ENUM('a','b','c')中的值列表
    """
    arg_types = {
        "this": True,        # 参数值（必需）
        "expression": False, # 参数表达式（可选）
    }

    @property
    def name(self) -> str:
        """
        获取数据类型参数的名称。
        
        参数名称来自this属性的name。
        
        返回:
            str: 参数名称
        """
        # 参数的名称来自this属性
        return self.this.name


# nullable参数在从其他方言转译到ClickHouse时很有用，因为ClickHouse默认假设非空类型。
# 值None和True表示类型可为空。
class DataType(Expression):
    """
    数据类型表达式类。
    
    表示SQL中的数据类型定义，这是数据库模式定义的核心。
    数据类型决定了数据的存储格式、取值范围和操作规则。
    
    支持的数据类型特性：
    - 基础类型：INT、VARCHAR、DATE等
    - 复合类型：ARRAY、STRUCT、MAP等
    - 精度控制：DECIMAL(10,2)、VARCHAR(255)等
    - 空值控制：NULLABLE类型修饰符
    - 用户定义类型：自定义数据类型支持
    
    这个类包含了几乎所有主流数据库支持的数据类型。
    """
    arg_types = {
        "this": True,        # 数据类型名称（必需）
        "expressions": False, # 类型参数列表（如精度、标度）
        "nested": False,     # 嵌套类型信息
        "values": False,     # 枚举值列表
        "prefix": False,     # 类型前缀
        "kind": False,       # 类型种类
        "nullable": False,   # 是否可为空（ClickHouse等数据库重要）
    }

    class Type(AutoName):
        """
        数据类型枚举类。
        
        定义了SQLGlot支持的所有数据类型。
        这个枚举涵盖了主流数据库的数据类型，包括：
        
        基础数据类型：
        - 整数类型：INT、BIGINT、SMALLINT等
        - 浮点类型：FLOAT、DOUBLE、DECIMAL等
        - 文本类型：VARCHAR、TEXT、CHAR等
        - 日期时间：DATE、TIME、TIMESTAMP等
        - 二进制：BLOB、BINARY、VARBINARY等
        
        复合数据类型：
        - 数组类型：ARRAY、LIST
        - 结构类型：STRUCT、OBJECT、NESTED
        - 映射类型：MAP
        - JSON类型：JSON、JSONB、VARIANT
        
        特殊类型：
        - 地理类型：GEOGRAPHY、GEOMETRY、POINT等
        - 网络类型：INET、IPADDRESS、IPV4、IPV6
        - 范围类型：DATERANGE、NUMRANGE等
        - 用户定义：USERDEFINED、UNKNOWN
        
        这个枚举的设计考虑了跨数据库兼容性，
        不同数据库的同类型可能有不同的名称。
        """
        ARRAY = auto()
        AGGREGATEFUNCTION = auto()
        SIMPLEAGGREGATEFUNCTION = auto()
        BIGDECIMAL = auto()
        BIGINT = auto()
        BIGSERIAL = auto()
        BINARY = auto()
        BIT = auto()
        BLOB = auto()
        BOOLEAN = auto()
        BPCHAR = auto()
        CHAR = auto()
        DATE = auto()
        DATE32 = auto()
        DATEMULTIRANGE = auto()
        DATERANGE = auto()
        DATETIME = auto()
        DATETIME2 = auto()
        DATETIME64 = auto()
        DECIMAL = auto()
        DECIMAL32 = auto()
        DECIMAL64 = auto()
        DECIMAL128 = auto()
        DECIMAL256 = auto()
        DOUBLE = auto()
        DYNAMIC = auto()
        ENUM = auto()
        ENUM8 = auto()
        ENUM16 = auto()
        FIXEDSTRING = auto()
        FLOAT = auto()
        GEOGRAPHY = auto()
        GEOGRAPHYPOINT = auto()
        GEOMETRY = auto()
        POINT = auto()
        RING = auto()
        LINESTRING = auto()
        MULTILINESTRING = auto()
        POLYGON = auto()
        MULTIPOLYGON = auto()
        HLLSKETCH = auto()
        HSTORE = auto()
        IMAGE = auto()
        INET = auto()
        INT = auto()
        INT128 = auto()
        INT256 = auto()
        INT4MULTIRANGE = auto()
        INT4RANGE = auto()
        INT8MULTIRANGE = auto()
        INT8RANGE = auto()
        INTERVAL = auto()
        IPADDRESS = auto()
        IPPREFIX = auto()
        IPV4 = auto()
        IPV6 = auto()
        JSON = auto()
        JSONB = auto()
        LIST = auto()
        LONGBLOB = auto()
        LONGTEXT = auto()
        LOWCARDINALITY = auto()
        MAP = auto()
        MEDIUMBLOB = auto()
        MEDIUMINT = auto()
        MEDIUMTEXT = auto()
        MONEY = auto()
        NAME = auto()
        NCHAR = auto()
        NESTED = auto()
        NOTHING = auto()
        NULL = auto()
        NUMMULTIRANGE = auto()
        NUMRANGE = auto()
        NVARCHAR = auto()
        OBJECT = auto()
        RANGE = auto()
        ROWVERSION = auto()
        SERIAL = auto()
        SET = auto()
        SMALLDATETIME = auto()
        SMALLINT = auto()
        SMALLMONEY = auto()
        SMALLSERIAL = auto()
        STRUCT = auto()
        SUPER = auto()
        TEXT = auto()
        TINYBLOB = auto()
        TINYTEXT = auto()
        TIME = auto()
        TIMETZ = auto()
        TIMESTAMP = auto()
        TIMESTAMPNTZ = auto()
        TIMESTAMPLTZ = auto()
        TIMESTAMPTZ = auto()
        TIMESTAMP_S = auto()
        TIMESTAMP_MS = auto()
        TIMESTAMP_NS = auto()
        TINYINT = auto()
        TSMULTIRANGE = auto()
        TSRANGE = auto()
        TSTZMULTIRANGE = auto()
        TSTZRANGE = auto()
        UBIGINT = auto()
        UINT = auto()
        UINT128 = auto()
        UINT256 = auto()
        UMEDIUMINT = auto()
        UDECIMAL = auto()
        UDOUBLE = auto()
        UNION = auto()
        UNKNOWN = auto()  # Sentinel value, useful for type annotation
        USERDEFINED = "USER-DEFINED"
        USMALLINT = auto()
        UTINYINT = auto()
        UUID = auto()
        VARBINARY = auto()
        VARCHAR = auto()
        VARIANT = auto()
        VECTOR = auto()
        XML = auto()
        YEAR = auto()
        TDIGEST = auto()
        # DB2特有数据类型
        LONG_VARCHAR = auto()
        DECFLOAT = auto()
        LONG_VARGRAPHIC = auto()
        # TERADATA特有数据类型
        BYTEINT = auto()
        

    STRUCT_TYPES = {
        Type.NESTED,    # ClickHouse嵌套类型
        Type.OBJECT,    # 通用对象类型
        Type.STRUCT,    # 结构体类型
        Type.UNION,     # 联合类型
    }

    # 数组类型集合：包含数组和列表类型
    ARRAY_TYPES = {
        Type.ARRAY,     # 标准数组类型
        Type.LIST,      # 列表类型
    }

    # 嵌套类型集合：包含所有嵌套结构类型
    NESTED_TYPES = {
        *STRUCT_TYPES,  # 展开结构类型
        *ARRAY_TYPES,   # 展开数组类型
        Type.MAP,       # 映射类型
    }

    # 文本类型集合：包含所有字符串类型
    TEXT_TYPES = {
        Type.CHAR,      # 固定长度字符
        Type.NCHAR,     # Unicode固定长度字符
        Type.NVARCHAR,  # Unicode可变长度字符
        Type.TEXT,      # 长文本
        Type.VARCHAR,   # 可变长度字符
        Type.NAME,      # 名称类型
    }

    # 有符号整数类型集合
    SIGNED_INTEGER_TYPES = {
        Type.BIGINT,    # 大整数
        Type.INT,       # 标准整数
        Type.INT128,    # 128位整数
        Type.INT256,    # 256位整数
        Type.MEDIUMINT, # 中等整数
        Type.SMALLINT,  # 小整数
        Type.TINYINT,   # 微整数
    }

    # 无符号整数类型集合
    UNSIGNED_INTEGER_TYPES = {
        Type.UBIGINT,   # 无符号大整数
        Type.UINT,      # 无符号整数
        Type.UINT128,   # 无符号128位整数
        Type.UINT256,   # 无符号256位整数
        Type.UMEDIUMINT,# 无符号中等整数
        Type.USMALLINT, # 无符号小整数
        Type.UTINYINT,  # 无符号微整数
    }

    # 整数类型集合：包含所有整数类型
    INTEGER_TYPES = {
        *SIGNED_INTEGER_TYPES,   # 展开有符号整数
        *UNSIGNED_INTEGER_TYPES, # 展开无符号整数
        Type.BIT,                # 位类型
    }

    # 浮点类型集合
    FLOAT_TYPES = {
        Type.DOUBLE,    # 双精度浮点
        Type.FLOAT,     # 单精度浮点
        Type.DECFLOAT,
    }

    # 实数类型集合：包含浮点和精确数值类型
    REAL_TYPES = {
        *FLOAT_TYPES,           # 展开浮点类型
        Type.BIGDECIMAL,        # 大精度小数
        Type.DECIMAL,           # 标准小数
        Type.DECIMAL32,         # 32位小数
        Type.DECIMAL64,         # 64位小数
        Type.DECIMAL128,        # 128位小数
        Type.DECIMAL256,        # 256位小数
        Type.DECFLOAT,
        Type.MONEY,             # 货币类型
        Type.SMALLMONEY,        # 小货币类型
        Type.UDECIMAL,          # 无符号小数
        Type.UDOUBLE,           # 无符号双精度
    }

    # 数值类型集合：包含所有数值类型
    NUMERIC_TYPES = {
        *INTEGER_TYPES, # 展开整数类型
        *REAL_TYPES,    # 展开实数类型
    }

    # 时间类型集合：包含所有日期时间类型
    TEMPORAL_TYPES = {
        Type.DATE,          # 日期
        Type.DATE32,        # 32位日期
        Type.DATETIME,      # 日期时间
        Type.DATETIME2,     # 日期时间2
        Type.DATETIME64,    # 64位日期时间
        Type.SMALLDATETIME, # 小日期时间
        Type.TIME,          # 时间
        Type.TIMESTAMP,     # 时间戳
        Type.TIMESTAMPNTZ,  # 无时区时间戳
        Type.TIMESTAMPLTZ,  # 本地时区时间戳
        Type.TIMESTAMPTZ,   # 带时区时间戳
        Type.TIMESTAMP_MS,  # 毫秒时间戳
        Type.TIMESTAMP_NS,  # 纳秒时间戳
        Type.TIMESTAMP_S,   # 秒时间戳
        Type.TIMETZ,        # 带时区时间
    }

    @classmethod
    def build(
        cls,
        dtype: DATA_TYPE,
        dialect: DialectType = None,
        udt: bool = False,
        copy: bool = True,
        **kwargs,
    ) -> DataType:
        """
        构建DataType对象。
        
        这是一个工厂方法，用于从各种输入创建DataType实例。
        支持字符串、枚举值、现有DataType对象等多种输入格式。
        
        参数:
            dtype: 要构建的数据类型
                可以是字符串、DataType.Type枚举、DataType对象等
            dialect: 用于解析dtype的SQL方言（当dtype是字符串时）
            udt: 当设置为True时，如果无法解析为DataType，
                则将dtype用作用户定义类型
            copy: 是否复制数据类型
            kwargs: 传递给DataType构造函数的其他参数
            
        返回:
            DataType: 构建的DataType对象
            
        异常:
            ValueError: 当dtype类型无效时
            ParseError: 当字符串解析失败且udt=False时
        """
        from sqlglot import parse_one

        # 处理字符串类型的dtype
        if isinstance(dtype, str):
            # 特殊处理UNKNOWN类型
            if dtype.upper() == "UNKNOWN":
                return DataType(this=DataType.Type.UNKNOWN, **kwargs)

            try:
                # 尝试解析字符串为DataType
                data_type_exp = parse_one(
                    dtype, read=dialect, into=DataType, error_level=ErrorLevel.IGNORE
                )
            except ParseError:
                # 解析失败时的处理
                if udt:
                    # 如果允许用户定义类型，创建USERDEFINED类型
                    return DataType(this=DataType.Type.USERDEFINED, kind=dtype, **kwargs)
                raise  # 否则重新抛出异常
        elif isinstance(dtype, (Identifier, Dot)) and udt:
            # 处理标识符类型的用户定义类型
            return DataType(this=DataType.Type.USERDEFINED, kind=dtype, **kwargs)
        elif isinstance(dtype, DataType.Type):
            # 处理枚举类型
            data_type_exp = DataType(this=dtype)
        elif isinstance(dtype, DataType):
            # 处理现有DataType对象
            return maybe_copy(dtype, copy)
        else:
            # 无效类型处理
            raise ValueError(f"Invalid data type: {type(dtype)}. Expected str or DataType.Type")

        # 合并参数并返回新实例
        return DataType(**{**data_type_exp.args, **kwargs})

    def is_type(self, *dtypes: DATA_TYPE, check_nullable: bool = False) -> bool:
        """
        检查此DataType是否匹配提供的任一数据类型。
        
        嵌套类型或精度使用"结构等价"语义进行比较，
        例如array<int> != array<float>。
        
        参数:
            dtypes: 要比较的数据类型列表
            check_nullable: 是否在比较中考虑NULLABLE类型构造器
                如果为false，则NULLABLE<INT>等价于INT
                
        返回:
            bool: 当且仅当存在匹配的数据类型时返回True
            
        比较逻辑:
            1. 对于复杂类型（有expressions）或需要检查nullable的情况：
               使用完整的结构比较（==）
            2. 对于用户定义类型：
               使用完整的结构比较（==）
            3. 对于简单类型：
               只比较基础类型（this）
        """
        # 获取当前类型的nullable状态
        self_is_nullable = self.args.get("nullable")
        
        # 遍历所有要比较的类型
        for dtype in dtypes:
            # 构建比较类型
            other_type = DataType.build(dtype, copy=False, udt=True)
            other_is_nullable = other_type.args.get("nullable")
            
            # 决定比较策略
            if (
                other_type.expressions  # 有类型参数，需要结构比较
                or (check_nullable and (self_is_nullable or other_is_nullable))  # 需要检查nullable
                or self.this == DataType.Type.USERDEFINED  # 用户定义类型
                or other_type.this == DataType.Type.USERDEFINED  # 用户定义类型
            ):
                # 使用完整的结构比较
                matches = self == other_type
            else:
                # 只比较基础类型
                matches = self.this == other_type.this

            # 找到匹配则立即返回True
            if matches:
                return True
                
        # 没有找到匹配
        return False



# https://www.postgresql.org/docs/15/datatype-pseudo.html
class PseudoType(DataType):
    """
    伪类型表达式类。
    
    表示PostgreSQL中的伪类型，这些是特殊的数据类型占位符。
    伪类型不能作为列的数据类型，但可以用作函数参数或返回值类型。
    
    常见的伪类型：
    - RECORD：表示函数返回复合类型
    - TRIGGER：用于触发器函数
    - VOID：表示函数无返回值
    - UNKNOWN：表示未确定的类型
    - ANYARRAY：表示任意数组类型
    - ANYELEMENT：表示任意元素类型
    
    伪类型在函数重载和多态函数中起重要作用。
    """
    arg_types = {"this": True}  # 伪类型名称（必需）


# 参考：https://www.postgresql.org/docs/15/datatype-oid.html
class ObjectIdentifier(DataType):
    """
    对象标识符数据类型类。
    
    表示PostgreSQL中的OID（对象标识符）数据类型。
    OID用于在PostgreSQL内部唯一标识数据库对象。
    
    OID相关类型：
    - OID：基本对象标识符
    - REGPROC：函数名到OID的映射
    - REGTYPE：类型名到OID的映射
    - REGCLASS：表名到OID的映射
    - REGCONFIG：文本搜索配置到OID的映射
    
    这些类型主要用于系统内部和管理功能。
    """
    arg_types = {"this": True}  # OID类型名称（必需）


# WHERE x <OP> EXISTS|ALL|ANY|SOME(SELECT ...)
class SubqueryPredicate(Predicate):
    """
    子查询谓词基类。
    
    表示SQL中使用子查询的谓词表达式，如EXISTS、ALL、ANY等。
    这些谓词用于将标量值与子查询结果进行比较。
    
    子查询谓词的特点：
    - 使用子查询作为比较对象
    - 返回布尔值（TRUE/FALSE/NULL）
    - 支持复杂的存在性和量化比较
    
    语法形式：value <operator> predicate(subquery)
    """
    pass


class All(SubqueryPredicate):
    """
    ALL子查询谓词类。
    
    表示SQL中的ALL谓词，用于与子查询的所有结果进行比较。
    只有当条件对子查询返回的所有行都成立时，ALL谓词才返回TRUE。
    
    语法示例：
    - WHERE salary > ALL(SELECT salary FROM employees WHERE dept = 'IT')
    - WHERE price >= ALL(SELECT min_price FROM products)
    
    ALL的逻辑：
    - 如果子查询返回空集，ALL返回TRUE
    - 如果所有比较都为TRUE，返回TRUE
    - 如果任何比较为FALSE，返回FALSE
    - 如果有NULL值且无FALSE，返回NULL
    """
    pass


class Any(SubqueryPredicate):
    """
    ANY/SOME子查询谓词类。
    
    表示SQL中的ANY或SOME谓词，用于与子查询的任一结果进行比较。
    只要条件对子查询返回的任何一行成立，ANY谓词就返回TRUE。
    
    语法示例：
    - WHERE salary > ANY(SELECT salary FROM employees WHERE dept = 'IT')
    - WHERE price <= SOME(SELECT max_price FROM products)
    
    ANY/SOME的逻辑：
    - 如果子查询返回空集，ANY返回FALSE
    - 如果任何比较为TRUE，返回TRUE
    - 如果所有比较都为FALSE，返回FALSE
    - 如果有NULL值且无TRUE，返回NULL
    
    注意：ANY和SOME是等价的，只是语法上的差异。
    """
    pass


# 与数据库或引擎交互的命令。对于大多数命令表达式，
# 我们将命令名称后面的内容解析为字符串。
class Command(Expression):
    """
    命令表达式类。
    
    表示数据库管理命令，如SHOW、DESCRIBE、EXPLAIN等。
    这些命令用于数据库管理、信息查询和系统操作。
    
    常见命令类型：
    - SHOW：显示数据库信息
    - DESCRIBE/DESC：描述表结构
    - EXPLAIN：显示执行计划
    - USE：切换数据库
    - SET：设置变量
    
    命令表达式的设计原则是简化解析，
    将命令参数作为字符串处理，避免复杂的语法解析。
    """
    arg_types = {
        "this": True,        # 命令名称（必需）
        "expression": False, # 命令参数表达式（可选）
    }


class Transaction(Expression):
    """
    事务表达式类。
    
    表示事务控制语句，如BEGIN、START TRANSACTION等。
    事务是数据库操作的基本单位，确保数据的一致性和完整性。
    
    事务特性（ACID）：
    - 原子性（Atomicity）：事务是不可分割的单位
    - 一致性（Consistency）：事务保持数据一致性
    - 隔离性（Isolation）：事务之间相互独立
    - 持久性（Durability）：事务提交后持久保存
    """
    arg_types = {
        "this": False,  # 事务类型（BEGIN/START等）
        "modes": False, # 事务模式（隔离级别、访问模式等）
        "mark": False,  # 保存点标记
    }


class Commit(Expression):
    """
    提交表达式类。
    
    表示事务提交语句，用于确认事务中的所有操作。
    COMMIT使事务中的所有更改永久生效。
    """
    arg_types = {
        "chain": False,      # 是否链接到新事务
        "this": False,       # 提交目标（通常为空）
        "durability": False, # 持久性选项
    }


class Rollback(Expression):
    """
    回滚表达式类。
    
    表示事务回滚语句，用于撤销事务中的所有操作。
    ROLLBACK将数据库状态恢复到事务开始前的状态。
    """
    arg_types = {
        "savepoint": False, # 保存点名称（部分回滚）
        "this": False,      # 回滚目标（通常为空）
    }


class Alter(Expression):
    """
    ALTER表达式类。
    
    表示SQL中的ALTER语句，用于修改数据库对象的结构。
    ALTER是DDL（数据定义语言）的核心命令，支持动态模式修改。
    
    支持的ALTER操作：
    - ALTER TABLE：修改表结构
    - ALTER INDEX：修改索引
    - ALTER VIEW：修改视图
    - ALTER FUNCTION：修改函数
    - ALTER PROCEDURE：修改存储过程
    """
    arg_types = {
        "this": False,     # 要修改的对象名称
        "kind": True,     # 对象类型（TABLE/INDEX/VIEW等）（必需）
        "actions": True,  # 修改操作列表（必需）
        "exists": False,  # IF EXISTS选项
        "only": False,    # ONLY选项（仅影响当前表，不包括继承表）
        "options": False, # 其他选项
        "cluster": False, # 集群相关选项
        "not_valid": False, # NOT VALID选项（PostgreSQL）
        "check": False,
        "cascade": False,
    }

    @property
    def kind(self) -> t.Optional[str]:
        """
        获取ALTER操作的对象类型。
        
        返回对象类型的大写形式，如TABLE、INDEX、VIEW等。
        """
        # 获取kind参数并转换为大写
        # 大写转换确保类型名称的一致性
        kind = self.args.get("kind")
        return kind and kind.upper()

    @property
    def actions(self) -> t.List[Expression]:
        """
        获取ALTER操作的动作列表。
        
        返回要执行的修改操作列表，如ADD COLUMN、DROP CONSTRAINT等。
        """
        # 返回actions列表，如果不存在则返回空列表
        # 这确保总是返回列表类型，便于迭代处理
        return self.args.get("actions") or []
    

class AlterSession(Expression):
    arg_types = {"expressions": True, "unset": False}


class Analyze(Expression):
    """
    ANALYZE表达式类。
    
    表示SQL中的ANALYZE语句，用于收集表的统计信息。
    统计信息帮助查询优化器选择最优的执行计划。
    """
    arg_types = {
        "kind": False,       # 分析类型
        "this": False,       # 要分析的表名
        "options": False,    # 分析选项
        "mode": False,       # 分析模式
        "partition": False,  # 分区信息
        "expression": False, # 分析表达式
        "properties": False, # 分析属性
    }


class AnalyzeStatistics(Expression):
    """
    统计信息分析表达式类。
    
    表示对数据库统计信息的专门分析操作。
    统计信息是查询优化器选择执行计划的重要依据。
    
    统计信息类型：
    - 表统计：行数、数据页数量、平均行长度
    - 列统计：唯一值数量、空值数量、数据分布
    - 索引统计：索引的选择性、键值分布
    
    这个类用于更新或重新计算特定的统计信息，
    帮助优化器更准确地估算查询成本和选择最优执行计划。
    """
    arg_types = {
        "kind": True,        # 统计类型（必需）- 指定要分析的统计信息类型
        "option": False,     # 统计选项（可选）- 分析的具体选项和参数
        "this": False,       # 目标对象（可选）- 要分析的表或索引
        "expressions": False, # 统计表达式列表（可选）- 具体的统计计算表达式
    }


class AnalyzeHistogram(Expression):
    """
    直方图分析表达式类。
    
    表示对列值分布的直方图分析。
    直方图提供比基本统计更详细的数据分布信息。
    
    直方图的作用：
    - 显示数据值的分布模式（均匀分布、正态分布、偏态分布等）
    - 帮助优化器更准确地估算选择性
    - 支持复杂查询的成本估算
    - 识别数据倾斜和异常值
    
    直方图分析对于大数据表和复杂查询的性能优化特别重要。
    """
    arg_types = {
        "this": True,           # 目标列（必需）- 要分析直方图的列名
        "expressions": True,    # 直方图表达式（必需）- 直方图构建的具体表达式
        "expression": False,    # 附加表达式（可选）- 额外的分析表达式
        "update_options": False, # 更新选项（可选）- 直方图更新的策略和选项
    }


class AnalyzeSample(Expression):
    """
    采样分析表达式类。
    
    表示使用采样数据进行统计分析。
    采样分析在大表上可以显著提高分析速度，减少资源消耗。
    
    采样策略：
    - 随机采样：随机选择数据子集，保证统计的代表性
    - 系统采样：按固定间隔选择数据，提高采样效率
    - 分层采样：按分区或组进行采样，保证各组的代表性
    - 自适应采样：根据数据特征动态调整采样策略
    
    采样分析在保持统计准确性的同时，大幅提升分析性能。
    """
    arg_types = {
        "kind": True,   # 采样类型（必需）- 指定采样的方法和策略
        "sample": True, # 采样规范（必需）- 采样的具体参数和配置
    }


class AnalyzeListChainedRows(Expression):
    """
    链式行分析表达式类。
    
    表示对链式行（chained rows）的分析。
    链式行是Oracle等数据库中由于行迁移产生的现象。
    
    链式行的产生原因：
    - 行更新导致数据增长，原数据块空间不足
    - 行被迁移到其他数据块，原位置保留指向新位置的指针
    - 多次迁移可能形成行链
    
    链式行的问题：
    - 影响查询性能：需要额外的I/O操作
    - 增加存储开销：指针占用额外空间
    - 降低缓存效率：数据分散在多个块中
    
    这个类用于分析和处理链式行，优化存储结构。
    """
    arg_types = {
        "expression": False, # 分析表达式（可选）- 链式行分析的具体表达式
    }


class AnalyzeDelete(Expression):
    """
    删除分析表达式类。
    
    表示分析删除操作或删除分析结果。
    用于清理过时的统计信息或分析删除操作的影响。
    
    删除分析的应用场景：
    - 清理过时的统计信息：删除不再需要的统计数据
    - 分析删除操作的影响：评估删除操作对性能的影响
    - 统计信息维护：定期清理和重建统计信息
    
    这个类支持统计信息生命周期管理，保持统计信息的时效性。
    """
    arg_types = {
        "kind": False, # 删除类型（可选）- 指定要删除的统计信息类型
    }


class AnalyzeWith(Expression):
    """
    WITH子句分析表达式类。
    
    表示带有WITH子句的分析操作。
    WITH子句可以指定分析的具体参数和选项。
    
    WITH子句的作用：
    - 提供分析参数：指定分析的具体选项和配置
    - 条件分析：根据条件执行不同的分析策略
    - 参数化分析：使用参数化的方式进行分析
    
    这个类支持灵活的分析配置，满足不同场景的分析需求。
    """
    arg_types = {
        "expressions": True, # WITH表达式列表（必需）- WITH子句中的表达式列表
    }


class AnalyzeValidate(Expression):
    """
    验证分析表达式类。
    
    表示分析验证操作，用于检查数据完整性和约束。
    验证分析确保数据质量和约束的有效性。
    
    验证类型：
    - 约束验证：检查外键、检查约束、唯一约束等
    - 结构验证：检查表结构的一致性和完整性
    - 数据验证：检查数据的完整性和一致性
    - 索引验证：检查索引的完整性和有效性
    
    验证分析是数据质量保证的重要环节，
    确保数据库的完整性和一致性。
    """
    arg_types = {
        "kind": True,        # 验证类型（必需）- 指定要执行的验证类型
        "this": False,       # 验证目标（可选）- 要验证的对象
        "expression": False, # 验证表达式（可选）- 具体的验证逻辑
    }


class AnalyzeColumns(Expression):
    """
    列分析表达式类。
    
    表示专门针对列的分析操作。
    用于分析特定列的统计信息和数据分布。
    
    列分析的内容：
    - 唯一值数量：统计列中不同值的数量
    - 空值比例：计算NULL值的占比
    - 数据分布模式：分析数据的分布特征
    - 最小值和最大值：确定数据的取值范围
    - 数据长度分布：分析字符串或二进制数据的长度分布
    - 数据模式：识别数据的常见模式和格式
    
    列分析为查询优化提供详细的列级统计信息，
    帮助优化器做出更准确的决策。
    """
    pass


class UsingData(Expression):
    """
    USING DATA表达式类。
    
    表示使用特定数据进行操作的语法结构。
    常用于统计分析、数据验证等场景。
    
    USING DATA的应用：
    - 指定分析使用的数据源：明确分析的数据范围
    - 控制操作的数据范围：限制操作影响的数据
    - 提供数据处理的上下文：为操作提供数据环境
    - 数据源选择：在多个数据源中选择特定的数据
    
    这个类支持灵活的数据源指定，
    为各种数据操作提供精确的数据控制。
    """
    pass


class AddConstraint(Expression):
    """
    添加约束表达式类。
    
    表示ALTER TABLE语句中的ADD CONSTRAINT操作。
    用于向现有表添加各种类型的约束。
    
    支持的约束类型：
    - PRIMARY KEY：主键约束
    - FOREIGN KEY：外键约束
    - UNIQUE：唯一约束
    - CHECK：检查约束
    - NOT NULL：非空约束
    
    约束的作用：
    - 保证数据完整性
    - 维护数据一致性
    - 提供数据验证规则
    - 支持引用完整性
    """
    arg_types = {"expressions": True}  # 约束表达式列表（必需）


class AddPartition(Expression):
    """
    添加分区表达式类。
    
    表示ALTER TABLE语句中的ADD PARTITION操作。
    用于向分区表添加新的分区。
    
    分区的作用：
    - 提高查询性能：只扫描相关分区
    - 简化数据管理：按分区进行维护
    - 支持并行处理：不同分区可并行操作
    - 优化存储：按分区存储和压缩
    
    分区类型：
    - 范围分区：按数值或日期范围分区
    - 列表分区：按离散值列表分区
    - 哈希分区：按哈希值分区
    """
    arg_types = {
        "this": True,      # 分区名称（必需）
        "exists": False,   # IF NOT EXISTS选项（可选）
        "location": False, # 分区存储位置（可选）
    }

# 支持GaussDB的分区 ADD PARTITION ... VALUES 语法
class AddGaussDBPartition(Expression):
    """
    GaussDB分区添加表达式类。
    
    表示GaussDB数据库特有的分区添加语法。
    GaussDB是华为云数据库，支持PostgreSQL兼容的分区语法。
    
    语法特点：
    - 支持VALUES子句指定分区值
    - 支持FOR VALUES子句指定分区范围
    - 兼容PostgreSQL分区语法
    - 支持IF NOT EXISTS选项
    
    应用场景：
    - 华为云GaussDB数据库
    - PostgreSQL兼容环境
    - 企业级数据仓库
    """
    arg_types = {
        "this": True,        # 分区名称（必需）
        "expressions": True, # 分区值列表（必需）
        "exists": False,     # IF NOT EXISTS选项（可选）
        "values": False,     # VALUES/FOR VALUES类型（可选）
    }

class AttachOption(Expression):
    """
    附加选项表达式类。
    
    表示数据库操作中的附加选项和参数。
    用于指定操作的具体行为和配置。
    
    常见选项类型：
    - 存储选项：指定存储引擎、压缩方式等
    - 性能选项：指定缓存、索引等性能相关设置
    - 安全选项：指定访问权限、加密等安全设置
    - 兼容性选项：指定语法兼容性设置
    
    这个类提供了灵活的选项配置机制。
    """
    arg_types = {
        "this": True,        # 选项名称（必需）
        "expression": False, # 选项值表达式（可选）
    }


class DropPartition(Expression):
    """
    删除分区表达式类。
    
    表示ALTER TABLE语句中的DROP PARTITION操作。
    用于从分区表中删除指定的分区。
    
    删除分区的考虑：
    - 数据备份：删除前需要备份重要数据
    - 依赖关系：检查是否有外键依赖
    - 性能影响：删除操作可能影响查询性能
    - 存储回收：释放分区占用的存储空间
    
    支持的分区删除方式：
    - 按名称删除：DROP PARTITION partition_name
    - 按值删除：DROP PARTITION (values)
    - 批量删除：DROP PARTITION partition1, partition2
    """
    arg_types = {
        "expressions": True, # 分区表达式列表（必需）
        "exists": False,     # IF EXISTS选项（可选）
    }


# 参考：https://clickhouse.com/docs/en/sql-reference/statements/alter/partition#replace-partition
class ReplacePartition(Expression):
    """
    替换分区表达式类。
    
    表示ClickHouse数据库中的REPLACE PARTITION操作。
    用于替换分区表中的分区数据。
    
    ClickHouse分区替换特点：
    - 原子操作：替换操作是原子的，要么全部成功要么全部失败
    - 高性能：直接替换分区文件，避免逐行操作
    - 数据一致性：保证替换过程中数据的一致性
    - 支持批量：可以同时替换多个分区
    
    应用场景：
    - 数据更新：用新数据替换旧分区
    - 数据迁移：将数据从一个表迁移到另一个表
    - 数据修复：修复损坏的分区数据
    """
    arg_types = {
        "expression": True, # 目标分区表达式（必需）
        "source": True,     # 源分区表达式（必需）
    }


# 二元表达式，如 (ADD a b)
class Binary(Condition):
    """
    二元表达式基类。
    
    表示具有两个操作数的表达式。
    二元表达式是SQL中最常见的表达式类型。
    
    二元表达式的特点：
    - 左操作数：第一个操作数
    - 右操作数：第二个操作数
    - 操作符：定义两个操作数之间的关系
    - 结果类型：根据操作符和操作数类型确定
    
    常见的二元表达式：
    - 算术运算：+、-、*、/、%
    - 比较运算：=、!=、<、>、<=、>=
    - 逻辑运算：AND、OR
    - 位运算：&、|、^、<<、>>
    """
    arg_types = {
        "this": True,      # 左操作数（必需）
        "expression": True, # 右操作数（必需）
    }

    @property
    def left(self) -> Expression:
        """
        获取左操作数。
        
        返回二元表达式的第一个操作数。
        这是对this属性的语义化访问。
        
        返回:
            Expression: 左操作数表达式
        """
        # 左操作数存储在this属性中
        return self.this

    @property
    def right(self) -> Expression:
        """
        获取右操作数。
        
        返回二元表达式的第二个操作数。
        这是对expression属性的语义化访问。
        
        返回:
            Expression: 右操作数表达式
        """
        # 右操作数存储在expression属性中
        return self.expression


class Add(Binary):
    """
    加法运算表达式类。
    
    表示SQL中的加法运算（+）。
    用于数值相加或字符串连接。
    
    加法运算的特点：
    - 数值加法：两个数值相加
    - 字符串连接：在某些数据库中支持字符串连接
    - 日期运算：日期加时间间隔
    - NULL处理：任何数与NULL相加结果为NULL
    """
    pass


class Connector(Binary):
    """
    连接器表达式类。
    
    表示SQL中的连接操作。
    用于连接不同的表达式或条件。
    
    连接器的应用：
    - 条件连接：WHERE子句中的条件连接
    - 表达式连接：复杂表达式的组合
    - 逻辑连接：AND、OR等逻辑连接
    """
    pass


class BitwiseAnd(Binary):
    """
    按位与运算表达式类。
    
    表示SQL中的按位与运算（&）。
    对两个整数的二进制表示进行按位与操作。
    
    按位与的特点：
    - 逐位比较：对每一位进行AND操作
    - 结果规则：1&1=1, 1&0=0, 0&1=0, 0&0=0
    - 应用场景：权限控制、标志位操作
    """
    pass


class BitwiseLeftShift(Binary):
    """
    左移位运算表达式类。
    
    表示SQL中的左移位运算（<<）。
    将二进制数向左移动指定位数。
    
    左移位的特点：
    - 二进制左移：相当于乘以2的n次方
    - 高位丢弃：超出范围的位被丢弃
    - 低位补零：右侧空位补0
    """
    pass


class BitwiseOr(Binary):
    """
    按位或运算表达式类。
    
    表示SQL中的按位或运算（|）。
    对两个整数的二进制表示进行按位或操作。
    
    按位或的特点：
    - 逐位比较：对每一位进行OR操作
    - 结果规则：1|1=1, 1|0=1, 0|1=1, 0|0=0
    - 应用场景：权限合并、标志位设置
    """
    pass


class BitwiseRightShift(Binary):
    """
    右移位运算表达式类。
    
    表示SQL中的右移位运算（>>）。
    将二进制数向右移动指定位数。
    
    右移位的特点：
    - 二进制右移：相当于除以2的n次方
    - 低位丢弃：超出范围的位被丢弃
    - 高位补零：左侧空位补0（逻辑右移）
    """
    pass


class BitwiseXor(Binary):
    """
    按位异或运算表达式类。
    
    表示SQL中的按位异或运算（^）。
    对两个整数的二进制表示进行按位异或操作。
    
    按位异或的特点：
    - 逐位比较：对每一位进行XOR操作
    - 结果规则：1^1=0, 1^0=1, 0^1=1, 0^0=0
    - 应用场景：数据加密、校验和计算
    """
    pass


class Div(Binary):
    """
    除法运算表达式类。
    
    表示SQL中的除法运算（/）。
    用于数值除法运算。
    
    除法运算的特点：
    - 数值除法：两个数值相除
    - 精度控制：支持指定结果精度
    - 安全除法：支持避免除零错误的选项
    - 类型转换：支持指定结果类型
    
    特殊选项：
    - typed：指定结果的数据类型
    - safe：安全除法，避免除零错误
    """
    arg_types = {
        "this": True,      # 被除数（必需）
        "expression": True, # 除数（必需）
        "typed": False,    # 结果类型（可选）
        "safe": False,     # 安全除法选项（可选）
    }


class Overlaps(Binary):
    """
    重叠比较表达式类。
    
    表示SQL中的OVERLAPS操作符。
    用于比较两个时间区间或空间范围是否重叠。
    
    重叠比较的特点：
    - 时间重叠：比较两个时间段是否有交集
    - 空间重叠：比较两个几何对象是否重叠
    - 返回布尔值：TRUE表示重叠，FALSE表示不重叠
    - NULL处理：任何操作数为NULL时返回NULL
    
    语法示例：
    - WHERE (start1, end1) OVERLAPS (start2, end2)
    - WHERE period1 OVERLAPS period2
    """
    pass


class Dot(Binary):
    """
    点号表达式类。
    
    表示SQL中的点号操作符（.），用于限定标识符。
    点号用于分隔数据库对象的层次结构。
    
    点号的应用场景：
    - 表限定：schema.table
    - 列限定：table.column
    - 数据库限定：database.schema.table
    - 函数调用：schema.function()
    
    层次结构：
    - catalog.database.schema.table.column
    - 从最外层到最内层的限定
    """
    
    @property
    def is_star(self) -> bool:
        """
        检查是否为星号表达式。
        
        判断点号表达式的右操作数是否为星号（*）。
        星号表示选择所有列。
        
        返回:
            bool: 如果右操作数是星号则返回True
        """
        # 检查右操作数（expression）是否为星号
        # 星号在SQL中表示选择所有列
        return self.expression.is_star

    @property
    def name(self) -> str:
        """
        获取点号表达式的名称。
        
        返回点号表达式右操作数的名称。
        这通常是列名、表名等标识符的名称。
        
        返回:
            str: 右操作数的名称
        """
        # 返回右操作数的名称
        # 在点号表达式中，右操作数通常是目标标识符
        return self.expression.name

    @property
    def output_name(self) -> str:
        """
        获取输出名称。
        
        返回点号表达式的输出名称。
        对于点号表达式，输出名称就是其名称。
        
        返回:
            str: 输出名称
        """
        # 点号表达式的输出名称就是其名称
        return self.name

    @classmethod
    def build(cls, expressions: t.Sequence[Expression]) -> Dot:
        """
        构建点号表达式对象。
        
        从表达式序列构建点号表达式。
        使用reduce函数将多个表达式连接成点号表达式链。
        
        参数:
            expressions: 表达式序列，至少需要2个表达式
            
        返回:
            Dot: 构建的点号表达式对象
            
        异常:
            ValueError: 当表达式数量少于2个时抛出
            
        示例:
            Dot.build([table, column]) -> table.column
            Dot.build([db, schema, table]) -> db.schema.table
        """
        # 检查表达式数量，点号表达式至少需要2个操作数
        if len(expressions) < 2:
            raise ValueError("Dot requires >= 2 expressions.")

        # 使用reduce函数从左到右构建点号表达式链
        # lambda函数将两个表达式用点号连接
        # 最终结果是一个嵌套的点号表达式
        return t.cast(Dot, reduce(lambda x, y: Dot(this=x, expression=y), expressions))

    @property
    def parts(self) -> t.List[Expression]:
        """
        获取点号表达式的各个部分。
        
        返回点号表达式的所有部分，按照标准顺序排列：
        catalog, database, schema, table, column
        
        返回:
            List[Expression]: 按顺序排列的表达式部分列表
            
        处理逻辑:
        1. 展平点号表达式链
        2. 反转顺序以便从内到外处理
        3. 按照COLUMN_PARTS顺序收集各部分
        4. 再次反转得到正确的顺序
        """
        # 展平点号表达式链，获取所有部分
        this, *parts = self.flatten()

        # 反转部分列表，因为flatten是从外到内的顺序
        # 我们需要从内到外处理
        parts.reverse()

        # 按照COLUMN_PARTS定义的顺序收集各部分
        # COLUMN_PARTS = ("this", "table", "db", "catalog")
        for arg in COLUMN_PARTS:
            part = this.args.get(arg)
            
            # 如果部分存在且是表达式，则添加到列表中
            if isinstance(part, Expression):
                parts.append(part)

        # 再次反转，得到正确的从外到内的顺序
        parts.reverse()
        return parts


# 数据类型联合类型定义
DATA_TYPE = t.Union[str, Identifier, Dot, DataType, DataType.Type]


class DPipe(Binary):
    """
    双管道表达式类。
    
    表示SQL中的双管道操作符（||）。
    用于字符串连接操作。
    
    双管道的特点：
    - 字符串连接：将两个字符串连接成一个
    - 标准SQL：||是SQL标准的字符串连接操作符
    - 数据库支持：大多数现代数据库都支持
    - NULL处理：任何操作数为NULL时返回NULL
    
    语法示例：
    - SELECT first_name || ' ' || last_name AS full_name
    - WHERE 'Hello' || 'World' = 'HelloWorld'
    """
    arg_types = {
        "this": True,      # 左操作数（必需）
        "expression": True, # 右操作数（必需）
        "safe": False,     # 安全连接选项（可选）
    }


class EQ(Binary, Predicate):
    """
    等于比较表达式类。
    
    表示SQL中的等于操作符（=）。
    用于比较两个值是否相等。
    
    等于比较的特点：
    - 值比较：比较两个表达式的值
    - 类型转换：支持隐式类型转换
    - NULL处理：NULL = NULL 返回NULL（不是TRUE）
    - 三值逻辑：TRUE、FALSE、NULL
    """
    pass


class NullSafeEQ(Binary, Predicate):
    """
    空值安全等于比较表达式类。
    
    表示SQL中的空值安全等于操作符（<=>）。
    用于比较两个值是否相等，对NULL值有特殊处理。
    
    空值安全等于的特点：
    - NULL处理：NULL <=> NULL 返回TRUE
    - 值比较：非NULL值的比较与普通等于相同
    - MySQL特有：主要用于MySQL数据库
    - 避免NULL陷阱：解决NULL比较的常见问题
    """
    pass


class NullSafeNEQ(Binary, Predicate):
    """
    空值安全不等于比较表达式类。
    
    表示SQL中的空值安全不等于操作符。
    用于比较两个值是否不相等，对NULL值有特殊处理。
    
    空值安全不等于的特点：
    - NULL处理：NULL与任何值的比较都返回FALSE
    - 值比较：非NULL值的比较与普通不等于相同
    - 逻辑清晰：避免NULL比较的歧义
    """
    pass


# 表示例如DuckDB中的:=，主要用于设置参数
class PropertyEQ(Binary):
    """
    属性赋值表达式类。
    
    表示SQL中的属性赋值操作符（:=）。
    主要用于设置参数和变量。
    
    属性赋值的特点：
    - 参数设置：设置会话参数或变量
    - DuckDB支持：DuckDB数据库中的特殊语法
    - 赋值语义：与普通等于不同，这是赋值操作
    - 作用域：通常影响当前会话或连接
    
    语法示例：
    - SET param := value
    - variable := expression
    """
    pass


class Distance(Binary):
    """
    距离计算表达式类。
    
    表示SQL中的距离计算操作。
    用于计算两个几何对象或空间点之间的距离。
    
    距离计算的特点：
    - 空间计算：计算几何对象间的距离
    - 多种距离：支持欧几里得距离、曼哈顿距离等
    - 地理信息：支持地理坐标系统
    - 性能优化：支持空间索引优化
    """
    pass


class Escape(Binary):
    """
    转义表达式类。
    
    表示SQL中的转义操作。
    用于在字符串中处理特殊字符。
    
    转义的特点：
    - 字符转义：处理字符串中的特殊字符
    - 模式匹配：在LIKE等模式匹配中使用
    - 安全处理：防止SQL注入等安全问题
    - 字符编码：处理不同字符编码的转义
    """
    pass


class Glob(Binary, Predicate):
    """
    通配符匹配表达式类。
    
    表示SQL中的GLOB操作符。
    用于文件名模式匹配，类似于Unix shell的通配符。
    
    GLOB的特点：
    - 通配符：支持*、?、[]等通配符
    - 大小写敏感：区分大小写
    - SQLite支持：主要用于SQLite数据库
    - 模式匹配：类似正则表达式的简化版本
    """
    pass


class GT(Binary, Predicate):
    """
    大于比较表达式类。
    
    表示SQL中的大于操作符（>）。
    用于比较两个值的大小关系。
    
    大于比较的特点：
    - 数值比较：比较数值的大小
    - 字符串比较：按字典序比较字符串
    - 日期比较：比较日期时间的先后
    - NULL处理：任何操作数为NULL时返回NULL
    """
    pass


class GTE(Binary, Predicate):
    """
    大于等于比较表达式类。
    
    表示SQL中的大于等于操作符（>=）。
    用于比较两个值的大小关系。
    
    大于等于比较的特点：
    - 包含等于：大于或等于都返回TRUE
    - 数值比较：比较数值的大小关系
    - 字符串比较：按字典序比较字符串
    - NULL处理：任何操作数为NULL时返回NULL
    """
    pass


class ILike(Binary, Predicate):
    """
    不区分大小写的LIKE表达式类。
    
    表示SQL中的ILIKE操作符。
    用于不区分大小写的模式匹配。
    
    ILIKE的特点：
    - 不区分大小写：'Hello' ILIKE 'hello' 返回TRUE
    - 模式匹配：支持%和_通配符
    - PostgreSQL支持：主要用于PostgreSQL数据库
    - 性能考虑：比LIKE稍慢，因为需要大小写转换
    """
    pass


class IntDiv(Binary):
    """
    整数除法表达式类。
    
    表示SQL中的整数除法操作符（DIV或/）。
    用于执行整数除法运算，结果向下取整。
    
    整数除法的特点：
    - 向下取整：结果总是整数
    - 截断小数：不进行四舍五入
    - 性能优化：比浮点除法更快
    - 数据库差异：不同数据库的语法可能不同
    
    示例：
    - 7 DIV 3 = 2
    - 8 DIV 3 = 2
    """
    pass


class Is(Binary, Predicate):
    """
    IS比较表达式类。
    
    表示SQL中的IS操作符。
    用于与NULL值进行比较。
    
    IS操作符的特点：
    - NULL比较：专门用于NULL值比较
    - 三值逻辑：IS NULL、IS NOT NULL
    - 类型安全：避免NULL比较的歧义
    - 标准SQL：所有数据库都支持
    
    语法示例：
    - WHERE column IS NULL
    - WHERE column IS NOT NULL
    - WHERE column IS TRUE
    """
    pass


class Kwarg(Binary):
    """
    关键字参数表达式类。
    
    表示特殊函数中的关键字参数。
    用于函数调用中的命名参数传递。
    
    关键字参数的特点：
    - 命名参数：使用名称而不是位置传递参数
    - 函数调用：在特殊函数中使用
    - 语法形式：func(kwarg => value)
    - 灵活性：可以改变参数顺序
    
    应用场景：
    - 自定义函数调用
    - 存储过程调用
    - 特殊SQL函数
    """
    pass


class Like(Binary, Predicate):
    """
    LIKE模式匹配表达式类。
    
    表示SQL中的LIKE操作符。
    用于字符串的模式匹配。
    
    LIKE的特点：
    - 模式匹配：支持通配符%和_
    - 区分大小写：'Hello' LIKE 'hello' 返回FALSE
    - 通配符：%匹配任意字符，_匹配单个字符
    - 转义支持：支持转义特殊字符
    
    语法示例：
    - WHERE name LIKE 'John%'
    - WHERE name LIKE '_ohn'
    - WHERE name LIKE 'John\_%' ESCAPE '\'
    """
    pass


class LT(Binary, Predicate):
    """
    小于比较表达式类。
    
    表示SQL中的小于操作符（<）。
    用于比较两个值的大小关系。
    
    小于比较的特点：
    - 严格小于：不包含等于的情况
    - 数值比较：比较数值的大小
    - 字符串比较：按字典序比较
    - NULL处理：任何操作数为NULL时返回NULL
    """
    pass


class LTE(Binary, Predicate):
    """
    小于等于比较表达式类。
    
    表示SQL中的小于等于操作符（<=）。
    用于比较两个值的大小关系。
    
    小于等于比较的特点：
    - 包含等于：小于或等于都返回TRUE
    - 数值比较：比较数值的大小关系
    - 字符串比较：按字典序比较字符串
    - NULL处理：任何操作数为NULL时返回NULL
    """
    pass


class Mod(Binary):
    """
    取模运算表达式类。
    
    表示SQL中的取模操作符（%或MOD）。
    用于计算两个数相除的余数。
    
    取模运算的特点：
    - 余数计算：返回除法的余数部分
    - 整数运算：通常用于整数运算
    - 周期性：常用于周期性计算
    - 负数处理：不同数据库对负数的处理可能不同
    
    示例：
    - 7 % 3 = 1
    - 8 % 3 = 2
    """
    pass


class Mul(Binary):
    """
    乘法运算表达式类。
    
    表示SQL中的乘法操作符（*）。
    用于数值乘法运算。
    
    乘法运算的特点：
    - 数值乘法：两个数值相乘
    - 类型提升：结果类型可能提升
    - 精度处理：需要考虑精度和溢出
    - NULL处理：任何操作数为NULL时返回NULL
    """
    pass


class NEQ(Binary, Predicate):
    """
    不等于比较表达式类。
    
    表示SQL中的不等于操作符（!=或<>）。
    用于比较两个值是否不相等。
    
    不等于比较的特点：
    - 值比较：比较两个表达式的值
    - 类型转换：支持隐式类型转换
    - NULL处理：NULL != NULL 返回NULL
    - 三值逻辑：TRUE、FALSE、NULL
    """
    pass


# 参考：https://www.postgresql.org/docs/current/ddl-schemas.html#DDL-SCHEMAS-PATH
class Operator(Binary):
    """
    操作符表达式类。
    
    表示SQL中的自定义操作符。
    用于支持数据库特定的操作符。
    
    操作符的特点：
    - 自定义操作符：支持用户定义的操作符
    - PostgreSQL支持：主要用于PostgreSQL
    - 操作符重载：同一操作符可以有不同的实现
    - 模式路径：支持操作符的模式路径
    
    应用场景：
    - 几何操作符
    - 数组操作符
    - 自定义数据类型操作符
    """
    arg_types = {
        "this": True,      # 左操作数（必需）
        "operator": True,  # 操作符名称（必需）
        "expression": True, # 右操作数（必需）
    }


class SimilarTo(Binary, Predicate):
    """
    SIMILAR TO模式匹配表达式类。
    
    表示SQL中的SIMILAR TO操作符。
    用于正则表达式模式匹配。
    
    SIMILAR TO的特点：
    - 正则表达式：支持POSIX正则表达式
    - 模式匹配：比LIKE更强大的模式匹配
    - PostgreSQL支持：主要用于PostgreSQL
    - 转义字符：支持转义特殊字符
    
    语法示例：
    - WHERE name SIMILAR TO 'John%'
    - WHERE name SIMILAR TO '[A-Z][a-z]*'
    """
    pass


class Slice(Binary):
    """
    切片表达式类。
    
    表示SQL中的切片操作。
    用于从字符串或数组中提取子序列。
    
    切片的特点：
    - 子序列提取：从序列中提取部分元素
    - 索引支持：支持起始和结束索引
    - 字符串切片：从字符串中提取子字符串
    - 数组切片：从数组中提取子数组
    
    语法示例：
    - SUBSTRING(string FROM start FOR length)
    - array[start:end]
    """
    arg_types = {
        "this": False,      # 源序列（可选）
        "expression": False, # 切片参数（可选）
    }


class Sub(Binary):
    """
    减法运算表达式类。
    
    表示SQL中的减法操作符（-）。
    用于数值减法运算。
    
    减法运算的特点：
    - 数值减法：两个数值相减
    - 类型转换：支持隐式类型转换
    - 精度处理：需要考虑精度和溢出
    - NULL处理：任何操作数为NULL时返回NULL
    """
    pass


# 一元表达式
# (NOT a)
class Unary(Condition):
    """
    一元表达式基类。
    
    表示具有单个操作数的表达式。
    一元表达式是SQL中的重要表达式类型。
    
    一元表达式的特点：
    - 单操作数：只有一个操作数
    - 操作符：定义对操作数的操作
    - 结果类型：根据操作符和操作数类型确定
    - 优先级：通常比二元表达式优先级高
    
    常见的一元表达式：
    - 逻辑非：NOT
    - 算术负号：-
    - 按位非：~
    - 括号：()
    """
    pass


class BitwiseNot(Unary):
    """
    按位非运算表达式类。
    
    表示SQL中的按位非操作符（~）。
    对整数的二进制表示进行按位取反。
    
    按位非的特点：
    - 逐位取反：对每一位进行NOT操作
    - 结果规则：~1=0, ~0=1
    - 整数运算：通常用于整数运算
    - 应用场景：位操作、掩码处理
    """
    pass


class Not(Unary):
    """
    逻辑非运算表达式类。
    
    表示SQL中的逻辑非操作符（NOT）。
    用于逻辑值的取反。
    
    逻辑非的特点：
    - 逻辑取反：TRUE变FALSE，FALSE变TRUE
    - NULL处理：NOT NULL 返回NULL
    - 三值逻辑：TRUE、FALSE、NULL
    - 优先级：通常比AND、OR优先级高
    """
    pass


class Paren(Unary):
    """
    括号表达式类。
    
    表示SQL中的括号操作符（()）。
    用于改变表达式的计算优先级。
    
    括号的特点：
    - 优先级控制：改变运算的优先级
    - 分组操作：将表达式分组
    - 子查询：用于子查询的语法
    - 函数调用：用于函数参数列表
    
    输出名称处理：
    - 括号表达式本身没有名称
    - 输出名称来自内部表达式
    """
    
    @property
    def output_name(self) -> str:
        """
        获取输出名称。
        
        括号表达式本身没有名称，输出名称来自内部表达式。
        这确保了括号不会影响表达式的输出名称。
        
        返回:
            str: 内部表达式的名称
        """
        # 括号表达式没有自己的名称，使用内部表达式的名称
        return self.this.name


class Neg(Unary):
    """
    负号运算表达式类。
    
    表示SQL中的负号操作符（-）。
    用于数值的取反运算。
    
    负号的特点：
    - 数值取反：正数变负数，负数变正数
    - 一元运算：只有一个操作数
    - 类型保持：结果类型与操作数相同
    - NULL处理：-NULL 返回NULL
    """
    
    def to_py(self) -> int | Decimal:
        """
        转换为Python对象。
        
        将负号表达式转换为Python的数值对象。
        如果操作数是数字，则返回其负值。
        
        返回:
            int | Decimal: Python数值对象，如果操作数是数字则返回负值
        """
        # 如果操作数是数字，返回其负值
        if self.is_number:
            # 将操作数转换为Python对象并取负
            return self.this.to_py() * -1
        # 如果不是数字，调用父类方法
        return super().to_py()
    

class Alias(Expression):
    """
    别名表达式类。
    
    表示SQL中的别名（AS子句）。
    用于为表、列、子查询等提供别名。
    
    别名的应用场景：
    - 列别名：SELECT column AS alias
    - 表别名：FROM table AS alias
    - 子查询别名：FROM (SELECT ...) AS alias
    - 函数别名：SELECT func() AS alias
    
    别名的好处：
    - 简化引用：使用简短名称引用复杂表达式
    - 提高可读性：使用有意义的名称
    - 避免冲突：解决名称冲突问题
    - 支持自引用：在递归查询中使用
    """
    arg_types = {
        "this": True,    # 被别名的表达式（必需）
        "alias": False,  # 别名名称（可选）
    }

    @property
    def output_name(self) -> str:
        """
        获取输出名称。
        
        对于别名表达式，输出名称就是别名本身。
        如果没有别名，则返回空字符串。
        
        返回:
            str: 别名名称
        """
        # 别名的输出名称就是别名本身
        return self.alias


# BigQuery要求UNPIVOT列列表别名必须是字符串或整数，
# 但其他方言要求标识符。这使我们能够轻松地在它们之间进行转译。
class PivotAlias(Alias):
    """
    透视表别名表达式类。
    
    表示透视表操作中的特殊别名。
    用于处理不同数据库方言对透视表别名的不同要求。
    
    方言差异：
    - BigQuery：要求UNPIVOT列别名是字符串或整数
    - 其他数据库：要求别名是标识符
    - 转译支持：支持在不同方言间转换
    
    应用场景：
    - UNPIVOT操作：列转行操作中的别名
    - PIVOT操作：行转列操作中的别名
    - 跨数据库兼容：处理方言差异
    """
    pass


# 表示Snowflake的ANY [ ORDER BY ... ]语法
# 参考：https://docs.snowflake.com/en/sql-reference/constructs/pivot
class PivotAny(Expression):
    """
    Snowflake透视表ANY表达式类。
    
    表示Snowflake数据库中的ANY [ ORDER BY ... ]语法。
    用于透视表操作中的特殊排序处理。
    
    Snowflake ANY语法特点：
    - 任意值选择：从多个值中选择任意一个
    - 排序支持：支持ORDER BY子句
    - 聚合函数：通常与聚合函数一起使用
    - Snowflake特有：主要用于Snowflake数据库
    
    语法示例：
    - PIVOT (ANY(column) FOR category IN (...)) ORDER BY ...
    """
    arg_types = {
        "this": False,  # 表达式（可选）
    }


class Aliases(Expression):
    """
    多别名表达式类。
    
    表示具有多个别名的表达式。
    用于处理需要多个别名的复杂情况。
    
    多别名的应用：
    - 多列别名：为多个列提供别名
    - 复杂表达式：为复杂表达式提供多个引用名称
    - 视图定义：在视图定义中使用多个别名
    - 子查询：在子查询中使用多个别名
    """
    arg_types = {
        "this": True,        # 主表达式（必需）
        "expressions": True, # 别名表达式列表（必需）
    }

    @property
    def aliases(self):
        """
        获取别名列表。
        
        返回所有别名表达式的列表。
        这是对expressions属性的语义化访问。
        
        返回:
            List[Expression]: 别名表达式列表
        """
        # 别名列表就是表达式列表
        return self.expressions


# 参考：https://docs.aws.amazon.com/redshift/latest/dg/query-super.html
class AtIndex(Expression):
    """
    数组索引表达式类。
    
    表示数组或JSON对象的索引访问。
    用于从数组或JSON对象中获取特定位置的元素。
    
    索引访问的特点：
    - 数组索引：从数组中获取指定位置的元素
    - JSON索引：从JSON对象中获取指定键的值
    - 零基索引：通常使用零基索引
    - 边界检查：需要处理索引越界的情况
    
    语法示例：
    - array[0]：获取数组第一个元素
    - json['key']：获取JSON对象的键值
    - super_column[1]：获取SUPER类型的元素
    """
    arg_types = {
        "this": True,      # 数组或JSON对象（必需）
        "expression": True, # 索引表达式（必需）
    }


class AtTimeZone(Expression):
    """
    时区转换表达式类。
    
    表示将时间戳转换到指定时区的操作。
    用于处理不同时区的时间转换。
    
    时区转换的特点：
    - 时区转换：将时间戳转换到目标时区
    - 时区支持：支持标准时区名称和偏移量
    - 夏令时：自动处理夏令时转换
    - 数据库差异：不同数据库的语法可能不同
    
    语法示例：
    - timestamp AT TIME ZONE 'UTC'
    - timestamp AT TIME ZONE '+08:00'
    """
    arg_types = {
        "this": True,  # 时间戳表达式（必需）
        "zone": True,  # 时区表达式（必需）
    }


class FromTimeZone(Expression):
    """
    从时区转换表达式类。
    
    表示从指定时区转换时间戳的操作。
    用于处理时区感知的时间转换。
    
    从时区转换的特点：
    - 时区感知：明确指定源时区
    - 时区转换：从源时区转换到目标时区
    - 精度保持：保持时间戳的精度
    - 应用场景：处理来自不同时区的时间数据
    
    语法示例：
    - FROM_TIMEZONE(timestamp, 'UTC')
    - FROM_TIMEZONE(timestamp, '+08:00')
    """
    arg_types = {
        "this": True,  # 时间戳表达式（必需）
        "zone": True,  # 源时区表达式（必需）
    }


class FormatPhrase(Expression):
    """
    格式短语表达式类。
    
    表示Teradata数据库中的列格式覆盖。
    可以根据需要扩展到其他方言。
    
    参考：https://docs.teradata.com/r/Enterprise_IntelliFlex_VMware/SQL-Data-Types-and-Literals/Data-Type-Formats-and-Format-Phrases/FORMAT
    
    格式短语的特点：
    - 格式覆盖：为列指定特定的显示格式
    - Teradata特有：主要用于Teradata数据库
    - 数据类型：支持各种数据类型的格式
    - 可扩展性：可以扩展到其他数据库方言
    
    应用场景：
    - 日期格式：指定日期的显示格式
    - 数字格式：指定数字的显示格式
    - 字符串格式：指定字符串的显示格式
    """
    arg_types = {
        "this": True,   # 列表达式（必需）
        "format": True, # 格式表达式（必需）
    }


class Between(Predicate):
    """
    BETWEEN范围比较表达式类。
    
    表示SQL中的BETWEEN操作符。
    用于检查值是否在指定范围内。
    
    BETWEEN的特点：
    - 范围检查：检查值是否在[low, high]范围内
    - 包含边界：默认包含边界值
    - 对称性：支持对称范围检查
    - 性能优化：通常比AND组合更高效
    
    语法示例：
    - WHERE age BETWEEN 18 AND 65
    - WHERE date BETWEEN '2023-01-01' AND '2023-12-31'
    - WHERE value BETWEEN SYMMETRIC 10 AND 5
    """
    arg_types = {
        "this": True,      # 要检查的值（必需）
        "low": True,       # 下界（必需）
        "high": True,      # 上界（必需）
        "symmetric": False, # 对称范围选项（可选）
    }


class Bracket(Condition):
    """
    方括号表达式类。
    
    表示SQL中的方括号操作符（[]）。
    用于数组下标访问和JSON路径访问。
    
    参考：https://cloud.google.com/bigquery/docs/reference/standard-sql/operators#array_subscript_operator
    
    方括号的特点：
    - 数组访问：从数组中获取元素
    - JSON访问：从JSON对象中获取值
    - 多维度：支持多维数组访问
    - 安全访问：支持安全访问模式
    
    语法示例：
    - array[0]：获取数组第一个元素
    - json['key']：获取JSON对象的键值
    - array[1, 2]：多维数组访问
    - array[SAFE 0]：安全访问
    """
    arg_types = {
        "this": True,                    # 数组或JSON对象（必需）
        "expressions": True,             # 索引表达式列表（必需）
        "offset": False,                 # 偏移量（可选）
        "safe": False,                   # 安全访问选项（可选）
        "returns_list_for_maps": False,  # 映射返回列表选项（可选）
    }

    @property
    def output_name(self) -> str:
        """
        获取输出名称。
        
        如果只有一个索引表达式，返回该表达式的输出名称。
        否则返回父类的输出名称。
        
        返回:
            str: 输出名称
        """
        # 如果只有一个索引表达式，使用其输出名称
        if len(self.expressions) == 1:
            return self.expressions[0].output_name

        # 否则使用父类的输出名称
        return super().output_name


class Distinct(Expression):
    """
    去重表达式类。
    
    表示SQL中的DISTINCT操作符。
    用于去除重复值。
    
    DISTINCT的特点：
    - 去重：去除重复的行或值
    - 性能影响：可能影响查询性能
    - 排序：某些数据库需要排序来实现去重
    - 聚合：通常与聚合函数一起使用
    
    语法示例：
    - SELECT DISTINCT column
    - SELECT DISTINCT ON (column) *
    - COUNT(DISTINCT column)
    """
    arg_types = {
        "expressions": False, # 去重表达式列表（可选）
        "on": False,         # ON子句（可选）
    }


class In(Predicate):
    """
    IN成员检查表达式类。
    
    表示SQL中的IN操作符。
    用于检查值是否在指定的值列表中。
    
    IN的特点：
    - 成员检查：检查值是否在列表中
    - 子查询支持：支持子查询作为值列表
    - 性能优化：通常比多个OR条件更高效
    - NULL处理：NULL IN (...) 返回NULL
    
    语法示例：
    - WHERE column IN (1, 2, 3)
    - WHERE column IN (SELECT id FROM table)
    - WHERE column NOT IN (1, 2, 3)
    """
    arg_types = {
        "this": True,        # 要检查的值（必需）
        "expressions": False, # 值列表（可选）
        "query": False,      # 子查询（可选）
        "unnest": False,     # UNNEST操作（可选）
        "field": False,      # 字段访问（可选）
        "is_global": False,  # 全局IN选项（可选）
    }


# 参考：https://cloud.google.com/bigquery/docs/reference/standard-sql/procedural-language#for-in
class ForIn(Expression):
    """
    FOR-IN循环表达式类。
    
    表示BigQuery中的FOR-IN循环语法。
    用于过程化语言中的循环控制。
    
    参考：https://cloud.google.com/bigquery/docs/reference/standard-sql/procedural-language#for-in
    
    FOR-IN循环的特点：
    - 循环控制：用于过程化语言中的循环
    - 迭代变量：自动处理迭代变量
    - 集合遍历：遍历集合中的每个元素
    - BigQuery特有：主要用于BigQuery的过程化语言
    
    语法示例：
    - FOR item IN (SELECT * FROM table) DO ...
    - FOR i IN 1..10 DO ...
    """
    arg_types = {
        "this": True,      # 迭代变量（必需）
        "expression": True, # 迭代集合（必需）
    }


class TimeUnit(Expression):
    """
    时间单位表达式类。
    
    表示SQL中的时间单位，如DAY、HOUR、MINUTE等。
    自动将单位参数转换为Var对象，确保单位名称的一致性。
    
    时间单位的特点：
    - 标准化：将缩写形式转换为完整形式
    - 大小写统一：统一转换为大写
    - 类型安全：确保时间单位的正确性
    - 跨数据库兼容：支持不同数据库的时间单位语法
    
    应用场景：
    - 时间间隔：INTERVAL 1 DAY
    - 日期函数：DATE_ADD(date, INTERVAL 1 HOUR)
    - 时间戳操作：时间戳的加减运算
    """
    # 自动将单位参数转换为Var对象
    arg_types = {"unit": False}  # 时间单位（可选）

    # 时间单位缩写到完整名称的映射
    UNABBREVIATED_UNIT_NAME = {
        "D": "DAY",           # 天
        "H": "HOUR",          # 小时
        "M": "MINUTE",        # 分钟
        "MS": "MILLISECOND",  # 毫秒
        "NS": "NANOSECOND",   # 纳秒
        "Q": "QUARTER",       # 季度
        "S": "SECOND",        # 秒
        "US": "MICROSECOND",  # 微秒
        "W": "WEEK",          # 周
        "Y": "YEAR",          # 年
    }

    # 支持Var-like类型的类集合
    VAR_LIKE = (Column, Literal, Var)

    def __init__(self, **args):
        """
        初始化时间单位表达式。
        
        自动处理时间单位的标准化：
        1. 将缩写形式转换为完整形式
        2. 统一转换为大写
        3. 确保类型一致性
        """
        unit = args.get("unit")

        # 如果单位是Var-like类型，进行标准化处理
        if type(unit) in self.VAR_LIKE and not (isinstance(unit, Column) and len(unit.parts) != 1):
            # 获取完整的时间单位名称，如果找不到则使用原名称
            # 然后转换为大写并创建Var对象
            args["unit"] = Var(
                this=(self.UNABBREVIATED_UNIT_NAME.get(unit.name) or unit.name).upper()
            )
        elif isinstance(unit, Week):
            # 如果单位是Week类型，将其转换为Var对象
            # 并确保名称为大写
            unit.set("this", Var(this=unit.this.name.upper()))

        # 调用父类构造函数
        super().__init__(**args)

    @property
    def unit(self) -> t.Optional[Var | IntervalSpan]:
        """
        获取时间单位。
        
        返回时间单位参数，可能是Var对象或IntervalSpan对象。
        
        返回:
            Optional[Var | IntervalSpan]: 时间单位对象
        """
        return self.args.get("unit")


class IntervalOp(TimeUnit):
    """
    时间间隔操作表达式类。
    
    表示时间间隔操作，如加减时间间隔。
    继承自TimeUnit，增加了表达式参数。
    
    时间间隔操作的特点：
    - 表达式操作：对时间表达式进行间隔操作
    - 单位支持：支持各种时间单位
    - 类型安全：确保操作的类型正确性
    - 跨数据库兼容：支持不同数据库的语法
    
    应用场景：
    - 日期计算：DATE + INTERVAL 1 DAY
    - 时间戳操作：TIMESTAMP + INTERVAL 1 HOUR
    - 时间差计算：计算两个时间之间的间隔
    """
    arg_types = {
        "unit": False,      # 时间单位（可选）
        "expression": True, # 时间表达式（必需）
    }

    def interval(self):
        """
        创建Interval对象。
        
        将当前IntervalOp转换为Interval对象。
        复制表达式和单位，确保对象的独立性。
        
        返回:
            Interval: 新创建的Interval对象
        """
        # 创建新的Interval对象
        # 复制表达式和单位，避免引用共享
        return Interval(
            this=self.expression.copy(),  # 复制时间表达式
            unit=self.unit.copy() if self.unit else None,  # 复制时间单位（如果存在）
        )


# 参考：
# https://www.oracletutorial.com/oracle-basics/oracle-interval/
# https://trino.io/docs/current/language/types.html#interval-day-to-second
# https://docs.databricks.com/en/sql/language-manual/data-types/interval-type.html
class IntervalSpan(DataType):
    """
    时间间隔跨度数据类型类。
    
    表示时间间隔的跨度，如DAY TO SECOND、YEAR TO MONTH等。
    用于定义时间间隔的精度和范围。
    
    时间间隔跨度的特点：
    - 精度控制：指定时间间隔的精度
    - 范围定义：定义时间间隔的范围
    - 数据库支持：Oracle、Trino、Databricks等数据库支持
    - 类型安全：确保时间间隔的类型正确性
    
    常见的间隔跨度：
    - DAY TO SECOND：天到秒的间隔
    - YEAR TO MONTH：年到月的间隔
    - HOUR TO MINUTE：小时到分钟的间隔
    
    语法示例：
    - INTERVAL '1 2:3:4' DAY TO SECOND
    - INTERVAL '1-2' YEAR TO MONTH
    """
    arg_types = {
        "this": True,      # 间隔类型（必需）
        "expression": True, # 间隔值（必需）
    }


class Interval(TimeUnit):
    """
    时间间隔表达式类。
    
    表示SQL中的时间间隔，如INTERVAL 1 DAY。
    用于时间计算和日期操作。
    
    时间间隔的特点：
    - 时间计算：用于时间的加减运算
    - 单位支持：支持各种时间单位
    - 精度控制：支持不同的精度设置
    - 跨数据库兼容：支持不同数据库的语法
    
    应用场景：
    - 日期计算：DATE + INTERVAL 1 DAY
    - 时间戳操作：TIMESTAMP + INTERVAL 1 HOUR
    - 时间差：计算两个时间之间的间隔
    - 调度任务：定义任务执行的时间间隔
    """
    arg_types = {
        "this": False,  # 间隔值（可选）
        "unit": False,  # 时间单位（可选）
    }


class IgnoreNulls(Expression):
    """
    忽略空值表达式类。
    
    表示SQL中的IGNORE NULLS子句。
    用于聚合函数中忽略NULL值。
    
    IGNORE NULLS的特点：
    - 聚合函数：在聚合函数中忽略NULL值
    - 性能优化：提高聚合函数的性能
    - 数据清理：自动过滤无效数据
    - 标准SQL：符合SQL标准
    
    应用场景：
    - 窗口函数：FIRST_VALUE() IGNORE NULLS
    - 聚合函数：MAX() IGNORE NULLS
    - 数据统计：统计时忽略缺失值
    """
    pass


class RespectNulls(Expression):
    """
    尊重空值表达式类。
    
    表示SQL中的RESPECT NULLS子句。
    用于聚合函数中保留NULL值。
    
    RESPECT NULLS的特点：
    - 保留NULL：在聚合函数中保留NULL值
    - 默认行为：大多数聚合函数的默认行为
    - 数据完整性：保持数据的完整性
    - 明确指定：明确指定处理NULL值的方式
    
    应用场景：
    - 窗口函数：FIRST_VALUE() RESPECT NULLS
    - 聚合函数：MAX() RESPECT NULLS
    - 数据统计：统计时保留缺失值
    """
    pass


# 参考：https://cloud.google.com/bigquery/docs/reference/standard-sql/aggregate-function-calls#max_min_clause
class HavingMax(Expression):
    """
    HAVING MAX表达式类。
    
    表示BigQuery中的HAVING MAX子句。
    用于在聚合函数中指定最大值条件。
    
    参考：https://cloud.google.com/bigquery/docs/reference/standard-sql/aggregate-function-calls#max_min_clause
    
    HAVING MAX的特点：
    - 最大值条件：指定聚合函数的最大值条件
    - BigQuery特有：主要用于BigQuery数据库
    - 聚合函数：与聚合函数一起使用
    - 性能优化：优化聚合查询的性能
    
    应用场景：
    - 聚合查询：在聚合查询中指定最大值条件
    - 数据分析：分析数据中的最大值
    - 性能优化：优化聚合查询的执行
    """
    arg_types = {
        "this": True,      # 聚合表达式（必需）
        "expression": True, # 条件表达式（必需）
        "max": True,       # 最大值表达式（必需）
    }


# 函数表达式
class Func(Condition):
    """
    所有函数表达式的基类。
    
    这是SQLGlot中所有函数表达式的基类，提供了函数的基本功能。
    函数表达式是SQL中最重要的表达式类型之一。
    
    属性:
        is_var_len_args (bool): 如果设置为True，arg_types中定义的最后一个参数
            将被视为可变长度参数，参数的值将作为列表存储。
        _sql_names (list): 此函数表达式的SQL名称（列表中的第1项）和别名（后续项）。
            这些值用于在解析期间将此节点映射到名称，以及在SQL字符串生成期间
            提供函数的名称。默认情况下，SQL名称设置为转换为蛇形命名法的表达式类名。
    """

    is_var_len_args = False  # 是否支持可变长度参数

    @classmethod
    def from_arg_list(cls, args):
        """
        从参数列表创建函数实例。
        
        根据函数是否支持可变长度参数，采用不同的参数处理策略：
        - 如果支持可变长度参数：将最后一个参数作为可变长度参数处理
        - 如果不支持：按固定参数处理
        
        参数:
            args: 参数列表
            
        返回:
            Func: 函数实例
        """
        if cls.is_var_len_args:
            # 获取所有参数键
            all_arg_keys = list(cls.arg_types)
            # 如果此函数支持可变长度参数，将最后一个参数视为可变长度参数
            non_var_len_arg_keys = all_arg_keys[:-1] if cls.is_var_len_args else all_arg_keys
            num_non_var = len(non_var_len_arg_keys)

            # 构建参数字典：固定参数按位置分配
            args_dict = {arg_key: arg for arg, arg_key in zip(args, non_var_len_arg_keys)}
            # 剩余参数作为可变长度参数
            args_dict[all_arg_keys[-1]] = args[num_non_var:]
        else:
            # 不支持可变长度参数时，按固定参数处理
            args_dict = {arg_key: arg for arg, arg_key in zip(args, cls.arg_types)}

        return cls(**args_dict)

    @classmethod
    def sql_names(cls):
        """
        获取函数的SQL名称列表。
        
        返回函数的SQL名称和别名列表。
        如果类没有定义_sql_names，则使用类名的蛇形命名法作为默认名称。
        
        返回:
            list: SQL名称列表
            
        异常:
            NotImplementedError: 当在基类Func上调用时抛出
        """
        if cls is Func:
            raise NotImplementedError(
                "SQL name is only supported by concrete function implementations"
            )
        # 如果类没有定义_sql_names，使用类名的蛇形命名法作为默认名称
        if "_sql_names" not in cls.__dict__:
            cls._sql_names = [camel_to_snake_case(cls.__name__)]
        return cls._sql_names

    @classmethod
    def sql_name(cls):
        """
        获取函数的主要SQL名称。
        
        返回函数的主要SQL名称（列表中的第一个名称）。
        
        返回:
            str: 主要的SQL名称
        """
        sql_names = cls.sql_names()
        assert sql_names, f"Expected non-empty 'sql_names' for Func: {cls.__name__}."
        return sql_names[0]

    @classmethod
    def default_parser_mappings(cls):
        """
        获取默认的解析器映射。
        
        返回函数名称到解析器的映射字典。
        用于在解析SQL时识别函数调用。
        
        返回:
            dict: 名称到解析器的映射
        """
        return {name: cls.from_arg_list for name in cls.sql_names()}


class Typeof(Func):
    """
    类型检查函数类。
    
    表示SQL中的TYPEOF函数。
    用于获取表达式的数据类型。
    
    类型检查的特点：
    - 类型识别：返回表达式的数据类型
    - 动态类型：在运行时确定类型
    - 调试支持：用于调试和类型验证
    - 跨数据库兼容：不同数据库的语法可能不同
    """
    pass


class Acos(Func):
    pass


class Acosh(Func):
    pass


class Asin(Func):
    pass


class Asinh(Func):
    pass


class Atan(Func):
    arg_types = {"this": True, "expression": False}


class Atanh(Func):
    pass


class Atan2(Func):
    arg_types = {"this": True, "expression": True}


class Cot(Func):
    pass


class Coth(Func):
    pass


class Cos(Func):
    pass


class Csc(Func):
    pass


class Csch(Func):
    pass


class Sec(Func):
    pass


class Sech(Func):
    pass


class Sin(Func):
    pass


class Sinh(Func):
    pass


class Tan(Func):
    pass


class Tanh(Func):
    pass


class Degrees(Func):
    pass


class Cosh(Func):
    pass


class CosineDistance(Func):
    arg_types = {"this": True, "expression": True}


class EuclideanDistance(Func):
    arg_types = {"this": True, "expression": True}


class JarowinklerSimilarity(Func):
    arg_types = {"this": True, "expression": True}


class AggFunc(Func):
    """
    聚合函数基类。
    
    表示SQL中的聚合函数，如SUM、COUNT、AVG等。
    聚合函数用于对一组值进行计算并返回单个结果。
    
    聚合函数的特点：
    - 数据聚合：对多行数据进行聚合计算
    - 分组支持：通常与GROUP BY一起使用
    - 窗口函数：支持窗口函数功能
    - 性能优化：数据库通常对聚合函数进行优化
    """
    pass


class BitwiseAndAgg(AggFunc):
    """
    按位与聚合函数类。
    
    表示SQL中的BIT_AND聚合函数。
    对一组整数进行按位与运算。
    
    按位与聚合的特点：
    - 位运算：对每个位进行AND操作
    - 整数运算：通常用于整数类型
    - 权限控制：常用于权限位操作
    - 性能优化：位运算通常很快
    """
    pass


class BitwiseOrAgg(AggFunc):
    """
    按位或聚合函数类。
    
    表示SQL中的BIT_OR聚合函数。
    对一组整数进行按位或运算。
    
    按位或聚合的特点：
    - 位运算：对每个位进行OR操作
    - 权限合并：常用于权限合并
    - 标志位操作：用于标志位的聚合
    """
    pass


class BitwiseXorAgg(AggFunc):
    """
    按位异或聚合函数类。
    
    表示SQL中的BIT_XOR聚合函数。
    对一组整数进行按位异或运算。
    
    按位异或聚合的特点：
    - 位运算：对每个位进行XOR操作
    - 奇偶校验：常用于奇偶校验
    - 数据加密：用于简单的数据加密
    """
    pass
    
class BitwiseCount(Func):
    pass

class BitwiseCountAgg(AggFunc):
    """
    位计数聚合函数类。
    
    表示SQL中的BIT_COUNT聚合函数。
    计算一组整数中设置位的数量。
    
    位计数的特点：
    - 位统计：统计设置位的数量
    - 性能分析：用于性能分析
    - 数据压缩：用于数据压缩算法
    """
    pass


class ByteLength(Func):
    pass


class Boolnot(Func):
    pass


class Booland(Func):
    arg_types = {"this": True, "expression": True}


class Boolor(Func):
    arg_types = {"this": True, "expression": True}


# https://cloud.google.com/bigquery/docs/reference/standard-sql/json_functions#bool_for_json
class JSONBool(Func):
    pass


class ArrayRemove(Func):
    """
    数组移除函数类。
    
    表示SQL中的数组移除函数。
    用于从数组中移除指定的元素。
    
    数组移除的特点：
    - 元素移除：从数组中移除指定元素
    - 类型安全：确保元素类型匹配
    - 性能考虑：可能需要重新分配数组
    - 数据库支持：不同数据库的语法可能不同
    """
    arg_types = {
        "this": True,      # 数组表达式（必需）
        "expression": True, # 要移除的元素（必需）
    }


class ParameterizedAgg(AggFunc):
    """
    参数化聚合函数类。
    
    表示SQL中的参数化聚合函数。
    用于支持参数的聚合函数，如PERCENTILE_CONT等。
    
    参数化聚合的特点：
    - 参数支持：支持额外的参数
    - 灵活配置：可以根据参数调整行为
    - 高级统计：常用于高级统计分析
    - 数据库差异：不同数据库的参数可能不同
    """
    arg_types = {
        "this": True,        # 聚合表达式（必需）
        "expressions": True, # 参数表达式列表（必需）
        "params": True,      # 参数列表（必需）
    }


class Abs(Func):
    """
    绝对值函数类。
    
    表示SQL中的ABS函数。
    用于计算数值的绝对值。
    
    绝对值函数的特点：
    - 数值运算：计算数值的绝对值
    - 类型支持：支持各种数值类型
    - 简单运算：基本的数学运算
    - 广泛支持：所有数据库都支持
    """
    pass


class ArgMax(AggFunc):
    """
    参数最大值聚合函数类。
    
    表示SQL中的ARG_MAX/MAX_BY函数。
    返回使表达式值最大的参数值。
    
    参数最大值的特点：
    - 条件聚合：根据条件返回参数值
    - 多列支持：支持多列比较
    - 性能优化：数据库通常优化此类函数
    - 应用场景：常用于数据分析
    """
    arg_types = {
        "this": True,      # 参数表达式（必需）
        "expression": True, # 比较表达式（必需）
        "count": False,    # 计数参数（可选）
    }
    _sql_names = ["ARG_MAX", "ARGMAX", "MAX_BY"]  # 支持的SQL名称


class ArgMin(AggFunc):
    """
    参数最小值聚合函数类。
    
    表示SQL中的ARG_MIN/MIN_BY函数。
    返回使表达式值最小的参数值。
    
    参数最小值的特点：
    - 条件聚合：根据条件返回参数值
    - 多列支持：支持多列比较
    - 性能优化：数据库通常优化此类函数
    - 应用场景：常用于数据分析
    """
    arg_types = {
        "this": True,      # 参数表达式（必需）
        "expression": True, # 比较表达式（必需）
        "count": False,    # 计数参数（可选）
    }
    _sql_names = ["ARG_MIN", "ARGMIN", "MIN_BY"]  # 支持的SQL名称


class ApproxTopK(AggFunc):
    """
    近似TopK聚合函数类。
    
    表示SQL中的近似TopK函数。
    用于高效地计算近似的前K个值。
    
    近似TopK的特点：
    - 近似计算：使用近似算法提高性能
    - 内存效率：使用有限内存处理大数据
    - 统计精度：在性能和精度间平衡
    - 大数据支持：特别适合大数据场景
    """
    arg_types = {
        "this": True,        # 值表达式（必需）
        "expression": False, # 可选表达式（可选）
        "counters": False,   # 计数器参数（可选）
    }


class ApproxTopSum(AggFunc):
    arg_types = {"this": True, "expression": True, "count": True}


class ApproxQuantiles(AggFunc):
    arg_types = {"this": True, "expression": False}


class FarmFingerprint(Func):
    arg_types = {"expressions": True}
    is_var_len_args = True
    _sql_names = ["FARM_FINGERPRINT", "FARMFINGERPRINT64"]


class Flatten(Func):
    """
    扁平化函数类。
    
    表示SQL中的扁平化函数。
    用于将嵌套数组或结构扁平化。
    
    扁平化的特点：
    - 结构扁平：将嵌套结构扁平化
    - 数组处理：处理嵌套数组
    - 数据转换：用于数据格式转换
    - 性能考虑：可能需要大量内存
    """
    pass


class Float64(Func):
    arg_types = {"this": True, "expression": False}


# https://spark.apache.org/docs/latest/api/sql/index.html#transform
class Transform(Func):
    """
    转换函数类。
    
    表示SQL中的TRANSFORM函数。
    用于对数组元素进行转换操作。
    
    参考：https://spark.apache.org/docs/latest/api/sql/index.html#transform
    
    转换函数的特点：
    - 数组转换：对数组中的每个元素进行转换
    - 函数应用：应用转换函数到每个元素
    - 类型转换：支持类型转换
    - Spark支持：主要用于Spark SQL
    """
    arg_types = {
        "this": True,      # 数组表达式（必需）
        "expression": True, # 转换表达式（必需）
    }


class Translate(Func):
    arg_types = {"this": True, "from": True, "to": True}


class Grouping(AggFunc):
    arg_types = {"expressions": True}
    is_var_len_args = True


class Anonymous(Func):
    """
    匿名函数类。
    
    表示SQL中的匿名函数调用。
    用于处理没有预定义名称的函数。
    
    匿名函数的特点：
    - 动态名称：函数名称在运行时确定
    - 可变参数：支持可变长度参数
    - 灵活调用：支持各种函数调用方式
    - 数据库差异：不同数据库的语法可能不同
    """
    arg_types = {
        "this": True,        # 函数名称（必需）
        "expressions": False, # 参数表达式列表（可选）
    }
    is_var_len_args = True  # 支持可变长度参数

    @property
    def name(self) -> str:
        """
        获取函数名称。
        
        返回函数的名称，支持字符串和表达式类型。
        
        返回:
            str: 函数名称
        """
        # 如果this是字符串，直接返回；否则返回表达式的名称
        return self.this if isinstance(self.this, str) else self.this.name


class AnonymousAggFunc(AggFunc):
    arg_types = {
        "this": True,        # 函数名称（必需）
        "expressions": False, # 参数表达式列表（可选）
    }
    is_var_len_args = True  # 支持可变长度参数


# 参考：https://clickhouse.com/docs/en/sql-reference/aggregate-functions/combinators
class CombinedAggFunc(AnonymousAggFunc):
    """
    组合聚合函数类。
    
    表示ClickHouse中的组合聚合函数。
    用于组合多个聚合函数的功能。
    
    参考：https://clickhouse.com/docs/en/sql-reference/aggregate-functions/combinators
    
    组合聚合函数的特点：
    - 函数组合：组合多个聚合函数
    - ClickHouse特有：主要用于ClickHouse
    - 高级功能：提供高级聚合功能
    - 性能优化：优化聚合计算性能
    """
    arg_types = {
        "this": True,        # 函数名称（必需）
        "expressions": False, # 参数表达式列表（可选）
    }


class CombinedParameterizedAgg(ParameterizedAgg):
    """
    组合参数化聚合函数类。
    
    表示组合的参数化聚合函数。
    结合了组合聚合函数和参数化聚合函数的功能。
    
    组合参数化聚合的特点：
    - 函数组合：组合多个聚合函数
    - 参数支持：支持额外的参数
    - 高级功能：提供高级聚合功能
    - 灵活配置：可以根据参数调整行为
    """
    arg_types = {
        "this": True,        # 函数名称（必需）
        "expressions": True, # 参数表达式列表（必需）
        "params": True,      # 参数列表（必需）
    }


# 参考：
# https://docs.snowflake.com/en/sql-reference/functions/hll
# https://docs.aws.amazon.com/redshift/latest/dg/r_HLL_function.html
class Hll(AggFunc):
    """
    超对数聚合函数类。
    
    表示SQL中的HLL（HyperLogLog）函数。
    用于高效地计算近似唯一值数量。
    
    参考：
    - https://docs.snowflake.com/en/sql-reference/functions/hll
    - https://docs.aws.amazon.com/redshift/latest/dg/r_HLL_function.html
    
    超对数的特点：
    - 近似计算：使用HyperLogLog算法
    - 内存效率：使用有限内存处理大数据
    - 高精度：在低内存下保持高精度
    - 大数据支持：特别适合大数据场景
    """
    arg_types = {
        "this": True,        # 值表达式（必需）
        "expressions": False, # 参数表达式列表（可选）
    }
    is_var_len_args = True  # 支持可变长度参数


class ApproxDistinct(AggFunc):
    """
    近似去重聚合函数类。
    
    表示SQL中的APPROX_DISTINCT函数。
    用于高效地计算近似唯一值数量。
    
    近似去重的特点：
    - 近似计算：使用近似算法提高性能
    - 内存效率：使用有限内存处理大数据
    - 精度控制：支持精度参数
    - 大数据支持：特别适合大数据场景
    """
    arg_types = {
        "this": True,      # 值表达式（必需）
        "accuracy": False, # 精度参数（可选）
    }
    _sql_names = ["APPROX_DISTINCT", "APPROX_COUNT_DISTINCT"]  # 支持的SQL名称


class Apply(Func):
    """
    应用函数类。
    
    表示SQL中的APPLY函数。
    用于将函数应用到表达式上。
    
    应用函数的特点：
    - 函数应用：将函数应用到表达式
    - 动态调用：支持动态函数调用
    - 灵活配置：支持各种函数调用方式
    - 数据库差异：不同数据库的语法可能不同
    """
    arg_types = {
        "this": True,      # 函数表达式（必需）
        "expression": True, # 应用表达式（必需）
    }


class Array(Func):
    """
    数组构造函数类。
    
    表示SQL中的数组构造函数。
    用于创建数组字面量或数组表达式。
    
    数组构造的特点：
    - 字面量创建：创建数组字面量
    - 表达式数组：从表达式创建数组
    - 可变参数：支持可变数量的元素
    - 类型推断：自动推断数组元素类型
    """
    arg_types = {
        "expressions": False,      # 数组元素表达式列表（可选）
        "bracket_notation": False, # 方括号表示法（可选）
    }
    is_var_len_args = True  # 支持可变长度参数


class Ascii(Func):
    """
    ASCII码函数类。
    
    表示SQL中的ASCII函数。
    用于获取字符的ASCII码值。
    
    ASCII函数的特点：
    - 字符转换：将字符转换为ASCII码
    - 单字符：通常只处理字符串的第一个字符
    - 数值返回：返回整数值
    - 广泛支持：大多数数据库都支持
    """
    pass


# 参考：https://docs.snowflake.com/en/sql-reference/functions/to_array
class ToArray(Func):
    """
    转换为数组函数类。
    
    表示SQL中的TO_ARRAY函数。
    用于将其他类型转换为数组类型。
    
    参考：https://docs.snowflake.com/en/sql-reference/functions/to_array
    
    转换为数组的特点：
    - 类型转换：将其他类型转换为数组
    - Snowflake支持：主要用于Snowflake数据库
    - 数据转换：用于数据格式转换
    - 类型安全：确保转换的类型正确性
    """
    pass


# 参考：https://materialize.com/docs/sql/types/list/
class List(Func):
    """
    列表构造函数类。
    
    表示SQL中的列表构造函数。
    用于创建列表类型的数据结构。
    
    参考：https://materialize.com/docs/sql/types/list/
    
    列表构造的特点：
    - 列表创建：创建列表数据结构
    - Materialize支持：主要用于Materialize数据库
    - 可变参数：支持可变数量的元素
    - 类型推断：自动推断列表元素类型
    """
    arg_types = {
        "expressions": False, # 列表元素表达式列表（可选）
    }
    is_var_len_args = True  # 支持可变长度参数


# 字符串填充，kind True -> LPAD，False -> RPAD
class Pad(Func):
    """
    字符串填充函数类。
    
    表示SQL中的字符串填充函数。
    用于在字符串的左侧或右侧填充字符。
    
    字符串填充的特点：
    - 左填充：LPAD，在字符串左侧填充
    - 右填充：RPAD，在字符串右侧填充
    - 填充字符：可以指定填充字符
    - 长度控制：控制填充后的字符串长度
    """
    arg_types = {
        "this": True,         # 源字符串（必需）
        "expression": True,   # 目标长度（必需）
        "fill_pattern": False, # 填充模式（可选）
        "is_left": True,      # 是否左填充（必需）
    }

# 参考：
# https://docs.snowflake.com/en/sql-reference/functions/to_char
# https://docs.oracle.com/en/database/oracle/oracle-database/23/sqlrf/TO_CHAR-number.html
class ToChar(Func):
    """
    转换为字符串函数类。
    
    表示SQL中的TO_CHAR函数。
    用于将其他类型转换为字符串类型。
    
    参考：
    - https://docs.snowflake.com/en/sql-reference/functions/to_char
    - https://docs.oracle.com/en/database/oracle/oracle-database/23/sqlrf/TO_CHAR-number.html
    
    转换为字符串的特点：
    - 类型转换：将数值、日期等转换为字符串
    - 格式控制：支持格式字符串
    - 本地化：支持本地化参数
    - 广泛支持：大多数数据库都支持
    """
    arg_types = {
        "this": True,     # 要转换的表达式（必需）
        "format": False,  # 格式字符串（可选）
        "nlsparam": False, # 本地化参数（可选）
	"is_numeric": False,
    }


class ToCodePoints(Func):
    pass

# 参考：
# https://docs.snowflake.com/en/sql-reference/functions/to_decimal
# https://docs.oracle.com/en/database/oracle/oracle-database/23/sqlrf/TO_NUMBER.html
class ToNumber(Func):
    """
    转换为数值函数类。
    
    表示SQL中的TO_NUMBER/TO_DECIMAL函数。
    用于将字符串转换为数值类型。
    
    参考：
    - https://docs.snowflake.com/en/sql-reference/functions/to_decimal
    - https://docs.oracle.com/en/database/oracle/oracle-database/23/sqlrf/TO_NUMBER.html
    
    转换为数值的特点：
    - 字符串转换：将字符串转换为数值
    - 格式支持：支持格式字符串
    - 精度控制：支持精度和标度参数
    - 本地化：支持本地化参数
    """
    arg_types = {
        "this": True,      # 要转换的字符串（必需）
        "format": False,   # 格式字符串（可选）
        "nlsparam": False, # 本地化参数（可选）
        "precision": False, # 精度（可选）
        "scale": False,    # 标度（可选）
    }





# 参考：https://docs.snowflake.com/en/sql-reference/functions/to_double
class ToDouble(Func):
    """
    转换为双精度浮点数函数类。
    
    表示SQL中的TO_DOUBLE函数。
    用于将其他类型转换为双精度浮点数。
    
    参考：https://docs.snowflake.com/en/sql-reference/functions/to_double
    
    转换为双精度的特点：
    - 类型转换：将其他类型转换为双精度浮点数
    - 格式支持：支持格式字符串
    - 精度保证：保证双精度精度
    - Snowflake支持：主要用于Snowflake数据库
    """
    arg_types = {
        "this": True,    # 要转换的表达式（必需）
        "format": False, # 格式字符串（可选）
    }


class CodePointsToBytes(Func):
    pass


class Columns(Func):
    """
    列函数类。
    
    表示SQL中的COLUMNS函数。
    用于处理列相关的操作。
    
    列函数的特点：
    - 列操作：处理列相关的操作
    - 解包支持：支持列的解包操作
    - 灵活配置：支持各种列操作选项
    - 数据库差异：不同数据库的语法可能不同
    """
    arg_types = {
        "this": True,    # 列表达式（必需）
        "unpack": False, # 解包选项（可选）
    }


# 参考：https://learn.microsoft.com/en-us/sql/t-sql/functions/cast-and-convert-transact-sql?view=sql-server-ver16#syntax
class Convert(Func):
    """
    转换函数类。
    
    表示SQL中的CONVERT函数。
    用于类型转换操作。
    
    参考：https://learn.microsoft.com/en-us/sql/t-sql/functions/cast-and-convert-transact-sql?view=sql-server-ver16#syntax
    
    转换函数的特点：
    - 类型转换：将一种类型转换为另一种类型
    - 样式支持：支持转换样式参数
    - SQL Server支持：主要用于SQL Server
    - 灵活配置：支持各种转换选项
    """
    arg_types = {
        "this": True,      # 要转换的表达式（必需）
        "expression": True, # 目标类型（必需）
        "style": False,    # 转换样式（可选）
    }


# 参考：https://docs.oracle.com/en/database/oracle/oracle-database/19/sqlrf/CONVERT.html
class ConvertToCharset(Func):
    """
    字符集转换函数类。
    
    表示SQL中的字符集转换函数。
    用于在不同字符集之间转换字符串。
    
    参考：https://docs.oracle.com/en/database/oracle/oracle-database/19/sqlrf/CONVERT.html
    
    字符集转换的特点：
    - 字符集转换：在不同字符集间转换
    - 编码支持：支持各种字符编码
    - Oracle支持：主要用于Oracle数据库
    - 国际化：支持国际化字符处理
    """
    arg_types = {
        "this": True,    # 源字符串（必需）
        "dest": True,    # 目标字符集（必需）
        "source": False, # 源字符集（可选）
    }


class ConvertTimezone(Func):
    """
    时区转换函数类。
    
    表示SQL中的时区转换函数。
    用于在不同时区之间转换时间戳。
    
    时区转换的特点：
    - 时区转换：在不同时区间转换时间
    - 时区支持：支持标准时区名称和偏移量
    - 夏令时：自动处理夏令时转换
    - 精度保持：保持时间戳的精度
    """
    arg_types = {
        "source_tz": False,  # 源时区（可选）
        "target_tz": True,   # 目标时区（必需）
        "timestamp": True,   # 时间戳（必需）
        "options": False,    # 转换选项（可选）
    }


class CodePointsToString(Func):
    pass


class GenerateSeries(Func):
    """
    生成序列函数类。
    
    表示SQL中的GENERATE_SERIES函数。
    用于生成数值序列。
    
    生成序列的特点：
    - 序列生成：生成指定范围的数值序列
    - 步长控制：支持自定义步长
    - 边界控制：支持包含或排除结束值
    - PostgreSQL支持：主要用于PostgreSQL
    """
    arg_types = {
        "start": True,           # 起始值（必需）
        "end": True,             # 结束值（必需）
        "step": False,           # 步长（可选）
        "is_end_exclusive": False, # 是否排除结束值（可选）
    }


# PostgreSQL的GENERATE_SERIES函数返回行集，即当它在投影中使用时隐式展开，
# 所以这个表达式是一个辅助工具，便于转译到其他方言。
# 例如，我们会在DuckDB中生成UNNEST(GENERATE_SERIES(...))
class ExplodingGenerateSeries(GenerateSeries):
    """
    展开生成序列函数类。
    
    表示PostgreSQL的GENERATE_SERIES函数的展开版本。
    用于处理需要显式展开的序列生成。
    
    展开生成序列的特点：
    - 隐式展开：PostgreSQL中隐式展开为行集
    - 转译支持：便于转译到其他数据库方言
    - 显式展开：在其他数据库中需要显式展开
    - 兼容性：提供跨数据库的兼容性
    """
    pass


class ArrayAgg(AggFunc):
    """
    数组聚合函数类。
    
    表示SQL中的ARRAY_AGG函数。
    用于将多行数据聚合为数组。
    
    数组聚合的特点：
    - 行聚合：将多行数据聚合为单行数组
    - NULL处理：支持排除NULL值
    - 分组支持：通常与GROUP BY一起使用
    - 性能优化：数据库通常优化此类函数
    """
    arg_types = {
        "this": True,         # 聚合表达式（必需）
        "nulls_excluded": False, # 是否排除NULL值（可选）
    }


class XMLAgg(AggFunc):
    """
    XML聚合函数类。
    
    表示SQL中的XMLAGG函数，用于将多行XML数据聚合为单个XML文档。
    主要用于Teradata等数据库中的XML数据处理。
    
    XMLAGG语法特点：
    - XML聚合：将多行XML值聚合为单个XML文档
    - ORDER BY支持：支持指定聚合顺序
    - RETURNING子句：支持CONTENT或SEQUENCE返回类型
    - Teradata特有：主要在Teradata数据库中使用
    """
    arg_types = {
        "this": True,           # XML表达式（必需）
        "order": False,         # ORDER BY子句（可选）
        "returning": False,     # RETURNING类型：CONTENT或SEQUENCE（可选）
    }


class ArrayUniqueAgg(AggFunc):
    """
    唯一数组聚合函数类。
    
    表示SQL中的唯一数组聚合函数。
    用于将多行数据聚合为唯一值数组。
    
    唯一数组聚合的特点：
    - 去重聚合：聚合时自动去除重复值
    - 唯一性保证：确保数组中的值唯一
    - 性能考虑：去重操作可能影响性能
    - 数据清理：自动清理重复数据
    """
    pass


class AIAgg(AggFunc):
    arg_types = {"this": True, "expression": True}
    _sql_names = ["AI_AGG"]


class AISummarizeAgg(AggFunc):
    _sql_names = ["AI_SUMMARIZE_AGG"]


class AIClassify(Func):
    arg_types = {"this": True, "categories": True, "config": False}
    _sql_names = ["AI_CLASSIFY"]


class ArrayAll(Func):
    """
    数组全量检查函数类。
    
    表示SQL中的数组全量检查函数。
    用于检查数组中的所有元素是否满足条件。
    
    数组全量检查的特点：
    - 全量检查：检查数组中的所有元素
    - 条件满足：所有元素都必须满足条件
    - 布尔返回：返回TRUE或FALSE
    - 空数组处理：空数组通常返回TRUE
    """
    arg_types = {
        "this": True,      # 数组表达式（必需）
        "expression": True, # 检查条件（必需）
    }


# 表示Python的`any(f(x) for x in array)`，其中`array`是`this`，`f`是`expression`
class ArrayAny(Func):
    """
    数组任意检查函数类。
    
    表示SQL中的数组任意检查函数。
    用于检查数组中是否有任何元素满足条件。
    
    相当于Python的`any(f(x) for x in array)`，其中`array`是`this`，`f`是`expression`
    
    数组任意检查的特点：
    - 任意检查：检查数组中是否有元素满足条件
    - 短路求值：找到满足条件的元素就返回TRUE
    - 布尔返回：返回TRUE或FALSE
    - 空数组处理：空数组通常返回FALSE
    """
    arg_types = {
        "this": True,      # 数组表达式（必需）
        "expression": True, # 检查条件（必需）
    }


class ArrayConcat(Func):
    """
    数组连接函数类。
    
    表示SQL中的数组连接函数。
    用于将多个数组连接成一个数组。
    
    数组连接的特点：
    - 多数组连接：支持连接多个数组
    - 可变参数：支持可变数量的数组参数
    - 顺序保持：保持元素的原始顺序
    - 类型兼容：确保数组元素类型兼容
    """
    _sql_names = ["ARRAY_CONCAT", "ARRAY_CAT"]  # 支持的SQL名称
    arg_types = {
        "this": True,        # 第一个数组（必需）
        "expressions": False, # 其他数组列表（可选）
    }
    is_var_len_args = True  # 支持可变长度参数


class ArrayConcatAgg(AggFunc):
    """
    数组连接聚合函数类。
    
    表示SQL中的数组连接聚合函数。
    用于将多行数组聚合为单个连接数组。
    
    数组连接聚合的特点：
    - 行聚合：将多行数组聚合为单行
    - 连接操作：聚合时进行数组连接
    - 分组支持：通常与GROUP BY一起使用
    - 性能优化：数据库通常优化此类函数
    """
    pass


class ArrayConstructCompact(Func):
    """
    紧凑数组构造函数类。
    
    表示SQL中的紧凑数组构造函数。
    用于构造紧凑的数组，自动去除空值。
    
    紧凑数组构造的特点：
    - 紧凑构造：构造时自动去除空值
    - 空间优化：减少数组的存储空间
    - 数据清理：自动清理无效数据
    - 性能优化：提高数组操作性能
    """
    arg_types = {
        "expressions": True, # 数组元素表达式列表（必需）
    }
    is_var_len_args = True  # 支持可变长度参数


class ArrayContains(Binary, Func):
    """
    数组包含检查函数类。
    
    表示SQL中的数组包含检查函数。
    用于检查数组是否包含指定元素。
    
    数组包含检查的特点：
    - 元素检查：检查数组是否包含指定元素
    - 布尔返回：返回TRUE或FALSE
    - 类型匹配：确保元素类型匹配
    - 性能优化：通常使用索引优化
    """
    arg_types = {"this": True, "expression": True, "ensure_variant": False}
    _sql_names = ["ARRAY_CONTAINS", "ARRAY_HAS"]



class ArrayContainsAll(Binary, Func):
    """
    数组包含全部检查函数类。
    
    表示SQL中的数组包含全部检查函数。
    用于检查数组是否包含所有指定元素。
    
    数组包含全部检查的特点：
    - 全量检查：检查数组是否包含所有指定元素
    - 布尔返回：返回TRUE或FALSE
    - 集合操作：类似于集合的包含关系
    - 性能考虑：全量检查可能影响性能
    """
    _sql_names = ["ARRAY_CONTAINS_ALL", "ARRAY_HAS_ALL"]  # 支持的SQL名称


class ArrayFilter(Func):
    """
    数组过滤函数类。
    
    表示SQL中的数组过滤函数。
    用于根据条件过滤数组元素。
    
    数组过滤的特点：
    - 条件过滤：根据条件过滤数组元素
    - 函数应用：对每个元素应用过滤函数
    - 结果数组：返回过滤后的新数组
    - 性能优化：通常使用并行处理
    """
    arg_types = {
        "this": True,      # 源数组（必需）
        "expression": True, # 过滤条件（必需）
    }
    _sql_names = ["FILTER", "ARRAY_FILTER"]  # 支持的SQL名称


class ArrayFirst(Func):
    """
    数组首元素函数类。
    
    表示SQL中的数组首元素函数。
    用于获取数组的第一个元素。
    
    数组首元素的特点：
    - 首元素获取：获取数组的第一个元素
    - 边界处理：处理空数组的情况
    - 类型保持：保持元素的原始类型
    - 性能优化：直接访问第一个元素
    """
    pass


class ArrayLast(Func):
    """
    数组末元素函数类。
    
    表示SQL中的数组末元素函数。
    用于获取数组的最后一个元素。
    
    数组末元素的特点：
    - 末元素获取：获取数组的最后一个元素
    - 边界处理：处理空数组的情况
    - 类型保持：保持元素的原始类型
    - 性能考虑：需要遍历到最后一个元素
    """
    pass


class ArrayReverse(Func):
    """
    数组反转函数类。
    
    表示SQL中的数组反转函数。
    用于反转数组元素的顺序。
    
    数组反转的特点：
    - 顺序反转：反转数组元素的顺序
    - 原地操作：通常创建新数组
    - 类型保持：保持元素的原始类型
    - 性能考虑：反转操作需要遍历整个数组
    """
    pass


class ArraySlice(Func):
    """
    数组切片函数类。
    
    表示SQL中的数组切片函数。
    用于从数组中提取子数组。
    
    数组切片的特点：
    - 子数组提取：从数组中提取指定范围的元素
    - 索引支持：支持起始和结束索引
    - 步长支持：支持自定义步长
    - 边界处理：处理索引越界的情况
    """
    arg_types = {
        "this": True,   # 源数组（必需）
        "start": True,  # 起始索引（必需）
        "end": False,   # 结束索引（可选）
        "step": False,  # 步长（可选）
    }


class ArrayToString(Func):
    """
    数组转字符串函数类。
    
    表示SQL中的数组转字符串函数。
    用于将数组转换为字符串。
    
    数组转字符串的特点：
    - 字符串转换：将数组转换为字符串
    - 分隔符支持：支持自定义分隔符
    - NULL处理：支持NULL值的处理
    - 格式控制：支持字符串格式控制
    """
    arg_types = {
        "this": True,      # 源数组（必需）
        "expression": True, # 分隔符（必需）
        "null": False,     # NULL值处理（可选）
    }
    _sql_names = ["ARRAY_TO_STRING", "ARRAY_JOIN"]  # 支持的SQL名称


class ArrayIntersect(Func):
    """
    数组交集函数类。
    
    表示SQL中的数组交集函数。
    用于计算多个数组的交集。
    
    数组交集的特点：
    - 交集计算：计算多个数组的交集
    - 去重处理：自动去除重复元素
    - 类型兼容：确保数组元素类型兼容
    - 性能考虑：交集计算可能影响性能
    """
    arg_types = {
        "expressions": True, # 数组表达式列表（必需）
    }
    is_var_len_args = True  # 支持可变长度参数
    _sql_names = ["ARRAY_INTERSECT", "ARRAY_INTERSECTION"]  # 支持的SQL名称


class StPoint(Func):
    """
    空间点函数类。
    
    表示SQL中的ST_POINT函数。
    用于创建空间点对象。
    
    空间点函数的特点：
    - 点创建：创建空间点对象
    - 坐标支持：支持X、Y坐标
    - 空间计算：支持空间计算操作
    - 地理信息：支持地理信息系统
    """
    arg_types = {
        "this": True,      # X坐标（必需）
        "expression": True, # Y坐标（必需）
        "null": False,     # NULL值处理（可选）
    }
    _sql_names = ["ST_POINT", "ST_MAKEPOINT"]  # 支持的SQL名称


class StDistance(Func):
    """
    空间距离函数类。
    
    表示SQL中的ST_DISTANCE函数。
    用于计算两个空间对象之间的距离。
    
    空间距离函数的特点：
    - 距离计算：计算空间对象间的距离
    - 椭球体支持：支持椭球体距离计算
    - 精度控制：支持距离计算精度控制
    - 地理信息：支持地理信息系统
    """
    arg_types = {
        "this": True,        # 第一个空间对象（必需）
        "expression": True,  # 第二个空间对象（必需）
        "use_spheroid": False, # 是否使用椭球体（可选）
    }


# 参考：https://cloud.google.com/bigquery/docs/reference/standard-sql/timestamp_functions#string
class String(Func):
    """
    字符串函数类。
    
    表示SQL中的STRING函数。
    用于将时间戳转换为字符串。
    
    参考：https://cloud.google.com/bigquery/docs/reference/standard-sql/timestamp_functions#string
    
    字符串函数的特点：
    - 时间戳转换：将时间戳转换为字符串
    - 时区支持：支持时区参数
    - 格式控制：支持字符串格式控制
    - BigQuery支持：主要用于BigQuery
    """
    arg_types = {
        "this": True,  # 时间戳表达式（必需）
        "zone": False, # 时区参数（可选）
    }


class StringToArray(Func):
    """
    字符串转数组函数类。
    
    表示SQL中的字符串转数组函数。
    用于将字符串分割为数组。
    
    字符串转数组的特点：
    - 字符串分割：将字符串分割为数组元素
    - 分隔符支持：支持自定义分隔符
    - NULL处理：支持NULL值的处理
    - 格式控制：支持分割格式控制
    """
    arg_types = {
        "this": True,      # 源字符串（必需）
        "expression": False, # 分隔符（可选）
        "null": False,     # NULL值处理（可选）
    }
    _sql_names = ["STRING_TO_ARRAY", "SPLIT_BY_STRING", "STRTOK_TO_ARRAY"]  # 支持的SQL名称


class ArrayOverlaps(Binary, Func):
    """
    数组重叠检查函数类。
    
    表示SQL中的数组重叠检查函数。
    用于检查两个数组是否有重叠元素。
    
    数组重叠检查的特点：
    - 重叠检查：检查两个数组是否有重叠元素
    - 布尔返回：返回TRUE或FALSE
    - 集合操作：类似于集合的交集检查
    - 性能优化：通常使用索引优化
    """
    pass


class ArraySize(Func):
    """
    数组大小函数类。
    
    表示SQL中的数组大小函数。
    用于获取数组的大小（元素数量）。
    
    数组大小的特点：
    - 大小获取：获取数组的元素数量
    - 整数返回：返回整数值
    - 性能优化：直接返回数组大小
    - 边界处理：处理空数组的情况
    """
    arg_types = {
        "this": True,      # 数组表达式（必需）
        "expression": False, # 可选参数（可选）
    }
    _sql_names = ["ARRAY_SIZE", "ARRAY_LENGTH"]  # 支持的SQL名称


class ArraySort(Func):
    """
    数组排序函数类。
    
    表示SQL中的数组排序函数。
    用于对数组元素进行排序。
    
    数组排序的特点：
    - 元素排序：对数组元素进行排序
    - 排序函数：支持自定义排序函数
    - 原地操作：通常创建新数组
    - 性能考虑：排序操作需要比较所有元素
    """
    arg_types = {
        "this": True,      # 源数组（必需）
        "expression": False, # 排序函数（可选）
    }


class ArraySum(Func):
    """
    数组求和函数类。
    
    表示SQL中的数组求和函数。
    用于计算数组元素的和。
    
    数组求和的特点：
    - 元素求和：计算数组元素的和
    - 数值类型：通常用于数值数组
    - 类型提升：结果类型可能提升
    - 性能优化：通常使用并行计算
    """
    arg_types = {
        "this": True,      # 源数组（必需）
        "expression": False, # 可选参数（可选）
    }


class ArrayUnionAgg(AggFunc):
    """
    数组并集聚合函数类。
    
    表示SQL中的数组并集聚合函数。
    用于将多行数组聚合为并集数组。
    
    数组并集聚合的特点：
    - 并集聚合：将多行数组聚合为并集
    - 去重处理：自动去除重复元素
    - 分组支持：通常与GROUP BY一起使用
    - 性能优化：数据库通常优化此类函数
    """
    pass


class Avg(AggFunc):
    """
    平均值聚合函数类。
    
    表示SQL中的AVG函数。
    用于计算数值的平均值。
    
    平均值函数的特点：
    - 数值聚合：计算一组数值的平均值
    - NULL处理：自动忽略NULL值
    - 精度控制：保持适当的计算精度
    - 广泛支持：所有数据库都支持
    
    语法示例：
    - SELECT AVG(salary) FROM employees
    - SELECT AVG(age) GROUP BY department
    """
    pass


class AnyValue(AggFunc):
    """
    任意值聚合函数类。
    
    表示SQL中的ANY_VALUE函数。
    用于从一组值中返回任意一个非NULL值。
    
    任意值函数的特点：
    - 非确定性：返回的值可能不确定
    - NULL跳过：优先返回非NULL值
    - 性能优化：通常比其他聚合函数更快
    - 分组使用：通常在GROUP BY中使用
    
    应用场景：
    - 获取组中的任意代表值
    - 性能优化的聚合查询
    - 数据采样
    """
    pass


class Lag(AggFunc):
    """
    滞后窗口函数类。
    
    表示SQL中的LAG函数。
    用于访问当前行之前的行的值。
    
    滞后函数的特点：
    - 向前查看：访问当前行之前的行
    - 偏移控制：可以指定向前偏移的行数
    - 默认值：当没有前面的行时返回默认值
    - 窗口函数：通常与OVER子句一起使用
    
    语法示例：
    - LAG(column) OVER (ORDER BY date)
    - LAG(column, 2, 0) OVER (PARTITION BY id ORDER BY date)
    """
    arg_types = {
        "this": True,    # 要访问的列（必需）
        "offset": False, # 偏移量（可选，默认为1）
        "default": False, # 默认值（可选，默认为NULL）
    }


class Lead(AggFunc):
    """
    超前窗口函数类。
    
    表示SQL中的LEAD函数。
    用于访问当前行之后的行的值。
    
    超前函数的特点：
    - 向后查看：访问当前行之后的行
    - 偏移控制：可以指定向后偏移的行数
    - 默认值：当没有后面的行时返回默认值
    - 窗口函数：通常与OVER子句一起使用
    
    语法示例：
    - LEAD(column) OVER (ORDER BY date)
    - LEAD(column, 2, 0) OVER (PARTITION BY id ORDER BY date)
    """
    arg_types = {
        "this": True,    # 要访问的列（必需）
        "offset": False, # 偏移量（可选，默认为1）
        "default": False, # 默认值（可选，默认为NULL）
    }


# 某些方言区分first和first_value，通常first是聚合函数，
# 而first_value是窗口函数
class First(AggFunc):
    """
    第一个值聚合函数类。
    
    表示SQL中的FIRST函数。
    用于获取一组值中的第一个值。
    
    第一个值函数的特点：
    - 聚合函数：作为聚合函数使用
    - 顺序相关：结果依赖于数据的顺序
    - 分组使用：通常在GROUP BY中使用
    - 方言差异：与FIRST_VALUE在某些数据库中有区别
    
    语法示例：
    - SELECT FIRST(column) FROM table GROUP BY category
    """
    arg_types = {"this": True, "expression": False}


class Last(AggFunc):
    """
    最后一个值聚合函数类。
    
    表示SQL中的LAST函数。
    用于获取一组值中的最后一个值。
    
    最后一个值函数的特点：
    - 聚合函数：作为聚合函数使用
    - 顺序相关：结果依赖于数据的顺序
    - 分组使用：通常在GROUP BY中使用
    - 方言差异：与LAST_VALUE在某些数据库中有区别
    
    语法示例：
    - SELECT LAST(column) FROM table GROUP BY category
    """
    arg_types = {"this": True, "expression": False}


class FirstValue(AggFunc):
    """
    第一个值窗口函数类。
    
    表示SQL中的FIRST_VALUE函数。
    用于在窗口中获取第一个值。
    
    第一个值窗口函数的特点：
    - 窗口函数：作为窗口函数使用
    - 窗口边界：在指定窗口内获取第一个值
    - 排序相关：结果依赖于ORDER BY
    - 帧控制：支持窗口帧的控制
    
    语法示例：
    - FIRST_VALUE(column) OVER (ORDER BY date)
    - FIRST_VALUE(column) OVER (PARTITION BY id ORDER BY date)
    """
    pass


class LastValue(AggFunc):
    """
    最后一个值窗口函数类。
    
    表示SQL中的LAST_VALUE函数。
    用于在窗口中获取最后一个值。
    
    最后一个值窗口函数的特点：
    - 窗口函数：作为窗口函数使用
    - 窗口边界：在指定窗口内获取最后一个值
    - 排序相关：结果依赖于ORDER BY
    - 帧控制：支持窗口帧的控制
    
    语法示例：
    - LAST_VALUE(column) OVER (ORDER BY date)
    - LAST_VALUE(column) OVER (PARTITION BY id ORDER BY date)
    """
    pass


class NthValue(AggFunc):
    """
    第N个值窗口函数类。
    
    表示SQL中的NTH_VALUE函数。
    用于在窗口中获取第N个值。
    
    第N个值窗口函数的特点：
    - 窗口函数：作为窗口函数使用
    - 位置指定：可以指定获取第几个值
    - 窗口边界：在指定窗口内获取指定位置的值
    - 边界处理：处理超出窗口范围的情况
    
    语法示例：
    - NTH_VALUE(column, 3) OVER (ORDER BY date)
    - NTH_VALUE(column, 2) OVER (PARTITION BY id ORDER BY date)
    """
    arg_types = {
        "this": True,   # 要访问的列（必需）
        "offset": True, # 位置偏移量（必需）
    }


class Case(Func):
    """
    条件表达式类。
    
    表示SQL中的CASE表达式。
    用于实现条件逻辑，类似于编程语言中的if-else语句。
    
    CASE表达式的特点：
    - 条件分支：支持多个条件分支
    - 类型统一：所有分支的返回类型必须兼容
    - 短路求值：按顺序评估条件，找到匹配就停止
    - 默认值：支持ELSE子句作为默认值
    
    语法示例：
    - CASE WHEN condition1 THEN value1 WHEN condition2 THEN value2 ELSE default END
    - CASE column WHEN value1 THEN result1 WHEN value2 THEN result2 ELSE default END
    """
    arg_types = {
        "this": False,   # CASE表达式的主体（可选，用于简单CASE）
        "ifs": True,     # IF条件列表（必需）
        "default": False, # 默认值（ELSE子句）（可选）
    }

    def when(self, condition: ExpOrStr, then: ExpOrStr, copy: bool = True, **opts) -> Case:
        """
        添加WHEN条件分支。
        
        为CASE表达式添加一个新的WHEN-THEN条件分支。
        
        参数:
            condition: WHEN条件表达式
            then: THEN结果表达式
            copy: 是否复制当前实例
            **opts: 解析选项
            
        返回:
            Case: 添加了新条件分支的CASE表达式
        """
        # 根据copy参数决定是否复制当前实例
        instance = maybe_copy(self, copy)
        
        # 添加新的IF条件到ifs列表中
        # If对象包含条件和对应的结果
        instance.append(
            "ifs",
            If(
                this=maybe_parse(condition, copy=copy, **opts),  # 解析条件表达式
                true=maybe_parse(then, copy=copy, **opts),       # 解析结果表达式
            ),
        )
        return instance

    def else_(self, condition: ExpOrStr, copy: bool = True, **opts) -> Case:
        """
        设置ELSE默认值。
        
        为CASE表达式设置ELSE子句的默认值。
        
        参数:
            condition: 默认值表达式
            copy: 是否复制当前实例
            **opts: 解析选项
            
        返回:
            Case: 设置了默认值的CASE表达式
        """
        # 根据copy参数决定是否复制当前实例
        instance = maybe_copy(self, copy)
        
        # 设置default参数为解析后的条件表达式
        instance.set("default", maybe_parse(condition, copy=copy, **opts))
        return instance


class Cast(Func):
    """
    类型转换函数类。
    
    表示SQL中的CAST函数。
    用于将一种数据类型转换为另一种数据类型。
    
    类型转换的特点：
    - 显式转换：明确指定目标类型
    - 格式支持：支持转换格式参数
    - 安全转换：支持安全转换选项
    - 错误处理：支持转换失败时的处理
    
    语法示例：
    - CAST(column AS INTEGER)
    - CAST(date_string AS DATE FORMAT 'YYYY-MM-DD')
    - SAFE_CAST(string_value AS INTEGER)
    """
    arg_types = {
        "this": True,     # 要转换的表达式（必需）
        "to": True,       # 目标数据类型（必需）
        "format": False,  # 转换格式（可选）
        "safe": False,    # 安全转换选项（可选）
        "action": False,  # 转换动作（可选）
        "default": False, # 默认值（可选）
    }

    @property
    def name(self) -> str:
        """
        获取被转换表达式的名称。
        
        返回:
            str: 被转换表达式的名称
        """
        # 返回被转换表达式的名称
        return self.this.name

    @property
    def to(self) -> DataType:
        """
        获取目标数据类型。
        
        返回:
            DataType: 目标数据类型对象
        """
        # 返回目标数据类型
        return self.args["to"]

    @property
    def output_name(self) -> str:
        """
        获取输出名称。
        
        对于CAST表达式，输出名称就是被转换表达式的名称。
        
        返回:
            str: 输出名称
        """
        # CAST表达式的输出名称就是原表达式的名称
        return self.name

    def is_type(self, *dtypes: DATA_TYPE) -> bool:
        """
        检查CAST的目标类型是否匹配提供的数据类型之一。
        
        嵌套类型如数组或结构体将使用"结构等价"语义进行比较，
        例如 array<int> != array<float>。
        
        参数:
            dtypes: 要比较的数据类型列表
            
        返回:
            bool: 如果目标类型与提供的类型之一匹配则返回True
        """
        # 使用目标类型的is_type方法进行比较
        return self.to.is_type(*dtypes)


class TryCast(Cast):
    """
    尝试类型转换函数类。
    
    表示SQL中的TRY_CAST函数。
    用于尝试类型转换，转换失败时返回NULL而不是报错。
    
    尝试转换的特点：
    - 安全转换：转换失败时返回NULL
    - 错误处理：避免转换错误导致查询失败
    - 类型检查：在运行时进行类型检查
    - 字符串要求：某些情况下需要字符串输入
    
    语法示例：
    - TRY_CAST(column AS INTEGER)
    - TRY_CAST(string_value AS DATE)
    """
    # 继承Cast的arg_types，并添加requires_string参数
    arg_types = {**Cast.arg_types, "requires_string": False}


# 参考：https://clickhouse.com/docs/sql-reference/data-types/newjson#reading-json-paths-as-sub-columns
class JSONCast(Cast):
    """
    JSON类型转换函数类。
    
    表示ClickHouse中的JSON类型转换。
    用于将JSON路径读取为子列。
    
    参考：https://clickhouse.com/docs/sql-reference/data-types/newjson#reading-json-paths-as-sub-columns
    
    JSON转换的特点：
    - JSON路径：支持JSON路径表达式
    - 子列读取：将JSON路径读取为列
    - ClickHouse特有：主要用于ClickHouse数据库
    - 动态类型：支持动态类型推断
    
    应用场景：
    - JSON数据处理
    - 半结构化数据查询
    - 动态列提取
    """
    pass


# PostgreSQL日期间隔调整函数：处理日期间隔的规范化显示
class JustifyDays(Func):
    """将小时数转换为天数的PostgreSQL函数"""
    pass


class JustifyHours(Func):
    """将分钟数转换为小时数的PostgreSQL函数"""
    pass


class JustifyInterval(Func):
    """同时调整天数和小时数的PostgreSQL综合间隔调整函数"""
    pass


# 异常处理函数：用于捕获和处理SQL执行中的错误
class Try(Func):
    """尝试执行表达式，如果失败则返回NULL的函数"""
    pass


# 类型转换函数：将值转换为字符串类型
class CastToStrType(Func):
    """强制类型转换为字符串类型的函数"""
    # arg_types定义了函数参数的要求：this为必需参数（被转换的值），to为目标类型
    arg_types = {"this": True, "to": True}


# Teradata字符翻译函数：用于字符串中字符的替换操作
# 参考文档：https://docs.teradata.com/r/Enterprise_IntelliFlex_VMware/SQL-Functions-Expressions-and-Predicates/String-Operators-and-Functions/TRANSLATE/TRANSLATE-Function-Syntax
class TranslateCharacters(Expression):
    """Teradata的TRANSLATE函数，用于字符串字符替换"""
    # with_error参数为可选，用于控制遇到错误时的行为
    arg_types = {"this": True, "expression": True, "with_error": False}


# 排序规则函数：继承自Binary和Func，用于指定字符串比较的排序规则
class Collate(Binary, Func):
    """指定字符串排序规则的函数，影响字符串比较和排序行为"""
    pass

class Collation(Func):
    pass

# 向上取整函数：数学运算函数
class Ceil(Func):
    """向上取整函数，支持指定小数位数"""
    # decimals和to为可选参数，用于控制精度和目标类型
    arg_types = {"this": True, "decimals": False, "to": False}
    # _sql_names列表包含了不同数据库中的函数名变体，确保跨数据库兼容性
    _sql_names = ["CEIL", "CEILING"]


# 合并函数：返回第一个非NULL值
class Coalesce(Func):
    """返回参数列表中第一个非NULL值的函数"""
    # is_nvl和is_null用于标识函数的特殊行为模式
    arg_types = {"this": True, "expressions": False, "is_nvl": False, "is_null": False}
    # is_var_len_args=True表示该函数支持可变长度参数列表
    is_var_len_args = True
    # 包含多个SQL方言中的等价函数名，实现跨数据库的语义统一
    _sql_names = ["COALESCE", "IFNULL", "NVL"]


# 字符函数：将ASCII码转换为字符
class Chr(Func):
    """将ASCII码值转换为对应字符的函数"""
    # charset参数为可选，用于指定字符集
    arg_types = {"expressions": True, "charset": False}
    # 支持多个表达式参数，允许批量转换
    is_var_len_args = True
    # CHR和CHAR是不同数据库中的等价函数名
    _sql_names = ["CHR", "CHAR"]


# 字符串连接函数：将多个字符串连接成一个
class Concat(Func):
    """字符串连接函数，将多个字符串合并为一个"""
    # safe参数用于安全模式，coalesce用于处理NULL值
    arg_types = {"expressions": True, "safe": False, "coalesce": False}
    # 支持可变长度参数，可以连接任意数量的字符串
    is_var_len_args = True


# 带分隔符的字符串连接函数：继承自Concat
class ConcatWs(Concat):
    """使用指定分隔符连接字符串的函数，第一个参数为分隔符"""
    _sql_names = ["CONCAT_WS"]



# https://cloud.google.com/bigquery/docs/reference/standard-sql/string_functions#contains_substr
# 包含检查函数：检查字符串是否包含指定子串
class Contains(Func):
    """检查字符串中是否包含指定子字符串的函数"""
    arg_types = {"this": True, "expression": True, "json_scope": False}


# Oracle层次查询函数：用于树形结构查询
# 参考文档：https://docs.oracle.com/cd/B13789_01/server.101/b10759/operators004.htm#i1035022
class ConnectByRoot(Func):
    """Oracle的CONNECT BY ROOT操作符，用于层次查询中获取根节点值"""
    pass


# 计数聚合函数：统计记录数量
class Count(AggFunc):
    """计数聚合函数，统计满足条件的记录数量"""
    # this为可选（COUNT(*)时不需要），big_int用于大数值处理
    arg_types = {"this": False, "expressions": False, "big_int": False}
    # 支持多种计数方式：COUNT(*)、COUNT(col1, col2, ...)
    is_var_len_args = True


# 条件计数聚合函数：基于条件的计数
class CountIf(AggFunc):
    """基于条件表达式的计数聚合函数"""
    # 不同数据库中条件计数函数的名称变体
    _sql_names = ["COUNT_IF", "COUNTIF"]


# 立方根函数：数学运算
class Cbrt(Func):
    """计算立方根的数学函数"""
    pass


# 当前日期函数：获取系统当前日期
class CurrentDate(Func):
    """获取当前日期的函数（不包含时间部分）"""
    # this为可选参数，某些数据库可能需要括号
    arg_types = {"this": False}


# 当前日期时间函数：获取完整的日期时间
class CurrentDatetime(Func):
    """获取当前日期和时间的函数"""
    arg_types = {"this": False}


# 当前时间函数：获取当前时间
class CurrentTime(Func):
    """获取当前时间的函数（不包含日期部分）"""
    arg_types = {"this": False}


# 当前时间戳函数：获取精确时间戳
class CurrentTimestamp(Func):
    """获取当前时间戳的函数，包含完整的日期时间信息"""
    # sysdate参数用于Oracle的SYSDATE函数行为
    arg_types = {"this": False, "sysdate": False}


# 本地时区当前时间戳函数：考虑时区的时间戳
class CurrentTimestampLTZ(Func):
    """获取本地时区当前时间戳的函数"""
    # 不需要任何参数
    arg_types = {}


# 当前模式函数：获取当前数据库模式
class CurrentSchema(Func):
    """获取当前数据库模式（schema）名称的函数"""
    arg_types = {"this": False}


# 当前用户函数：获取当前登录用户
class CurrentUser(Func):
    """获取当前数据库用户名的函数"""
    arg_types = {"this": False}

class UtcDate(Func):
    arg_types = {}


class UtcTime(Func):
    arg_types = {"this": False}


class UtcTimestamp(Func):
    arg_types = {"this": False}

# 日期加法函数：在日期上添加时间间隔
class DateAdd(Func, IntervalOp):
    """在指定日期上添加时间间隔的函数"""
    # this为基准日期，expression为要添加的数值，unit为时间单位
    arg_types = {"this": True, "expression": True, "unit": False}


# 日期分组函数：将时间戳分组到指定的时间间隔内
class DateBin(Func, IntervalOp):
    """将时间戳按指定间隔进行分组的函数，常用于时间序列数据聚合"""
    # zone参数用于处理时区转换，在不同时区间隔计算时很重要
    arg_types = {"this": True, "expression": True, "unit": False, "zone": False, "origin": False}


# 日期减法函数：从日期中减去时间间隔
class DateSub(Func, IntervalOp):
    """从指定日期中减去时间间隔的函数"""
    arg_types = {"this": True, "expression": True, "unit": False}


# 日期差值计算函数：计算两个日期之间的时间差
class DateDiff(Func, TimeUnit):
    """计算两个日期之间差值的函数"""
    # 支持多种数据库的函数名变体，确保跨平台兼容性
    _sql_names = ["DATEDIFF", "DATE_DIFF"]
    # zone参数用于时区相关的日期差值计算
    arg_types = {"this": True, "expression": True, "unit": False, "zone": False}


# 日期截断函数：将日期截断到指定的时间单位
class DateTrunc(Func):
    """将日期截断到指定时间单位的函数，例如截断到月初、年初等"""
    # unit和this的参数顺序在不同数据库中可能不同，需要统一处理
    arg_types = {"unit": True, "this": True, "zone": False}

    def __init__(self, **args):
        # 在大多数数据库中可以安全地展开时间单位缩写（如'Q' -> 'QUARTER'），但Oracle除外
        # 参考文档：https://docs.oracle.com/en/database/oracle/oracle-database/21/sqlrf/ROUND-and-TRUNC-Date-Functions.html
        # unabbreviate控制是否展开缩写，默认为True以提高可读性
        unabbreviate = args.pop("unabbreviate", True)

        unit = args.get("unit")
        # 检查是否为时间单位变量类型，需要进行名称标准化处理
        if isinstance(unit, TimeUnit.VAR_LIKE) and not (
            isinstance(unit, Column) and len(unit.parts) != 1
        ):
            unit_name = unit.name.upper()
            # 如果启用展开且存在于映射表中，则转换为完整名称
            if unabbreviate and unit_name in TimeUnit.UNABBREVIATED_UNIT_NAME:
                unit_name = TimeUnit.UNABBREVIATED_UNIT_NAME[unit_name]

            # 将处理后的单位名称转换为字符串字面量，确保SQL生成正确
            args["unit"] = Literal.string(unit_name)

        super().__init__(**args)

    @property
    def unit(self) -> Expression:
        """获取时间单位参数的属性访问器"""
        return self.args["unit"]


# BigQuery日期时间函数：创建或转换日期时间对象
# 参考文档：https://cloud.google.com/bigquery/docs/reference/standard-sql/datetime_functions#datetime
# expression参数可以是时间表达式或时区信息
class Datetime(Func):
    """创建或转换日期时间对象的函数，支持时区处理"""
    # expression为可选，可以是时间表达式或时区
    arg_types = {"this": True, "expression": False}


# 日期时间加法函数：在日期时间上添加间隔
class DatetimeAdd(Func, IntervalOp):
    """在日期时间值上添加指定间隔的函数"""
    arg_types = {"this": True, "expression": True, "unit": False}


# 日期时间减法函数：从日期时间中减去间隔
class DatetimeSub(Func, IntervalOp):
    """从日期时间值中减去指定间隔的函数"""
    arg_types = {"this": True, "expression": True, "unit": False}


# 日期时间差值函数：计算两个日期时间的差值
class DatetimeDiff(Func, TimeUnit):
    """计算两个日期时间值之间差值的函数"""
    arg_types = {"this": True, "expression": True, "unit": False}


# 日期时间截断函数：截断日期时间到指定精度
class DatetimeTrunc(Func, TimeUnit):
    """将日期时间截断到指定时间单位的函数"""
    # zone参数处理时区相关的截断操作
    arg_types = {"this": True, "unit": True, "zone": False}


# Unix日期转换函数：将Unix日期数值转换为标准日期
class DateFromUnixDate(Func):
    """将Unix日期数值（自1970-01-01以来的天数）转换为日期对象的函数"""
    pass


# 一周中的第几天函数：返回日期在一周中的位置
class DayOfWeek(Func):
    """返回指定日期在一周中第几天的函数"""
    # 不同数据库使用不同的函数名，需要映射处理
    _sql_names = ["DAY_OF_WEEK", "DAYOFWEEK"]


# ISO标准一周中的第几天函数：按ISO标准计算星期几
# 参考文档：https://duckdb.org/docs/sql/functions/datepart.html#part-specifiers-only-usable-as-date-part-specifiers
# DuckDB中的ISO星期几函数名为ISODOW
class DayOfWeekIso(Func):
    """按ISO标准返回一周中第几天的函数（周一为1，周日为7）"""
    # ISO标准与某些数据库的默认标准不同，需要特殊处理
    _sql_names = ["DAYOFWEEK_ISO", "ISODOW"]


# 一个月中的第几天函数：返回日期在月份中的天数
class DayOfMonth(Func):
    """返回指定日期在当月中第几天的函数"""
    _sql_names = ["DAY_OF_MONTH", "DAYOFMONTH"]


# 一年中的第几天函数：返回日期在年份中的天数
class DayOfYear(Func):
    """返回指定日期在当年中第几天的函数（1-366）"""
    _sql_names = ["DAY_OF_YEAR", "DAYOFYEAR"]


# 转换为天数函数：将日期转换为数值天数
class ToDays(Func):
    """将日期转换为从某个基准日期开始的天数的函数"""
    pass


# 一年中的第几周函数：返回日期在年份中的周数
class WeekOfYear(Func):
    """返回指定日期在当年中第几周的函数"""
    _sql_names = ["WEEK_OF_YEAR", "WEEKOFYEAR"]

class YearOfWeek(Func):
    _sql_names = ["YEAR_OF_WEEK", "YEAROFWEEK"]

class YearOfWeekIso(Func):
    _sql_names = ["YEAR_OF_WEEK_ISO", "YEAROFWEEKISO"]

# 月份间隔函数：计算两个日期之间的月份数差
class MonthsBetween(Func):
    """计算两个日期之间相差多少个月的函数"""
    # roundoff参数控制是否对结果进行四舍五入处理
    arg_types = {"this": True, "expression": True, "roundoff": False}


# 创建时间间隔函数：根据指定的时间组件创建间隔对象
class MakeInterval(Func):
    """根据年、月、日、时、分、秒等组件创建时间间隔对象的函数"""
    # 所有时间组件都是可选的，可以灵活组合创建不同的间隔
    arg_types = {
        "year": False,      # 年份组件
        "month": False,     # 月份组件
        "day": False,       # 天数组件
        "hour": False,      # 小时组件
        "minute": False,    # 分钟组件
        "second": False,    # 秒数组件
    }


# 月末日期函数：获取指定日期所在月份的最后一天
class LastDay(Func, TimeUnit):
    """获取指定日期所在月份最后一天的函数"""
    # 支持不同的函数名变体，提供更好的兼容性
    _sql_names = ["LAST_DAY", "LAST_DAY_OF_MONTH"]
    # unit参数可以指定其他时间单位的"最后一天"概念
    arg_types = {"this": True, "unit": False}


class PreviousDay(Func):
    arg_types = {"this": True, "expression": True}


class LaxBool(Func):
    pass


class LaxFloat64(Func):
    pass


class LaxInt64(Func):
    pass


class LaxString(Func):
    pass


# 提取时间部分函数：从日期时间中提取指定的时间组件
class Extract(Func):
    """从日期时间值中提取指定时间组件（年、月、日等）的函数"""
    # this为要提取的时间组件名，expression为日期时间值
    arg_types = {"this": True, "expression": True}


# 存在性检查函数：检查子查询是否返回结果
class Exists(Func, SubqueryPredicate):
    """检查子查询是否存在匹配记录的谓词函数"""
    # 继承SubqueryPredicate表明这是用于子查询的谓词
    # expression为可选，用于某些特殊的存在性检查场景
    arg_types = {"this": True, "expression": False}


# 时间戳函数：创建或转换时间戳对象
class Timestamp(Func):
    """创建或转换时间戳对象的函数"""
    # with_tz参数控制是否包含时区信息，影响时间戳的表示方式
    arg_types = {"this": False, "zone": False, "with_tz": False}


# 时间戳加法函数：在时间戳上添加时间间隔
class TimestampAdd(Func, TimeUnit):
    """在时间戳上添加指定时间间隔的函数"""
    arg_types = {"this": True, "expression": True, "unit": False}


# 时间戳减法函数：从时间戳中减去时间间隔
class TimestampSub(Func, TimeUnit):
    """从时间戳中减去指定时间间隔的函数"""
    arg_types = {"this": True, "expression": True, "unit": False}


# 时间戳差值函数：计算两个时间戳的差值
class TimestampDiff(Func, TimeUnit):
    """计算两个时间戳之间差值的函数"""
    # 支持多种数据库的函数名，确保广泛兼容性
    _sql_names = ["TIMESTAMPDIFF", "TIMESTAMP_DIFF"]
    arg_types = {"this": True, "expression": True, "unit": False}


# 时间戳截断函数：将时间戳截断到指定精度
class TimestampTrunc(Func, TimeUnit):
    """将时间戳截断到指定时间单位的函数"""
    # zone参数用于时区相关的截断操作
    arg_types = {"this": True, "unit": True, "zone": False}


class TimeSlice(Func, TimeUnit):
    arg_types = {"this": True, "expression": True, "unit": True, "kind": False}
    
    
# 时间加法函数：在时间值上添加间隔
class TimeAdd(Func, TimeUnit):
    """在时间值上添加指定间隔的函数"""
    arg_types = {"this": True, "expression": True, "unit": False}


# 时间减法函数：从时间值中减去间隔
class TimeSub(Func, TimeUnit):
    """从时间值中减去指定间隔的函数"""
    arg_types = {"this": True, "expression": True, "unit": False}


# 时间差值函数：计算两个时间值的差值
class TimeDiff(Func, TimeUnit):
    """计算两个时间值之间差值的函数"""
    arg_types = {"this": True, "expression": True, "unit": False}


# 时间截断函数：将时间截断到指定精度
class TimeTrunc(Func, TimeUnit):
    """将时间值截断到指定时间单位的函数"""
    arg_types = {"this": True, "unit": True, "zone": False}


# 从组件构造日期函数：通过年月日组件创建日期
class DateFromParts(Func):
    """通过年、月、日组件构造日期对象的函数"""
    # 不同数据库使用不同的函数名，需要统一映射
    _sql_names = ["DATE_FROM_PARTS", "DATEFROMPARTS"]
    # 年月日都是必需参数，确保日期的完整性
    arg_types = {"year": True, "month": True, "day": True}


# 从组件构造时间函数：通过时分秒组件创建时间
class TimeFromParts(Func):
    """通过时、分、秒等组件构造时间对象的函数"""
    _sql_names = ["TIME_FROM_PARTS", "TIMEFROMPARTS"]
    arg_types = {
        "hour": True,       # 小时（必需）
        "min": True,        # 分钟（必需）
        "sec": True,        # 秒（必需）
        "nano": False,      # 纳秒（可选，用于高精度时间）
        "fractions": False, # 小数秒（可选）
        "precision": False, # 精度控制（可选）
    }


# 日期字符串转日期函数：将字符串转换为日期对象
class DateStrToDate(Func):
    """将日期字符串转换为日期对象的函数"""
    pass


# 日期转日期字符串函数：将日期对象转换为字符串
class DateToDateStr(Func):
    """将日期对象转换为日期字符串的函数"""
    pass


# 日期转整数函数：将日期转换为整数表示
class DateToDi(Func):
    """将日期转换为整数表示的函数（通常用于特定数据库的内部格式）"""
    pass


# BigQuery日期函数：创建日期对象
# 参考文档：https://cloud.google.com/bigquery/docs/reference/standard-sql/date_functions#date
class Date(Func):
    """创建或转换日期对象的通用函数"""
    # expressions支持多种参数组合，is_var_len_args允许灵活的参数数量
    arg_types = {"this": False, "zone": False, "expressions": False}
    # 支持可变长度参数，可以接受不同数量的输入参数
    is_var_len_args = True


# 日期中的天数函数：从日期中提取天数部分
class Day(Func):
    """从日期中提取天数部分的函数"""
    pass


# 解码函数：将编码的字符串按指定字符集解码
class Decode(Func):
    """将编码字符串按指定字符集解码为原始字符串的函数"""
    # charset指定解码使用的字符集，replace控制遇到无法解码字符时的处理方式
    arg_types = {"this": True, "charset": True, "replace": False}


# Oracle风格的DECODE条件函数：类似于CASE WHEN的多条件判断
class DecodeCase(Func):
    """Oracle的DECODE函数，提供多分支条件判断功能，类似于CASE表达式"""
    # expressions包含条件-值对的序列，支持可变长度参数
    arg_types = {"expressions": True}
    # 支持多个条件-值对：DECODE(expr, search1, result1, search2, result2, ..., default)
    is_var_len_args = True


class DenseRank(AggFunc):
    arg_types = {"expressions": False}
    is_var_len_args = True


# 整数转日期函数：将数值型日期标识符转换为日期对象
class DiToDate(Func):
    """将数值型日期标识符（DI）转换为标准日期对象的函数"""
    pass


# 编码函数：将字符串按指定字符集编码
class Encode(Func):
    """将字符串按指定字符集编码的函数"""
    # charset参数指定编码使用的字符集类型
    arg_types = {"this": True, "charset": True}


class EqualNull(Func):
    arg_types = {"this": True, "expression": True}


# 指数函数：计算e的指定次幂
class Exp(Func):
    """计算自然对数底数e的指定次幂的数学函数"""
    pass

class Factorial(Func):
    pass

# https://docs.snowflake.com/en/sql-reference/functions/flatten
class Explode(Func, UDTF):
    """将数组、MAP或结构体展开为多行的表值函数"""
    # 继承UDTF表明这是用户定义的表值函数，返回表格结果
    # expressions为可选，用于指定额外的展开参数
    arg_types = {"this": True, "expressions": False}
    # 支持可变参数，可以同时展开多个数组或结构
    is_var_len_args = True


# Spark内联函数：将结构体数组展开为多行多列
# 参考文档：https://spark.apache.org/docs/latest/api/sql/#inline
class Inline(Func):
    """Spark的INLINE函数，将结构体数组展开为表格形式"""
    pass


# 外部展开函数：包含NULL值的数组展开
class ExplodeOuter(Explode):
    """EXPLODE的OUTER版本，即使数组为空或NULL也会产生一行NULL结果"""
    pass


# 带位置的展开函数：展开数组同时返回元素位置
class Posexplode(Explode):
    """展开数组并返回每个元素位置索引的函数"""
    pass


# 带位置的外部展开函数：结合位置信息和OUTER语义
class PosexplodeOuter(Posexplode, ExplodeOuter):
    """POSEXPLODE的OUTER版本，同时提供位置信息和NULL处理"""
    pass


# 位置列表达式：表示基于位置的列引用
class PositionalColumn(Expression):
    """基于位置索引的列引用表达式，用于SELECT中的位置参数"""
    pass


# 数组拆分函数：PostgreSQL风格的数组展开
class Unnest(Func, UDTF):
    """PostgreSQL的UNNEST函数，将数组展开为行集合"""
    arg_types = {
        "expressions": True,        # 要展开的数组表达式
        "alias": False,            # 结果列的别名
        "offset": False,           # 是否包含偏移量列
        "explode_array": False,    # 是否按数组方式展开
    }

    @property
    def selects(self) -> t.List[Expression]:
        """生成SELECT子句中的列列表，包括可能的偏移量列"""
        # 获取基础的选择列
        columns = super().selects
        offset = self.args.get("offset")
        # 如果启用了偏移量，添加offset列到结果中
        if offset:
            # offset为True时使用默认名称，否则使用指定的偏移量列名
            columns = columns + [to_identifier("offset") if offset is True else offset]
        return columns


# 向下取整函数：数学函数，向负无穷方向取整
class Floor(Func):
    """向下取整函数，返回不大于给定数值的最大整数"""
    # decimals指定小数位数，to指定目标数据类型
    arg_types = {"this": True, "decimals": False, "to": False}


class FromBase32(Func):
    pass


# Base64解码函数：将Base64编码字符串解码为原始数据
class FromBase64(Func):
    """将Base64编码的字符串解码为原始二进制数据的函数"""
    pass


class ToBase32(Func):
    pass

# Base64编码函数：将数据编码为Base64字符串
class ToBase64(Func):
    """将二进制数据编码为Base64字符串的函数"""
    pass


# https://docs.snowflake.com/en/sql-reference/functions/base64_decode_binary
class Base64DecodeBinary(Func):
    arg_types = {"this": True, "alphabet": False}


# https://docs.snowflake.com/en/sql-reference/functions/base64_decode_string
class Base64DecodeString(Func):
    arg_types = {"this": True, "alphabet": False}


# https://docs.snowflake.com/en/sql-reference/functions/base64_encode
class Base64Encode(Func):
    arg_types = {"this": True, "max_line_length": False, "alphabet": False}


# https://docs.snowflake.com/en/sql-reference/functions/try_base64_decode_binary
class TryBase64DecodeBinary(Func):
    arg_types = {"this": True, "alphabet": False}


# https://docs.snowflake.com/en/sql-reference/functions/try_base64_decode_string
class TryBase64DecodeString(Func):
    arg_types = {"this": True, "alphabet": False}


# https://docs.snowflake.com/en/sql-reference/functions/try_hex_decode_binary
class TryHexDecodeBinary(Func):
    pass


# https://docs.snowflake.com/en/sql-reference/functions/try_hex_decode_string
class TryHexDecodeString(Func):
    pass


# https://trino.io/docs/current/functions/datetime.html#from_iso8601_timestamp
class FromISO8601Timestamp(Func):
    """将ISO8601格式的字符串转换为时间戳的函数"""
    # 使用明确的函数名映射，确保与Trino兼容
    _sql_names = ["FROM_ISO8601_TIMESTAMP"]


# 间隙填充函数：填充时间序列数据中的缺失时间点
class GapFill(Func):
    """时间序列数据的间隙填充函数，用于补全缺失的时间点"""
    arg_types = {
        "this": True,                   # 数据源表或查询
        "ts_column": True,              # 时间戳列名（必需）
        "bucket_width": True,           # 时间间隔宽度（必需）
        "partitioning_columns": False,  # 分区列，用于分组填充
        "value_columns": False,         # 需要填充的值列
        "origin": False,                # 时间轴的起始点
        "ignore_nulls": False,          # 是否忽略NULL值
    }


# BigQuery日期数组生成函数：生成指定范围内的日期数组
# 参考文档：https://cloud.google.com/bigquery/docs/reference/standard-sql/array_functions#generate_date_array
class GenerateDateArray(Func):
    """生成指定日期范围内的日期数组的BigQuery函数"""
    # step为可选，默认为1天间隔
    arg_types = {"start": True, "end": True, "step": False}


# BigQuery时间戳数组生成函数：生成指定范围内的时间戳数组
# 参考文档：https://cloud.google.com/bigquery/docs/reference/standard-sql/array_functions#generate_timestamp_array
class GenerateTimestampArray(Func):
    """生成指定时间戳范围内的时间戳数组的BigQuery函数"""
    # step为必需参数，指定时间间隔
    arg_types = {"start": True, "end": True, "step": True}


# Snowflake GET函数：从半结构化数据中提取值
# 参考文档：https://docs.snowflake.com/en/sql-reference/functions/get
class GetExtract(Func):
    """从半结构化数据（JSON、VARIANT等）中提取指定路径值的Snowflake函数"""
    # this为数据源，expression为提取路径
    arg_types = {"this": True, "expression": True}
    
class Getbit(Func):
    arg_types = {"this": True, "expression": True}


# 最大值函数：返回多个值中的最大值
class Greatest(Func):
    """返回多个输入值中最大值的函数"""
    # expressions为可选，支持任意数量的比较值
    arg_types = {"this": True, "expressions": False}
    # 支持可变长度参数：GREATEST(val1, val2, val3, ...)
    is_var_len_args = True


class GreatestIgnoreNulls(Func):
    arg_types = {"expressions": True}
    is_var_len_args = True


class LeastIgnoreNulls(Func):
    arg_types = {"expressions": True}
    is_var_len_args = True
    
    
# Trino溢出截断行为配置：控制聚合函数溢出时的行为
# Trino的 `ON OVERFLOW TRUNCATE [filler_string] {WITH | WITHOUT} COUNT` 语法
# 参考文档：https://trino.io/docs/current/functions/aggregate.html#listagg
class OverflowTruncateBehavior(Expression):
    """Trino中控制聚合函数溢出截断行为的配置表达式"""
    # this为可选的填充字符串，with_count控制是否显示截断计数
    arg_types = {"this": False, "with_count": True}


# 分组连接聚合函数：将组内的值连接成字符串
class GroupConcat(AggFunc):
    """将分组内的多个字符串值连接成单个字符串的聚合函数"""
    # separator指定连接分隔符，on_overflow控制溢出处理策略
    arg_types = {"this": True, "separator": False, "on_overflow": False}


# 十六进制转换函数：将数值转换为十六进制字符串
class Hex(Func):
    """将数值转换为十六进制字符串表示的函数"""
    pass

# https://docs.snowflake.com/en/sql-reference/functions/hex_decode_string
class HexDecodeString(Func):
    pass


# https://docs.snowflake.com/en/sql-reference/functions/hex_encode
class HexEncode(Func):
    arg_types = {"this": True, "case": False}


class Hour(Func):
    pass


class Minute(Func):
    pass


class Second(Func):
    pass


# T-SQL: https://learn.microsoft.com/en-us/sql/t-sql/functions/compress-transact-sql?view=sql-server-ver17
# Snowflake: https://docs.snowflake.com/en/sql-reference/functions/compress
class Compress(Func):
    arg_types = {"this": True, "method": False}


# Snowflake: https://docs.snowflake.com/en/sql-reference/functions/decompress_binary
class DecompressBinary(Func):
    arg_types = {"this": True, "method": True}


# Snowflake: https://docs.snowflake.com/en/sql-reference/functions/decompress_string
class DecompressString(Func):
    arg_types = {"this": True, "method": True}


# 小写十六进制函数：继承自Hex，返回小写的十六进制字符串
class LowerHex(Hex):
    """返回小写十六进制字符串的函数，继承自Hex"""
    pass


# 逻辑与连接器：逻辑AND操作
class And(Connector, Func):
    """逻辑AND操作符，用于连接多个布尔条件"""
    # 继承Connector表示这是一个连接器，可以连接多个表达式
    pass


# 逻辑或连接器：逻辑OR操作
class Or(Connector, Func):
    """逻辑OR操作符，用于连接多个布尔条件"""
    pass


# 逻辑异或连接器：逻辑XOR操作
class Xor(Connector, Func):
    """逻辑XOR（异或）操作符，当且仅当一个条件为真时返回真"""
    # 支持多种参数形式：单个表达式、两个表达式或表达式列表
    arg_types = {"this": False, "expression": False, "expressions": False}


# 条件判断函数：类似三元操作符的IF表达式
class If(Func):
    """条件判断函数，根据条件返回不同的值：IF(condition, true_value, false_value)"""
    # false参数为可选，某些数据库的IF函数可以省略else部分
    arg_types = {"this": True, "true": True, "false": False}
    # IIF是SQL Server的变体名称，与IF功能相同
    _sql_names = ["IF", "IIF"]


# NULL值比较函数：如果两个值相等则返回NULL
class Nullif(Func):
    """如果两个参数相等则返回NULL，否则返回第一个参数的函数"""
    arg_types = {"this": True, "expression": True}


# 首字母大写函数：将字符串的首字母转换为大写
class Initcap(Func):
    """将字符串中每个单词的首字母转换为大写的函数"""
    # expression为可选，用于指定分隔符或其他格式化选项
    arg_types = {"this": True, "expression": False}


# ASCII字符检查函数：检查字符串是否只包含ASCII字符
class IsAscii(Func):
    """检查字符串是否只包含ASCII字符的函数"""
    pass


# NaN检查函数：检查数值是否为NaN（非数字）
class IsNan(Func):
    """检查数值是否为NaN（Not a Number）的函数"""
    # 不同数据库使用不同的函数名
    _sql_names = ["IS_NAN", "ISNAN"]


# BigQuery的64位整数转换函数：用于JSON处理
# 参考文档：https://cloud.google.com/bigquery/docs/reference/standard-sql/json_functions#int64_for_json
class Int64(Func):
    """BigQuery中将值转换为64位整数的函数，主要用于JSON数据处理"""
    pass


# 无穷大检查函数：检查数值是否为无穷大
class IsInf(Func):
    """检查数值是否为无穷大（正无穷或负无穷）的函数"""
    _sql_names = ["IS_INF", "ISINF"]


class IsNullValue(Func):
    pass
    
    
# JSON表达式：PostgreSQL的JSON数据类型表示
# 参考文档：https://www.postgresql.org/docs/current/functions-json.html
class JSON(Expression):
    """JSON数据类型表达式，用于表示JSON值和相关操作"""
    # with用于WITH子句，unique控制键的唯一性
    arg_types = {"this": False, "with": False, "unique": False}


# JSON路径表达式：用于在JSON数据中导航的路径
class JSONPath(Expression):
    """JSON路径表达式，用于在JSON结构中定位特定元素"""
    # expressions包含路径的各个部分，escape控制转义字符处理
    arg_types = {"expressions": True, "escape": False}

    @property
    def output_name(self) -> str:
        """从路径的最后一部分提取输出列名"""
        # 获取路径的最后一个段作为输出名称，这通常是最终访问的字段名
        last_segment = self.expressions[-1].this
        # 只有当最后一段是字符串时才返回，否则返回空字符串
        return last_segment if isinstance(last_segment, str) else ""


# JSON路径组件基类：所有JSON路径部分的基础类
class JSONPathPart(Expression):
    """JSON路径中各种组件的基础类"""
    arg_types = {}


# JSON路径过滤器：用于条件过滤的路径组件
class JSONPathFilter(JSONPathPart):
    """JSON路径中的过滤器组件，用于根据条件筛选元素"""
    arg_types = {"this": True}


# JSON路径键：用于访问对象键的路径组件
class JSONPathKey(JSONPathPart):
    """JSON路径中的键组件，用于访问JSON对象的特定键"""
    arg_types = {"this": True}


# JSON路径递归：用于递归搜索的路径组件
class JSONPathRecursive(JSONPathPart):
    """JSON路径中的递归组件，用于递归搜索所有嵌套层级"""
    # this为可选，可以指定递归的起始条件
    arg_types = {"this": False}


# JSON路径根：表示JSON路径的根节点
class JSONPathRoot(JSONPathPart):
    """JSON路径的根节点，表示路径的起始点"""
    pass


# JSON路径脚本：用于执行脚本表达式的路径组件
class JSONPathScript(JSONPathPart):
    """JSON路径中的脚本组件，用于执行动态表达式"""
    arg_types = {"this": True}


# JSON路径切片：用于数组切片的路径组件
class JSONPathSlice(JSONPathPart):
    """JSON路径中的数组切片组件，类似Python的切片语法"""
    # 支持start:end:step的切片语法，所有参数都是可选的
    arg_types = {"start": False, "end": False, "step": False}


# JSON路径选择器：用于选择特定元素的路径组件
class JSONPathSelector(JSONPathPart):
    """JSON路径中的选择器组件，用于选择特定的元素"""
    arg_types = {"this": True}


# JSON路径下标：用于数组索引访问的路径组件
class JSONPathSubscript(JSONPathPart):
    """JSON路径中的下标组件，用于通过索引访问数组元素"""
    arg_types = {"this": True}


# JSON路径联合：用于组合多个路径的路径组件
class JSONPathUnion(JSONPathPart):
    """JSON路径中的联合组件，用于同时匹配多个路径"""
    # expressions包含要联合的多个路径表达式
    arg_types = {"expressions": True}


# JSON路径通配符：匹配所有元素的路径组件
class JSONPathWildcard(JSONPathPart):
    """JSON路径中的通配符组件，匹配当前层级的所有元素"""
    pass


# JSON格式化表达式：用于JSON格式化的表达式
class FormatJson(Expression):
    """用于JSON数据格式化的表达式"""
    pass


class Format(Func):
    arg_types = {"this": True, "expressions": False}
    is_var_len_args = True


# JSON键值对：表示JSON对象中的键值对
class JSONKeyValue(Expression):
    """JSON对象中的键值对表达式"""
    # this为键，expression为值
    arg_types = {"this": True, "expression": True}

# https://cloud.google.com/bigquery/docs/reference/standard-sql/json_functions#json_keys
class JSONKeysAtDepth(Func):
    arg_types = {"this": True, "expression": False, "mode": False}


# JSON对象构造函数：创建JSON对象
class JSONObject(Func):
    """构造JSON对象的函数，从键值对创建JSON对象"""
    arg_types = {
        "expressions": False,      # 键值对表达式列表
        "null_handling": False,    # NULL值处理方式
        "unique_keys": False,      # 是否要求键唯一
        "return_type": False,      # 返回值类型
        "encoding": False,         # 编码方式
    }


# JSON对象聚合函数：将多行数据聚合为JSON对象
class JSONObjectAgg(AggFunc):
    """将多行键值对聚合为单个JSON对象的聚合函数"""
    # 参数与JSONObject相同，但这是聚合函数
    arg_types = {
        "expressions": False,
        "null_handling": False,
        "unique_keys": False,
        "return_type": False,
        "encoding": False,
    }


# PostgreSQL的JSONB对象聚合函数：性能更优的二进制JSON聚合
# 参考文档：https://www.postgresql.org/docs/9.5/functions-aggregate.html
class JSONBObjectAgg(AggFunc):
    """PostgreSQL的JSONB对象聚合函数，将键值对聚合为JSONB对象"""
    # JSONB是PostgreSQL的二进制JSON格式，性能更好
    arg_types = {"this": True, "expression": True}


# Oracle的JSON数组构造函数：创建JSON数组
# 参考文档：https://docs.oracle.com/en/database/oracle/oracle-database/19/sqlrf/JSON_ARRAY.html
class JSONArray(Func):
    """Oracle的JSON数组构造函数，从多个值创建JSON数组"""
    arg_types = {
        "expressions": False,     # 数组元素表达式
        "null_handling": False,   # NULL值处理策略
        "return_type": False,     # 返回类型规范
        "strict": False,          # 严格模式，控制类型检查
    }


# Oracle的JSON数组聚合函数：将多行聚合为JSON数组
# 参考文档：https://docs.oracle.com/en/database/oracle/oracle-database/19/sqlrf/JSON_ARRAYAGG.html
class JSONArrayAgg(Func):
    """Oracle的JSON数组聚合函数，将多行值聚合为JSON数组"""
    arg_types = {
        "this": True,            # 要聚合的表达式
        "order": False,          # 排序子句
        "null_handling": False,  # NULL处理方式
        "return_type": False,    # 返回类型
        "strict": False,         # 严格模式
    }


# JSON存在性检查函数：检查JSON路径是否存在
class JSONExists(Func):
    """检查JSON数据中指定路径是否存在的函数"""
    # passing用于传递变量，on_condition处理错误条件
    arg_types = {"this": True, "path": True, "passing": False, "on_condition": False}


# Oracle的JSON表列定义：定义JSON_TABLE的列结构
# 参考文档：https://docs.oracle.com/en/database/oracle/oracle-database/19/sqlrf/JSON_TABLE.html
# 注意：JSON列定义的解析目前还不完整
class JSONColumnDef(Expression):
    """Oracle JSON_TABLE函数中的列定义表达式"""
    # nested_schema用于嵌套的JSON结构定义
    arg_types = {
        "this": False,
        "kind": False,
        "path": False,
        "nested_schema": False,
        "ordinality": False,
    }



# JSON模式定义：描述JSON数据的结构模式
class JSONSchema(Expression):
    """JSON数据结构的模式定义表达式"""
    # expressions包含模式的各个组件定义
    arg_types = {"expressions": True}

class JSONSet(Func):
    arg_types = {"this": True, "expressions": True}
    is_var_len_args = True
    _sql_names = ["JSON_SET"]


# https://cloud.google.com/bigquery/docs/reference/standard-sql/json_functions#json_strip_nulls
class JSONStripNulls(Func):
    arg_types = {
        "this": True,
        "expression": False,
        "include_arrays": False,
        "remove_empty": False,
    }
    _sql_names = ["JSON_STRIP_NULLS"]


# MySQL的JSON值提取表达式：从JSON中提取标量值
# 参考文档：https://dev.mysql.com/doc/refman/8.4/en/json-search-functions.html#function_json-value
class JSONValue(Expression):
    """MySQL的JSON_VALUE表达式，从JSON中提取标量值"""
    arg_types = {
        "this": True,           # JSON数据源
        "path": True,           # 提取路径
        "returning": False,     # 返回类型规范
        "on_condition": False,  # 错误处理条件
    }


# JSON值数组函数：提取JSON数组值
class JSONValueArray(Func):
    """从JSON中提取数组值的函数"""
    arg_types = {"this": True, "expression": False}

class JSONRemove(Func):
    arg_types = {"this": True, "expressions": True}
    is_var_len_args = True
    _sql_names = ["JSON_REMOVE"]


# Oracle的JSON表函数：将JSON数据转换为关系表格
# 参考文档：https://docs.oracle.com/en/database/oracle/oracle-database/19/sqlrf/JSON_TABLE.html
class JSONTable(Func):
    """Oracle的JSON_TABLE函数，将JSON数据转换为关系表格形式"""
    arg_types = {
        "this": True,             # JSON数据源
        "schema": True,           # 表结构模式定义
        "path": False,            # 根路径
        "error_handling": False,  # 错误处理策略
        "empty_handling": False,  # 空值处理策略
    }


# JSON类型检测函数：返回JSON值的数据类型
# 参考文档：
# https://cloud.google.com/bigquery/docs/reference/standard-sql/json_functions#json_type
# https://doris.apache.org/docs/sql-manual/sql-functions/scalar-functions/json-functions/json-type#description
class JSONType(Func):
    """检测JSON值数据类型的函数，返回类型字符串（如'string', 'number'等）"""
    # expression为可选，用于某些数据库的额外参数
    arg_types = {"this": True, "expression": False}
    _sql_names = ["JSON_TYPE"]


# Snowflake的对象插入函数：向JSON对象中插入键值对
# 参考文档：https://docs.snowflake.com/en/sql-reference/functions/object_insert
class ObjectInsert(Func):
    """Snowflake的OBJECT_INSERT函数，向JSON对象中插入或更新键值对"""
    arg_types = {
        "this": True,          # 目标JSON对象
        "key": True,           # 要插入的键
        "value": True,         # 要插入的值
        "update_flag": False,  # 是否更新已存在的键
    }


# SQL Server的OpenJSON列定义：定义OpenJSON的返回列
class OpenJSONColumnDef(Expression):
    """SQL Server OpenJSON函数的列定义表达式"""
    # as_json指示是否将值作为JSON返回
    arg_types = {"this": True, "kind": True, "path": False, "as_json": False}


# SQL Server的OpenJSON函数：解析JSON字符串为表格
class OpenJSON(Func):
    """SQL Server的OPENJSON函数，将JSON字符串解析为表格形式"""
    # expressions为列定义列表
    arg_types = {"this": True, "path": False, "expressions": False}


# PostgreSQL的JSONB包含检查：检查JSONB是否包含另一个JSONB
class JSONBContains(Binary, Func):
    """PostgreSQL的JSONB包含检查函数，检查一个JSONB是否包含另一个"""
    # 继承Binary表示这是二元操作
    _sql_names = ["JSONB_CONTAINS"]

# https://www.postgresql.org/docs/9.5/functions-json.html
class JSONBContainsAnyTopKeys(Binary, Func):
    pass


# https://www.postgresql.org/docs/9.5/functions-json.html
class JSONBContainsAllTopKeys(Binary, Func):
    pass


# PostgreSQL的JSONB存在性检查：检查JSONB路径是否存在
class JSONBExists(Func):
    """PostgreSQL的JSONB路径存在性检查函数"""
    arg_types = {"this": True, "path": True}
    _sql_names = ["JSONB_EXISTS"]


# https://www.postgresql.org/docs/9.5/functions-json.html
class JSONBDeleteAtPath(Binary, Func):
    pass

class JSONExtract(Binary, Func):
    """从JSON数据中提取值的通用函数，支持多种提取选项"""
    arg_types = {
        "this": True,              # JSON数据源
        "expression": True,        # 提取路径
        "only_json_types": False,  # 是否只返回JSON类型
        "expressions": False,      # 多个路径表达式
        "variant_extract": False,  # Snowflake的VARIANT提取
        "json_query": False,       # JSON查询模式
        "option": False,           # 提取选项
        "quote": False,            # 引号处理
        "on_condition": False,     # 错误条件处理
        "requires_json": False,    # 是否要求输入为JSON
    }
    _sql_names = ["JSON_EXTRACT"]
    # 支持可变长度参数，可以同时提取多个路径
    is_var_len_args = True

    @property
    def output_name(self) -> str:
        """根据提取路径确定输出列名"""
        # 如果没有多个表达式，使用单个表达式的输出名
        return self.expression.output_name if not self.expressions else ""


# Trino的JSON提取引号配置：控制JSON查询的引号行为
# 参考文档：https://trino.io/docs/current/functions/json.html#json-query
class JSONExtractQuote(Expression):
    """Trino JSON查询中的引号处理配置表达式"""
    # scalar控制是否返回标量值
    arg_types = {
        "option": True,
        "scalar": False,
    }


# JSON数组提取函数：提取JSON数组
class JSONExtractArray(Func):
    """从JSON中提取数组的函数"""
    arg_types = {"this": True, "expression": False}
    _sql_names = ["JSON_EXTRACT_ARRAY"]


# JSON标量提取函数：提取JSON标量值
class JSONExtractScalar(Binary, Func):
    """从JSON中提取标量值（非对象、非数组）的函数"""
    # only_json_types控制是否只返回JSON原生类型
    arg_types = {
        "this": True,
        "expression": True,
        "only_json_types": False,
        "expressions": False,
        "json_type": False,
    }
    _sql_names = ["JSON_EXTRACT_SCALAR"]
    is_var_len_args = True

    @property
    def output_name(self) -> str:
        """使用提取路径的输出名作为列名"""
        return self.expression.output_name


# PostgreSQL的JSONB提取函数：从JSONB中提取值
class JSONBExtract(Binary, Func):
    """PostgreSQL的JSONB值提取函数"""
    _sql_names = ["JSONB_EXTRACT"]


# PostgreSQL的JSONB标量提取函数：从JSONB中提取标量值
class JSONBExtractScalar(Binary, Func):
    """PostgreSQL的JSONB标量值提取函数"""
    arg_types = {"this": True, "expression": True, "json_type": False}
    _sql_names = ["JSONB_EXTRACT_SCALAR"]


# JSON格式化函数：格式化JSON输出
class JSONFormat(Func):
    """格式化JSON输出的函数，控制JSON的显示格式"""
    # is_json指示输入是否已经是JSON格式
    arg_types = {"this": False, "options": False, "is_json": False, "to_json": False}
    _sql_names = ["JSON_FORMAT"]


class JSONArrayAppend(Func):
    arg_types = {"this": True, "expressions": True}
    is_var_len_args = True
    _sql_names = ["JSON_ARRAY_APPEND"]

# https://dev.mysql.com/doc/refman/8.0/en/json-search-functions.html#operator_member-of
class JSONArrayContains(Binary, Predicate, Func):
    """MySQL的JSON数组包含检查函数，检查值是否在JSON数组中"""
    # 继承Predicate表示这是一个谓词函数，返回布尔值
    arg_types = {"this": True, "expression": True, "json_type": False}
    _sql_names = ["JSON_ARRAY_CONTAINS"]


class JSONArrayInsert(Func):
    arg_types = {"this": True, "expressions": True}
    is_var_len_args = True
    _sql_names = ["JSON_ARRAY_INSERT"]


class ParseBignumeric(Func):
    pass


class ParseNumeric(Func):
    pass

class ParseJSON(Func):
    """将JSON字符串解析为JSON对象的函数"""
    # BigQuery和Snowflake使用PARSE_JSON，Presto使用JSON_PARSE
    # Snowflake还有TRY_PARSE_JSON，通过safe参数表示
    _sql_names = ["PARSE_JSON", "JSON_PARSE"]
    # safe参数表示是否使用安全模式（遇到错误返回NULL而不是抛出异常）
    arg_types = {"this": True, "expression": False, "safe": False}


# Snowflake: https://docs.snowflake.com/en/sql-reference/functions/parse_url
# Databricks: https://docs.databricks.com/aws/en/sql/language-manual/functions/parse_url
class ParseUrl(Func):
    arg_types = {"this": True, "part_to_extract": False, "key": False, "permissive": False}


class ParseIp(Func):
    arg_types = {"this": True, "type": True, "permissive": False}

class ParseTime(Func):
    """将时间字符串按指定格式解析为时间对象的函数"""
    # format参数是必需的，用于指定时间字符串的格式模式
    arg_types = {"this": True, "format": True}


# 日期时间解析函数：将字符串解析为日期时间对象
class ParseDatetime(Func):
    """将日期时间字符串解析为日期时间对象的函数"""
    # format为可选，某些数据库可以自动推断格式；zone用于时区处理
    arg_types = {"this": True, "format": False, "zone": False}


# 最小值函数：返回多个值中的最小值
class Least(Func):
    """返回多个输入值中最小值的函数"""
    # expressions为可选，支持任意数量的比较值
    arg_types = {"this": True, "expressions": False}
    # 支持可变长度参数：LEAST(val1, val2, val3, ...)
    is_var_len_args = True


# 左子字符串函数：从字符串左侧提取指定长度的子串
class Left(Func):
    """从字符串左侧提取指定长度子字符串的函数"""
    # expression指定要提取的字符数量
    arg_types = {"this": True, "expression": True}


# 右子字符串函数：从字符串右侧提取指定长度的子串
class Right(Func):
    """从字符串右侧提取指定长度子字符串的函数"""
    arg_types = {"this": True, "expression": True}


class Reverse(Func):
    pass

class Length(Func):
    """计算字符串长度的函数，支持不同的计算方式"""
    # binary控制是否按字节计算，encoding指定字符编码
    arg_types = {"this": True, "binary": False, "encoding": False}
    # 不同数据库使用不同的函数名，需要统一映射
    _sql_names = ["LENGTH", "LEN", "CHAR_LENGTH", "CHARACTER_LENGTH"]


class RtrimmedLength(Func):
    pass


class BitLength(Func):
    pass


class Levenshtein(Func):
    """计算两个字符串之间编辑距离的函数，用于字符串相似度分析"""
    arg_types = {
        "this": True,           # 第一个字符串
        "expression": False,    # 第二个字符串（可选）
        "ins_cost": False,      # 插入操作的成本权重
        "del_cost": False,      # 删除操作的成本权重
        "sub_cost": False,      # 替换操作的成本权重
        "max_dist": False,      # 最大距离阈值，超过则不计算
    }


# 自然对数函数：计算自然对数
class Ln(Func):
    """计算自然对数（以e为底）的数学函数"""
    pass


# 对数函数：计算指定底数的对数
class Log(Func):
    """计算对数的数学函数，可指定底数"""
    # expression为可选，指定对数的底数，默认通常为10
    arg_types = {"this": True, "expression": False}


# 逻辑或聚合函数：对布尔值进行OR聚合
class LogicalOr(AggFunc):
    """对多个布尔值进行逻辑OR聚合的函数，任一为真则结果为真"""
    # 不同数据库使用不同的函数名
    _sql_names = ["LOGICAL_OR", "BOOL_OR", "BOOLOR_AGG"]


# 逻辑与聚合函数：对布尔值进行AND聚合
class LogicalAnd(AggFunc):
    """对多个布尔值进行逻辑AND聚合的函数，全部为真则结果为真"""
    _sql_names = ["LOGICAL_AND", "BOOL_AND", "BOOLAND_AGG"]


# 小写转换函数：将字符串转换为小写
class Lower(Func):
    """将字符串转换为小写的函数"""
    # LCASE是某些数据库的别名
    _sql_names = ["LOWER", "LCASE"]


# 映射数据结构：表示键值对的映射类型
class Map(Func):
    """创建和操作映射（键值对）数据结构的函数"""
    # keys和values都是可选，可以为空映射
    arg_types = {"keys": False, "values": False}

    @property
    def keys(self) -> t.List[Expression]:
        """获取映射中所有键的列表"""
        # 从参数中提取keys，如果存在则返回其表达式列表
        keys = self.args.get("keys")
        # 安全地获取表达式列表，避免None值引起的错误
        return keys.expressions if keys else []

    @property
    def values(self) -> t.List[Expression]:
        """获取映射中所有值的列表"""
        # 类似地处理values参数
        values = self.args.get("values")
        return values.expressions if values else []


# DuckDB的映射转换函数：将结构体转换为映射
# 表示DuckDB中的MAP {...}语法 - 基本上是将struct转换为MAP
class ToMap(Func):
    """DuckDB的函数，将结构体转换为映射类型"""
    pass


# 从条目创建映射函数：从键值对数组创建映射
class MapFromEntries(Func):
    """从键值对条目数组创建映射对象的函数"""
    pass


# SQL Server的作用域解析操作符：用于命名空间解析
# 参考文档：https://learn.microsoft.com/en-us/sql/t-sql/language-elements/scope-resolution-operator-transact-sql?view=sql-server-ver16
class ScopeResolution(Expression):
    """SQL Server的作用域解析操作符（::），用于解析命名空间和对象引用"""
    # this为可选的作用域名，expression为要解析的对象名
    arg_types = {"this": False, "expression": True}


# 流表达式：表示数据流概念
class Stream(Expression):
    """表示数据流的表达式，用于流式数据处理"""
    pass


# 星型映射函数：特殊的映射操作
class StarMap(Func):
    """星型映射函数，用于特殊的映射变换操作"""
    pass


# 可变参数映射函数：支持可变数量键值对的映射
class VarMap(Func):
    """支持可变长度参数的映射构造函数"""
    # keys和values都是必需的，且支持可变长度参数
    arg_types = {"keys": True, "values": True}
    # 允许传入任意数量的键值对
    is_var_len_args = True

    @property
    def keys(self) -> t.List[Expression]:
        """获取映射的键列表"""
        # 直接返回keys参数的表达式列表，因为keys是必需的
        return self.args["keys"].expressions

    @property
    def values(self) -> t.List[Expression]:
        """获取映射的值列表"""
        # 同样直接返回values参数的表达式列表
        return self.args["values"].expressions


# MySQL全文搜索函数：MATCH AGAINST全文检索
# 参考文档：https://dev.mysql.com/doc/refman/8.0/en/fulltext-search.html
class MatchAgainst(Func):
    """MySQL的MATCH AGAINST全文搜索函数"""
    # expressions为要搜索的关键词列表，modifier为搜索模式修饰符
    arg_types = {"this": True, "expressions": True, "modifier": False}


# 最大值聚合函数：返回一组值中的最大值
class Max(AggFunc):
    """返回一组值中最大值的聚合函数"""
    # expressions为可选，支持多列的最大值比较
    arg_types = {"this": True, "expressions": False}
    # 支持可变参数：MAX(col1, col2, ...)某些数据库支持多列比较
    is_var_len_args = True


# MD5哈希函数：计算MD5哈希值
class MD5(Func):
    """计算MD5哈希值的函数，返回十六进制字符串"""
    _sql_names = ["MD5"]


# MD5摘要函数：返回二进制MD5哈希值
# 表示返回二进制值的MD5函数变体
class MD5Digest(Func):
    """计算MD5哈希值并返回二进制结果的函数"""
    # 与MD5不同，这个版本返回二进制数据而非十六进制字符串
    _sql_names = ["MD5_DIGEST"]


# https://docs.snowflake.com/en/sql-reference/functions/md5_number_lower64
class MD5NumberLower64(Func):
    pass


# https://docs.snowflake.com/en/sql-reference/functions/md5_number_upper64
class MD5NumberUpper64(Func):
    pass


class Median(AggFunc):
    """计算一组数值中位数的聚合函数"""
    pass


class Min(AggFunc):
    """返回一组值中最小值的聚合函数"""
    # 类似Max，支持多列比较
    arg_types = {"this": True, "expressions": False}
    is_var_len_args = True


class Month(Func):
    """从日期中提取月份部分的函数"""
    pass


class Monthname(Func):
    pass

class AddMonths(Func):
    """在指定日期上添加指定月数的函数"""
    # expression为要添加的月数
    arg_types = {"this": True, "expression": True}


class Nvl2(Func):
    """Oracle的NVL2函数，根据第一个参数是否为NULL返回不同的值"""
    # 如果this不为NULL，返回true；如果this为NULL，返回false（可选）
    arg_types = {"this": True, "true": True, "false": False}


class Ntile(AggFunc):
    arg_types = {"this": False}


class Normalize(Func):
    """将Unicode字符串按指定形式进行规范化的函数"""
    # form指定规范化形式（NFC、NFD、NFKC、NFKD等）
    arg_types = {"this": True, "form": False, "is_casefold": False}


class Overlay(Func):
    """在字符串的指定位置覆盖另一个字符串的函数"""
    # from指定起始位置，for指定覆盖长度（可选）
    arg_types = {"this": True, "expression": True, "from": True, "for": False}


# https://cloud.google.com/bigquery/docs/reference/standard-sql/bigqueryml-syntax-predict#mlpredict_function
class Predict(Func):
    """BigQuery ML的预测函数，使用机器学习模型对数据进行预测"""
    # this为模型，expression为输入数据，params_struct为预测参数
    arg_types = {"this": True, "expression": True, "params_struct": False}

# https://cloud.google.com/bigquery/docs/reference/standard-sql/bigqueryml-syntax-translate#mltranslate_function
class MLTranslate(Func):
    arg_types = {"this": True, "expression": True, "params_struct": True}


# https://cloud.google.com/bigquery/docs/reference/standard-sql/bigqueryml-syntax-feature-time
class FeaturesAtTime(Func):
    """在指定时间点获取特征数据的时间序列分析函数"""
    arg_types = {
        "this": True,                    # 特征数据源
        "time": False,                   # 目标时间点
        "num_rows": False,              # 返回的行数限制
        "ignore_feature_nulls": False,  # 是否忽略特征中的NULL值
    }


# https://cloud.google.com/bigquery/docs/reference/standard-sql/bigqueryml-syntax-generate-embedding
class GenerateEmbedding(Func):
    arg_types = {"this": True, "expression": True, "params_struct": False, "is_text": False}


class MLForecast(Func):
    arg_types = {"this": True, "expression": False, "params_struct": False}


# Represents Snowflake's <model>!<attribute> syntax. For example: SELECT model!PREDICT(INPUT_DATA => {*})
# See: https://docs.snowflake.com/en/guides-overview-ml-functions
class ModelAttribute(Expression):
    arg_types = {"this": True, "expression": True}


# https://cloud.google.com/bigquery/docs/reference/standard-sql/search_functions#vector_search
class VectorSearch(Func):
    arg_types = {
        "this": True,
        "column_to_search": True,
        "query_table": True,
        "query_column_to_search": False,
        "top_k": False,
        "distance_type": False,
        "options": False,
    }


class Pi(Func):
    arg_types = {}


class Pow(Binary, Func):
    """计算幂次方的数学函数，返回底数的指数次幂"""
    # 继承Binary表示这是二元操作：底数^指数
    _sql_names = ["POWER", "POW"]


class PercentileCont(AggFunc):
    """计算连续分布百分位数的聚合函数，使用线性插值"""
    # expression为可选的排序表达式，用于指定计算基准
    arg_types = {"this": True, "expression": False}


class PercentileDisc(AggFunc):
    """计算离散分布百分位数的聚合函数，返回实际存在的值"""
    # 与PercentileCont不同，不进行插值，只返回数据集中实际存在的值
    arg_types = {"this": True, "expression": False}


class PercentRank(AggFunc):
    arg_types = {"expressions": False}
    is_var_len_args = True


class Quantile(AggFunc):
    """计算指定分位数的聚合函数"""
    # quantile参数指定要计算的分位数（0-1之间的值）
    arg_types = {"this": True, "quantile": True}


class ApproxQuantile(Quantile):
    """计算近似分位数的聚合函数，适用于大数据集的高效计算"""
    # accuracy控制精度，weight为权重（用于加权分位数计算）
    arg_types = {
        "this": True,
        "quantile": True,
        "accuracy": False,
        "weight": False,
        "error_tolerance": False,
    }


class Quarter(Func):
    """从日期中提取季度（1-4）的函数"""
    pass


# https://docs.teradata.com/r/Enterprise_IntelliFlex_VMware/SQL-Functions-Expressions-and-Predicates/Arithmetic-Trigonometric-Hyperbolic-Operators/Functions/RANDOM/RANDOM-Function-Syntax
# teradata lower and upper bounds
class Rand(Func):
    """生成随机数的函数，支持指定范围"""
    _sql_names = ["RAND", "RANDOM"]
    # this为种子值，lower和upper为Teradata的下界和上界参数
    arg_types = {"this": False, "lower": False, "upper": False}


class Randn(Func):
    """生成符合标准正态分布的随机数的函数"""
    # this为可选的种子值
    arg_types = {"this": False}


class RangeN(Func):
    """生成指定范围内数值序列的函数"""
    # expressions为序列参数，each控制是否为每行生成独立序列
    arg_types = {"this": True, "expressions": True, "each": False}

class RangeBucket(Func):
    arg_types = {"this": True, "expression": True}


class Rank(AggFunc):
    arg_types = {"expressions": False}
    is_var_len_args = True


class ReadCSV(Func):
    """从CSV文件读取数据的函数，常用于数据导入"""
    _sql_names = ["READ_CSV"]
    # 支持可变长度参数，可以指定多个文件或选项
    is_var_len_args = True
    # expressions为可选的读取选项（分隔符、编码等）
    arg_types = {"this": True, "expressions": False}


class ReadParquet(Func):
    is_var_len_args = True
    arg_types = {"expressions": True}


class Reduce(Func):
    """函数式编程中的reduce操作，对数组进行累积计算"""
    # initial为初始值，merge为合并函数，finish为最终处理函数
    arg_types = {"this": True, "initial": True, "merge": True, "finish": False}


class RegexpExtract(Func):
    """使用正则表达式从字符串中提取匹配内容的函数"""
    arg_types = {
        "this": True,           # 源字符串
        "expression": True,     # 正则表达式模式
        "position": False,      # 开始搜索的位置
        "occurrence": False,    # 匹配的第几个出现
        "parameters": False,    # 正则表达式参数
        "group": False,         # 要提取的捕获组编号
    }


class RegexpExtractAll(Func):
    """使用正则表达式提取字符串中所有匹配项的函数"""
    # 参数与RegexpExtract相同，但返回所有匹配而非第一个
    arg_types = {
        "this": True,
        "expression": True,
        "position": False,
        "occurrence": False,
        "parameters": False,
        "group": False,
    }



class RegexpReplace(Func):
    """使用正则表达式进行字符串替换的函数"""
    arg_types = {
        "this": True,           # 源字符串
        "expression": True,     # 正则表达式模式
        "replacement": False,   # 替换内容
        "position": False,      # 开始位置
        "occurrence": False,    # 替换第几个匹配
        "modifiers": False,     # 正则表达式修饰符（i、g、m等）
        "single_replace": False,
    }


class RegexpLike(Binary, Func):
    """检查字符串是否匹配正则表达式的函数"""
    # 继承Binary表示这是二元谓词操作
    # flag为正则表达式标志（大小写敏感等）
    arg_types = {"this": True, "expression": True, "flag": False}


class RegexpILike(Binary, Func):
    """不区分大小写的正则表达式匹配函数"""
    # 类似RegexpLike，但默认忽略大小写
    arg_types = {"this": True, "expression": True, "flag": False}


class RegexpFullMatch(Binary, Func):
    arg_types = {"this": True, "expression": True, "options": False}


class RegexpInstr(Func):
    arg_types = {
        "this": True,
        "expression": True,
        "position": False,
        "occurrence": False,
        "option": False,
        "parameters": False,
        "group": False,
    }


# https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.functions.split.html
# limit is the number of times a pattern is applied
class RegexpSplit(Func):
    """使用正则表达式分割字符串的函数"""
    # limit控制最大分割次数，避免过度分割
    arg_types = {"this": True, "expression": True, "limit": False}


class RegexpCount(Func):
    arg_types = {
        "this": True,
        "expression": True,
        "position": False,
        "parameters": False,
    }


class RegrValx(Func):
    arg_types = {"this": True, "expression": True}


class RegrValy(Func):
    arg_types = {"this": True, "expression": True}


class Repeat(Func):
    """将字符串重复指定次数的函数"""
    # times指定重复次数
    arg_types = {"this": True, "times": True}


# Some dialects like Snowflake support two argument replace
class Replace(Func):
    """字符串替换函数，将指定子串替换为新内容"""
    # replacement为可选，某些数据库支持只删除不替换
    arg_types = {"this": True, "expression": True, "replacement": False}


class Radians(Func):
    pass


# https://learn.microsoft.com/en-us/sql/t-sql/functions/round-transact-sql?view=sql-server-ver16
# tsql third argument function == trunctaion if not 0
class Round(Func):
    """数值四舍五入函数，支持指定小数位数和截断模式"""
    # truncate参数在SQL Server中控制是截断还是舍入
    arg_types = {"this": True, "decimals": False, "truncate": False}


class RowNumber(Func):
    """窗口函数，为查询结果的每一行分配连续的行号"""
    # this为可选，某些情况下可能需要额外参数
    arg_types = {"this": False}


class SafeAdd(Func):
    arg_types = {"this": True, "expression": True}


class SafeDivide(Func):
    """安全的除法函数，当除数为零时返回NULL而不是错误"""
    # 提供除零保护，避免运行时错误
    arg_types = {"this": True, "expression": True}

class SafeMultiply(Func):
    arg_types = {"this": True, "expression": True}


class SafeNegate(Func):
    pass


class SafeSubtract(Func):
    arg_types = {"this": True, "expression": True}


class SafeConvertBytesToString(Func):
    pass


class SHA(Func):
    """计算SHA-1哈希值的函数"""
    # SHA1是SHA的别名
    _sql_names = ["SHA", "SHA1"]


class SHA2(Func):
    """计算SHA-2哈希值的函数，支持不同的哈希长度"""
    _sql_names = ["SHA2"]
    # length指定哈希位数（如224、256、384、512）
    arg_types = {"this": True, "length": False}

# Represents the variant of the SHA1 function that returns a binary value
class SHA1Digest(Func):
    pass


# Represents the variant of the SHA2 function that returns a binary value
class SHA2Digest(Func):
    arg_types = {"this": True, "length": False}


class Sign(Func):
    """返回数值符号的函数，正数返回1，负数返回-1，零返回0"""
    # SIGNUM是某些数据库的别名
    _sql_names = ["SIGN", "SIGNUM"]


class SortArray(Func):

    """对数组元素进行排序的函数"""
    # asc控制是否升序排序，默认为升序
    arg_types = {"this": True, "asc": False, "nulls_first": False}


class Soundex(Func):
    pass


# https://docs.snowflake.com/en/sql-reference/functions/soundex_p123
class SoundexP123(Func):
    pass


class Split(Func):
    """按指定分隔符分割字符串的函数"""
    # limit控制最大分割次数
    arg_types = {"this": True, "expression": True, "limit": False}


# https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.functions.split_part.html
# https://docs.snowflake.com/en/sql-reference/functions/split_part
# https://docs.snowflake.com/en/sql-reference/functions/strtok
class SplitPart(Func):
    """分割字符串并返回指定位置部分的函数"""
    # part_index指定要返回的部分索引（通常从1开始）
    arg_types = {"this": True, "delimiter": False, "part_index": False}


# Start may be omitted in the case of postgres
# https://www.postgresql.org/docs/9.1/functions-string.html @ Table 9-6
class Substring(Func):
    """提取字符串子串的函数"""
    _sql_names = ["SUBSTRING", "SUBSTR"]
    # start和length都是可选的，PostgreSQL允许省略start
    arg_types = {"this": True, "start": False, "length": False}


class SubstringIndex(Func):
    """
    SUBSTRING_INDEX(str, delim, count) - MySQL的字符串分割索引函数

    count > 0  → 返回第count个分隔符之前的左侧部分
    count < 0  → 返回第|count|个分隔符之后的右侧部分
    """
    # count的正负号决定返回左侧还是右侧部分
    arg_types = {"this": True, "delimiter": True, "count": True}


class StandardHash(Func):
    """计算标准化哈希值的函数，确保跨平台一致性"""
    # expression为可选的哈希算法参数
    arg_types = {"this": True, "expression": False}


class StartsWith(Func):
    """检查字符串是否以指定前缀开始的函数"""
    # 不同数据库使用不同的函数名
    _sql_names = ["STARTS_WITH", "STARTSWITH"]
    arg_types = {"this": True, "expression": True}


class EndsWith(Func):
    """检查字符串是否以指定后缀结束的函数"""
    _sql_names = ["ENDS_WITH", "ENDSWITH"]
    arg_types = {"this": True, "expression": True}


class StrPosition(Func):
    """在字符串中查找子串位置的函数，返回第一次出现的索引"""
    arg_types = {
        "this": True,        # 源字符串
        "substr": True,      # 要查找的子字符串
        "position": False,   # 开始搜索的起始位置
        "occurrence": False, # 查找第几次出现的位置
    }

# Snowflake: https://docs.snowflake.com/en/sql-reference/functions/search
# BigQuery: https://cloud.google.com/bigquery/docs/reference/standard-sql/search_functions#search
class Search(Func):
    arg_types = {
        "this": True,  # data_to_search / search_data
        "expression": True,  # search_query / search_string
        "json_scope": False,  # BigQuery: JSON_VALUES | JSON_KEYS | JSON_KEYS_AND_VALUES
        "analyzer": False,  # Both: analyzer / ANALYZER
        "analyzer_options": False,  # BigQuery: analyzer_options_values
        "search_mode": False,  # Snowflake: OR | AND
    }


class StrToDate(Func):
    """将日期字符串转换为日期对象的函数"""
    # format为可选，用于指定日期格式；safe控制错误处理模式
    arg_types = {"this": True, "format": False, "safe": False}


class StrToTime(Func):
    """将时间字符串转换为时间对象的函数"""
    # format通常是必需的，zone用于时区处理，safe控制异常处理
    arg_types = {"this": True, "format": True, "zone": False, "safe": False}


# Spark allows unix_timestamp()
# https://spark.apache.org/docs/3.1.3/api/python/reference/api/pyspark.sql.functions.unix_timestamp.html
class StrToUnix(Func):
    """将日期时间字符串转换为Unix时间戳的函数"""
    # this和format都是可选的，支持当前时间和默认格式的情况
    arg_types = {"this": False, "format": False}


# https://prestodb.io/docs/current/functions/string.html
# https://spark.apache.org/docs/latest/api/sql/index.html#str_to_map
class StrToMap(Func):
    """将键值对字符串解析为映射对象的函数"""
    arg_types = {
        "this": True,                              # 要解析的字符串
        "pair_delim": False,                       # 键值对之间的分隔符
        "key_value_delim": False,                  # 键和值之间的分隔符
        "duplicate_resolution_callback": False,    # 重复键的处理回调函数
    }


class NumberToStr(Func):
    """将数值按指定格式转换为字符串的函数"""
    # culture用于本地化格式（如千位分隔符、小数点符号等）
    arg_types = {"this": True, "format": True, "culture": False}


class FromBase(Func):
    """将指定进制的数字字符串转换为十进制数值的函数"""
    # expression指定源数据的进制基数
    arg_types = {"this": True, "expression": True}


class Space(Func):
    """
    SPACE(n) → 生成由n个空白字符组成的字符串
    """
    pass


class Struct(Func):
    """创建结构体（记录）数据类型的函数"""
    # expressions包含结构体的字段定义，支持可变数量的字段
    arg_types = {"expressions": False}
    # 支持可变长度参数，可以创建任意字段数量的结构体
    is_var_len_args = True


class StructExtract(Func):
    """从结构体对象中提取指定字段值的函数"""
    # expression指定要提取的字段名或索引
    arg_types = {"this": True, "expression": True}


# https://learn.microsoft.com/en-us/sql/t-sql/functions/stuff-transact-sql?view=sql-server-ver16
# https://docs.snowflake.com/en/sql-reference/functions/insert
class Stuff(Func):
    """在字符串的指定位置插入或替换内容的函数"""
    _sql_names = ["STUFF", "INSERT"]
    # start为起始位置，length为要替换的长度，expression为新内容
    arg_types = {"this": True, "start": True, "length": True, "expression": True}


class Sum(AggFunc):
    """计算数值总和的聚合函数"""
    pass


class Sqrt(Func):
    """计算平方根的数学函数"""
    pass


class Stddev(AggFunc):
    """计算标准差的聚合函数"""
    # STDEV是某些数据库的简写形式
    _sql_names = ["STDDEV", "STDEV"]


class StddevPop(AggFunc):
    """计算总体标准差的聚合函数（分母为N）"""
    pass


class StddevSamp(AggFunc):
    """计算样本标准差的聚合函数（分母为N-1）"""
    pass


# https://cloud.google.com/bigquery/docs/reference/standard-sql/time_functions#time
class Time(Func):
    """创建或转换时间对象的函数"""
    # this和zone都是可选的，支持多种时间创建方式
    arg_types = {"this": False, "zone": False}


class TimeToStr(Func):
    """将时间对象按指定格式转换为字符串的函数"""
    # culture用于本地化格式，zone用于时区转换
    arg_types = {"this": True, "format": True, "culture": False, "zone": False}


class TimeToTimeStr(Func):
    """将时间对象转换为标准时间字符串格式的函数"""
    pass


# 时间转Unix时间戳函数：将时间转换为Unix时间戳
class TimeToUnix(Func):
    """将时间对象转换为Unix时间戳的函数"""
    pass


# 时间字符串转日期函数：从时间字符串中提取日期部分
class TimeStrToDate(Func):
    """从时间字符串中提取日期部分的函数"""
    pass


# 时间字符串转时间函数：将时间字符串转换为时间对象
class TimeStrToTime(Func):
    """将时间字符串转换为时间对象的函数"""
    # zone用于时区处理和转换
    arg_types = {"this": True, "zone": False}


# 时间字符串转Unix时间戳函数：将时间字符串转换为Unix时间戳
class TimeStrToUnix(Func):
    """将时间字符串转换为Unix时间戳的函数"""
    pass


class Trim(Func):
    """去除字符串两端空白字符或指定字符的函数"""
    arg_types = {
        "this": True,        # 要修整的字符串
        "expression": False, # 要去除的字符集（默认为空白字符）
        "position": False,   # 修整位置（LEADING、TRAILING、BOTH）
        "collation": False,  # 字符串比较的排序规则
    }


# 时间戳或日期字符串加法函数：在时间值上添加指定间隔
class TsOrDsAdd(Func, TimeUnit):
    """在时间戳或日期字符串上添加时间间隔的函数"""
    # return_type用于在转译时正确转换此表达式的参数类型
    arg_types = {"this": True, "expression": True, "unit": False, "return_type": False}

    @property
    def return_type(self) -> DataType:
        """获取函数的返回数据类型"""
        # 如果没有指定返回类型，默认返回DATE类型
        return DataType.build(self.args.get("return_type") or DataType.Type.DATE)


# 时间戳或日期字符串差值函数：计算两个时间值的差
class TsOrDsDiff(Func, TimeUnit):
    """计算时间戳或日期字符串之间差值的函数"""
    arg_types = {"this": True, "expression": True, "unit": False}


# 时间戳或日期字符串转日期字符串函数：标准化日期字符串格式
class TsOrDsToDateStr(Func):
    """将时间戳或日期字符串转换为标准日期字符串格式的函数"""
    pass


# 时间戳或日期字符串转日期函数：转换为日期对象
class TsOrDsToDate(Func):
    """将时间戳或日期字符串转换为日期对象的函数"""
    # format和safe用于控制转换过程和错误处理
    arg_types = {"this": True, "format": False, "safe": False}


# 时间戳或日期字符串转日期时间函数：转换为日期时间对象
class TsOrDsToDatetime(Func):
    """将时间戳或日期字符串转换为日期时间对象的函数"""
    pass


# 时间戳或日期字符串转时间函数：提取时间部分
class TsOrDsToTime(Func):
    """将时间戳或日期字符串转换为时间对象的函数"""
    arg_types = {"this": True, "format": False, "safe": False}


# 时间戳或日期字符串转时间戳函数：标准化为时间戳格式
class TsOrDsToTimestamp(Func):
    """将时间戳或日期字符串转换为标准时间戳格式的函数"""
    pass


# 时间戳或日期标识符转日期标识符函数：格式转换
class TsOrDiToDi(Func):
    """将时间戳或日期标识符转换为日期标识符的函数"""
    pass


# 反十六进制函数：将十六进制字符串转换为二进制数据
class Unhex(Func):
    """将十六进制字符串转换为二进制数据的函数"""
    # expression为可选，某些数据库可能需要额外参数
    arg_types = {"this": True, "expression": False}


# Unicode码点函数：获取字符的Unicode码点值
class Unicode(Func):
    """获取字符的Unicode码点值的函数"""
    pass


# https://cloud.google.com/bigquery/docs/reference/standard-sql/date_functions#unix_date
class UnixDate(Func):
    """将日期转换为Unix日期数值的函数（自1970-01-01的天数）"""
    pass


class UnixToStr(Func):
    """将Unix时间戳按指定格式转换为字符串的函数"""
    # format为可选，默认使用标准格式
    arg_types = {"this": True, "format": False}


# https://prestodb.io/docs/current/functions/datetime.html
# presto has weird zone/hours/minutes
class UnixToTime(Func):
    """将Unix时间戳转换为时间对象的函数"""
    arg_types = {
        "this": True,     # Unix时间戳值
        "scale": False,   # 时间精度刻度
        "zone": False,    # 时区
        "hours": False,   # Presto特有的小时偏移
        "minutes": False, # Presto特有的分钟偏移
        "format": False,  # 输出格式
    }

    # 预定义的时间精度常量，用于scale参数
    SECONDS = Literal.number(0)      # 秒级精度
    DECIS = Literal.number(1)        # 十分之一秒
    CENTIS = Literal.number(2)       # 百分之一秒
    MILLIS = Literal.number(3)       # 毫秒级精度
    DECIMILLIS = Literal.number(4)   # 十分之一毫秒
    CENTIMILLIS = Literal.number(5)  # 百分之一毫秒
    MICROS = Literal.number(6)       # 微秒级精度
    DECIMICROS = Literal.number(7)   # 十分之一微秒
    CENTIMICROS = Literal.number(8)  # 百分之一微秒
    NANOS = Literal.number(9)        # 纳秒级精度


# Unix时间戳转时间字符串函数：将Unix时间戳转换为时间字符串
class UnixToTimeStr(Func):
    """将Unix时间戳转换为时间字符串的函数"""
    pass


# Unix秒数函数：获取当前时间的Unix秒数
class UnixSeconds(Func):
    """获取当前时间的Unix秒数时间戳的函数"""
    pass


# Unix微秒函数：获取当前时间的Unix微秒数
class UnixMicros(Func):
    """获取当前时间的Unix微秒数时间戳的函数"""
    pass


# Unix毫秒函数：获取当前时间的Unix毫秒数
class UnixMillis(Func):
    """获取当前时间的Unix毫秒数时间戳的函数"""
    pass


# UUID生成函数：生成唯一标识符
class Uuid(Func):
    """生成UUID（通用唯一标识符）的函数"""
    # 不同数据库使用不同的UUID生成函数名
    _sql_names = ["UUID", "GEN_RANDOM_UUID", "GENERATE_UUID", "UUID_STRING"]
    # this和name都是可选的，用于特定的UUID生成方式
    arg_types = {"this": False, "name": False}


TIMESTAMP_PARTS = {
    "year": False,
    "month": False,
    "day": False,
    "hour": False,
    "min": False,
    "sec": False,
    "nano": False,
}


class TimestampFromParts(Func):
    """通过年、月、日、时、分、秒等组件构造时间戳对象的函数"""
    _sql_names = ["TIMESTAMP_FROM_PARTS", "TIMESTAMPFROMPARTS"]
    arg_types = {
        **TIMESTAMP_PARTS,
        "zone": False,
        "milli": False,
        "this": False,
        "expression": False,
    }


class TimestampLtzFromParts(Func):
    _sql_names = ["TIMESTAMP_LTZ_FROM_PARTS", "TIMESTAMPLTZFROMPARTS"]
    arg_types = TIMESTAMP_PARTS.copy()


class TimestampTzFromParts(Func):
    _sql_names = ["TIMESTAMP_TZ_FROM_PARTS", "TIMESTAMPTZFROMPARTS"]
    arg_types = {
        **TIMESTAMP_PARTS,
        "zone": False,
    }

class Upper(Func):
    """将字符串转换为大写的函数"""
    # UCASE是某些数据库（如MySQL）的别名
    _sql_names = ["UPPER", "UCASE"]


class Corr(Binary, AggFunc):
    """计算两个数值列相关系数的聚合函数"""
    # 继承Binary表示需要两个输入列
    pass

# https://docs.oracle.com/en/database/oracle/oracle-database/19/sqlrf/CUME_DIST.html
class CumeDist(AggFunc):
    arg_types = {"expressions": False}
    is_var_len_args = True


class Variance(AggFunc):
    """计算样本方差的聚合函数"""
    # 多种函数名对应不同数据库的实现，但都计算样本方差
    _sql_names = ["VARIANCE", "VARIANCE_SAMP", "VAR_SAMP"]


class VariancePop(AggFunc):
    """计算总体方差的聚合函数（分母为N）"""
    # 与样本方差的区别是分母为N而不是N-1
    _sql_names = ["VARIANCE_POP", "VAR_POP"]


class WidthBucket(Func):
    arg_types = {"this": True, "min_value": True, "max_value": True, "num_buckets": True}


class CovarSamp(Binary, AggFunc):
    """计算两个数值列样本协方差的聚合函数"""
    pass


# 总体协方差聚合函数：计算两个变量的总体协方差
class CovarPop(Binary, AggFunc):
    """计算两个数值列总体协方差的聚合函数"""
    pass


# 周数提取函数：从日期中提取周数
class Week(Func):
    """从日期中提取周数的函数"""
    # mode参数控制周数计算的起始方式（周一还是周日开始等）
    arg_types = {"this": True, "mode": False}


# 周开始表达式：表示周的开始日期
class WeekStart(Expression):
    """表示一周开始日期的表达式"""
    pass


class NextDay(Func):
    arg_types = {"this": True, "expression": True}


class XMLElement(Func):
    """创建XML元素的函数"""
    _sql_names = ["XMLELEMENT"]
    # expressions为可选的XML属性和内容
    arg_types = {"this": True, "expressions": False}


# XML表函数：将XML数据转换为关系表格
class XMLTable(Func):
    """将XML数据解析为关系表格的函数"""
    arg_types = {
        "this": True,         # XML数据源
        "namespaces": False,  # XML命名空间定义
        "passing": False,     # 传递给XPath的变量
        "columns": False,     # 输出列定义
        "by_ref": False,      # 是否按引用传递
    }


# XML命名空间表达式：定义XML命名空间
class XMLNamespace(Expression):
    """XML命名空间定义表达式"""
    pass


# SQL Server的XML键值选项：用于FOR XML查询的选项
# 参考文档：https://learn.microsoft.com/en-us/sql/t-sql/queries/select-for-clause-transact-sql?view=sql-server-ver17#syntax
class XMLKeyValueOption(Expression):
    """SQL Server FOR XML查询中的键值对选项表达式"""
    # expression为可选的值表达式
    arg_types = {"this": True, "expression": False}


# 年份提取函数：从日期中提取年份部分
class Year(Func):
    """从日期中提取年份部分的函数"""
    pass


# USE语句表达式：切换数据库或模式上下文
class Use(Expression):
    """USE语句表达式，用于切换数据库或模式上下文"""
    # kind指定使用的类型（数据库、模式等）
    arg_types = {"this": False, "expressions": False, "kind": False}


# MERGE语句：数据合并操作的DML语句
class Merge(DML):
    """MERGE语句，用于根据条件进行INSERT、UPDATE或DELETE操作"""
    arg_types = {
        "this": True,      # 目标表
        "using": True,     # 源表或查询
        "on": False,       # 匹配条件
        "using_cond": False,
        "whens": True,     # WHEN子句集合
        "with": False,     # WITH子句（CTE）
        "returning": False, # RETURNING子句
    }


# WHEN子句：MERGE语句中的条件分支
class When(Expression):
    """MERGE语句中的WHEN子句，定义匹配条件和对应操作"""
    arg_types = {
        "matched": True,    # 是否匹配（MATCHED或NOT MATCHED）
        "source": False,    # 源表条件
        "condition": False, # 额外的AND条件
        "then": True,      # 执行的操作（INSERT/UPDATE/DELETE）
    }



class Whens(Expression):
    """包装一个或多个WHEN [NOT] MATCHED [...] 子句的表达式"""
    # expressions包含多个When表达式
    arg_types = {"expressions": True}


# https://docs.oracle.com/javadb/10.8.3.0/ref/rrefsqljnextvaluefor.html
# https://learn.microsoft.com/en-us/sql/t-sql/functions/next-value-for-transact-sql?view=sql-server-ver16
class NextValueFor(Func):
    """获取序列下一个值的函数"""
    # order用于指定排序规则（在某些上下文中）
    arg_types = {"this": True, "order": False}


# 分号表达式：表示尾随分号，仅用于保留尾随注释
# 例如：select 1; -- my comment
class Semicolon(Expression):
    """表示尾随分号的表达式，用于保留分号后的注释"""
    # 不需要任何参数，仅作为语法标记
    arg_types = {}


# 表列表达式：BigQuery特有的表投影表达式
# BigQuery允许SELECT t FROM t，并将投影视为结构体值
# 此表达式类型由qualify构造，以便稍后正确标注其类型
class TableColumn(Expression):
    """BigQuery特有的表列表达式，用于表投影作为结构体值"""
    pass

# 模块级别的函数注册表：收集所有函数类
# 获取当前模块中所有的函数子类，排除基类和抽象类
ALL_FUNCTIONS = subclasses(__name__, Func, (AggFunc, Anonymous, Func))
# 创建函数名到函数类的映射，支持多个SQL名称映射到同一个函数类
FUNCTION_BY_NAME = {name: func for func in ALL_FUNCTIONS for name in func.sql_names()}

# JSON路径组件注册表：收集所有JSON路径部分类
JSON_PATH_PARTS = subclasses(__name__, JSONPathPart, (JSONPathPart,))

# 百分位数函数元组：包含所有百分位数相关的函数类
PERCENTILES = (PercentileCont, PercentileDisc)


# 辅助函数：优雅地处理可能的字符串或表达式
@t.overload
def maybe_parse(
    sql_or_expression: ExpOrStr,
    *,
    into: t.Type[E],
    dialect: DialectType = None,
    prefix: t.Optional[str] = None,
    copy: bool = False,
    **opts,
) -> E: ...


@t.overload
def maybe_parse(
    sql_or_expression: str | E,
    *,
    into: t.Optional[IntoType] = None,
    dialect: DialectType = None,
    prefix: t.Optional[str] = None,
    copy: bool = False,
    **opts,
) -> E: ...


def maybe_parse(
    sql_or_expression: ExpOrStr,
    *,
    into: t.Optional[IntoType] = None,
    dialect: DialectType = None,
    prefix: t.Optional[str] = None,
    copy: bool = False,
    **opts,
) -> Expression:
    """优雅地处理可能的字符串或表达式输入

    这个函数是SQLGlot的核心辅助函数，统一处理字符串和表达式对象，
    简化了API的使用，用户可以传入SQL字符串或已解析的表达式对象。

    示例:
        >>> maybe_parse("1")
        Literal(this=1, is_string=False)
        >>> maybe_parse(to_identifier("x"))
        Identifier(this=x, quoted=False)

    参数:
        sql_or_expression: SQL代码字符串或表达式对象
        into: 要解析到的SQLGlot表达式类型
        dialect: 用于解析输入表达式的SQL方言（当输入是SQL字符串时）
        prefix: 在解析前添加到SQL字符串前的前缀（自动包含空格）
        copy: 是否复制表达式对象
        **opts: 其他解析选项（同样用于SQL字符串输入的情况）

    返回:
        Expression: 解析后的表达式对象或原始表达式对象
    """
    # 如果输入已经是表达式对象，直接返回或复制
    if isinstance(sql_or_expression, Expression):
        if copy:
            # 复制表达式对象，避免意外修改原对象
            return sql_or_expression.copy()
        return sql_or_expression

    # 空值检查：确保输入不为None，提供清晰的错误信息
    if sql_or_expression is None:
        raise ParseError("SQL cannot be None")

    # 延迟导入sqlglot模块，避免循环导入问题
    import sqlglot

    # 将输入转换为字符串（处理数字等其他类型）
    sql = str(sql_or_expression)
    if prefix:
        # 添加前缀，用于创建完整的SQL语句
        sql = f"{prefix} {sql}"

    # 使用sqlglot.parse_one进行实际解析
    return sqlglot.parse_one(sql, read=dialect, into=into, **opts)


# 可选复制函数的重载定义：处理None和泛型类型
@t.overload
def maybe_copy(instance: None, copy: bool = True) -> None: ...


@t.overload
def maybe_copy(instance: E, copy: bool = True) -> E: ...


# 可选复制辅助函数：根据copy参数决定是否复制实例
def maybe_copy(instance, copy=True):
    """根据copy参数条件性地复制实例对象"""
    # 只有当copy为True且instance存在时才进行复制，避免不必要的开销
    return instance.copy() if copy and instance else instance


# 表达式树的文本表示生成函数：用于调试和显示
def _to_s(node: t.Any, verbose: bool = False, level: int = 0, repr_str: bool = False) -> str:
    """生成表达式树的文本表示，用于调试和可视化"""
    # 根据层级计算缩进，每层增加2个空格
    indent = "\n" + ("  " * (level + 1))
    delim = f",{indent}"

    # 处理Expression类型节点
    if isinstance(node, Expression):
        # 过滤空值和空列表，除非verbose模式要求显示所有属性
        args = {k: v for k, v in node.args.items() if (v is not None and v != []) or verbose}

        # 添加类型信息（非DataType节点且有类型信息时）
        if (node.type or verbose) and not isinstance(node, DataType):
            args["_type"] = node.type
        # 添加注释信息
        if node.comments or verbose:
            args["_comments"] = node.comments

        # verbose模式下添加对象ID，用于跟踪对象引用
        if verbose:
            args["_id"] = id(node)

        # 叶子节点使用内联格式，提供更紧凑的表示
        if node.is_leaf():
            indent = ""
            delim = ", "

        # 判断是否需要使用repr格式（字符串字面量或带引号的标识符）
        repr_str = node.is_string or (isinstance(node, Identifier) and node.quoted)
        # 递归处理所有参数，构建参数字符串
        items = delim.join(
            [f"{k}={_to_s(v, verbose, level + 1, repr_str=repr_str)}" for k, v in args.items()]
        )
        return f"{node.__class__.__name__}({indent}{items})"

    # 处理列表类型节点
    if isinstance(node, list):
        # 递归处理列表中的每个元素
        items = delim.join(_to_s(i, verbose, level + 1) for i in node)
        # 空列表不需要缩进
        items = f"{indent}{items}" if items else ""
        return f"[{items}]"

    # 使用字符串的repr表示以避免丢失重要的空白字符
    if repr_str and isinstance(node, str):
        node = repr(node)

    # 将多行字符串按当前层级进行缩进对齐
    return indent.join(textwrap.dedent(str(node).strip("\n")).splitlines())


def _is_wrong_expression(expression, into):
    """检查表达式是否为错误的类型，需要转换为目标类型"""
    # 如果expression是Expression但不是目标类型into的实例，则认为类型错误
    return isinstance(expression, Expression) and not isinstance(expression, into)


def _apply_builder(
    expression,      # 要设置的表达式
    instance,        # 目标实例
    arg,            # 要设置的参数名
    copy=True,      # 是否复制实例
    prefix=None,    # SQL前缀
    into=None,      # 目标表达式类型
    dialect=None,   # SQL方言
    into_arg="this", # 类型转换时使用的参数名
    **opts,
):
    """应用单个表达式到实例的指定参数，SQLGlot流畅API的核心构建函数"""
    # 类型检查和自动转换：如果表达式类型不匹配，自动包装为目标类型
    if _is_wrong_expression(expression, into):
        # 将表达式包装为目标类型，使用into_arg指定参数名
        expression = into(**{into_arg: expression})
    
    # 根据copy参数决定是否复制实例，避免修改原对象
    instance = maybe_copy(instance, copy)
    
    # 解析表达式：统一处理字符串和表达式对象
    expression = maybe_parse(
        sql_or_expression=expression,
        prefix=prefix,
        into=into,
        dialect=dialect,
        **opts,
    )
    
    # 设置解析后的表达式到实例的指定参数
    instance.set(arg, expression)
    return instance


# 子表达式列表构建器应用函数：处理多个表达式的复杂构建逻辑
def _apply_child_list_builder(
    *expressions,    # 可变数量的表达式
    instance,        # 目标实例
    arg,            # 要设置的参数名
    append=True,    # 是否追加到现有列表
    copy=True,      # 是否复制实例
    prefix=None,    # SQL前缀
    into=None,      # 目标表达式类型
    dialect=None,   # SQL方言
    properties=None, # 额外属性
    **opts,
):
    """应用多个表达式到实例的指定参数，支持列表操作和属性合并"""
    # 复制实例以避免副作用
    instance = maybe_copy(instance, copy)
    parsed = []
    # 初始化属性字典，用于收集非expressions的其他属性
    properties = {} if properties is None else properties

    # 逐个处理输入的表达式
    for expression in expressions:
        if expression is not None:
            # 类型检查和自动转换：将单个表达式包装为包含expressions的目标类型
            if _is_wrong_expression(expression, into):
                expression = into(expressions=[expression])

            # 解析表达式
            expression = maybe_parse(
                expression,
                into=into,
                dialect=dialect,
                prefix=prefix,
                **opts,
            )
            
            # 分离expressions和其他属性
            for k, v in expression.args.items():
                if k == "expressions":
                    # 将expressions列表扩展到parsed中
                    parsed.extend(v)
                else:
                    # 收集其他属性到properties字典中
                    properties[k] = v

    # 处理追加逻辑：如果append为True且已有表达式，则合并
    existing = instance.args.get(arg)
    if append and existing:
        # 将现有表达式列表与新解析的表达式合并
        parsed = existing.expressions + parsed

    # 创建新的子表达式对象
    child = into(expressions=parsed)
    # 设置收集到的额外属性
    for k, v in properties.items():
        child.set(k, v)
    
    # 将构建好的子表达式设置到实例
    instance.set(arg, child)

    return instance


# 列表构建器应用函数：处理简单表达式列表的构建
def _apply_list_builder(
    *expressions,    # 可变数量的表达式
    instance,        # 目标实例
    arg,            # 要设置的参数名
    append=True,    # 是否追加到现有列表
    copy=True,      # 是否复制实例
    prefix=None,    # SQL前缀
    into=None,      # 目标表达式类型
    dialect=None,   # SQL方言
    **opts,
):
    """应用多个表达式到实例的指定参数，构建简单的表达式列表"""
    # 复制实例以避免修改原对象
    inst = maybe_copy(instance, copy)

    # 解析并过滤表达式：只处理非None的表达式，避免空值污染
    expressions = [
        maybe_parse(
            sql_or_expression=expression,
            into=into,
            prefix=prefix,
            dialect=dialect,
            **opts,
        )
        for expression in expressions
        if expression is not None  # 过滤None值，保持列表清洁
    ]

    # 处理追加逻辑：如果需要追加且已有表达式列表，则合并
    existing_expressions = inst.args.get(arg)
    if append and existing_expressions:
        # 将现有表达式列表与新表达式合并
        expressions = existing_expressions + expressions

    # 直接设置表达式列表到实例参数
    inst.set(arg, expressions)
    return inst


# 逻辑连接构建器应用函数：专门处理AND连接的表达式组合
def _apply_conjunction_builder(
    *expressions,    # 可变数量的表达式
    instance,        # 目标实例
    arg,            # 要设置的参数名
    into=None,      # 目标包装类型
    append=True,    # 是否追加到现有条件
    copy=True,      # 是否复制实例
    dialect=None,   # SQL方言
    **opts,
):
    """应用多个表达式并用AND连接，专门用于WHERE、HAVING等条件构建"""
    # 过滤空表达式：移除None和空字符串，避免生成无效的AND条件
    expressions = [exp for exp in expressions if exp is not None and exp != ""]
    if not expressions:
        # 如果没有有效表达式，直接返回原实例，不做任何修改
        return instance

    # 复制实例
    inst = maybe_copy(instance, copy)

    # 处理现有条件的追加逻辑
    existing = inst.args.get(arg)
    if append and existing is not None:
        # 解包现有条件：如果指定了into包装类型，取出内部的this，否则直接使用
        expressions = [existing.this if into else existing] + list(expressions)

    # 使用and_函数创建AND连接的条件树
    node = and_(*expressions, dialect=dialect, copy=copy, **opts)

    # 根据是否有包装类型决定如何设置节点
    inst.set(arg, into(this=node) if into else node)
    return inst


def _apply_cte_builder(
    instance: E,               # 目标查询实例
    alias: ExpOrStr,          # CTE别名
    as_: ExpOrStr,            # CTE定义表达式
    recursive: t.Optional[bool] = None,     # 是否递归CTE
    materialized: t.Optional[bool] = None,  # 是否物化CTE
    append: bool = True,       # 是否追加到现有WITH子句
    dialect: DialectType = None, # SQL方言
    copy: bool = True,         # 是否复制实例
    scalar: bool = False,      # 是否标量CTE
    **opts,
) -> E:
    """构建并添加CTE（公共表表达式）到查询的WITH子句中"""
    # 解析CTE别名为TableAlias对象
    alias_expression = maybe_parse(alias, dialect=dialect, into=TableAlias, **opts)
    # 解析CTE定义表达式
    as_expression = maybe_parse(as_, dialect=dialect, copy=copy, **opts)
    
    # 标量CTE特殊处理：标量CTE必须包装在子查询中
    if scalar and not isinstance(as_expression, Subquery):
        # 标量CTE必须包装在子查询中，确保语义正确性
        as_expression = Subquery(this=as_expression)
    
    # 创建CTE对象，包含所有必要的属性
    cte = CTE(this=as_expression, alias=alias_expression, materialized=materialized, scalar=scalar)
    
    # 使用子列表构建器将CTE添加到WITH子句中
    return _apply_child_list_builder(
        cte,
        instance=instance,
        arg="with",          # 设置到with参数
        append=append,
        copy=copy,
        into=With,           # 包装为With对象
        properties={"recursive": recursive or False},  # 设置递归属性
    )


# 表达式组合函数：使用指定连接符组合多个表达式
def _combine(
    expressions: t.Sequence[t.Optional[ExpOrStr]], # 表达式序列
    operator: t.Type[Connector],  # 连接操作符类型（如And、Or）
    dialect: DialectType = None,  # SQL方言
    copy: bool = True,           # 是否复制表达式
    wrap: bool = True,           # 是否用括号包装Connector类型
    **opts,
) -> Expression:
    """使用指定的连接操作符组合多个表达式，构建操作符树"""
    # 解析并过滤表达式：使用condition函数处理每个表达式
    conditions = [
        condition(expression, dialect=dialect, copy=copy, **opts)
        for expression in expressions
        if expression is not None  # 过滤None值
    ]

    # 解构第一个表达式和其余表达式
    this, *rest = conditions
    
    # 如果有多个表达式且需要包装，则包装第一个表达式
    if rest and wrap:
        this = _wrap(this, Connector)
    
    # 遍历其余表达式，逐个构建连接树
    for expression in rest:
        # 创建连接节点：左侧为当前累积结果，右侧为新表达式
        this = operator(this=this, expression=_wrap(expression, Connector) if wrap else expression)

    return this


# 表达式包装函数的重载定义：处理None类型
@t.overload
def _wrap(expression: None, kind: t.Type[Expression]) -> None: ...


# 表达式包装函数的重载定义：处理泛型表达式类型
@t.overload
def _wrap(expression: E, kind: t.Type[Expression]) -> E | Paren: ...


# 表达式包装辅助函数：根据表达式类型决定是否添加括号
def _wrap(expression: t.Optional[E], kind: t.Type[Expression]) -> t.Optional[E] | Paren:
    """根据表达式类型条件性地添加括号包装"""
    # 只有当表达式是指定类型时才添加括号，避免运算符优先级问题
    return Paren(this=expression) if isinstance(expression, kind) else expression


# 集合操作应用函数：处理UNION、INTERSECT、EXCEPT等集合运算
def _apply_set_operation(
    *expressions: ExpOrStr,      # 要进行集合运算的表达式
    set_operation: t.Type[S],    # 集合操作类型
    distinct: bool = True,       # 是否使用DISTINCT
    dialect: DialectType = None, # SQL方言
    copy: bool = True,           # 是否复制表达式
    **opts,
) -> S:
    """应用集合操作到多个表达式，构建左结合的集合运算树"""
    # 使用reduce函数构建左结合的集合操作树
    # 这确保了(A UNION B) UNION C的结构，而不是A UNION (B UNION C)
    return reduce(
        lambda x, y: set_operation(this=x, expression=y, distinct=distinct, **opts),
        # 生成器表达式：解析每个输入表达式
        (maybe_parse(e, dialect=dialect, copy=copy, **opts) for e in expressions),
    )


def union(
    *expressions: ExpOrStr,      # 要进行UNION的表达式
    distinct: bool = True,       # 是否使用DISTINCT（默认UNION DISTINCT）
    dialect: DialectType = None, # SQL方言
    copy: bool = True,           # 是否复制表达式
    **opts,
) -> Union:
    """
    初始化UNION操作的语法树

    UNION操作用于合并两个或多个SELECT语句的结果集，去除重复行。

    示例:
        >>> union("SELECT * FROM foo", "SELECT * FROM bla").sql()
        'SELECT * FROM foo UNION SELECT * FROM bla'

    参数:
        expressions: SQL代码字符串，对应UNION的操作数。
            如果传入Expression实例，将直接使用。
        distinct: 当且仅当此参数为True时设置DISTINCT标志。
        dialect: 用于解析输入表达式的SQL方言。
        copy: 是否复制表达式。
        opts: 用于解析输入表达式的其他选项。

    返回:
        新的Union实例。
    """
    # 断言检查：UNION至少需要两个表达式
    assert len(expressions) >= 2, "At least two expressions are required by `union`."
    # 调用通用的集合操作应用函数
    return _apply_set_operation(
        *expressions, set_operation=Union, distinct=distinct, dialect=dialect, copy=copy, **opts
    )


# INTERSECT操作构建函数：创建SQL INTERSECT操作的语法树
def intersect(
    *expressions: ExpOrStr,      # 要进行INTERSECT的表达式
    distinct: bool = True,       # 是否使用DISTINCT（默认为True）
    dialect: DialectType = None, # SQL方言
    copy: bool = True,           # 是否复制表达式
    **opts,
) -> Intersect:
    """
    初始化INTERSECT操作的语法树

    INTERSECT操作返回两个或多个SELECT语句结果集的交集，即同时出现在所有结果集中的行。

    示例:
        >>> intersect("SELECT * FROM foo", "SELECT * FROM bla").sql()
        'SELECT * FROM foo INTERSECT SELECT * FROM bla'

    参数:
        expressions: SQL代码字符串，对应INTERSECT的操作数。
            如果传入Expression实例，将直接使用。
        distinct: 当且仅当此参数为True时设置DISTINCT标志。
        dialect: 用于解析输入表达式的SQL方言。
        copy: 是否复制表达式。
        opts: 用于解析输入表达式的其他选项。

    返回:
        新的Intersect实例。
    """
    # 断言检查：INTERSECT至少需要两个表达式才有数学意义
    assert len(expressions) >= 2, "At least two expressions are required by `intersect`."
    # 调用通用的集合操作应用函数，使用Intersect作为操作类型
    return _apply_set_operation(
        *expressions, set_operation=Intersect, distinct=distinct, dialect=dialect, copy=copy, **opts
    )


# EXCEPT操作构建函数：创建SQL EXCEPT（差集）操作的语法树
def except_(
    *expressions: ExpOrStr,      # 要进行EXCEPT的表达式
    distinct: bool = True,       # 是否使用DISTINCT（默认为True）
    dialect: DialectType = None, # SQL方言
    copy: bool = True,           # 是否复制表达式
    **opts,
) -> Except:
    """
    初始化EXCEPT操作的语法树

    EXCEPT操作返回第一个SELECT语句的结果集中存在，但在后续SELECT语句结果集中不存在的行（差集）。

    示例:
        >>> except_("SELECT * FROM foo", "SELECT * FROM bla").sql()
        'SELECT * FROM foo EXCEPT SELECT * FROM bla'

    参数:
        expressions: SQL代码字符串，对应EXCEPT的操作数。
            如果传入Expression实例，将直接使用。
        distinct: 当且仅当此参数为True时设置DISTINCT标志。
        dialect: 用于解析输入表达式的SQL方言。
        copy: 是否复制表达式。
        opts: 用于解析输入表达式的其他选项。

    返回:
        新的Except实例。
    """
    # 断言检查：EXCEPT需要至少两个表达式（被减数和减数）
    assert len(expressions) >= 2, "At least two expressions are required by `except_`."
    # 调用通用的集合操作应用函数，使用Except作为操作类型
    return _apply_set_operation(
        *expressions, set_operation=Except, distinct=distinct, dialect=dialect, copy=copy, **opts
    )


# SELECT语句构建函数：创建SELECT查询的语法树
def select(*expressions: ExpOrStr, dialect: DialectType = None, **opts) -> Select:
    """
    从一个或多个SELECT表达式初始化语法树

    这是SQLGlot中最重要的查询构建函数，用于创建SELECT语句的起点。
    设计为流畅接口的入口，支持方法链式调用。

    示例:
        >>> select("col1", "col2").from_("tbl").sql()
        'SELECT col1, col2 FROM tbl'

    参数:
        *expressions: 要解析为SELECT语句表达式的SQL代码字符串。
            如果传入Expression实例，将直接使用。
        dialect: 用于解析输入表达式的SQL方言（当输入表达式为SQL字符串时）。
        **opts: 用于解析输入表达式的其他选项（同样用于SQL字符串输入的情况）。

    返回:
        Select: SELECT语句的语法树。
    """
    # 创建空的Select对象，然后调用其select方法设置选择列
    # 这种设计模式允许链式调用：select().from_().where()...
    return Select().select(*expressions, dialect=dialect, **opts)


# FROM子句构建函数：从FROM表达式创建SELECT查询的语法树
def from_(expression: ExpOrStr, dialect: DialectType = None, **opts) -> Select:
    """
    从FROM表达式初始化语法树

    这是另一种创建SELECT查询的方式，以FROM子句为起点。
    适用于查询逻辑更关注数据源而非选择列的场景。

    示例:
        >>> from_("tbl").select("col1", "col2").sql()
        'SELECT col1, col2 FROM tbl'

    参数:
        expression: 要解析为SELECT语句FROM表达式的SQL代码字符串。
            如果传入Expression实例，将直接使用。
        dialect: 用于解析输入表达式的SQL方言（当输入表达式为SQL字符串时）。
        **opts: 用于解析输入表达式的其他选项（同样用于SQL字符串输入的情况）。

    返回:
        Select: SELECT语句的语法树。
    """
    # 创建空的Select对象，然后调用其from_方法设置数据源
    # 与select()函数互补，提供不同的查询构建起点
    return Select().from_(expression, dialect=dialect, **opts)


# UPDATE语句构建函数：创建数据更新语句
def update(
    table: str | Table,                              # 要更新的表名或Table对象
    properties: t.Optional[dict] = None,             # 要更新的列和值的字典
    where: t.Optional[ExpOrStr] = None,             # WHERE条件
    from_: t.Optional[ExpOrStr] = None,             # FROM子句（用于多表更新）
    with_: t.Optional[t.Dict[str, ExpOrStr]] = None, # CTE定义字典
    dialect: DialectType = None,                     # SQL方言
    **opts,
) -> Update:
    """
    创建UPDATE语句

    UPDATE语句用于修改表中现有的数据行，支持复杂的更新场景包括多表关联更新和CTE。

    示例:
        >>> update("my_table", {"x": 1, "y": "2", "z": None}, from_="baz_cte", where="baz_cte.id > 1 and my_table.id = baz_cte.id", with_={"baz_cte": "SELECT id FROM foo"}).sql()
        "WITH baz_cte AS (SELECT id FROM foo) UPDATE my_table SET x = 1, y = '2', z = NULL FROM baz_cte WHERE baz_cte.id > 1 AND my_table.id = baz_cte.id"

    参数:
        properties: 要设置的属性字典，会自动转换为SQL对象，如None -> NULL
        where: 解析为WHERE语句的SQL条件表达式
        from_: 解析为FROM语句的SQL语句（用于多表更新）
        with_: CTE别名/select语句的字典，用于WITH子句
        dialect: 用于解析输入表达式的SQL方言
        **opts: 用于解析输入表达式的其他选项

    返回:
        Update: UPDATE语句的语法树
    """
    # 创建基础的Update对象，解析目标表
    update_expr = Update(this=maybe_parse(table, into=Table, dialect=dialect))
    
    # 处理SET子句：将properties字典转换为赋值表达式列表
    if properties:
        update_expr.set(
            "expressions",
            [
                # 为每个属性创建EQ（等号）表达式：column = value
                # convert()函数将Python值转换为SQL字面量（如None -> NULL）
                EQ(this=maybe_parse(k, dialect=dialect, **opts), expression=convert(v))
                for k, v in properties.items()
            ],
        )
    
    # 处理FROM子句：用于多表更新场景
    if from_:
        update_expr.set(
            "from",
            # 使用"FROM"前缀确保正确解析
            maybe_parse(from_, into=From, dialect=dialect, prefix="FROM", **opts),
        )
    
    # 处理WHERE子句：特殊处理已经是Condition类型的情况
    if isinstance(where, Condition):
        # 如果已经是Condition对象，直接包装为Where
        where = Where(this=where)
    if where:
        update_expr.set(
            "where",
            # 使用"WHERE"前缀确保正确解析
            maybe_parse(where, into=Where, dialect=dialect, prefix="WHERE", **opts),
        )
    
    # 处理WITH子句：构建CTE列表
    if with_:
        cte_list = [
            # 为每个CTE创建别名，table=True表示这是表级别的别名
            alias_(CTE(this=maybe_parse(qry, dialect=dialect, **opts)), alias, table=True)
            for alias, qry in with_.items()
        ]
        update_expr.set(
            "with",
            With(expressions=cte_list),
        )
    return update_expr


def delete(
    table: ExpOrStr,                        # 要删除数据的表
    where: t.Optional[ExpOrStr] = None,     # WHERE条件
    returning: t.Optional[ExpOrStr] = None, # RETURNING子句
    dialect: DialectType = None,            # SQL方言
    **opts,
) -> Delete:
    """
    构建DELETE语句

    DELETE语句用于从表中删除满足条件的数据行，支持RETURNING子句返回被删除的数据。

    示例:
        >>> delete("my_table", where="id > 1").sql()
        'DELETE FROM my_table WHERE id > 1'

    参数:
        where: 解析为WHERE语句的SQL条件表达式
        returning: 解析为RETURNING语句的SQL条件表达式
        dialect: 用于解析输入表达式的SQL方言
        **opts: 用于解析输入表达式的其他选项

    返回:
        Delete: DELETE语句的语法树
    """
    # 创建Delete对象并设置目标表，copy=False避免不必要的复制
    delete_expr = Delete().delete(table, dialect=dialect, copy=False, **opts)
    
    # 链式调用添加WHERE条件
    if where:
        delete_expr = delete_expr.where(where, dialect=dialect, copy=False, **opts)
    
    # 链式调用添加RETURNING子句
    if returning:
        delete_expr = delete_expr.returning(returning, dialect=dialect, copy=False, **opts)
    
    return delete_expr


def insert(
    expression: ExpOrStr,                                    # 要插入的数据表达式
    into: ExpOrStr,                                         # 目标表
    columns: t.Optional[t.Sequence[str | Identifier]] = None, # 可选的列名列表
    overwrite: t.Optional[bool] = None,                     # 是否使用INSERT OVERWRITE
    returning: t.Optional[ExpOrStr] = None,                 # RETURNING子句
    dialect: DialectType = None,                            # SQL方言
    copy: bool = True,                                      # 是否复制表达式
    **opts,
) -> Insert:
    """
    构建INSERT语句

    INSERT语句用于向表中插入新的数据行，支持多种插入模式和RETURNING功能。

    示例:
        >>> insert("VALUES (1, 2, 3)", "tbl").sql()
        'INSERT INTO tbl VALUES (1, 2, 3)'

    参数:
        expression: INSERT语句的SQL字符串或表达式
        into: 要插入数据的目标表
        columns: 可选的表列名列表
        overwrite: 是否使用INSERT OVERWRITE模式
        returning: 解析为RETURNING语句的SQL条件表达式
        dialect: 用于解析输入表达式的SQL方言
        copy: 是否复制表达式
        **opts: 用于解析输入表达式的其他选项

    返回:
        Insert: INSERT语句的语法树
    """
    # 解析插入的数据表达式（如VALUES子句或SELECT语句）
    expr = maybe_parse(expression, dialect=dialect, copy=copy, **opts)
    # 解析目标表
    this: Table | Schema = maybe_parse(into, into=Table, dialect=dialect, copy=copy, **opts)

    # 处理列名列表：如果指定了columns，需要包装为Schema对象
    if columns:
        # Schema对象包含表和列的定义，用于指定插入的目标列
        this = Schema(this=this, expressions=[to_identifier(c, copy=copy) for c in columns])

    # 创建Insert对象
    insert = Insert(this=this, expression=expr, overwrite=overwrite)

    # 添加RETURNING子句
    if returning:
        # copy=False因为insert对象是新创建的，不需要复制
        insert = insert.returning(returning, dialect=dialect, copy=False, **opts)

    return insert


def merge(
    *when_exprs: ExpOrStr,                  # 可变数量的WHEN子句
    into: ExpOrStr,                        # 目标表
    using: ExpOrStr,                       # 源表或查询
    on: ExpOrStr,                          # 匹配条件
    returning: t.Optional[ExpOrStr] = None, # RETURNING子句
    dialect: DialectType = None,           # SQL方言
    copy: bool = True,                     # 是否复制表达式
    **opts,
) -> Merge:
    """
    构建MERGE语句

    MERGE语句是复杂的DML操作，根据匹配条件执行INSERT、UPDATE或DELETE操作，
    常用于数据同步和upsert（更新或插入）场景。

    示例:
        >>> merge("WHEN MATCHED THEN UPDATE SET col1 = source_table.col1",
        ...       "WHEN NOT MATCHED THEN INSERT (col1) VALUES (source_table.col1)",
        ...       into="my_table",
        ...       using="source_table",
        ...       on="my_table.id = source_table.id").sql()
        'MERGE INTO my_table USING source_table ON my_table.id = source_table.id WHEN MATCHED THEN UPDATE SET col1 = source_table.col1 WHEN NOT MATCHED THEN INSERT (col1) VALUES (source_table.col1)'

    参数:
        *when_exprs: 指定匹配和非匹配行操作的WHEN子句
        into: 要合并数据的目标表
        using: 合并数据的源表或查询
        on: 合并的连接条件
        returning: 从合并操作返回的列
        dialect: 用于解析输入表达式的SQL方言
        copy: 是否复制表达式
        **opts: 用于解析输入表达式的其他选项

    返回:
        Merge: MERGE语句的语法树
    """
    # 收集所有WHEN表达式
    expressions: t.List[Expression] = []
    for when_expr in when_exprs:
        # 解析WHEN表达式，可能是单个When或包含多个When的Whens
        expression = maybe_parse(when_expr, dialect=dialect, copy=copy, into=Whens, **opts)
        # 统一处理：如果是单个When对象，包装为列表；如果是Whens对象，提取其expressions
        expressions.extend([expression] if isinstance(expression, When) else expression.expressions)

    # 创建Merge对象，包含所有必要的组件
    merge = Merge(
        this=maybe_parse(into, dialect=dialect, copy=copy, **opts),    # 目标表
        using=maybe_parse(using, dialect=dialect, copy=copy, **opts),  # 源表
        on=maybe_parse(on, dialect=dialect, copy=copy, **opts),        # 连接条件
        whens=Whens(expressions=expressions),                          # WHEN子句集合
    )
    
    # 添加RETURNING子句
    if returning:
        merge = merge.returning(returning, dialect=dialect, copy=False, **opts)

    if isinstance(using_clause := merge.args.get("using"), Alias):
        using_clause.replace(alias_(using_clause.this, using_clause.args["alias"], table=True))

    return merge


# 逻辑条件构建函数：创建基础条件表达式
def condition(
    expression: ExpOrStr, dialect: DialectType = None, copy: bool = True, **opts
) -> Condition:
    """
    初始化逻辑条件表达式

    这是构建复杂逻辑条件的基础函数，用于将字符串或表达式转换为Condition对象，
    支持后续的逻辑运算符组合（AND、OR、NOT等）。

    示例:
        >>> condition("x=1").sql()
        'x = 1'

        这对于构建更大的逻辑语法树很有帮助:
        >>> where = condition("x=1")
        >>> where = where.and_("y=1")
        >>> Select().from_("tbl").select("*").where(where).sql()
        'SELECT * FROM tbl WHERE x = 1 AND y = 1'

    参数:
        expression: 要解析的SQL代码字符串。
            如果传入Expression实例，则直接使用。
        dialect: 用于解析输入表达式的SQL方言（仅当输入表达式是SQL字符串时）。
        copy: 是否复制`expression`（仅适用于表达式）。
        **opts: 用于解析输入表达式的其他选项（同样，仅当输入表达式是SQL字符串时）。

    返回:
        新的Condition实例
    """
    # 使用maybe_parse统一处理字符串和Expression对象
    # into=Condition确保返回的是Condition类型，支持后续逻辑运算
    return maybe_parse(
        expression,
        into=Condition,
        dialect=dialect,
        copy=copy,
        **opts,
    )


# AND逻辑运算函数：用AND操作符组合多个条件
def and_(
    *expressions: t.Optional[ExpOrStr],  # 可变数量的表达式
    dialect: DialectType = None,         # SQL方言
    copy: bool = True,                   # 是否复制表达式
    wrap: bool = True,                   # 是否用括号包装操作数
    **opts,
) -> Condition:
    """
    使用AND逻辑操作符组合多个条件

    构建AND逻辑树，支持嵌套的逻辑表达式组合。默认会添加括号以避免运算符优先级问题。

    示例:
        >>> and_("x=1", and_("y=1", "z=1")).sql()
        'x = 1 AND (y = 1 AND z = 1)'

    参数:
        *expressions: 要解析的SQL代码字符串。
            如果传入Expression实例，则直接使用。
        dialect: 用于解析输入表达式的SQL方言。
        copy: 是否复制`expressions`（仅适用于表达式）。
        wrap: 是否用`Paren`包装操作数。默认为true以避免优先级问题，
            但在生成的AST过深并导致递归相关问题时可以关闭。
        **opts: 用于解析输入表达式的其他选项。

    返回:
        新的条件
    """
    # 使用_combine函数构建AND操作符树，并强制转换为Condition类型
    # wrap=True确保复杂表达式被正确括号包装，避免运算符优先级错误
    return t.cast(Condition, _combine(expressions, And, dialect, copy=copy, wrap=wrap, **opts))


# OR逻辑运算函数：用OR操作符组合多个条件
def or_(
    *expressions: t.Optional[ExpOrStr],  # 可变数量的表达式
    dialect: DialectType = None,         # SQL方言
    copy: bool = True,                   # 是否复制表达式
    wrap: bool = True,                   # 是否用括号包装操作数
    **opts,
) -> Condition:
    """
    使用OR逻辑操作符组合多个条件

    构建OR逻辑树，支持析取（或）逻辑的表达式组合。默认会添加括号以确保正确的逻辑分组。

    示例:
        >>> or_("x=1", or_("y=1", "z=1")).sql()
        'x = 1 OR (y = 1 OR z = 1)'

    参数:
        *expressions: 要解析的SQL代码字符串。
            如果传入Expression实例，则直接使用。
        dialect: 用于解析输入表达式的SQL方言。
        copy: 是否复制`expressions`（仅适用于表达式）。
        wrap: 是否用`Paren`包装操作数。默认为true以避免优先级问题，
            但在生成的AST过深并导致递归相关问题时可以关闭。
        **opts: 用于解析输入表达式的其他选项。

    返回:
        新的条件
    """
    # 使用_combine函数构建OR操作符树，强制转换为Condition类型
    # OR操作符的优先级低于AND，因此括号包装尤其重要
    return t.cast(Condition, _combine(expressions, Or, dialect, copy=copy, wrap=wrap, **opts))


# XOR逻辑运算函数：用XOR操作符组合多个条件
def xor(
    *expressions: t.Optional[ExpOrStr],  # 可变数量的表达式
    dialect: DialectType = None,         # SQL方言
    copy: bool = True,                   # 是否复制表达式
    wrap: bool = True,                   # 是否用括号包装操作数
    **opts,
) -> Condition:
    """
    使用XOR逻辑操作符组合多个条件

    构建XOR（异或）逻辑树，当且仅当两个条件中恰好有一个为真时，结果才为真。
    这是一个不常用但重要的逻辑操作符，主要用于特殊的逻辑判断场景。

    示例:
        >>> xor("x=1", xor("y=1", "z=1")).sql()
        'x = 1 XOR (y = 1 XOR z = 1)'

    参数:
        *expressions: 要解析的SQL代码字符串。
            如果传入Expression实例，则直接使用。
        dialect: 用于解析输入表达式的SQL方言。
        copy: 是否复制`expressions`（仅适用于表达式）。
        wrap: 是否用`Paren`包装操作数。默认为true以避免优先级问题，
            但在生成的AST过深并导致递归相关问题时可以关闭。
        **opts: 用于解析输入表达式的其他选项。

    返回:
        新的条件
    """
    # 使用_combine函数构建XOR操作符树，强制转换为Condition类型
    # XOR的语义要求精确的逻辑分组，因此括号包装至关重要
    return t.cast(Condition, _combine(expressions, Xor, dialect, copy=copy, wrap=wrap, **opts))


def not_(expression: ExpOrStr, dialect: DialectType = None, copy: bool = True, **opts) -> Not:
    """
    用NOT操作符包装条件

    创建逻辑否定表达式，将条件的真假值反转。NOT是一元操作符，
    只作用于单个条件表达式。

    示例:
        >>> not_("this_suit='black'").sql()
        "NOT this_suit = 'black'"

    参数:
        expression: 要解析的SQL代码字符串。
            如果传入Expression实例，则直接使用。
        dialect: 用于解析输入表达式的SQL方言。
        copy: 是否复制表达式。
        **opts: 用于解析输入表达式的其他选项。

    返回:
        新的条件。
    """
    # 首先将输入转换为Condition对象
    this = condition(
        expression,
        dialect=dialect,
        copy=copy,
        **opts,
    )
    # 创建NOT对象，使用_wrap确保Connector类型的表达式被适当包装
    # 这避免了像NOT x AND y这样的歧义，确保生成NOT (x AND y)的正确语义
    return Not(this=_wrap(this, Connector))


# 括号包装函数：为表达式添加圆括号
def paren(expression: ExpOrStr, copy: bool = True) -> Paren:
    """
    用圆括号包装表达式

    在SQL中，圆括号用于明确运算符优先级和逻辑分组，确保表达式按预期方式求值。
    这个函数提供了一个简便的方式来为任何表达式添加括号包装。

    示例:
        >>> paren("5 + 3").sql()
        '(5 + 3)'

    参数:
        expression: 要解析的SQL代码字符串。
            如果传入Expression实例，则直接使用。
        copy: 是否复制表达式。

    返回:
        包装后的表达式。
    """
    # 使用maybe_parse处理输入，然后用Paren对象包装
    # 这确保了无论输入是字符串还是Expression都能正确处理
    return Paren(this=maybe_parse(expression, copy=copy))


# 安全标识符正则表达式：匹配不需要引号的标识符
# 标识符必须以字母或下划线开头，后跟字母、数字或下划线
SAFE_IDENTIFIER_RE: t.Pattern[str] = re.compile(r"^[_a-zA-Z][\w]*$")


# 函数重载：处理None输入的情况
@t.overload
def to_identifier(name: None, quoted: t.Optional[bool] = None, copy: bool = True) -> None: ...


# 函数重载：处理字符串或Identifier输入的情况
@t.overload
def to_identifier(
    name: str | Identifier, quoted: t.Optional[bool] = None, copy: bool = True
) -> Identifier: ...


# 标识符构建函数：将字符串或现有标识符转换为Identifier对象
def to_identifier(name, quoted=None, copy=True):
    """构建标识符

    将输入转换为SQL标识符，自动处理引号需求。SQL标识符可能需要用引号包围，
    特别是当它们包含特殊字符、空格或与保留关键字冲突时。

    参数:
        name: 要转换为标识符的名称。
        quoted: 是否强制给标识符加引号。
        copy: 如果name是Identifier，是否复制它。

    返回:
        标识符AST节点。
    """
    # 处理None输入：直接返回None
    if name is None:
        return None

    # 处理已经是Identifier的情况：根据copy参数决定是否复制
    if isinstance(name, Identifier):
        identifier = maybe_copy(name, copy)
    elif isinstance(name, str):
        # 处理字符串输入：创建新的Identifier对象
        identifier = Identifier(
            this=name,
            # 智能引号决策：如果quoted未指定，则根据SAFE_IDENTIFIER_RE正则表达式判断
            # 不符合安全标识符模式的名称需要加引号（如包含空格、特殊字符等）
            quoted=not SAFE_IDENTIFIER_RE.match(name) if quoted is None else quoted,
        )
    else:
        # 类型错误：输入必须是字符串或Identifier
        raise ValueError(f"Name needs to be a string or an Identifier, got: {name.__class__}")
    return identifier


# 标识符解析函数：尝试解析字符串为标识符，支持容错处理
def parse_identifier(name: str | Identifier, dialect: DialectType = None) -> Identifier:
    """
    将给定字符串解析为标识符

    这个函数提供了比to_identifier更强大的解析能力，它会尝试使用完整的SQL解析器
    来处理输入，如果解析失败则回退到简单的标识符构建。

    参数:
        name: 要解析为标识符的名称。
        dialect: 要解析的SQL方言。

    返回:
        标识符AST节点。
    """
    try:
        # 尝试使用完整的SQL解析器解析标识符
        # 这可以处理更复杂的标识符格式，如带引号的标识符
        expression = maybe_parse(name, dialect=dialect, into=Identifier)
    except (ParseError, TokenError):
        # 如果解析失败，回退到简单的标识符构建
        # 这确保了即使在复杂解析失败的情况下也能创建基本的标识符
        expression = to_identifier(name)

    return expression


# 时间间隔字符串正则表达式：匹配"数字 + 单位"格式的时间间隔
# 支持可选的负号、小数点，以及字母单位（如day、month、year等）
INTERVAL_STRING_RE = re.compile(r"\s*(-?[0-9]+(?:\.[0-9]+)?)\s*([a-zA-Z]+)\s*")

# Matches day-time interval strings that contain
# - A number of days (possibly negative or with decimals)
# - At least one space
# - Portions of a time-like signature, potentially negative
#   - Standard format                   [-]h+:m+:s+[.f+]
#   - Just minutes/seconds/frac seconds [-]m+:s+.f+
#   - Just hours, minutes, maybe colon  [-]h+:m+[:]
#   - Just hours, maybe colon           [-]h+[:]
#   - Just colon                        :
INTERVAL_DAY_TIME_RE = re.compile(
    r"\s*-?\s*\d+(?:\.\d+)?\s+(?:-?(?:\d+:)?\d+:\d+(?:\.\d+)?|-?(?:\d+:){1,2}|:)\s*"
)


# 时间间隔构建函数：从字符串构建INTERVAL表达式
def to_interval(interval: str | Literal) -> Interval:
    """从类似'1 day'或'5 months'的字符串构建时间间隔表达式
    
    时间间隔在SQL中用于日期时间运算，如日期加减、时间差计算等。
    这个函数标准化了时间间隔的创建过程。
    """
    # 处理Literal类型的输入
    if isinstance(interval, Literal):
        # 验证Literal必须是字符串类型
        if not interval.is_string:
            raise ValueError("Invalid interval string.")
        
        # 提取Literal对象中的实际字符串值
        interval = interval.this

    # 构造完整的INTERVAL SQL语句并解析
    # 添加"INTERVAL"关键字是因为SQL标准要求时间间隔必须以INTERVAL开头
    interval = maybe_parse(f"INTERVAL {interval}")
    
    # 类型断言：确保解析结果确实是Interval对象
    # 这是一个安全检查，防止解析产生意外的表达式类型
    assert isinstance(interval, Interval)
    return interval


def to_table(
    sql_path: str | Table, dialect: DialectType = None, copy: bool = True, **kwargs
) -> Table:
    """
    从`[catalog].[schema].[table]`SQL路径创建表表达式，目录和模式是可选的。
    如果传入的是Table对象，则直接返回该表。

    这个函数处理了SQL中表名的完整层次结构，支持三层命名空间：
    catalog.schema.table，这在企业级数据库环境中很常见。

    参数:
        sql_path: `[catalog].[schema].[table]`格式的字符串。
        dialect: 用于解析表名的源SQL方言。
        copy: 如果传入Table对象，是否复制它。
        kwargs: 用于实例化结果`Table`表达式的额外参数。

    返回:
        Table表达式。
    """
    # 如果输入已经是Table对象，根据copy参数决定是否复制
    if isinstance(sql_path, Table):
        return maybe_copy(sql_path, copy=copy)

    try:
        # 尝试使用完整的SQL解析器解析表路径
        # 这可以处理复杂的表名格式，如带引号的标识符或特殊字符
        table = maybe_parse(sql_path, into=Table, dialect=dialect)
    except ParseError:
        # 如果解析失败，使用手动分割的方式处理路径
        # split_num_words按"."分割字符串，最多分割成3部分
        catalog, db, this = split_num_words(sql_path, ".", 3)

        # 如果连表名都没有，则重新抛出原始解析错误
        if not this:
            raise

        # 使用table_函数构建Table对象，支持catalog.schema.table层次结构
        table = table_(this, db=db, catalog=catalog)

    # 应用额外的kwargs参数到Table对象
    # 这允许设置额外的属性，如表的元数据信息
    for k, v in kwargs.items():
        table.set(k, v)

    return table


# 列对象构建函数：从SQL路径创建Column表达式
def to_column(
    sql_path: str | Column,             # SQL路径字符串或现有Column对象
    quoted: t.Optional[bool] = None,    # 是否强制给标识符加引号
    dialect: DialectType = None,        # SQL方言
    copy: bool = True,                  # 是否复制现有Column对象
    **kwargs,                           # 额外的Column属性
) -> Column:
    """
    从`[table].[column]`SQL路径创建列，表名是可选的。
    如果传入的是Column对象，则直接返回该列。

    列引用在SQL中可以是简单的列名，也可以是完全限定的table.column格式，
    这在多表查询中用于消除歧义。

    参数:
        sql_path: `[table].[column]`格式的字符串。
        quoted: 是否强制给标识符加引号。
        dialect: 用于解析列名的源SQL方言。
        copy: 如果传入Column对象，是否复制它。
        kwargs: 用于实例化结果`Column`表达式的额外参数。

    返回:
        Column表达式。
    """
    # 如果输入已经是Column对象，根据copy参数决定是否复制
    if isinstance(sql_path, Column):
        return maybe_copy(sql_path, copy=copy)

    try:
        # 尝试使用完整的SQL解析器解析列路径
        col = maybe_parse(sql_path, into=Column, dialect=dialect)
    except ParseError:
        # 解析失败时使用简单的字符串分割方式
        # reversed()是因为column()函数期望的参数顺序是(column, table)
        # 而sql_path的格式是"table.column"，所以需要反转
        return column(*reversed(sql_path.split(".")), quoted=quoted, **kwargs)

    # 应用额外的kwargs参数到Column对象
    for k, v in kwargs.items():
        col.set(k, v)

    # 如果指定了quoted=True，则强制为所有标识符添加引号
    if quoted:
        # 遍历Column中的所有Identifier节点并设置quoted属性
        # 这确保了表名和列名都会被正确引用
        for i in col.find_all(Identifier):
            i.set("quoted", True)

    return col


# 别名构建函数：为表达式创建别名
def alias_(
    expression: ExpOrStr,                                    # 要添加别名的表达式
    alias: t.Optional[str | Identifier],                    # 别名名称
    table: bool | t.Sequence[str | Identifier] = False,     # 是否创建表别名或列别名列表
    quoted: t.Optional[bool] = None,                        # 是否给别名加引号
    dialect: DialectType = None,                            # SQL方言
    copy: bool = True,                                      # 是否复制表达式
    **opts,
):
    """创建Alias表达式

    别名在SQL中用于给表、列、子查询等提供临时名称，提高查询可读性或解决命名冲突。
    支持简单别名和表别名（包含列名列表）两种模式。

    示例:
        >>> alias_('foo', 'bar').sql()
        'foo AS bar'

        >>> alias_('(select 1, 2)', 'bar', table=['a', 'b']).sql()
        '(SELECT 1, 2) AS bar(a, b)'

    参数:
        expression: 要解析的SQL代码字符串。
            如果传入Expression实例，则直接使用。
        alias: 要使用的别名名称。如果名称包含特殊字符，会自动加引号。
        table: 是否创建表别名，也可以是列名列表。
        quoted: 是否给别名加引号
        dialect: 用于解析输入表达式的SQL方言。
        copy: 是否复制表达式。
        **opts: 用于解析输入表达式的其他选项。

    返回:
        Alias: 添加了别名的表达式
    """
    # 解析输入表达式
    exp = maybe_parse(expression, dialect=dialect, copy=copy, **opts)
    # 将别名转换为Identifier对象
    alias = to_identifier(alias, quoted=quoted)

    # 处理表别名的情况
    if table:
        # 创建TableAlias对象，用于表级别的别名
        table_alias = TableAlias(this=alias)
        exp.set("alias", table_alias)

        # 如果table不是简单的bool值，而是列名列表
        if not isinstance(table, bool):
            # 为每个列名创建标识符并添加到表别名的columns中
            # 这支持形如"SELECT ... FROM (subquery) AS alias(col1, col2)"的语法
            for column in table:
                table_alias.append("columns", to_identifier(column, quoted=quoted))

        return exp

    # 特殊处理Window表达式的别名
    # Window表达式的别名处理有特殊性，需要创建ALIAS节点而不是IDENTIFIER节点
    # 这是因为Window表达式的"alias"参数表示的是"named_window"构造（如BigQuery中的用法）
    # 而我们需要的是为整个Window表达式创建别名
    #
    # 参考: https://cloud.google.com/bigquery/docs/reference/standard-sql/window-function-calls

    if "alias" in exp.arg_types and not isinstance(exp, Window):
        # 对于支持alias参数且不是Window的表达式，直接设置alias属性
        exp.set("alias", alias)
        return exp
    
    # 对于其他情况（包括Window表达式），创建独立的Alias节点包装表达式
    return Alias(this=exp, alias=alias)


# 子查询构建函数：创建可查询的子查询表达式
def subquery(
    expression: ExpOrStr,                # 子查询的SQL表达式
    alias: t.Optional[Identifier | str] = None,  # 子查询别名
    dialect: DialectType = None,         # SQL方言
    **opts,
) -> Select:
    """
    构建用于查询的子查询表达式

    子查询是嵌套在另一个查询中的查询，常用于复杂的数据检索场景。
    这个函数将任何表达式包装为可以在FROM子句中使用的子查询。

    示例:
        >>> subquery('select x from tbl', 'bar').select('x').sql()
        'SELECT x FROM (SELECT x FROM tbl) AS bar'

    参数:
        expression: 要解析的SQL代码字符串。
            如果传入Expression实例，则直接使用。
        alias: 要使用的别名名称。
        dialect: 用于解析输入表达式的SQL方言。
        **opts: 用于解析输入表达式的其他选项。

    返回:
        包含子查询表达式的新Select实例。
    """
    # 解析表达式并调用其subquery方法创建子查询
    # subquery()方法会将表达式包装在括号中并可选地添加别名
    expression = maybe_parse(expression, dialect=dialect, **opts).subquery(alias, **opts)
    
    # 创建一个新的Select查询，将子查询作为FROM子句的源
    # 这使得子查询可以被进一步查询，支持链式操作
    return Select().from_(expression, dialect=dialect, **opts)


# 函数重载：带有fields参数时返回Dot对象
@t.overload
def column(
    col: str | Identifier,
    table: t.Optional[str | Identifier] = None,
    db: t.Optional[str | Identifier] = None,
    catalog: t.Optional[str | Identifier] = None,
    *,
    fields: t.Collection[t.Union[str, Identifier]],
    quoted: t.Optional[bool] = None,
    copy: bool = True,
) -> Dot:
    pass


# 函数重载：不带fields参数时返回Column对象
@t.overload
def column(
    col: str | Identifier | Star,
    table: t.Optional[str | Identifier] = None,
    db: t.Optional[str | Identifier] = None,
    catalog: t.Optional[str | Identifier] = None,
    *,
    fields: Lit[None] = None,
    quoted: t.Optional[bool] = None,
    copy: bool = True,
) -> Column:
    pass


# 列构建函数：创建Column或Dot表达式
def column(
    col,                    # 列名
    table=None,             # 表名
    db=None,                # 数据库名
    catalog=None,           # 目录名
    *,
    fields=None,            # 附加字段（用点分隔）
    quoted=None,            # 是否强制给标识符加引号
    copy=True,              # 是否复制标识符
):
    """
    构建Column

    支持从简单列名到完全限定的catalog.db.table.column格式，
    还可以通过fields参数支持嵌套字段访问（如JSON或结构体字段）。

    参数:
        col: 列名。
        table: 表名。
        db: 数据库名。
        catalog: 目录名。
        fields: 使用点访问的附加字段。
        quoted: 是否强制给列的标识符加引号。
        copy: 如果传入标识符，是否复制它们。

    返回:
        新的Column实例。
    """
    # 特殊处理Star对象：Star（*）不需要转换为标识符
    if not isinstance(col, Star):
        col = to_identifier(col, quoted=quoted, copy=copy)

    # 创建基本的Column对象，支持完整的四层命名空间
    this = Column(
        this=col,                                                    # 列名
        table=to_identifier(table, quoted=quoted, copy=copy),       # 表名
        db=to_identifier(db, quoted=quoted, copy=copy),             # 数据库名
        catalog=to_identifier(catalog, quoted=quoted, copy=copy),   # 目录名
    )

    # 处理嵌套字段访问：如果指定了fields，创建Dot表达式链
    if fields:
        # 使用Dot.build创建点访问链，支持JSON字段或结构体字段访问
        # 如：column('data', fields=['user', 'name']) -> data.user.name
        this = Dot.build(
            (this, *(to_identifier(field, quoted=quoted, copy=copy) for field in fields))
        )
    return this


# 类型转换函数：将表达式转换为指定数据类型
def cast(
    expression: ExpOrStr,           # 要转换的表达式
    to: DATA_TYPE,                  # 目标数据类型
    copy: bool = True,              # 是否复制提供的表达式
    dialect: DialectType = None,    # 目标SQL方言
    **opts
) -> Cast:
    """将表达式转换为数据类型

    CAST是SQL中的类型转换操作，用于显式地将一个数据类型转换为另一个。
    这个函数包含智能优化，避免不必要的重复转换。

    示例:
        >>> cast('x + 1', 'int').sql()
        'CAST(x + 1 AS INT)'

    参数:
        expression: 要转换的表达式。
        to: 要转换到的数据类型。
        copy: 是否复制提供的表达式。
        dialect: 目标方言。用于防止以下场景中的重复转换：
            - 要转换的表达式已经是exp.Cast表达式
            - 现有的转换是到与新类型逻辑等价的类型

            例如，如果:expression='CAST(x as DATETIME)'且:to=Type.TIMESTAMP，
            但在目标方言中DATETIME映射到TIMESTAMP，那么我们不会返回`CAST(x (as DATETIME) as TIMESTAMP)`
            而是只返回原始表达式`CAST(x as DATETIME)`。

            这是为了防止在目标方言生成器中应用DATETIME -> TIMESTAMP
            映射后输出为双重转换`CAST(x (as TIMESTAMP) as TIMESTAMP)`。

    返回:
        新的Cast实例。
    """
    # 解析输入表达式
    expr = maybe_parse(expression, copy=copy, dialect=dialect, **opts)
    # 构建目标数据类型
    data_type = DataType.build(to, copy=copy, dialect=dialect, **opts)

    # 智能优化：避免对已经正确类型转换的表达式进行重复转换
    if isinstance(expr, Cast):
        from sqlglot.dialects.dialect import Dialect

        # 获取目标方言并查找类型映射表
        target_dialect = Dialect.get_or_raise(dialect)
        type_mapping = target_dialect.generator_class.TYPE_MAPPING

        # 提取现有转换的类型和新转换的类型
        existing_cast_type: DataType.Type = expr.to.this
        new_cast_type: DataType.Type = data_type.this
        
        # 检查类型是否在目标方言中等价
        # type_mapping将某些类型映射为等价类型（如DATETIME -> TIMESTAMP）
        types_are_equivalent = type_mapping.get(
            existing_cast_type, existing_cast_type.value
        ) == type_mapping.get(new_cast_type, new_cast_type.value)

        # 如果已经是目标类型或类型等价，则返回现有表达式
        if expr.is_type(data_type) or types_are_equivalent:
            return expr

    # 创建新的Cast表达式
    expr = Cast(this=expr, to=data_type)
    # 设置类型信息以便后续优化使用
    expr.type = data_type

    return expr


# 表构建函数：创建Table表达式
def table_(
    table: Identifier | str,                    # 表名
    db: t.Optional[Identifier | str] = None,    # 数据库名
    catalog: t.Optional[Identifier | str] = None,  # 目录名
    quoted: t.Optional[bool] = None,            # 是否强制给标识符加引号
    alias: t.Optional[Identifier | str] = None, # 表别名
) -> Table:
    """构建Table

    创建表表达式，支持完整的三层命名空间（catalog.db.table）
    和可选的表别名。

    参数:
        table: 表名。
        db: 数据库名。
        catalog: 目录名。
        quote: 是否强制给表的标识符加引号。
        alias: 表的别名。

    返回:
        新的Table实例。
    """
    return Table(
        # 条件性转换：只有在值存在时才转换为标识符
        this=to_identifier(table, quoted=quoted) if table else None,
        db=to_identifier(db, quoted=quoted) if db else None,
        catalog=to_identifier(catalog, quoted=quoted) if catalog else None,
        # 表别名需要包装在TableAlias对象中
        alias=TableAlias(this=to_identifier(alias)) if alias else None,
    )


# VALUES语句构建函数：创建VALUES表达式
def values(
    values: t.Iterable[t.Tuple[t.Any, ...]],                    # 值的元组集合
    alias: t.Optional[str] = None,                              # 可选别名
    columns: t.Optional[t.Iterable[str] | t.Dict[str, DataType]] = None,  # 可选列名或列类型映射
) -> Values:
    """构建VALUES语句

    VALUES语句用于指定一组常量行，常用于INSERT语句或作为临时表使用。
    支持可选的别名和列名定义。

    示例:
        >>> values([(1, '2')]).sql()
        "VALUES (1, '2')"

    参数:
        values: 将被转换为SQL的值语句
        alias: 可选别名
        columns: 可选的有序列名列表或列名到类型的有序字典。
         如果提供了任一个，则也需要提供别名。

    返回:
        Values: Values表达式对象
    """
    # 验证：如果提供了列定义，必须也提供别名
    if columns and not alias:
        raise ValueError("Alias is required when providing columns")

    return Values(
        # 将每个元组转换为SQL表达式
        expressions=[convert(tup) for tup in values],
        # 构建表别名：根据是否提供列名选择不同的构造方式
        alias=(
            # 如果提供了列名，创建包含列定义的TableAlias
            TableAlias(this=to_identifier(alias), columns=[to_identifier(x) for x in columns])
            if columns
            # 如果只提供了别名，创建简单的TableAlias
            else (TableAlias(this=to_identifier(alias)) if alias else None)
        ),
    )


# 变量构建函数：创建SQL变量
def var(name: t.Optional[ExpOrStr]) -> Var:
    """构建SQL变量

    SQL变量用于参数化查询或存储过程中的变量引用。
    可以从字符串或表达式的名称创建变量。

    示例:
        >>> repr(var('x'))
        'Var(this=x)'

        >>> repr(var(column('x', table='y')))
        'Var(this=x)'

    参数:
        name: 变量的名称或名称将成为变量的表达式。

    返回:
        新的变量节点。
    """
    # 验证：变量名不能为空
    if not name:
        raise ValueError("Cannot convert empty name into var.")

    # 如果输入是表达式，提取其名称作为变量名
    if isinstance(name, Expression):
        # 使用表达式的name属性，这会提取表达式的核心标识符
        name = name.name
    
    # 创建Var对象
    return Var(this=name)


def rename_table(
    old_name: str | Table,          # 旧表名
    new_name: str | Table,          # 新表名
    dialect: DialectType = None,    # SQL方言
) -> Alter:
    """构建ALTER TABLE... RENAME...表达式

    表重命名是常见的DDL操作，用于修改现有表的名称。
    这个函数生成标准的ALTER TABLE语句来执行重命名操作。

    参数:
        old_name: 表的旧名称
        new_name: 表的新名称
        dialect: 用于解析表名的SQL方言。

    返回:
        Alter表达式
    """
    # 将输入的表名转换为Table对象，支持字符串和Table对象
    old_table = to_table(old_name, dialect=dialect)
    new_table = to_table(new_name, dialect=dialect)
    
    # 构建ALTER语句：指定操作类型为TABLE，添加RENAME动作
    return Alter(
        this=old_table,                         # 要修改的目标表
        kind="TABLE",                           # 指定ALTER的对象类型为表
        actions=[
            AlterRename(this=new_table),        # 重命名动作，指向新的表名
        ],
    )


def rename_column(
    table_name: str | Table,                    # 表名
    old_column_name: str | Column,              # 旧列名
    new_column_name: str | Column,              # 新列名
    exists: t.Optional[bool] = None,            # 是否添加IF EXISTS子句
    dialect: DialectType = None,                # SQL方言
) -> Alter:
    """构建ALTER TABLE... RENAME COLUMN...表达式

    列重命名是DDL操作中的常见需求，用于修改表中现有列的名称。
    支持可选的IF EXISTS子句来避免列不存在时的错误。

    参数:
        table_name: 表的名称
        old_column: 列的旧名称
        new_column: 列的新名称
        exists: 是否添加`IF EXISTS`子句
        dialect: 用于解析表名/列名的SQL方言。

    返回:
        Alter表达式
    """
    # 转换输入参数为相应的AST对象
    table = to_table(table_name, dialect=dialect)              # 目标表
    old_column = to_column(old_column_name, dialect=dialect)   # 旧列名
    new_column = to_column(new_column_name, dialect=dialect)   # 新列名
    
    # 构建ALTER TABLE语句，包含列重命名动作
    return Alter(
        this=table,                                             # 要修改的表
        kind="TABLE",                                           # 指定ALTER的对象类型
        actions=[
            # RenameColumn动作：从旧列名重命名为新列名
            RenameColumn(this=old_column, to=new_column, exists=exists),
        ],
    )


# Python值转换函数：将Python对象转换为SQL表达式
def convert(value: t.Any, copy: bool = False) -> Expression:
    """将Python值转换为表达式对象

    这是SQLGlot的核心转换函数，负责将各种Python数据类型映射为相应的SQL表达式。
    支持基本类型、复合类型、时间类型等的全面转换。

    如果无法转换则抛出错误。

    参数:
        value: Python对象。
        copy: 是否复制`value`（仅适用于表达式和集合）。

    返回:
        等价的表达式对象。
    """
    # 处理已经是Expression的情况：根据copy参数决定是否复制
    if isinstance(value, Expression):
        return maybe_copy(value, copy)
    
    # 字符串转换：创建字符串字面量
    if isinstance(value, str):
        return Literal.string(value)
    
    # 布尔值转换：创建Boolean表达式
    if isinstance(value, bool):
        return Boolean(this=value)
    
    # NULL值处理：None或NaN都转换为SQL的NULL
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return null()
    
    # 数值转换：整数、浮点数等转换为数值字面量
    if isinstance(value, numbers.Number):
        return Literal.number(value)
    
    # 字节串转换：转换为十六进制字符串表示
    if isinstance(value, bytes):
        return HexString(this=value.hex())
    
    # 日期时间转换：处理带时区信息的完整日期时间
    if isinstance(value, datetime.datetime):
        # 使用ISO格式并用空格分隔日期和时间（更符合SQL标准）
        datetime_literal = Literal.string(value.isoformat(sep=" "))

        tz = None
        if value.tzinfo:
            # 提取时区信息：支持zoneinfo.ZoneInfo、pytz.timezone和datetime.datetime.utc
            # 返回IANA时区名称如"America/Los_Angeles"而不是缩写如"PDT"
            # 这与SQLGlot中其他时区处理函数保持一致
            tz = Literal.string(str(value.tzinfo))

        # 创建时间字符串到时间的转换表达式，包含时区信息
        return TimeStrToTime(this=datetime_literal, zone=tz)
    
    # 日期转换：仅日期部分，使用标准格式
    if isinstance(value, datetime.date):
        date_literal = Literal.string(value.strftime("%Y-%m-%d"))
        return DateStrToDate(this=date_literal)
    
    # 时间转换：仅时间部分，使用ISO格式
    if isinstance(value, datetime.time):
        time_literal = Literal.string(value.isoformat())
        return TsOrDsToTime(this=time_literal)
    
    # 元组转换：区分命名元组和普通元组
    if isinstance(value, tuple):
        # 命名元组（namedtuple）转换为结构体，保留字段名
        if hasattr(value, "_fields"):
            return Struct(
                expressions=[
                    # 为每个字段创建属性等号表达式：field_name = field_value
                    PropertyEQ(
                        this=to_identifier(k), expression=convert(getattr(value, k), copy=copy)
                    )
                    for k in value._fields
                ]
            )
        # 普通元组转换为Tuple表达式
        return Tuple(expressions=[convert(v, copy=copy) for v in value])
    
    # 列表转换：转换为数组表达式
    if isinstance(value, list):
        return Array(expressions=[convert(v, copy=copy) for v in value])
    
    # 字典转换：转换为Map表达式，分别处理键和值
    if isinstance(value, dict):
        return Map(
            # 键数组：所有字典键转换为数组
            keys=Array(expressions=[convert(k, copy=copy) for k in value]),
            # 值数组：所有字典值转换为数组
            values=Array(expressions=[convert(v, copy=copy) for v in value.values()]),
        )
    
    # 自定义对象转换：具有__dict__属性的对象转换为结构体
    if hasattr(value, "__dict__"):
        return Struct(
            expressions=[
                # 将对象的每个属性转换为PropertyEQ表达式
                PropertyEQ(this=to_identifier(k), expression=convert(v, copy=copy))
                for k, v in value.__dict__.items()
            ]
        )
    
    # 无法转换的类型：抛出错误
    raise ValueError(f"Cannot convert {value}")


# 子节点替换函数：用函数结果替换表达式的子节点
def replace_children(expression: Expression, fun: t.Callable, *args, **kwargs) -> None:
    """
    用lambda函数fun(child) -> exp的结果替换表达式的子节点

    这个函数遍历表达式的所有直接子节点，对每个Expression类型的子节点应用转换函数，
    然后用结果替换原始子节点。支持单个节点和节点列表两种参数类型。
    """
    # 遍历表达式的所有参数，使用tuple避免在迭代过程中修改字典
    for k, v in tuple(expression.args.items()):
        # 检查参数是否为列表类型，用于后续的不同处理逻辑
        is_list_arg = type(v) is list

        # 统一处理：将单个值包装为列表，列表保持原样
        child_nodes = v if is_list_arg else [v]
        new_child_nodes = []

        # 处理每个子节点
        for cn in child_nodes:
            if isinstance(cn, Expression):
                # 对Expression类型的子节点应用转换函数
                # ensure_collection确保返回值是可迭代的集合
                for child_node in ensure_collection(fun(cn, *args, **kwargs)):
                    new_child_nodes.append(child_node)
            else:
                # 非Expression类型的节点直接保留
                new_child_nodes.append(cn)

        # 根据原始参数类型设置新值：列表保持列表，单值取第一个元素
        expression.set(k, new_child_nodes if is_list_arg else seq_get(new_child_nodes, 0))


# 表达式树替换函数：用函数调用结果替换整个表达式树
def replace_tree(
    expression: Expression,                                    # 要处理的根表达式
    fun: t.Callable,                                          # 应用于每个节点的转换函数
    prune: t.Optional[t.Callable[[Expression], bool]] = None, # 可选的剪枝函数
) -> Expression:
    """
    用函数调用结果替换整个表达式树的每个节点

    这个函数使用反向深度优先搜索（DFS）遍历表达式树，从叶子节点开始处理。
    如果函数调用产生了新节点，这些新节点也会被遍历处理。

    参数:
        expression: 要处理的表达式树
        fun: 应用于每个节点的转换函数
        prune: 可选的剪枝函数，返回True的节点及其子树将被跳过

    返回:
        处理后的表达式树根节点
    """
    # 使用DFS获取所有需要处理的节点，存储在栈中
    # 反向DFS确保叶子节点先被处理
    stack = list(expression.dfs(prune=prune))

    # 处理栈中的每个节点
    while stack:
        # 从栈顶取出节点（确保叶子优先处理）
        node = stack.pop()
        # 应用转换函数
        new_node = fun(node)

        # 如果产生了新节点，执行替换操作
        if new_node is not node:
            # 在树中用新节点替换旧节点
            node.replace(new_node)

            # 如果新节点也是Expression，将其加入栈中以便进一步处理
            # 这处理了转换函数可能创建新的复合表达式的情况
            if isinstance(new_node, Expression):
                stack.append(new_node)

    # 返回最终的根节点
    return new_node


def find_tables(expression: Expression) -> t.Set[Table]:
    """
    Find all tables referenced in a query.

    Args:
        expressions: The query to find the tables in.

    Returns:
        A set of all the tables.
    """
    from sqlglot.optimizer.scope import traverse_scope

    return {
        table
        for scope in traverse_scope(expression)
        for table in scope.tables
        if table.name and table.name not in scope.cte_sources
    }


def column_table_names(expression: Expression, exclude: str = "") -> t.Set[str]:
    """
    返回表达式中通过列引用的所有表名

    这个函数分析SQL表达式，找出所有形如table.column格式的列引用，
    并提取其中的表名。常用于依赖分析和查询优化。

    示例:
        >>> import sqlglot
        >>> sorted(column_table_names(sqlglot.parse_one("a.b AND c.d AND c.e")))
        ['a', 'c']

    参数:
        expression: 要查找表名的表达式
        exclude: 要排除的表名

    返回:
        唯一表名的集合
    """
    return {
        table
        # 从表达式中找出所有Column节点，提取其table属性
        for table in (column.table for column in expression.find_all(Column))
        # 过滤掉空值和需要排除的表名
        if table and table != exclude
    }


def table_name(
    table: Table | str,              # 表表达式节点或字符串
    dialect: DialectType = None,     # 生成表名的SQL方言
    identify: bool = False           # 是否强制引用标识符
) -> str:
    """获取表的完整名称字符串

    将Table表达式或字符串转换为完整的表名字符串，支持多层命名空间
    （如catalog.schema.table）和智能引号处理。

    参数:
        table: Table表达式节点或字符串
        dialect: 生成表名的SQL方言
        identify: 确定何时应该引用标识符。可能的值：
            False（默认）：除非方言强制要求，否则不引用
            True：总是引用

    示例:
        >>> from sqlglot import exp, parse_one
        >>> table_name(parse_one("select * from a.b.c").find(exp.Table))
        'a.b.c'

    返回:
        表名字符串
    """
    # 将输入转换为Table对象
    table = maybe_parse(table, into=Table, dialect=dialect)

    # 验证解析结果
    if not table:
        raise ValueError(f"Cannot parse {table}")

    # 构建完整的表名：将表的各个部分用点连接
    return ".".join(
        (
            # 根据identify参数和标识符安全性决定是否引用
            part.sql(dialect=dialect, identify=True, copy=False, comments=False)
            if identify or not SAFE_IDENTIFIER_RE.match(part.name)  # 强制引用或不安全标识符需要引用
            else part.name  # 安全标识符直接使用名称
        )
        for part in table.parts  # 遍历表的所有部分（catalog、schema、table等）
    )


def normalize_table_name(
    table: str | Table,              # 要标准化的表
    dialect: DialectType = None,     # 标准化规则使用的方言
    copy: bool = True                # 是否复制表达式
) -> str:
    """返回去除引号的大小写标准化表名

    这个函数对表名进行标准化处理，去除引号并根据方言规则统一大小写。
    主要用于表名比较和映射操作中的键标准化。

    参数:
        table: 要标准化的表
        dialect: 标准化规则使用的方言
        copy: 是否复制表达式

    示例:
        >>> normalize_table_name("`A-B`.c", dialect="bigquery")
        'A-B.c'
    """
    # 导入标识符标准化模块
    from sqlglot.optimizer.normalize_identifiers import normalize_identifiers

    # 执行标准化并重新组合表名：去除引号，统一大小写，用点连接各部分
    return ".".join(
        p.name  # 提取每个部分的名称（已去除引号）
        for p in normalize_identifiers(
            to_table(table, dialect=dialect, copy=copy), dialect=dialect
        ).parts  # 获取标准化后表的所有部分（catalog、schema、table等）
    )


# 表替换函数：根据映射替换表达式中的所有表
def replace_tables(
    expression: E,                           # 要转换和替换的表达式节点
    mapping: t.Dict[str, str],               # 表名映射字典
    dialect: DialectType = None,             # 映射表的方言
    copy: bool = True                        # 是否复制表达式
) -> E:
    """根据映射替换表达式中的所有表

    这个函数遍历表达式树，找到所有Table节点并根据提供的映射进行替换。
    常用于查询重写、表迁移和别名替换等场景。

    参数:
        expression: 要转换和替换的表达式节点
        mapping: 表名映射字典
        dialect: 映射表的方言
        copy: 是否复制表达式

    示例:
        >>> from sqlglot import exp, parse_one
        >>> replace_tables(parse_one("select * from a.b"), {"a.b": "c"}).sql()
        'SELECT * FROM c /* a.b */'

    返回:
        映射后的表达式
    """
    # 标准化映射字典的键：确保映射键使用统一的标准化表名
    mapping = {normalize_table_name(k, dialect=dialect): v for k, v in mapping.items()}

    # 内部转换函数：处理单个节点的表替换逻辑
    def _replace_tables(node: Expression) -> Expression:
        # 检查是否为Table节点且允许替换（meta.replace不为False）
        if isinstance(node, Table) and node.meta.get("replace") is not False:
            # 获取当前表的标准化名称
            original = normalize_table_name(node, dialect=dialect)
            # 在映射中查找新名称
            new_name = mapping.get(original)

            if new_name:
                # 创建新的表对象，保留除TABLE_PARTS外的所有原始属性
                table = to_table(
                    new_name,
                    # 过滤掉表的核心部分（this、db、catalog等），保留其他属性如别名
                    **{k: v for k, v in node.args.items() if k not in TABLE_PARTS},
                    dialect=dialect,
                )
                # 添加注释记录原始表名，便于调试和追踪
                table.add_comments([original])
                return table
        # 不是Table节点或不需要替换，返回原节点
        return node

    # 应用转换函数到整个表达式树
    return expression.transform(_replace_tables, copy=copy)  # type: ignore


# 占位符替换函数：替换表达式中的占位符
def replace_placeholders(
    expression: Expression,          # 要转换和替换的表达式节点
    *args,                          # 按顺序替换无名占位符的位置参数
    **kwargs                        # 替换命名占位符的关键字参数
) -> Expression:
    """替换表达式中的占位符

    这个函数处理SQL中的参数化查询，将占位符替换为实际值。
    支持两种占位符：无名占位符（?）和命名占位符（:name）。

    参数:
        expression: 要转换和替换的表达式节点
        args: 按给定顺序替换无名占位符的位置参数
        kwargs: 替换命名占位符的关键字参数

    示例:
        >>> from sqlglot import exp, parse_one
        >>> replace_placeholders(
        ...     parse_one("select * from :tbl where ? = ?"),
        ...     exp.to_identifier("str_col"), "b", tbl=exp.to_identifier("foo")
        ... ).sql()
        "SELECT * FROM foo WHERE str_col = 'b'"

    返回:
        映射后的表达式
    """
    # 内部转换函数：处理单个占位符节点的替换逻辑
    def _replace_placeholders(node: Expression, args, **kwargs) -> Expression:
        if isinstance(node, Placeholder):
            # 处理命名占位符（如:name）
            if node.this:
                new_name = kwargs.get(node.this)
                if new_name is not None:
                    # 将替换值转换为适当的表达式
                    return convert(new_name)
            else:
                # 处理无名占位符（如?）
                try:
                    # 从args迭代器中获取下一个值
                    return convert(next(args))
                except StopIteration:
                    # 参数用完了，保持占位符不变
                    pass
        # 不是占位符或无法替换，返回原节点
        return node

    # 应用转换函数，将args转换为迭代器以支持按顺序消费
    return expression.transform(_replace_placeholders, iter(args), **kwargs)


# 源展开函数：将引用的源展开为子查询
def expand(
    expression: Expression,                                 # 要展开的表达式
    sources: t.Dict[str, Query | t.Callable[[], Query]],   # 名称到查询的字典或提供查询的可调用对象
    dialect: DialectType = None,                           # sources字典或可调用对象的方言
    copy: bool = True,                                     # 转换期间是否复制表达式
) -> Expression:
    """通过将所有引用的源展开为子查询来转换表达式

    这个函数实现了查询展开功能，将表引用替换为对应的子查询。
    支持递归展开和惰性求值（通过可调用对象）。

    示例:
        >>> from sqlglot import parse_one
        >>> expand(parse_one("select * from x AS z"), {"x": parse_one("select * from y")}).sql()
        'SELECT * FROM (SELECT * FROM y) AS z /* source: x */'

        >>> expand(parse_one("select * from x AS z"), {"x": parse_one("select * from y"), "y": parse_one("select * from z")}).sql()
        'SELECT * FROM (SELECT * FROM (SELECT * FROM z) AS y /* source: y */) AS z /* source: x */'

    参数:
        expression: 要展开的表达式
        sources: 名称到查询的字典或按需提供查询的可调用对象
        dialect: sources字典或可调用对象的方言
        copy: 转换期间是否复制表达式。默认为True

    返回:
        转换后的表达式
    """
    # 标准化sources字典的键：确保查找时使用统一的表名格式
    normalized_sources = {normalize_table_name(k, dialect=dialect): v for k, v in sources.items()}

    # 内部展开函数：处理单个节点的展开逻辑
    def _expand(node: Expression):
        if isinstance(node, Table):
            # 获取表的标准化名称
            name = normalize_table_name(node, dialect=dialect)
            # 在sources中查找对应的源
            source = normalized_sources.get(name)

            if source:
                # 获取源查询：如果是可调用对象则调用，否则直接使用
                parsed_source = source() if callable(source) else source
                # 创建子查询，使用相同的别名（或表名如果没有别名）
                subquery = parsed_source.subquery(node.alias or name)
                # 添加注释标记源信息，便于调试和追踪
                subquery.comments = [f"source: {name}"]

                # 在子查询内部继续展开：支持递归展开，copy=False避免不必要的复制
                return subquery.transform(_expand, copy=False)

        # 不是Table节点或没有对应源，返回原节点
        return node

    # 应用展开转换到整个表达式树
    return expression.transform(_expand, copy=copy)


def func(name: str, *args, copy: bool = True, dialect: DialectType = None, **kwargs) -> Func:
    """
    Returns a Func expression.

    Examples:
        >>> func("abs", 5).sql()
        'ABS(5)'

        >>> func("cast", this=5, to=DataType.build("DOUBLE")).sql()
        'CAST(5 AS DOUBLE)'

    Args:
        name: the name of the function to build.
        args: the args used to instantiate the function of interest.
        copy: whether to copy the argument expressions.
        dialect: the source dialect.
        kwargs: the kwargs used to instantiate the function of interest.

    Note:
        The arguments `args` and `kwargs` are mutually exclusive.

    Returns:
        An instance of the function of interest, or an anonymous function, if `name` doesn't
        correspond to an existing `sqlglot.expressions.Func` class.
    """
    """
    返回一个Func表达式。
    
    这是一个工厂函数，用于动态创建SQL函数表达式。
    根据函数名称和参数，自动选择合适的函数类进行实例化。
    
    Examples:
        >>> func("abs", 5).sql()
        'ABS(5)'

        >>> func("cast", this=5, to=DataType.build("DOUBLE")).sql()
        'CAST(5 AS DOUBLE)'

    Args:
        name: 要构建的函数名称
        args: 用于实例化目标函数的位置参数
        copy: 是否复制参数表达式
        dialect: 源方言类型
        kwargs: 用于实例化目标函数的关键字参数

    Note:
        参数 `args` 和 `kwargs` 是互斥的，不能同时使用

    Returns:
        目标函数的实例，如果名称不对应现有的函数类，则返回匿名函数
    """
    # 检查参数使用方式：args和kwargs不能同时使用
    if args and kwargs:
        raise ValueError("Can't use both args and kwargs to instantiate a function.")

    from sqlglot.dialects.dialect import Dialect

    # 获取方言对象，如果方言无效则抛出异常
    dialect = Dialect.get_or_raise(dialect)
    
    # 将位置参数转换为Expression对象，支持SQL字符串解析
    # 如果copy=True，会创建参数的副本以避免修改原始对象
    converted: t.List[Expression] = [maybe_parse(arg, dialect=dialect, copy=copy) for arg in args]
    
    # 将关键字参数也转换为Expression对象
    kwargs = {key: maybe_parse(value, dialect=dialect, copy=copy) for key, value in kwargs.items()}


    # 首先尝试从方言特定的函数映射中获取构造函数
    # 方言可能有自己的函数实现，优先级更高
    constructor = dialect.parser_class.FUNCTIONS.get(name.upper())
    if constructor:
        # 如果找到了方言特定的函数构造函数
        if converted:
            # 有位置参数的情况
            if "dialect" in constructor.__code__.co_varnames:
                # 如果构造函数接受dialect参数，则传入
                function = constructor(converted, dialect=dialect)
            else:
                # 否则只传入位置参数
                function = constructor(converted)
        elif constructor.__name__ == "from_arg_list":
            # 特殊处理：如果构造函数是from_arg_list方法
            # 则调用其所属类的实例方法，传入关键字参数
            function = constructor.__self__(**kwargs)  # type: ignore
        else:
            # 如果没有位置参数，尝试从全局函数映射中查找
            constructor = FUNCTION_BY_NAME.get(name.upper())
            if constructor:
                # 使用关键字参数创建函数实例
                function = constructor(**kwargs)
            else:
                # 如果都找不到，抛出错误
                raise ValueError(
                    f"Unable to convert '{name}' into a Func. Either manually construct "
                    "the Func expression of interest or parse the function call."
                )
    else:
        # 如果没有找到方言特定的函数，创建匿名函数
        # 如果没有关键字参数，使用位置参数作为expressions
        kwargs = kwargs or {"expressions": converted}
        function = Anonymous(this=name, **kwargs)

    # 验证函数参数的有效性，如果发现错误则抛出异常
    for error_message in function.error_messages(converted):
        raise ValueError(error_message)

    return function


def case(
    expression: t.Optional[ExpOrStr] = None,  # 可选的输入表达式（非所有方言都支持）
    **opts,                                   # 解析expression的额外关键字参数
) -> Case:
    """
    初始化CASE语句

    CASE语句是SQL中实现条件逻辑的核心结构，支持多重条件判断和默认值设置。
    可以创建简单CASE（基于表达式值）或搜索CASE（基于条件表达式）。

    示例:
        case().when("a = 1", "foo").else_("bar")

    参数:
        expression: 可选的输入表达式（不是所有方言都支持此特性）
        **opts: 解析`expression`的额外关键字参数
    """
    # 处理简单CASE的输入表达式：如果提供了expression，则解析它
    if expression is not None:
        # 简单CASE：CASE expression WHEN value1 THEN result1...
        this = maybe_parse(expression, **opts)
    else:
        # 搜索CASE：CASE WHEN condition1 THEN result1...
        this = None
    
    # 创建Case对象，初始化时ifs列表为空，后续通过when()方法添加条件
    return Case(this=this, ifs=[])


def array(
    *expressions: ExpOrStr,          # 可变数量的表达式参数
    copy: bool = True,               # 是否复制参数表达式
    dialect: DialectType = None,     # 源方言
    **kwargs                         # 用于实例化相关函数的其他参数
) -> Array:
    """
    返回数组表达式

    数组是现代SQL中的重要数据类型，用于存储同类型元素的有序集合。
    不同数据库对数组的语法支持有所差异。

    示例:
        >>> array(1, 'x').sql()
        'ARRAY(1, x)'

    参数:
        expressions: 要添加到数组中的表达式
        copy: 是否复制参数表达式
        dialect: 源方言
        kwargs: 用于实例化相关函数的关键字参数

    返回:
        数组表达式
    """
    return Array(
        # 将每个输入表达式解析并添加到数组中
        expressions=[
            maybe_parse(expression, copy=copy, dialect=dialect, **kwargs)
            for expression in expressions
        ]
    )


def tuple_(
    *expressions: ExpOrStr,          # 可变数量的表达式参数
    copy: bool = True,               # 是否复制参数表达式
    dialect: DialectType = None,     # 源方言
    **kwargs                         # 用于实例化相关函数的其他参数
) -> Tuple:
    """
    返回元组表达式

    元组在SQL中用于组合多个值，常用于IN子句、比较操作和函数参数传递。
    与数组不同，元组通常用括号表示，且可以包含不同类型的元素。

    示例:
        >>> tuple_(1, 'x').sql()
        '(1, x)'

    参数:
        expressions: 要添加到元组中的表达式
        copy: 是否复制参数表达式
        dialect: 源方言
        kwargs: 用于实例化相关函数的关键字参数

    返回:
        元组表达式
    """
    return Tuple(
        # 将每个输入表达式解析并添加到元组中
        expressions=[
            maybe_parse(expression, copy=copy, dialect=dialect, **kwargs)
            for expression in expressions
        ]
    )


def true() -> Boolean:
    """
    返回真值布尔表达式

    创建表示SQL中TRUE值的布尔表达式，用于条件判断、默认值设置等场景。
    """
    # 创建值为True的Boolean对象
    return Boolean(this=True)


def false() -> Boolean:
    """
    返回假值布尔表达式

    创建表示SQL中FALSE值的布尔表达式，用于条件判断、默认值设置等场景。
    """
    # 创建值为False的Boolean对象
    return Boolean(this=False)


def null() -> Null:
    """
    返回NULL表达式

    创建表示SQL中NULL值的表达式，NULL在SQL中表示未知或缺失的值，
    具有特殊的比较语义和处理规则。
    """
    # 创建Null对象，不需要参数因为NULL是唯一的
    return Null()


NONNULL_CONSTANTS = (
    Literal,
    Boolean,
)

CONSTANTS = (
    Literal,
    Boolean,
    Null,
)


class PartitionListProperty(Property):
    """表示分区列表属性"""
    arg_types = {"this": True, "partition_list": False}

class ToGroupProperty(Property):
    """Property for specifying table distribution to a specific node group"""
    arg_types = {"this": True}

class ToNodeProperty(Property):  
    """Property for specifying table distribution to specific nodes"""
    arg_types = {"this": True}