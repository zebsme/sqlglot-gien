import json
import re
import os
import sys
from utils import *
from hbase import parse_hbase_ddl
import hashlib
from sqlglot import parse_one,parse,exp
from sqlglot.expressions import *
from sqlglot.tokens import TokenType



# 添加安全配置
MAX_SQL_LENGTH = 1000000  # 最大SQL长度限制

# =============================================================================
# 分布键处理模块 - 统一管理所有分布键相关的逻辑
# =============================================================================
class DistributeKeyProcessor:
    """分布键处理器 - 统一管理分布键的提取、解析和设置"""
    
    # 类级别的缓存，避免重复解析相同的SQL
    _cache = {}

    @staticmethod
    def extract_distribute_info(sql, dialect):
        """
        统一的分布键信息提取函数，优先使用AST解析，回退到正则表达式

        Args:
            sql: SQL语句
            dialect: 数据库方言

        Returns:
            dict: 包含分布键信息的字典，格式为:
            {
                "distribute_flag": "DISTRIBUTE BY HASH" | "DISTRIBUTED BY",
                "fields": ["field1", "field2", ...]
            }
        """
        try:
            # 检查缓存
            cache_key = f"{dialect}:{hash(sql)}"
            if cache_key in DistributeKeyProcessor._cache:
                return DistributeKeyProcessor._cache[cache_key]
            
            result = None
            
            # 对于特定方言，优先使用正则表达式（更可靠）
            if dialect in ['dws', 'qianbasempp']:
                result = DistributeKeyProcessor._extract_with_regex(sql, dialect)
                if not result:
                    # 如果正则表达式失败，尝试AST解析
                    result = DistributeKeyProcessor._extract_from_ast(sql, dialect)
            else:
                # 对于其他方言，优先使用AST解析
                result = DistributeKeyProcessor._extract_from_ast(sql, dialect)
                if not result:
                    # 如果AST解析失败，回退到正则表达式
                    result = DistributeKeyProcessor._extract_with_regex(sql, dialect)
            
            # 缓存结果
            DistributeKeyProcessor._cache[cache_key] = result
            return result

        except Exception as e:
            logger.debug(f"分布键信息提取失败: {e}")
            return None



    @staticmethod
    def _is_distribute_token(token):
        """判断是否为分布键相关的token"""
        return (token.token_type == TokenType.DISTRIBUTE_BY or
                (hasattr(token, 'text') and token.text.upper() == 'DISTRIBUTE'))

    @staticmethod
    def _parse_distribute_tokens(tokens, start_index, dialect):
        """解析分布键相关的token序列"""
        distribute_info = {
            "distribute_flag": "DISTRIBUTE BY",
            "fields": []
        }

        j = start_index + 1
        in_parens = False
        paren_depth = 0
        found_hash = False

        while j < len(tokens):
            current_token = tokens[j]

            if current_token.token_type == TokenType.VAR:
                if current_token.text.upper() == 'HASH':
                    distribute_info["distribute_flag"] = "DISTRIBUTE BY HASH"
                    found_hash = True
                elif in_parens:
                    distribute_info["fields"].append(current_token.text)
            elif current_token.token_type == TokenType.L_PAREN:
                paren_depth += 1
                if paren_depth == 1:
                    in_parens = True
            elif current_token.token_type == TokenType.R_PAREN:
                paren_depth -= 1
                if paren_depth == 0:
                    break
            elif current_token.token_type in [TokenType.SEMICOLON, TokenType.BREAK]:
                break

            j += 1

        # 返回结果
        if distribute_info["fields"]:
            return distribute_info
        elif found_hash:
            return distribute_info

        return None

    @staticmethod
    def _extract_from_ast(sql, dialect):
        """
        从AST中提取分布键信息
        
        优先使用sqlglot的AST解析来提取分布键信息，支持：
        - DistributedByProperty节点
        - 各种分布键类型（HASH、RANGE、LIST等）
        - 字段名清理（去除引号等）
        
        Args:
            sql: SQL语句
            dialect: 数据库方言
            
        Returns:
            dict: 分布键信息字典，解析失败返回None
        """
        try:
            parsed = parse(sql, dialect=dialect)

            create_expr = None
            if isinstance(parsed, list):
                for expr in parsed:
                    if isinstance(expr, exp.Create):
                        create_expr = expr
                        break
            elif isinstance(parsed, exp.Create):
                create_expr = parsed

            if create_expr:
                # 查找分布键属性
                distribute_props = list(create_expr.find_all(exp.DistributedByProperty))
                if distribute_props:
                    prop = distribute_props[0]
                    # 提取字段名，清理引号
                    fields = []
                    for expr in prop.args.get('expressions', []):
                        if hasattr(expr, 'sql'):
                            field_name = expr.sql()
                            # 清理引号
                            field_name = field_name.strip('"`\'')
                            fields.append(field_name)
                        else:
                            fields.append(str(expr))
                    
                    # 确定分布键类型
                    kind = prop.args.get('kind', 'HASH')
                    if kind.upper() == 'HASH':
                        distribute_flag = "DISTRIBUTE BY HASH"
                    elif kind.upper() == 'RANGE':
                        distribute_flag = "DISTRIBUTE BY RANGE"
                    elif kind.upper() == 'LIST':
                        distribute_flag = "DISTRIBUTE BY LIST"
                    else:
                        distribute_flag = f"DISTRIBUTE BY {kind}"
                    
                    return {
                        "distribute_flag": distribute_flag,
                        "fields": fields
                    }
        except Exception as e:
            logger.debug(f"AST解析失败: {e}")

        return None

    @staticmethod
    def _extract_with_regex(sql, dialect):
        """
        使用正则表达式提取分布键信息（备用方法）
        
        当AST解析失败时，使用正则表达式作为备用方案。
        支持多种方言的分布键语法：
        - DWS: DISTRIBUTE BY HASH(field1, field2)
        - Greenplum: DISTRIBUTE BY HASH/RANGE/LIST/RANDOM
        - QianBaseMPP: DISTRIBUTED BY (field1, field2)
        
        Args:
            sql: SQL语句
            dialect: 数据库方言
            
        Returns:
            dict: 分布键信息字典，解析失败返回None
        """
        try:
            if dialect == 'dws':
                # DWS方言: DISTRIBUTE BY HASH(field1, field2)
                pattern = r'DISTRIBUTE\s+BY\s+HASH\s*\(([^)]+)\)'
                match = re.search(pattern, sql, re.IGNORECASE)
                if match:
                    fields = [f.strip() for f in match.group(1).split(',')]
                    return {
                        "distribute_flag": "DISTRIBUTE BY HASH",
                        "fields": fields
                    }
            elif dialect == 'greenplum':
                # Greenplum方言支持多种分布键语法
                patterns = [
                    # DISTRIBUTE BY HASH(field1, field2)
                    (r'DISTRIBUTE\s+BY\s+HASH\s*\(([^)]+)\)', "DISTRIBUTE BY HASH"),
                    # DISTRIBUTE BY (field1, field2) - 简单分布
                    (r'DISTRIBUTE\s+BY\s*\(([^)]+)\)', "DISTRIBUTE BY"),
                    # DISTRIBUTE BY RANDOM
                    (r'DISTRIBUTE\s+BY\s+RANDOM', "DISTRIBUTE BY RANDOM"),
                    # DISTRIBUTE BY RANGE(field1, field2)
                    (r'DISTRIBUTE\s+BY\s+RANGE\s*\(([^)]+)\)', "DISTRIBUTE BY RANGE"),
                    # DISTRIBUTE BY LIST(field1, field2)
                    (r'DISTRIBUTE\s+BY\s+LIST\s*\(([^)]+)\)', "DISTRIBUTE BY LIST"),
                ]

                for pattern, flag in patterns:
                    match = re.search(pattern, sql, re.IGNORECASE)
                    if match:
                        if flag in ["DISTRIBUTE BY RANDOM"]:
                            # RANDOM分布没有字段
                            return {
                                "distribute_flag": flag,
                                "fields": []
                            }
                        else:
                            # 其他分布类型有字段
                            fields = [f.strip() for f in match.group(1).split(',')]
                            return {
                                "distribute_flag": flag,
                                "fields": fields
                            }
            elif dialect == 'qianbasempp':
                # 易鲸捷方言: DISTRIBUTED BY (field1, field2)
                pattern = r'DISTRIBUTED\s+BY\s*\(([^)]+)\)'
                match = re.search(pattern, sql, re.IGNORECASE)
                if match:
                    fields = [f.strip() for f in match.group(1).split(',')]
                    return {
                        "distribute_flag": "DISTRIBUTED BY",
                        "fields": fields
                    }
        except Exception as e:
            logger.debug(f"正则表达式提取失败: {e}")

        return None

    @staticmethod
    def update_table_distribute_keys(table_data, distribute_info):
        """
        更新表的分布键信息

        Args:
            table_data: 表数据字典
            distribute_info: 分布键信息

        Returns:
            dict: 更新后的表数据
        """
        if not distribute_info or not distribute_info.get('fields'):
            return table_data

        distribute_fields = distribute_info['fields']

        # 更新列信息中的分布键标志
        for column in table_data.get('tableColInfoList', []):
            if column.get('fieldName') in distribute_fields:
                column['distributeKey'] = True

        return table_data

    @staticmethod
    def get_distribute_fields_for_table(table_name, metadata):
        """
        从元数据中获取指定表的分布键字段

        Args:
            table_name: 表名
            metadata: 元数据字典

        Returns:
            list: 分布键字段列表
        """
        table_data = metadata.get('tables', {}).get(table_name, {})
        return table_data.get('distribute_fields', [])




# =============================================================================
# AST节点处理函数 - 处理各种SQL语句类型
# =============================================================================
def handle_alter_table(node, comments, primary_keys):
    """处理ALTER TABLE语句"""
    try:
        table_name = safe_get_table_name(node.this)
        if not table_name:
            return
        # 遍历所有ALTER操作
        for action in node.actions:
            # 处理主键约束
            for cons in getattr(action, 'expressions', []):
                if isinstance(cons, PrimaryKey):
                    # 提取主键列名
                    pk_columns = [col.this.this for col in cons.expressions]
                    primary_keys.setdefault(table_name, []).extend(pk_columns)

                elif isinstance(cons, Constraint):
                    # 处理约束中的主键定义
                    for expr in cons.args.get('expressions', []):
                        if isinstance(expr, PrimaryKey):
                            # 提取主键列名
                            pk_columns = [col.this.this for col in expr.expressions]
                            primary_keys.setdefault(table_name, []).extend(pk_columns)
            # 处理列定义（ClickHouse注释特殊处理）
            if isinstance(action, ColumnDef):
                col_name = action.args.get('this').name
                col_path = f"{table_name}.{col_name}"

                # 提取列注释
                for constraint in getattr(action, 'constraints', []):
                    if hasattr(constraint, 'kind'):
                        comments[col_path] = constraint.kind.name
    except Exception as e:
        logger.error(f"处理ALTER TABLE语句时出错: {str(e)}")


def handle_comment(node, comments):
    """处理COMMENT语句"""
    try:
        kind = node.args.get('kind')
        expression = node.expression.this if hasattr(node.expression, 'this') else None

        # 获取kind的实际值（处理Identifier类型）
        kind_value = None
        if isinstance(kind, str):
            kind_value = kind
        elif hasattr(kind, 'name'):
            kind_value = kind.name
        elif hasattr(kind, 'this'):
            kind_value = kind.this

        # 表注释
        if kind_value == 'TABLE':
            table_name = safe_get_table_name(node.this)
            if table_name and expression:
                comments[table_name] = expression

        # 列注释
        elif kind_value == 'COLUMN':
            # 处理不同方言的列路径表示方式
            column_ref = node.this
            if isinstance(column_ref, Column):
                # 格式: schema.table.column 或 table.column
                table_name = column_ref.table
                col_name = column_ref.name
                if table_name and col_name and expression:
                    # 处理带schema的情况
                    if hasattr(table_name, 'this'):
                        table_name = table_name.this
                    if hasattr(col_name, 'this'):
                        col_name = col_name.this
                    comments[f"{table_name}.{col_name}"] = expression

            elif isinstance(column_ref, Identifier):
                # 格式: table.column (某些方言)
                parts = column_ref.parts
                if len(parts) >= 2 and expression:
                    # 处理 schema.table.column 格式
                    if len(parts) == 3:
                        schema_name = parts[0]
                        table_name = parts[1]
                        col_name = parts[2]
                        comments[f"{table_name}.{col_name}"] = expression
                    elif len(parts) == 2:
                        table_name = parts[0]
                        col_name = parts[1]
                        comments[f"{table_name}.{col_name}"] = expression

        # 索引注释
        elif kind_value == 'INDEX':
            # 处理索引注释
            index_ref = node.this

            if isinstance(index_ref, Identifier):
                # 处理带schema的索引名，如 ltytest.v1
                if hasattr(index_ref, 'this'):
                    full_name = index_ref.this
                else:
                    full_name = index_ref.name

                if full_name and expression:
                    # 提取索引名部分（去掉schema前缀）
                    if '.' in full_name:
                        # 如果包含schema，只取索引名部分
                        index_name = full_name.split('.')[-1]
                    else:
                        index_name = full_name
                    
                    comments[index_name] = expression

    except Exception as e:
        logger.error(f"处理COMMENT语句时出错: {str(e)}")


# =============================================================================
# 工具函数 - 安全提取表结构信息
# =============================================================================
def _extract_table_sql_from_original(original_sql, table_name, schema, dialect, pre_dialect, node):
    """
    从原始SQL中提取指定表的SQL片段
    
    统一处理表名模式匹配和SQL片段提取，避免重复的正则表达式逻辑
    
    Args:
        original_sql: 原始SQL语句
        table_name: 表名
        schema: schema名
        dialect: 目标方言
        pre_dialect: 源数据库方言
        node: CREATE TABLE的AST节点
        
    Returns:
        str: 提取的表SQL片段，如果未找到返回None
    """
    if not original_sql:
        # 如果没有原始SQL，使用重新生成的SQL作为备选
        return node.sql(dialect=dialect)
    
    # 构建表名搜索模式
    table_name_patterns = [table_name, f"`{table_name}`", f'"{table_name}"']
    if schema:
        table_name_patterns.extend([
            f"{schema}.{table_name}",
            f"`{schema}`.`{table_name}`",
            f'"{schema}"."{table_name}"'
        ])

    # 在原始SQL中查找该表对应的SQL片段
    for pattern in table_name_patterns:
        table_sql_match = re.search(
            rf'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?{re.escape(pattern)}.*?(?=CREATE\s+TABLE|COMMENT\s+ON|$)',
            original_sql,
            re.IGNORECASE | re.DOTALL
        )
        
        if table_sql_match:
            return table_sql_match.group(0)
    
    # 如果未找到匹配的表SQL，使用重新生成的SQL作为备选
    return node.sql(dialect=dialect)

def _handle_partition_columns(prop, table_name, tables, comments, dialect):
    """
    处理分区列定义
    
    从表属性中提取分区列信息，避免与handle_table_property中的逻辑重复
    
    Args:
        prop: 表属性对象
        table_name: 表名
        tables: 表信息字典
        comments: 注释信息字典
        dialect: 目标方言
    """
    if not (hasattr(prop, "this") and getattr(prop, "this", None) is not None):
        return
    
    existing_col_names = {col['name'] for col in tables[table_name]['cols']}
    
    for part_col in getattr(prop.this, "expressions", []):
        try:
            # 提取列名
            if isinstance(part_col, Identifier):
                col_name = part_col.name
            elif hasattr(part_col, "this") and hasattr(part_col.this, "this"):
                col_name = part_col.this.this
            else:
                col_name = str(part_col)
            
            # 跳过重复字段
            if col_name in existing_col_names:
                continue
            
            # 提取数据类型
            data_type = part_col.args['kind'].sql(dialect=dialect)
            
            # 提取长度和精度
            length, scale = None, None
            if part_col.kind.expressions:
                try:
                    length = part_col.kind.expressions[0].this.this if len(part_col.kind.expressions) > 0 else None
                    scale = part_col.kind.expressions[1].this.this if len(part_col.kind.expressions) > 1 else None
                except:
                    pass
            
            # 提取列注释
            col_comment = ''
            for cons in getattr(part_col, 'constraints', []):
                if isinstance(cons.kind, CommentColumnConstraint):
                    col_comment = cons.kind.this.this
            
            # 添加列信息
            tables[table_name]['cols'].append({
                'name': col_name,
                'type': data_type,
                'length': length,
                'scale': scale,
                'nullable': False,
                'defaultValue': None,
            })
            comments[f'{table_name}.{col_name}'] = col_comment
            
        except Exception as e:
            logger.error(f"处理分区列时出错: {str(e)}")
            continue
def safe_get_table_name(table_expr):
    """
    安全获取表名，支持多种表名表示方式
    
    处理不同方言中表名的各种表示方式：
    - 简单表名：table_name
    - 带schema：schema.table_name
    - 带catalog：catalog.schema.table_name
    - 深层嵌套结构
    
    Args:
        table_expr: 表名表达式AST节点
        
    Returns:
        str: 提取的表名，如果提取失败返回None
    """
    try:
        if table_expr is None:
            return None

        # 处理简单标识符: table_name
        if isinstance(table_expr, Identifier):
            return table_expr.name

        # 处理Schema类型: schema.table_name
        if hasattr(table_expr, 'this') and hasattr(table_expr.this, 'name'):
            return table_expr.this.name

        # 处理带schema的表名: schema.table_name
        if (hasattr(table_expr, 'this') and
                isinstance(table_expr.this, Identifier)):
            return table_expr.this.name

        # 处理多层表名: catalog.schema.table_name
        if (hasattr(table_expr, 'this') and
                hasattr(table_expr.this, 'this') and
                isinstance(table_expr.this.this, Identifier)):
            return table_expr.this.this.name

        # 处理某些方言的特殊表名结构
        if (hasattr(table_expr, 'this') and
                hasattr(table_expr.this, 'this') and
                hasattr(table_expr.this.this, 'this') and
                isinstance(table_expr.this.this.this, Identifier)):
            return table_expr.this.this.this.name

        return None
    except Exception as e:
        logger.error(f"提取表名时出错: {str(e)}")
        return None

def safe_get_schema_name(table_expr):
    """
    安全获取schema名，支持多种表名表示方式
    
    从表名表达式中提取schema信息，支持：
    - schema.table_name
    - catalog.schema.table_name
    - 深层嵌套结构
    
    Args:
        table_expr: 表名表达式AST节点
        
    Returns:
        str: 提取的schema名，如果没有schema返回None
    """
    try:
        if table_expr is None:
            return None

        # 处理简单标识符: table_name
        if isinstance(table_expr, Identifier):
            return None

        # 处理Schema类型: schema.table_name
        if hasattr(table_expr, 'this') and hasattr(table_expr.this, 'db'):
            return table_expr.this.db

        # 处理带schema的表名: schema.table_name
        if (hasattr(table_expr, 'db') and table_expr.db):
            return table_expr.db

        # 递归查找更深层级
        if hasattr(table_expr, 'this'):
            return safe_get_schema_name(table_expr.this)

        return None
    except Exception as e:
        logger.error(f"提取schema名时出错: {str(e)}")
        return None

def safe_get_catalog_name(table_expr):
    """
    安全获取catalog名，支持多种表名表示方式
    
    从表名表达式中提取catalog信息，支持：
    - catalog.schema.table_name
    - 深层嵌套结构
    
    Args:
        table_expr: 表名表达式AST节点
        
    Returns:
        str: 提取的catalog名，如果没有catalog返回None
    """
    try:
        if table_expr is None:
            return None

        # 处理简单标识符: table_name
        if isinstance(table_expr, Identifier):
            return None

        # 处理catalog.schema.table_name
        # 递归查找最外层的catalog
        if hasattr(table_expr, 'catalog') and table_expr.catalog:
            return table_expr.catalog
        # 兼容部分AST结构
        if hasattr(table_expr, 'this'):
            return safe_get_catalog_name(table_expr.this)
        return None
    except Exception as e:
        logger.error(f"提取catalog名时出错: {str(e)}")
        return None

# =============================================================================
# 特殊语法处理函数 - 处理各种方言的特殊语法
# =============================================================================
def extract_create_addition_options(sql, field_attributes=None):
    """
    支持OPTIONS、WITH、TBLPROPERTIES三种属性块，LOCATION行可在前或后，保留原始格式
    根据字段中已定义的属性过滤掉相应的选项
    当同时存在OPTIONS和TBLPROPERTIES时，TBLPROPERTIES在前面
    
    注意：此函数使用正则表达式进行复杂的文本模式匹配，
    因为需要处理各种方言的特殊语法和嵌套结构，难以用AST完全替代。

    Args:
        sql: SQL语句
        field_attributes: 字段属性字典，包含primaryKey、partitionKey、distributeKey等
    """
    # 若存在 "USING hudi"，优先从其后开始抽取
    using_hudi = re.search(r'USING\s+hudi\b', sql, re.IGNORECASE)
    if using_hudi:
        tail = sql[using_hudi.end():]

        # 匹配所有属性块和LOCATION行
        prop_pattern = re.compile(r'(TBLPROPERTIES|OPTIONS|WITH)\s*\((?:[^()]*|\([^()]*\))*\)\s*;?',
                                  re.IGNORECASE | re.DOTALL)
        loc_pattern = re.compile(r'([\t ]*--\s*location\b[^\n]*|[\t ]*LOCATION\b[^\n;]*;?)', re.IGNORECASE)

        # 找到所有属性块
        prop_matches = list(prop_pattern.finditer(tail))
        loc_matches = list(loc_pattern.finditer(tail))

        # 分离TBLPROPERTIES和其他属性块
        tblproperties_segments = []
        other_segments = []
        
        for match in prop_matches:
            segment = match.group(0)
            prop_type = match.group(1).upper()
            
            # 删除分区定义块
            segment = re.sub(r'PARTITIONED\s+BY\s*\([^)]*\)\s*', '', segment, flags=re.IGNORECASE | re.DOTALL)
            # 只删除描述项和primaryKey，保留preCombineField
            segment = re.sub(r"\n?\s*['\"]?hoodie\.table\.description['\"]?\s*=\s*['\"][^'\"]*['\"]\s*,?\s*\n?", '\n',
                             segment, flags=re.IGNORECASE)
            # 删除 primaryKey 项
            segment = re.sub(r"\n?\s*['\"]?primaryKey['\"]?\s*=\s*['\"][^'\"]*['\"]\s*,?\s*\n?", '\n', segment,
                             flags=re.IGNORECASE)
            # 删除独立表级注释行
            segment = re.sub(r"^\s*comment\s*['\"][^'\"]*['\"]\s*$", '', segment, flags=re.IGNORECASE | re.MULTILINE)
            # 规范逗号
            segment = re.sub(r',\s*,', ',', segment)
            segment = re.sub(r'\n\s*,', '\n', segment)
            segment = re.sub(r',\s*\)', ')', segment)
            # 去掉末尾分号
            segment = re.sub(r';\s*$', '', segment)
            
            if segment.strip():
                if prop_type == 'TBLPROPERTIES':
                    tblproperties_segments.append(segment)
                else:
                    other_segments.append(segment)

        # 构建结果：TBLPROPERTIES在前，其他在后
        segments = tblproperties_segments + other_segments

        # 添加LOCATION行
        for match in loc_matches:
            location_line = match.group(1) if match.lastindex else match.group(0)
            if location_line.strip():
                segments.append(location_line)

        if segments:
            return '\n'.join(segments)
        return None

    # 回退逻辑：处理多个属性块的情况
    prop_pattern = re.compile(
        r'(TBLPROPERTIES|OPTIONS|WITH)\s*\((?:[^()]*|\([^()]*\))*\)\s*;?', re.DOTALL | re.IGNORECASE
    )
    location_line_pattern = re.compile(r'(?:^|\s)(--\s*location\b[^\n]*|LOCATION\b[^\n;]*;?)', re.IGNORECASE)

    # 找到所有属性块匹配
    prop_matches = list(prop_pattern.finditer(sql))
    location_match = location_line_pattern.search(sql)

    if prop_matches:
        # 分离TBLPROPERTIES和其他属性块
        tblproperties_segments = []
        other_segments = []
        
        for match in prop_matches:
            segment = match.group(0)
            prop_type = match.group(1).upper()
            
            # 清理segment
            # 删除 PARTITIONED BY 块
            segment = re.sub(r'PARTITIONED\s+BY\s*\([^)]*\)\s*', '', segment, flags=re.IGNORECASE | re.DOTALL)
            # 只删除描述项和primaryKey，保留preCombineField
            segment = re.sub(r"\n?\s*['\"]?hoodie\.table\.description['\"]?\s*=\s*['\"][^'\"]*['\"]\s*,?\s*\n?", '\n',
                            segment, flags=re.IGNORECASE)
            # 删除 primaryKey 项
            segment = re.sub(r"\n?\s*['\"]?primaryKey['\"]?\s*=\s*['\"][^'\"]*['\"]\s*,?\s*\n?", '\n', segment,
                            flags=re.IGNORECASE)
            # 删除独立表级注释行
            segment = re.sub(r"^\s*comment\s*['\"][^'\"]*['\"]\s*$", '', segment, flags=re.IGNORECASE | re.MULTILINE)
            # 清理属性块中可能多余的逗号
            segment = re.sub(r',\s*,', ',', segment)
            segment = re.sub(r'\n\s*,', '\n', segment)
            segment = re.sub(r',\s*\)', ')', segment)
            # 去掉末尾分号
            segment = re.sub(r';\s*$', '', segment)
            
            if segment.strip():
                if prop_type == 'TBLPROPERTIES':
                    tblproperties_segments.append(segment)
                else:
                    other_segments.append(segment)

        # 构建结果：TBLPROPERTIES在前，其他在后
        segments = tblproperties_segments + other_segments
        
        # 添加LOCATION行
        if location_match:
            location_line = location_match.group(1) if location_match.lastindex else location_match.group(0)
            if location_line.strip():
                segments.append(location_line)

        if segments:
            return '\n'.join(segments)

    # 如果只有LOCATION行
    if location_match:
        location_line = location_match.group(1) if location_match.lastindex else location_match.group(0)
        return location_line

    return None


# =============================================================================
# 主要解析函数 - 核心DDL解析逻辑
# =============================================================================
def parse_ast(sql_data, pre_dialect, dialect):
    """
    解析SQL DDL语句并提取表结构信息
    
    这是主要的解析函数，负责：
    1. 验证输入SQL的有效性
    2. 预处理SQL语句
    3. 使用sqlglot解析AST
    4. 收集表结构元数据
    5. 构建最终结果
    
    Args:
        sql_data: 原始SQL DDL语句
        pre_dialect: 源数据库方言标识
        dialect: 目标方言标识
        
    Returns:
        list: 包含表结构信息的列表，每个元素包含：
            - 表基本信息（名称、schema、catalog等）
            - 列信息（字段名、类型、约束等）
            - 索引信息
            - 分区信息
            - 分布键信息
            - 注释信息
    """
    # 输入验证
    if not sql_data or len(sql_data) > MAX_SQL_LENGTH:
        logger.error("输入SQL为空或过长")
        return {"error": "输入SQL无效"}
    logger.info(f"开始解析SQL, 源数据库: {pre_dialect}, 目标方言: {dialect}")

    try:
        # 预处理SQL语句
        processed_sql = preprocess(sql_data, pre_dialect, dialect)

        # HBase特殊处理
        if dialect == 'hbase':
            return parse_hbase_ddl(processed_sql, dialect)

        # 使用sqlglot解析AST
        ast = parse(processed_sql, dialect=dialect)
        logger.info('SQL解析成功')

        # 从AST收集元数据
        metadata = collect_metadata(ast, dialect, pre_dialect, original_sql=sql_data)

        # 收集字段属性信息，用于后续处理
        field_attributes = {}
        if metadata and 'tables' in metadata:
            for table_name, table_data in metadata['tables'].items():
                has_primary_key = any(metadata['primary_keys'].get(table_name, []))
                has_partition_key = any(metadata['partitions'].get(table_name, []))
                has_distribute_key = bool(table_data.get('distribute_fields', []))

                field_attributes[table_name] = {
                    'primaryKey': has_primary_key,
                    'partitionKey': has_partition_key,
                    'distributeKey': has_distribute_key
                }

        # 构建最终结果
        result = build_result(metadata, pre_dialect,  original_sql=sql_data)

        # 安全日志输出
        if type(result) == list:
            logger.info(f"解析完成，共发现{len(result)}张表")
            return result

    except ParseError as e:
        logger.error(f"SQL解析错误: {str(e)}")
        return {"error": f"SQL解析失败: {str(e)}"}
    except Exception as e:
        logger.exception("解析过程中发生未预期错误")
        return {"error": f"系统错误: {str(e)}"}


def collect_metadata(ast_nodes, dialect, pre_dialect, original_sql=None):
    """
    从AST节点收集表结构元数据
    
    遍历AST中的所有节点，提取以下信息：
    - 表定义信息（列、约束、属性等）
    - 注释信息（表注释、列注释、索引注释）
    - 主键信息
    - 分区信息
    - 索引信息
    - 分布键信息
    
    Args:
        ast_nodes: 解析后的AST节点列表
        dialect: 目标方言
        pre_dialect: 源数据库方言
        original_sql: 原始SQL语句（用于提取特殊语法）
        
    Returns:
        dict: 包含所有元数据的字典
    """
    tables = {}
    comments = {}
    primary_keys = {}
    partitions = {}
    indexes = {}  # 收集索引信息
    partition_details = {}  # 收集分区详细信息
    distribute_fields = {}

    for node in ast_nodes:
        try:
            # 处理CREATE TABLE语句
            if isinstance(node, Create) and node.args.get('kind') == 'TABLE':
                handle_create_table(node, tables, comments, primary_keys, partitions, distribute_fields, dialect, pre_dialect, original_sql, partition_details)

            # 处理CREATE INDEX语句（目前主要支持DWS方言）
            elif isinstance(node, Create) and node.args.get('kind') == 'INDEX' and dialect == 'dws':
                handle_create_index(node, indexes, dialect)

            # 处理ALTER TABLE语句
            elif isinstance(node, Alter) and node.args.get('kind') == 'TABLE':
                handle_alter_table(node, comments, primary_keys)

            # 处理COMMENT语句
            elif isinstance(node, Comment):
                handle_comment(node, comments)

        except Exception as e:
            logger.error(f"处理AST节点时出错: {type(node).__name__}, 错误: {str(e)}")
            continue

    return {
        "tables": tables,
        "comments": comments,
        "primary_keys": primary_keys,
        "partitions": partitions,
        "distribute_fields": distribute_fields,  # 返回分布键信息
        "indexes": indexes,  # 返回索引信息
        "partition_details": partition_details  # 返回分区详细信息
    }



def handle_constraint(expr, table_name, primary_keys):
    """
    处理约束表达式中的主键定义

    参数:
        expr: 约束表达式对象 (Constraint)
        table_name: 当前表名
        primary_keys: 主键字典（将被更新）
    """
    try:
        # 确保我们有表达式列表
        if 'expressions' not in expr.args or not expr.args['expressions']:
            return

        # 遍历约束中的所有表达式
        for exp in expr.args['expressions']:
            # 只处理主键约束
            if isinstance(exp, PrimaryKey):
                # 提取主键列名
                pk_columns = []
                expressions = exp.args.get('expressions', [])

                # 安全处理各种主键列表示方式
                for i in range(len(expressions)):
                    col_expr = expressions[i]

                    # 处理不同列表示方式
                    if hasattr(col_expr, 'this') and hasattr(col_expr.this, 'this') and hasattr(col_expr.this.this,
                                                                                                'this'):
                        # 处理深层嵌套的列名 (e.g., col_expr.this.this.this)
                        pk_columns.append(col_expr.this.this.this)
                    elif hasattr(col_expr, 'this') and hasattr(col_expr.this, 'name'):
                        # 处理标准列表示 (e.g., Column)
                        pk_columns.append(col_expr.this.name)
                    elif hasattr(col_expr, 'name'):
                        # 处理简单标识符
                        pk_columns.append(col_expr.name)
                    else:
                        # 尝试直接获取SQL表示
                        try:
                            col_name = col_expr.sql().split('.')[-1].strip('"[]`')
                            pk_columns.append(col_name)
                        except:
                            logger.warning(f"无法解析主键列表达式: {type(col_expr)}")

                # 更新主键字典
                if pk_columns:
                    if table_name not in primary_keys:
                        primary_keys[table_name] = pk_columns
                    else:
                        # 合并已存在的主键
                        primary_keys[table_name].extend(pk_columns)
                # 只处理第一个主键约束（表通常只有一个主键）
                # break

    except Exception as e:
        logger.error(f"处理约束时出错: {str(e)}")

def handle_create_table(node, tables, comments, primary_keys, partitions, distribute_fields, dialect, pre_dialect, original_sql=None, partition_details=None):
    """
    处理CREATE TABLE语句，提取表结构信息
    
    从CREATE TABLE AST节点中提取：
    - 表基本信息（名称、schema、catalog等）
    - 列定义（字段名、类型、约束、默认值等）
    - 表属性（分区、分布键、注释等）
    - 特殊语法处理（如Hudi-Spark的OPTIONS、DWS的WITH (ORIENTATION=COLUMN)-- 列式存储等）
    
    Args:
        node: CREATE TABLE的AST节点
        tables: 表信息字典（将被更新）
        comments: 注释信息字典（将被更新）
        primary_keys: 主键信息字典（将被更新）
        partitions: 分区信息字典（将被更新）
        distribute: 分布键信息字典（将被更新）
        dialect: 目标方言
        pre_dialect: 源数据库方言
        original_sql: 原始SQL语句（用于提取特殊语法）
        partition_details: 分区详细信息字典（将被更新）
    """
    table_name = safe_get_table_name(node.this)
    schema = safe_get_schema_name(node.this)
    if not schema and table_name and '.' in table_name:
        schema, table_name = table_name.split('.', 1)
    if not schema:
        schema = None
    if not table_name:
        return
    catalog = safe_get_catalog_name(node.this)
    is_temp = getattr(node, 'exists', False)
    tables[table_name] = {
        'cols': [],
        'props': {},
        'is_temp': is_temp,
        'catalog': catalog,
        'schema': schema,
        'addition_options': None,  # 存储该表的WITH语句
        'distribute_fields': []    # 存储该表的分布键字段
    }

    # 处理列定义
    for expr in node.this.expressions:
        try:
            if isinstance(expr, ColumnDef):
                handle_column_definition(expr, table_name, tables, comments, primary_keys, dialect)
            elif isinstance(expr, PrimaryKey):
                primary_keys[table_name] = [col.name for col in expr.args.get('expressions')]
            elif isinstance(expr, Constraint):
                handle_constraint(expr, table_name, primary_keys)
        except Exception as e:
            logger.error(f"处理表结构时出错: {str(e)}")

    # 统一处理表属性和特殊信息提取
    try:
        # 处理表属性（AST解析）
        if node.args.get('properties'):
            for prop in node.args.get('properties').expressions:
                handle_table_property(prop, table_name, comments, partitions, primary_keys, distribute_fields, tables)
                # 处理分区列定义
                _handle_partition_columns(prop, table_name, tables, comments, dialect)

        # 从原始SQL中提取表片段，用于特殊语法处理
        table_sql = _extract_table_sql_from_original(original_sql, table_name, schema, dialect, pre_dialect, node)
        if 'Hudi_Spark' in pre_dialect or 'DWS' in pre_dialect:
            tables[table_name]['addition_options'] = extract_create_addition_options(table_sql)

    except Exception as e:
        logger.error(f"处理表属性和特殊信息时出错: {str(e)}")
        if partition_details is not None:
            partition_details[table_name] = {}


def handle_create_index(node, indexes, dialect):
    """
    处理CREATE INDEX语句，提取索引信息
    
    支持dws方言的索引解析，提取以下信息：
    - 索引名称
    - 索引类型和方法
    - 索引字段列表
    - 索引描述
    
    Args:
        node: CREATE INDEX的AST节点
        indexes: 索引信息字典（将被更新）
        dialect: 数据库方言
    """
    try:
        # 检查方言支持
        if not _is_index_supported_dialect(dialect):
            logger.debug(f"方言 {dialect} 暂不支持索引解析")
            return

        # 提取索引基本信息
        index_name = _extract_index_name(node)
        table_name = _extract_table_name_from_index(node)
        index_fields = _extract_index_fields(node)
        
        # 验证必要信息
        if not table_name:
            logger.warning("无法提取表名，跳过索引处理")
            return
            
        if not index_fields:
            logger.warning(f"索引 {index_name} 没有字段信息，跳过处理")
            return

        # 提取索引类型和方法
        index_type, index_method = _extract_index_type_and_method(node, dialect)

        # 存储索引信息
        if table_name not in indexes:
            indexes[table_name] = []

        indexes[table_name].append({
            'indexName': index_name,
            'indexType': index_type,
            'indexFun': index_method,
            'indexFieldNames': index_fields,
            'pubExplain': ''  # 可通过COMMENT语句补充
        })

        logger.debug(f"成功处理索引: {index_name} on {table_name}")

    except Exception as e:
        logger.error(f"处理CREATE INDEX时出错: {str(e)}")

def _is_index_supported_dialect(dialect):
    """检查方言是否支持索引解析"""
    supported_dialects = ['dws', 'mysql', 'postgres', 'oracle', 'sqlite']
    return dialect in supported_dialects


def _extract_index_name(node):
    """提取索引名称"""
    try:
        if hasattr(node, 'this'):
            if hasattr(node.this, 'name'):
                return node.this.name
            elif hasattr(node.this, 'this'):
                return node.this.this
        return None
    except:
        return None


def _extract_table_name_from_index(node):
    """从索引节点中提取表名"""
    try:
        # 方法1：从properties中提取
        if hasattr(node, 'args') and 'properties' in node.args and node.args['properties']:
            properties = node.args['properties']
            if hasattr(properties, 'expressions'):
                for prop in properties.expressions:
                    if hasattr(prop, 'this') and prop.this == 'TABLE':
                        if hasattr(prop, 'args') and 'value' in prop.args:
                            table_expr = prop.args['value']
                            return safe_get_table_name(table_expr)
        
        # 方法2：兼容旧结构
        if hasattr(node, 'this') and hasattr(node.this, 'args'):
            if 'table' in node.this.args and hasattr(node.this.args['table'], 'args'):
                if 'this' in node.this.args['table'].args:
                    return node.this.args['table'].args['this'].name
        
        return None
    except:
        return None


def _extract_index_fields(node):
    """提取索引字段列表"""
    try:
        index_fields = []
        
        # 方法1：从properties中提取
        if hasattr(node, 'args') and 'properties' in node.args and node.args['properties']:
            properties = node.args['properties']
            if hasattr(properties, 'expressions'):
                for prop in properties.expressions:
                    if hasattr(prop, 'this') and prop.this == 'COLUMNS':
                        if hasattr(prop, 'args') and 'value' in prop.args and prop.args['value']:
                            col_array = prop.args['value']
                            if hasattr(col_array, 'expressions'):
                                for col_expr in col_array.expressions:
                                    field_name = _extract_field_name_from_expr(col_expr)
                                    if field_name:
                                        index_fields.append(field_name)
        
        # 方法2：兼容旧结构
        if not index_fields and hasattr(node, 'this') and hasattr(node.this, 'args'):
            if 'params' in node.this.args and hasattr(node.this.args['params'], 'args'):
                if 'columns' in node.this.args['params'].args:
                    for column in node.this.args['params'].args['columns']:
                        if hasattr(column, 'name'):
                            index_fields.append(column.name)
        
        return index_fields
    except:
        return []


def _extract_field_name_from_expr(col_expr):
    """从列表达式中提取字段名"""
    try:
        # 处理Ordered对象
        if hasattr(col_expr, 'this') and hasattr(col_expr.this, 'name'):
            return col_expr.this.name
        elif hasattr(col_expr, 'name'):
            return col_expr.name
        elif hasattr(col_expr, 'this') and hasattr(col_expr.this, 'this') and hasattr(col_expr.this.this, 'name'):
            return col_expr.this.this.name
        return None
    except:
        return None


def _extract_index_type_and_method(node, dialect):
    """提取索引类型和方法"""
    try:
        index_type = None  # 默认类型
        index_method = _get_default_index_method(dialect)  # 根据方言获取默认方法

        # 从SQL中提取USING子句
        index_sql = node.sql(dialect=dialect).upper()
        if "USING" in index_sql:
            using_match = re.search(r'USING\s+(\w+)', index_sql)
            if using_match:
                index_method = using_match.group(1)
        
        # 根据方言设置索引类型
        if dialect in ['mysql', 'postgres']:
            # 检查是否是唯一索引
            if "UNIQUE" in index_sql:
                index_type = "UNIQUE"
            elif "PRIMARY" in index_sql:
                index_type = "PRIMARY"
        
        return index_type, index_method
    except:
        return None, _get_default_index_method(dialect)


def _get_default_index_method(dialect):
    """根据方言获取默认索引方法"""
    default_methods = {        'dws': 'BTREE',
        'mysql': 'BTREE',
        'postgres': 'BTREE',
        'oracle': 'BTREE',
        'sqlite': 'BTREE'

    }
    return default_methods.get(dialect, 'BTREE')


def handle_column_definition(expr, table_name, tables, comments, primary_keys, dialect):
    """
    处理列定义，提取列的结构信息
    
    从列定义AST节点中提取：
    - 列名和数据类型
    - 长度和精度信息
    - 约束信息（主键、非空、默认值等）
    - 注释信息
    - 枚举值（ENUM类型）
    
    Args:
        expr: 列定义的AST节点
        table_name: 所属表名
        tables: 表信息字典（将被更新）
        comments: 注释信息字典（将被更新）
        primary_keys: 主键信息字典（将被更新）
        dialect: 目标方言
    """
    col_name = expr.this.this
    data_type = expr.args['kind'].sql(dialect=dialect)
    # 安全获取长度和精度
    length, scale = None, None
    if expr.kind.expressions:
        try:
            # 尝试不同的访问方式来获取长度和精度
            length_expr = expr.kind.expressions[0] if len(expr.kind.expressions) > 0 else None
            scale_expr = expr.kind.expressions[1] if len(expr.kind.expressions) > 1 else None
            
            # 处理长度
            if length_expr:
                if hasattr(length_expr, 'this') and hasattr(length_expr.this, 'this'):
                    length = length_expr.this.this
                elif hasattr(length_expr, 'this'):
                    length = length_expr.this
                else:
                    length = str(length_expr)
            else:
                length = None
                
            # 处理精度
            if scale_expr:
                if hasattr(scale_expr, 'this') and hasattr(scale_expr.this, 'this'):
                    scale = scale_expr.this.this
                elif hasattr(scale_expr, 'this'):
                    scale = scale_expr.this
                else:
                    scale = str(scale_expr)
            else:
                scale = None
        except Exception as e:
            length, scale = None, None
    
    # 特殊处理Oracle的FLOAT类型 - 从SQL中提取长度信息
    if data_type == "FLOAT" and length is None and dialect == "oracle":
        # 尝试从原始SQL中提取FLOAT的长度信息
        try:
            # 获取列的原始SQL
            col_sql = expr.sql(dialect=dialect)
            # 使用正则表达式提取FLOAT(数字)中的数字
            import re
            match = re.search(r'FLOAT\((\d+)\)', col_sql)
            if match:
                length = match.group(1)
        except Exception as e:
            pass

    # 特殊处理Oracle的INT(0) —— 期望将 length 识别为 0
    if data_type == "INT" and length is None and dialect == "oracle":
        try:
            col_sql = expr.sql(dialect=dialect).upper()
            import re
            match = re.search(r'INT\((\d+)\)', col_sql)
            if match:
                length = match.group(1)
            else:
                # 若生成 SQL 中已无参数（因解析阶段丢失），回退为 0
                length = "0"
        except Exception:
            length = "0"

    # 处理ENUM类型的枚举值
    enum_values = None
    if hasattr(expr.kind, 'this') and expr.kind.this == exp.DataType.Type.ENUM:
        if hasattr(expr.kind, 'expressions') and expr.kind.expressions:
            enum_values = []
            for enum_expr in expr.kind.expressions:
                if hasattr(enum_expr, 'this'):
                    enum_values.append(enum_expr.this)
                else:
                    enum_values.append(str(enum_expr))

    # 处理列约束
    is_nullable = True    # 是否允许null
    default_value = None  # 默认值
    for cons in getattr(expr, 'constraints', []):
        if isinstance(cons.kind, PrimaryKeyColumnConstraint):
            primary_keys.setdefault(table_name, []).append(col_name)
        if isinstance(cons.kind, CommentColumnConstraint):
            comments[f"{table_name}.{col_name}"] = cons.kind.this.this
        if isinstance(cons.kind, NotNullColumnConstraint):
            # 关键：判断是 NOT NULL 还是 NULL
            kind_sql = cons.kind.sql(dialect=dialect).upper()
            if kind_sql == 'NOT NULL':
                is_nullable = False
            elif kind_sql == 'NULL':
                is_nullable = True
        if isinstance(cons.kind, DefaultColumnConstraint):  # 新增
            # 兼容各种类型的默认值
            try:
                if hasattr(cons.kind, "this"):
                    if hasattr(cons.kind.this, "this"):
                        default_value = cons.kind.this.this
                    else:
                        default_value = cons.kind.this
                else:
                    default_value = None
            except Exception as e:
                default_value = None

    # 保存列信息
    col_info = {
        'name': col_name,
        'type': data_type,
        'length': length,
        'scale': scale,
        'nullable': is_nullable,
        'defaultValue': default_value,
        'enumValue': enum_values
    }

    tables[table_name]['cols'].append(col_info)


def handle_table_property(prop, table_name, comments, partitions, primary_keys, distribute_fields, tables=None):
    """处理表属性"""
    if prop is None:
        return
    if isinstance(prop, SchemaCommentProperty):
        comments[table_name] = getattr(getattr(prop, "this", None), "this", "")
    elif isinstance(prop, PartitionedByProperty):
        # 处理PartitionedByProperty，支持新的MySQL分区结构
        partition_expr = prop.args.get('this')
        if partition_expr and hasattr(partition_expr, 'expressions'):
            # 从Partition对象的expressions中提取分区字段
            partition_fields = []
            for col in partition_expr.expressions:
                if hasattr(col, 'name'):
                    partition_fields.append(col.name)
                elif hasattr(col, 'this'):
                    partition_fields.append(col.this)
                else:
                    partition_fields.append(str(col))
            partitions[table_name] = partition_fields
        else:
            # 兼容旧的结构
            partitions[table_name] = [getattr(col, "name", None) for col in getattr(getattr(prop, "this", None), "expressions", [])]
    elif isinstance(prop, PartitionListProperty):
        # 处理DWS的PartitionListProperty
        partition_expr = prop.args.get('this')
        if partition_expr and hasattr(partition_expr, 'expressions'):
            # 提取分区字段
            partition_fields = [getattr(col, 'name', None) for col in partition_expr.expressions]
            partitions[table_name] = partition_fields
    elif isinstance(prop, PartitionByListProperty):
        # 处理MySQL的PartitionByListProperty
        partition_exprs = prop.args.get('partition_expressions', [])
        if partition_exprs:
            # 从partition_expressions中提取分区字段
            partition_fields = []
            for col in partition_exprs:
                if hasattr(col, 'name'):
                    partition_fields.append(col.name)
                elif hasattr(col, 'this'):
                    partition_fields.append(col.this)
                else:
                    partition_fields.append(str(col))
            partitions[table_name] = partition_fields
    elif isinstance(prop, PartitionByRangeProperty):
        # 处理MySQL的PartitionByRangeProperty
        partition_exprs = prop.args.get('partition_expressions', [])
        if partition_exprs:
            # 从partition_expressions中提取分区字段
            partition_fields = []
            for col in partition_exprs:
                if hasattr(col, 'name'):
                    partition_fields.append(col.name)
                elif hasattr(col, 'this'):
                    partition_fields.append(col.this)
                else:
                    partition_fields.append(str(col))
            partitions[table_name] = partition_fields
    elif isinstance(prop, PrimaryKey):
        primary_keys[table_name] = [getattr(col, "name", None) for col in prop.args.get('expressions', [])]
    elif hasattr(prop, "this") and isinstance(prop.this, Literal) and getattr(prop.this,"name",None) == 'primaryKey':
        value = getattr(prop, "args", {}).get('value')
        if value and hasattr(value, "this"):
            primary_keys[table_name] = value.this.split(',')
    elif hasattr(prop, "this") and isinstance(prop.this, Var) and getattr(prop.this, "name", None) == 'primaryKey':
        value = getattr(prop, "args", {}).get('value')
        if value and hasattr(value, "this"):
            primary_keys[table_name] = value.this.split(',')
    elif isinstance(prop, DistributedByProperty):
        # 处理DistributedByProperty，提取分布键字段
        # print(f"DEBUG: 处理DistributedByProperty for {table_name}")
        if tables and table_name in tables:
            distribute_field_list = []
            for expr in prop.args.get('expressions', []):
                if hasattr(expr, 'sql'):
                    field_name = expr.sql()
                    # 去掉引号
                    if field_name.startswith('"') and field_name.endswith('"'):
                        field_name = field_name[1:-1]
                    elif field_name.startswith('`') and field_name.endswith('`'):
                        field_name = field_name[1:-1]
                    distribute_field_list.append(field_name)
                elif hasattr(expr, 'name'):
                    distribute_field_list.append(expr.name)
                else:
                    distribute_field_list.append(str(expr))
            # 设置分布键字段到全局字典中，与分区键保持一致
            distribute_fields[table_name] = distribute_field_list
    elif isinstance(prop, Property):
        this_arg = prop.args.get('this')
        value_arg = prop.args.get('value')
        if isinstance(this_arg, Literal):
            if this_arg.name == 'hoodie.table.description' and isinstance(value_arg, Literal):
                comments[table_name] = value_arg.name
            elif this_arg.name == 'primarykey' and isinstance(value_arg,Literal):
                primary_keys[table_name] = [field.strip() for field in value_arg.name.split(',')]
            # 处理FMRS方言中的primary.key属性
            elif this_arg.name == 'primary.key' and isinstance(value_arg, Literal):
                # 解析主键字段列表，支持逗号分隔的多个字段
                primary_keys[table_name] = [field.strip() for field in value_arg.name.split(',')]




def build_hudi_partition_config(pre_dialect, metadata, table_name=None):
    """根据方言构建 Hudi_Spark 的分区配置。

    Args:
        pre_dialect: 预处理方言
        metadata: 元数据
        table_name: 表名，如果指定则只返回该表的分区信息

    返回值:
        (tablePartitionInfo, partitionVOList)
    """
    if 'Hudi_Spark' in pre_dialect:
        # 获取指定表的分区信息
        partition_fields = []
        if table_name and table_name in metadata['partitions']:
            partition_fields = metadata['partitions'][table_name]

        # 如果没有分区字段，返回null
        if not partition_fields:
            return (None, [])

        partition_field = ','.join(partition_fields)
        return ({
            "partitionType": "SingleValue",
            "partitionExpression": "",
            "partitionField": partition_field,
            "partitionNum": None
        }, [])
    if 'DWS' in pre_dialect:
        # DWS方言分区配置
        return build_dws_partition_config(metadata, table_name)

    return (None, None)

def build_dws_partition_config(metadata, table_name=None):
    """构建DWS方言的分区配置

    Args:
        metadata: 元数据
        table_name: 表名，如果指定则只返回该表的分区信息

    返回值:
        (tablePartitionInfo, partitionVOList)
    """
    try:
        # 获取分区字段信息
        partition_fields = []
        partition_type = "SingleValue"  # 默认类型

        # 如果指定了表名，只处理该表的分区信息
        if table_name:
            if table_name in metadata['partitions']:
                partitions = metadata['partitions'][table_name]
                if partitions:
                    partition_fields = partitions
            else:
                # 该表没有分区信息，直接返回空
                return (None, [])
        else:
            # 兼容旧逻辑：从metadata中提取所有表的分区信息
            for table_name, partitions in metadata['partitions'].items():
                if partitions:
                    partition_fields.extend(partitions)

        # 从partition_details中确定分区类型
        partition_details = metadata.get('partition_details', {})
        target_table = table_name if table_name else list(partition_details.keys())[0] if partition_details else None

        if target_table and target_table in partition_details:
            details = partition_details[target_table]
            if details and 'type' in details:
                if details['type'] == 'LIST':
                    partition_type = "List"
                elif details['type'] == 'RANGE':
                    partition_type = "Range"
                elif details['type'] == 'HASH':
                    partition_type = "Hash"

        if not partition_fields:
            return (None, [])

        # 构建tablePartitionInfo
        tablePartitionInfo = {
            "partitionType": partition_type,  # 根据实际分区类型设置
            "partitionExpression": "",  # 分区表达式，DWS中通常为空
            "partitionField": ",".join(partition_fields),  # 分区字段，多个字段用英文逗号分隔
            "partitionNum": None  # 分区数，DWS中不使用
        }

        # 构建partitionVOList
        partitionVOList = extract_dws_partition_values(metadata, table_name)

        return (tablePartitionInfo, partitionVOList)

    except Exception as e:
        logger.error(f"构建DWS分区配置时出错: {str(e)}")
        return (None, [])

def extract_dws_partition_values(metadata, table_name=None):
    """从metadata中提取DWS分区值信息

    Args:
        metadata: 元数据
        table_name: 表名，如果指定则只返回该表的分区值信息

    返回值:
        list: partitionVOList格式的分区值列表
    """
    try:
        partitionVOList = []

        # 从partition_details中提取分区值信息
        partition_details = metadata.get('partition_details', {})

        # 如果指定了表名，只处理该表的分区值
        if table_name and table_name in partition_details:
            target_tables = [table_name]
        else:
            # 兼容旧逻辑：处理所有表
            target_tables = partition_details.keys()

        for table_name in target_tables:
            if table_name not in partition_details:
                continue

            details = partition_details[table_name]
            if not details or 'type' not in details:
                continue

            partition_type = details.get('type')
            partition_names = details.get('partition_names', [])
            # 新增：每个分区的原始值列表
            partition_raw_values = details.get('partition_raw_values', []) if 'partition_raw_values' in details else None

            if partition_type == 'LIST':
                # LIST分区：每个PARTITION为一个对象，值合并
                if partition_raw_values:
                    for i, value_list in enumerate(partition_raw_values):
                        partition_name = partition_names[i] if i < len(partition_names) else f"p{i+1}"
                        # 保留原始引号，拼接
                        value_str = ', '.join(value_list)
                        partitionVOList.append({
                            "partitionName": partition_name,
                            "partitionValue": value_str
                        })
                else:
                    # 兼容旧结构
                    for i, name in enumerate(partition_names):
                        partitionVOList.append({
                            "partitionName": name,
                            "partitionValue": ''
                        })
            elif partition_type == 'RANGE':
                # RANGE分区：处理所有分区
                if partition_names:
                    values = details.get('values', [])
                    for i, partition_name in enumerate(partition_names):
                        partition_value = values[i] if i < len(values) else ''
                        partitionVOList.append({
                            "partitionName": partition_name,
                            "partitionValue": partition_value
                        })
            elif partition_type == 'HASH':
                # HASH分区：通常没有具体的值
                partition_name = partition_names[0] if partition_names else "p1"
                partitionVOList.append({
                    "partitionName": partition_name,
                    "partitionValue": ""
                })

        return partitionVOList

    except Exception as e:
        logger.error(f"提取DWS分区值时出错: {str(e)}")
        return []




def extract_dws_indexes_from_sql(sql, table_name):
    """
    从SQL中提取DWS索引信息
    
    注意：此函数使用正则表达式进行文本模式匹配，
    作为AST解析的补充，处理复杂的索引语法。
    """
    indexes = []

    try:
        # 查找CREATE INDEX语句，排除注释行
        index_pattern = rf'^[^-]*CREATE\s+INDEX\s+(\w+)\s+ON\s+{re.escape(table_name)}\s*\(([^)]+)\)'
        matches = re.finditer(index_pattern, sql, re.IGNORECASE | re.MULTILINE)

        for match in matches:
            # 检查匹配的行是否以注释符号开头
            line_start = sql.rfind('\n', 0, match.start()) + 1
            line_content = sql[line_start:match.start()].strip()

            # 跳过注释行
            if line_content.startswith('--') or line_content.startswith('#'):
                continue

            index_name = match.group(1)
            index_columns = [col.strip() for col in match.group(2).split(',')]

            # 提取索引方法（如果有USING子句）
            index_method = "BTREE"  # 默认方法

            # 查找该索引的完整SQL片段
            index_sql_start = match.start()
            index_sql_end = sql.find(';', index_sql_start)
            if index_sql_end == -1:
                index_sql_end = len(sql)

            index_sql = sql[index_sql_start:index_sql_end]

            # 检查是否有USING子句
            using_match = re.search(r'USING\s+(\w+)', index_sql, re.IGNORECASE)
            if using_match:
                index_method = using_match.group(1)

            indexes.append({
                'indexName': index_name,
                'indexType': None,
                'indexFun': index_method,
                'indexFieldNames': index_columns,
                'pubExplain': ''
            })

    except Exception as e:
        logger.error(f"从SQL中提取DWS索引信息时出错: {str(e)}")

    return indexes


def deduplicate_indexes(indexes_list):
    """去重索引列表，基于索引名称和字段组合"""
    if not indexes_list:
        return []

    seen = set()
    unique_indexes = []

    for index in indexes_list:
        # 创建唯一标识：索引名 + 字段名组合
        index_name = index.get('indexName', '')
        field_names = tuple(sorted(index.get('indexFieldNames', [])))
        unique_key = (index_name, field_names)

        if unique_key not in seen:
            seen.add(unique_key)
            unique_indexes.append(index)

    return unique_indexes


def build_result(metadata, pre_dialect, original_sql=None):
    """
    构建最终的解析结果
    
    将收集到的元数据转换为标准化的表结构信息，包括：
    - 表基本信息（ID、名称、schema、catalog等）
    - 列信息列表（字段名、类型、约束、注释等）
    - 索引信息列表
    - 分区信息（分区类型、分区字段、分区值等）
    - 表创建选项（WITH语句等）
    
    Args:
        metadata: 从AST收集的元数据
        pre_dialect: 源数据库方言
        distr_dict: 分布键信息字典
        original_sql: 原始SQL语句
        
    Returns:
        list: 包含所有表结构信息的列表
    """
    result = []

    for table_name, data in metadata['tables'].items():
        columns = []

        for idx, col_data in enumerate(data['cols'], 1):
            col_name = col_data['name']
            col_comments = metadata['comments'].get(f"{table_name}.{col_name}", '')

            # 安全处理字段类型中的括号
            field_type = col_data['type'].split('(')[0] if '(' in col_data['type'] else col_data['type']

            # 安全处理默认值，确保Null对象被转换为None
            default_value = col_data['defaultValue']
            if default_value is not None:
                # 检查是否是sqlglot的Null对象
                if hasattr(default_value, '__class__') and default_value.__class__.__name__ == 'Null':
                    default_value = None
                # 检查是否是其他不可序列化的对象
                elif not isinstance(default_value, (str, int, float, bool, type(None))):
                    try:
                        # 尝试转换为字符串
                        default_value = str(default_value)
                    except:
                        default_value = None

            tablecolinfo = {
                'id': idx,
                'fieldName': col_name,
                'fieldNameCn': sanitize_string(col_comments),
                'fieldType': field_type,
                'fieldLength': col_data['length'],
                'fieldScale': col_data['scale'],
                'nullable': col_data['nullable'],
                'defaultValue': default_value,
                'primaryKey': col_name in metadata['primary_keys'].get(table_name, []),
                'partitionKey': col_name in metadata['partitions'].get(table_name, []),
                'distributeKey': col_name in metadata['distribute_fields'].get(table_name, []),
                'enumValue': col_data['enumValue'] if 'enumValue' in col_data else ""
            }

            columns.append(tablecolinfo)

        # 安全生成表ID
        table_id = generate_table_id(table_name)

        # 这两个字段暂时都只做Hudi_Spark 和 DWS, 其他方言都是None
        tablePartitionInfo, partitionVOList = build_hudi_partition_config(pre_dialect, metadata, table_name)

        # 使用每个表自己的addition_options
        table_addition_options = data.get('addition_options', None)

        # 获取该表的索引信息，并填充注释内容
        table_indexes = metadata['indexes'].get(table_name, [])

        # 如果从AST中没有获取到索引信息，尝试从原始SQL中提取
        if not table_indexes and original_sql and 'DWS' in pre_dialect:
            table_indexes = extract_dws_indexes_from_sql(original_sql, table_name)

        # 为索引添加注释信息
        for index_info in table_indexes:
            index_name = index_info.get('indexName', '')
            if index_name:
                # 直接通过索引名查找注释
                index_comment = metadata['comments'].get(index_name, '')
                index_info['pubExplain'] = sanitize_string(index_comment)

        # 对索引列表进行去重处理
        # table_indexes = deduplicate_indexes(table_indexes)

        tableinfo = {
            'id': table_id,
            'catalog': data.get('catalog'),
            'schema': data['schema'],
            'tableName': table_name,
            'tableNameCn': sanitize_string(metadata['comments'].get(table_name, '')),
            'isTemporaryTable': data['is_temp'],
            'tableCreateOption': table_addition_options,
            'tablePartitionInfo': tablePartitionInfo,
            'partitionVOList': partitionVOList,
            'tableColInfoList': columns,
            'tableIndexList': table_indexes,  # 新增：索引信息（已去重）
        }

        result.append(tableinfo)

    return result


def generate_table_id(table_name):
    """
    生成安全的表ID
    
    使用哈希算法生成唯一的表ID，避免冲突
    
    Args:
        table_name: 表名
        
    Returns:
        int: 生成的表ID
    """
    try:
        # 使用哈希代替简单取模，避免冲突
        return int(hashlib.sha256(table_name.encode()).hexdigest()[:8], 16)
    except:
        return abs(hash(table_name)) % 1000000


def sanitize_string(value):
    """
    清理字符串中的特殊字符
    
    移除控制字符和不可见字符，确保字符串安全
    
    Args:
        value: 需要清理的字符串
        
    Returns:
        str: 清理后的字符串
    """
    if not value:
        return value
    return re.sub(r'[\x00-\x1F\x7F-\x9F]', '', value)




# =============================================================================
# 预处理函数 - SQL语句预处理
# =============================================================================
def preprocess(sql_data, pre_dialect, dialect):
    """
    预处理SQL语句，进行基本的验证和清理
    
    Args:
        sql_data: 原始SQL语句
        pre_dialect: 源数据库方言
        dialect: 目标方言
        
    Returns:
        str: 处理后的SQL语句
    """
    if not sql_data:
        return ""

    try:
        # 检查SQL长度限制
        if len(sql_data) > MAX_SQL_LENGTH:
            logger.warning("SQL过长，进行截断处理")
            sql_data = sql_data[:MAX_SQL_LENGTH]

        # 由于sqlglot已经支持各种方言，不再需要复杂的预处理
        return sql_data
        
    except Exception as e:
        logger.error(f"预处理失败: {str(e)}")
        return sql_data



# =============================================================================
# 文件处理函数 - 文件读取和工具函数
# =============================================================================
def read_file(file_path):
    """
    读取文件内容
    
    Args:
        file_path: 文件路径
        
    Returns:
        str: 文件内容
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content

def read_files(folder_path):
    """
    读取文件夹中的所有文件内容
    
    Args:
        folder_path: 文件夹路径
        
    Returns:
        str: 所有文件内容的合并字符串
    """
    contents = []

    # 获取文件夹中所有文件的完整路径，并排序（可按文件名）
    all_files = sorted([
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
    ])

    # 读取所有文件
    for file_path in all_files:
        with open(file_path, 'r', encoding='utf-8') as file:
            contents.append(file.read())

    result = "".join(contents)
    return result

# =============================================================================
# 主程序入口 - 测试和示例代码
# =============================================================================
if __name__ == "__main__":
    # sql_data = """
    # CREATE TABLE t_loan_loan_app (
    # cust_name varchar(32)  COMMENT '客户姓名'
    # ,PRIMARY KEY USING BTREE (cust_name)
    # ,INDEX indexl USING BTREE (cust_name)
    # ) ENGINE=oceanbase
    # DEFAULT CHARSET=utf8mb COMMENT '放款申请表';
    # """


    # sql_data = "ALTER TABLE ltytest.table_test0808 ADD PARTITION p3 VALUES LESS THAN ('hij');"
    # sql_data = "ALTER TABLE table1 DROP PARTITION p1"
    # all_tables = parse_ast(sql_data, 'ClickHouse','clickhouse')
    # all_tables = parse_ast(sql_data, 'DB2', 'db2')
    # all_tables = parse_ast(sql_data,'OceanBase_MySQL','mysql')
    # all_tables = parse_ast(sql_data, 'Hudi_Spark', 'spark')
    # all_tables = parse_ast(sql_data,'mysql','mysql')
    # all_tables = parse_ast(sql_data,'postgreSQL','postgres')
    # folder = '/Users/ynkang/Downloads/抽取代码信息/DDL/failed/dwstable_081802_12025-08-18.sql'
    # folder = '/Users/ynkang/Downloads/抽取代码信息/DDL/clickhouse.sql'
    # folder = '/Users/ynkang/Downloads/DDL/hwmrs_主键.sql'
    # folder = '/Users/ynkang/Downloads/抽取代码信息/DDL/failed/Hudi_spark-0811.sql'
    # folder = '/Users/ynkang/Downloads/抽取代码信息/DDL/failed/hudi-Spark_test.sql'
    # folder = '/Users/ynkang/Downloads/抽取代码信息/DDL/failed/hudi-Spark.sql'
    # folder = '/Users/ynkang/Downloads/抽取代码信息/DDL/failed/hudi-spark-2025-08-07.sql'
    # folder = '/Users/ynkang/Downloads/抽取代码信息/DDL/failed/gaussdb.sql'
    # folder = '/Users/ynkang/Downloads/抽取代码信息/DDL/failed/DWS2025-08-18.sql'
    # folder = '/Users/ynkang/Downloads/抽取代码信息/DDL/failed/DWStable_081802_12025-08-18.sql'
    # folder = '/Users/ynkang/Downloads/抽取代码信息/DDL/failed/DWStable_0818022025-08-18.sql'
    # folder = '/Users/ynkang/Downloads/抽取代码信息/DDL/failed/DB2.sql'
    # folder = '/Users/ynkang/Downloads/抽取代码信息/test/hive.sql'
    # folder = '/Users/ynkang/Downloads/抽取代码信息/DDL/failed/DWS2025-08-18.sql'
    # folder = '/Users/ynkang/Downloads/抽取代码信息/DDL/failed/qianBaseMPP.sql'
    # folder = '/Users/ynkang/Downloads/抽取代码信息/DDL/failed/DWS_index.sql'
    # folder = '/Users/ynkang/Downloads/抽取代码信息/DDL/failed/DWS.sql'
    # folder = '/Users/ynkang/Downloads/抽取代码信息/DDL/failed/DWS2025-08-18.sql'
    # folder = '/Users/ynkang/Downloads/抽取代码信息/DDL/failed/DWS_0813.sql'
    # folder = '/Users/ynkang/Downloads/抽取代码信息/DDL/failed/DWS_index.sql'
    # folder = '/Users/ynkang/Downloads/抽取代码信息/DDL/failed/DWS_test.sql'
    # folder = '/Users/ynkang/Downloads/抽取代码信息/DDL/failed/OceanBase-mysql-0910.sql'
    # folder = '/Users/ynkang/Downloads/抽取代码信息/test/hwmrs_test.sql'
    # folder = './badcases/hudi-Spark.sql'
    # folder = '/Users/ynkang/Downloads/抽取代码信息/DDL/failed/l&Ptest_12025-08-04.sql'
    # folder = '/Users/ynkang/Downloads/抽取代码信息/DDL/failed/hudi-Spark_test.sql'
    # folder = '/Users/ynkang/Downloads/抽取代码信息/test/TIDB.sql'
    # folder = '/Users/ynkang/Downloads/抽取代码信息/test/greenplum.sql'
    # folder = '/Users/ynkang/Downloads/抽取代码信息/DDL/failed/DWS2025-08-27.sql'
    # folder = '/Users/ynkang/Downloads/抽取代码信息/DDL/failed/Hudi_Spark_0904.sql'
    # folder = '/Users/ynkang/Downloads/抽取代码信息/DDL/failed/hudi-Spark-0909.sql'
    # folder = '/Users/ynkang/Downloads/抽取代码信息/DDL/pass/oracle_0912.sql'
    folder = './badcases/ORACLE_分区.sql'
    # folder = '/Users/ynkang/Downloads/抽取代码信息/DDL/failed/hudi-Spark.sql'
    # folder = '/Users/ynkang/Downloads/抽取代码信息/DDL/failed/hudi-flink.sql'
    # folder = './badcases/DB2.sql'
    # folder = './badcases/hive_test.sql'
    # folder = './badcases/qianBaseMPP.sql'
    # folder = './badcases/clickhouse.sql'
    # folder = './badcases/FMRS.sql'
    # folder = './badcases/mysql_分区.sql'
    # folder = './badcases/OceanBase_MySQL.sql'
    # folder = './badcases/mysql_分区.sql'
    # folder = './badcases/hive.sql'
    # folder = './badcases/mysql2025-09-18.sql'
    # folder = './badcases/DWS2025-08-18.sql'
    # folder = './badcases/gaussdb_test.sql'
    # folder = './badcases/mysql_enum.sql'
    # folder = './badcases/DWStable_0818022025-08-18.sql'
    # folder = './badcases/dm_multi_table.sql'
    # folder = './badcases/goldendb_分区.sql'
    # folder = './badcases/tdsql_分区.sql'
    # folder = './badcases/greenplum_fixed.sql'
    # folder = './badcases/qianBaseMPP.sql'
    # folder = './badcases/oracle_test.sql'
    sql_data = read_file(folder)
    # all_tables = parse_ast(sql_data, 'FMRS', 'databricks') # databricks
    # all_tables = parse_ast(sql_data, 'FMRS', 'fmrs') # databricks
    # all_tables = parse_ast(sql_data, 'hive', 'hive')
    # all_tables = parse_ast(sql_data, 'DB2', 'db2')
    # all_tables = parse_ast(sql_data, 'qianBaseMPP', 'qianbasempp')
    # all_tables = parse_ast(sql_data,'gaussDB','gaussdb')
    # all_tables = parse_ast(sql_data, 'FMRS', 'databricks')
    # all_tables = parse_ast(sql_data, 'mysql', 'mysql')
    # all_tables = parse_ast(sql_data, 'Hudi_Spark', 'spark')
    # all_tables = parse_ast(sql_data,'DWS', 'dws')
    # all_tables = parse_ast(sql_data, 'DWS2025-08-12', 'dws')
    # all_tables = parse_ast(sql_data,'ClickHouse','clickhouse')
    # all_tables = parse_ast(sql_data, 'Hudi_Spark', 'spark')
    # all_tables = parse_ast(sql_data, 'Hudi_Spark', 'hudi_spark')
    # all_tables = parse_ast(sql_data,'TIDB', 'mysql')
    # all_tables = parse_ast(sql_data,'greenplum','athena')
    # all_tables = parse_ast(sql_data,'greenplum','greenplum')
    # all_tables = parse_ast(sql_data, 'OceanBase-mysql-0910', 'mysql')
    all_tables = parse_ast(sql_data, 'oracle', 'oracle')
    # all_tables = parse_ast(sql_data, 'Hudi_Flink', 'databricks')
    # all_tables = parse_ast(sql_data, 'DB2', 'db2')
    # all_tables = parse_ast(sql_data, 'hive', 'hive')
    # all_tables = parse_ast(sql_data, 'qianBaseMPP', 'qianbasempp')
    # all_tables = parse_ast(sql_data, 'ClickHouse', 'clickhouse')
    # all_tables = parse_ast(sql_data, 'FMRS', 'fmrs')
    # all_tables = parse_ast(sql_data, 'oceanbase-mysql', 'oceanbase_mysql')
    # all_tables = parse_ast(sql_data, 'gaussDB', 'gaussdb')
    # all_tables = parse_ast(sql_data, 'dm', 'dm')
    # all_tables = parse_ast(sql_data, 'GoldenDB', 'mysql')
    # all_tables = parse_ast(sql_data, 'TDSQL', 'mysql')
    # all_tables = parse_ast(sql_data, 'greenplum', 'greenplum')
    # all_tables = parse_ast(sql_data, 'qianBaseMPP', 'qianbasempp')
    # all_tables = parse_ast(sql_data, 'oceanbase_mysql', 'oceanbase_mysql')
    print(json.dumps(all_tables, indent=2, ensure_ascii=False))

    
    # root = "test"
    # directory = "/Users/ynkang/Downloads/抽取代码信息/test"
    # directory = "/Users/ynkang/Downloads/抽取代码信息/DDL"
    # for root, _, files in os.walk(directory):
    #     for i,filename in enumerate(files):
    #         if filename.lower().endswith('.sql'):
    #             # if filename != ('OceanBase_MySQL.sql'):
    #             #     continue
    #             full_path = os.path.join(root, filename)
    #             print(filename)
    #             sql_data = read_sql_file(full_path)
    #             key = filename.split(".")[0]
    #             value = dialect_map[key]
    #             print(key, dialect_map[key])
                # all_tables = parse_ast(sql_data, 'Hudi_Spark', 'hudi_spark')
                # all_tables = parse_ast(sql_data, key, value)
                # print(json.dumps(all_tables, indent=2, ensure_ascii=False))

    # filename = 'argodb.sql'
    # filename = 'Oracle.sql'
    # filename = 'gaussdb.sql'
    # filename = 'hive.sql'
    # filename = 'oracle_test.sql'

    # full_path = os.path.join(root, filename)
    # sql_data = read_sql_file(full_path)
    # all_tables = parse_ast(sql_data, 'argodb', dialect_map['argodb'])
    # all_tables = parse_ast(sql_data, 'gaussdb', dialect_map['gaussdb'])
    # all_tables = parse_ast(sql_data, 'hive', dialect_map['hive'])
    # all_tables = parse_ast(sql_data, 'Oracle', dialect_map['Oracle'])
    # all_tables = parse_ast(sql_data, 'oracle_test', dialect_map['oracle_test'])
    # print(json.dumps(all_tables, indent=2, ensure_ascii=False))
    print('1')

