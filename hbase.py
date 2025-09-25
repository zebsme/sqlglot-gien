import re
import hashlib
import json

# def generate_table_id(table_name):
#     """生成安全的表ID"""
#     try:
#         # 使用哈希代替简单取模，避免冲突
#         return int(hashlib.sha256(table_name.encode()).hexdigest()[:8], 16)
#     except:
#         return abs(hash(table_name)) % 1000000

# def parse_hbase_ddl(ddl,dialect='hbase'):
#     # 准备正则
#     pattern1 = re.compile(r"create\s+'(?P<table>[^']+)'\s*,(.*?);", re.IGNORECASE | re.DOTALL)
#     col_pattern1 = re.compile(r"\{NAME\s*=>\s*'([^']+)'\}", re.IGNORECASE)
#
#     # pattern2 = re.compile(r"create\s+'(?P<table>[^']+)'\s*(?:,\s*'([^']+)')+", re.IGNORECASE)
#     pattern2 = re.compile(r"create\s+'(?P<table>[^']+)'\s*(?:,\s*'([^']+)')+\s*;?", re.IGNORECASE)
#     col_pattern2 = re.compile(r"'([^']+)'")
#
#     tables = []
#
#     # 处理第一种格式
#     for m in pattern1.finditer(ddl):
#         table = m.group('table')
#         block = m.group(2)
#         cols = col_pattern1.findall(block)
#         # 生成 column_info
#         column_info = []
#         for idx, col in enumerate(cols, start=1):
#             column_info.append({
#                 "id": idx,
#                 "fieldName": col,
#                 "fieldNameCn": "",
#                 "fieldType": "String",
#                 "fieldLength": None,
#                 "fieldScale": None,
#                 "nullable": True,
#                 "primaryKey": False,
#                 "partitionKey": False,
#                 "distributeKey": False
#             })
#         # 生成表结构
#         table_id = generate_table_id(table)
#         tables.append({
#             "id": table_id,
#             "catalog": None,
#             "schema": "public",
#             "tableName": table,
#             "tableNameCn": "",
#             "isTemporaryTable": False,
#             "tableColInfoList": column_info
#         })
#
#     # 处理第二种格式
#     for line in ddl.splitlines():
#         m2 = pattern2.search(line)
#         if m2:
#             parts = col_pattern2.findall(line)
#             table = parts[0]
#             cols = parts[1:]
#             column_info = []
#             for idx, col in enumerate(cols, start=1):
#                 column_info.append({
#                     "id": idx,
#                     "fieldName": col,
#                     "fieldNameCn": "",
#                     "fieldType": "String",
#                     "fieldLength": None,
#                     "fieldScale": None,
#                     "nullable": True,
#                     "primaryKey": False,
#                     "partitionKey": False,
#                     "distributeKey": False
#                 })
#             tables.append({
#                 "id": hash(table) % 1000000,
#                 "catalog": None,
#                 "schema": "public",
#                 "tableName": table,
#                 "tableNameCn": "",
#                 "isTemporaryTable": False,
#                 "tableColInfoList": column_info
#             })
#     return tables


def parse_hbase_ddl(ddl, dialect='hbase'):
    # 统一表ID生成方式
    def generate_table_id(table_name):
        return abs(hash(table_name)) % 1000000

    # 构建列信息 (复用函数)
    def build_column_info(cols):
        return [
            {
                "id": idx,
                "fieldName": col,
                "fieldNameCn": "",
                "fieldType": "String",
                "fieldLength": None,
                "fieldScale": None,
                "nullable": True,
                "primaryKey": False,
                "partitionKey": False,
                "distributeKey": False
            }
            for idx, col in enumerate(cols, start=1)
        ]

    # 构建表信息 (复用函数)
    def build_table_info(table, cols):
        return {
            "id": generate_table_id(table),
            "catalog": None,
            "schema": "public",
            "tableName": table,
            "tableNameCn": "",
            "isTemporaryTable": False,
            "tableColInfoList": build_column_info(cols)
        }

    # 统一正则模式
    table_pattern = re.compile(
        r"create\s+'(?P<table>[^']+)'\s*"  # 捕获表名
        r"(?:,\s*)?(?P<columns>\{.*?\}|(?:'[^']+',?\s*)+)\s*;?",  # 捕获列定义
        re.IGNORECASE | re.DOTALL
    )

    # 两种列定义模式的解析器
    def parse_columns(column_str):
        # 模式1: {NAME => 'col1'}, {NAME => 'col2'}
        if column_str.startswith('{'):
            return re.findall(r"\{NAME\s*=>\s*'([^']+)'", column_str)
        # 模式2: 'col1', 'col2', 'col3'
        else:
            return re.findall(r"'([^']+)'", column_str)[1:]  # 跳过表名

    tables = []
    seen_tables = set()  # 防止重复表

    try:
        for match in table_pattern.finditer(ddl):
            table = match.group('table')
            if table in seen_tables:
                continue

            seen_tables.add(table)
            column_str = match.group('columns')

            try:
                cols = parse_columns(column_str)
                if cols:
                    tables.append(build_table_info(table, cols))
                else:
                    logger.warning(f"No columns found for table {table}")
            except Exception as e:
                logger.error(f"Column parsing error for {table}: {str(e)}")

    except Exception as e:
        logger.error(f"DDL parsing failed: {str(e)}")

    return tables





if __name__ == "__main__":
    ddl = """
    create_namespace 'table_lzs0512';
    
    drop 'fq_account_draw';
    
    create 'fq_account_draw', 
     {NAME => 'org_num'},
     {NAME => 'data_dt'},
     {NAME => 'dzbs_edit'},
     {NAME => 'gmbs'},
     {NAME => 'gmje_edit'},
     {NAME => 'fieldName7'}
    ;
    
    create 'user_info', 'basic', 'detail';
    create 'user_info2', 'basic','contact','preference';
    create 'product','info','price','inventory';
    """

    tables = parse_hbase_ddl(ddl)
    print(json.dumps(tables, indent=4, ensure_ascii=False))
