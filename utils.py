import logging
import sys
import json
from pathlib import Path
# from jinja2 import Environment, FileSystemLoader
import os

def getLogger(name, file_name, use_formatter=True):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s    %(message)s')
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    if file_name:
        handler = logging.FileHandler(file_name, encoding='utf8')
        handler.setLevel(logging.INFO)
        if use_formatter:
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
            handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
logger = getLogger('sql_extractor', 'sql_extractor.log')

def file_reader(file_path):
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'shift-jis', 'iso-8859-1']
    content = None
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            # print(f"成功使用 {encoding} 编码读取文件")
            break
        except UnicodeDecodeError:
            continue
    return content

def save_res(file_path, sqls):
    with open(file_path, 'w', encoding='utf-8') as w:
        w.write(json.dumps(sqls, ensure_ascii=False, indent=4))  

# def perl_mock_generator(tmpl_name,**kwargs):
#     props = kwargs
#     current_dir_path = Path(__file__).parent
#     path_to_template = os.path.join(current_dir_path, "templates")
#     logger.info(f"template path: {path_to_template}")
#     if os.path.exists(path_to_template):
#         logger.info(f"template path exists")
#     env = Environment(loader=FileSystemLoader(path_to_template))
#     tmpl = env.get_template(tmpl_name)
#     try:
#         text = tmpl.render(**props)
#         logger.info(f"template name: {text}")
#     except Exception as e:
#         logger.error(f"Error in template: {e}")
#         text = ""
#     return text

def read_sql_file(file_path: str) -> str:
    """
    读取 SQL 文件内容
    :param file_path: SQL 文件路径
    :return: SQL 文件内容字符串
    :raises: FileNotFoundError, IOError
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"SQL 文件不存在: {file_path}")
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='GBK') as f:  # 尝试 GBK 编码
            return f.read()
