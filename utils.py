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