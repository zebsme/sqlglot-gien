from sqlglot import parse_one,parse

# GAUSSDB
# k="""DROP FOREIGN TABLE IF EXISTS SDB.S02_KXD_HBKHLOG_EXT;"""
k="""
INSERT INTO SDB.S00_DBA_TAB_COLUMNS0101
SELECT 
    nvl(trim(OWNER),' ')
    ,nvl(trim(TABLE_NAME),' ')
    ,nvl(trim(COLUMN_NAME),' ')
    ,nvl(trim(DATA_DEFAULT),' ')
    ,nvl(trim(NULLABLE),' ')
    ,nvl(trim(DATA_TYPE),' ')
    ,to_number(nvl(trim(CHAR_LENGTH),0))
    ,to_number(nvl(trim(DATA_PRECISION),0))
    ,to_number(nvl(trim(DATA_SCALE),0))
    ,to_number(nvl(trim(COLUMN_ID),0))
 FROM SDB.S00_DBA_TAB_COLUMNS_EXT
;
"""
# WITH err_HR_staffS_ft
# LOG INTO SDB.S01_BIFM5A_ERR PER NODE REJECT LIMIT 'unlimited'; 
# ORACLE
# k="""CREATE TABLE A
#   (
#     "KLXBBH" VARCHAR2(10) NOT NULL ENABLE
#   ) 
# TABLESPACE "CBSDATA" 
# ;"""

ex1=parse(k, dialect = "gaussdb")
print(1)

# ex1[0]