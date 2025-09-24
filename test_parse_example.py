from sqlglot import parse_one,parse

# GAUSSDB
# k="""DROP FOREIGN TABLE IF EXISTS SDB.S02_KXD_HBKHLOG_EXT;"""
k="""
CREATE  TABLE d60_paf_settle_acct_dtl (
	data_dt timestamp without time zone,
)
TABLESPACE h_tbs_cdb
DISTRIBUTE BY HASH(acct_no)
TO GROUP group_version1
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