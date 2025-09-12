from sqlglot import parse_one,parse

# GAUSSDB
# k="""DROP FOREIGN TABLE IF EXISTS SDB.S02_KXD_HBKHLOG_EXT;"""
k="""
CREATE  TABLE d60_paf_settle_acct_dtl (
	data_dt timestamp without time zone,
	acct_no character varying(100),
    CUST_ACCT_NO   character varying(100),
	acct_typ character varying(20),
	cust_no character varying(100),
	open_dt timestamp(0) without time zone,
	id_type character varying(10),
	id_no character varying(100),
	last_tx_dt timestamp(0) without time zone,
	acct_bal numeric(22,2),
	no_up_days character varying(30),
	CHG_FLAG   character varying(10)      -- 
)
WITH (orientation=column, compression=low, colversion=2.0, enable_delta=false)
TABLESPACE h_tbs_cdb
DISTRIBUTE BY HASH(acct_no)
TO GROUP group_version1
PARTITION BY LIST (data_dt)
( 
	 PARTITION p_19000101 VALUES ('1900-01-01 00:00:00'::timestamp without time zone),
	 PARTITION p_20241031 VALUES ('2024-10-31 00:00:00'::timestamp without time zone)
)
ENABLE ROW MOVEMENT;
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