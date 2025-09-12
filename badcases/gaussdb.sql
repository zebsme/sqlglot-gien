--ADD PARTITION partition_name VALUES (partition_value)
ALTER TABLE FQ.FQ_RPT_CD_089 ADD PARTITION P_20240101 VALUES(DATE'2024-01-01'); 
--ALTER TABLE table_name DROP PARTITION partition_name
ALTER TABLE CDB.T66_ACCT_TRANS_DETAIL DROP PARTITION P_20240101;
--SET TABLESPACE space_name
ALTER TABLE FQ.FQ_RPT_CD_089 SET TABLESPACE H_TBS_TMP;
--CREATE TABLE table_name TABLESPACE space_name
CREATE TABLE CDB.hx_check_bal_info_RESULT
 WITH (orientation=column)
 TABLESPACE h_tbs_cdb
 AS
 SELECT ORG_NO    
   FROM CDB.hx_check_bal_info ;
SELECT TO_CHAR(DATE '2024-01-01' - '2024-01-01' + 1) FROM TABLE_A;
--PARTITION BY VALUES (values)
CREATE TABLE IF.AUDIT_CBS_LNFM23(
    L23ORGNO VARCHAR2(27)
)
PARTITION BY VALUES (DATA_DT)
;
--PARTITION BY LIST (column_name) ... ([PARTITION partition_name VALUES (values),PARTITION partition_name VALUES (values)])
CREATE  TABLE fq_agent_national_pay_fee_detail (
	data_dt date,
)
PARTITION BY LIST (data_dt) 
(	 PARTITION p_19000101 VALUES ('1900-01-01'))
;
--INTERVAL 'value,unit' 暂不处理
select   t1.ORG_NUM
from fq.fq_item_gl_bal t1
where t1.data_dt =  DATE '2024-01-01' + INTERVAL '-12,month' and sum_typ_cd='日';

--SERVER，OPTIONS，READ ONLY
CREATE  TABLE SDB.S02_KXD_HBKHLOG_EXT
( 
	BZBJG	VARCHAR2(60) 
)
SERVER gsmpp_server 
OPTIONS (
  LOCATION 'DATA_PATH2/CMS_DATA/RECEIVE/20240101/S02_KXD_HBKHLOG.txt',
  FORMAT 'TEXT',
  MODE 'NORMAL',
  ENCODING 'UTF8',
  DELIMITER E'',
  NOESCAPING 'TRUE',
  NULL '',COMPATIBLE_ILLEGAL_CHARS 'TRUE',
  FILL_MISSING_FIELDS 'TRUE',
  IGNORE_EXTRA_DATA 'TRUE'
)
READ ONLY
;
-- TO GROUP group_name， ENABLE ROW MOVEMENT
CREATE  TABLE d60_paf_settle_acct_dtl (
	data_dt timestamp without time zone,
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