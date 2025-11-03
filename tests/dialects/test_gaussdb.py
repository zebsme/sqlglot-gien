from sqlglot.dialects.dialect import Dialects
from tests.dialects.test_dialect import Validator

class TestGaussDB(Validator):
    dialect = "gaussdb"

    sql = """CREATE TABLE CDB.hx_check_bal_info_RESULT
WITH (orientation=column)
TABLESPACE h_tbs_cdb
DISTRIBUTE BY HASH (ORG_NO,SBJ_NO,DATA_DT)
AS
SELECT       DATA_DT               -- 统计日期      
            ,ORG_NO              -- 借据号
            ,SBJ_NO  
            ,SRC_SYS 
            ,DIF_VAL      
  FROM CDB.hx_check_bal_info 
 WHERE DATA_DT = DATE'2024-01-01'
   AND ((ORG_NO NOT IN('601101','601102','601103','601151','601171','601201','601211','601221'
                 ,'601241','601421','601441','601461','601481','601121','601463','601462') AND ABS(DIF_VAL) > 0)
          OR (ORG_NO IN('601101','601102','601103','601151','601171','601201','601211','601221'
                 ,'601241','601421','601441','601461','601481','601121','601463','601462')
               AND ABS(DIF_VAL) > 1) 
       )
  ;"""
    def test_druid(self):
        self.validate_identity(self.sql)
        self.validate_identity("SELECT CEIL(__time TO WEEK) FROM t")
        self.validate_identity("SELECT CEIL(col) FROM t")
        self.validate_identity("SELECT CEIL(price, 2) AS rounded_price FROM t")
        self.validate_identity("SELECT FLOOR(__time TO WEEK) FROM t")
        self.validate_identity("SELECT FLOOR(col) FROM t")
        self.validate_identity("SELECT FLOOR(price, 2) AS rounded_price FROM t")
        self.validate_identity("SELECT CURRENT_TIMESTAMP")

        # validate across all dialects
        write = {dialect.value: "FLOOR(__time TO WEEK)" for dialect in Dialects}
        self.validate_all(
            "FLOOR(__time TO WEEK)",
            write=write,
        )
        

# k="ALTER TABLE FQ.FQ_RPT_CD_089 ADD PARTITION P_20240101 VALUES(DATE'2024-01-01'); "
# k="""
# 	CREATE TABLE TMP.CUS_BLK_LIST_TMP_12345678 	WITH (orientation=column) 
#     DISTRIBUTE BY HASH (cus_id) 
#     AS
# 	  SELECT   T.AGRI_FLG	    AS CITY_VILLAGE_FLG    /*农户标志*/
#   from ODB.S56_BIZ_CUS_INDIV T  
#    ;   """
# k="ALTER TABLE CDB.T66_ACCT_TRANS_DETAIL DROP PARTITION P_20240101;"
k="""ALTER TABLE FQ.FQ_RPT_CD_089 SET TABLESPACE H_TBS_TMP"""