from textwrap import dedent

from sqlglot import exp
from sqlglot.dialects.dialect import Dialect, Dialects
from tests.dialects.test_dialect import Validator


class TestGaussDB(Validator):
    dialect = "gaussdb"

    def test_create_table_with_tablespace_and_options(self):
        ctases = (
            (
                dedent(
                    """
                    CREATE TABLE CDB.hx_check_bal_info_RESULT
                    WITH (orientation=column)
                    TABLESPACE h_tbs_cdb
                    DISTRIBUTE BY HASH (ORG_NO,SBJ_NO,DATA_DT)
                    AS
                    SELECT DATA_DT, ORG_NO, SBJ_NO, SRC_SYS, DIF_VAL
                    FROM CDB.hx_check_bal_info
                    WHERE DATA_DT = DATE'2024-01-01'
                    """
                ).strip(),
                "CREATE TABLE CDB.hx_check_bal_info_RESULT TABLESPACE=h_tbs_cdb DISTRIBUTED BY HASH "
                "(ORG_NO, SBJ_NO, DATA_DT) WITH (orientation=column) AS SELECT DATA_DT, ORG_NO, SBJ_NO, "
                "SRC_SYS, DIF_VAL FROM CDB.hx_check_bal_info WHERE DATA_DT = CAST('2024-01-01' AS DATE)",
            ),
            (
                dedent(
                    """
                    CREATE TABLE TMP.CUS_BLK_LIST_TMP_12345678
                    WITH (orientation=column)
                    DISTRIBUTE BY HASH (cus_id)
                    AS
                    SELECT T.AGRI_FLG AS CITY_VILLAGE_FLG
                    FROM ODB.S56_BIZ_CUS_INDIV T
                    """
                ).strip(),
                "CREATE TABLE TMP.CUS_BLK_LIST_TMP_12345678 DISTRIBUTED BY HASH (cus_id) WITH "
                "(orientation=column) AS SELECT T.AGRI_FLG AS CITY_VILLAGE_FLG FROM ODB.S56_BIZ_CUS_INDIV AS T",
            ),
        )

        for sql, expected in ctases:
            expr = self.validate_identity(sql, write_sql=expected)
            expr.assert_is(exp.Create)

    def test_scalar_rounding_functions(self):
        for sql in (
            "SELECT CEIL(__time TO WEEK) FROM t",
            "SELECT CEIL(col) FROM t",
            "SELECT CEIL(price, 2) AS rounded_price FROM t",
            "SELECT FLOOR(__time TO WEEK) FROM t",
            "SELECT FLOOR(col) FROM t",
            "SELECT FLOOR(price, 2) AS rounded_price FROM t",
            "SELECT CURRENT_TIMESTAMP",
        ):
            expr = self.validate_identity(sql)
            expr.assert_is(exp.Select)

        write = {}
        for dialect in Dialects:
            if not dialect.value:
                continue
            try:
                Dialect.get_or_raise(dialect.value)
            except Exception:
                continue
            write[dialect.value] = "FLOOR(__time TO WEEK)"

        self.validate_all("FLOOR(__time TO WEEK)", write=write)

    def test_enable_row_movement_property(self):
        sql = "CREATE TABLE t (c INT) ENABLE ROW MOVEMENT"
        expr = self.validate_identity(sql, write_sql="CREATE TABLE t (c INT4) ENABLE ROW MOVEMENT")
        expr.assert_is(exp.Create)
        self.assertEqual("CREATE TABLE t (c INT) ENABLE ROW MOVEMENT", expr.sql())

    def test_alter_table_add_partition_values(self):
        sql = "ALTER TABLE FQ.FQ_RPT_CD_089 ADD PARTITION P_20240101 VALUES(DATE'2024-01-01')"
        expected = (
            "ALTER TABLE FQ.FQ_RPT_CD_089 ADD PARTITION P_20240101 VALUES (CAST('2024-01-01' AS DATE))"
        )
        expr = self.validate_identity(sql, write_sql=expected)
        alter = expr.assert_is(exp.Alter)
        action = alter.args['actions'][0].assert_is(exp.AddGaussDBPartition)
        action.this.assert_is(exp.Identifier)
        self.assertIsInstance(action.args["values"], exp.Var)
        self.assertEqual(action.args["values"].name, "VALUES")

    def test_alter_table_drop_partition(self):
        sql = "ALTER TABLE CDB.T66_ACCT_TRANS_DETAIL DROP PARTITION P_20240101"
        expr = self.validate_identity(sql)
        alter = expr.assert_is(exp.Alter)
        alter.args['actions'][0].assert_is(exp.DropPartition)

    def test_alter_table_drop_partition_for(self):
        sql = "ALTER TABLE t DROP PARTITION FOR (1)"
        expr = self.validate_identity(sql)
        alter = expr.assert_is(exp.Alter)
        partition = alter.args["actions"][0].expressions[0].assert_is(exp.Partition)
        self.assertEqual(sql, expr.sql(dialect=self.dialect))
        self.assertEqual(len(partition.expressions), 1)

    def test_alter_table_add_partition_for_values_in(self):
        sql = "ALTER TABLE t ADD PARTITION p FOR VALUES IN (1, 2)"
        expr = self.validate_identity(sql)
        alter = expr.assert_is(exp.Alter)
        action = alter.args["actions"][0].assert_is(exp.AddGaussDBPartition)
        self.assertIsInstance(action.args["values"], exp.Var)
        self.assertEqual(action.args["values"].name, "FOR_VALUES_IN")
        self.assertEqual(sql, expr.sql(dialect=self.dialect))

    def test_alter_table_add_partition_for_values_range(self):
        sql = "ALTER TABLE t ADD PARTITION p FOR VALUES FROM (1) TO (2)"
        expr = self.validate_identity(sql)
        alter = expr.assert_is(exp.Alter)
        action = alter.args["actions"][0].assert_is(exp.AddGaussDBPartition)
        self.assertIsInstance(action.args["values"], exp.Var)
        self.assertEqual(action.args["values"].name, "FOR_VALUES_RANGE")
        self.assertEqual(sql, expr.sql(dialect=self.dialect))

    def test_alter_table_set_tablespace(self):
        sql = "ALTER TABLE FQ.FQ_RPT_CD_089 SET TABLESPACE H_TBS_TMP"
        expr = self.validate_identity(sql)
        alter = expr.assert_is(exp.Alter)
        alter.args['actions'][0].assert_is(exp.AlterSet)

    def test_alter_table_owner(self):
        sql = "ALTER TABLE fq.fq_mkt_dept_cust_detail_a OWNER TO sjck"
        expr = self.parse_one(sql)
        alter = expr.assert_is(exp.Alter)
        action = alter.expressions[0].assert_is(exp.AlterOwner)
        action.this.assert_is(exp.Table).assert_name("fq.fq_mkt_dept_cust_detail_a")
        action.expression.assert_is(exp.Identifier).assert_name("sjck")
        self.assertEqual(sql, expr.sql(dialect=self.dialect))
        self.assertEqual("ALTER TABLE fq.fq_mkt_dept_cust_detail_a OWNER TO sjck", expr.sql())


    def test_create_foreign_table_with_options(self):
        sql = dedent(
            """
            CREATE FOREIGN TABLE IF NOT EXISTS SDB.S01_VAFW54_EXT (
                V54NODE VARCHAR2(54)
            )
            SERVER gsmpp_server
            OPTIONS (
              LOCATION '/FCB_DATA/20240101/S01_VAFW54.txt',
              FORMAT 'TEXT',
              MODE 'NORMAL',
              ENCODING 'UTF8'
            )
            READ ONLY
            LOG INTO SDB.S01_VAFW54_ERR PER NODE REJECT LIMIT 'unlimited'
            """
        ).strip()
        expected = (
            "CREATE FOREIGN TABLE IF NOT EXISTS SDB.S01_VAFW54_EXT (V54NODE VARCHAR(54)) "
            "SERVER gsmpp_server LOCATION '/FCB_DATA/20240101/S01_VAFW54.txt' READ ONLY LOG INTO "
            "SDB.S01_VAFW54_ERR PER NODE REJECT LIMIT 'unlimited' WITH (FORMAT='TEXT', MODE='NORMAL', ENCODING='UTF8')"
        )
        expr = self.validate_identity(sql, write_sql=expected)
        expr.assert_is(exp.Create)
    
    def test_create_foreign_table_with_reject_limit_rows(self):
        sql = dedent(
            """
            CREATE FOREIGN TABLE t (
                c INT
            )
            SERVER s
            OPTIONS (FORMAT 'TEXT')
            LOG INTO err PER NODE REJECT LIMIT 10 ROWS
            """
        ).strip()
        expr = self.validate_identity(
            sql,
            write_sql="CREATE FOREIGN TABLE t (c INT4) SERVER s LOG INTO err PER NODE REJECT LIMIT 10 ROWS WITH (FORMAT='TEXT')",
        )
        prop = expr.assert_is(exp.Create).args["properties"].find(exp.PerNodeRejectLimitProperty)
        self.assertTrue(prop.args.get("rows"))

    def test_partition_by_list(self):
        sql = "CREATE TABLE t PARTITION BY LIST (c) (PARTITION p1 VALUES IN (1, 2))"
        expected = "CREATE TABLE t PARTITION BY LIST (c) (PARTITION p1 VALUES IN (1, 2))"
        expr = self.validate_identity(sql, write_sql=expected)
        properties = expr.args.get("properties")
        partition_prop = properties and properties.find(exp.PartitionByListProperty)
        self.assertIsNotNone(partition_prop)

    def test_partition_by_range_values_less_than(self):
        sql = "CREATE TABLE t PARTITION BY RANGE (c) (PARTITION p1 VALUES LESS THAN (10))"
        expr = self.validate_identity(sql)
        part_range = expr.find(exp.PartitionRange)
        self.assertIsNotNone(part_range)
        self.assertEqual(sql, expr.sql(dialect=self.dialect))
