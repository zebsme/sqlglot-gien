from sqlglot import exp
from tests.dialects.test_dialect import Validator


class TestLindorm(Validator):
    dialect = "lindorm"

    def test_ddl(self):
        create = self.validate_identity(
            """
            CREATE TABLE test.T2 (
              fieldName1 STRING,
              fieldName2 STRING,
              fieldName3 STRING,
              fieldName4 TIMESTAMP,
              fieldName5 DECIMAL(6, 2),
              fieldName6 FLOAT,
              fieldName7 INTEGER,
              fieldName8 BINARY,
              fieldName9 STRING,
              fieldName10 STRING,
              fieldName11 DATE,
              fieldName12 DOUBLE,
              fieldName13 STRING,
              fieldName14 STRING,
              fieldName15 STRING,
              fieldName16 INTEGER,
              fieldName17 BIGINT,
              fieldName18 VARBINARY,
              fieldName19 SMALLINT,
              fieldName20 SMALLINT,
              fieldName21 TIMESTAMP,
              fieldName22 STRING,
              fieldName23 STRING,
              fieldName24 STRING,
              fieldName25 STRING
            )
            WITH (
              TTL='31536000',
              COMPRESSION='SNAPPY',
              MAX_VERSIONS='3',
              BLOOMFILTER='ROW',
              DATA_BLOCK_ENCODING='PREFIX'
            )
            """,
            "CREATE TABLE test.T2 (fieldName1 STRING, fieldName2 STRING, fieldName3 STRING, fieldName4 TIMESTAMP, fieldName5 DECIMAL(6, 2), fieldName6 FLOAT, fieldName7 INTEGER, fieldName8 BINARY, fieldName9 STRING, fieldName10 STRING, fieldName11 DATE, fieldName12 DOUBLE, fieldName13 STRING, fieldName14 STRING, fieldName15 STRING, fieldName16 INTEGER, fieldName17 BIGINT, fieldName18 VARBINARY, fieldName19 SMALLINT, fieldName20 SMALLINT, fieldName21 TIMESTAMP, fieldName22 STRING, fieldName23 STRING, fieldName24 STRING, fieldName25 STRING) WITH (TTL='31536000', COMPRESSION='SNAPPY', MAX_VERSIONS='3', BLOOMFILTER='ROW', DATA_BLOCK_ENCODING='PREFIX')",
        )

        ttl = create.args["properties"].expressions[0]
        ttl.assert_is(exp.Property)
        self.assertEqual(ttl.name, "TTL")

        self.validate_identity(
            "CREATE INDEX IF NOT EXISTS A2 USING SEARCH ON test.T2 (fieldName1)",
            "CREATE INDEX IF NOT EXISTS A2 USING SEARCH ON test.T2(fieldName1)",
        )
        self.validate_identity(
            "CREATE INDEX A1 ON test.T2 (fieldName4)",
            "CREATE INDEX A1 ON test.T2(fieldName4)",
        )
