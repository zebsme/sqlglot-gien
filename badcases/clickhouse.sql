-- ----------------------------
-- TABLE NAME : TT1_edit
-- CREATE DATE : 2025-05-16 16:21:33
-- ----------------------------
CREATE TABLE IF NOT EXISTS TT1_edit (
  fieldName1 String,
  fieldName2 FixedString(256),
  fieldName3_edit DateTime,
  fieldName4 Date,
  fieldName5 Float64,
  fieldName6 Float32,
  fieldName7 UInt32,
  fieldName8 String,
  fieldName9 String,
  fieldName10 String,
  fieldName11 Date,
  fieldName12 Float64,
  fieldName13 String,
  fieldName14 String,
  fieldName15 String,
  fieldName16 UInt16,
  fieldName17 UInt64,
  fieldName18 String,
  fieldName19 UInt8,
  fieldName20 UInt32,
  fieldName21 DateTime64(3),
  fieldName22 String, 
  fieldName23 String,
  fieldName25 Float32,
  INDEX A1(fieldName1) TYPE minmax GRANULARITY 8192
)
ENGINE = ReplicatedMergeTree('/clickhouse/tables/{shard}/default/TT1_edit', '{replica}')
PRIMARY KEY (fieldName1, fieldName2)
ORDER BY (fieldName1)
PARTITION BY toYYYYMM(fieldName3_edit)
SETTINGS index_granularity = 8192;

-- 添加列注释
ALTER TABLE TT1_edit COMMENT COLUMN fieldName1 '字段1';
ALTER TABLE TT1_edit COMMENT COLUMN fieldName2 '字段2';
ALTER TABLE TT1_edit COMMENT COLUMN fieldName3_edit '字段3修改';
ALTER TABLE TT1_edit COMMENT COLUMN fieldName4 '字段4';
ALTER TABLE TT1_edit COMMENT COLUMN fieldName5 '字段5';
ALTER TABLE TT1_edit COMMENT COLUMN fieldName6 '字段6';
ALTER TABLE TT1_edit COMMENT COLUMN fieldName7 '字段7';
ALTER TABLE TT1_edit COMMENT COLUMN fieldName8 '字段8';
ALTER TABLE TT1_edit COMMENT COLUMN fieldName9 '字段9';
ALTER TABLE TT1_edit COMMENT COLUMN fieldName10 '字段10';
ALTER TABLE TT1_edit COMMENT COLUMN fieldName11 '字段11';
ALTER TABLE TT1_edit COMMENT COLUMN fieldName12 '字段12';
ALTER TABLE TT1_edit COMMENT COLUMN fieldName13 '字段13';
ALTER TABLE TT1_edit COMMENT COLUMN fieldName14 '字段14';
ALTER TABLE TT1_edit COMMENT COLUMN fieldName15 '字段15';
ALTER TABLE TT1_edit COMMENT COLUMN fieldName16 '字段16';
ALTER TABLE TT1_edit COMMENT COLUMN fieldName17 '字段17';
ALTER TABLE TT1_edit COMMENT COLUMN fieldName18 '字段18';
ALTER TABLE TT1_edit COMMENT COLUMN fieldName19 '字段19';
ALTER TABLE TT1_edit COMMENT COLUMN fieldName20 '字段20';
ALTER TABLE TT1_edit COMMENT COLUMN fieldName21 '字段21';
ALTER TABLE TT1_edit COMMENT COLUMN fieldName22 '字段22'; 
ALTER TABLE TT1_edit COMMENT COLUMN fieldName23 '字段23';
ALTER TABLE TT1_edit COMMENT COLUMN fieldName25 '字段25';