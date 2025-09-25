-- ofrs.dashboard_board definition

-- drop table if exists

drop table if exists ofrs.dashboard_board;

CREATE TABLE ofrs.dashboard_board (
    board_id varchar(32) NOT NULL, -- 看板ID
    username text NULL, -- 用户名
    category_id varchar(32) NULL DEFAULT NULL::character varying, -- 看板分类ID
    board_name varchar(100) NULL DEFAULT NULL::character varying, -- 看板名称
    layout_json text NULL, -- 看板设计数据
    create_time timestamp NULL DEFAULT now(), -- 创建时间
    update_time timestamp NULL DEFAULT now(), -- 最后更新时间
    CONSTRAINT dashboard_board_pkey PRIMARY KEY (board_id)
)
WITH (
    orientation=row,
    compression=no,
    storage_type=USTORE,
    segment=off
);

COMMENT ON TABLE ofrs.dashboard_board IS '看板设计表';

-- Column comments
COMMENT ON COLUMN ofrs.dashboard_board.board_id IS '看板ID';
COMMENT ON COLUMN ofrs.dashboard_board.username IS '用户名';
COMMENT ON COLUMN ofrs.dashboard_board.category_id IS '看板分类ID';
COMMENT ON COLUMN ofrs.dashboard_board.board_name IS '看板名称';
COMMENT ON COLUMN ofrs.dashboard_board.layout_json IS '看板设计数据';
COMMENT ON COLUMN ofrs.dashboard_board.create_time IS '创建时间';
COMMENT ON COLUMN ofrs.dashboard_board.update_time IS '最后更新时间';

-- ofrs.dashboard_board definition

-- drop table if exists

drop table if exists ofrs.dashboard_board;

CREATE TABLE ofrs.dashboard_board (
    board_id varchar(32) NOT NULL, -- 看板ID
    username text NULL, -- 用户名
    category_id varchar(32) NULL DEFAULT NULL::character varying, -- 看板分类ID
    board_name varchar(100) NULL DEFAULT NULL::character varying, -- 看板名称
    layout_json text NULL, -- 看板设计数据
    create_time timestamp NULL DEFAULT now(), -- 创建时间
    update_time timestamp NULL DEFAULT now(), -- 最后更新时间
    CONSTRAINT dashboard_board_pkey PRIMARY KEY (board_id)
)
WITH (
    orientation=row,
    compression=no,
    storage_type=USTORE,
    segment=off
);

COMMENT ON TABLE ofrs.dashboard_board IS '看板设计表';

-- Column comments
COMMENT ON COLUMN ofrs.dashboard_board.board_id IS '看板ID';
COMMENT ON COLUMN ofrs.dashboard_board.username IS '用户名';
COMMENT ON COLUMN ofrs.dashboard_board.category_id IS '看板分类ID';
COMMENT ON COLUMN ofrs.dashboard_board.board_name IS '看板名称';
COMMENT ON COLUMN ofrs.dashboard_board.layout_json IS '看板设计数据';
COMMENT ON COLUMN ofrs.dashboard_board.create_time IS '创建时间';
COMMENT ON COLUMN ofrs.dashboard_board.update_time IS '最后更新时间';

-- ofrs.dashboard_board_param definition

-- drop table if exists

-- drop table if exists ofrs.dashboard_board_param;

CREATE TABLE ofrs.dashboard_board_param (
    board_param_id varchar(32) NOT NULL, -- 看板参数ID
    username text NULL, -- 用户名
    board_id varchar(32) NULL DEFAULT NULL::character varying, -- 看板ID
    config text NULL, -- 参数配置数据
    CONSTRAINT dashboard_board_param_pkey PRIMARY KEY (board_param_id)
)
WITH (
    orientation=row,
    compression=no,
    storage_type=USTORE,
    segment=off
);

COMMENT ON TABLE ofrs.dashboard_board_param IS '看板参数表';

-- Column comments
COMMENT ON COLUMN ofrs.dashboard_board_param.board_param_id IS '看板参数ID';
COMMENT ON COLUMN ofrs.dashboard_board_param.username IS '用户名';
COMMENT ON COLUMN ofrs.dashboard_board_param.board_id IS '看板ID';
COMMENT ON COLUMN ofrs.dashboard_board_param.config IS '参数配置数据';

-- ofrs.dashboard_board_publish definition

-- drop table if exists

-- drop table if exists ofrs.dashboard_board_publish;

CREATE TABLE ofrs.dashboard_board_publish (
    board_id varchar(32) NOT NULL, -- 看板id
    publish_type bpchar(1) NOT NULL, -- 发布类型 1:发布菜单；2:发布组件
    publish_key varchar(64) NOT NULL, -- 发布后对应的主键
    publish_time bpchar(19) NULL DEFAULT NULL::bpchar, -- 发布时间
    publish_by varchar(32) NULL DEFAULT NULL::character varying, -- 发布人
    CONSTRAINT dashboard_board_publish_pkey PRIMARY KEY (board_id, publish_type, publish_key)
)
WITH (
    orientation=row,
    compression=no,
    storage_type=USTORE,
    segment=off
);

COMMENT ON TABLE ofrs.dashboard_board_publish IS '看板发布表';

-- Column comments
COMMENT ON COLUMN ofrs.dashboard_board_publish.board_id IS '看板id';
COMMENT ON COLUMN ofrs.dashboard_board_publish.publish_type IS '发布类型 1:发布菜单；2:发布组件';
COMMENT ON COLUMN ofrs.dashboard_board_publish.publish_key IS '发布后对应的主键';
COMMENT ON COLUMN ofrs.dashboard_board_publish.publish_time IS '发布时间';
COMMENT ON COLUMN ofrs.dashboard_board_publish.publish_by IS '发布人';

-- ofrs.dashboard_board_url definition

-- drop table if exists

-- drop table if exists ofrs.dashboard_board_url;

CREATE TABLE ofrs.dashboard_board_url (
    board_id varchar(32) NULL DEFAULT NULL::character varying, -- 看板ID
    board_url varchar(100) NULL DEFAULT NULL::character varying -- 看板路
)
WITH (
    orientation=row,
    compression=no,
    storage_type=USTORE,
    segment=off
);

COMMENT ON TABLE ofrs.dashboard_board_url IS '看板发布信息表';

-- Column comments
COMMENT ON COLUMN ofrs.dashboard_board_url.board_id IS '看板ID';
COMMENT ON COLUMN ofrs.dashboard_board_url.board_url IS '看板路';

-- ofrs.dashboard_category definition

-- drop table if exists

-- drop table if exists ofrs.dashboard_category;

CREATE TABLE ofrs.dashboard_category (
    category_id varchar(32) NOT NULL, -- 分类ID
    category_name varchar(100) NULL DEFAULT NULL::character varying, -- 分类名称
    username text NULL, -- 用户名
    CONSTRAINT dashboard_category_pkey PRIMARY KEY (category_id)
)
WITH (
    orientation=row,
    compression=no,
    storage_type=USTORE,
    segment=off
);

COMMENT ON TABLE ofrs.dashboard_category IS '看板分类表';

-- Column comments
COMMENT ON COLUMN ofrs.dashboard_category.category_id IS '分类ID';
COMMENT ON COLUMN ofrs.dashboard_category.category_name IS '分类名称';
COMMENT ON COLUMN ofrs.dashboard_category.username IS '用户名';

-- ofrs.dashboard_dataset definition

-- drop table if exists

-- drop table if exists ofrs.dashboard_dataset;

CREATE TABLE ofrs.dashboard_dataset (
    dataset_id varchar(32) NOT NULL, -- 数据集ID
    username text NULL, -- 用户名
    category_name varchar(100) NULL DEFAULT NULL::character varying, -- 分类名称
    dataset_name varchar(100) NULL DEFAULT NULL::character varying, -- 数据集名称
    data_json text NULL, -- 数据集配置信息
    create_time timestamp NULL DEFAULT now(), -- 创建时间
    update_time timestamp NULL DEFAULT now(), -- 最后更新时间
    CONSTRAINT dashboard_dataset_pkey PRIMARY KEY (dataset_id)
)
WITH (
    orientation=row,
    compression=no,
    storage_type=USTORE,
    segment=off
);

COMMENT ON TABLE ofrs.dashboard_dataset IS '数据集管理表';

-- Column comments
COMMENT ON COLUMN ofrs.dashboard_dataset.dataset_id IS '数据集ID';
COMMENT ON COLUMN ofrs.dashboard_dataset.username IS '用户名';
COMMENT ON COLUMN ofrs.dashboard_dataset.category_name IS '分类名称';
COMMENT ON COLUMN ofrs.dashboard_dataset.dataset_name IS '数据集名称';
COMMENT ON COLUMN ofrs.dashboard_dataset.data_json IS '数据集配置信息';
COMMENT ON COLUMN ofrs.dashboard_dataset.create_time IS '创建时间';
COMMENT ON COLUMN ofrs.dashboard_dataset.update_time IS '最后更新时间';

-- ofrs.dashboard_datasource definition

-- drop table if exists

-- drop table if exists ofrs.dashboard_datasource;

CREATE TABLE ofrs.dashboard_datasource (
    datasource_id varchar(32) NOT NULL, -- 数据源ID
    username text NULL, -- 用户名
    source_name varchar(100) NULL DEFAULT NULL::character varying, -- 数据源名称
    source_type varchar(100) NULL DEFAULT NULL::character varying, -- 数据源类型
    config text NULL, -- 数据源配置信息
    create_time timestamp NULL DEFAULT now(), -- 创建时间
    update_time timestamp NULL DEFAULT now(), -- 最后更新时间
    CONSTRAINT dashboard_datasource_pkey PRIMARY KEY (datasource_id)
)
WITH (
    orientation=row,
    compression=no,
    storage_type=USTORE,
    segment=off
);

COMMENT ON TABLE ofrs.dashboard_datasource IS '数据源管理表';

-- Column comments
COMMENT ON COLUMN ofrs.dashboard_datasource.datasource_id IS '数据源ID';
COMMENT ON COLUMN ofrs.dashboard_datasource.username IS '用户名';
COMMENT ON COLUMN ofrs.dashboard_datasource.source_name IS '数据源名称';
COMMENT ON COLUMN ofrs.dashboard_datasource.source_type IS '数据源类型';
COMMENT ON COLUMN ofrs.dashboard_datasource.config IS '数据源配置信息';
COMMENT ON COLUMN ofrs.dashboard_datasource.create_time IS '创建时间';
COMMENT ON COLUMN ofrs.dashboard_datasource.update_time IS '最后更新时间';

-- ofrs.dashboard_job definition

-- drop table if exists

-- drop table if exists ofrs.dashboard_job;

CREATE TABLE ofrs.dashboard_job (
    job_id varchar(32) NOT NULL,
    job_name varchar(200) NULL DEFAULT NULL::character varying, -- 任务名称
    cron_exp varchar(200) NULL DEFAULT NULL::character varying, -- 执行周期表达式
    start_date timestamp NULL, -- 开始时间
    end_date timestamp NULL, -- 结束时间
    job_type varchar(200) NULL DEFAULT NULL::character varying, -- 任务类型
    job_config text NULL, -- 任务配置
    username text NULL, -- 用户名
    last_exec_time timestamp NULL, -- 最后一次函数执行时间
    job_status numeric(65, 30) NULL DEFAULT NULL::numeric, -- 任务状态
    exec_log text NULL, -- 执行日志
    CONSTRAINT dashboard_job_pkey PRIMARY KEY (job_id)
)
WITH (
    orientation=row,
    compression=no,
    storage_type=USTORE,
    segment=off
);

COMMENT ON TABLE ofrs.dashboard_job IS '管理驾驶舱定时任务表';

-- Column comments
COMMENT ON COLUMN ofrs.dashboard_job.job_id IS '任务ID';
COMMENT ON COLUMN ofrs.dashboard_job.job_name IS '任务名称';
COMMENT ON COLUMN ofrs.dashboard_job.cron_exp IS '执行周期表达式';
COMMENT ON COLUMN ofrs.dashboard_job.start_date IS '开始时间';
COMMENT ON COLUMN ofrs.dashboard_job.end_date IS '结束时间';
COMMENT ON COLUMN ofrs.dashboard_job.job_type IS '任务类型';
COMMENT ON COLUMN ofrs.dashboard_job.job_config IS '任务配置';
COMMENT ON COLUMN ofrs.dashboard_job.username IS '用户名';
COMMENT ON COLUMN ofrs.dashboard_job.last_exec_time IS '最后一次函数执行时间';
COMMENT ON COLUMN ofrs.dashboard_job.job_status IS '任务状态';
COMMENT ON COLUMN ofrs.dashboard_job.exec_log IS '执行日志';

-- ofrs.dashboard_board definition

-- drop table if exists

drop table if exists ofrs.dashboard_board;

CREATE TABLE ofrs.dashboard_board (
    board_id varchar(32) NOT NULL, -- 看板ID
    username text NULL, -- 用户名
    category_id varchar(32) NULL DEFAULT NULL::character varying, -- 看板分类ID
    board_name varchar(100) NULL DEFAULT NULL::character varying, -- 看板名称
    layout_json text NULL, -- 看板设计数据
    create_time timestamp NULL DEFAULT now(), -- 创建时间
    update_time timestamp NULL DEFAULT now(), -- 最后更新时间
    CONSTRAINT dashboard_board_pkey PRIMARY KEY (board_id)
)
WITH (
    orientation=row,
    compression=no,
    storage_type=USTORE,
    segment=off
);

COMMENT ON TABLE ofrs.dashboard_board IS '看板设计表';

-- Column comments
COMMENT ON COLUMN ofrs.dashboard_board.board_id IS '看板ID';
COMMENT ON COLUMN ofrs.dashboard_board.username IS '用户名';
COMMENT ON COLUMN ofrs.dashboard_board.category_id IS '看板分类ID';
COMMENT ON COLUMN ofrs.dashboard_board.board_name IS '看板名称';
COMMENT ON COLUMN ofrs.dashboard_board.layout_json IS '看板设计数据';
COMMENT ON COLUMN ofrs.dashboard_board.create_time IS '创建时间';
COMMENT ON COLUMN ofrs.dashboard_board.update_time IS '最后更新时间';

-- ofrs.dashboard_board_param definition

-- drop table if exists

-- drop table if exists ofrs.dashboard_board_param;

CREATE TABLE ofrs.dashboard_board_param (
    board_param_id varchar(32) NOT NULL, -- 看板参数ID
    username text NULL, -- 用户名
    board_id varchar(32) NULL DEFAULT NULL::character varying, -- 看板ID
    config text NULL, -- 参数配置数据
    CONSTRAINT dashboard_board_param_pkey PRIMARY KEY (board_param_id)
)
WITH (
    orientation=row,
    compression=no,
    storage_type=USTORE,
    segment=off
);

COMMENT ON TABLE ofrs.dashboard_board_param IS '看板参数表';

-- Column comments
COMMENT ON COLUMN ofrs.dashboard_board_param.board_param_id IS '看板参数ID';
COMMENT ON COLUMN ofrs.dashboard_board_param.username IS '用户名';
COMMENT ON COLUMN ofrs.dashboard_board_param.board_id IS '看板ID';
COMMENT ON COLUMN ofrs.dashboard_board_param.config IS '参数配置数据';

-- ofrs.dashboard_board_publish definition

-- drop table if exists

-- drop table if exists ofrs.dashboard_board_publish;

CREATE TABLE ofrs.dashboard_board_publish (
    board_id varchar(32) NOT NULL, -- 看板id
    publish_type bpchar(1) NOT NULL, -- 发布类型 1:发布菜单；2:发布组件
    publish_key varchar(64) NOT NULL, -- 发布后对应的主键
    publish_time bpchar(19) NULL DEFAULT NULL::bpchar, -- 发布时间
    publish_by varchar(32) NULL DEFAULT NULL::character varying, -- 发布人
    CONSTRAINT dashboard_board_publish_pkey PRIMARY KEY (board_id, publish_type, publish_key)
)
WITH (
    orientation=row,
    compression=no,
    storage_type=USTORE,
    segment=off
);

COMMENT ON TABLE ofrs.dashboard_board_publish IS '看板发布表';

-- Column comments
COMMENT ON COLUMN ofrs.dashboard_board_publish.board_id IS '看板id';
COMMENT ON COLUMN ofrs.dashboard_board_publish.publish_type IS '发布类型 1:发布菜单；2:发布组件';
COMMENT ON COLUMN ofrs.dashboard_board_publish.publish_key IS '发布后对应的主键';
COMMENT ON COLUMN ofrs.dashboard_board_publish.publish_time IS '发布时间';
COMMENT ON COLUMN ofrs.dashboard_board_publish.publish_by IS '发布人';

-- ofrs.dashboard_board_url definition

-- drop table if exists

-- drop table if exists ofrs.dashboard_board_url;

CREATE TABLE ofrs.dashboard_board_url (
    board_id varchar(32) NULL DEFAULT NULL::character varying, -- 看板ID
    board_url varchar(100) NULL DEFAULT NULL::character varying -- 看板路
)
WITH (
    orientation=row,
    compression=no,
    storage_type=USTORE,
    segment=off
);

COMMENT ON TABLE ofrs.dashboard_board_url IS '看板发布信息表';

-- Column comments
COMMENT ON COLUMN ofrs.dashboard_board_url.board_id IS '看板ID';
COMMENT ON COLUMN ofrs.dashboard_board_url.board_url IS '看板路';

-- ofrs.dashboard_category definition

-- drop table if exists

-- drop table if exists ofrs.dashboard_category;

CREATE TABLE ofrs.dashboard_category (
    category_id varchar(32) NOT NULL, -- 分类ID
    category_name varchar(100) NULL DEFAULT NULL::character varying, -- 分类名称
    username text NULL, -- 用户名
    CONSTRAINT dashboard_category_pkey PRIMARY KEY (category_id)
)
WITH (
    orientation=row,
    compression=no,
    storage_type=USTORE,
    segment=off
);

COMMENT ON TABLE ofrs.dashboard_category IS '看板分类表';

-- Column comments
COMMENT ON COLUMN ofrs.dashboard_category.category_id IS '分类ID';
COMMENT ON COLUMN ofrs.dashboard_category.category_name IS '分类名称';
COMMENT ON COLUMN ofrs.dashboard_category.username IS '用户名';

-- ofrs.dashboard_dataset definition

-- drop table if exists

-- drop table if exists ofrs.dashboard_dataset;

CREATE TABLE ofrs.dashboard_dataset (
    dataset_id varchar(32) NOT NULL, -- 数据集ID
    username text NULL, -- 用户名
    category_name varchar(100) NULL DEFAULT NULL::character varying, -- 分类名称
    dataset_name varchar(100) NULL DEFAULT NULL::character varying, -- 数据集名称
    data_json text NULL, -- 数据集配置信息
    create_time timestamp NULL DEFAULT now(), -- 创建时间
    update_time timestamp NULL DEFAULT now(), -- 最后更新时间
    CONSTRAINT dashboard_dataset_pkey PRIMARY KEY (dataset_id)
)
WITH (
    orientation=row,
    compression=no,
    storage_type=USTORE,
    segment=off
);

COMMENT ON TABLE ofrs.dashboard_dataset IS '数据集管理表';

-- Column comments
COMMENT ON COLUMN ofrs.dashboard_dataset.dataset_id IS '数据集ID';
COMMENT ON COLUMN ofrs.dashboard_dataset.username IS '用户名';
COMMENT ON COLUMN ofrs.dashboard_dataset.category_name IS '分类名称';
COMMENT ON COLUMN ofrs.dashboard_dataset.dataset_name IS '数据集名称';
COMMENT ON COLUMN ofrs.dashboard_dataset.data_json IS '数据集配置信息';
COMMENT ON COLUMN ofrs.dashboard_dataset.create_time IS '创建时间';
COMMENT ON COLUMN ofrs.dashboard_dataset.update_time IS '最后更新时间';

-- ofrs.dashboard_datasource definition

-- drop table if exists

-- drop table if exists ofrs.dashboard_datasource;

CREATE TABLE ofrs.dashboard_datasource (
    datasource_id varchar(32) NOT NULL, -- 数据源ID
    username text NULL, -- 用户名
    source_name varchar(100) NULL DEFAULT NULL::character varying, -- 数据源名称
    source_type varchar(100) NULL DEFAULT NULL::character varying, -- 数据源类型
    config text NULL, -- 数据源配置信息
    create_time timestamp NULL DEFAULT now(), -- 创建时间
    update_time timestamp NULL DEFAULT now(), -- 最后更新时间
    CONSTRAINT dashboard_datasource_pkey PRIMARY KEY (datasource_id)
)
WITH (
    orientation=row,
    compression=no,
    storage_type=USTORE,
    segment=off
);

COMMENT ON TABLE ofrs.dashboard_datasource IS '数据源管理表';

-- Column comments
COMMENT ON COLUMN ofrs.dashboard_datasource.datasource_id IS '数据源ID';
COMMENT ON COLUMN ofrs.dashboard_datasource.username IS '用户名';
COMMENT ON COLUMN ofrs.dashboard_datasource.source_name IS '数据源名称';
COMMENT ON COLUMN ofrs.dashboard_datasource.source_type IS '数据源类型';
COMMENT ON COLUMN ofrs.dashboard_datasource.config IS '数据源配置信息';
COMMENT ON COLUMN ofrs.dashboard_datasource.create_time IS '创建时间';
COMMENT ON COLUMN ofrs.dashboard_datasource.update_time IS '最后更新时间';

-- ofrs.dashboard_job definition

-- drop table if exists

-- drop table if exists ofrs.dashboard_job;

CREATE TABLE ofrs.dashboard_job (
    job_id varchar(32) NOT NULL,
    job_name varchar(200) NULL DEFAULT NULL::character varying, -- 任务名称
    cron_exp varchar(200) NULL DEFAULT NULL::character varying, -- 执行周期表达式
    start_date timestamp NULL, -- 开始时间
    end_date timestamp NULL, -- 结束时间
    job_type varchar(200) NULL DEFAULT NULL::character varying, -- 任务类型
    job_config text NULL, -- 任务配置
    username text NULL, -- 用户名
    last_exec_time timestamp NULL, -- 最后一次函数执行时间
    job_status numeric(65, 30) NULL DEFAULT NULL::numeric, -- 任务状态
    exec_log text NULL, -- 执行日志
    CONSTRAINT dashboard_job_pkey PRIMARY KEY (job_id)
)
WITH (
    orientation=row,
    compression=no,
    storage_type=USTORE,
    segment=off
);

COMMENT ON TABLE ofrs.dashboard_job IS '管理驾驶舱定时任务表';

-- Column comments
COMMENT ON COLUMN ofrs.dashboard_job.job_id IS '任务ID';
COMMENT ON COLUMN ofrs.dashboard_job.job_name IS '任务名称';
COMMENT ON COLUMN ofrs.dashboard_job.cron_exp IS '执行周期表达式';
COMMENT ON COLUMN ofrs.dashboard_job.start_date IS '开始时间';
COMMENT ON COLUMN ofrs.dashboard_job.end_date IS '结束时间';
COMMENT ON COLUMN ofrs.dashboard_job.job_type IS '任务类型';
COMMENT ON COLUMN ofrs.dashboard_job.job_config IS '任务配置';
COMMENT ON COLUMN ofrs.dashboard_job.username IS '用户名';
COMMENT ON COLUMN ofrs.dashboard_job.last_exec_time IS '最后一次函数执行时间';
COMMENT ON COLUMN ofrs.dashboard_job.job_status IS '任务状态';
COMMENT ON COLUMN ofrs.dashboard_job.exec_log IS '执行日志';

-- ofrs.dashboard_role definition

-- drop table if exists

-- drop table if exists ofrs.dashboard_role;

CREATE TABLE ofrs.dashboard_role (
    role_id varchar(100) NOT NULL, -- 角色ID
    role_name varchar(100) NULL DEFAULT NULL::character varying, -- 角色名称
    user_id varchar(50) NULL DEFAULT NULL::character varying, -- 用户ID
    CONSTRAINT dashboard_role_pkey PRIMARY KEY (role_id)
)
WITH (
    orientation=row,
    compression=no,
    storage_type=USTORE,
    segment=off
);

COMMENT ON TABLE ofrs.dashboard_role IS '管理驾驶舱角色表';

-- Column comments
COMMENT ON COLUMN ofrs.dashboard_role.role_id IS '角色ID';
COMMENT ON COLUMN ofrs.dashboard_role.role_name IS '角色名称';
COMMENT ON COLUMN ofrs.dashboard_role.user_id IS '用户ID';

-- ofrs.dashboard_role_res definition

-- drop table if exists

-- drop table if exists ofrs.dashboard_role_res;

CREATE TABLE ofrs.dashboard_role_res (
    role_res_id varchar(32) NOT NULL, -- 角色资源ID
    role_id varchar(100) NULL DEFAULT NULL::character varying, -- 角色ID
    res_type varchar(100) NULL DEFAULT NULL::character varying, -- 资源类型
    res_id varchar(32) NULL DEFAULT NULL::character varying, -- 资源ID
    "permission" varchar(20) NULL DEFAULT NULL::character varying, -- 资源权限
    CONSTRAINT dashboard_role_res_pkey PRIMARY KEY (role_res_id)
)
WITH (
    orientation=row,
    compression=no,
    storage_type=USTORE,
    segment=off
);

COMMENT ON TABLE ofrs.dashboard_role_res IS '管理驾驶舱角色资源表';

-- Column comments
COMMENT ON COLUMN ofrs.dashboard_role_res.role_res_id IS '角色资源ID';
COMMENT ON COLUMN ofrs.dashboard_role_res.role_id IS '角色ID';
COMMENT ON COLUMN ofrs.dashboard_role_res.res_type IS '资源类型';
COMMENT ON COLUMN ofrs.dashboard_role_res.res_id IS '资源ID';
COMMENT ON COLUMN ofrs.dashboard_role_res."permission" IS '资源权限';

-- ofrs.dashboard_user definition

-- drop table if exists

-- drop table if exists ofrs.dashboard_user;

CREATE TABLE ofrs.dashboard_user (
    user_id varchar(50) NOT NULL, -- 用户ID
    login_name varchar(100) NULL DEFAULT NULL::character varying, -- 登录名
    user_name varchar(100) NULL DEFAULT NULL::character varying, -- 用户名
    user_password varchar(100) NULL DEFAULT NULL::character varying, -- 用户密码
    user_status varchar(100) NULL DEFAULT NULL::character varying, -- 用户状态
    CONSTRAINT dashboard_user_pkey PRIMARY KEY (user_id)
)
WITH (
    orientation=row,
    compression=no,
    storage_type=USTORE,
    segment=off
);

COMMENT ON TABLE ofrs.dashboard_user IS '管理驾驶舱用户表';

-- Column comments
COMMENT ON COLUMN ofrs.dashboard_user.user_id IS '用户ID';
COMMENT ON COLUMN ofrs.dashboard_user.login_name IS '登录名';
COMMENT ON COLUMN ofrs.dashboard_user.user_name IS '用户名';
COMMENT ON COLUMN ofrs.dashboard_user.user_password IS '用户密码';
COMMENT ON COLUMN ofrs.dashboard_user.user_status IS '用户状态';
