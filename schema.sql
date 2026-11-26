-- STRICT tables reject storage-class drift. ANY preserves integer, real, text, and BLOB values.
-- Enable on every SQLite connection; SQLite does not enable foreign keys by default.
PRAGMA foreign_keys = ON;

-- One concrete initialized Qwen3MLP. Each row owns one exact state_dict and config;
-- multiple test cases share this row when they exercise the same initialized module.
CREATE TABLE module_instance (
    id      TEXT PRIMARY KEY, -- Stable text identity shared by config and test-case rows.
    target  TEXT NOT NULL, -- Fully qualified class path used by the runner.
    state   BLOB NOT NULL  -- Safetensors bundle keyed by state_dict name; carries dtype/shape.
) STRICT;

-- One native SQLite scalar per config field. Store list/nested fields as indexed/path rows,
-- not serialized objects. torch_dtype records the weight precision needed for reconstruction.
CREATE TABLE module_config (
    module_id TEXT NOT NULL REFERENCES module_instance(id),
    name      TEXT NOT NULL,
    value     ANY, -- Keep numeric dimensions numeric and names such as hidden_act textual.
    PRIMARY KEY (module_id, name)
) STRICT;

-- One forward() invocation. The module_id links the case to its exact config and weights.
CREATE TABLE test_case (
    id        TEXT PRIMARY KEY,
    module_id TEXT NOT NULL REFERENCES module_instance(id)
) STRICT;

-- One row per named forward() argument. Tensor arguments use Safetensors BLOBs so dtype,
-- shape, name, and raw values survive the SQLite round trip; scalar args use SQLite scalars.
CREATE TABLE test_case_input (
    case_id TEXT NOT NULL REFERENCES test_case(id),
    name    TEXT NOT NULL,
    value   ANY NOT NULL, -- Tensor arguments are Safetensors BLOBs; scalar kwargs stay scalar.
    PRIMARY KEY (case_id, name)
) STRICT;

-- One expected result bundle per invocation; separate table keeps reference outputs out of
-- forward arguments and supports structured outputs. Comparison tolerances stay in the runner.
CREATE TABLE test_case_expected (
    case_id TEXT PRIMARY KEY REFERENCES test_case(id),
    value   BLOB NOT NULL -- Safetensors output bundle preserves names, dtype, shape, and values.
) STRICT;
