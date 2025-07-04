
# working code
import streamlit as st
from sqlalchemy import create_engine, text
import polars as pl
import duckdb
import io

# Setup
st.set_page_config(page_title="Data Comparison Tool", layout="wide")
st.title("🔍 Data Comparison Tool - File or DB vs DB")

# Connect to DuckDB in-memory
duckdiffDB = duckdb.connect(database=":memory:")

# Layout
left_col, right_col = st.columns(2)

# ---------- LEFT: Source Input (File + DB) ----------
with left_col:
    st.subheader("📂 Source Input")

    source_input_type = st.radio("Source Type", ["CSV", "JSON", "Database"], horizontal=True)

    uploaded_file = st.file_uploader("Upload Source File (optional)", type=["csv", "json"])
    source_db_url = st.text_input("Source DB URL", placeholder="e.g., sqlite:///source.db")
    source_query = st.text_area("Source Query", "SELECT * FROM employee")

# ---------- RIGHT: Target DB ----------
with right_col:
    st.subheader("🎯 Target Database")
    target_db_url = st.text_input("Target DB URL", placeholder="e.g., sqlite:///target.db")
    target_query = st.text_area("Target Query", "SELECT * FROM employee")

# ---------- Show IDs in Target but Not in Source ----------
def show_target_not_in_source():
    st.subheader("🚫 IDs in Target but Not in Source")
    try:
        source_cols = duckdiffDB.execute("PRAGMA table_info(source_table)").fetchdf()
        target_cols = duckdiffDB.execute("PRAGMA table_info(target_table)").fetchdf()

        source_col_names = set(col.strip().lower() for col in source_cols['name'])
        target_col_names = set(col.strip().lower() for col in target_cols['name'])
        common_cols = source_col_names.intersection(target_col_names)

        if not common_cols:
            st.warning("No common columns found.")
            return

        key_col = "id" if "id" in common_cols else list(common_cols)[0]
        st.markdown(f"🔑 Comparing by `{key_col}`")

        query = f"""
            SELECT {key_col}
            FROM target_table
            WHERE {key_col} NOT IN (SELECT {key_col} FROM source_table)
        """
        not_in_df = duckdiffDB.execute(query).fetchdf()
        if not not_in_df.empty:
            st.dataframe(not_in_df, use_container_width=True)
            st.success(f"Found {not_in_df.shape[0]} target IDs missing in source.")
        else:
            st.info("✅ All target IDs are present in source.")
    except Exception as e:
        st.warning(f"Error: {e}")

# ---------- Show Record-Level Differences ----------
def show_record_level_diff():
    st.subheader("🔄 Non-Matching Records")
    try:
        target_not_in_source = duckdiffDB.execute("""
            SELECT *, 'target' AS source
            FROM target_table
            EXCEPT
            SELECT *, 'target'
            FROM source_table
        """).fetchdf()

        source_not_in_target = duckdiffDB.execute("""
            SELECT *, 'source' AS source
            FROM source_table
            EXCEPT
            SELECT *, 'source'
            FROM target_table
        """).fetchdf()

        full_diff_df = pl.concat([
            pl.from_pandas(source_not_in_target),
            pl.from_pandas(target_not_in_source)
        ])

        if not full_diff_df.is_empty():
            st.dataframe(full_diff_df.to_pandas(), use_container_width=True)
            st.success(f"✅ Found {full_diff_df.shape[0]} non-matching records.")
        else:
            st.info("✅ No differences found.")
    except Exception as e:
        st.warning(f"❌ Diff error: {e}")

# ---------- Run Comparison ----------
if st.button("Run Comparison"):
    if (uploaded_file or (source_db_url.strip() and source_query.strip())) and \
       target_db_url.strip() and target_query.strip():
        try:
            duckdiffDB.execute("DROP TABLE IF EXISTS source_table")
            duckdiffDB.execute("DROP TABLE IF EXISTS target_table")

            # Load Source
            if uploaded_file:
                if source_input_type == "CSV":
                    source_df = pl.read_csv(uploaded_file)
                elif source_input_type == "JSON":
                    content = uploaded_file.read()
                    source_df = pl.read_json(io.BytesIO(content))
            elif source_db_url and source_query:
                source_engine = create_engine(source_db_url)
                with source_engine.connect() as conn:
                    source_df = pl.read_database(query=text(source_query), connection=conn)
            else:
                st.error("⚠️ Provide either a file or DB URL + query for source.")
                st.stop()

            source_df = source_df.rename({col: col.strip().lower() for col in source_df.columns})
            duckdiffDB.register("source_table", source_df)
            st.subheader("📘 Source Table Preview")
            st.dataframe(source_df.to_pandas())

            # Load Target
            target_engine = create_engine(target_db_url)
            with target_engine.connect() as conn:
                target_df = pl.read_database(query=text(target_query), connection=conn)
            target_df = target_df.rename({col: col.strip().lower() for col in target_df.columns})
            duckdiffDB.register("target_table", target_df)
            st.subheader("📙 Target Table Preview")
            st.dataframe(target_df.to_pandas())

            # Show diffs
            show_target_not_in_source()
            show_record_level_diff()

        except Exception as e:
            st.error(f"❌ Comparison failed:\n{e}")
    else:
        st.error("⚠️ Please provide all required input.")


# csv working 
import streamlit as st
from sqlalchemy import create_engine, text
import polars as pl
import duckdb
import io

# Setup
st.set_page_config(page_title="Data Comparison Tool", layout="wide")
st.title("🔍 Data Comparison Tool - File or DB vs DB")

# DuckDB in-memory instance
duckdiffDB = duckdb.connect(database=":memory:")

# Layout
left_col, right_col = st.columns(2)

# ---------- LEFT: Source Input ----------
with left_col:
    st.subheader("📂 Source Input")
    source_input_type = st.radio("Source Type", ["CSV", "JSON", "Database"], horizontal=True)
    uploaded_file = st.file_uploader("Upload Source File (optional)", type=["csv", "json"])
    source_db_url = st.text_input("Source DB URL", placeholder="e.g., sqlite:///source.db")
    source_query = st.text_area("Source Query", "SELECT * FROM source_table")

# ---------- RIGHT: Target DB ----------
with right_col:
    st.subheader("🎯 Target Database")
    target_db_url = st.text_input("Target DB URL", placeholder="e.g., sqlite:///target.db")
    target_query = st.text_area("Target Query", "SELECT * FROM target_table")

# ---------- Show Target Not in Source ----------
def show_target_not_in_source():
    st.subheader("🚫 IDs in Target but Not in Source")
    try:
        source_cols = duckdiffDB.execute("PRAGMA table_info(source_table)").fetchdf()
        target_cols = duckdiffDB.execute("PRAGMA table_info(target_table)").fetchdf()

        source_col_names = set(col.strip().lower() for col in source_cols['name'])
        target_col_names = set(col.strip().lower() for col in target_cols['name'])
        common_cols = source_col_names.intersection(target_col_names)

        if not common_cols:
            st.warning("No common columns found.")
            return

        key_col = "id" if "id" in common_cols else list(common_cols)[0]
        st.markdown(f"🔑 Comparing by `{key_col}`")

        query = f"""
            SELECT {key_col}
            FROM target_table
            WHERE {key_col} NOT IN (SELECT {key_col} FROM source_table)
        """
        not_in_df = duckdiffDB.execute(query).fetchdf()
        if not not_in_df.empty:
            st.dataframe(pl.from_pandas(not_in_df), use_container_width=True)
            st.success(f"Found {not_in_df.shape[0]} target IDs missing in source.")
        else:
            st.info("✅ All target IDs are present in source.")
    except Exception as e:
        st.warning(f"Error: {e}")

# ---------- Show Record-Level Differences ----------
def show_record_level_diff():
    st.subheader("🔄 Non-Matching Records")
    try:
        target_not_in_source = duckdiffDB.execute("""
            SELECT *, 'target' AS origin
            FROM target_table
            EXCEPT
            SELECT *, 'target'
            FROM source_table
        """).fetchdf()

        source_not_in_target = duckdiffDB.execute("""
            SELECT *, 'source' AS origin
            FROM source_table
            EXCEPT
            SELECT *, 'source'
            FROM target_table
        """).fetchdf()

        full_diff = pl.concat([
            pl.from_pandas(source_not_in_target),
            pl.from_pandas(target_not_in_source)
        ])

        if not full_diff.is_empty():
            st.dataframe(full_diff, use_container_width=True)
            st.success(f"✅ Found {full_diff.shape[0]} non-matching records.")
        else:
            st.info("✅ No differences found.")
    except Exception as e:
        st.warning(f"❌ Diff error: {e}")

# ---------- Run Comparison ----------
if st.button("Run Comparison"):
    if (uploaded_file or (source_db_url.strip() and source_query.strip())) and \
       target_db_url.strip() and target_query.strip():
        try:
            duckdiffDB.execute("DROP TABLE IF EXISTS source_table")
            duckdiffDB.execute("DROP TABLE IF EXISTS target_table")

            # Load Source
            if uploaded_file:
                if source_input_type == "CSV":
                    source_df = pl.read_csv(uploaded_file)
                elif source_input_type == "JSON":
                    content = uploaded_file.read()
                    source_df = pl.read_json(io.BytesIO(content))
                source_df = source_df.rename({col: col.strip().lower() for col in source_df.columns})
                duckdiffDB.register("source_table", source_df)
            elif source_db_url and source_query:
                source_engine = create_engine(source_db_url)
                with source_engine.connect() as conn:
                    source_df = pl.read_database(query=text(source_query), connection=conn)
                source_df = source_df.rename({col: col.strip().lower() for col in source_df.columns})
                duckdiffDB.register("source_table", source_df)
            else:
                st.error("⚠️ Provide either file or DB + query for source.")
                st.stop()

            # Preview or query result
            result_df = duckdiffDB.execute(source_query).fetchdf()
            st.subheader("📘 Source Table Preview")
            st.dataframe(pl.from_pandas(result_df), use_container_width=True)

            # Load Target
            target_engine = create_engine(target_db_url)
            with target_engine.connect() as conn:
                target_df = pl.read_database(query=text(target_query), connection=conn)
            target_df = target_df.rename({col: col.strip().lower() for col in target_df.columns})
            duckdiffDB.register("target_table", target_df)
            st.subheader("📙 Target Table Preview")
            st.dataframe(target_df, use_container_width=True)

            # Show diffs only if full table queries
            if "select *" in source_query.lower() and "select *" in target_query.lower():
                show_target_not_in_source()
                show_record_level_diff()

        except Exception as e:
            st.error(f"❌ Comparison failed:\n{e}")
    else:
        st.error("⚠️ Please provide all required input.")


# working good
    import streamlit as st
    from sqlalchemy import create_engine, text
    import polars as pl
    import duckdb
    import io

    # Setup
    st.set_page_config(page_title="Data Comparison Tool", layout="wide")
    st.title("🔍 Data Comparison Tool - File or DB vs DB")

    # DuckDB in-memory instance
    duckdiffDB = duckdb.connect(database=":memory:")

    # Layout
    left_col, right_col = st.columns(2)

    # ---------- LEFT: Source Input ----------
    with left_col:
        st.subheader("📂 Source Input")
        source_input_type = st.radio("Source Type", ["CSV", "JSON", "Database"], horizontal=True)
        uploaded_file = st.file_uploader("Upload Source File (optional)", type=["csv", "json"])
        source_db_url = st.text_input("Source DB URL", placeholder="e.g., sqlite:///source.db")
        source_query = st.text_area("Source Query", "SELECT * FROM source_table")

    # ---------- RIGHT: Target DB ----------
    with right_col:
        st.subheader("🎯 Target Database")
        target_db_url = st.text_input("Target DB URL", placeholder="e.g., sqlite:///target.db")
        target_query = st.text_area("Target Query", "SELECT * FROM target_table")

    # ---------- Show Target Not in Source ----------
    def show_target_not_in_source():
        st.subheader("🚫 IDs in Target but Not in Source")
        try:
            source_cols = duckdiffDB.execute("PRAGMA table_info(source_table)").fetchdf()
            target_cols = duckdiffDB.execute("PRAGMA table_info(target_table)").fetchdf()

            source_col_names = set(col.strip().lower() for col in source_cols['name'])
            target_col_names = set(col.strip().lower() for col in target_cols['name'])
            common_cols = source_col_names.intersection(target_col_names)

            if not common_cols:
                st.warning("No common columns found.")
                return

            key_col = "id" if "id" in common_cols else list(common_cols)[0]
            st.markdown(f"🔑 Comparing by `{key_col}`")

            query = f"""
                SELECT {key_col}
                FROM target_table
                WHERE {key_col} NOT IN (SELECT {key_col} FROM source_table)
            """
            not_in_df = duckdiffDB.execute(query).fetchdf()
            if not not_in_df.empty:
                st.dataframe(pl.from_pandas(not_in_df), use_container_width=True)
                st.success(f"Found {not_in_df.shape[0]} target IDs missing in source.")
            else:
                st.info("✅ All target IDs are present in source.")
        except Exception as e:
            st.warning(f"Error: {e}")

    # ---------- Show Record-Level Differences ----------
    def show_record_level_diff():
        st.subheader("🔄 Non-Matching Records")
        try:
            target_not_in_source = duckdiffDB.execute("""
                SELECT *, 'target' AS origin
                FROM target_table
                EXCEPT
                SELECT *, 'target'
                FROM source_table
            """).fetchdf()

            source_not_in_target = duckdiffDB.execute("""
                SELECT *, 'source' AS origin
                FROM source_table
                EXCEPT
                SELECT *, 'source'
                FROM target_table
            """).fetchdf()

            full_diff = pl.concat([
                pl.from_pandas(source_not_in_target),
                pl.from_pandas(target_not_in_source)
            ])

            if not full_diff.is_empty():
                st.dataframe(full_diff, use_container_width=True)
                st.success(f"✅ Found {full_diff.shape[0]} non-matching records.")
            else:
                st.info("✅ No differences found.")
        except Exception as e:
            st.warning(f"❌ Diff error: {e}")

    # ---------- Run Comparison ----------
    if st.button("Run Comparison"):
        if (uploaded_file or (source_db_url.strip() and source_query.strip())) and \
        target_db_url.strip() and target_query.strip():
            try:
                duckdiffDB.execute("DROP TABLE IF EXISTS source_table")
                duckdiffDB.execute("DROP TABLE IF EXISTS target_table")

                # Load Source
                if uploaded_file:
                    if source_input_type == "CSV":
                        source_df = pl.read_csv(uploaded_file)
                    elif source_input_type == "JSON":
                        content = uploaded_file.read()
                        source_df = pl.read_json(io.BytesIO(content))
                    source_df = source_df.rename({col: col.strip().lower() for col in source_df.columns})
                    duckdiffDB.register("source_table", source_df)
                elif source_db_url and source_query:
                    source_engine = create_engine(source_db_url)
                    with source_engine.connect() as conn:
                        source_df = pl.read_database(query=text(source_query), connection=conn)
                    source_df = source_df.rename({col: col.strip().lower() for col in source_df.columns})
                    duckdiffDB.register("source_table", source_df)
                else:
                    st.error("⚠️ Provide either file or DB + query for source.")
                    st.stop()

                # Preview or query result
                result_df = duckdiffDB.execute(source_query).fetchdf()
                st.subheader("📘 Source Table Preview")
                st.dataframe(pl.from_pandas(result_df), use_container_width=True)

                # Load Target
                target_engine = create_engine(target_db_url)
                with target_engine.connect() as conn:
                    target_df = pl.read_database(query=text(target_query), connection=conn)
                target_df = target_df.rename({col: col.strip().lower() for col in target_df.columns})
                duckdiffDB.register("target_table", target_df)
                st.subheader("📙 Target Table Preview")
                st.dataframe(target_df, use_container_width=True)

                # Show diffs only if full table queries
                if "select *" in source_query.lower() and "select *" in target_query.lower():
                    show_target_not_in_source()
                    show_record_level_diff()

            except Exception as e:
                st.error(f"❌ Comparison failed:\n{e}")
        else:
            st.error("⚠️ Please provide all required input.")


# all results together
        

import streamlit as st
from sqlalchemy import create_engine, text
import polars as pl
import duckdb
import io
import json

# Setup
st.set_page_config(page_title="Universal Validator", layout="wide")
st.title("🔍 DB Validation Tool (Source & Target) - Polars + DuckDB")

# DuckDB in-memory
duckdb_conn = duckdb.connect(database=":memory:")

# Layout
left, right = st.columns(2)

# Left: Source Input
with left:
    st.subheader("📂 Source Input")
    source_type = st.radio("Source Type", ["CSV", "JSON", "Database"], horizontal=True)
    uploaded_file = st.file_uploader("Upload Source File", type=["csv", "json"])
    source_db_url = st.text_input("Source DB URL", placeholder="e.g., sqlite:///source.db")
    source_query = st.text_area("Source Query", "SELECT * FROM employee")

# Right: Target Input
with right:
    st.subheader("🎯 Target Input")
    target_type = "Database"
    target_db_url = st.text_input("Target DB URL", placeholder="e.g., sqlite:///target.db")
    target_query = st.text_area("Target Query", "SELECT * FROM employee")

# ---------- Load Function ----------
def load_into_duckdb(name, file, db_url, query, input_type):
    if file:
        content = file.read()
        file.seek(0)
        if input_type == "CSV":
            df = pl.read_csv(io.BytesIO(content))
        elif input_type == "JSON":
            records = json.load(io.BytesIO(content))
            if isinstance(records, list):
                df = pl.DataFrame(records)
            else:
                st.error("JSON must be a list of objects.")
                st.stop()
        else:
            st.error("Unsupported type")
            st.stop()
    elif db_url and query:
        engine = create_engine(db_url)
        with engine.connect() as conn:
            df = pl.read_database(query=text(query), connection=conn)
    else:
        st.error("Provide valid input")
        st.stop()

    df = df.rename({col: col.strip().lower() for col in df.columns})
    duckdb_conn.register(name, df)
    return df

# ---------- Validation Function ----------
def run_validations(table_name):
    st.subheader(f"📋 Validations for `{table_name}`")

    # 1. Preview
    st.markdown("✅ Preview")
    st.dataframe(duckdb_conn.execute(f"SELECT * FROM {table_name} LIMIT 10").fetchdf(), use_container_width=True)

    # 2. Schema
    st.markdown("📐 Table Schema")
    schema_df = duckdb_conn.execute(f"PRAGMA table_info('{table_name}')").fetchdf()
    st.dataframe(schema_df, use_container_width=True)

    # 3. Self Join on name mismatch
    st.markdown("🔁 Name matches but dept differs")
    join_query = f"""
        SELECT a.id, a.name, a.dept_id 
        FROM {table_name} a 
        JOIN {table_name} b 
        ON a.name = b.name AND a.id != b.id AND a.dept_id != b.dept_id
    """
    mismatch_df = duckdb_conn.execute(join_query).fetchdf()
    st.dataframe(mismatch_df, use_container_width=True)

    # 4. Duplicate Names
    st.markdown("👥 Duplicate Names")
    dup_df = duckdb_conn.execute(f"""
        SELECT name, COUNT(*) as count 
        FROM {table_name}
        GROUP BY name
        HAVING COUNT(*) > 1
    """).fetchdf()
    st.dataframe(dup_df, use_container_width=True)

    # 5. Aggregates
    st.markdown("📊 Aggregates")
    agg_df = duckdb_conn.execute(f"""
        SELECT MAX(id) as max_id, MIN(id) as min_id, COUNT(DISTINCT dept_id) as distinct_depts 
        FROM {table_name}
    """).fetchdf()
    st.dataframe(agg_df, use_container_width=True)

# ---------- Run Button ----------
if st.button("Run Validations"):
    try:
        duckdb_conn.execute("DROP TABLE IF EXISTS source")
        duckdb_conn.execute("DROP TABLE IF EXISTS target")

        # Load source
        source_df = load_into_duckdb("source", uploaded_file, source_db_url, source_query, source_type)
        st.success("✅ Source loaded")
        run_validations("source")

        # Load and validate target if provided
        if target_db_url.strip() and target_query.strip():
            target_df = load_into_duckdb("target", None, target_db_url, target_query, "Database")
            st.success("✅ Target loaded")
            run_validations("target")

            # Optional: Show differences
            st.subheader("📍 Rows in Target NOT in Source")
            diff_query = "SELECT * FROM target EXCEPT SELECT * FROM source"
            diff_df = duckdb_conn.execute(diff_query).fetchdf()
            st.dataframe(pl.from_pandas(diff_df), use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error: {e}")


# niot working for csv / json (duplicated query)
import streamlit as st
from sqlalchemy import create_engine, text
import polars as pl
import duckdb
import io
import json

# Setup
st.set_page_config(page_title="Data Comparison Tool", layout="wide")
st.title("🔍 Data Comparison Tool - File or DB vs DB")

# In-memory DuckDB
duckdiffDB = duckdb.connect(database=":memory:")

# Layout
left_col, right_col = st.columns(2)

# ---------- LEFT: Source Input ----------
with left_col:
    st.subheader("📂 Source Input")
    source_input_type = st.radio("Source Type", ["CSV", "JSON", "Database"], horizontal=True)
    uploaded_file = st.file_uploader("Upload Source File (optional)", type=["csv", "json"])
    source_db_url = st.text_input("Source DB URL", placeholder="e.g., sqlite:///source.db")
    source_query = st.text_area("Source Query", "SELECT * FROM source_table")

# ---------- RIGHT: Target DB ----------
with right_col:
    st.subheader("🎯 Target Database")
    target_db_url = st.text_input("Target DB URL", placeholder="e.g., sqlite:///target.db")
    target_query = st.text_area("Target Query", "SELECT * FROM target_table")

# ---------- Comparison Utilities ----------
def show_target_not_in_source():
    st.subheader("🚫 IDs in Target but Not in Source")
    try:
        source_cols = duckdiffDB.execute("PRAGMA table_info(source_table)").fetchdf()
        target_cols = duckdiffDB.execute("PRAGMA table_info(target_table)").fetchdf()

        source_col_names = set(source_cols["name"].str.lower())
        target_col_names = set(target_cols["name"].str.lower())
        common_cols = source_col_names.intersection(target_col_names)

        if not common_cols:
            st.warning("No common columns found.")
            return

        key_col = "id" if "id" in common_cols else list(common_cols)[0]
        st.markdown(f"🔑 Comparing by `{key_col}`")

        query = f"""
            SELECT {key_col}
            FROM target_table
            WHERE {key_col} NOT IN (SELECT {key_col} FROM source_table)
        """
        result = duckdiffDB.execute(query).fetchdf()
        if not result.empty:
            st.dataframe(pl.from_pandas(result), use_container_width=True)
            st.success(f"✅ Found {result.shape[0]} records only in target.")
        else:
            st.info("✅ All target IDs are present in source.")
    except Exception as e:
        st.warning(f"Error: {e}")

def show_record_level_diff():
    st.subheader("🔄 Non-Matching Records")
    try:
        df1 = duckdiffDB.execute("""
            SELECT *, 'target' AS origin FROM target_table
            EXCEPT
            SELECT *, 'target' FROM source_table
        """).fetchdf()

        df2 = duckdiffDB.execute("""
            SELECT *, 'source' AS origin FROM source_table
            EXCEPT
            SELECT *, 'source' FROM target_table
        """).fetchdf()

        full_diff = pl.concat([
            pl.from_pandas(df2),
            pl.from_pandas(df1)
        ])

        if not full_diff.is_empty():
            st.dataframe(full_diff, use_container_width=True)
            st.success(f"✅ Found {full_diff.shape[0]} differing records.")
        else:
            st.info("✅ No differences found.")
    except Exception as e:
        st.warning(f"❌ Diff error: {e}")

# ---------- Run Comparison ----------
if st.button("Run Comparison"):
    if (uploaded_file or (source_db_url.strip() and source_query.strip())) and \
       target_db_url.strip() and target_query.strip():
        try:
            duckdiffDB.execute("DROP TABLE IF EXISTS source_table")
            duckdiffDB.execute("DROP TABLE IF EXISTS target_table")

            # Load Source
            if uploaded_file:
                content = uploaded_file.read()
                uploaded_file.seek(0)
                if source_input_type == "CSV":
                    source_df = pl.read_csv(io.BytesIO(content))
                elif source_input_type == "JSON":
                    records = json.load(io.BytesIO(content))
                    if not isinstance(records, list):
                        st.error("Uploaded JSON must be an array of records.")
                        st.stop()
                    source_df = pl.DataFrame(records)
                else:
                    st.error("⚠️ Unsupported file type.")
                    st.stop()
            else:
                source_engine = create_engine(source_db_url)
                with source_engine.connect() as conn:
                    source_df = pl.read_database(query=text(source_query), connection=conn)

            source_df = source_df.rename({col: col.strip().lower() for col in source_df.columns})
            duckdiffDB.register("source_table", source_df)

            # Preview Source
            preview_df = duckdiffDB.execute("SELECT * FROM source_table LIMIT 100").fetchdf()
            st.subheader("📘 Source Table Preview")
            st.dataframe(pl.from_pandas(preview_df), use_container_width=True)

            # Load Target
            target_engine = create_engine(target_db_url)
            with target_engine.connect() as conn:
                target_df = pl.read_database(query=text(target_query), connection=conn)

            target_df = target_df.rename({col: col.strip().lower() for col in target_df.columns})
            duckdiffDB.register("target_table", target_df)
            st.subheader("📙 Target Table Preview")
            st.dataframe(target_df, use_container_width=True)

            # Show diffs (only for SELECT * cases)
            if "select *" in source_query.lower() and "select *" in target_query.lower():
                show_target_not_in_source()
                show_record_level_diff()

        except Exception as e:
            st.error(f"❌ Comparison failed:\n{e}")
    else:
        st.error("⚠️ Please provide all required input.")

# ---------- Join File and Target DB ----------
if st.button("Join File and Target DB"):
    if uploaded_file and target_db_url.strip() and target_query.strip():
        try:
            content = uploaded_file.read()
            uploaded_file.seek(0)
            if source_input_type == "CSV":
                source_df = pl.read_csv(io.BytesIO(content))
            elif source_input_type == "JSON":
                records = json.load(io.BytesIO(content))
                if not isinstance(records, list):
                    st.error("Uploaded JSON must be an array of records.")
                    st.stop()
                source_df = pl.DataFrame(records)
            else:
                st.error("Unsupported file type.")
                st.stop()

            source_df = source_df.rename({col: col.strip().lower() for col in source_df.columns})

            # Load target
            target_engine = create_engine(target_db_url)
            with target_engine.connect() as conn:
                target_df = pl.read_database(query=text(target_query), connection=conn)
            target_df = target_df.rename({col: col.strip().lower() for col in target_df.columns})

            # Register in DuckDB
            duckdiffDB.register("source_file", source_df)
            duckdiffDB.register("target", target_df)

            # Join condition
            common_cols = set(source_df.columns).intersection(set(target_df.columns))
            if "dept_id" in source_df.columns and "id" in target_df.columns:
                join_condition = "s.dept_id = t.id"
            elif common_cols:
                join_col = list(common_cols)[0]
                join_condition = f"s.{join_col} = t.{join_col}"
            else:
                st.warning("⚠️ No common columns to join.")
                st.stop()

            # Perform join
            join_query = f"""
                SELECT s.*, t.name AS target_name
                FROM source_file s
                JOIN target t ON {join_condition}
            """
            join_df = duckdiffDB.execute(join_query).fetchdf()
            st.subheader("🔗 Joined Result")
            st.dataframe(pl.from_pandas(join_df), use_container_width=True)
            st.success(f"✅ Joined on: `{join_condition}`")

        except Exception as e:
            st.error(f"❌ Join failed: {e}")
    else:
        st.error("⚠️ Upload source file and provide target DB + query.")


#both csv and json in right side as well
import streamlit as st
from sqlalchemy import create_engine, text
import polars as pl
import duckdb
import io
import json

st.set_page_config(page_title="Data Comparison Tool", layout="wide")
st.title("🔍 Data Comparison Tool - File or DB vs DB")

# Setup DuckDB
if "duckdiffDB" not in st.session_state:
    st.session_state.duckdiffDB = duckdb.connect(database=":memory:")
duckdiffDB = st.session_state.duckdiffDB

# Initialize session state for source/target
for key in ["source_df", "target_df"]:
    if key not in st.session_state:
        st.session_state[key] = None

# Layout
left_col, right_col = st.columns(2)

# Source Input
with left_col:
    st.subheader("📂 Source Input")
    source_input_type = st.radio("Source Type", ["CSV", "JSON", "Database"], horizontal=True, key="source_type")
    uploaded_file = st.file_uploader("Upload Source File", type=["csv", "json"], key="source_file")
    source_db_url = st.text_input("Source DB URL", placeholder="e.g., sqlite:///source.db", key="src_url")
    source_query = st.text_area("Source Query", "SELECT * FROM source_table", key="src_query")

# Target Input
with right_col:
    st.subheader("🎯 Target Input")
    target_input_type = st.radio("Target Type", ["CSV", "JSON", "Database"], horizontal=True, key="target_type")
    uploaded_target_file = st.file_uploader("Upload Target File", type=["csv", "json"], key="target_file")
    target_db_url = st.text_input("Target DB URL", placeholder="e.g., sqlite:///target.db", key="tgt_url")
    target_query = st.text_area("Target Query", "SELECT * FROM target_table", key="tgt_query")

# Load Source
def load_source():
    df = None
    if source_input_type in ["CSV", "JSON"] and uploaded_file:
        content = uploaded_file.read()
        uploaded_file.seek(0)
        if source_input_type == "CSV":
            df = pl.read_csv(io.BytesIO(content))
        elif source_input_type == "JSON":
            records = json.load(io.BytesIO(content))
            if not isinstance(records, list):
                st.error("Uploaded JSON must be an array of records.")
                st.stop()
            df = pl.DataFrame(records)

        df = df.rename({col: col.strip().lower() for col in df.columns})
        st.session_state.source_df = df
        duckdiffDB.execute("DROP VIEW IF EXISTS source_table")
        duckdiffDB.execute("DROP TABLE IF EXISTS source_table")
        duckdiffDB.register("source_table", df)

    elif source_input_type == "Database" and source_db_url.strip() and source_query.strip():
        try:
            engine = create_engine(source_db_url)
            with engine.connect() as conn:
                df = pl.read_database(query=text(source_query), connection=conn)
            df = df.rename({col: col.strip().lower() for col in df.columns})
            st.session_state.source_df = df
            duckdiffDB.execute("DROP VIEW IF EXISTS source_table")
            duckdiffDB.execute("DROP TABLE IF EXISTS source_table")
            duckdiffDB.register("source_table", df)
        except Exception as e:
            st.error(f"❌ Source DB error: {e}")

# Load Target
def load_target():
    df = None
    if target_input_type in ["CSV", "JSON"] and uploaded_target_file:
        content = uploaded_target_file.read()
        uploaded_target_file.seek(0)
        if target_input_type == "CSV":
            df = pl.read_csv(io.BytesIO(content))
        elif target_input_type == "JSON":
            records = json.load(io.BytesIO(content))
            if not isinstance(records, list):
                st.error("Uploaded JSON must be an array of records.")
                st.stop()
            df = pl.DataFrame(records)

        df = df.rename({col: col.strip().lower() for col in df.columns})
        st.session_state.target_df = df
        duckdiffDB.execute("DROP VIEW IF EXISTS target_table")
        duckdiffDB.execute("DROP TABLE IF EXISTS target_table")
        duckdiffDB.register("target_table", df)

    elif target_input_type == "Database" and target_db_url.strip() and target_query.strip():
        try:
            engine = create_engine(target_db_url)
            with engine.connect() as conn:
                df = pl.read_database(query=text(target_query), connection=conn)
            df = df.rename({col: col.strip().lower() for col in df.columns})
            st.session_state.target_df = df
            duckdiffDB.execute("DROP VIEW IF EXISTS target_table")
            duckdiffDB.execute("DROP TABLE IF EXISTS target_table")
            duckdiffDB.register("target_table", df)
        except Exception as e:
            st.error(f"❌ Target DB error: {e}")

# Show ID Differences
def show_target_not_in_source():
    try:
        result = duckdiffDB.execute("""
            SELECT id FROM target_table
            WHERE id NOT IN (SELECT id FROM source_table)
        """).fetchdf()
        if not result.empty:
            st.subheader("🚫 IDs in Target but Not in Source")
            st.dataframe(result, use_container_width=True)
        else:
            st.success("✅ All IDs in target exist in source.")
    except Exception as e:
        st.error(f"❌ ID diff error: {e}")

# Show Record Differences
def show_record_level_diff():
    try:
        df_source = duckdiffDB.execute("""
            SELECT * FROM source_table
            EXCEPT
            SELECT * FROM target_table
            ORDER BY id DESC
        """).fetchall()

        df_target = duckdiffDB.execute("""
            SELECT * FROM target_table
            EXCEPT
            SELECT * FROM source_table
            ORDER BY id DESC
        """).fetchall()

        column_names = [desc[0] for desc in duckdiffDB.description]
        interleaved = []
        for i in range(max(len(df_source), len(df_target))):
            if i < len(df_source):
                interleaved.append(df_source[i] + ("source",))
            if i < len(df_target):
                interleaved.append(df_target[i] + ("target",))

        if not interleaved:
            st.success("✅ No differences found.")
            return

        final_columns = column_names + ["origin"]
        html = "<table style='width:100%; border-collapse:collapse;'>"
        html += "<thead><tr>" + "".join([f"<th style='border:1px solid #ccc; padding:6px'>{col}</th>" for col in final_columns]) + "</tr></thead><tbody>"
        for row in interleaved:
            color = "green" if row[-1] == "source" else "red"
            html += "<tr>" + "".join([f"<td style='color:{color}; border:1px solid #ccc; padding:6px'>{cell}</td>" for cell in row]) + "</tr>"
        html += "</tbody></table>"
        st.markdown(html, unsafe_allow_html=True)
        st.success(f"✅ Found {len(interleaved)} differing records.")
    except Exception as e:
        st.error(f"❌ Diff error: {e}")

# Run Comparison
if st.button("Run Comparison"):
    load_source()
    load_target()
    if st.session_state.source_df is not None:
        st.subheader("📘 Source Preview")
        st.dataframe(st.session_state.source_df, use_container_width=True)
    if st.session_state.target_df is not None:
        st.subheader("📙 Target Preview")
        st.dataframe(st.session_state.target_df, use_container_width=True)

    if duckdiffDB.execute("PRAGMA show_tables").fetchall():
        show_target_not_in_source()
        show_record_level_diff()

# Join Logic
if st.button("Join File and Target DB"):
    try:
        if st.session_state.source_df is None or st.session_state.target_df is None:
            st.error("❌ Source or Target data is missing.")
            st.stop()

        source_df = st.session_state.source_df
        target_df = st.session_state.target_df

        common_cols = set(source_df.columns).intersection(set(target_df.columns))
        if "dept_id" in source_df.columns and "id" in target_df.columns:
            join_col_source = "dept_id"
            join_col_target = "id"
        elif common_cols:
            join_col_source = join_col_target = list(common_cols)[0]
        else:
            st.warning("⚠️ No common columns to join.")
            st.stop()

        # Rename columns to avoid duplicates
        source_cols_renamed = [pl.col(c).alias(f"s_{c}") if c != join_col_source else pl.col(c) for c in source_df.columns]
        target_cols_renamed = [pl.col(c).alias(f"t_{c}") if c != join_col_target else pl.col(c) for c in target_df.columns]
        source_df = source_df.select(source_cols_renamed)
        target_df = target_df.select(target_cols_renamed)

        joined_df = source_df.join(
            target_df,
            left_on=join_col_source,
            right_on=join_col_target,
            how="inner"
        )

        st.subheader("🔗 Joined Result (Polars Only)")
        st.dataframe(joined_df, use_container_width=True)
        st.success(f"✅ Joined on: `s.{join_col_source} = t.{join_col_target}`")

    except Exception as e:
        st.error(f"❌ Join failed: {e}")

