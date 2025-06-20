import os
import shutil
import tempfile
from io import StringIO

import example_duck
import streamlit as st
from code_editor import code_editor

from sqlalchemy import create_engine, text
from snowflake.sqlalchemy import URL as SnowflakeURL
from sqlalchemy.exc import SQLAlchemyError

# --- Streamlit config ---
st.set_page_config(page_title="ducklit", page_icon=":duck:")

# --- DuckDB Connection ---
def get_db_connection():
    if "duck_conn" not in st.session_state:
        st.session_state["duck_conn"] = example_duck.connect(":memory:")
    return st.session_state["duck_conn"]

# --- Snowflake SQLAlchemy URL ---
def get_snowflake_url():
    return SnowflakeURL(
        user="Ambika",
        password="Snowflake#2025",
        account="POEVRBR-DW28551",
        warehouse="COMPUTE_WH",
        database="RAW",
        schema="TEST",
        role="ACCOUNTADMIN"
    )

# --- Main ---
def main():
    conn = get_db_connection()
    create_side_bar(conn)
    create_page(conn)

# --- Sidebar UI ---
def create_side_bar(conn: example_duck.DuckDBPyConnection):
    cur = conn.cursor()

    with st.sidebar:
        st.markdown("## 📁 Uploads")
        st.button("📥 Load Sample Data", on_click=load_sample_data, args=[conn])

        files = st.file_uploader("Upload CSV or JSON files", accept_multiple_files=True)
        load_files(conn, files)

        st.divider()
        st.markdown("## ❄️ Snowflake")
        if st.checkbox("Connect and load from Snowflake"):
            table_name = st.text_input("Enter Snowflake Table Name")
            if st.button("🚀 Load Table from Snowflake"):
                load_from_snowflake(table_name, conn)

        st.divider()
        st.markdown("## 🦆 Import DuckDB Database")
        db_file = st.file_uploader("Upload a .duckdb file", type=["duckdb"])
        if db_file is not None:
            import_duckdb_file(conn, db_file)

        st.divider()
        st.markdown("## 📊 Tables in DuckDB")
        table_list = ""

        try:
            cur.execute("SHOW TABLES")
            recs = cur.fetchall()
            if len(recs) > 0:
                for rec in recs:
                    table_name = rec[0]
                    table_list += f"- `{table_name}`\n"
                    cur.execute(f"DESCRIBE {table_name}")
                    for col in cur.fetchall():
                        table_list += f"    - {col[0]} {col[1]}\n"
            else:
                table_list = "_No tables found in DuckDB_"
        except Exception as e:
            table_list = f"❌ Error: {e}"

        st.markdown(table_list)

# --- Load CSV / JSON into DuckDB ---
def load_files(conn: example_duck.DuckDBPyConnection, files: list):
    for file in files:
        stringio = StringIO(file.getvalue().decode("utf-8"))

        if file.name.endswith(".csv"):
            conn.read_csv(stringio).create(file.name[:-4])
        elif file.name.endswith(".json"):
            with open(file.name, "w") as temp_file:
                stringio.seek(0)
                shutil.copyfileobj(stringio, temp_file)
            conn.read_json(file.name).create(file.name[:-5])
            os.remove(file.name)

# --- Load Sample JSON ---
def load_sample_data(conn: example_duck.DuckDBPyConnection):
    conn.read_json("sample_data/posts.json").create("posts")

# --- Load from Snowflake using SQLAlchemy ---
def load_from_snowflake(table_name: str, duckdb_conn: example_duck.DuckDBPyConnection):
    try:
        sf_engine = create_engine(get_snowflake_url())
        with sf_engine.connect() as sf_conn:
            result = sf_conn.execute(text(f"SELECT * FROM {table_name}"))
            rows = result.fetchall()

            if not rows:
                st.warning("Snowflake table is empty.")
                return

            columns = result.keys()
            col_defs = ", ".join([f"{col} TEXT" for col in columns])
            duckdb_conn.execute(f"CREATE OR REPLACE TABLE {table_name} ({col_defs})")

            values = [tuple(str(val) if val is not None else "" for val in row) for row in rows]
            placeholders = ", ".join(["?"] * len(columns))

            duckdb_conn.executemany(
                f"INSERT INTO {table_name} VALUES ({placeholders})", values
            )

            st.success(f"✅ Loaded {len(values)} rows from Snowflake into DuckDB")

    except SQLAlchemyError as e:
        st.error(f"❌ Snowflake error: {e}")
    except Exception as e:
        st.error(f"⚠️ DuckDB insert error: {e}")

# --- Import DuckDB File ---
def import_duckdb_file(conn: example_duck.DuckDBPyConnection, db_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".duckdb") as tmp_file:
            tmp_file.write(db_file.getbuffer())
            db_path = tmp_file.name

        conn.execute(f"ATTACH DATABASE '{db_path}' AS extdb")

        schemas = conn.execute("SELECT schema_name FROM extdb.information_schema.schemata").fetchall()
        table_count = 0
        for (schema,) in schemas:
            tables = conn.execute(
                f"""
                SELECT table_name 
                FROM extdb.information_schema.tables 
                WHERE table_schema = '{schema}'
                """
            ).fetchall()
            for (table,) in tables:
                conn.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM extdb.\"{schema}\".\"{table}\"")
                table_count += 1

        st.success(f"✅ Loaded {table_count} table(s) from {db_file.name}")
    except Exception as e:
        st.error(f"❌ Failed to load DuckDB file: {e}")

# --- Main Query Page ---
def create_page(conn: example_duck.DuckDBPyConnection):
    st.title("ducklit :duck:")
    st.write("Query your files or Snowflake data using DuckDB SQL")
    st.divider()

    cur = conn.cursor()
    st.write("💡 Hint: End each SQL statement with a semicolon (;)")
    st.write("⌨️ Press Ctrl+Enter to execute")

    res = code_editor(code="", lang="sql", key="editor")

    for query in res["text"].split(";"):
        query = query.strip()
        if query:
            try:
                cur.execute(query)
                df = cur.fetch_df()
                if not df.empty:
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("✅ Query successful. No rows returned.")
            except Exception as e:
                st.error(f"⚠️ {e}")

    if st.button("🔄 Reset Database"):
        st.cache_resource.clear()
        st.session_state["editor"]["text"] = ""
        st.session_state.pop("duck_conn", None)
        st.rerun()

if __name__ == "__main__":
    main()


# showing diff

import streamlit as st
from sqlalchemy import create_engine, text
import example_duck
import polars as pl
import pandas as pd

st.set_page_config(page_title="Data Comparison Tool - Row Differences Only", layout="wide")
st.title("🔍 Data Comparison Tool - Row-by-Row Differences")

source, target = st.columns(2)

duckdiffDB = example_duck.connect(database=":memory:")

with source:
    source_db_url = st.text_input("Source DB URL", placeholder="e.g., sqlite:///mydb.sqlite")
    source_query = st.text_area("Source Query", "SELECT * FROM my_table LIMIT 10")

with target:
    target_db_url = st.text_input("Target DB URL", placeholder="e.g., sqlite:///mydb.sqlite")
    target_query = st.text_area("Target Query", "SELECT * FROM my_table LIMIT 10")

if st.button("Run SQL"):
    if not source_db_url or not target_db_url or not source_query.strip() or not target_query.strip():
        st.error("Please enter both DB URLs and SQL queries.")
    else:
        log = st.empty()
        try:
            duckdiffDB.execute("DROP TABLE IF EXISTS source_table")
            duckdiffDB.execute("DROP TABLE IF EXISTS target_table")

            source_engine = create_engine(source_db_url)
            target_engine = create_engine(target_db_url)

            with source_engine.connect() as conn:
                df_source = pl.read_database(text(source_query), connection=conn)
            with target_engine.connect() as conn:
                df_target = pl.read_database(text(target_query), connection=conn)

            # Convert to pandas for comparison
            pdf_source = df_source.to_pandas()
            pdf_target = df_target.to_pandas()

            # Truncate to the minimum row count
            min_len = min(len(pdf_source), len(pdf_target))
            pdf_source = pdf_source.head(min_len)
            pdf_target = pdf_target.head(min_len)

            # Pad shorter column lists if number of columns mismatch
            max_cols = max(pdf_source.shape[1], pdf_target.shape[1])

            mismatches = []
            for i in range(min_len):
                row_src = pdf_source.iloc[i].tolist()
                row_tgt = pdf_target.iloc[i].tolist()

                # Pad shorter rows to align comparison
                while len(row_src) < max_cols:
                    row_src.append(None)
                while len(row_tgt) < max_cols:
                    row_tgt.append(None)

                if row_src != row_tgt:
                    mismatch = {"Row #": i + 1}
                    for j in range(max_cols):
                        mismatch[f"SRC_col{j+1}"] = str(row_src[j])
                        mismatch[f"TGT_col{j+1}"] = str(row_tgt[j])
                    mismatches.append(mismatch)

            if mismatches:
                st.warning(f"⚠️ {len(mismatches)} row(s) differ.")
                st.dataframe(pd.DataFrame(mismatches))
            else:
                st.success("✅ All rows match.")

        except Exception as e:
            st.error(f"An error occurred:\n{e}")


# set 3 ( to show only non matchig rows)
import streamlit as st
from sqlalchemy import create_engine, text
import example_duck
import polars as pl

# ---------- UI ----------
st.set_page_config(page_title="Data Comparison Tool - Differences Only", layout="wide")
st.title("🔍 Data Comparison Tool - Row-based Differences")

source, target = st.columns(2)
duckdiffDB = example_duck.connect(database=":memory:")

with source:
    source_db_url = st.text_input("Enter SQLAlchemy Source Database URL", placeholder="e.g., sqlite:///mydb.sqlite")
    st.subheader("Source Query")
    source_query = st.text_area("Enter SQL query for Source", "SELECT * FROM my_table LIMIT 10")

with target:
    target_db_url = st.text_input("Enter SQLAlchemy Target Database URL", placeholder="e.g., sqlite:///mydb.sqlite")
    st.subheader("Target Query")
    target_query = st.text_area("Enter SQL query for Target", "SELECT * FROM my_table LIMIT 10")

if st.button("Run SQL"):
    if not source_db_url or not target_db_url or not source_query.strip() or not target_query.strip():
        st.error("Please provide both database URLs and SQL queries.")
    else:
        log = st.empty()
        try:
            log.text("Starting comparison...")
            duckdiffDB.execute("DROP TABLE IF EXISTS source_table")
            duckdiffDB.execute("DROP TABLE IF EXISTS target_table")

            source_engine = create_engine(source_db_url)
            target_engine = create_engine(target_db_url)

            # Read source
            with source_engine.connect() as conn:
                df_source = pl.read_database(query=text(source_query), connection=conn)
                duckdiffDB.register("source_table", df_source)

            # Read target
            with target_engine.connect() as conn:
                df_target = pl.read_database(query=text(target_query), connection=conn)
                duckdiffDB.register("target_table", df_target)

            # Combine and find rows not in both
            query = """
            SELECT 'source' as __origin__, * FROM source_table
            EXCEPT
            SELECT 'source' as __origin__, * FROM target_table
            UNION ALL
            SELECT 'target' as __origin__, * FROM target_table
            EXCEPT
            SELECT 'target' as __origin__, * FROM source_table
            """

            result = duckdiffDB.execute(query).fetchdf()
            st.success("Comparison complete. Showing non-matching rows:")
            st.dataframe(result, use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred: {e}")

# bala's code

import streamlit as st
from sqlalchemy import create_engine, text
import duckdb
import polars as pl

# ---------- UI ----------
st.set_page_config(page_title="Data Comparison Tool - Differences Only", layout="wide")
st.title("🔍 Data Comparison Tool - Query-based Differences")

source, target = st.columns(2)

# Create DuckDB in-memory instance
duckdiffDB = duckdb.connect(database=":memory:")

with source:
    source_db_url = st.text_input("Enter SQLAlchemy Source Database URL",
                                   placeholder="e.g., sqlite:///mydb.sqlite or snowflake://user:pass@account/db/schema")
    st.subheader("Source Query")
    source_query = st.text_area("Enter SQL query for Source", "SELECT * FROM my_table LIMIT 10")

with target:
    target_db_url = st.text_input("Enter SQLAlchemy Target Database URL",
                                   placeholder="e.g., sqlite:///mydb.sqlite or snowflake://user:pass@account/db/schema")
    st.subheader("Target Query")
    target_query = st.text_area("Enter SQL query for Target", "SELECT * FROM my_table LIMIT 10")

# Run button
if st.button("Run SQL"):
    if not source_db_url or not source_query.strip() or not target_db_url or not target_query.strip():
        st.error("Please provide both a database URL and a SQL query.")
    else:
        log = st.empty()
        try:
            log.text("🟡 Starting the DuckDiff process...")
            duckdiffDB.execute("DROP TABLE IF EXISTS source_table")
            duckdiffDB.execute("DROP TABLE IF EXISTS target_table")

            # Connect to databases
            source_engine = create_engine(source_db_url)
            target_engine = create_engine(target_db_url)
            sourceHasResults = targetHasResults = False
            source_columns = target_columns = []

            # Source
            log.text("🔗 Connecting to source database...")
            with source_engine.connect() as connection:
                df = pl.read_database(query=text(source_query), connection=connection)
                if not df.is_empty():
                    sourceHasResults = True
                    duckdiffDB.register("source_table", df)
                    source_columns = df.columns
                    df.clear()
                    log.text("✅ Source results loaded into DuckDB.")
                else:
                    log.text("⚠️ Source returned no data.")

            # Target
            log.text("🔗 Connecting to target database...")
            with target_engine.connect() as connection:
                df = pl.read_database(query=text(target_query), connection=connection)
                if not df.is_empty():
                    targetHasResults = True
                    duckdiffDB.register("target_table", df)
                    target_columns = df.columns
                    df.clear()
                    log.text("✅ Target results loaded into DuckDB.")
                else:
                    log.text("⚠️ Target returned no data.")

            # Proceed if both have data
            if sourceHasResults and targetHasResults:
                # Show registered tables
                log.text("📋 Listing tables in DuckDB...")
                tables_df = duckdiffDB.execute("SHOW TABLES").fetchdf()
                st.subheader("📋 Tables in DuckDB")
                st.dataframe(tables_df)

                # Compare data
                log.text("🔍 Performing data comparison...")
                order_by_clause = ", ".join(str(i + 2) for i in range(len(target_columns)))
                st.info(f"Order By Clause: {order_by_clause}")

                duckdb_result = duckdiffDB.execute(f"""
                    SELECT 'source' AS __duckdiff_source__, * FROM source_table
                    UNION ALL
                    SELECT 'target' AS __duckdiff_source__, * FROM target_table
                    ORDER BY {order_by_clause}, 1
                """)
                st.success("✅ Query executed successfully!")
                st.dataframe(duckdb_result.fetchdf())

            else:
                st.warning("Query executed successfully. But one or both queries returned no data.")

        except Exception as e:
            st.error(f"❌ An error occurred:\n\n{e}")

# bala's modifie code
import streamlit as st
from sqlalchemy import create_engine, text
import example_duck
import polars as pl
import pandas as pd  # Only for final display

st.set_page_config(page_title="Data Comparison Tool - Match & Difference", layout="wide")
st.title("🔍 Data Comparison Tool - Row-wise Match & Difference")

source, target = st.columns(2)
duckdiffDB = example_duck.connect(database=":memory:")

with source:
    source_db_url = st.text_input("Enter Source DB URL", placeholder="e.g., sqlite:///mydb.sqlite")
    source_query = st.text_area("Source Query", "SELECT * FROM my_table LIMIT 10")

with target:
    target_db_url = st.text_input("Enter Target DB URL", placeholder="e.g., sqlite:///mydb.sqlite")
    target_query = st.text_area("Target Query", "SELECT * FROM my_table LIMIT 10")

if st.button("Run Comparison"):
    if not source_db_url or not source_query.strip() or not target_db_url or not target_query.strip():
        st.error("Provide both DB URLs and queries.")
    else:
        try:
            st.info("Running comparison...")
            duckdiffDB.execute("DROP TABLE IF EXISTS source_table")
            duckdiffDB.execute("DROP TABLE IF EXISTS target_table")

            # Load Source
            source_engine = create_engine(source_db_url)
            with source_engine.connect() as conn:
                source_df = pl.read_database(query=text(source_query), connection=conn)
                duckdiffDB.register("source_table", source_df)

            # Load Target
            target_engine = create_engine(target_db_url)
            with target_engine.connect() as conn:
                target_df = pl.read_database(query=text(target_query), connection=conn)
                duckdiffDB.register("target_table", target_df)

            # Normalize column order (not names!)
            cols = duckdiffDB.execute("PRAGMA table_info(source_table)").fetchall()
            col_expr = ", ".join([f'"{col[1]}"' for col in cols])  # preserve case

            # Matching rows: SOURCE_MATCH and TARGET_MATCH
            source_match_query = f"""
                SELECT 'SOURCE_MATCH' AS status, {col_expr}
                FROM source_table
                INTERSECT
                SELECT 'SOURCE_MATCH' AS status, {col_expr}
                FROM target_table
            """
            target_match_query = f"""
                SELECT 'TARGET_MATCH' AS status, {col_expr}
                FROM target_table
                INTERSECT
                SELECT 'TARGET_MATCH' AS status, {col_expr}
                FROM source_table
            """

            # Non-matching rows
            source_only_query = f"""
                SELECT 'SOURCE_NON_MATCH' AS status, {col_expr}
                FROM source_table
                EXCEPT
                SELECT 'SOURCE_NON_MATCH' AS status, {col_expr}
                FROM target_table
            """
            target_only_query = f"""
                SELECT 'TARGET_NON_MATCH' AS status, {col_expr}
                FROM target_table
                EXCEPT
                SELECT 'TARGET_NON_MATCH' AS status, {col_expr}
                FROM source_table
            """

            # Combine all
            final_query = f"""
                {source_match_query}
                UNION ALL
                {target_match_query}
                UNION ALL
                {source_only_query}
                UNION ALL
                {target_only_query}
            """

            result_df = duckdiffDB.execute(final_query).fetchdf()
            st.success("Comparison complete!")
            st.dataframe(result_df, use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred:\n{e}")

# version 2
import streamlit as st
from sqlalchemy import create_engine, text
import polars as pl
import pandas as pd
import duckdb

# Streamlit setup
st.set_page_config(page_title="Data Comparison Tool - Match & Difference", layout="wide")
st.title("🔍 Data Comparison Tool - Row-wise Match & Difference")

# Layout for input
source, target = st.columns(2)
duckdiffDB = duckdb.connect(database=":memory:")

# Source DB and Query
with source:
    source_db_url = st.text_input("Enter Source DB URL", placeholder="e.g., sqlite:///source.db")
    source_query = st.text_area("Source Query", "SELECT * FROM my_table LIMIT 10")

# Target DB and Query
with target:
    target_db_url = st.text_input("Enter Target DB URL", placeholder="e.g., sqlite:///target.db")
    target_query = st.text_area("Target Query", "SELECT * FROM my_table LIMIT 10")

# Compare Button
if st.button("Run Comparison"):
    if not source_db_url or not source_query.strip() or not target_db_url or not target_query.strip():
        st.error("Provide both DB URLs and queries.")
    else:
        try:
            st.info("Running comparison...")

            # Reset temp tables
            duckdiffDB.execute("DROP TABLE IF EXISTS source_table")
            duckdiffDB.execute("DROP TABLE IF EXISTS target_table")

            # Load Source
            source_engine = create_engine(source_db_url)
            with source_engine.connect() as conn:
                source_df = pl.read_database(query=text(source_query), connection=conn)
                duckdiffDB.register("source_table", source_df)

            # Load Target
            target_engine = create_engine(target_db_url)
            with target_engine.connect() as conn:
                target_df = pl.read_database(query=text(target_query), connection=conn)
                duckdiffDB.register("target_table", target_df)

            # Get column list
            cols = duckdiffDB.execute("PRAGMA table_info(source_table)").fetchall()
            col_expr = ", ".join([f'"{col[1]}"' for col in cols])  # preserve column case

            # Queries for matches and diffs
            source_match_query = f"""
                SELECT 'SOURCE_MATCH' AS status, {col_expr}
                FROM source_table
                INTERSECT
                SELECT 'SOURCE_MATCH' AS status, {col_expr}
                FROM target_table
            """
            target_match_query = f"""
                SELECT 'TARGET_MATCH' AS status, {col_expr}
                FROM target_table
                INTERSECT
                SELECT 'TARGET_MATCH' AS status, {col_expr}
                FROM source_table
            """
            source_only_query = f"""
                SELECT 'SOURCE_NON_MATCH' AS status, {col_expr}
                FROM source_table
                EXCEPT
                SELECT 'SOURCE_NON_MATCH' AS status, {col_expr}
                FROM target_table
            """
            target_only_query = f"""
                SELECT 'TARGET_NON_MATCH' AS status, {col_expr}
                FROM target_table
                EXCEPT
                SELECT 'TARGET_NON_MATCH' AS status, {col_expr}
                FROM source_table
            """

            final_query = f"""
                {source_match_query}
                UNION ALL
                {target_match_query}
                UNION ALL
                {source_only_query}
                UNION ALL
                {target_only_query}
            """

            # Execute and get result
            result_df = duckdiffDB.execute(final_query).fetchdf()

            # Highlight styling
            def highlight_diff(row):
                if row['status'] == 'SOURCE_NON_MATCH':
                    return ['background-color: #155724'] * len(row)  # green
                elif row['status'] == 'TARGET_NON_MATCH':
                    return ['background-color: #721c24'] * len(row)  # red
                else:
                    return [''] * len(row)

            styled_df = result_df.style.apply(highlight_diff, axis=1)

            # Display results
            st.success("✅ Comparison complete!")
            st.write(styled_df)

        except Exception as e:
            st.error(f"❌ An error occurred:\n{e}")


# recent with color tags math and not match
import streamlit as st
from sqlalchemy import create_engine, text
import polars as pl
import pandas as pd
import duckdb

# Streamlit setup
st.set_page_config(page_title="Data Comparison Tool - Match & Difference", layout="wide")
st.title("🔍 Data Comparison Tool - Row-wise Match & Difference")

# Layout for input
source, target = st.columns(2)
duckdiffDB = duckdb.connect(database=":memory:")

# Source DB and Query
with source:
    source_db_url = st.text_input("Enter Source DB URL", placeholder="e.g., sqlite:///source.db")
    source_query = st.text_area("Source Query", "SELECT * FROM my_table LIMIT 10")

# Target DB and Query
with target:
    target_db_url = st.text_input("Enter Target DB URL", placeholder="e.g., sqlite:///target.db")
    target_query = st.text_area("Target Query", "SELECT * FROM my_table LIMIT 10")

# Compare Button
if st.button("Run Comparison"):
    if not source_db_url or not source_query.strip() or not target_db_url or not target_query.strip():
        st.error("Provide both DB URLs and queries.")
    else:
        try:
            st.info("Running comparison...")

            # Reset temp tables
            duckdiffDB.execute("DROP TABLE IF EXISTS source_table")
            duckdiffDB.execute("DROP TABLE IF EXISTS target_table")

            # Load Source
            source_engine = create_engine(source_db_url)
            with source_engine.connect() as conn:
                source_df = pl.read_database(query=text(source_query), connection=conn)
                duckdiffDB.register("source_table", source_df)

                # ✅ Show source table
                st.subheader("📘 Source Table Preview")
                st.dataframe(source_df.to_pandas())

            # Load Target
            target_engine = create_engine(target_db_url)
            with target_engine.connect() as conn:
                target_df = pl.read_database(query=text(target_query), connection=conn)
                duckdiffDB.register("target_table", target_df)

                # ✅ Show target table
                st.subheader("📙 Target Table Preview")
                st.dataframe(target_df.to_pandas())

            # Get column list
            cols = duckdiffDB.execute("PRAGMA table_info(source_table)").fetchall()
            col_expr = ", ".join([f'"{col[1]}"' for col in cols])  # preserve column case

            # Queries for matches and diffs
            source_match_query = f"""
                SELECT 'SOURCE_MATCH' AS status, {col_expr}
                FROM source_table
                INTERSECT
                SELECT 'SOURCE_MATCH' AS status, {col_expr}
                FROM target_table
            """
            target_match_query = f"""
                SELECT 'TARGET_MATCH' AS status, {col_expr}
                FROM target_table
                INTERSECT
                SELECT 'TARGET_MATCH' AS status, {col_expr}
                FROM source_table
            """
            source_only_query = f"""
                SELECT 'SOURCE_NON_MATCH' AS status, {col_expr}
                FROM source_table
                EXCEPT
                SELECT 'SOURCE_NON_MATCH' AS status, {col_expr}
                FROM target_table
            """
            target_only_query = f"""
                SELECT 'TARGET_NON_MATCH' AS status, {col_expr}
                FROM target_table
                EXCEPT
                SELECT 'TARGET_NON_MATCH' AS status, {col_expr}
                FROM source_table
            """

            final_query = f"""
                {source_match_query}
                UNION ALL
                {target_match_query}
                UNION ALL
                {source_only_query}
                UNION ALL
                {target_only_query}
            """

            # Execute and get result
            result_df = duckdiffDB.execute(final_query).fetchdf()

            # Highlight styling
            def highlight_diff(row):
                if row['status'] == 'SOURCE_NON_MATCH':
                    return ['background-color: #155724'] * len(row)  # green
                elif row['status'] == 'TARGET_NON_MATCH':
                    return ['background-color: #721c24'] * len(row)  # red
                else:
                    return [''] * len(row)

            styled_df = result_df.style.apply(highlight_diff, axis=1)

            # Display results
            st.success("✅ Comparison complete!")
            st.write(styled_df)

        except Exception as e:
            st.error(f"❌ An error occurred:\n{e}")
