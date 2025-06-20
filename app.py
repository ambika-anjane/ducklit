import streamlit as st
from sqlalchemy import create_engine, text
import polars as pl
import pandas as pd
import duckdb

# Streamlit setup
st.set_page_config(page_title="Data Comparison Tool - Differences Only", layout="wide")
st.title("🔍 Data Comparison Tool - Row-wise Differences")

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

                st.subheader("📘 Source Table Preview")
                st.dataframe(source_df.to_pandas())

            # Load Target
            target_engine = create_engine(target_db_url)
            with target_engine.connect() as conn:
                target_df = pl.read_database(query=text(target_query), connection=conn)
                duckdiffDB.register("target_table", target_df)

                st.subheader("📙 Target Table Preview")
                st.dataframe(target_df.to_pandas())

            # Get column list
            cols = duckdiffDB.execute("PRAGMA table_info(source_table)").fetchall()
            col_names = [f'"{col[1]}"' for col in cols]
            col_expr = ", ".join(col_names)

            # Add 'origin' column to label source/target
            source_only_query = f"""
                SELECT 'source' AS origin, {col_expr}
                FROM source_table
                EXCEPT
                SELECT 'source' AS origin, {col_expr}
                FROM target_table
            """
            target_only_query = f"""
                SELECT 'target' AS origin, {col_expr}
                FROM target_table
                EXCEPT
                SELECT 'target' AS origin, {col_expr}
                FROM source_table
            """

            final_query = f"""
                {source_only_query}
                UNION ALL
                {target_only_query}
            """

            # Execute and get result
            result_df = duckdiffDB.execute(final_query).fetchdf()

            st.success("✅ Differences retrieved (with origin labels)!")
            st.dataframe(result_df)

        except Exception as e:
            st.error(f"❌ An error occurred:\n{e}")
