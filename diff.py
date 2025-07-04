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

# Initialize session state
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
    source_query = st.text_area("Source Query", "SELECT * FROM uploaded_source_data", key="src_query")

# Target Input
with right_col:
    st.subheader("🌟 Target Input")
    target_input_type = st.radio("Target Type", ["CSV", "JSON", "Database"], horizontal=True, key="target_type")
    uploaded_target_file = st.file_uploader("Upload Target File", type=["csv", "json"], key="target_file")
    target_db_url = st.text_input("Target DB URL", placeholder="e.g., sqlite:///target.db", key="tgt_url")
    target_query = st.text_area("Target Query", "SELECT * FROM uploaded_target_data", key="tgt_query")

# Load Source
def load_source():
    if source_input_type in ["CSV", "JSON"] and uploaded_file:
        content = uploaded_file.read()
        uploaded_file.seek(0)
        if source_input_type == "CSV":
            df = pl.read_csv(io.BytesIO(content))
        else:  # JSON
            records = json.load(io.BytesIO(content))
            if not isinstance(records, list):
                st.error("Uploaded JSON must be an array of records.")
                return
            df = pl.DataFrame(records)

        df = df.rename({col: col.strip().lower() for col in df.columns})
        duckdiffDB.register("uploaded_source_data", df)

        if source_query.strip():
            try:
                result_df = duckdiffDB.execute(source_query).pl()
                duckdiffDB.register("source_table", result_df)
                st.session_state.source_df = result_df
            except Exception as e:
                st.error(f"❌ Source query error: {e}")

    elif source_input_type == "Database" and source_db_url.strip() and source_query.strip():
        try:
            engine = create_engine(source_db_url)
            with engine.connect() as conn:
                df = pl.read_database(query=text(source_query), connection=conn)
            df = df.rename({col: col.strip().lower() for col in df.columns})
            duckdiffDB.register("source_table", df)
            st.session_state.source_df = df
        except Exception as e:
            st.error(f"❌ Source DB error: {e}")

# Load Target
def load_target():
    if target_input_type in ["CSV", "JSON"] and uploaded_target_file:
        content = uploaded_target_file.read()
        uploaded_target_file.seek(0)
        if target_input_type == "CSV":
            df = pl.read_csv(io.BytesIO(content))
        else:  # JSON
            records = json.load(io.BytesIO(content))
            if not isinstance(records, list):
                st.error("Uploaded JSON must be an array of records.")
                return
            df = pl.DataFrame(records)

        df = df.rename({col: col.strip().lower() for col in df.columns})
        duckdiffDB.register("uploaded_target_data", df)

        if target_query.strip():
            try:
                result_df = duckdiffDB.execute(target_query).pl()
                duckdiffDB.register("target_table", result_df)
                st.session_state.target_df = result_df
            except Exception as e:
                st.error(f"❌ Target query error: {e}")

    elif target_input_type == "Database" and target_db_url.strip() and target_query.strip():
        try:
            engine = create_engine(target_db_url)
            with engine.connect() as conn:
                df = pl.read_database(query=text(target_query), connection=conn)
            df = df.rename({col: col.strip().lower() for col in df.columns})
            duckdiffDB.register("target_table", df)
            st.session_state.target_df = df
        except Exception as e:
            st.error(f"❌ Target DB error: {e}")

# Show ID Diff
def show_target_not_in_source():
    try:
        result = duckdiffDB.execute("""
            SELECT * FROM target_table
            EXCEPT
            SELECT * FROM source_table
        """).fetchdf()

        if result.shape[0] > 0:
            st.subheader("🚫 Rows in Target but Not in Source")
            st.dataframe(result, use_container_width=True)
        else:
            st.success("✅ All rows in target exist in source.")
    except Exception as e:
        st.error(f"❌ Target-Not-In-Source error: {e}")

# Show Record-Level Diff
def show_record_level_diff():
    try:
        df_source = duckdiffDB.execute("SELECT * FROM source_table EXCEPT SELECT * FROM target_table").fetchall()
        df_target = duckdiffDB.execute("SELECT * FROM target_table EXCEPT SELECT * FROM source_table").fetchall()

        column_names = [desc[0] for desc in duckdiffDB.description]
        interleaved = []
        for i in range(max(len(df_source), len(df_target))):
            if i < len(df_source):
                interleaved.append(df_source[i] + ("source",))
            if i < len(df_target):
                interleaved.append(df_target[i] + ("target",))

        if interleaved:
            final_columns = column_names + ["origin"]
            html = "<table style='width:100%; border-collapse:collapse;'>"
            html += "<thead><tr>" + "".join([f"<th style='border:1px solid #ccc; padding:6px'>{col}</th>" for col in final_columns]) + "</tr></thead><tbody>"
            for row in interleaved:
                color = "green" if row[-1] == "source" else "red"
                html += "<tr>" + "".join([f"<td style='color:{color}; border:1px solid #ccc; padding:6px'>{cell}</td>" for cell in row]) + "</tr>"
            html += "</tbody></table>"
            st.markdown(html, unsafe_allow_html=True)
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

    # Skip diffing for aggregation queries only
    aggregation_keywords = ["group by", "sum(", "count(", "avg(", "min(", "max(", "having"]
    if not any(kw in source_query.lower() for kw in aggregation_keywords) and \
       not any(kw in target_query.lower() for kw in aggregation_keywords):

        show_target_not_in_source()
        show_record_level_diff()
    else:
        st.info("ℹ️ Skipped ID and record-level comparison due to aggregation in source or target query.")


if st.button("Join File and Target DB"):
    try:
        if st.session_state.source_df is None or st.session_state.target_df is None:
            st.error("❌ Source or Target data is missing.")
            st.stop()

        source_df = st.session_state.source_df
        target_df = st.session_state.target_df

        # Determine join columns based on common columns
        common_cols = list(set(source_df.columns).intersection(set(target_df.columns)))

        if not common_cols:
            st.warning("⚠️ No common columns to perform join.")
            st.stop()

        # Select first 1–2 common columns for joining
        join_cols = common_cols[:2]  # You can adjust this as needed

        # Add prefix to avoid duplicate column names
        source_renamed = source_df.rename({col: f"s_{col}" for col in source_df.columns if col not in join_cols})
        target_renamed = target_df.rename({col: f"t_{col}" for col in target_df.columns if col not in join_cols})

        # Perform the join
        joined_df = source_renamed.join(
            target_renamed,
            left_on=join_cols,
            right_on=join_cols,
            how="inner"
        )

        st.subheader("🔗 Joined Result")
        st.dataframe(joined_df, use_container_width=True)
        st.success(f"✅ Joined on column(s): {', '.join(join_cols)}")

    except Exception as e:
        st.error(f"❌ Join failed: {e}")
