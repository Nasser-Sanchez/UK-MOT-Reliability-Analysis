"""
UK Car Terminal Mileage Analyser - v1.0
Streamlit frontend for querying terminal mileage predictions.
"""

import streamlit as st
import duckdb
import os
import glob

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PREDICTION_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "terminal_predictions")

# ---------------------------------------------------------------------------
# Helper: query all parquet files via DuckDB
# ---------------------------------------------------------------------------

def query_car(registration: str):
    """Look up a registration across all prediction parquet files."""
    # DuckDB can glob parquet files directly
    pattern = os.path.join(PREDICTION_DIR, "batch_*.parquet")
    files = sorted(glob.glob(pattern))
    
    if not files:
        return None, "No prediction files found in data/terminal_predictions/"
    
    # Build a UNION ALL query (DuckDB handles this efficiently)
    # Generator expression (no brackets) to produce strings for join
    union_parts = " UNION ALL ".join(
        f"SELECT *, '{f}' AS _source FROM read_parquet('{f}')"
        for f in files
    )
    query = f"SELECT * FROM ({union_parts}) WHERE UPPER(registration) = '{registration.upper()}' LIMIT 1"
    
    try:
        result = duckdb.query(query).df()
        if result.empty:
            return None, None
        return result.iloc[0], None
    except Exception as e:
        return None, str(e)


def query_all_cars():
    """Return all predictions as a single DataFrame."""
    pattern = os.path.join(PREDICTION_DIR, "batch_*.parquet")
    files = sorted(glob.glob(pattern))
    
    if not files:
        return None
    
    union_parts = " UNION ALL ".join(f"SELECT *, '{f}' AS _source FROM read_parquet('{f}')" for f in files)
    query = f"SELECT * FROM ({union_parts})"
    
    try:
        return duckdb.query(query).df()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="UK Car Terminal Mileage", page_icon="car", layout="wide")
st.title("UK Car Terminal Mileage Analyser")
st.caption("Query terminal mileage predictions by registration.")

# Search box
st.subheader("Search by Registration")
reg_input = st.text_input("Registration", placeholder="e.g. AB12CDE", max_chars=8)

if reg_input.strip():
    registration = reg_input.strip().upper()
    
    with st.spinner(f"Looking up {registration}..."):
        row, err = query_car(registration)
    
    if err:
        st.error(f"Error: {err}")
    elif row is None:
        st.warning(f"No predictions found for {registration}.")
    else:
        st.success(f"Found: {registration}")
        
        # Car details
        st.subheader("Vehicle Details")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Make", row.get("make", "N/A"))
        col2.metric("Model", row.get("model", "N/A"))
        col3.metric("Fuel Type", row.get("fuelType", "N/A"))
        col4.metric("Engine Size", row.get("engineSize_bucket", "N/A"))
        
        # Current status
        st.subheader("Current Status")
        col5, col6, col7 = st.columns(3)
        col5.metric("Current Mileage", f"{row.get('current_mileage', 0):,.0f} miles")
        col6.metric("Advisory Defects", int(row.get("defect_count_advisory", 0)))
        col7.metric("Dangerous Defects", int(row.get("defect_count_dangerous", 0)))
        
        # Terminal mileage predictions
        st.subheader("Terminal Mileage Prediction")
        col8, col9, col10 = st.columns(3)
        col8.metric("Median Terminal", f"{row.get('terminal_median', 0):,.0f} miles")
        col9.metric("Mean Terminal", f"{row.get('terminal_mean', 0):,.0f} miles")
        col10.metric("P75 Terminal", f"{row.get('terminal_p75', 0):,.0f} miles")
        
        # Remaining life
        st.subheader("Remaining Life Estimate")
        col11, col12, col13 = st.columns(3)
        remaining_median = row.get("remaining_life_median", 0)
        remaining_mean = row.get("remaining_life_mean", 0)
        remaining_p75 = row.get("remaining_life_p75", 0)
        
        col11.metric(
            "Median Remaining",
            f"{remaining_median:,.0f} miles" if remaining_median > 0 else "N/A"
        )
        col12.metric(
            "Mean Remaining",
            f"{remaining_mean:,.0f} miles" if remaining_mean > 0 else "N/A"
        )
        col13.metric(
            "P75 Remaining",
            f"{remaining_p75:,.0f} miles" if remaining_p75 > 0 else "N/A"
        )
        
        # Depreciation / cost-of-ownership calculator
        st.subheader("Ownership Cost Calculator")
        col_cost1, col_cost2 = st.columns(2)
        car_price = col_cost1.number_input(
            "Car Price (£)",
            min_value=0,
            step=500,
            help="Enter the current market value of the car. Optional.",
            key="car_price",
        )
        annual_mileage = col_cost2.number_input(
            "Annual Mileage",
            min_value=0,
            step=500,
            value=7000,
            help="Estimated miles driven per year. Defaults to 7,000.",
            key="annual_mileage",
        )

        if remaining_median > 0 and car_price > 0 and annual_mileage > 0:
            years_remaining = remaining_median / annual_mileage
            cost_per_year = car_price / years_remaining
            cost_per_mile = car_price / remaining_median

            col_d1, col_d2, col_d3 = st.columns(3)
            col_d1.metric("Years Remaining", f"{years_remaining:.1f} years")
            col_d2.metric("Cost Per Year", f"£{cost_per_year:,.0f}")
            col_d3.metric("Cost Per Mile", f"{cost_per_mile:.2f} p/mile")
        elif remaining_median > 0 and car_price > 0 and annual_mileage == 0:
            st.caption("Enter a non-zero annual mileage to calculate costs.")
        elif remaining_median > 0 and car_price == 0:
            st.caption("Enter a car price above to calculate ownership costs.")
        
        # Visual bar for remaining life vs terminal
        if remaining_median > 0 and row.get("terminal_median", 0) > 0:
            st.subheader("Progress to Terminal Mileage")
            progress_pct = min(100, (row.get("current_mileage", 0) / row.get("terminal_median", 1)) * 100)
            st.progress(progress_pct / 100)
            st.caption(f"{progress_pct:.1f}% of estimated terminal mileage reached")
        
        # Model metadata
        st.subheader("Model Metadata")
        col_meta1, col_meta2 = st.columns(2)
        col_meta1.caption("Prediction Source")
        col_meta2.caption(row.get("_source", "Unknown"))
        
        # Quick stats across all cars
        st.divider()
        st.subheader("Dataset Overview")
        all_df = query_all_cars()
        if all_df is not None and not all_df.empty:
            col_s1, col_s2, col_s3 = st.columns(3)
            col_s1.metric("Total Cars in Dataset", f"{len(all_df):,}")
            col_s2.metric("Average Terminal Median", f"{all_df['terminal_median'].mean():,.0f} miles")
            col_s3.metric("Average Current Mileage", f"{all_df['current_mileage'].mean():,.0f} miles")

else:
    st.info("Enter a registration number above to look up its terminal mileage prediction.")
