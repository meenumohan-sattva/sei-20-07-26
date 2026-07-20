# SEI Harmonized Index Dashboard — Pipeline, Methodology, Results, Divergence Analysis, Interpretation
# Co-authored with CoCo
import os
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="SEI Harmonization Dashboard", layout="wide")

conn = st.connection("snowflake", ttl=os.getenv("SNOWFLAKE_CONNECTION_TTL"))


@st.cache_data
def load_harmonized_data():
    df = conn.query("""
        SELECT * FROM GATES_FOUNDATION_DB.BRONZE.MH_SEI_HARMONIZED
        WHERE YEAR >= 2015 AND YEAR <= 2024
        ORDER BY DISTRICT, YEAR, QUARTER
    """)
    return df


df = load_harmonized_data()

# --- SIDEBAR ---
with st.sidebar:
    st.title("SEI Dashboard")
    st.markdown("**Unified Harmonization Pipeline**")
    section = st.radio(
        "Navigate",
        ["Pipeline", "Methodology", "Results", "Divergence Analysis", "Interpretation"],
        index=2,
    )
    st.divider()
    districts = sorted(df["DISTRICT"].unique())
    selected_districts = st.multiselect(
        "Filter Districts", districts, default=districts[:5]
    )
    year_range = st.slider(
        "Year Range",
        2015,
        2024,
        (2015, 2024),
    )
    st.divider()
    if st.button("Clear cache"):
        load_harmonized_data.clear()
        st.rerun()

# Apply filters
filtered = df[
    (df["DISTRICT"].isin(selected_districts))
    & (df["YEAR"] >= year_range[0])
    & (df["YEAR"] <= year_range[1])
]

# ==========================================
# SECTION 1: PIPELINE
# ==========================================
if section == "Pipeline":
    st.header("Pipeline Architecture")
    st.markdown("""
    The SEI Harmonization Pipeline transforms **7 monthly geospatial source tables** (15 indicators)
    into two robust, harmonized indices through a 4-phase process.
    """)

    st.subheader("Source Tables & Indicators")
    source_data = pd.DataFrame({
        "Abbreviation": ["RAIN_MED", "LST_MEAN", "LST_MED", "NTL_MEAN", "NTL_SUM",
                         "HOSP_CT", "BANK_CT", "ROAD_KM", "SCHOOL_CT", "NDBI_MEAN",
                         "NDVI_MEAN", "POP_DENSITY", "MEAN_CROPS", "MEAN_BUILT", "MEAN_BARE"],
        "Source Table": ["MH_CHIRPS_MONTHLY_2014_2025", "MH_LST_MONTHLY_2014_2025",
                         "MH_LST_MONTHLY_2014_2025", "MH_VIIRS_MONTHLY_2014_2025",
                         "MH_VIIRS_MONTHLY_2014_2025", "MH_OSM_MONTHLY_2014_2025",
                         "MH_OSM_MONTHLY_2014_2025", "MH_OSM_MONTHLY_2014_2025",
                         "MH_OSM_MONTHLY_2014_2025", "MH_NDBI_MONTHLY_2014_2025_INTERPOLATED",
                         "MH_NDVI_MONTHLY_2014_2025_INTERPOLATED", "MH_HRSL_MONTHLY_2014_2025",
                         "MH_LULC_MONTHLY_2015_2024", "MH_LULC_MONTHLY_2015_2024",
                         "MH_LULC_MONTHLY_2015_2024"],
        "Raw Column": ["CHIRPS_MEDIAN_MM", "LST_MEAN", "LST_MEDIAN", "VIIRS_MEAN",
                       "VIIRS_SUM", "HOSPITAL_COUNT", "BANK_COUNT", "ROAD_LENGTH_KM",
                       "SCHOOL_COUNT", "MEAN_NDBI", "MEAN_NDVI", "POP_DENSITY_HRSL",
                       "MEAN_CROPS", "MEAN_BUILT", "MEAN_BARE"],
        "Category": ["Climate", "Climate", "Climate", "Nightlights", "Nightlights",
                     "Infrastructure", "Infrastructure", "Infrastructure", "Infrastructure",
                     "Land Use", "Vegetation", "Population", "Land Cover", "Land Cover", "Land Cover"],
    })
    st.dataframe(source_data, hide_index=True, use_container_width=True)

    st.subheader("Pipeline Flow")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        with st.container(border=True):
            st.markdown("**Phase 1**")
            st.caption("Pre-Processing")
            st.markdown("""
            - Quarterly aggregation
            - Targeted imputation
            - STL decomposition
            """)
    with col2:
        with st.container(border=True):
            st.markdown("**Phase 2**")
            st.caption("Weight Extraction")
            st.markdown("""
            - PCA branch (PC1)
            - ICW branch (inv-cov)
            - Unity normalization
            """)
    with col3:
        with st.container(border=True):
            st.markdown("**Phase 3**")
            st.caption("Synthesis")
            st.markdown("""
            - Meta-PCA fusion
            - Envelope weighting
            - Min-Max scaling
            """)
    with col4:
        with st.container(border=True):
            st.markdown("**Phase 4**")
            st.caption("Output")
            st.markdown("""
            - SEI_METAPCA_FINAL
            - SEI_ENVELOPE_FINAL
            - Write to Snowflake
            """)

# ==========================================
# SECTION 2: METHODOLOGY
# ==========================================
elif section == "Methodology":
    st.header("Methodology")

    tab1, tab2, tab3 = st.tabs(["PCA & ICW Branches", "Meta-PCA (Branch A)", "Envelope (Branch B)"])

    with tab1:
        st.subheader("Phase 2: Dual Weight Extraction")
        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                st.markdown("**PCA Branch (Infrastructure-Heavy)**")
                st.markdown("""
                1. Compute covariance matrix of MinMax-scaled trend lines
                2. Eigen-decompose to extract PC1 factor loadings
                3. Flip sign if loading sum < 0 (directional alignment)
                4. **Normalize**: divide by sum so weights sum to 1
                5. Score = weighted sum; Min-Max scale to [0,1]
                """)
                st.info("PCA naturally assigns higher weights to correlated infrastructure indicators (NTL, roads, banks).")

        with col2:
            with st.container(border=True):
                st.markdown("**ICW Branch (Environment-Sensitive)**")
                st.markdown("""
                1. Compute covariance matrix of MinMax-scaled trend lines
                2. Invert the covariance matrix (pseudo-inverse if singular)
                3. Row-sum the inverse covariance matrix
                4. **Normalize**: divide absolute row-sums by their total so weights sum to 1
                5. Score = weighted sum; Min-Max scale to [0,1]
                """)
                st.info("ICW penalizes redundancy — indicators unique to the system (NDVI, rainfall) receive higher weight.")

        st.markdown("---")
        st.markdown("**Unity Constraint**: Both methodologies explicitly normalize weights so they sum to exactly 1.0. "
                    "This ensures comparability across methods and interpretability as portfolio allocations.")

    with tab2:
        st.subheader("Branch A: Meta-PCA Framework")
        st.markdown("""
        **Goal**: Maximize shared information space between PCA and ICW signals without human subjectivity.

        **Steps**:
        1. Stack `SEI_PCA` and `SEI_ICW` into a 2-dimensional matrix
        2. Standardize (z-score) both columns
        3. Compute 2x2 covariance matrix
        4. Extract Meta-PC1 (first eigenvector)
        5. Directional alignment: flip if loading sum < 0
        6. Project standardized data onto Meta-PC1
        7. Min-Max scale result to [0,1] -> **SEI_METAPCA_FINAL**

        **Interpretation**: The Meta-PCA finds the axis of maximum joint variance between the two
        sub-indices, mathematically capturing what both PCA and ICW "agree" on.
        """)

    with tab3:
        st.subheader("Branch B: Envelope Weighting Framework")
        st.markdown("""
        **Goal**: Retain the strongest signal per indicator from either method.

        **Steps**:
        1. For each indicator: W_env(i) = max(W_PCA(i), W_ICW(i))
        2. **Unity normalization**: W_norm(i) = W_env(i) / sum(W_env) so weights sum to 1
        3. Matrix multiply: normalized weights x MinMax-scaled trend matrix
        4. Min-Max scale result to [0,1] -> **SEI_ENVELOPE_FINAL**

        **Interpretation**: The envelope never discards information — it always picks the methodology
        that assigns the stronger weight to each indicator, then re-normalizes.
        """)

# ==========================================
# SECTION 3: RESULTS
# ==========================================
elif section == "Results":
    st.header("Results")

    # KPI row
    with st.container(horizontal=True):
        st.metric("Districts", f"{df['DISTRICT'].nunique()}", border=True)
        st.metric("Quarters", f"{df.groupby(['YEAR','QUARTER']).ngroups}", border=True)
        st.metric("MetaPCA Mean", f"{df['SEI_METAPCA_FINAL'].mean():.3f}", border=True)
        st.metric("Envelope Mean", f"{df['SEI_ENVELOPE_FINAL'].mean():.3f}", border=True)
        corr = np.corrcoef(df["SEI_METAPCA_FINAL"], df["SEI_ENVELOPE_FINAL"])[0, 1]
        st.metric("Correlation", f"{corr:.3f}", border=True)

    st.divider()

    # Time series
    st.subheader("Index Time-Series (Mean Across Selected Districts)")
    ts = filtered.groupby(["YEAR", "QUARTER"]).agg(
        MetaPCA=("SEI_METAPCA_FINAL", "mean"),
        Envelope=("SEI_ENVELOPE_FINAL", "mean"),
    ).reset_index()
    ts["Period"] = ts["YEAR"].astype(str) + "-Q" + ts["QUARTER"].astype(str)

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("**SEI MetaPCA**")
            st.line_chart(ts, x="Period", y="MetaPCA", color="#9C27B0")
    with col2:
        with st.container(border=True):
            st.markdown("**SEI Envelope**")
            st.line_chart(ts, x="Period", y="Envelope", color="#009688")

    # Statewide SEI (mean across ALL districts for 40 quarters)
    st.subheader("Statewide SEI (All Districts, 40 Quarters)")
    state_ts = df.groupby(["YEAR", "QUARTER"]).agg(
        MetaPCA=("SEI_METAPCA_FINAL", "mean"),
        Envelope=("SEI_ENVELOPE_FINAL", "mean"),
    ).reset_index()
    state_ts["Period"] = state_ts["YEAR"].astype(str) + "-Q" + state_ts["QUARTER"].astype(str)
    with st.container(border=True):
        st.line_chart(state_ts, x="Period", y=["MetaPCA", "Envelope"])

    # Distribution
    st.subheader("Distribution Comparison")
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("**SEI_METAPCA_FINAL**")
            bins_m = pd.cut(filtered["SEI_METAPCA_FINAL"], bins=20)
            hist_m = bins_m.value_counts().sort_index().reset_index()
            hist_m.columns = ["Bin", "Count"]
            hist_m["Bin"] = hist_m["Bin"].astype(str)
            st.bar_chart(hist_m, x="Bin", y="Count")
    with col2:
        with st.container(border=True):
            st.markdown("**SEI_ENVELOPE_FINAL**")
            bins_e = pd.cut(filtered["SEI_ENVELOPE_FINAL"], bins=20)
            hist_e = bins_e.value_counts().sort_index().reset_index()
            hist_e.columns = ["Bin", "Count"]
            hist_e["Bin"] = hist_e["Bin"].astype(str)
            st.bar_chart(hist_e, x="Bin", y="Count")

    # Scatter
    st.subheader("Method Agreement (Scatter)")
    scatter_df = filtered[["SEI_METAPCA_FINAL", "SEI_ENVELOPE_FINAL"]].copy()
    st.scatter_chart(scatter_df, x="SEI_METAPCA_FINAL", y="SEI_ENVELOPE_FINAL")

    # Statistics table
    st.subheader("Summary Statistics")
    stats = df[["SEI_METAPCA_FINAL", "SEI_ENVELOPE_FINAL", "SEI_PCA_INTERMEDIATE", "SEI_ICW_INTERMEDIATE"]].describe()
    stats.insert(0, "STATISTIC", stats.index)
    st.dataframe(stats, hide_index=True, use_container_width=True)

    # Raw data
    st.subheader("Filtered Data")
    st.dataframe(filtered, hide_index=True, use_container_width=True)

# ==========================================
# SECTION 4: DIVERGENCE ANALYSIS
# ==========================================
elif section == "Divergence Analysis":
    st.header("Divergence Analysis: What Drives Differences Between Methods?")
    st.markdown("""
    This section investigates **why** SEI_METAPCA and SEI_ENVELOPE produce different rankings,
    identifying which indicators are responsible for the divergence.
    """)

    # Weights from pipeline run (PCA and ICW, unity-normalized)
    INDICATOR_COLS = ["RAIN_MED", "LST_MEAN", "LST_MED", "NTL_MEAN", "NTL_SUM",
                     "HOSP_CT", "BANK_CT", "ROAD_KM", "SCHOOL_CT", "NDBI_MEAN",
                     "NDVI_MEAN", "POP_DENSITY", "MEAN_CROPS", "MEAN_BUILT", "MEAN_BARE"]

    w_pca = np.array([-0.0315, 0.3635, 0.3661, -0.0182, -0.0135,
                      -0.0195, -0.0162, -0.0258, -0.0159, 0.0967,
                      -0.2494, 0.0161, 0.3777, -0.0086, 0.1785])
    w_icw = np.array([0.0770, 0.0048, 0.0416, 0.0210, 0.0008,
                      0.0429, 0.1426, 0.0405, 0.2643, 0.0361,
                      0.1319, 0.0066, 0.0284, 0.0909, 0.0705])

    # Derived
    weight_diff = w_pca - w_icw
    w_envelope = np.maximum(w_pca, w_icw)
    w_envelope_norm = w_envelope / w_envelope.sum()

    # Correlation from notebook: indicators vs (MetaPCA - Envelope)
    corr_data = pd.DataFrame({
        "Indicator": ["LST_MED", "LST_MEAN", "MEAN_CROPS", "NDVI_MEAN", "MEAN_BARE",
                      "NDBI_MEAN", "RAIN_MED", "HOSP_CT", "ROAD_KM", "SCHOOL_CT",
                      "BANK_CT", "NTL_MEAN", "NTL_SUM", "MEAN_BUILT", "POP_DENSITY"],
        "Corr_with_Divergence": [-0.962, -0.960, -0.868, 0.778, -0.723,
                                  -0.376, 0.250, 0.222, 0.217, 0.215,
                                  0.206, 0.180, 0.143, 0.115, -0.024],
        "Direction": ["Envelope", "Envelope", "Envelope", "MetaPCA", "Envelope",
                      "Envelope", "MetaPCA", "MetaPCA", "MetaPCA", "MetaPCA",
                      "MetaPCA", "MetaPCA", "MetaPCA", "MetaPCA", "Envelope"],
    })

    # KPI row
    corr = np.corrcoef(df["SEI_METAPCA_FINAL"], df["SEI_ENVELOPE_FINAL"])[0, 1]
    with st.container(horizontal=True):
        st.metric("Method Correlation", f"{corr:.4f}", border=True)
        st.metric("Total Weight Divergence", f"{np.abs(weight_diff).sum():.3f}", border=True)
        st.metric("Max Single Indicator Gap", f"{np.abs(weight_diff).max():.3f}", border=True)

    st.divider()

    # Weight comparison table
    st.subheader("Weight Comparison Table")
    weight_table = pd.DataFrame({
        "Indicator": INDICATOR_COLS,
        "W_PCA": w_pca,
        "W_ICW": w_icw,
        "W_Envelope": w_envelope_norm,
        "Diff (PCA-ICW)": weight_diff,
        "Dominant": ["PCA" if d > 0 else "ICW" for d in weight_diff],
    }).sort_values("Diff (PCA-ICW)", key=abs, ascending=False)
    st.dataframe(weight_table, hide_index=True, use_container_width=True)

    # Weight divergence chart
    st.subheader("Weight Divergence (PCA - ICW per Indicator)")
    sorted_idx = np.argsort(weight_diff)
    chart_df = pd.DataFrame({
        "Indicator": [INDICATOR_COLS[i] for i in sorted_idx],
        "PCA minus ICW": weight_diff[sorted_idx],
    })
    st.bar_chart(chart_df, x="Indicator", y="PCA minus ICW")

    st.divider()

    # Correlation with divergence
    st.subheader("Indicator Correlation with (MetaPCA - Envelope)")
    st.markdown("Positive = pushes MetaPCA higher; Negative = pushes Envelope higher")
    st.dataframe(corr_data, hide_index=True, use_container_width=True)

    st.divider()

    # Key drivers
    st.subheader("Key Divergence Drivers")

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("**Indicators Driving Envelope HIGHER**")
            st.markdown("""
            These have high PCA weights (picked by envelope's max operation) but low ICW weights:

            | Indicator | W_PCA | W_ICW | r with divergence |
            |-----------|-------|-------|-------------------|
            | **LST_MED** | 0.366 | 0.042 | -0.962 |
            | **LST_MEAN** | 0.364 | 0.005 | -0.960 |
            | **MEAN_CROPS** | 0.378 | 0.028 | -0.868 |

            PCA assigns >35% weight to temperature/crops (correlated cluster).
            Envelope inherits these large weights directly.
            """)

    with col2:
        with st.container(border=True):
            st.markdown("**Indicators Driving MetaPCA HIGHER**")
            st.markdown("""
            These have high ICW weights or negative PCA weights (captured by meta-synthesis):

            | Indicator | W_PCA | W_ICW | r with divergence |
            |-----------|-------|-------|-------------------|
            | **NDVI_MEAN** | -0.249 | 0.132 | +0.778 |
            | **RAIN_MED** | -0.032 | 0.077 | +0.250 |
            | **HOSP_CT** | -0.020 | 0.043 | +0.222 |

            NDVI has a **negative** PCA weight (inverse to development).
            ICW sees it as unique. MetaPCA synthesizes both sub-index scores,
            partially preserving this environmental signal.
            """)

    st.divider()

    # District-level divergence
    st.subheader("District-Level Divergence")
    st.markdown("Districts where MetaPCA and Envelope disagree most:")

    district_div = df.copy()
    district_div["Divergence"] = district_div["SEI_METAPCA_FINAL"] - district_div["SEI_ENVELOPE_FINAL"]
    district_summary = district_div.groupby("DISTRICT").agg(
        Mean_Divergence=("Divergence", "mean"),
        Std_Divergence=("Divergence", "std"),
    ).reset_index()
    district_summary["Abs_Mean"] = district_summary["Mean_Divergence"].abs()
    top_divergent = district_summary.nlargest(20, "Abs_Mean")
    top_divergent["Direction"] = top_divergent["Mean_Divergence"].apply(
        lambda x: "MetaPCA > Envelope" if x > 0 else "Envelope > MetaPCA"
    )

    st.dataframe(
        top_divergent[["DISTRICT", "Mean_Divergence", "Std_Divergence", "Direction"]],
        hide_index=True,
        use_container_width=True,
    )

    # Divergence over time
    st.subheader("Divergence Over Time (Statewide)")
    div_ts = df.copy()
    div_ts["Divergence"] = div_ts["SEI_METAPCA_FINAL"] - div_ts["SEI_ENVELOPE_FINAL"]
    div_time = div_ts.groupby(["YEAR", "QUARTER"]).agg(
        Mean_Divergence=("Divergence", "mean"),
    ).reset_index()
    div_time["Period"] = div_time["YEAR"].astype(str) + "-Q" + div_time["QUARTER"].astype(str)
    with st.container(border=True):
        st.line_chart(div_time, x="Period", y="Mean_Divergence")

    st.divider()

    # Root cause explanation
    st.subheader("Root Cause Explanation")
    with st.container(border=True):
        st.markdown("""
        **Why do the methods diverge?**

        1. **Weight concentration**: PCA creates a highly concentrated weight vector dominated by
           LST and CROPS (a correlated temperature/agricultural cluster with >35% each).
           ICW distributes weight toward unique signals (SCHOOL_CT=26%, BANK_CT=14%, NDVI=13%).

        2. **Envelope inherits PCA's concentration**: Since envelope takes max(W_PCA, W_ICW) per indicator,
           it inherits PCA's large LST/CROPS weights, making it temperature/agriculture-driven.

        3. **MetaPCA partially cancels concentration**: By operating on the two *sub-index scores*
           (not raw weights), MetaPCA finds the shared variance axis. Since SEI_PCA (temperature-heavy)
           and SEI_ICW (infrastructure-heavy) pull in different directions, MetaPCA finds a compromise
           that neither method dominates.

        4. **Negative PCA loading on NDVI**: PCA treats vegetation as *inversely* related to development
           (W=-0.25), while ICW gives it a positive weight (W=+0.13). This fundamental disagreement
           about NDVI's direction is the single largest driver of divergence.
        """)

# ==========================================
# SECTION 5: INTERPRETATION
# ==========================================
elif section == "Interpretation":
    st.header("Interpretation Guide")

    st.subheader("What the Indices Measure")
    st.markdown("""
    Both indices aim to capture **socio-economic development** at the sub-district level
    by combining 15 geospatial proxies spanning infrastructure, environment, population, and land use.

    | Index | Core Signal | Best For |
    |-------|-------------|----------|
    | **SEI_METAPCA** | Joint variance between PCA and ICW | Detecting consensus development patterns |
    | **SEI_ENVELOPE** | Maximum-weight per indicator | Capturing strongest signal per dimension |
    """)

    st.subheader("Reading the Values")
    st.markdown("""
    - **0.0** = Lowest development in the dataset (relative minimum)
    - **1.0** = Highest development in the dataset (relative maximum)
    - Values are **relative rankings**, not absolute measures
    - Trends over time indicate structural improvement/decline after seasonal removal (STL)
    """)

    st.subheader("When Methods Agree vs Diverge")
    corr = np.corrcoef(df["SEI_METAPCA_FINAL"], df["SEI_ENVELOPE_FINAL"])[0, 1]
    with st.container(border=True):
        st.metric("Rank Correlation (Pearson)", f"{corr:.4f}", border=True)

    if corr > 0.8:
        st.success("High agreement — both methods identify similar development patterns.")
    elif corr > 0.5:
        st.warning("Moderate agreement — methods emphasize different dimensions. Use both for robustness.")
    else:
        st.error("Low/negative agreement — methods capture fundamentally different signals. "
                 "Investigate which indicators drive divergence (see Divergence Analysis section).")

    st.markdown("""
    **Key differences in weighting philosophy:**
    - **PCA** over-weights correlated clusters (e.g., LST + CROPS dominate with >35% each)
    - **ICW** up-weights unique signals that are uncorrelated with others (e.g., SCHOOL_CT=26%, NDVI=13%)
    - **Envelope** is a hybrid that never assigns a zero-weight to any important signal
    - **Meta-PCA** finds the single axis that both PCA and ICW agree on

    **Recommended use:**
    - Use **SEI_ENVELOPE** for policy where all dimensions matter equally
    - Use **SEI_METAPCA** for research where statistical consensus is preferred
    - Report **both** to show robustness (or lack thereof) in rankings
    """)

    st.subheader("Limitations")
    st.markdown("""
    1. **Relative scale**: Values are dataset-specific — cannot compare across states or time windows without re-fitting
    2. **Census codes**: District identifiers are PC11_SD_ID codes (not human-readable names)
    3. **Temporal coverage**: LULC data limited to 2015-2024; edge effects possible in early/late quarters
    4. **STL assumptions**: Period=4 assumes annual seasonality; sub-annual structural breaks may be smoothed
    5. **Negative correlation**: If MetaPCA and Envelope are negatively correlated, the underlying PCA/ICW branches
       weight indicators in opposing directions — check directional alignment
    """)
