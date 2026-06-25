import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="Bank Fraud EDA Dashboard",
    page_icon="🏦",
    layout="wide"
)

# =========================
# Modern Professional CSS
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif;
    background-color: #f8fafc;
}

/* Titles */
.main-title {
    font-size: 38px;
    font-weight: 800;
    color: #0f172a;
    letter-spacing: -0.5px;
    margin-bottom: 5px;
}
.sub-title {
    font-size: 16px;
    color: #64748b;
    margin-bottom: 30px;
}

/* Custom KPI Metric Cards */
.kpi-container {
    display: flex;
    gap: 15px;
    margin-bottom: 25px;
}
.kpi-card {
    flex: 1;
    background: white;
    padding: 22px;
    border-radius: 16px;
    box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.05), 0 2px 4px -2px rgb(0 0 0 / 0.05);
    border: 1px solid #e2e8f0;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 15px -3px rgb(0 0 0 / 0.05), 0 4px 6px -4px rgb(0 0 0 / 0.05);
}
.kpi-label {
    font-size: 13px;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.kpi-value {
    font-size: 28px;
    font-weight: 700;
    color: #0f172a;
    margin-top: 8px;
}

/* Analytical Alert & Info Boxes */
.insight-box {
    background-color: #f0fdf4;
    padding: 20px;
    border-radius: 12px;
    border-left: 5px solid #16a34a;
    font-size: 15px;
    color: #14532d;
    box-shadow: 0 1px 3px rgb(0 0 0 / 0.05);
    margin-bottom: 20px;
}
.warning-box {
    background-color: #fef2f2;
    padding: 20px;
    border-radius: 12px;
    border-left: 5px solid #dc2626;
    font-size: 15px;
    color: #7f1d1d;
    box-shadow: 0 1px 3px rgb(0 0 0 / 0.05);
    margin-bottom: 20px;
}

/* Sidebar Customizations */
[data-testid="stSidebar"] {
    background-color: #0f172a;
}
[data-testid="stSidebar"] *, [data-testid="stSidebar"] p, [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 {
    color: #f8fafc !important;
}
div[data-testid="stSidebarUserContent"] {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)


# =========================
# Custom Plotly Theme Config
# =========================
def apply_plotly_theme(fig):
    fig.update_layout(
        template="plotly_white",
        font_family="Inter",
        title_font_size=18,
        title_font_color="#0f172a",
        title_font_weight="bold",
        margin=dict(l=40, r=40, t=50, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig


# =========================
# Data Loading Pipeline
# =========================
@st.cache_data(show_spinner=True)
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        return df

    try:
        df = pd.read_csv("data/bank_fraud.csv")
        return df
    except Exception:
        pass

    try:
        import kagglehub
        from kagglehub import KaggleDatasetAdapter

        file_path = "bank_fraud.csv"
        df = kagglehub.dataset_load(
            KaggleDatasetAdapter.PANDAS,
            "nafiulislam490/bank-transaction-fraud-detection-dataset",
            file_path,
        )
        return df
    except Exception as e:
        st.error(
            "Could not load the dataset automatically. "
            "Please upload bank_fraud.csv from the sidebar, "
            "or add it inside a data/ folder in GitHub."
        )
        st.exception(e)
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def clean_data(df):
    df = df.copy()
    if "fraud_type" in df.columns:
        df["fraud_type"] = df["fraud_type"].fillna("Legitimate")

    if "customer_id" in df.columns and "customer_serial_num" not in df.columns:
        df["customer_serial_num"] = df.groupby("customer_id").ngroup() + 1

    if (
        "customer_id" in df.columns
        and "transaction_id" in df.columns
        and "account_transaction_count" not in df.columns
    ):
        df["account_transaction_count"] = (
            df.groupby("customer_id")["transaction_id"].transform("count")
        )

    if "customer_age" in df.columns and "age_group" not in df.columns:
        bins = [18, 25, 35, 45, 55, 65, 100]
        labels = ["18-25", "26-35", "36-45", "46-55", "56-65", "65+"]
        df["age_group"] = pd.cut(
            df["customer_age"],
            bins=bins,
            labels=labels,
            right=False
        )
    return df


def safe_number(value):
    try:
        return f"{value:,.0f}"
    except Exception:
        return str(value)


def show_kpis(df):
    total_transactions = len(df)
    fraud_transactions = int(df["is_fraud"].sum()) if "is_fraud" in df.columns else 0
    fraud_rate = (fraud_transactions / total_transactions * 100) if total_transactions > 0 else 0
    avg_amount = df["transaction_amount"].mean() if "transaction_amount" in df.columns else 0

    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-card">
            <div class="kpi-label">Total Transactions</div>
            <div class="kpi-value">{safe_number(total_transactions)}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Fraud Transactions</div>
            <div class="kpi-value" style="color: #dc2626;">{safe_number(fraud_transactions)}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Fraud Rate</div>
            <div class="kpi-value" style="color: #dc2626;">{fraud_rate:.2f}%</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Average Amount</div>
            <div class="kpi-value">${avg_amount:,.2f}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def apply_filters(df):
    filtered = df.copy()
    st.sidebar.markdown("### 🔎 Filters")

    if "country" in filtered.columns:
        countries = sorted(filtered["country"].dropna().unique())
        selected = st.sidebar.multiselect("Country", countries, default=countries)
        filtered = filtered[filtered["country"].isin(selected)]

    if "payment_method" in filtered.columns:
        methods = sorted(filtered["payment_method"].dropna().unique())
        selected = st.sidebar.multiselect("Payment Method", methods, default=methods)
        filtered = filtered[filtered["payment_method"].isin(selected)]

    if "merchant_category" in filtered.columns:
        categories = sorted(filtered["merchant_category"].dropna().unique())
        selected = st.sidebar.multiselect("Merchant Category", categories, default=categories)
        filtered = filtered[filtered["merchant_category"].isin(selected)]

    if "is_fraud" in filtered.columns:
        fraud_options = st.sidebar.multiselect(
            "Fraud Status",
            options=[0, 1],
            default=[0, 1],
            format_func=lambda x: "Fraud" if x == 1 else "Legitimate"
        )
        filtered = filtered[filtered["is_fraud"].isin(fraud_options)]

    return filtered


# =========================
# Sidebar & Navigation
# =========================
st.sidebar.markdown("## 🏦 Bank Fraud EDA")
uploaded_file = st.sidebar.file_uploader("Upload bank_fraud.csv", type=["csv"])

df_raw = load_data(uploaded_file)
if df_raw.empty:
    st.stop()

df = clean_data(df_raw)
df_filtered = apply_filters(df)

st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation Menus",
    [
        "Home",
        "Data Overview",
        "Data Cleaning",
        "Univariate Analysis",
        "Relationships",
        "Final Insights"
    ]
)


# =========================
# Home Page
# =========================
if page == "Home":
    st.markdown('<div class="main-title">🏦 Bank Fraud Analytics Portal</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">An interactive intelligence dashboard designed to track, explore, and analyze fraudulent transaction behaviors.</div>', unsafe_allow_html=True)

    show_kpis(df_filtered)
    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("""
        ### 🎯 Project Goal
        This advanced operational analytical dashboard breaks down corporate bank transaction datasets to uncover anomaly risk vectors. By comparing legitimate user habits against fraudulent markers, the application surfaces useful actionable insights within customer demographics, transaction velocities, payment channels, and merchant profiles.
        
        ### 📈 Exploratory Data Workflow
        1. **Ingestion & Inspection:** Primary loading phase alongside dimensional schema inspection.
        2. **Pipeline Feature Engineering:** Imputation of conditional flags and automated transaction mapping.
        3. **Univariate Profiling:** Dissecting isolated metric densities and continuous field distributions.
        4. **Relational Analysis:** Correlating high-dimensional independent variables against binary fraud outcomes.
        """)

    with col2:
        if "is_fraud" in df_filtered.columns:
            fraud_counts = df_filtered["is_fraud"].value_counts().reset_index()
            fraud_counts.columns = ["is_fraud", "count"]
            fraud_counts["Status"] = fraud_counts["is_fraud"].map({0: "Legitimate", 1: "Fraud"})
            fig = px.pie(
                fraud_counts,
                names="Status",
                values="count",
                hole=0.4,
                color_discrete_sequence=["#2563eb", "#dc2626"]
            )
            fig.update_layout(title_text="Overall Volumetric Breakdown")
            st.plotly_chart(apply_plotly_theme(fig), use_container_width=True)


# =========================
# Data Overview
# =========================
elif page == "Data Overview":
    st.markdown('<div class="main-title">📊 Dataset Structural Overview</div>', unsafe_allow_html=True)
    st.write("")
    show_kpis(df_filtered)

    c1, c2 = st.columns(2)
    with c1:
        st.info(f"🔢 Total Sample Rows: **{df_filtered.shape[0]:,}**")
    with c2:
        st.info(f"🔠 Schema Total Columns: **{df_filtered.shape[1]:,}**")

    st.subheader("Data Matrix Preview (First 10 Head Rows)")
    st.dataframe(df_filtered.head(10), use_container_width=True)

    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.subheader("Data Column Metadata Types")
        dtype_df = pd.DataFrame({
            "Column Reference": df_filtered.columns,
            "System Data Type": df_filtered.dtypes.astype(str).values
        })
        st.dataframe(dtype_df, use_container_width=True)
    with col_b:
        st.subheader("Descriptive Descriptive Statistics Summary")
        st.dataframe(df_filtered.describe().T, use_container_width=True)


# =========================
# Data Cleaning
# =========================
elif page == "Data Cleaning":
    st.markdown('<div class="main-title">🧹 Data Pipelines & Feature Engineering</div>', unsafe_allow_html=True)
    st.write("")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Missing Elements Prior to Cleaning")
        missing_before = df_raw.isnull().sum().sort_values(ascending=False).reset_index()
        missing_before.columns = ["Column Reference", "Missing Elements Count"]
        st.dataframe(missing_before, use_container_width=True)

    with col2:
        st.subheader("Missing Elements Post Transformation")
        missing_after = df.isnull().sum().sort_values(ascending=False).reset_index()
        missing_after.columns = ["Column Reference", "Missing Elements Count"]
        st.dataframe(missing_after, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    💡 <b>Imputation Strategy Note:</b> Missing items within <b>fraud_type</b> were explicitly filled with <b>Legitimate</b>. This ensures raw null entries represent non-fraud base financial transactions.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Generated Features Manifest")
    features = []
    if "customer_serial_num" in df.columns:
        features.append("**customer_serial_num**: Encodes complex alphanumeric strings (`customer_id`) into sequential discrete integers.")
    if "account_transaction_count" in df.columns:
        features.append("**account_transaction_count**: Computes rolling historical volume count generated by the individual customer account.")
    if "age_group" in df.columns:
        features.append("**age_group**: Discretizes continuous age float metrics into readable consumer segment clusters.")

    for feature in features:
        st.markdown(f"✅ {feature}")


# =========================
# Univariate Analysis
# =========================
elif page == "Univariate Analysis":
    st.markdown('<div class="main-title">📈 Univariate Feature Distributions</div>', unsafe_allow_html=True)
    st.write("")
    show_kpis(df_filtered)

    chart = st.selectbox(
        "Select Visual Target Metric",
        [
            "Total Monetary Value: Legitimate vs Fraud",
            "PIN Changed Recently",
            "Fraud Distribution",
            "Transaction Amount Distribution",
            "Customer Age Distribution"
        ]
    )

    if chart == "Total Monetary Value: Legitimate vs Fraud":
        if {"is_fraud", "transaction_amount"}.issubset(df_filtered.columns):
            total_money = df_filtered.groupby("is_fraud")["transaction_amount"].sum().reset_index()
            total_money["Status"] = total_money["is_fraud"].map({0: "Legitimate", 1: "Fraud"})
            fig = px.bar(
                total_money, x="Status", y="transaction_amount", text_auto=".2s",
                color="Status", color_discrete_map={"Legitimate": "#2563eb", "Fraud": "#dc2626"}
            )
            st.plotly_chart(apply_plotly_theme(fig), use_container_width=True)

    elif chart == "PIN Changed Recently":
        if "pin_changed_recently" in df_filtered.columns:
            pin_counts = df_filtered["pin_changed_recently"].value_counts().reset_index()
            pin_counts.columns = ["pin_changed_recently", "count"]
            pin_counts["Status"] = pin_counts["pin_changed_recently"].map({0: "Never Changed", 1: "Changed Recently"})
            fig = px.pie(pin_counts, names="Status", values="count", color_discrete_sequence=["#475569", "#f59e0b"])
            st.plotly_chart(apply_plotly_theme(fig), use_container_width=True)

    elif chart == "Fraud Distribution":
        if "is_fraud" in df_filtered.columns:
            fraud_counts = df_filtered["is_fraud"].value_counts().reset_index()
            fraud_counts.columns = ["is_fraud", "count"]
            fraud_counts["Status"] = fraud_counts["is_fraud"].map({0: "Legitimate", 1: "Fraud"})
            fig = px.bar(
                fraud_counts, x="Status", y="count", text_auto=True,
                color="Status", color_discrete_map={"Legitimate": "#2563eb", "Fraud": "#dc2626"}
            )
            st.plotly_chart(apply_plotly_theme(fig), use_container_width=True)

    elif chart == "Transaction Amount Distribution":
        if "transaction_amount" in df_filtered.columns:
            sample_df = df_filtered.sample(min(len(df_filtered), 50000), random_state=42)
            fig = px.histogram(sample_df, x="transaction_amount", nbins=50, color_discrete_sequence=["#3b82f6"])
            st.plotly_chart(apply_plotly_theme(fig), use_container_width=True)

    elif chart == "Customer Age Distribution":
        if "customer_age" in df_filtered.columns:
            sample_df = df_filtered.sample(min(len(df_filtered), 50000), random_state=42)
            fig = px.histogram(sample_df, x="customer_age", nbins=30, color_discrete_sequence=["#10b981"])
            st.plotly_chart(apply_plotly_theme(fig), use_container_width=True)


# =========================
# Relationships Analysis
# =========================
elif page == "Relationships":
    st.markdown('<div class="main-title">🔗 Relational Explorations</div>', unsafe_allow_html=True)
    st.write("")

    chart = st.selectbox(
        "Select Multi-Variable Intersections",
        [
            "Fraud by Customer Age",
            "Fraud by Age Group",
            "Fraud Rate: Night vs Day",
            "Fraud Rate: Weekend vs Weekday",
            "Merchant Category Comparison",
            "Fraud by Payment Method",
            "Fraud Type Distribution",
            "Correlation Heatmap"
        ]
    )

    if chart == "Fraud by Customer Age":
        if {"is_fraud", "customer_age"}.issubset(df_filtered.columns):
            fraud_data = df_filtered[df_filtered["is_fraud"] == 1]
            sample_df = fraud_data.sample(min(len(fraud_data), 50000), random_state=42) if len(fraud_data) > 0 else fraud_data
            fig = px.histogram(sample_df, x="customer_age", nbins=25, color_discrete_sequence=["#dc2626"])
            st.plotly_chart(apply_plotly_theme(fig), use_container_width=True)

    elif chart == "Fraud by Age Group":
        if {"is_fraud", "age_group"}.issubset(df_filtered.columns):
            fraud_data = df_filtered[df_filtered["is_fraud"] == 1]
            age_group_counts = fraud_data["age_group"].value_counts().reset_index()
            age_group_counts.columns = ["Age Group", "Count"]
            fig = px.bar(age_group_counts, x="Age Group", y="Count", text_auto=True, color_discrete_sequence=["#ef4444"])
            st.plotly_chart(apply_plotly_theme(fig), use_container_width=True)

    elif chart == "Fraud Rate: Night vs Day":
        if {"is_night_transaction", "is_fraud"}.issubset(df_filtered.columns):
            night_fraud = df_filtered.groupby("is_night_transaction")["is_fraud"].mean().mul(100).reset_index()
            night_fraud["Period"] = night_fraud["is_night_transaction"].map({0: "Day", 1: "Night"})
            fig = px.bar(night_fraud, x="Period", y="is_fraud", text_auto=".2f", color="Period", color_discrete_sequence=["#f59e0b", "#1e1b4b"])
            fig.update_layout(yaxis_title="Fraud Rate (%)")
            st.plotly_chart(apply_plotly_theme(fig), use_container_width=True)

    elif chart == "Fraud Rate: Weekend vs Weekday":
        if {"is_weekend", "is_fraud"}.issubset(df_filtered.columns):
            weekend_fraud = df_filtered.groupby("is_weekend")["is_fraud"].mean().mul(100).reset_index()
            weekend_fraud["Period"] = weekend_fraud["is_weekend"].map({0: "Weekday", 1: "Weekend"})
            fig = px.bar(weekend_fraud, x="Period", y="is_fraud", text_auto=".2f", color="Period", color_discrete_sequence=["#64748b", "#3b82f6"])
            fig.update_layout(yaxis_title="Fraud Rate (%)")
            st.plotly_chart(apply_plotly_theme(fig), use_container_width=True)

    elif chart == "Merchant Category Comparison":
        if {"merchant_category", "is_fraud"}.issubset(df_filtered.columns):
            merchant_summary = df_filtered.groupby("merchant_category")["is_fraud"].agg(
                total_transactions="count", fraud_transactions="sum", fraud_rate="mean"
            ).reset_index()
            merchant_summary["fraud_rate"] *= 100
            merchant_summary = merchant_summary.sort_values("fraud_transactions", ascending=False)

            fig = px.bar(merchant_summary, y="merchant_category", x="fraud_transactions", orientation="h", text_auto=True, color_discrete_sequence=["#6366f1"])
            st.plotly_chart(apply_plotly_theme(fig), use_container_width=True)
            st.dataframe(merchant_summary, use_container_width=True)

    elif chart == "Fraud by Payment Method":
        if {"payment_method", "is_fraud"}.issubset(df_filtered.columns):
            fraud_data = df_filtered[df_filtered["is_fraud"] == 1]
            payment_counts = fraud_data["payment_method"].value_counts().reset_index()
            payment_counts.columns = ["Payment Method", "Count"]
            fig = px.bar(payment_counts, x="Payment Method", y="Count", text_auto=True, color_discrete_sequence=["#06b6d4"])
            st.plotly_chart(apply_plotly_theme(fig), use_container_width=True)

    elif chart == "Fraud Type Distribution":
        if {"fraud_type", "is_fraud"}.issubset(df_filtered.columns):
            fraud_data = df_filtered[df_filtered["is_fraud"] == 1]
            fraud_type_counts = fraud_data["fraud_type"].value_counts().reset_index()
            fraud_type_counts.columns = ["Fraud Type", "Count"]
            fig = px.bar(fraud_type_counts, x="Fraud Type", y="Count", text_auto=True, color_discrete_sequence=["#ec4899"])
            st.plotly_chart(apply_plotly_theme(fig), use_container_width=True)

    elif chart == "Correlation Heatmap":
        numeric_df = df_filtered.select_dtypes(include=np.number)
        if numeric_df.shape[1] > 1:
            corr = numeric_df.corr()
            fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r")
            st.plotly_chart(apply_plotly_theme(fig), use_container_width=True)


# =========================
# Final Insights Page
# =========================
elif page == "Final Insights":
    st.markdown('<div class="main-title">💡 Strategic Findings & Insights</div>', unsafe_allow_html=True)
    st.write("")
    show_kpis(df_filtered)

    st.markdown("""
    <div class="insight-box">
    <b>Data Density & Capacity:</b> The foundational source houses heavy relational transaction registers, creating a robust baseline for feature correlation and behavior tracking.
    </div>
    """, unsafe_allow_html=True)

    if "is_fraud" in df_filtered.columns:
        fraud_rate = df_filtered["is_fraud"].mean() * 100
        st.markdown(f"""
        <div class="warning-box">
        🚨 <b>Anomaly Critical Ratio:</b> Confirmed fraud entries account for approximately <b>{fraud_rate:.2f}%</b> of your current scoped records.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    ### 🔑 Definitive Analytical Takeaways

    * **Severe Class Imbalance:** The overwhelming baseline of data represents normal client behavior. Fraud remains a highly concentrated structural outlier requiring explicit conditional tracking.
    * **Temporal and Channel Triggers:** Anomaly occurrences spike systematically within nighttime transactional windows, particular merchant classifications, and specific payment processing channels.
    * **Demographic Clusters:** Discretized age binning (`age_group`) maps high-risk vectors immediately to targeted age brackets, revealing vulnerability patterns.
    * **Interactive Operational Utility:** Moving static calculations from a linear notebook pipeline into an multi-filter Streamlit UI gives instant, dynamic segmentation capability to risk stakeholders.
    """)
