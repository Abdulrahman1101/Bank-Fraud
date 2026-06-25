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
# Custom CSS
# =========================
st.markdown("""
<style>
.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #002147;
}
.sub-title {
    font-size: 19px;
    color: #475569;
}
.metric-card {
    background-color: #F8FAFC;
    padding: 18px;
    border-radius: 14px;
    border: 1px solid #E2E8F0;
}
.insight-box {
    background-color: #EFF6FF;
    padding: 18px;
    border-radius: 14px;
    border-left: 6px solid #2563EB;
    font-size: 17px;
}
.warning-box {
    background-color: #FEF2F2;
    padding: 18px;
    border-radius: 14px;
    border-left: 6px solid #DC2626;
}
</style>
""", unsafe_allow_html=True)


# =========================
# Load Data
# =========================
@st.cache_data(show_spinner=True)
def load_data(uploaded_file=None):
    """
    Load dataset in this order:
    1. Uploaded CSV from Streamlit sidebar
    2. data/bank_fraud.csv from GitHub repo
    3. KaggleHub dataset
    """
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

    total_customers = df["customer_id"].nunique() if "customer_id" in df.columns else 0
    avg_amount = df["transaction_amount"].mean() if "transaction_amount" in df.columns else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions", safe_number(total_transactions))
    c2.metric("Fraud Transactions", safe_number(fraud_transactions))
    c3.metric("Fraud Rate", f"{fraud_rate:.2f}%")
    c4.metric("Average Amount", f"${avg_amount:,.2f}")


def apply_filters(df):
    filtered = df.copy()

    st.sidebar.header("🔎 Filters")

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
# Sidebar
# =========================
st.sidebar.title("🏦 Bank Fraud EDA")
uploaded_file = st.sidebar.file_uploader("Upload bank_fraud.csv", type=["csv"])

df_raw = load_data(uploaded_file)
if df_raw.empty:
    st.stop()

df = clean_data(df_raw)
df_filtered = apply_filters(df)

page = st.sidebar.radio(
    "Navigation",
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
# Home
# =========================
if page == "Home":
    st.markdown('<div class="main-title">🏦 Bank Fraud Detection EDA Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Interactive Streamlit dashboard based on your Bank Fraud EDA project.</div>',
        unsafe_allow_html=True
    )

    st.write("")
    show_kpis(df_filtered)

    st.write("")
    st.markdown("""
    ### Project Goal

    This dashboard analyzes bank transaction data to understand fraud patterns, compare legitimate and fraudulent transactions,
    and extract useful insights from transaction behavior, customer information, payment methods, and merchant categories.
    """)

    st.markdown("""
    ### EDA Workflow

    1. Load and look at the data  
    2. Understand the columns  
    3. Clean missing values and create new features  
    4. Explore one column at a time  
    5. Explore relationships between columns  
    6. Write final insights  
    """)

    if "is_fraud" in df_filtered.columns:
        fraud_counts = df_filtered["is_fraud"].value_counts().reset_index()
        fraud_counts.columns = ["is_fraud", "count"]
        fraud_counts["Status"] = fraud_counts["is_fraud"].map({0: "Legitimate", 1: "Fraud"})
        fig = px.pie(
            fraud_counts,
            names="Status",
            values="count",
            title="Fraud vs Legitimate Transactions"
        )
        st.plotly_chart(fig, use_container_width=True)


# =========================
# Data Overview
# =========================
elif page == "Data Overview":
    st.title("📊 Data Overview")

    show_kpis(df_filtered)

    st.subheader("Dataset Shape")
    st.write(f"Rows: **{df_filtered.shape[0]:,}**")
    st.write(f"Columns: **{df_filtered.shape[1]:,}**")

    st.subheader("First 10 Rows")
    st.dataframe(df_filtered.head(10), use_container_width=True)

    st.subheader("Column Types")
    dtype_df = pd.DataFrame({
        "Column": df_filtered.columns,
        "Data Type": df_filtered.dtypes.astype(str).values
    })
    st.dataframe(dtype_df, use_container_width=True)

    st.subheader("Numerical Summary")
    st.dataframe(df_filtered.describe().T, use_container_width=True)


# =========================
# Data Cleaning
# =========================
elif page == "Data Cleaning":
    st.title("🧹 Data Cleaning & Feature Engineering")

    st.subheader("Missing Values Before Cleaning")
    missing_before = df_raw.isnull().sum().sort_values(ascending=False).reset_index()
    missing_before.columns = ["Column", "Missing Values"]
    st.dataframe(missing_before, use_container_width=True)

    st.subheader("Missing Values After Cleaning")
    missing_after = df.isnull().sum().sort_values(ascending=False).reset_index()
    missing_after.columns = ["Column", "Missing Values"]
    st.dataframe(missing_after, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    Missing values in <b>fraud_type</b> were replaced with <b>Legitimate</b> because null values represent transactions that are not fraud.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Created Features")
    features = []
    if "customer_serial_num" in df.columns:
        features.append("customer_serial_num: gives each unique customer a serial number.")
    if "account_transaction_count" in df.columns:
        features.append("account_transaction_count: counts how many transactions each customer made.")
    if "age_group" in df.columns:
        features.append("age_group: groups customer ages into readable categories.")

    for feature in features:
        st.write("✅", feature)


# =========================
# Univariate Analysis
# =========================
elif page == "Univariate Analysis":
    st.title("📈 Univariate Analysis")

    show_kpis(df_filtered)

    chart = st.selectbox(
        "Choose a chart",
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
            total_money = (
                df_filtered.groupby("is_fraud")["transaction_amount"]
                .sum()
                .reset_index()
            )
            total_money["Status"] = total_money["is_fraud"].map({0: "Legitimate", 1: "Fraud"})
            fig = px.bar(
                total_money,
                x="Status",
                y="transaction_amount",
                text_auto=".2s",
                title="Total Monetary Value: Legitimate vs Fraud"
            )
            st.plotly_chart(fig, use_container_width=True)

    elif chart == "PIN Changed Recently":
        if "pin_changed_recently" in df_filtered.columns:
            pin_counts = df_filtered["pin_changed_recently"].value_counts().reset_index()
            pin_counts.columns = ["pin_changed_recently", "count"]
            pin_counts["Status"] = pin_counts["pin_changed_recently"].map({
                0: "Never Changed",
                1: "Changed Recently"
            })
            fig = px.pie(
                pin_counts,
                names="Status",
                values="count",
                title="PIN Changed Recently"
            )
            st.plotly_chart(fig, use_container_width=True)

    elif chart == "Fraud Distribution":
        if "is_fraud" in df_filtered.columns:
            fraud_counts = df_filtered["is_fraud"].value_counts().reset_index()
            fraud_counts.columns = ["is_fraud", "count"]
            fraud_counts["Status"] = fraud_counts["is_fraud"].map({0: "Legitimate", 1: "Fraud"})
            fig = px.bar(
                fraud_counts,
                x="Status",
                y="count",
                text_auto=True,
                title="Distribution of Fraud vs Legitimate Transactions"
            )
            st.plotly_chart(fig, use_container_width=True)

    elif chart == "Transaction Amount Distribution":
        if "transaction_amount" in df_filtered.columns:
            sample_df = df_filtered.sample(min(len(df_filtered), 50000), random_state=42)
            fig = px.histogram(
                sample_df,
                x="transaction_amount",
                nbins=50,
                title="Transaction Amount Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

    elif chart == "Customer Age Distribution":
        if "customer_age" in df_filtered.columns:
            sample_df = df_filtered.sample(min(len(df_filtered), 50000), random_state=42)
            fig = px.histogram(
                sample_df,
                x="customer_age",
                nbins=30,
                title="Customer Age Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)


# =========================
# Relationships
# =========================
elif page == "Relationships":
    st.title("🔗 Relationships Between Columns")

    chart = st.selectbox(
        "Choose relationship analysis",
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
            fig = px.histogram(
                sample_df,
                x="customer_age",
                nbins=25,
                title="Distribution of Fraudulent Transactions by Customer Age"
            )
            st.plotly_chart(fig, use_container_width=True)

    elif chart == "Fraud by Age Group":
        if {"is_fraud", "age_group"}.issubset(df_filtered.columns):
            fraud_data = df_filtered[df_filtered["is_fraud"] == 1]
            age_group_counts = fraud_data["age_group"].value_counts().reset_index()
            age_group_counts.columns = ["Age Group", "Count"]
            fig = px.bar(
                age_group_counts,
                x="Age Group",
                y="Count",
                text_auto=True,
                title="Fraudulent Transactions by Age Group"
            )
            st.plotly_chart(fig, use_container_width=True)

    elif chart == "Fraud Rate: Night vs Day":
        if {"is_night_transaction", "is_fraud"}.issubset(df_filtered.columns):
            night_fraud = (
                df_filtered.groupby("is_night_transaction")["is_fraud"]
                .mean()
                .mul(100)
                .reset_index()
            )
            night_fraud["Period"] = night_fraud["is_night_transaction"].map({
                0: "Day",
                1: "Night"
            })
            fig = px.bar(
                night_fraud,
                x="Period",
                y="is_fraud",
                text_auto=".2f",
                title="Fraud Rate: Night vs Day"
            )
            fig.update_layout(yaxis_title="Fraud Rate (%)")
            st.plotly_chart(fig, use_container_width=True)

    elif chart == "Fraud Rate: Weekend vs Weekday":
        if {"is_weekend", "is_fraud"}.issubset(df_filtered.columns):
            weekend_fraud = (
                df_filtered.groupby("is_weekend")["is_fraud"]
                .mean()
                .mul(100)
                .reset_index()
            )
            weekend_fraud["Period"] = weekend_fraud["is_weekend"].map({
                0: "Weekday",
                1: "Weekend"
            })
            fig = px.bar(
                weekend_fraud,
                x="Period",
                y="is_fraud",
                text_auto=".2f",
                title="Fraud Rate: Weekend vs Weekday"
            )
            fig.update_layout(yaxis_title="Fraud Rate (%)")
            st.plotly_chart(fig, use_container_width=True)

    elif chart == "Merchant Category Comparison":
        if {"merchant_category", "is_fraud"}.issubset(df_filtered.columns):
            merchant_summary = (
                df_filtered.groupby("merchant_category")
                .agg(
                    total_transactions=("is_fraud", "count"),
                    fraud_transactions=("is_fraud", "sum"),
                    fraud_rate=("is_fraud", "mean")
                )
                .reset_index()
            )
            merchant_summary["fraud_rate"] *= 100
            merchant_summary = merchant_summary.sort_values("fraud_transactions", ascending=False)

            fig = px.bar(
                merchant_summary,
                y="merchant_category",
                x="fraud_transactions",
                orientation="h",
                text_auto=True,
                title="Fraud Transactions by Merchant Category"
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Merchant Category Summary")
            st.dataframe(merchant_summary, use_container_width=True)

    elif chart == "Fraud by Payment Method":
        if {"payment_method", "is_fraud"}.issubset(df_filtered.columns):
            fraud_data = df_filtered[df_filtered["is_fraud"] == 1]
            payment_counts = fraud_data["payment_method"].value_counts().reset_index()
            payment_counts.columns = ["Payment Method", "Count"]
            fig = px.bar(
                payment_counts,
                x="Payment Method",
                y="Count",
                text_auto=True,
                title="Fraud Cases by Payment Method"
            )
            st.plotly_chart(fig, use_container_width=True)

    elif chart == "Fraud Type Distribution":
        if {"fraud_type", "is_fraud"}.issubset(df_filtered.columns):
            fraud_data = df_filtered[df_filtered["is_fraud"] == 1]
            fraud_type_counts = fraud_data["fraud_type"].value_counts().reset_index()
            fraud_type_counts.columns = ["Fraud Type", "Count"]
            fig = px.bar(
                fraud_type_counts,
                x="Fraud Type",
                y="Count",
                text_auto=True,
                title="Fraud Type Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

    elif chart == "Correlation Heatmap":
        numeric_df = df_filtered.select_dtypes(include=np.number)
        if numeric_df.shape[1] > 1:
            corr = numeric_df.corr()
            fig = px.imshow(
                corr,
                text_auto=".2f",
                aspect="auto",
                title="Correlation Heatmap"
            )
            st.plotly_chart(fig, use_container_width=True)


# =========================
# Final Insights
# =========================
elif page == "Final Insights":
    st.title("💡 Final Insights")

    show_kpis(df_filtered)

    st.markdown("""
    <div class="insight-box">
    ✅ The dataset contains a very large number of bank transactions, which makes it useful for exploring fraud patterns.
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    if "is_fraud" in df_filtered.columns:
        fraud_rate = df_filtered["is_fraud"].mean() * 100
        st.markdown(f"""
        <div class="warning-box">
        🚨 Fraud transactions represent approximately <b>{fraud_rate:.2f}%</b> of the filtered dataset.
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    st.markdown("""
    ### Main Findings

    - Most transactions are legitimate, while fraud transactions are much fewer.
    - `fraud_type` missing values were handled as `Legitimate`.
    - Fraud can be compared across payment methods, merchant categories, night transactions, and weekend transactions.
    - Age grouping helps identify which customer age groups appear more in fraud cases.
    - Merchant category and payment method are important columns for understanding fraud behavior.
    - The dashboard filters make the analysis easier and more interactive than the original notebook.
    """)

    st.markdown("""
    ### Conclusion

    This Streamlit dashboard converts the original EDA notebook into an interactive web application.
    It allows users to explore the data, understand fraud patterns, and present the results clearly using filters and visual charts.
    """)
