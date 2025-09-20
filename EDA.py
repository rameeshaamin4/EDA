import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Streamlit page config
st.set_page_config(page_title="E-Commerce EDA App", layout="wide")

# Title
st.title("🛒 Comprehensive E-Commerce Exploratory Data Analysis (EDA)")

# File uploader
uploaded_file = st.file_uploader("Upload your E-Commerce CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Dataset preview
    st.subheader("📂 Dataset Preview")
    st.write(df.head())

    # Dataset shape
    st.subheader("📏 Dataset Shape")
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    # Data types
    st.subheader("🧾 Column Data Types")
    st.write(df.dtypes)

    # Missing values
    st.subheader("❌ Missing Values")
    st.write(df.isnull().sum())

    # Summary statistics
    st.subheader("📊 Summary Statistics")
    st.write(df.describe(include="all"))

    # Column-wise analysis
    st.subheader("🔍 Column-wise Analysis")
    column = st.selectbox("Select a column for analysis", df.columns)

    if pd.api.types.is_numeric_dtype(df[column]):
        st.write(f"Summary of {column}:")
        st.write(df[column].describe())

        # Histogram
        fig, ax = plt.subplots()
        df[column].hist(ax=ax, bins=20, color="skyblue", edgecolor="black")
        ax.set_title(f"Histogram of {column}")
        st.pyplot(fig)

        # Boxplot
        fig, ax = plt.subplots()
        sns.boxplot(x=df[column], ax=ax, color="lightgreen")
        ax.set_title(f"Boxplot of {column}")
        st.pyplot(fig)

        # Density plot
        fig, ax = plt.subplots()
        sns.kdeplot(df[column], ax=ax, fill=True, color="orange")
        ax.set_title(f"Density Plot of {column}")
        st.pyplot(fig)

    else:
        st.write(f"Value counts of {column}:")
        st.write(df[column].value_counts())

        # Bar plot
        fig, ax = plt.subplots()
        df[column].value_counts().plot(kind="bar", ax=ax, color="lightblue")
        ax.set_title(f"Bar Plot of {column}")
        st.pyplot(fig)

    # Correlation heatmap
    st.subheader("📌 Correlation Heatmap")
    numeric_df = df.select_dtypes(include=np.number)
    if not numeric_df.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)
    else:
        st.write("No numeric columns available for correlation heatmap.")

    # Scatter plot with regression line
    st.subheader("📉 Scatter Plot with Regression Line")
    numeric_columns = df.select_dtypes(include=np.number).columns
    if len(numeric_columns) >= 2:
        x_axis = st.selectbox("Select X-axis", numeric_columns, index=0)
        y_axis = st.selectbox("Select Y-axis", numeric_columns, index=1)

        fig, ax = plt.subplots()
        sns.regplot(x=df[x_axis], y=df[y_axis], ax=ax, scatter_kws={"alpha":0.6})
        ax.set_title(f"Regression Line: {x_axis} vs {y_axis}")
        st.pyplot(fig)
    else:
        st.write("Not enough numeric columns for scatter plot with regression line.")

    # Pairplot
    st.subheader("📷 Pairplot")
    if numeric_df.shape[1] <= 5 and df.shape[0] <= 500:
        fig = sns.pairplot(df[numeric_columns])
        st.pyplot(fig)
    else:
        st.write("Pairplot skipped (too many columns or rows).")

    # ============================
    # EXTRA E-COMMERCE-SPECIFIC EDA
    # ============================

    # Sales by Date
    if "InvoiceDate" in df.columns:
        st.subheader("🗓️ Sales Over Time")
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
        daily_sales = df.groupby(df["InvoiceDate"].dt.date).size()
        fig, ax = plt.subplots()
        daily_sales.plot(ax=ax, color="purple")
        ax.set_title("Daily Sales Count")
        st.pyplot(fig)

    # Top Products
    if "Product" in df.columns or "Description" in df.columns:
        prod_col = "Product" if "Product" in df.columns else "Description"
        st.subheader("🏆 Top 10 Products Sold")
        top_products = df[prod_col].value_counts().head(10)
        fig, ax = plt.subplots()
        top_products.plot(kind="bar", ax=ax, color="gold")
        ax.set_title("Top 10 Products")
        st.pyplot(fig)

    # Revenue Analysis
    if "Quantity" in df.columns and ("UnitPrice" in df.columns or "Price" in df.columns):
        price_col = "UnitPrice" if "UnitPrice" in df.columns else "Price"
        df["Revenue"] = df["Quantity"] * df[price_col]
        st.subheader("💰 Revenue Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df["Revenue"], bins=50, ax=ax, color="teal")
        ax.set_title("Revenue Distribution")
        st.pyplot(fig)

        st.write("Total Revenue:", df["Revenue"].sum())

    # Customer Analysis
    if "CustomerID" in df.columns:
        st.subheader("👥 Top 10 Customers by Purchases")
        top_customers = df["CustomerID"].value_counts().head(10)
        fig, ax = plt.subplots()
        top_customers.plot(kind="bar", ax=ax, color="salmon")
        ax.set_title("Top 10 Customers")
        st.pyplot(fig)

    # Geographic Analysis
    if "Country" in df.columns:
        st.subheader("🌍 Sales by Country")
        country_sales = df["Country"].value_counts().head(10)
        fig, ax = plt.subplots()
        country_sales.plot(kind="bar", ax=ax, color="skyblue")
        ax.set_title("Top 10 Countries by Sales")
        st.pyplot(fig)
