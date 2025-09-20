import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Title
st.title("E-commerce Dataset EDA App")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Show dataset preview
    st.subheader("Dataset Preview")
    st.write(df.head())

    # Show basic info
    st.subheader("Summary Statistics")
    st.write(df.describe(include="all"))

    # Check missing values
    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    # Column selection for analysis
    st.subheader("Column-wise Analysis")
    column = st.selectbox("Select a column for analysis", df.columns)

    if pd.api.types.is_numeric_dtype(df[column]):
        st.write(f"Summary of {column}:")
        st.write(df[column].describe())

        # Histogram
        fig, ax = plt.subplots()
        df[column].hist(ax=ax, bins=20)
        ax.set_title(f"Histogram of {column}")
        st.pyplot(fig)

        # Boxplot
        fig, ax = plt.subplots()
        ax.boxplot(df[column].dropna())
        ax.set_title(f"Boxplot of {column}")
        st.pyplot(fig)

    else:
        st.write(f"Value counts of {column}:")
        st.write(df[column].value_counts())

        # Bar chart
        fig, ax = plt.subplots()
        df[column].value_counts().plot(kind="bar", ax=ax)
        ax.set_title(f"Bar Chart of {column}")
        st.pyplot(fig)

    # Correlation Heatmap
    st.subheader("Correlation Heatmap (Numeric Features)")
    numeric_df = df.select_dtypes(include=["number"])
    if not numeric_df.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.matshow(numeric_df.corr(), cmap="coolwarm")
        fig.colorbar(cax)
        ax.set_xticks(range(len(numeric_df.columns)))
        ax.set_yticks(range(len(numeric_df.columns)))
        ax.set_xticklabels(numeric_df.columns, rotation=90)
        ax.set_yticklabels(numeric_df.columns)
        st.pyplot(fig)

    # Top Categories for Categorical Columns
    st.subheader("Top Categories Overview")
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols[:3]:  # limit to first 3 for simplicity
        st.write(f"Top categories in {col}:")
        st.write(df[col].value_counts().head())

        fig, ax = plt.subplots()
        df[col].value_counts().head(10).plot(kind="bar", ax=ax)
        ax.set_title(f"Top 10 Categories of {col}")
        st.pyplot(fig)
