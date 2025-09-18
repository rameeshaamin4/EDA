import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Title
st.title("Comprehensive Exploratory Data Analysis (EDA) App")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Show dataset preview
    st.subheader("Dataset Preview")
    st.write(df.head())

    # Show dataset shape
    st.subheader("Dataset Shape")
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    # Show basic info
    st.subheader("Summary Statistics")
    st.write(df.describe(include="all"))

    # Missing values
    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    # Data types
    st.subheader("Column Data Types")
    st.write(df.dtypes)

    # Column selection for analysis
    st.subheader("Column-wise Analysis")
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

    # Correlation heatmap (only numeric)
    st.subheader("Correlation Heatmap")
    numeric_df = df.select_dtypes(include=np.number)
    if not numeric_df.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)
    else:
        st.write("No numeric columns available for correlation heatmap.")

    # Scatter plot with regression line
    st.subheader("Scatter Plot with Regression Line")
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

    # Pairplot (if dataset is small enough)
    st.subheader("Pairplot")
    if numeric_df.shape[1] <= 5 and df.shape[0] <= 500:
        fig = sns.pairplot(df[numeric_columns])
        st.pyplot(fig)
    else:
        st.write("Pairplot skipped (too many columns or rows).")
