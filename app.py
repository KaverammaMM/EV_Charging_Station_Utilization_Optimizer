import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="EV Charging Dashboard", layout="wide")

st.title("⚡ EV Charging Station Utilization Dashboard")
st.write("Upload your EV charging dataset to explore insights.")

# ---------------------------
# File Upload
# ---------------------------
uploaded_file = st.file_uploader("Upload EV dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("Dataset loaded successfully!")

        st.subheader("📄 Data Preview")
        st.dataframe(df.head())

        # ---------------------------
        # Basic Info
        # ---------------------------
        st.subheader("📊 Dataset Summary")
        st.write("Shape:", df.shape)
        st.write("Missing values:")
        st.write(df.isnull().sum())

        # ---------------------------
        # Column selection safety
        # ---------------------------
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

        if len(numeric_cols) == 0:
            st.warning("No numeric columns found for visualization.")
        else:
            st.subheader("📈 Basic Analysis")

            col = st.selectbox("Select column for analysis", numeric_cols)

            # Histogram
            fig, ax = plt.subplots()
            ax.hist(df[col].dropna(), bins=20)
            ax.set_title(f"Distribution of {col}")
            st.pyplot(fig)

            # Boxplot
            fig2, ax2 = plt.subplots()
            ax2.boxplot(df[col].dropna())
            ax2.set_title(f"Boxplot of {col}")
            st.pyplot(fig2)

        # ---------------------------
        # Optional: Correlation heatmap
        # ---------------------------
        if len(numeric_cols) > 1:
            st.subheader("🔥 Correlation Heatmap")

            corr = df[numeric_cols].corr()

            fig3, ax3 = plt.subplots()
            cax = ax3.imshow(corr, cmap="coolwarm")

            ax3.set_xticks(range(len(numeric_cols)))
            ax3.set_yticks(range(len(numeric_cols)))
            ax3.set_xticklabels(numeric_cols, rotation=90)
            ax3.set_yticklabels(numeric_cols)

            fig3.colorbar(cax)
            st.pyplot(fig3)

    except Exception as e:
        st.error(f"Error reading file: {e}")

else:
    st.info("Please upload a CSV file to begin.")
