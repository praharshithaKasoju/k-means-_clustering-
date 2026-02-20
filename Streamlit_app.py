import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Mall Customer Clustering",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model = joblib.load("kmeans_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

@st.cache_data
def load_data():
    return pd.read_csv("Mall_Customers.csv")

model, scaler = load_model()
df = load_data()

st.title("🛍️ Mall Customer Clustering Prediction")

if model is not None and df is not None:

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### 📊 Customer Information")

        age = st.slider(
            "👤 Age",
            int(df['Age'].min()),
            int(df['Age'].max()),
            30
        )

        annual_income = st.slider(
            "💰 Annual Income (k$)",
            int(df['Annual Income (k$)'].min()),
            int(df['Annual Income (k$)'].max()),
            50
        )

        spending_score = st.slider(
            "🎯 Spending Score (1-100)",
            1, 100, 50
        )

    with col2:
        st.markdown("### 📈 Dataset Statistics")
        st.metric("Total Customers", len(df))
        st.metric("Avg Age", f"{df['Age'].mean():.1f}")
        st.metric("Avg Income", f"{df['Annual Income (k$)'].mean():.1f}")

    st.markdown("---")

    # ---------------- PREDICTION ----------------
    if st.button("🚀 Predict Cluster"):

        # IMPORTANT: ONLY 2 FEATURES (same as training)
        input_data = pd.DataFrame({
            "Annual Income (k$)": [annual_income],
            "Spending Score (1-100)": [spending_score]
        })

        scaled_input = scaler.transform(input_data)
        cluster = model.predict(scaled_input)[0]

        st.success(f"Customer belongs to Cluster {cluster}")

        # ---------------- CREATE CLUSTERS FOR VISUALIZATION ----------------
        X = df[["Annual Income (k$)", "Spending Score (1-100)"]]
        X_scaled = scaler.transform(X)
        df["Cluster"] = model.predict(X_scaled)

        # ---------------- 2D SCATTER PLOT ----------------
        fig = px.scatter(
            df,
            x="Annual Income (k$)",
            y="Spending Score (1-100)",
            color=df["Cluster"].astype(str),
            title="Customer Clusters"
        )

        # Add prediction point
        fig.add_scatter(
            x=[annual_income],
            y=[spending_score],
            mode="markers",
            marker=dict(size=15, color="red"),
            name="Your Input"
        )

        st.plotly_chart(fig, use_container_width=True)

        # ---------------- PIE CHART ----------------
        cluster_counts = df["Cluster"].value_counts().sort_index()

        fig_pie = go.Figure(data=[go.Pie(
            labels=[f"Cluster {i}" for i in cluster_counts.index],
            values=cluster_counts.values
        )])

        fig_pie.update_layout(title="Customer Distribution by Cluster")
        st.plotly_chart(fig_pie, use_container_width=True)

else:
    st.error("Model or dataset not found.")