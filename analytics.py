import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from chatbot_backend import training

def analytics_tab():
    st.header("📊 Interactive Dataset Analytics & Distribution")

    # ---------------- Dataset Preview ----------------
    with st.expander("📄 View Dataset"):
        st.dataframe(training.head(100))

    # ---------------- Disease Distribution ----------------
    st.subheader("🦠 Disease Distribution")
    disease_counts = training["prognosis"].value_counts()
    fig1 = px.bar(
        disease_counts,
        x=disease_counts.index,
        y=disease_counts.values,
        color=disease_counts.values,
        color_continuous_scale="Viridis",
        labels={"x":"Disease", "y":"Number of Samples"},
        title="Number of Samples per Disease"
    )
    fig1.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig1, use_container_width=True)

    # ---------------- Symptom Frequency ----------------
    st.subheader("🤒 Most Common Symptoms")
    symptom_sum = training.drop("prognosis", axis=1).sum().sort_values(ascending=False)
    fig2 = px.bar(
        x=symptom_sum.index[:15],
        y=symptom_sum.values[:15],
        labels={"x":"Symptom", "y":"Occurrences"},
        color=symptom_sum.values[:15],
        color_continuous_scale="Turbo",
        title="Top 15 Most Frequent Symptoms"
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ---------------- PCA Scatter Plot ----------------
    st.subheader("📈 Disease Clustering (PCA)")
    X = training.drop("prognosis", axis=1)
    y = training["prognosis"]
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    pca_df = pd.DataFrame({"PC1": X_pca[:,0], "PC2": X_pca[:,1], "Disease": y})
    
    fig3 = px.scatter(
        pca_df, x="PC1", y="PC2", color="Disease",
        hover_data=["Disease"], opacity=0.7,
        color_discrete_sequence=px.colors.qualitative.Bold,
        title="2D PCA of Disease Samples"
    )
    st.plotly_chart(fig3, use_container_width=True)

    # ---------------- Symptom Correlation Heatmap ----------------
    st.subheader("🔥 Symptom Correlation Heatmap")
    corr = training.drop("prognosis", axis=1).corr()
    fig4 = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale="Viridis"
        )
    )
    fig4.update_layout(title="Correlation Between Symptoms", width=900, height=700)
    st.plotly_chart(fig4, use_container_width=True)

    # ---------------- Interactive Filtered Data ----------------
    st.subheader("🔍 Filter & Explore Data")
    disease_filter = st.multiselect("Select Disease(s) to explore:", options=training["prognosis"].unique())
    filtered_data = training[training["prognosis"].isin(disease_filter)] if disease_filter else training
    st.dataframe(filtered_data)

    st.success("💡 All visualizations are fully interactive! Hover, zoom, and filter to explore your dataset.")
