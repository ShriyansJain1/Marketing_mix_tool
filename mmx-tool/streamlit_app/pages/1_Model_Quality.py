import streamlit as st
import matplotlib.pyplot as plt
from utils import load_csv, load_parquet

st.header("📈 Model Quality")

metrics = load_csv("08_reporting/model_metrics.csv")
st.dataframe(metrics)

pred = load_parquet("07_model_output/validation_predictions.parquet")

fig, ax = plt.subplots()
ax.scatter(pred["sales"], pred["predicted_sales"], alpha=0.4)
ax.plot(
    [pred["sales"].min(), pred["sales"].max()],
    [pred["sales"].min(), pred["sales"].max()],
    linestyle="--"
)
ax.set_xlabel("Actual Sales")
ax.set_ylabel("Predicted Sales")

st.pyplot(fig)
