import streamlit as st
import matplotlib.pyplot as plt
from utils import load_csv

st.header("💰 ROI & Budget Optimization")

roi = load_csv("08_reporting/roi_table.csv")
st.subheader("ROI by Channel")
st.dataframe(roi)

fig, ax = plt.subplots()
ax.bar(roi["channel"], roi["ROI"])
st.pyplot(fig)

budget = load_csv("08_reporting/budget_comparison.csv")

st.subheader("Budget Reallocation")
st.dataframe(budget)

budget.set_index("channel")[["current_budget", "optimized_budget"]].plot(
    kind="bar", ax=ax
)
st.pyplot(ax.figure)
