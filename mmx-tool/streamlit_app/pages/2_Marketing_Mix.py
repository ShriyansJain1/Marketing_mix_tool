import streamlit as st
import matplotlib.pyplot as plt
from utils import load_csv

st.header("📊 Marketing Mix Effects")

coef = load_csv("08_reporting/model_results.csv")
coef = coef[coef["feature"].str.contains("_sat")]

fig, ax = plt.subplots()
ax.barh(coef["feature"], coef["coefficient"])
ax.set_xlabel("Impact on Sales")

st.pyplot(fig)
