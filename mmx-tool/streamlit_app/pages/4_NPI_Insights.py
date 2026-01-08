import streamlit as st
import matplotlib.pyplot as plt
from utils import load_parquet

st.header("🧠 NPI-Level Responsiveness")

npi = load_parquet("07_model_output/npi_random_effects.parquet")

st.dataframe(npi.head(20))

fig, ax = plt.subplots()
ax.hist(npi.iloc[:, 1], bins=30)
ax.set_xlabel("Random Intercept")

st.pyplot(fig)
