import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
from app_working_V5 import GBWM_dynamic_Programming_V2
from app_working_V6 import GBWM_dynamic_programming_v1


def main():
    st.sidebar.title("Choose monthly investment type")
    program_selection = st.sidebar.selectbox("Select Investment Type", ["Varied monthly investment", "Fixed monthly investment"])

    if program_selection == "Fixed monthly investment":
        GBWM_dynamic_programming_v1()
    elif program_selection == "Varied monthly investment":
        GBWM_dynamic_Programming_V2()


if __name__ == "__main__":
    main()