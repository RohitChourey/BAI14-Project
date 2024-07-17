import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from forward_recursion import forward_recursion_goal_programming
from backward_recursion import backward_recursion_goal_programming
from data_processing import process_data

def get_user_data(file):
    try:
        return pd.read_excel(file)
    except Exception as e:
        st.error(f"Error reading Excel file: {e}")
        return None

def plot_history(history, title, ylabel):
    plt.figure(figsize=(10, 6))
    for client_name, client_history in history.items():
        plt.plot(client_history, label=client_name)
    plt.title(title)
    plt.xlabel("Time (months)")
    plt.ylabel(ylabel)
    plt.legend()
    st.pyplot(plt.gcf())
    plt.close()

def streamlit_ui():
    st.title("Investment Strategy Optimization")

    recursion_method = st.radio("Select Recursion Method", ('Forward Recursion', 'Backward Recursion'))
    uploaded_file = st.file_uploader("Upload an Excel file with user goals", type=["xlsx"])

    if uploaded_file is not None:
        user_data = get_user_data(uploaded_file)
        if user_data is not None:
            st.write("User Data:")
            st.write(user_data)

            monthly_investment_needed_df = pd.DataFrame(columns=['Client', 'Goal', 'Years', 'Priority', 'Monthly Investment Needed', 'Risk Tolerance'])
            weight_history = {}
            wealth_history = {}
            best_return_history = {}
            best_risk_history = {}
            monthly_investment_history = {}

            for index, user in user_data.iterrows():
                client_name = user['Client']
                risk_tolerance = user['Risk Tolerance']
                initial_wealth = user['Lumpsum Investment Amount']
                goals = []

                for i in range(1, 11):
                    goal_key = f'Goal_{i}'
                    years_key = f'Years_{i}'
                    priority_key = f'priority {i}'

                    if goal_key in user and pd.notna(user[goal_key]):
                        goal_amount = user[goal_key]
                        years = user[years_key]
                        priority = user[priority_key]

                        goals.append((goal_amount, years, priority))

                goals = sorted(goals, key=lambda x: x[2], reverse=True)

                for goal_index, (goal_amount, years, priority) in enumerate(goals):
                    combined_returns = process_data()

                    # combined_returns['Equity'] = combined_returns['Nifty50']
                    # combined_returns['Debt'] = combined_returns[['Debt Long', 'Debt Short']].mean(axis=1)
                    # combined_returns = combined_returns[['Equity', 'Debt']]

                    try:
                        if recursion_method == 'Forward Recursion':
                            min_monthly_investment, w_history, wealth_hist, return_hist, risk_hist, investment_hist = forward_recursion_goal_programming(goal_amount, years, risk_tolerance, combined_returns, initial_wealth)
                        else:
                            min_monthly_investment, w_history, wealth_hist, return_hist, risk_hist, investment_hist = backward_recursion_goal_programming(goal_amount, years, risk_tolerance, combined_returns, initial_wealth)
                    except ValueError as e:
                        st.error(f"Error for client {client_name}, goal {goal_amount}: {e}")
                        continue

                    weight_history[client_name] = w_history
                    wealth_history[client_name] = wealth_hist
                    best_return_history[client_name] = return_hist
                    best_risk_history[client_name] = risk_hist
                    monthly_investment_history[client_name] = investment_hist

                    new_entry = pd.DataFrame([{
                        'Client': client_name,
                        'Goal': goal_amount,
                        'Years': years,
                        'Priority': priority,
                        'Monthly Investment Needed': min_monthly_investment,
                        'Risk Tolerance': risk_tolerance
                    }])
                    monthly_investment_needed_df = pd.concat([monthly_investment_needed_df, new_entry], ignore_index=True)

            st.subheader("Monthly Investment Needed")
            st.write(monthly_investment_needed_df)

            st.subheader("Weight History")
            plot_history(weight_history, "Weight History Over Time", "Weight")

            st.subheader("Wealth History")
            plot_history(wealth_history, "Wealth History Over Time", "Wealth")

            st.subheader("Best Return History")
            plot_history(best_return_history, "Best Return History Over Time", "Return")

            st.subheader("Best Risk History")
            plot_history(best_risk_history, "Best Risk History Over Time", "Risk")

            st.subheader("Monthly Investment History")
            plot_history(monthly_investment_history, "Monthly Investment History Over Time", "Monthly Investment")

            towrite = BytesIO()
            monthly_investment_needed_df.to_excel(towrite, index=False, engine='openpyxl')
            towrite.seek(0)
            b64 = base64.b64encode(towrite.read()).decode()
            linko = f'<a href="data:application/octet-stream;base64,{b64}" download="monthly_investment_needed.xlsx">Download Excel file</a>'
            st.markdown(linko, unsafe_allow_html=True)

if __name__ == "__main__":
    streamlit_ui()
