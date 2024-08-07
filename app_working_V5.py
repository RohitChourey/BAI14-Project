'''
Modification History:
This code is based on code deployed on Streamlit site
1. Updated Cashflow as given in paper
2. 2 heatmap and wealth Trajectories

'''

import streamlit as st
from io import BytesIO
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from scipy.optimize import minimize
from scipy.optimize import linprog
import io
import openpyxl
import seaborn as sns
from scipy.stats import norm

# Define forward recursion functions
def load_data(nifty50_data, debt_long_data, debt_short_data):
    #nifty50_data = pd.read_csv('./historic_data/Nifty 50 Historical Data.csv')
    #debt_long_data = pd.read_csv('./historic_data/India 10-Year Bond Yield Historical Data.csv')
    #debt_short_data = pd.read_csv('./historic_data/India 3-Month Bond Yield Historical Data.csv')
    nifty50_data['Date'] = pd.to_datetime(nifty50_data['Date'], format='%d-%m-%Y')
    debt_long_data['Date'] = pd.to_datetime(debt_long_data['Date'], format='%d-%m-%Y')
    debt_short_data['Date'] = pd.to_datetime(debt_short_data['Date'], format='%d-%m-%Y')
    nifty50_data = nifty50_data.sort_values(by='Date')
    debt_long_data = debt_long_data.sort_values(by='Date')
    debt_short_data = debt_short_data.sort_values(by='Date')
    nifty50_data['Change %'] = pd.to_numeric(nifty50_data['Change %'].str.rstrip('%')) / 100.0
    debt_long_data['Change %'] = pd.to_numeric(debt_long_data['Change %'].str.rstrip('%')) / 100.0
    debt_short_data['Change %'] = pd.to_numeric(debt_short_data['Change %'].str.rstrip('%')) / 100.0
    combined_returns = pd.DataFrame({
        'Nifty50': nifty50_data['Change %'],
        'Debt Long': debt_long_data['Change %'],
        'Debt Short': debt_short_data['Change %']
    })
    combined_returns['Equity'] = combined_returns['Nifty50']
    combined_returns['Debt'] = combined_returns[['Debt Long', 'Debt Short']].mean(axis=1)
    return combined_returns[['Equity', 'Debt']]

# def future_value_of_annuity(monthly_investment, months, monthly_return, mean_returns, best_weights, Goal):
#     def objective(x):
#         return x[0] + x[1]

#     def constraints(x, RE_years, RD_years, p_e, p_d, Goal):
#         return [
#             {'type': 'ineq', 'fun': lambda x: (RE_years * x[0]) + (RD_years * x[1]) - Goal},
#             {'type': 'ineq', 'fun': lambda x: p_e * x[0] + p_d * x[1] - x[0]}
#         ]

#     def solve(RE_years, RD_years, months, p_e, p_d, Goal):
#         x0 = np.array([Goal * p_e / months, Goal * p_d / months])
#         bnds = ((0, None), (0, None))
#         res = minimize(objective, x0, bounds=bnds, method='SLSQP', constraints=constraints(x0, RE_years, RD_years, p_e, p_d, Goal))
#         return res.x[0] + res.x[1]

#     def goal_programming_solver(combined_returns, months, best_weights, Goal):
#         p_d = best_weights[0]
#         p_e = best_weights[1]
#         RE = combined_returns[0]
#         RD = combined_returns[1]

#         Years = months / 12
#         RE_years = 12 * (((1 + RE) ** Years - 1) / RE)
#         RD_years = 12 * (((1 + RD) ** Years - 1) / RD)
#         original_solution = solve(RE_years, RD_years, Years, p_e, p_d, Goal)
#         return original_solution

#     original_solution = goal_programming_solver(mean_returns, months, best_weights, Goal)
#     return original_solution

def future_value_of_annuity(monthly_investment, months, monthly_return):
    if monthly_return == 0:
        return monthly_investment * months
    else:
        return monthly_investment * (((1 + monthly_return) ** months - 1) / monthly_return)

def forward_recursion_goal_programming(financial_goal, time_horizon_years, risk_tolerance, data, initial_wealth):
    months = time_horizon_years * 12
    wealth_history = []
    weight_history = []
    best_return_history = []
    best_risk_history = []
    monthly_investment_history = []

    current_wealth = initial_wealth
    monthly_investment = 100  # Initial guess for monthly investment
    monthly_int = []
    for year in range(1, time_horizon_years + 1):
        end_date = 2014 + year
        current_data = data.loc[:str(end_date)]

        # Get the best return and risk for the given risk tolerance and current data
        best_weights, best_return, best_risk = get_best_return_and_risk_fr(risk_tolerance, current_data)
        #print('best return', best_return)
        #print('best weights', best_weights)
        best_return_history.append(best_return)
        best_risk_history.append(best_risk)
        weight_history.append(best_weights)
        #print('weight history', weight_history)

        # Update the current wealth with the monthly investment
        current_wealth += future_value_of_annuity(monthly_investment, 12, best_return)
        monthly_int.append(monthly_investment)
        #print(f"Year: {year}, Monthly Investment: {monthly_investment}, Current Wealth: {current_wealth}, Best Return: {best_return}")

        # Objective function to minimize the difference between future value and the financial goal
        def objective_function(monthly_investment):
            fv = future_value_of_annuity(monthly_investment, months - year * 12, best_return)
            return (fv + current_wealth - financial_goal) ** 2

        # Use scipy.optimize.minimize to find the minimum monthly investment
        result = minimize(objective_function, x0=[monthly_investment], bounds=[(0, None)])
        monthly_investment = result.x[0]
        monthly_investment_history.append(monthly_investment)

        #print(f"Year: {year}, Optimized Monthly Investment: {monthly_investment}")

        wealth_history.append(current_wealth)

    return monthly_investment, weight_history, wealth_history, best_return_history, best_risk_history, monthly_investment_history, monthly_int

def get_best_return_and_risk_fr(risk_tolerance, data):
    mean_returns = data.mean()
    cov_matrix = data.cov()
    cov_matrix = (cov_matrix + cov_matrix.T) / 2
    #print('mean return', mean_returns)
    def optimize_portfolio(target_return, mean_returns, cov_matrix):
        N = len(mean_returns)
        w = cp.Variable(N)
        objective = cp.Minimize(cp.quad_form(w, cov_matrix))
        constraints = [
            cp.sum(w) == 1,
            w @ mean_returns >= target_return,
            w >= 0
        ]
        problem = cp.Problem(objective, constraints)
        problem.solve()
        return w.value

    target_returns = np.linspace(mean_returns.min(), mean_returns.max(), 50)
    portfolio_risks = []
    portfolio_returns = []
    portfolio_weights = []

    for tr in target_returns:
        optimal_weights = optimize_portfolio(tr, mean_returns, cov_matrix)
        if optimal_weights is not None:
            portfolio_risks.append(np.sqrt(optimal_weights @ cov_matrix @ optimal_weights))
            portfolio_returns.append(optimal_weights @ mean_returns)
            portfolio_weights.append(optimal_weights)
    
        def get_best_portfolio(risk_tolerance):
            allocations = {
                'Conservative': 0.2,
                'Moderate': 0.4,
                'Assertive': 0.5,
                'Aggressive': 0.6,
                'Highly Aggressive': 0.8
            }
            if risk_tolerance not in allocations:
                raise ValueError("Invalid risk tolerance.")
        
            equity_alloc = allocations[risk_tolerance]
            debt_alloc = 1 - equity_alloc
            
            best_return_idx = np.argmax(portfolio_returns)
            min_risk_idx = np.argmin(portfolio_risks)
            
            combined_allocation = equity_alloc * np.array(portfolio_weights[best_return_idx]) + debt_alloc * np.array(portfolio_weights[min_risk_idx])
            combined_return = equity_alloc * portfolio_returns[best_return_idx] + debt_alloc * portfolio_returns[min_risk_idx]
            combined_risk = equity_alloc * portfolio_risks[best_return_idx] + debt_alloc * portfolio_risks[min_risk_idx]
            return combined_allocation, combined_return, combined_risk 
        
    return get_best_portfolio(risk_tolerance)

    def get_best_portfolio(risk_tolerance):
        allocations = {
            'Conservative': 0.2,
            'Moderate': 0.4,
            'Assertive': 0.5,
            'Aggressive': 0.6,
            'Highly Aggressive': 0.8
        }
        if risk_tolerance not in allocations:
            raise ValueError("Invalid risk tolerance.")
        
        equity_alloc = allocations[risk_tolerance]
        debt_alloc = 1 - equity_alloc
        
        best_return_idx = np.argmax(portfolio_returns)
        min_risk_idx = np.argmin(portfolio_risks)
        
        combined_allocation = equity_alloc * np.array(portfolio_weights[best_return_idx]) + debt_alloc * np.array(portfolio_weights[min_risk_idx])
        combined_return = equity_alloc * portfolio_returns[best_return_idx] + debt_alloc * portfolio_returns[min_risk_idx]
        combined_risk = equity_alloc * portfolio_risks[best_return_idx] + debt_alloc * portfolio_risks[min_risk_idx]
        
        return combined_allocation, combined_return, combined_risk 

    combined_allocation, combined_return, combined_risk = get_best_portfolio(risk_tolerance)
    return combined_allocation, combined_return, combined_risk, mean_returns

def process_clients(client_data, combined_returns):
    #downloadable_files_FR = []
    for index, row in client_data.iterrows():
        client_id = row['Client']
        initial_wealth = row['Lumpsum Investment Amount']
        risk_tolerance = row['Risk Tolerance']

        # Initialize lists to store the results for each client
        all_goals_results = []

        for goal_num in range(1, 11):  # Up to 10 goals per client
            goal_column = f'Goal_{goal_num}'
            years_column = f'Years_{goal_num}'
            if goal_column in row and years_column in row and pd.notna(row[goal_column]) and pd.notna(row[years_column]):
                financial_goal = row[goal_column]
                time_horizon_years = int(row[years_column])
                
                # Calculate the minimum monthly investment and other metrics needed to achieve the financial goal
                min_monthly_investment, weight_history, wealth_history, best_return_history, best_risk_history, monthly_investment_history, monthly_int = forward_recursion_goal_programming(financial_goal, time_horizon_years, risk_tolerance, combined_returns, initial_wealth)
                
                # Store the results for each goal
                goal_result = {
                    'Goal': financial_goal,
                    'Years': time_horizon_years,
                    'Min Monthly Investment': min_monthly_investment,
                    'Weight History': weight_history,
                    'Wealth History': wealth_history,
                    'Best Return History': best_return_history,
                    'Best Risk History': best_risk_history,
                    'Monthly Investment History': monthly_investment_history,
                    'Monthly Investment change': monthly_int
                }
                all_goals_results.append(goal_result)

        # Save results to an Excel file for each client
        client_filename = f'Client_{client_id}_Investment_Plan.xlsx'
        #path_name = f'.\model_output\Forward_R\{client_filename}'
        downloadable_files_FR = io.BytesIO()
        with pd.ExcelWriter(downloadable_files_FR, engine='openpyxl') as writer:
            for goal_num, goal_result in enumerate(all_goals_results, start=1):
                # Save each goal's results in separate sheets
                min_monthly_investment_df = pd.DataFrame([[goal_result['Min Monthly Investment']]], columns=['Minimum Monthly Investment'])
                min_monthly_investment_df.to_excel(writer, sheet_name=f'Goal_{goal_num}_Min Monthly Investment', index=False)

                weights_df = pd.DataFrame(goal_result['Weight History'], columns=['Equity', 'Debt'])
                weights_df.to_excel(writer, sheet_name=f'Goal_{goal_num}_Weights', index=False)

                wealth_history_df = pd.DataFrame(goal_result['Wealth History'], columns=["Wealth History"])
                wealth_history_df.to_excel(writer, sheet_name=f'Goal_{goal_num}_Wealth History', index=False)

                monthly_investment_df = pd.DataFrame(goal_result['Monthly Investment History'], columns=["Monthly Investment"])
                monthly_investment_df.to_excel(writer, sheet_name=f'Goal_{goal_num}_Monthly Investment', index=False)

                monthly_investment_change_df = pd.DataFrame(goal_result['Monthly Investment change'], columns=["Monthly Investment change"])
                monthly_investment_change_df.to_excel(writer, sheet_name=f'Goal_{goal_num}_Mnthly_inv_chnge', index=False)

                st.markdown(f"<h2 style='text-align: center; color: blue;'>Client: {client_id} and Goal: {goal_num} </h2>", unsafe_allow_html=True)
                st.markdown(f"<h4>Portfolio Return : {round(max(best_return_history), 4)}</h4>", unsafe_allow_html=True)
                #st.markdown(round(max(best_return_history), 4))
                st.markdown(f"<h4>Portfolio Risk : {round(max(best_risk_history), 4)} </h4>", unsafe_allow_html=True)
                #st.markdown(round(max(best_risk_history), 4))
                st.markdown(f"<h4>Suggested Minimum Monthly Investment : {round(goal_result['Min Monthly Investment'], 2)}</h4>", unsafe_allow_html=True)
                #st.markdown(round(goal_result['Min Monthly Investment'], 2))

                # Plotting the results
                #st.subheader(f'**Client: {client_id} and Goal: {goal_num}**')
                plt.figure(figsize=(14, 30))
                
                st.markdown("<h3 style='text-align: center; color: red;'>Wealth Level over Time</h4>", unsafe_allow_html=True)
                plt.figure(figsize=(10, 6))
                plt.plot(range(1, goal_result['Years'] + 1), goal_result['Wealth History'], marker='o')
                #plt.title('Wealth Level Over Time')
                plt.xlabel('Years')
                plt.ylabel('Wealth')
                plt.grid(True)
                st.pyplot()
                plt.close()
                
                st.markdown("<h3 style='text-align: center; color: red;'>Best Return and Risk over Time</h4>", unsafe_allow_html=True)
                plt.figure(figsize=(10, 6))
                plt.plot(range(1, goal_result['Years'] + 1), goal_result['Best Return History'], marker='o', label='Return')
                plt.plot(range(1, goal_result['Years'] + 1), goal_result['Best Risk History'], marker='x', label='Risk')
                #plt.title('Best Return and Risk Over Time')
                plt.xlabel('Years')
                plt.ylabel('Return/Risk')
                plt.xlim([0,10])
                plt.ylim([0,1])
                plt.legend()
                plt.grid(True)
                st.pyplot()
                plt.close()

                st.markdown("<h3 style='text-align: center; color: red;'>Portfolio weights over time</h4>", unsafe_allow_html=True)
                plt.figure(figsize=(10, 6))
                weights = np.array(goal_result['Weight History'])
                for i in range(weights.shape[1]):
                    plt.plot(range(1, goal_result['Years'] + 1), weights[:, i], marker='o', label=f'Asset {i + 1}')
                #plt.title('Portfolio Weights Over Time')
                plt.xlabel('Years')
                plt.ylabel('Weights')
                plt.xlim([0,10])
                plt.ylim([0,1])
                plt.legend()
                plt.grid(True)
                st.pyplot()
                plt.close()

                # plt.subplot(5, 1, 4)
                # plt.plot(range(1, goal_result['Years'] + 1), goal_result['Monthly Investment change'], marker='o')
                # plt.title('Monthly investment change')
                # plt.xlabel('Years')
                # plt.ylabel('Change in Monthly Investment')
                # plt.legend()
                # plt.grid(True)
                #plt.close()

                # plt.subplot(5, 1, 5)
                # plt.plot(range(1, goal_result['Years'] + 1), goal_result['Monthly Investment History'], marker='o')
                # plt.title('Monthly Investment Over Time')
                # plt.xlabel('Years')
                # plt.ylabel('Monthly Investment')
                # plt.grid(True)
                #plt.savefig(f"./model_output/Forward_R/Client_{client_id}_Goal_{goal_num}_plot.png")
                #st.pyplot()
                #plt.close()

                #workbook = writer.book
                #worksheet = workbook.add_worksheet(f"Goal_{goal_num}_Graph")
                #worksheet.insert_image('A1', f"Client_{client_id}_Goal_{goal_num}_plot_monthly_investment.png")
        st.download_button(
            label="Download Result",
            data=downloadable_files_FR,
            file_name=f"FR{client_id}.xlsx",
            mime='application/vnd.ms-excel'
        )    
    return weights_df, wealth_history_df, monthly_investment_df, monthly_investment_change_df, downloadable_files_FR

# Define backward recursion functions
def get_user_data(file_path):
    try:
        return pd.read_excel(file_path)
    except Exception as e:
        st.error(f"Error reading Excel file: {e}")
        return None

def get_best_return_and_risk(risk_tolerance, data):

    # Function to optimize portfolio for a given target return
    def optimize_portfolio(target_return, mean_returns, cov_matrix):
        N = len(mean_returns)
        w = cp.Variable(N)
        objective = cp.Minimize(cp.quad_form(w, cov_matrix))
        constraints = [
            cp.sum(w) == 1,  # Full investment constraint
            w @ mean_returns >= target_return,  # Target return constraint
            w >= 0  # No short selling constraint
        ]
        problem = cp.Problem(objective, constraints)
        problem.solve()
        return w.value

    mean_returns = data.mean()
    cov_matrix = data.cov()
    cov_matrix = (cov_matrix + cov_matrix.T) / 2
    # Calculate the efficient frontier
    target_returns = np.linspace(mean_returns.min(), mean_returns.max(), 50)
    portfolio_risks = []
    portfolio_returns = []
    portfolio_weights = []

    for tr in target_returns:
        optimal_weights = optimize_portfolio(tr, mean_returns, cov_matrix)
        portfolio_risks.append(np.sqrt(optimal_weights @ cov_matrix @ optimal_weights))
        portfolio_returns.append(optimal_weights @ mean_returns)
        portfolio_weights.append(optimal_weights)

    # Function to provide the best portfolio based on user risk tolerance
    def get_best_portfolio(risk_tolerance):
        # Define the allocations for each risk category
        allocations = {
            'Conservative': (0.2, 0.8),  # 20% in equity, 80% in debt
            'Moderate': (0.4, 0.6),      # 40% in equity, 60% in debt
            'Assertive': (0.5, 0.5),     # 50% in equity, 50% in debt
            'Aggressive': (0.6, 0.4),    # 60% in equity, 40% in debt
            'Highly Aggressive': (0.8, 0.2) # 80% in equity, 20% in debt
        }

        if risk_tolerance not in allocations:
            raise ValueError("Risk tolerance must be 'Conservative', 'Moderate', 'Assertive', 'Aggressive', or 'Highly Aggressive'.")

        # Get the corresponding allocation
        equity_alloc, debt_alloc = allocations[risk_tolerance]

        # Determine the number of equity and debt assets
        num_assets = len(mean_returns)
        #print('mean return', mean_returns, 'num_asset' ,num_assets)
        num_equity = max(1, round(equity_alloc * num_assets))  # Ensure at least one equity
        num_debt = num_assets - num_equity
        if num_debt == 0:
            num_debt = 0.2

        if num_equity == 0:
            raise ValueError("Number of equities is zero. Check the input data or logic determining equity allocation.")
        # Create an allocation array with the specified proportions
        w = np.zeros(num_assets)
        w[:num_equity] = equity_alloc / num_equity
        w[num_equity:] = debt_alloc / num_debt

        # Calculate the portfolio return and risk
        portfolio_return = w @ mean_returns
        portfolio_risk = np.sqrt(w @ cov_matrix @ w)

        #print('weights - w', w)

        return w, portfolio_return, portfolio_risk

    # Get best portfolio based on user risk tolerance
    best_weights, best_return, best_risk = get_best_portfolio(risk_tolerance)

    return best_return, best_risk, best_weights
# def calculate_cashflows(target_wealth, time_horizon, mu, sigma, dt):
#     cashflows = np.zeros(time_horizon)
#     required_wealth = target_wealth
#     for t in reversed(range(time_horizon)):
#         required_wealth /= np.exp((mu - 0.5 * sigma**2) * dt)
#         cashflow = required_wealth - (required_wealth / np.exp(mu * dt))
#         cashflows[t] = max(cashflow, 0)
#     return cashflows



def calculate_cashflows_min_max(W0, mu_min, mu_max, sigma, sigma_max, target_wealth, time_horizon, dt):
    # Ensure time_horizon is an integer
    #print('time Horizon',time_horizon)
    #print('dt', dt)
    time_horizon = int((len(time_horizon))/dt)
    
    def W_min(t, C):
        sum_C = sum(C[tau] * np.exp((mu_min - 0.5 * sigma**2) * (t - tau) - 3 * sigma_max * np.sqrt(t - tau)) for tau in range(t+1))
        return W0 * np.exp((mu_min - 0.5 * sigma**2) * t - 3 * sigma_max * np.sqrt(t)) + sum_C
    
    def W_max(t, C):
        sum_C = sum(C[tau] * np.exp((mu_max - 0.5 * sigma**2) * (t - tau) + 3 * sigma_max * np.sqrt(t - tau)) for tau in range(t+1))
        return W0 * np.exp((mu_max - 0.5 * sigma**2) * t + 3 * sigma_max * np.sqrt(t)) + sum_C

    # Initialize the cashflows array
    cashflows = np.zeros(time_horizon)
    
    for t in range(time_horizon):
        C_t = target_wealth / np.exp((mu_max - 0.5 * sigma**2) * (time_horizon - t) + 3 * sigma_max * np.sqrt(time_horizon - t))
        cashflows[t] = C_t

    return cashflows

def process_clients_backward(user_data, combined_returns):
    def utility_function(wealth):
        return np.log(wealth + 1)

    def next_wealth(wealth, allocation, mu, sigma, dt, cashflow):
        return (wealth + cashflow) * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.normal())

    # def calculate_transition_probabilities(wealth_levels, mu, sigma, dt):
    #     probabilities = np.zeros((len(wealth_levels), len(wealth_levels)))
    #     for i, W_i in enumerate(wealth_levels):
    #         for j, W_j in enumerate(wealth_levels):
    #             transition_return = (W_j / (W_i + 1e-6)) - 1  # Add a small value to avoid division by zero
    #             probabilities[i, j] = norm.pdf(transition_return, (mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt))
    #         probabilities[i, :] /= np.sum(probabilities[i, :])  # Normalize to make it a proper probability distribution
    #     return probabilities

    if user_data is None:
        raise ValueError("Failed to read user goals data.")

    monthly_investment_needed_df = pd.DataFrame(columns=['Client', 'Goal', 'Monthly Investment Needed', 'Risk Tolerance'])

    for index, user in user_data.iterrows():
        client_name = user['Client']
        risk_tolerance = user['Risk Tolerance']
        initial_wealth = 0
        goals = []

        for i in range(1, 11):
            goal_key = f'Goal_{i}'
            years_key = f'Years_{i}'
            priority_key = f'priority {i}'

            if goal_key in user and pd.notna(user[goal_key]):
                goal_amount = user[goal_key]
                years = user[years_key]
                priority = user[priority_key]

                if priority == 10:
                    initial_wealth = user['Lumpsum Investment Amount']
                else:
                    initial_wealth = 0

                goals.append((goal_amount, years, priority))

        client_filename = f"{client_name}_investment_plan.xlsx"
        downloadable_files_BR = io.BytesIO()

        with pd.ExcelWriter(downloadable_files_BR, engine='openpyxl') as writer:
            for goal_index, (goal_amount, years, priority) in enumerate(goals):
                target_wealth = goal_amount
                investment_horizon = years
                time_steps = investment_horizon

                try:
                    best_return, best_risk, portfolio_weights = get_best_return_and_risk(risk_tolerance, combined_returns)
                except ValueError as e:
                    print(f"Error for client {client_name}, goal {goal_amount}: {e}")
                    continue

                mu = best_return
                sigma = best_risk

                mu_min = sum(combined_returns.min()) / len(combined_returns.min())
                mu_max = sum(combined_returns.max()) / len(combined_returns.max())
                sigma_min = sum(combined_returns.std()) / len(combined_returns.std())
                sigma_max = combined_returns.max().std()

                wealth_levels = np.linspace(0, target_wealth, num=21)
                time_intervals = np.linspace(0, investment_horizon, num=time_steps + 1)

                value_function = np.zeros((len(wealth_levels), len(time_intervals)))
                value_function[:, -1] = utility_function(wealth_levels)

                dt = time_intervals[1] - time_intervals[0]
                cashflows = calculate_cashflows_min_max(initial_wealth, mu_min, mu_max, sigma_min, sigma_max, target_wealth, time_intervals, dt)

                # transition_probabilities = calculate_transition_probabilities(wealth_levels, mu, sigma, dt)

                for t in reversed(range(len(time_intervals) - 1)):
                    for w in range(len(wealth_levels)):
                        expected_values = []
                        for allocation in np.linspace(0, 1, num=11):
                            next_w = next_wealth(wealth_levels[w], allocation, mu, sigma, dt, cashflows[t])
                            next_w_index = np.searchsorted(wealth_levels, next_w, side='right') - 1
                            next_w_index = min(next_w_index, len(wealth_levels) - 1)
                            expected_value = value_function[next_w_index, t + 1]
                            expected_values.append(expected_value)
                        value_function[w, t] = np.max(expected_values)

                optimal_policy = np.zeros((len(wealth_levels), len(time_intervals) - 1))
                for t in range(len(time_intervals) - 1):
                    for w in range(len(wealth_levels)):
                        allocation_values = []
                        for allocation in np.linspace(0, 1, num=11):
                            next_w = next_wealth(wealth_levels[w], allocation, mu, sigma, dt, cashflows[t])
                            next_w_index = np.searchsorted(wealth_levels, next_w, side='right') - 1
                            next_w_index = min(next_w_index, len(wealth_levels) - 1)
                            expected_value = value_function[next_w_index, t + 1]
                            allocation_values.append(expected_value)
                        optimal_policy[w, t] = np.linspace(0, 1, num=11)[np.argmax(allocation_values)]

                optimal_policy_df = pd.DataFrame(optimal_policy, columns=[f'Time {t}' for t in range(len(time_intervals) - 1)], index=wealth_levels)
                optimal_policy_df.to_excel(writer, sheet_name=f"Optimal Policy {goal_index + 1}")

                stock_symbols = combined_returns.columns
                portfolio_weights = np.array(portfolio_weights)
                if portfolio_weights.ndim == 1:
                    portfolio_weights = portfolio_weights.reshape(1, -1)

                if portfolio_weights.shape[1] != len(stock_symbols):
                    raise ValueError(f"Shape of portfolio_weights {portfolio_weights.shape} does not match number of stock symbols {len(stock_symbols)}.")

                portfolio_weights_df = pd.DataFrame(portfolio_weights, columns=stock_symbols)
                portfolio_weights_df.to_excel(writer, sheet_name=f"Portfolio Weights {goal_index + 1}")

                num_simulations = 1000
                wealth_trajectories = np.zeros((num_simulations, len(time_intervals)))

                for i in range(num_simulations):
                    wealth = initial_wealth
                    wealth_trajectories[i, 0] = wealth
                    for t in range(len(time_intervals) - 1):
                        allocation = optimal_policy[np.searchsorted(wealth_levels, wealth) - 1, t]
                        wealth = next_wealth(wealth, allocation, mu, sigma, dt, cashflows[t])
                        wealth_trajectories[i, t + 1] = wealth

                probability_thresholds = np.linspace(0, target_wealth, num=50)
                cumulative_probabilities = np.zeros((len(time_intervals), len(probability_thresholds)))

                for t in range(len(time_intervals)):
                    for j, threshold in enumerate(probability_thresholds):
                        cumulative_probabilities[t, j] = np.mean(wealth_trajectories[:, t] <= threshold)

                
                
                st.markdown(f"<h2 style='text-align: center; color: blue;'>Client: {client_name} and Goal: {goal_index + 1} </h2>", unsafe_allow_html=True)

                wealth_trajectories_df = pd.DataFrame(wealth_trajectories, columns=[f'Year {int(t)}' for t in time_intervals])
                wealth_trajectories_df.to_excel(writer, sheet_name=f"Wealth Trajectories {goal_index + 1}")

                terminal_wealth = wealth_trajectories[:, -1]
                terminal_wealth_df = pd.DataFrame(terminal_wealth, columns=['Terminal Wealth'])
                terminal_wealth_summary = terminal_wealth_df.describe()
                st.markdown(f"<h4>Portfolio Return: {round(mu, 4)}</h4>", unsafe_allow_html=True)
                st.markdown(f"<h4>Portfolio Risk: {round(sigma, 4)}</h4>", unsafe_allow_html=True)
                st.markdown(f"<h4>Suggested Minimum Monthly Investment : {round((sum((cashflows) / len(cashflows)) / 12), 2)}</h4>", unsafe_allow_html=True)
                terminal_wealth_df.to_excel(writer, sheet_name=f"Terminal Wealth {goal_index + 1}")
                terminal_wealth_summary.to_excel(writer, sheet_name=f"Terminal Wealth Summary {goal_index + 1}")

                new_entry = pd.DataFrame([{
                    'Client': client_name,
                    'Goal': goal_amount,
                    'Years': years,
                    'Priority': priority,
                    'Monthly Investment Needed': cashflows[0],
                    'Risk Tolerance': risk_tolerance
                }])
                if not new_entry.empty:
                    monthly_investment_needed_df = pd.concat([monthly_investment_needed_df, new_entry], ignore_index=True)

                num_simulations = wealth_trajectories.shape[0]
                terminal_wealth = wealth_trajectories[:, -1]

                goal_achievement_probability = np.mean(terminal_wealth >= target_wealth)
                st.markdown(f"<h4>Goal Amount to Achieve : {goal_amount}</h4>", unsafe_allow_html=True)
                #st.markdown(f"<h4>Terminal wealth Achieved : {terminal_wealth}</h4>", unsafe_allow_html=True)
                st.markdown(f"<h4>Goal Achievement Probability : {goal_achievement_probability}</h4>", unsafe_allow_html=True)
                standard_deviation_terminal_wealth = np.std(terminal_wealth)
                st.markdown(f"<h4>Standard Deviation of Terminal Wealth : {standard_deviation_terminal_wealth}</h4>", unsafe_allow_html=True)
                value_at_risk_95 = np.percentile(terminal_wealth, 5)
                st.markdown(f"<h4>Value at Risk at 95% : {round(value_at_risk_95, 2)}</h4>", unsafe_allow_html=True)
                conditional_value_at_risk_95 = np.mean(terminal_wealth[terminal_wealth <= value_at_risk_95])
                st.markdown(f"<h4>Conditional value at risk at 95% : {round(conditional_value_at_risk_95, 2)}</h4>", unsafe_allow_html=True)
                st.markdown("<h4>Statistical Summary</h4>", unsafe_allow_html=True)
                st.dataframe(terminal_wealth_summary)
                
                st.markdown("<h3 style='text-align: center; color: red;'>Probability Distribution of Wealth Over Time</h4>", unsafe_allow_html=True)
                plt.figure(figsize=(20, 10))
                for t in range(len(time_intervals)):
                    plt.plot(probability_thresholds, 1 - cumulative_probabilities[t, :], label=f'Time {t}')
                plt.xlabel('Wealth')
                plt.ylabel('1 - Cumulative Probability')
                plt.legend()
                plt.grid(True)
                st.pyplot()
                plt.close()

                T = len(time_intervals) - 1
                WGrid = wealth_levels
                V = 1 - cumulative_probabilities[:, ::5].T

                st.markdown("<h3 style='text-align: center; color: red;'>Probability of Achieving Goal</h4>", unsafe_allow_html=True)
                plt.figure(figsize=(10, 8))
                sns.heatmap(V, xticklabels=range(T + 1), yticklabels=np.flip((WGrid[:21:2])), cmap="YlGnBu")
                plt.xlabel("Time Period")
                plt.ylabel("Wealth Level")
                #plt.title("Probability of Achieving Goal")
                st.pyplot()
                plt.close()

                metrics_df = pd.DataFrame({
                    'Metric': ['Goal Achievement Probability', 'Goal Amount', 'Standard Deviation of Terminal Wealth', 'Value at Risk (95%)', 'Conditional Value at Risk (95%)'],
                    'Value': [goal_achievement_probability, goal_amount, standard_deviation_terminal_wealth, value_at_risk_95, conditional_value_at_risk_95]
                })
                metrics_df.to_excel(writer, sheet_name=f'investment_strategy_metrics {goal_index + 1}', index=False)

                # New graph implementation
                WPath = wealth_trajectories.mean(axis=0)
                VPath = 1 - cumulative_probabilities.mean(axis=1)
                
                st.markdown("<h3 style='text-align: center; color: red;'>Optimal Strategy Outcome</h4>", unsafe_allow_html=True)
                plt.figure(figsize=(12, 8))
                fig, ax1 = plt.subplots()

                ax1.set_xlabel('Time Period')
                ax1.set_ylabel('Wealth Level', color='tab:blue')
                ax1.plot(range(T + 1), WPath, color='tab:blue', label="Wealth path")
                ax1.axhline(y=target_wealth, color='k', linestyle='--', label="Wealth goal")
                ax1.tick_params(axis='y', labelcolor='tab:blue')

                ax2 = ax1.twinx()
                ax2.set_ylabel('Probability of Success', color='tab:red')
                ax2.plot(range(T + 1), VPath, color='tab:red', label="Probability of Success")
                ax2.tick_params(axis='y', labelcolor='tab:red')

                fig.tight_layout()
                #plt.title('Optimal Strategy Outcome')
                fig.legend(loc="upper left", bbox_to_anchor=(0.1,0.9))
                plt.grid(True)
                st.pyplot(fig)
                plt.close()

                #print("optimal_policy:", optimal_policy)
                st.markdown("<h3 style='text-align: center; color: red;'>Optimal Portfolio Aggressiveness</h4>", unsafe_allow_html=True)
                plt.figure(figsize=(10, 8))
                sns.heatmap(np.flip(optimal_policy[::5, :], axis=0), 
                            xticklabels=range(T), 
                            yticklabels=np.flip(wealth_levels[::5]), 
                            cmap="YlGnBu", 
                            annot=True, 
                            linewidths=0.5, 
                            linecolor='black')
                plt.xlabel("Time Period")
                plt.ylabel("Wealth Level")
                #plt.title("Portfolio Aggressiveness")
                st.pyplot()
                plt.close()

        st.download_button(
            label="Download Result",
            data=downloadable_files_BR.getvalue(),
            file_name=f"BR{client_name}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    return monthly_investment_needed_df, terminal_wealth_summary, mu, sigma

def GBWM_dynamic_Programming_V2():
    # Streamlit UI
    st.title("Investment Goal Planning")
    st.sidebar.title("Choose Recursion Type")
    recursion_type = st.sidebar.selectbox("Select Recursion Type", ["Forward Recursion", "Backward Recursion", "Both"])

    nifty50_data = st.file_uploader("Upload your CSV file with nifty50_data", type=["csv"])
    debt_long_data = st.file_uploader("Upload your CSV file with debt_long_data_df", type=["csv"])
    debt_short_data = st.file_uploader("Upload your CSV file with debt_short_data_df", type=["csv"])
    uploaded_file = st.file_uploader("Upload your Excel file with client data", type=["xlsx"])

    if uploaded_file:
        st.set_option('deprecation.showPyplotGlobalUse', False)
        client_data = pd.read_excel(uploaded_file)
        nifty50_data_df = pd.read_csv(nifty50_data)
        debt_long_data_df = pd.read_csv(debt_long_data)
        debt_short_data_df = pd.read_csv(debt_short_data)
        combined_returns = load_data(nifty50_data_df, debt_long_data_df, debt_short_data_df)
        if recursion_type == "Forward Recursion":
            st.header("Forward Recursion Results")
            weights_df, wealth_history_df, monthly_investment_df, monthly_investment_change_df, downloadable_files_FR = process_clients(client_data, combined_returns)

            # st.download_button(
            #     label="Download Result",
            #     data=downloadable_files_FR,
            #     file_name="FR.xlsx",
            #     mime='application/vnd.ms-excel'
            # )

        elif recursion_type == "Backward Recursion":
            st.header("Backward Recursion Results")
            downloadable_files_BR, terminal_wealth_summary, mu, sigma = process_clients_backward(client_data, combined_returns)

            # st.download_button(
            #     label="Download Result",
            #     data=downloadable_files_BR,
            #     file_name="BR.xlsx",
            #     mime='application/vnd.ms-excel'
            # )
        elif recursion_type == "Both":
            st.header("Forward and Backward Recursion Results")
            st.subheader("Forward Recursion")
            weights_df, wealth_history_df, monthly_investment_df, monthly_investment_change_df, downloadable_files_FR = process_clients(client_data, combined_returns)

            # st.download_button(
            #     label="Download Result",
            #     data=downloadable_files_FR,
            #     file_name="FR.xlsx",
            #     mime='application/vnd.ms-excel'
            # )
            st.subheader("Backward Recursion")
            downloadable_files_BR, terminal_wealth_summary, mu, sigma = process_clients_backward(client_data, combined_returns)

            # st.download_button(
            #     label="Download Result",
            #     data=downloadable_files_BR,
            #     file_name="BR.xlsx",
            #     mime='application/vnd.ms-excel'
            # )