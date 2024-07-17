from scipy.optimize import minimize
import numpy as np
import cvxpy as cp


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
        best_weights, best_return, best_risk = get_best_return_and_risk(risk_tolerance, current_data)
        best_return_history.append(best_return)
        best_risk_history.append(best_risk)
        weight_history.append(best_weights)

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

    return monthly_investment, weight_history, wealth_history, best_return_history, 

def get_best_return_and_risk(risk_tolerance, data):
    mean_returns = data.mean()
    cov_matrix = data.cov()

    # Ensure the covariance matrix is symmetric
    cov_matrix = (cov_matrix + cov_matrix.T) / 2

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

    def get_best_portfolio(risk_tolerance):
        allocations = {
            'Conservative': (0.2, 0.8),  # 20% in equity, 80% in debt
            'Moderate': (0.4, 0.6),      # 40% in equity, 60% in debt
            'Assertive': (0.5, 0.5),     # 50% in equity, 50% in debt
            'Aggressive': (0.6, 0.4),    # 60% in equity, 40% in debt
            'Highly Aggressive': (0.8, 0.2) # 80% in equity, 20% in debt
        }

        if risk_tolerance not in allocations:
            raise ValueError("Risk tolerance must be 'Conservative', 'Moderate', 'Assertive', 'Aggressive', or 'Highly Aggressive'.")

        equity_alloc, debt_alloc = allocations[risk_tolerance]

        combined_allocation = equity_alloc * np.array(portfolio_weights[np.argmax(portfolio_returns)]) + debt_alloc * np.array(portfolio_weights[np.argmin(portfolio_risks)])
        return combined_allocation, np.max(portfolio_returns), np.min(portfolio_risks)

    return get_best_portfolio(risk_tolerance)