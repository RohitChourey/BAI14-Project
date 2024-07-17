import pandas as pd
import numpy as np
import cvxpy as cp

def future_value_of_annuity(payment, rate, periods):
    return payment * ((1 + rate) ** periods - 1) / rate

def backward_recursion_goal_programming(goal_amount, years, risk_tolerance, combined_returns, initial_wealth):
    monthly_investment = cp.Variable()
    w_history = []
    wealth_hist = []
    return_hist = []
    risk_hist = []
    investment_hist = []

    for month in reversed(range(years * 12)):
        return_rate = combined_returns.mean(axis=0).values
        risk = combined_returns.std(axis=0).values

        weights = cp.Variable(2)
        constraints = [cp.sum(weights) == 1, weights >= 0]

        if risk_tolerance == 'high':
            objective = cp.Maximize(return_rate @ weights)
        else:
            objective = cp.Minimize(risk @ weights)

        problem = cp.Problem(objective, constraints)
        problem.solve()

        best_weights = weights.value
        w_history.append(best_weights)

        monthly_return = best_weights @ return_rate
        monthly_risk = best_weights @ risk

        wealth = future_value_of_annuity(monthly_investment.value, monthly_return, month)
        wealth_hist.append(wealth)
        return_hist.append(monthly_return)
        risk_hist.append(monthly_risk)
        investment_hist.append(monthly_investment.value)

    return monthly_investment.value, w_history, wealth_hist, return_hist, risk_hist, investment_hist
