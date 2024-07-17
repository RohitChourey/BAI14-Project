def future_value_of_annuity(monthly_investment, months, monthly_return, best_weights):
    def objective(x):
        return x[0] + x[1]
    
    def constraints(x, RE_years, RD_years, p_e, p_d, Goal):
        return [
        {'type': 'ineq', 'fun': lambda x, lb = Goal, i = 0: (RE_years*x[0]) + (RD_years*x[1]) - lb},
        {'type': 'ineq', 'fun': lambda x, lb = 0.0, i = 1: p_e * x[0] + p_e * x[1] - x[0] - lb}
        ]
    
    def solve(RE_years, RD_years, months, p_e, p_d, Goal):
        x0 = np.array([Goal*p_e/months, Goal*p_d/months])
        bnds = ((0, None), (0, None))
        # Using scipy's minimize function for optimization
        res = minimize(objective, x0, args=(), bounds=bnds, method='SLSQP', constraints=constraints(x0, RE_years, RD_years, p_e, p_d, Goal))
        return (res.x[0]+res.x[1])
    
    def goal_programming_solver(combined_returns, months, best_weights, Goal):
        p_d = best_weights[0]
        p_e = best_weights[1]
        RE = combined_returns[0]
        RD = combined_returns[1]

        years = months/12
        # Getting the original solution
        RE_years = 12*((((1+RE)**Years)-1)/RE)
        RD_years = 12*((((1+RD)**Years)-1)/RD)
        original_solution = solve(RE_years, RD_years, Years, p_e, p_d, Goal)
        # perturbations = [0.9, 0.95, 1.05, 1.1]
        # perturbed_solutions = []
        # # Looping through perturbations and getting perturbed solutions
        # for perturbation in perturbations:
        #     perturbed_solution = solve(RE*perturbation, RD*perturbation, Years, p_e*perturbation, p_d*perturbation, Goal*perturbation)
        #     perturbed_solutions.append(perturbed_solution)
        return original_solution