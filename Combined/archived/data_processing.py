import pandas as pd

def process_data():
    nifty50_data = pd.read_csv('Nifty 50 Historical Data.csv')
    debt_long_data = pd.read_csv('India 10-Year Bond Yield Historical Data.csv')
    debt_short_data = pd.read_csv('India 3-Month Bond Yield Historical Data.csv')

    # Ensure the 'Date' columns are in datetime format and sort by date
    nifty50_data['Date'] = pd.to_datetime(nifty50_data['Date'], format='%d-%m-%Y')
    debt_long_data['Date'] = pd.to_datetime(debt_long_data['Date'], format='%d-%m-%Y')
    debt_short_data['Date'] = pd.to_datetime(debt_short_data['Date'], format='%d-%m-%Y')

    nifty50_data = nifty50_data.sort_values(by='Date')
    debt_long_data = debt_long_data.sort_values(by='Date')
    debt_short_data = debt_short_data.sort_values(by='Date')

    # Convert the 'Change %' columns to numeric values
    nifty50_data['Change %'] = pd.to_numeric(nifty50_data['Change %'].str.rstrip('%')) / 100.0
    debt_long_data['Change %'] = pd.to_numeric(debt_long_data['Change %'].str.rstrip('%')) / 100.0
    debt_short_data['Change %'] = pd.to_numeric(debt_short_data['Change %'].str.rstrip('%')) / 100.0


    # Combine the returns into a single DataFrame
    combined_returns = pd.DataFrame({
        'Nifty50': nifty50_data['Change %'],
        'Debt Long': debt_long_data['Change %'],
        'Debt Short': debt_short_data['Change %']
    })
    print("Combined Returns DataFrame:")
    print(combined_returns.head())
    # Get the combined returns for equity and debt
    combined_returns1 = pd.DataFrame({})
    combined_returns1['Equity'] = combined_returns['Nifty50']
    combined_returns1['Debt'] = combined_returns[['Debt Long', 'Debt Short']].mean(axis=1)
    #combined_returns1 = combined_returns[['Equity', 'Debt']]
    
    return combined_returns1
