import pandas as pd
import numpy as np

def average_value_from_csv(df, column_name, selection_criteria=None):
    # Check if selection criteria are provided
    if selection_criteria:
        # Apply the selection criteria
        for key, value in selection_criteria.items():
            df = df[df[key] == value]

    # Calculate and return the average value of the specified column
    if not df.empty:
        return df[column_name].mean(), df[column_name].var()
    else:
        return "No matching data or empty column"

# Example usage
filename = 'with_ov_edge_mod.csv'
for i in range(46):
    ov = str(round(2.5 + i * 0.1, 1))
    ov_value = float(ov)
    df = pd.read_csv(filename)
    df = df[(df['ov_min'] <= ov_value) & (df['ov_max'] >= ov_value)]
    
    column_name = f'pde_{ov}'
    column_err_name = f'pde_{ov}_err'
    
    # With selection criteria
    selection_criteria = {'tsn': 4838}
    
    # Without selection criteria
    average_all, dev_all = average_value_from_csv(df, column_name)
    print(ov,  average_all, np.sqrt(dev_all))

