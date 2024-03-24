import pandas as pd

def get_value_from_csv(filename, column_name, selection_criteria):
    # Read the CSV file
    df = pd.read_csv(filename)

    # Apply the selection criteria
    # For example, if the criteria is that 'status' should be 1 and 'batch' should be 3
    filtered_df = df[(df['tsn'] == selection_criteria['tsn']) & (df['ch'] == selection_criteria['ch'])]

    # Retrieve the first value from the specified column
    # This example assumes you want the first value that matches the criteria
    if not filtered_df.empty:
        return filtered_df.iloc[0][column_name]
    else:
        return "No matching data found"

# Example usage
filename = 'bychannel.csv'
column_name = 'pde_2.5'
selection_criteria = {'ch': 1, 'tsn': 4838}

value = get_value_from_csv(filename, column_name, selection_criteria)
print("Retrieved Value:", value)

