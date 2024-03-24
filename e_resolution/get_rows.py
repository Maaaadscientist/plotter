import pandas as pd

# Load the CSV file
filename = 'bychannel.csv'
df = pd.read_csv(filename)

# Create a reference dictionary
reference_dict = {(row['tsn'], row['ch']): idx for idx, row in df.iterrows()}
# Get unique values from the 'tsn' column
unique_tsn_values = df['tsn'].unique()
# Function to get a specific value from a row based on tsn, ch, and column name
def get_value_by_tsn_ch(df, reference_dict, tsn, ch, column_name):
    row_index = reference_dict.get((tsn, ch))
    if row_index is not None and column_name in df.columns:
        return df.at[row_index, column_name]
    else:
        return "Value not found"


# Example usage
for tsn in unique_tsn_values:
    tsn_value = tsn # your tsn value
    ch_value = 5  # your ch value
    column_name = "pde_2.5"  # the column name from which you want to get the value
    
    value = get_value_by_tsn_ch(df, reference_dict, tsn_value, ch_value, column_name)
    
    print(tsn, value)

