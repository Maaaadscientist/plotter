import os
import sys
import pandas as pd
def map_to_nearest_smaller_ap(file_path, lambda_input, enf_residual_input):
    if lambda_input < 0.002 or lambda_input > 0.998:
        return 0.
    if enf_residual_input < 0.0001:
        return 0.
    # Load the data
    data = pd.read_csv(file_path)
    
    # Find the closest lambda value in the data
    lambda_closest = data['lambda'].iloc[(data['lambda'] - lambda_input).abs().argsort()[:1]].values[0]
    
    # Filter the data for the closest lambda
    data_lambda_filtered = data[data['lambda'] == lambda_closest]
    
    # If the given enf_residual exceeds the range, use the row with the maximum 'enf_residual'
    if enf_residual_input > data_lambda_filtered['enf_residual'].max():
        ap_value = data_lambda_filtered.loc[data_lambda_filtered['enf_residual'].idxmax(), 'ap']
    else:
        # Find the two closest enf_residual values in the filtered data
        closest_indices = data_lambda_filtered['enf_residual'].sub(enf_residual_input).abs().nsmallest(4).index
        closest_data = data_lambda_filtered.loc[closest_indices]
        
        # Get the 'ap' value corresponding to the smaller of the two closest 'enf_residual' values
        ap_value = closest_data['ap'].min()
    
    return ap_value

file_path = sys.argv[1]
lambda_input_1 = float(sys.argv[2])
enf_residual_input_1 = float(sys.argv[3])
# Apply the updated function for enf_residual = 0.06 and 0.09
ap_value_updated_1 = map_to_nearest_smaller_ap(file_path, lambda_input_1, enf_residual_input_1)
print(ap_value_updated_1)

