import os
import sys
import csv

# The name of the input text file
input_file_name = sys.argv[1]  # Replace with your file's name

# The name of the output CSV file
output_file_name = 'output.csv'

# Open the input file and read the lines
with open(input_file_name, 'r') as file:
    lines = file.readlines()

# Prepare the data to be written into the CSV file
data = []
for line in lines:
    # Split each line by the '-' character and strip any leading/trailing whitespace
    parts = line.strip().split('-')
    if len(parts) == 2:  # Ensure there are two parts (tsn and ch)
        data.append(parts)

# Open the output CSV file and write the data
with open(output_file_name, 'w', newline='') as csvfile:
    # Create a CSV writer object
    csvwriter = csv.writer(csvfile)

    # Write the header
    csvwriter.writerow(['tsn', 'ch'])

    # Write the data
    csvwriter.writerows(data)

print(f"Data has been successfully written to {output_file_name}")

