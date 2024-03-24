import os

# Define the directory containing your .txt files
directory = '/Users/wanghanwen/IHEPBox/TAO/TEST_0304'

# Iterate over all files in the directory
for filename in os.listdir(directory):
    # Check if the file is a .txt file
    if filename.endswith('.TXT'):
        # Construct the full file path
        file_path = os.path.join(directory, filename)
        # Open the file and read lines
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            lines = file.readlines()
        # Skip the first 4 lines and write the rest back to the file
        with open(file_path, 'w', encoding='utf-8') as file:
            file.writelines(lines[4:])

print("First 4 lines removed from all .txt files.")

