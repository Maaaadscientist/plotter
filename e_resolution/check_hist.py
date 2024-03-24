import os
import ROOT
import sys

def process_root_files(directory, file_pattern, hist_name):
    # List all files in the directory matching the pattern
    root_files = [f for f in os.listdir(directory) if f.endswith('.root') and file_pattern in f]

    # Loop over the files
    for filename in root_files:
        # Construct the full path to the file
        full_path = os.path.join(directory, filename)

        # Open the ROOT file
        root_file = ROOT.TFile(full_path, "READ")

        if root_file.IsOpen():
            #print(f"Processing file: {filename}")

            # Get the histogram
            histogram = root_file.Get(hist_name)

            if histogram:
                # Calculate the integral
                integral = histogram.Integral()
                if integral < 1000:
                    print(f"{filename}")
                    #print(f"Integral of {hist_name} in {filename}: {integral}")
            else:
                print(f"{filename}")

            # Close the ROOT file
            root_file.Close()
        else:
            print(f"Failed to open {filename}")

if __name__ == "__main__":
    directory = sys.argv[1]  # Update this with your directory path
    file_pattern = "hist_"
    hist_name = "dcr_poisson"

    process_root_files(directory, file_pattern, hist_name)

