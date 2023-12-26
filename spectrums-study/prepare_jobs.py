import os
import sys
import pandas as pd
import numpy as np

# Replace these file paths with the actual paths to your CSV files
input_csv = "test.csv"  # CSV containing "tsn" and "ch"
input_csv = os.path.abspath(sys.argv[1])
output_dir = os.path.abspath(sys.argv[2])
if not os.path.isdir(output_dir + "/jobs"):
    os.makedirs(output_dir + "/jobs")
run_pos_csv = "/Users/wanghanwen/codes/plotter/spectrums-study/merged_csv_Dec10.csv"  # CSV containing "run" and "pos" for each "tsn"
root_path = os.getcwd()#""
script_path = os.getcwd()
print(script_path)
# Read the input CSV file
df = pd.read_csv(input_csv)

# Define the volumes to loop over
volumes = [1, 2, 3, 4, 5, 6]

# Read the run and pos CSV file
run_pos_data = pd.read_csv(run_pos_csv)
scripts = [
"amplitude_spectra.py",
"charge_spectra.py",
"fit_spectrum.py",
"plot_waveforms.py",
]
# Start generating the scripts
for index, row in df.iterrows():
    tsn = row['tsn']
    ch = row['ch']
    print(tsn,ch)

    # Filter the run_pos_data for the current tsn
    selected_data = run_pos_data[run_pos_data['tsn'] == tsn]
    runs = np.unique(selected_data['run'].to_numpy())
    poss = np.unique(selected_data['pos'].to_numpy())

    print(runs)
    # Create a bash script for the current 'tsn' and 'ch'
    script_name = f"job_script_tsn_{tsn}_ch_{ch}.sh"
    with open(f"{output_dir}/jobs/" + script_name, 'w') as file:
        file.write("#!/bin/bash\n")
        file.write("#SBATCH --job-name=Job_TSN_{}\n".format(tsn))  # Example of a SLURM directive
        file.write('. /workfs2/juno/wanghanwen/sipm-massive/env_lcg.sh\n')

        # Loop over runs, positions, and volumes to generate commands
        for run in runs:
            for pos in poss:
                for vol in volumes:
                    for scr in scripts:
                        cmd = (f"python {script_path}/{scr} {root_path}/main_run_{str(run).zfill(4)}_ov_{vol}.00_sipmgr_{str(ch).zfill(2)}_tile.root "
                               f"{run_pos_csv} {pos} {output_dir}/outputs/{tsn}-{ch}\n")
                        file.write(cmd)

print("Bash scripts have been generated.")

