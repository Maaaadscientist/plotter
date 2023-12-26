import os
import sys
import pandas as pd
import numpy as np

# Replace these file paths with the actual paths to your CSV files
input_csv = os.path.abspath(sys.argv[1])
output_dir = os.path.abspath(sys.argv[2])
if not os.path.isdir(output_dir + "/jobs"):
    os.makedirs(output_dir + "/jobs")
#run_pos_csv = "/Users/wanghanwen/codes/plotter/spectrums-study/merged_csv_Dec10.csv"  # CSV containing "run" and "pos" for each "tsn"
run_pos_csv = "/junofs/users/wanghanwen/combined_csv_Dec26.csv"
root_path = "/junofs/users/wanghanwen/main-runs/main"#os.getcwd()#""
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
                        cmd = (f"python {script_path}/{scr} {root_path}/main_run_{str(run).zfill(4)}/root/main_run_{str(run).zfill(4)}_ov_{vol}.00_sipmgr_{str(ch).zfill(2)}_tile.root "
                               f"{run_pos_csv} {pos} {output_dir}/outputs/{tsn}-{ch}\n")
                        file.write(cmd)

print("Bash scripts have been generated.")
# Generate the initial_jobs_wrapper.sh script
wrapper_script_path = os.path.join(output_dir, 'initial_jobs_wrapper.sh')
with open(wrapper_script_path, 'w') as wrapper_script:
    wrapper_script.write("""#!/bin/bash
# Directory containing the job scripts
jobs_dir="{jobs_dir}"

# Argument handling for the specific job script to execute
if [ $# -ne 1 ]; then
    echo "Usage: $0 [ProcId]"
    exit 1
fi

procid=$1  # The provided ProcId determines which job script to run

# Get an array of job scripts
scripts=( $(ls $jobs_dir/job_script_tsn_*_ch_*.sh) )

# Calculate total available jobs
total_jobs=${{#scripts[@]}}

# Determine the job script to execute based on ProcId
if [ $procid -ge 0 ] && [ $procid -lt $total_jobs ]; then
    script_to_run="${{scripts[$procid]}}"
    echo "Running job script: $script_to_run"
    bash "$script_to_run"
else
    echo "Error: ProcId $procid is out of range. Only $total_jobs jobs are available."
    exit 1
fi
""".format(jobs_dir=os.path.join(output_dir, 'jobs')))
# Make the wrapper script executable
os.chmod(wrapper_script_path, 0o755)

print("Wrapper script has been generated.")

# Calculate the number of job scripts generated
num_job_scripts = len(os.listdir(os.path.join(output_dir, 'jobs')))

# Ask user if they want to submit jobs now
submit_jobs = input("Do you want to submit the jobs now? (yes/no): ").strip().lower()
if submit_jobs == 'yes':
    # Construct the submission command
    submit_command = f"hep_sub -e /dev/null -o /dev/null {wrapper_script_path} -argu \"%{{ProcId}}\" -n {num_job_scripts}"
    print("Submitting jobs...")
    os.system(submit_command)
    print(f"Jobs have been submitted with command: {submit_command}")
else:
    print("Jobs have not been submitted. You can submit them later with the following command:")
    print(f"hep_sub -e /dev/null -o /dev/null {wrapper_script_path} -argu \"%{{ProcId}}\" -n {num_job_scripts}")
