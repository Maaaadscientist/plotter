import sys
import os
import shutil

def find_bottom_level_dirs(directory):
    """Recursively find all bottom-level directories within the given directory."""
    if not os.path.isdir(directory):
        return []  # Not a directory
    subdirs = [os.path.join(directory, sub) for sub in os.listdir(directory) if os.path.isdir(os.path.join(directory, sub))]
    if not subdirs:  # If there are no subdirectories, this is a bottom-level directory
        return [directory]
    else:  # Otherwise, recursively find bottom-level directories in each subdirectory
        bottom_dirs = []
        for subdir in subdirs:
            bottom_dirs.extend(find_bottom_level_dirs(subdir))
        return bottom_dirs

def check_files_in_dir(directory):
    """Check if there are at least two files in the directory."""
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    return len(files) >= 30

def check_output_consistency(job_script):
    """
    Check if the bottom-level output directories and files for a given job script exist and are not empty.
    Return False if any issues are found.
    """
    with open(job_script, 'r') as file:
        lines = file.readlines()

    for line in lines:
        if line.startswith("$python"):
            parts = line.split()
            output_path = parts[-1]  # Assuming the second last argument is the output path

            # Find all bottom-level directories within the output path
            bottom_dirs = find_bottom_level_dirs(output_path)
            if not bottom_dirs:  # No bottom-level directories found
                return False
            
            # Check each bottom-level directory for at least two files
            for dir in bottom_dirs:
                if not check_files_in_dir(dir):
                    return False

    return True

# Existing code to get the directories...
output_dir = os.path.abspath(sys.argv[1])
jobs_dir = os.path.join(output_dir, "jobs")
outputs_dir = os.path.join(output_dir, "outputs")
resubmit_dir = os.path.join(output_dir, "jobs_resubmit")

# Ensure the resubmit directory exists
if not os.path.isdir(resubmit_dir):
    os.makedirs(resubmit_dir)

# Check each job script and copy potentially failed jobs to the resubmit directory
job_scripts = [os.path.join(jobs_dir, js) for js in os.listdir(jobs_dir) if js.endswith('.sh')]
for job_script in job_scripts:
    if not check_output_consistency(job_script):
        shutil.copy(job_script, resubmit_dir)

print("Checked all job scripts. Potentially failed jobs have been copied to:", resubmit_dir)

# Generate the resubmit_jobs_wrapper.sh script
wrapper_script_path = os.path.join(output_dir, 'resubmit_jobs_wrapper.sh')
with open(wrapper_script_path, 'w') as wrapper_script:
    wrapper_script_content = f"""#!/bin/bash
# Directory containing the job scripts for resubmission
jobs_dir="{resubmit_dir}"

# Argument handling for the specific job script to execute
if [ $# -ne 1 ]; then
    echo "Usage: $0 [ProcId]"
    exit 1
fi

procid=$1  # The provided ProcId determines which job script to run

# Get an array of job scripts
scripts=( $(ls $jobs_dir/*.sh) )

# Calculate total available jobs
total_jobs=${{#scripts[@]}}

# Determine the job script to execute based on ProcId
if [ $procid -ge 0 ] && [ $procid -lt $total_jobs ]; then
    script_to_run="${{scripts[$procid]}}"
    echo "Running job script for resubmission: $script_to_run"
    bash "$script_to_run"
else
    echo "Error: ProcId $procid is out of range. Only $total_jobs resubmission jobs are available."
    exit 1
fi
"""
    wrapper_script.write(wrapper_script_content)

# Make the wrapper script executable
os.chmod(wrapper_script_path, 0o755)

print(f"Resubmit wrapper script has been generated at: {wrapper_script_path}")
# Calculate the number of job scripts generated
num_job_scripts = len(os.listdir(resubmit_dir))

# Ask user if they want to submit jobs now
submit_jobs = input(f"{num_job_scripts} jobs failed. Do you want to submit the jobs now? (yes/no): ").strip().lower()
if submit_jobs == 'yes':
    # Construct the submission command
    submit_command = f"hep_sub -e /dev/null -o /dev/null {wrapper_script_path} -argu \"%{{ProcId}}\" -n {num_job_scripts}"
    print("Submitting jobs...")
    os.system(submit_command)
    print(f"Jobs have been submitted with command: {submit_command}")
else:
    print("Jobs have not been submitted. You can submit them later with the following command:")
    print(f"hep_sub -e /dev/null -o /dev/null {wrapper_script_path} -argu \"%{{ProcId}}\" -n {num_job_scripts}")
