import sys
import os
import shutil

output_dir = os.path.abspath(sys.argv[1])
jobs_dir = os.path.join(output_dir, "jobs")
outputs_dir = os.path.join(output_dir, "outputs")
resubmit_dir = os.path.join(output_dir, "jobs_resubmit")

# Ensure the resubmit directory exists
if not os.path.isdir(resubmit_dir):
    os.makedirs(resubmit_dir)

def check_output_consistency(job_script):
    """
    Check if the output directory and files for a given job script exist and are not empty.
    Return False if any issues are found.
    """
    with open(job_script, 'r') as file:
        lines = file.readlines()

    for line in lines:
        if line.startswith("python"):
            parts = line.split()
            output_path = parts[-1]  # Assuming the second last argument is the output path

            print(output_path)
            # Check if the output directory exists
            if not os.path.isdir(output_path):
                return False

            # Check if the directory is not empty (assuming files should be there)
            if not os.listdir(output_path):
                return False

    return True

# List all job scripts in the jobs directory
job_scripts = [os.path.join(jobs_dir, js) for js in os.listdir(jobs_dir) if js.endswith('.sh')]

# Check each job script
for job_script in job_scripts:
    if not check_output_consistency(job_script):
        # If the job is potentially failed, move/copy it to the resubmit directory
        shutil.copy(job_script, resubmit_dir)

print("Checked all job scripts. Potentially failed jobs have been copied to:", resubmit_dir)

