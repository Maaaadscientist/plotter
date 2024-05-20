import os
import sys
import pandas as pd
import numpy as np

input_str=sys.argv[1]
scr = sys.argv[2]
run_pos_csv = sys.argv[3]
#run_pos_csv = "/Users/wanghanwen/codes/plotter/spectrums-study/merged_csv_Dec10.csv"  # CSV containing "run" and "pos" for each "tsn"
#run_pos_csv = "/junofs/users/wanghanwen/special-harvest/merged.csv"
#run_pos_csv = "/Users/allen/codes/sipm-massive/merged.csv"
root_path = "/junofs/users/wanghanwen/main-runs/main"#os.getcwd()#""
root_path = "/Volumes/ExtDisk-1/main-runs/main"
root_path =  "~/codes/sipm-massive/main_run_0458"
script_path = os.getcwd()

# Read the run and pos CSV file
run_pos_data = pd.read_csv(run_pos_csv)
scripts = [
"amplitude_spectra.py",
"charge_spectra.py",
"fit_spectrum.py",
"plot_waveforms.py",
]
# Start generating the scripts
pos = int(input_str.split(",")[0])
run = int(input_str.split(",")[1])
ch = int(input_str.split(",")[2])
vol = int(input_str.split(",")[3])

# Filter the run_pos_data for the current tsn
selected_data = run_pos_data[(run_pos_data['pos'] == pos) & (run_pos_data['run'] == run) & (run_pos_data['ch'] == ch) & (run_pos_data['vol'] == vol)]

# Create a bash script for the current 'tsn' and 'ch'
script_name = f"tmp.sh"
with open(script_name, 'w') as file:
    file.write("#!/bin/bash\n")
    
    #file.write("source /cvmfs/juno.ihep.ac.cn/sw/anaconda/Anaconda3-2020.11-Linux-x86_64/bin/activate root622\n")
    #file.write('sleep 2\n')
    file.write('python=$(which python)\n')

    cmd = (f"$python {script_path}/{scr} {root_path}/main_run_{str(run).zfill(4)}_ov_{vol}.00_sipmgr_{str(ch).zfill(2)}_tile.root "
           f"{run_pos_csv} {pos} {pos}-{run}-{ch}-{vol}\n")
    file.write(cmd)

os.system(f"chmod a+x tmp.sh")
os.system('./tmp.sh')
print("Bash scripts have been generated and executed")
# Generate the initial_jobs_wrapper.sh script
