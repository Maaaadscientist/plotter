import csv
import subprocess

# Replace with your actual file paths
csv_file_path = 'bad_vbd_tiles.csv'
script_to_run = 'vbd_by_channel.py'
additional_arg = 'combined_csv_Dec26.csv'  # The constant argument for all calls

def call_script(tsn):
    """Call the existing script with the given tsn."""
    # Constructs the command like: python vbd_by_channel.py combined_csv_Dec26.csv 686
    cmd = ['python', script_to_run, additional_arg, str(tsn)]
    subprocess.run(cmd)

def main():
    with open(csv_file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            tsn = row['tsn']
            call_script(tsn)

if __name__ == '__main__':
    main()

