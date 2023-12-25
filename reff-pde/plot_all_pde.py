import os
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots

plt.style.use('science')
plt.style.use('nature')

labelsize=28
titlesize=40
textsize=24
size_marker = 5 #100
def plot_selected_pde_vs_ov(data_directory, selected_tsn):
    # Convert the list of tsn numbers to strings
    # Define a list of marker styles and line styles
    markers = ['o', 's', 'v', 'D','<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd', '|', '_']
    linestyles = ['-', '--', '-.', ':']

    # Create combinations of markers and line styles
    style_combinations = list(itertools.product(markers, linestyles))
    # Iterator for style combinations
    marker_iter = iter(markers)
    line_iter = iter(linestyles)
    selected_tsn = set(str(tsn) for tsn in selected_tsn)

    # List all CSV files in the given directory
    all_csv_files = [f for f in os.listdir(data_directory) if f.endswith('.csv') and f.split('_')[-1].split('.')[0] in selected_tsn]

    batch_1_tsn = [11675,11676,11677,11678,11671,11680]
    # Rearrange the files list to move batch_1_tsn files to the end
    batch_1_files = [f for f in all_csv_files if int(f.split('_')[-1].split('.')[0]) in batch_1_tsn]
    for f in all_csv_files:
        #print(f, int(f.split('_')[-1].split('.')[0]))
        if int(f.split('_')[-1].split('.')[0]) in batch_1_tsn:
            pass
            #print(f) 
    other_files = [f for f in all_csv_files if int(f.split('_')[-1].split('.')[0]) not in batch_1_tsn]
    print(batch_1_files, other_files)
    csv_files = other_files + batch_1_files
    #print(all_csv_files)
    #print(csv_files)

    plt.figure(figsize=(20, 15))

    count = 0
    for csv_file in all_csv_files:
        file_path = os.path.join(data_directory, csv_file)
        data = pd.read_csv(file_path)

        # Extract tsn number from filename
        tsn = csv_file.split('_')[-1].split('.')[0]
        if count % 7 == 0:
            marker = next(marker_iter)
            linestyle = next(line_iter)
        count += 1
        plt.errorbar(data['ov'], data['pde'], yerr=data['pde_err'], label=f'TSN {tsn}', capsize=3, markersize=size_marker, fmt=f'{marker}{linestyle}')

    # Customizing the plot
    plt.title('Ref. SiPM Photon Detection Efficiency at 404nm', fontsize=titlesize,pad=20)
    plt.xlabel('Over Voltage (V)', fontsize=labelsize,labelpad=15)
    plt.ylabel('PDE', fontsize=labelsize, labelpad=15)
    plt.legend(fontsize=textsize)
    plt.xticks(fontsize=labelsize)
    plt.yticks(fontsize=labelsize)
    plt.grid(True, axis='y')

    # Save the plot as an image file
    plt.savefig('selected_pde_vs_ov.pdf')

    # Optionally display the plot
    # plt.show()

    plt.clf()
    plt.figure(figsize=(20, 15))
    marker_iter = iter(markers)
    line_iter = iter(linestyles)
    count = 0
    for csv_file in csv_files:
        if count % 7 == 0:
            marker = next(marker_iter)
            linestyle = next(line_iter)
        count += 1
        file_path = os.path.join(data_directory, csv_file)
        data = pd.read_csv(file_path)

        # Extract tsn number from filename
        tsn = csv_file.split('_')[-1].split('.')[0]

        # Plotting pde vs ov for each selected file
        #plt.plot(data['vol'], data['pde'], label=f'TSN {tsn}')
        plt.errorbar(data['vol'], data['pde'], yerr=data['pde_err'], label=f'TSN {tsn}', capsize=3, markersize=size_marker, fmt=f'{marker}{linestyle}')

    # Customizing the plot
    plt.title('Ref. SiPM Photon Detection Efficiency at 404nm', fontsize=titlesize, pad=20)
    plt.xlabel('Voltage (V)', fontsize=labelsize,labelpad=15)
    plt.ylabel('PDE', fontsize=labelsize,labelpad=15)
    plt.legend(fontsize=textsize)
    plt.xticks(fontsize=labelsize)
    plt.yticks(fontsize=labelsize)
    plt.grid(True, axis='y')

    # Save the plot as an image file
    plt.savefig('selected_pde_vs_vol.pdf')


# List of selected tsn numbers
selected_tsn = [
    11671, 11678, 11676, 11677, 11675,
    11673, 11670, 11682, 11681, 11669,
    11674, 11683, 11680, 11668, 11667, 11679
]

# Replace 'path_to_your_csv_directory' with the path to your directory containing CSV files
plot_selected_pde_vs_ov('reff-output/csv', selected_tsn)

