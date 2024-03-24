import sys
import argparse
import yaml
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import scienceplots

plt.style.use('science')
plt.style.use('nature')

def parse_arguments():
    parser = argparse.ArgumentParser(description='Photon Simulation Parameters')
    parser.add_argument('--input', type=str, default="final_sipm_params.csv", help='the input csv file')
    parser.add_argument('--config', type=str, default="config.yaml", help='the YAML configuration file')
    parser.add_argument('--type', type=str, default="pde", help='the parameter type for plotter')
    parser.add_argument('--ov', type=float, default=-1.0, help='Overvoltage')
    parser.add_argument('--output', type=str, default='outputs', help='Output ROOT file directory path')
    parser.add_argument('--distribution', action='store_true', help='draw distribution')
    args = parser.parse_args()
    return args.input, args.config, args.type, args.ov, args.output, args.distribution

def average_value_from_csv(df, column_name, selection_criteria=None, upper_threshold=1e8, lower_threshold=0):
    # Check if selection criteria are provided
    if selection_criteria:
        # Apply the selection criteria
        for key, value in selection_criteria.items():
            df = df[df[key] == value]

    # Check if the dataframe is not empty
    if not df.empty and column_name in df:
        # Filter out rows where the column value is either too high or zero
        filtered_df = df[(df[column_name] < upper_threshold) & (df[column_name] > lower_threshold)]

        # If filtered dataframe is not empty, calculate mean and variance
        if not filtered_df.empty:
            data = filtered_df[column_name]
            mu, std = norm.fit(data)
            #return filtered_df[column_name].mean(), filtered_df[column_name].var()
            return mu, std**2
        else:
            return "No data within specified range"
    else:
        return "No matching data or empty column"


# Parse command line arguments
input_path, config_path, param, ov, output_path, draw_distribution = parse_arguments()
with open(config_path, 'r') as file:
    yaml_data = yaml.safe_load(file)

if ov < 0 :
    voltages = [round(float(key),1) if key != 'default' else None for key in yaml_data[param]['voltages']]  # example values, adjust as needed
else:
    voltages = [ov]
# Load the CSV file
df = pd.read_csv(input_path)

df = df[(df['match_x'] < 3) & (df['match_x'] > -3)]
df = df[(df['match_y'] < 3) & (df['match_y'] > -3)]

meancolor="black"
maxcolor="red"
#maxcolor="crimson"
mincolor="deepskyblue"
#mincolor="dodgerblue"
labelsize=28
titlesize=40
textsize=24
# Group by 'tsn'
bad_channels = pd.read_csv("fit_failure.csv")
grouped = df.groupby('tsn')
count = len(df['tsn'].unique()) 

if draw_distribution:
    if param == 'vbd_diff':
        # Initialize arrays to store the results
        tsns = []
        vbd_max_values = []
        vbd_min_values = []
        vbd_diff = []
        vbd_diff_err = []
        bins = [float(key) for key in yaml_data[param]['voltages']['default']['bins'].split(',')]
        
        bad_count = 0
        print("tsn,batch,boxrun,pos,vbd_max_ch,vbd_min_ch,vbd_max,vbd_min,vbd_median,pde_max,pde_min,dcr_max,dcr_min,gain_max,gain_min")
        for name, group in grouped:
            vbd_max = group['vbd'].max()
            vbd_min = group['vbd'].min()
            # Filter out the bad channels for the current tsn.
            current_bad_channels = bad_channels[bad_channels['tsn'] == name]
            good_channels_group = group[~group['ch'].isin(current_bad_channels['ch'])]
            
            # Calculate vbd_max and vbd_min for good channels only.
            vbd_max = good_channels_group['vbd'].max()
            vbd_min = good_channels_group['vbd'].min()
            max_diff = vbd_max - vbd_min
            tsn = np.unique(group['tsn'].to_numpy())[0]
            batch = np.unique(group['batch'].to_numpy())[0]
            box = np.unique(group['box'].to_numpy())[0]
            run = np.unique(group['run'].to_numpy())[0]
            pos = np.unique(group['pos'].to_numpy())[0]
            vbd_max_ch = group.loc[group['vbd'] == vbd_max, 'ch'].iloc[0]
            vbd_min_ch = group.loc[group['vbd'] == vbd_min, 'ch'].iloc[0]
            pde_max = group.loc[group['vbd'] == vbd_max, 'pde_4.0'].iloc[0]
            pde_min = group.loc[group['vbd'] == vbd_min, 'pde_4.0'].iloc[0]
            dcr_max = group.loc[group['vbd'] == vbd_max, 'dcr_4.0'].iloc[0]
            dcr_min = group.loc[group['vbd'] == vbd_min, 'dcr_4.0'].iloc[0]
            gain_max = group.loc[group['vbd'] == vbd_max, 'gain_4.0'].iloc[0]
            gain_min = group.loc[group['vbd'] == vbd_min, 'gain_4.0'].iloc[0]
            if max_diff > 0.3:
                bad_count += 1
                data = group['vbd'].to_numpy()
                channels = group['ch'].to_numpy()
                # Calculate Q1 and Q3
                Q2 = np.percentile(data, 50)
                print(f"{tsn},{batch},{box},{run},{pos},{vbd_max_ch},{vbd_min_ch},{vbd_max},{vbd_min},{Q2},{pde_max},{pde_min},{dcr_max},{dcr_min},{gain_max},{gain_min}")
                Q1 = np.percentile(data, 25)
                Q3 = np.percentile(data, 75)
                
                # Calculate the IQR
                IQR = Q3 - Q1
                
                # Determine the outlier bounds
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Create a boolean mask where true indicates the presence of an outlier
                outlier_mask = (data < lower_bound) | (data > upper_bound)
                
                # Find the indices of the outliers
                outlier_indices = np.where(outlier_mask)[0]
                #print(group['vbd'].to_numpy(), group['ch'].to_numpy())
                #print("Indices of outliers:", outlier_indices, "length:",len(outlier_indices))
                #for index in outlier_indices:
                    #print(np.unique(group['tsn'].to_numpy())[0],channels[outlier_indices[index]],data[outlier_indices[index]])
                    #print(f"{tsn},{run},{pos},{index+1},{data[index]}")
        
            # Assuming 'vbd_err' is the column for vbd error
            vbd_max_err = group.loc[group['vbd'] == vbd_max, 'vbd_err'].iloc[0]
            vbd_min_err = group.loc[group['vbd'] == vbd_min, 'vbd_err'].iloc[0]
            error = np.sqrt(vbd_max_err**2 + vbd_min_err**2)
        
            # Store the results in the arrays
            tsns.append(name)
            vbd_max_values.append(vbd_max)
            vbd_min_values.append(vbd_min)
            vbd_diff.append(max_diff)
            vbd_diff_err.append(error)
        
        vbd_diff = np.array(vbd_diff)
        vbd_diff_err = np.array(vbd_diff_err)
        # Optionally, you can print the arrays or process them further
        hist_mean, _ = np.histogram(vbd_diff, bins=bins)
        hist_1sigma, _ = np.histogram(vbd_diff - vbd_diff_err, bins=bins)
        hist_3sigma, _ = np.histogram(vbd_diff - 3 * vbd_diff_err, bins=bins)
        
        
        plt.figure(figsize=(20, 15))
        
        
        bin_pos = np.arange(bins[0],bins[-1] + (bins[-1] - bins[0]) / len(bins), (bins[-1] - bins[0]) / len(bins))
        bin_centers = (np.array(bin_pos[:-1]) + np.array(bin_pos[1:])) / 2
        
        bin_width = np.min(np.diff(bin_pos)) / 3  # Assuming we're plotting 3 histograms side by side
        # Determine the offset for each histogram to center them around bin center
        offset = bin_width / 2
        # Plot the histograms side by side
        bar_3sigma = plt.bar(bin_centers[:-1] - offset, hist_3sigma, width=bin_width, alpha=0.7, label="$\\mathbin{-\\phantom{+}}$3 $\\sigma$", align='center', color="mediumorchid")
        bar_1sigma = plt.bar(bin_centers[:-1], hist_1sigma, width=bin_width, alpha=0.7, label="$\\mathbin{-\\phantom{+}}$1 $\\sigma$", align='center', color=mincolor)
        bar_mean = plt.bar(bin_centers[:-1] + offset, hist_mean, width=bin_width, alpha=0.7, label="$\\mathrm{V}_\\mathrm{bd}$ diff.", align='center', color=meancolor)
        
        # Add text on top of the bars
        def add_text_on_bars(bars, color):
            for bar in bars:
                height = bar.get_height()
                if height != 0:  # Only add text if height is not zero
                    plt.text(bar.get_x() + bar.get_width()/2., 1.01*height, '%d' % int(height), ha='center', va='bottom', fontsize=textsize, color=color)
        
        xlim = plt.xlim()  # Get current x-axis limits
        ylim = plt.ylim()  # Get current y-axis limits
        x_pos = xlim[1] - 0.05 * (xlim[1] - xlim[0])  # 5% from the right edge
        y_pos = ylim[1] - 0.4 * (ylim[1] - ylim[0])  # 5% from the top edge
        plt.text(x_pos, y_pos, f"Total: {count}", ha="right", va="top", fontsize=labelsize + 2, color="black")  # Adjust font size as needed
        add_text_on_bars(bar_mean, meancolor)
        add_text_on_bars(bar_1sigma, "dodgerblue")
        add_text_on_bars(bar_3sigma, "mediumorchid")
        # Set x-axis ticks to match the binning
        
        plt.xticks(ticks=bin_pos[:-1], labels=bins,fontsize=labelsize)
        plt.yticks(fontsize=labelsize)
        plt.xlabel("$\\mathrm{V}_\\mathrm{breakdown}$ difference (V)",fontsize=titlesize)
        plt.ylabel("Number of Tiles",fontsize=titlesize)
        plt.legend(fontsize=titlesize)
        #plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.title(f"Breakdown Voltage Maximum Difference", fontsize=titlesize+6)
        #plt.tight_layout()
        plt.subplots_adjust(left=0.12)
        
        plt.savefig(f"vbd_diff_distribution_3sigma.pdf")
        exit()
    
    # Initialize numpy arrays for storing PDE values and errors
    variable_max = {v: np.zeros(len(df['tsn'].unique()), dtype=float) for v in voltages}
    variable_max_err = {v: np.zeros(len(df['tsn'].unique()), dtype=float) for v in voltages}
    variable_min = {v: np.zeros(len(df['tsn'].unique()), dtype=float) for v in voltages}
    variable_min_err = {v: np.zeros(len(df['tsn'].unique()), dtype=float) for v in voltages}
    variable_mean = {v: np.zeros(len(df['tsn'].unique()), dtype=float) for v in voltages}
    variable_mean_err = {v: np.zeros(len(df['tsn'].unique()), dtype=float) for v in voltages}
    
    scale = float(yaml_data[param]['scale'])
    bad_count_dcr =0
    for i, (name, group) in enumerate(grouped):
        for v in voltages:
            if v is None:
                continue
            variable_col = f'{param}_{v}'
            variable_err_col = f'{param}_{v}_err'
    
            # Calculate max, min, and mean PDE values
            variable_max[v][i] = group[variable_col].max() * scale
            variable_min[v][i] = group[variable_col].min() * scale
            variable_mean[v][i] = group[variable_col].mean() * scale
            if v == 4.0 and group[variable_col].mean() > 100:
                bad_count_dcr +=1
                tsn = np.unique(group['tsn'].to_numpy())[0]
                batch = np.unique(group['batch'].to_numpy())[0]
                box = np.unique(group['box'].to_numpy())[0]
                run = np.unique(group['run'].to_numpy())[0]
                pos = np.unique(group['pos'].to_numpy())[0]
                dcr_max = variable_max[v][i]
                dcr_mean = variable_mean[v][i]
                print(f"{tsn},{batch},{box},{run},{pos},{dcr_max},{dcr_mean}")
    
            # Calculate errors
            variable_max_err_index = group[variable_col].idxmax()
            variable_min_err_index = group[variable_col].idxmin()
            variable_max_err[v][i] = group.loc[variable_max_err_index, variable_err_col] * scale
            variable_min_err[v][i] = group.loc[variable_min_err_index, variable_err_col] * scale
            variable_mean_err[v][i] = group[variable_err_col].std() / np.sqrt(len(group)) * scale
    print(bad_count_dcr)
    
    for v in voltages:
        if v is None:
            continue
        ov_str = round(v,1)
        print(v,ov_str)
        if v not in yaml_data[param]['voltages']:
            ov_str = 'default'
        # check uniform binning keys
        if not 'bins' in yaml_data[param]['voltages'][ov_str]:
            print("no bins")
            bins = np.arange(yaml_data[param]['voltages'][ov_str]['lower_edge'], yaml_data[param]['voltages'][ov_str]['upper_edge'] + yaml_data[param]['voltages'][ov_str]['step'], yaml_data[param]['voltages'][ov_str]['step'])
        else:
            bins = [float(key) for key in yaml_data[param]['voltages'][ov_str]['bins'].split(',')]
        bins = [round(key,yaml_data[param]['digit']) for key in bins]
        print(bins)
        hist_max, _ = np.histogram(variable_max[v], bins=bins)
        hist_min, _ = np.histogram(variable_min[v], bins=bins)
        hist_mean, _ = np.histogram(variable_mean[v], bins=bins)
        
        bin_pos = np.arange(bins[0],bins[-1] + (bins[-1] - bins[0]) / len(bins), (bins[-1] - bins[0]) / len(bins))
        bin_centers = (np.array(bin_pos[:-1]) + np.array(bin_pos[1:])) / 2
        
        bin_width = np.min(np.diff(bin_pos)) / 3  # Assuming we're plotting 3 histograms side by side
        plt.figure(figsize=(20, 15))
        
        # Determine the offset for each histogram to center them around bin center
        offset = bin_width / 2
        
        # Plot the histograms side by side
        bar_max = plt.bar(bin_centers[:-1] + offset, hist_max, width=bin_width, alpha=0.7, label="Max", align='center', color=maxcolor, hatch='\\')
        bar_min = plt.bar(bin_centers[:-1] - offset, hist_min, width=bin_width, alpha=0.7, label="Min", align='center', color=mincolor, hatch='/')
        bar_mean = plt.bar(bin_centers[:-1], hist_mean, width=bin_width, alpha=0.6, label="Mean", align='center', color=meancolor, hatch='-')
        # Change the color of the hatch
        for bar in bar_mean:
            bar.set_edgecolor('white')
        
        # Add text on top of the bars
        def add_text_on_bars(bars, color):
            for bar in bars:
                height = bar.get_height()
                if height != 0:  # Only add text if height is not zero
                    plt.text(bar.get_x() + bar.get_width()/2., 1.01*height, '%d' % int(height), ha='center', va='bottom', fontsize=textsize, color=color)
    
        add_text_on_bars(bar_max, maxcolor)
        add_text_on_bars(bar_min, mincolor)
        add_text_on_bars(bar_mean, meancolor)
        xlim = plt.xlim()  # Get current x-axis limits
        ylim = plt.ylim()  # Get current y-axis limits
        x_pos = xlim[1] - 0.05 * (xlim[1] - xlim[0])  # 5% from the right edge
        y_pos = ylim[1] - 0.4 * (ylim[1] - ylim[0])  # 5% from the top edge
        plt.text(x_pos, y_pos, f"Total: {count}", ha="right", va="top", fontsize=labelsize + 2, color="black")  # Adjust font size as needed
    
    
        plt.xticks(ticks=bin_pos[:-1], labels=bins,fontsize=labelsize)
        plt.yticks(fontsize=labelsize)
        
        if scale == 1:
            plt.xlabel(yaml_data[param]['xlabel'],fontsize=titlesize,labelpad=15)
        else:
            plt.xlabel(yaml_data[param]['xlabel'] + f" ({1/scale:.1e})",fontsize=titlesize,labelpad=15)
        plt.ylabel(yaml_data[param]['ylabel'],fontsize=titlesize,labelpad=15)
        plt.legend(fontsize=titlesize)
        #plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.title(f"{yaml_data[param]['title']} ({v}V)", fontsize=titlesize+6, pad=20)
        plt.tight_layout()
        plt.subplots_adjust(left=0.15)
        
        plt.savefig(f"{param}_distribution_{v}.pdf")
        plt.clf()
else:

    ov_list = []
    value_list = []
    error_list = []
    nuf_list = []
    scale = float(yaml_data[param]['scale'])
    if param != "ap":
    #if True:
        for i in range(46):
            vol = str(round(2.5 + i * 0.1, 1))
            ov_value = float(vol)
            # Filter out rows where 'ov' is outside the 'ov_min' and 'ov_max' range
            df = df[(df['ov_min'] <= ov_value) & (df['ov_max'] >= ov_value)]
            
            column_name = f'{param}_{vol}'
            column_err_name = f'{param}_{vol}_err'
            
            # With selection criteria
            #selection_criteria = {'tsn': 4838}
            
            # Without selection criteria
            average_all, dev_all = average_value_from_csv(df, column_name, upper_threshold=yaml_data[param]['tolerance'])
            error_all, _ = average_value_from_csv(df, column_err_name, upper_threshold=yaml_data[param]['tolerance'] * 0.1) # 10% tolerance for error
            print(vol,  average_all, error_all)
            ov_list.append(vol)
            value_list.append(average_all * scale)
            error_list.append(error_all * scale)
            nuf_list.append(np.sqrt(dev_all) * scale)
    else:
        ov_list = [2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7.0]
        value_list = [0.00, 0.00, 0.00, 0.0023121856076775654, 0.0018802001816537676, 0.004592260188738675, 0.005603494363346231, 0.0066575250176804214, 0.007667411782256014, 0.008433497021108608, 0.00843078391998287, 0.008385036021587812, 0.007904183469901863, 0.008850636694014601, 0.008713131143955609, 0.008717673859595676, 0.007030457136347763, 0.004947314444334691, 0.0057115686477610976, 0.007579198898779196, 0.009598977237236433, 0.010580060518829364, 0.012581271931600347, 0.013350082161514336, 0.014199856115907178, 0.015615447989311698, 0.016500705177208246, 0.0169721013382356, 0.017793987388636398, 0.018500698254727343, 0.01892180184767479, 0.019482180748078306, 0.0207172660202462, 0.02106353453046995, 0.021925275947001836, 0.023004779571738715, 0.023665011882735572, 0.024921758712276504, 0.026189829123050276, 0.027538547025168413, 0.028165703300553683, 0.028201262809919716, 0.030093832832628876, 0.031007762004127252, 0.03217048726827821, 0.03363866384548797]
        nuf_list = [0.013136907394202825, 0.013879000207692603, 0.01172923061512177, 0.010437580563530273, 0.01112900812160648, 0.010279539098754405, 0.009858534054630946, 0.009249827358309785, 0.008998075568069717, 0.008634943451434748, 0.008487837635555604, 0.008860517664227519, 0.009563184268466036, 0.009529081220770057, 0.010475664805748276, 0.010819824618329464, 0.012590297157327852, 0.013858412445011351, 0.014384117922135047, 0.013254136449609677, 0.011778499331107771, 0.011529894384352164, 0.01106791166174325, 0.01070831818201628, 0.010539175464438394, 0.010560790895056203, 0.010482178624015619, 0.010446283187696616, 0.010958399296241466, 0.011146635146437975, 0.011451122398750138, 0.011636967688693612, 0.011666766713485903, 0.011951260518882574, 0.012237966097727392, 0.013144084192128922, 0.012321059822437842, 0.01262118755700827, 0.01321930297521373, 0.014301886151021428, 0.015250314457820394, 0.015672733891749942, 0.01611124996476415, 0.016697520707322947, 0.01693867961207885, 0.017062234632111564]
        error_list = [0 for _ in range(len(ov_list))]
        ov_list = [str(round(key,1)) for key in ov_list]
        #value_list = [round(key,4) for key in value_list]
        #nuf_list = [round(key,4) for key in nuf_list]
    # Calculate the ratio of non-uniformity to value
    ratio_nuf_value = np.array(nuf_list) / np.array(value_list)
    # Create subplots
    #fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 15))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 15), gridspec_kw={'height_ratios': [5, 2]})
    #fig.subplots_adjust(bottom=0.2)  # Adjust as needed
    # Create a list of x-ticks that you want to display
    selected_ticks = [str(float(x)/10) for x in range(25, 71, 5)]
    # First plot: Value vs Over Voltage with non-uniformity as error
    #ax1.errorbar(ov_list, value_list, yerr=error_list, fmt='o', color='black', ecolor='black', linewidth=1.5, capsize=5, markersize=6, label=f'{param.upper()} Average')
    ax1.errorbar(ov_list, value_list, yerr=error_list, fmt='o', color='black', ecolor='black', linewidth=1.5, capsize=5, markersize=6, label=f'Average')
    if 'hline' in yaml_data[param]:
        if 'maximum' in  yaml_data[param]['hline']:
            ax1.axhline(y=yaml_data[param]['hline']['maximum'], color='r', linestyle='--', label='Maximum', linewidth=2)
        elif 'minimum' in  yaml_data[param]['hline']:
            ax1.axhline(y=yaml_data[param]['hline']['minimum'], color='r', linestyle='--', label='Minimum', linewidth=2)
        else:
            print("No limit found in hline list.")
        
        if 'typical' in  yaml_data[param]['hline']:
            ax1.axhline(y=yaml_data[param]['hline']['typical'], color='darkviolet', linestyle='--', label='Typical', linewidth=2)
    #ax1.set_xticks([float(x)/10 for x in range(25, 71, 5)], fontsize=labelsize)
    ax1.set_xticks(selected_ticks)
    # Adjust font size for x-axis and y-axis tick labels
    ax1.tick_params(axis='x', labelsize=labelsize)  # Adjust x-axis tick label font size
    ax1.tick_params(axis='y', labelsize=labelsize)  # Adjust y-axis tick label font size
    

    ax1.set_title(f'{yaml_data[param]["title"]} vs Over voltage', fontsize=titlesize+6, pad=20)
    #ax1.set_xlabel('Over Voltage (V)', fontsize=titlesize, labelpad=15)
    if scale == 1:
        ax1.set_ylabel(yaml_data[param]['xlabel'], fontsize=titlesize, labelpad=15)
    else:
        ax1.set_ylabel(yaml_data[param]['xlabel'] + f" ({1/scale:.1e})", fontsize=titlesize, labelpad=15)
    ax1.grid(True)
    ax1.set_ylim(yaml_data[param]['y_min'],yaml_data[param]['y_max'])
    
    # Second plot: Ratio of Non-uniformity to Value vs Over Voltage
    ax2.plot(ov_list, ratio_nuf_value, '-s', color='blue',markersize=5, label='$\\frac{\\sqrt{\\mathrm{Var.}}}{\\mathrm{Mean}}$')
    ax2.set_xticks(selected_ticks)
    # Do the same for the second subplot (ax2) if needed
    ax2.tick_params(axis='x', labelsize=labelsize)  # Adjust x-axis tick label font size
    ax2.tick_params(axis='y', labelsize=labelsize)  # Adjust y-axis tick label font size
    #ax2.set_xticks([float(x)/10 for x in range(25, 71, 5)])
    #ax2.set_title('Non-uniformity ratio', fontsize=titlesize+6, pad=20)
    ax2.set_xlabel('Over Voltage (V)', fontsize=titlesize, labelpad=15)
    #ax2.set_ylabel('Non-uniformity ratio', fontsize=titlesize, labelpad=15)
    ax2.grid(True)
    
    # Adjust layout
    plt.tight_layout(pad=3.0)
    ax1.legend(fontsize=labelsize,loc=yaml_data[param]['legend1_pos'])
    ax2.legend(fontsize=labelsize,loc=yaml_data[param]['legend2_pos'])
    
    # Create the plot
    #plt.figure(figsize=(20, 15))
    #plt.errorbar(ov_list, value_list, yerr=nuf_list, fmt='s', color='black', ecolor='blue', linewidth=2, capsize=5, markersize=8)
    #

    ## Create a list of x-ticks that you want to display
    #selected_ticks = [str(float(x)/10) for x in range(25, 71, 5)]
    #
    ## Apply the selected x-ticks to the plot
    #plt.xticks(selected_ticks, fontsize=labelsize)

    #plt.yticks(fontsize=labelsize)
    ## Customizing the plot
    #plt.title(f'{yaml_data[param]["title"]} vs Over voltage',fontsize=titlesize+6,pad=20)
    #plt.xlabel('Over Voltage (V)',fontsize=titlesize,labelpad=15)
    #plt.ylabel(yaml_data[param]['xlabel'],fontsize=titlesize,labelpad=15)
    #plt.grid(True)
    
    # Show the plot
    plt.savefig(f"{param}_vs_ov.pdf")
    plt.clf()
