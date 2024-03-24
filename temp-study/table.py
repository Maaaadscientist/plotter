import pandas as pd

# Load the dataset
df = pd.read_csv('slope.csv')
# Remove the damaged channel (channel 13 for tile position 5)
df = df[~((df['pos'] == 5) & (df['ch'] == 13))]
# Assuming 'tsn' is constant for each 'pos', we can simply take the first 'tsn' for each tile group
df['tsn'] = df.groupby('pos')['tsn'].transform('first')

# Calculate average slope and statistical error (standard deviation) for each tile, including the tsn
grouped_stats = df.groupby(['pos', 'tsn'])['slope'].agg(['mean', 'std']).reset_index()

# Rename columns for clarity
grouped_stats.columns = ['Tile', 'TSN', 'Average Slope', 'Statistical Error']

# Function to generate LaTeX code for the table, now including the TSN column
def generate_latex_table(data):
    latex_str = "\\begin{table}[H]\n\\centering\n\\begin{tabular}{cccc}\n\\hline\n"
    latex_str += "Tile SN & Position & Average Slope & Error \\\\\n\\hline\n"
    
    for _, row in data.iterrows():
        latex_str += f"{int(row['TSN'])} & {int(row['Tile'])} & {row['Average Slope']:.4f} & {row['Statistical Error']:.4f} \\\\\n"
    
    latex_str += "\\hline\n\\end{tabular}\n\\caption{Average slope, statistical error, and TSN number for each tile.}\n\\label{tab:tile_stats}\n\\end{table}"
    
    return latex_str

# Generate LaTeX table
latex_table = generate_latex_table(grouped_stats)

# Print the LaTeX code
print(latex_table)
