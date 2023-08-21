import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import numpy as np
from scipy.interpolate import PchipInterpolator


# Read the coordinates from the text file
file_path = 'coordinates.txt'  # Update this with your file path
data = np.loadtxt(file_path, delimiter=',', unpack=True)

x = data[0]
y = data[1]

# Sort coordinates by x-values
sorted_indices = np.argsort(x)
x_sorted = x[sorted_indices]
y_sorted = y[sorted_indices]

# Find the index where the split should occur
split_index = np.searchsorted(x_sorted, 5.8e15)

# Split the data into two segments
x_segment1 = x_sorted[:split_index]
y_segment1 = y_sorted[:split_index]
x_segment2 = x_sorted[split_index:]
y_segment2 = y_sorted[split_index:]

# Create PCHIP interpolation functions for each segment
pchip_segment1 = PchipInterpolator(x_segment1, y_segment1)
pchip_segment2 = PchipInterpolator(x_segment2, y_segment2)

# Generate points for interpolation
x_interp1 = np.logspace(np.log10(min(x_segment1)), np.log10(max(x_segment1)), 500)
x_interp2 = np.logspace(np.log10(min(x_segment2)), np.log10(max(x_segment2)), 500)

# Calculate interpolated y-values using the PCHIP functions for each segment
y_interp1 = pchip_segment1(x_interp1)
y_interp2 = pchip_segment2(x_interp2)

# Combine the two segments for the final interpolated curve
x_combined = np.concatenate((x_interp1, x_interp2))
y_combined = np.concatenate((y_interp1, y_interp2))

# Create a logarithmic plot for both x and y axes with the combined interpolated curve
font_type = 'Times New Roman'
fig = plt.figure(figsize=(12, 8))
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    #"font.family": "helvet",
    "text.latex.preamble": r"\usepackage{courier}",
})
plt.loglog(x_combined, y_combined, linestyle='-', color='black', label = "energy spectrum", linewidth=1.5)
plt.xlabel('Neutrino Energy (eV)', fontsize=20, fontfamily=font_type)
# Add LaTeX text in Y-axis title
plt.ylabel(r'Cross Section $\left(\bar{\nu}_{\mathrm{e}} \mathrm{e}^{-} \rightarrow \bar{\nu}_{\mathrm{e}} \mathrm{e}^{-}\right)$ in $mb$', fontsize=20, fontfamily=font_type)
plt.xticks(fontsize=21, fontfamily=font_type)  # Adjust fontsize as needed
plt.yticks(fontsize=21, fontfamily=font_type)  # Adjust fontsize as needed
#plt.legend(fontsize=16)

# Set x and y axis limits
plt.xlim(1e-5, 1e18)
plt.ylim(1e-32, 1e-1)

# Define the vertices of the triangle
triangle_vertices_bigbang = [
(0.26909918997970017, 1.9996012346107427e-31),
(0.0012909263358923276, 1.8775214426474256e-31),
(0.020911027988655027, 9.267895581307134e-28),
]

# Create a Polygon patch representing the triangle
triangle_patch_bigbang = patches.Polygon(triangle_vertices_bigbang, closed=True, edgecolor='none', facecolor='mediumorchid', alpha=0.3)
# Add the triangle patch to the plot
plt.gca().add_patch(triangle_patch_bigbang)
# Add the text above the triangle
plt.text(0.0032453689750932516, 5.924360933334056e-26, 'Big Bang', fontsize=21, color='purple')
# Add the text inside the triangle
plt.text(0.003506546665044656, 1.1269036678953028e-29, 'PTOLEMY', fontsize=8, color='purple', fontfamily='Arial')

triangle_vertices_solar = [
(22.31314054071469, 1.2080237472422547e-31) ,
(13111339.374215685, 1.2865716052854267e-31),
(17364.992226827242, 4.946973940874884e-19),
]
# Create a Polygon patch representing the triangle
triangle_patch_solar = patches.Polygon(triangle_vertices_solar, closed=True, edgecolor='none', facecolor='gold', alpha=0.3)
# Add the triangle patch to the plot
plt.gca().add_patch(triangle_patch_solar)
solar_exp = 'Super-K\nBorexino\nJUNO\nSNO+\nHyper-k'
# Add the text above the triangle
plt.text(1795.01294324921, 1.0175298766472706e-17, 'Solar', fontsize=21, color='orange', fontweight='bold')
# Add the text inside the triangle
plt.text(3002.4227336844297, 4.806396160355268e-28, solar_exp, fontsize=10, color='orange', fontfamily='serif', fontweight='bold')
triangle_vertices_terrestrial = [
(307687.7016008227, 1.1795004562329368e-14),
(81261.97236907802, 1.7628948745308805e-31),
(1130294.1747470526, 1.7628948745308805e-31)
]
# Create a Polygon patch representing the triangle
triangle_patch_terrestrial = patches.Polygon(triangle_vertices_terrestrial, closed=True, edgecolor='none', facecolor='saddlebrown', alpha=0.3)
plt.gca().add_patch(triangle_patch_terrestrial)
plt.text(285.9152387622236, 1.770570432781873e-13, 'Terrestrial', fontsize=21, color='saddlebrown', fontweight='bold')

triangle_vertices_reactor = [
(739965.5726634334, 1.5952552724625842e-10),
(162982.19978532154, 1.7628948745308805e-31),
(3259428.2609118414, 1.7628948745308805e-31),
]
# Create a Polygon patch representing the triangle
triangle_patch_reactor = patches.Polygon(triangle_vertices_reactor, closed=True, edgecolor='none', facecolor='springgreen', alpha=0.5)
plt.gca().add_patch(triangle_patch_reactor)

reactor_exp = "KamLAND\nDouble Chooz\nDaya Bay\nJUNO\nJUNO-TAO\nRENO\nRICOCHET\nDANSS\nPROSPECT\nSTEREO"
plt.text(7005.381477490712, 3.9823019643036455e-9, 'Reactor', fontsize=21, color='green', fontweight='bold')
# Add the text inside the triangle
plt.text(107621.1486828271, 5.0446401103743125e-28, reactor_exp, fontsize=8, color='green', fontfamily='serif', fontweight='bold')

triangle_vertices_supernova = [
(1948664.939913141, 2.0390785788306192e-8),
(100433.21070505373, 1.7628948745309094e-31),
(36682216.474527076, 1.7628948745308805e-31),
]

# Create a Polygon patch representing the triangle
triangle_patch_supernova = patches.Polygon(triangle_vertices_supernova, closed=True, edgecolor='none', facecolor='tomato', alpha=0.3)
plt.gca().add_patch(triangle_patch_supernova)
supernova_exp = "Super-K\nBorexino\nJUNO\nKamLAND\nLVD\nIceCube/PINGU\nHyper-K\nDUNE\nSNO+\nLAGUNA\nWATCHMAN"
plt.text(48582.86295263112, 5.066604227226546e-7, 'Supernova', fontsize=21, color='red', fontweight='bold')
# Add the text inside the triangle
plt.text(655610.3593284058, 1.6161831181860865e-16, supernova_exp, fontsize=8, color='red', fontfamily='serif', fontweight='bold')

triangle_vertices_accelerator = [
(334025578.37085164, 2.277965643975087e-13),
(9399210.245782478, 1.4542090517123968e-31),
(11870474655.681412, 1.454321373901243e-31),
]
# Create a Polygon patch representing the triangle
triangle_patch_accelerator = patches.Polygon(triangle_vertices_accelerator, closed=True, edgecolor='none', facecolor='firebrick', alpha=0.3)
plt.gca().add_patch(triangle_patch_accelerator)
accelerator_exp = "MINOS+\nT2K\nNOvA\nHyper-K\nLBNO\nRADAR\nCHiPS\nDUNE\nMINERvA\nMicroBooNE\nMiniBooNE+\nICARUS/NESSiE\nLAr1, SciNOvA\nDAE$\delta$ALUS\nCSI, CENNS\nCAPTAIN, OscSNS\n" + r"$\nu$STORM" +"\nNuMAX"
plt.text(24357252.527021172, 5.314613643682534e-11, 'Accelerator', fontsize=21, color='maroon', fontweight='bold')
# Add the text inside the triangle
plt.text(45780473.5450542, 1.0466737218416046e-28, accelerator_exp, fontsize=8, color='maroon', fontfamily='serif', fontweight='bold')

triangle_vertices_atmospheric = [
(27696700517.547363, 1.4435570839111406e-18),
(6738072.345084716, 1.655266495526706e-31),
(113846687935652.75, 1.655266495526706e-31),
]

# Create a Polygon patch representing the triangle
triangle_patch_atmospheric = patches.Polygon(triangle_vertices_atmospheric, closed=True, edgecolor='none', facecolor='blue', alpha=0.3)
plt.gca().add_patch(triangle_patch_atmospheric)
atmospheric_exp = "Super-K\nJUNO\nMINOS+\nIceCube\nPINGU\nDUNE\nINO/ICAL\nHyper-K\nLAGUNA"
plt.text(4242868525.999314, 2.0346539975971988e-19, 'Atmospheric', fontsize=21, color='blue', fontweight='bold')
# Add the text inside the triangle
plt.text(17870474655.681412, 1.514218958007772e-28, atmospheric_exp, fontsize=8, color='blue', fontfamily='serif', fontweight='bold')


triangle_vertices_cosmic = [
(117344253713924.78, 1.3091250468239609e-17),
(15121661056.662975, 1.8775214426474256e-31),
(910592680796181000, 1.8775214426474256e-31),
]

# Create a Polygon patch representing the triangle
triangle_patch_cosmic = patches.Polygon(triangle_vertices_cosmic, closed=True, edgecolor='none', facecolor='gray', alpha=0.3)
plt.gca().add_patch(triangle_patch_cosmic)
cosmic_exp = "IceCube/PINGU\nANTARES\nANITA\nARA/ARIANNA\nKM3NET\nEVA"
plt.text(92115012000432.47, 6.927332567206935e-17, 'Cosmic', fontsize=21, color='black', fontweight='bold')
# Add the text inside the triangle
plt.text(28991490228064.697, 3.1554158577173575e-28, cosmic_exp, fontsize=8, color='black', fontfamily='serif', fontweight='bold')

# Load and insert another image
#inserted_image = plt.imread('hs-article-bigBang-2400x1200.jpeg')  # Replace with your image path
# Adjust the position and size of the inserted image using Axes
#image_extent = [1e10, 1e12, 1e-6, 1e-4]  # [xmin, xmax, ymin, ymax]
#plt.imshow(inserted_image,  alpha=0.5)  # Adjust transparency


bigbang_image_path = 'bigbang.jpeg'
bigbang_image = mpimg.imread(bigbang_image_path)

# Add an Axes (subplot) to the figure at a specific position and with specific size
# The values in add_axes are [left, bottom, width, height] in normalized coordinates (0 to 1)
left = 0.18
bottom = 0.3
width = 0.12
height = 0.12
ax_bigbang = fig.add_axes([left, bottom, width, height])

# Display the image on the subplot
ax_bigbang.imshow(bigbang_image)
ax_bigbang.axis('off')  # Turn off axis
# Customize plot appearance
solar_image_path = 'solar.jpeg'
solar_image = mpimg.imread(solar_image_path)

# Add an Axes (subplot) to the figure at a specific position and with specific size
# The values in add_axes are [left, bottom, width, height] in normalized coordinates (0 to 1)
left = 0.26
bottom = 0.45
width = 0.15
height = 0.10
ax_solar = fig.add_axes([left, bottom, width, height])

# Display the image on the subplot
ax_solar.imshow(solar_image)
ax_solar.axis('off')  # Turn off axis

# Customize plot appearance
terrestrial_image_path = 'terrestrial.webp'
terrestrial_image = mpimg.imread(terrestrial_image_path)

# Add an Axes (subplot) to the figure at a specific position and with specific size
# The values in add_axes are [left, bottom, width, height] in normalized coordinates (0 to 1)
left = 0.22
bottom = 0.58
width = 0.15
height = 0.10
ax_terrestrial = fig.add_axes([left, bottom, width, height])

# Display the image on the subplot
ax_terrestrial.imshow(terrestrial_image)
ax_terrestrial.axis('off')  # Turn off axis

# Customize plot appearance
reactor_image_path = 'reactor.jpeg'
reactor_image = mpimg.imread(reactor_image_path)

# Add an Axes (subplot) to the figure at a specific position and with specific size
# The values in add_axes are [left, bottom, width, height] in normalized coordinates (0 to 1)
left = 0.28
bottom = 0.70
width = 0.15
height = 0.10
ax_reactor = fig.add_axes([left, bottom, width, height])

# Display the image on the subplot
ax_reactor.imshow(reactor_image)
ax_reactor.axis('off')  # Turn off axis
# Customize plot appearance
supernova_image_path = 'supernova.webp'
supernova_image = mpimg.imread(supernova_image_path)

# Add an Axes (subplot) to the figure at a specific position and with specific size
# The values in add_axes are [left, bottom, width, height] in normalized coordinates (0 to 1)
left = 0.48
bottom = 0.775
width = 0.15
height = 0.10
ax_supernova = fig.add_axes([left, bottom, width, height])

# Display the image on the subplot
ax_supernova.imshow(supernova_image)
ax_supernova.axis('off')  # Turn off axis
#plt.grid(True)
# Customize plot appearance
accelerator_image_path = 'accelerator.png' # from https://phd.uniroma1.it/web/pagina.aspx?i=3504&l=EN&p=95 
accelerator_image = mpimg.imread(accelerator_image_path)

# Add an Axes (subplot) to the figure at a specific position and with specific size
# The values in add_axes are [left, bottom, width, height] in normalized coordinates (0 to 1)
left = 0.57
bottom = 0.67
width = 0.15
height = 0.10
ax_accelerator = fig.add_axes([left, bottom, width, height])

# Display the image on the subplot
ax_accelerator.imshow(accelerator_image)
ax_accelerator.axis('off')  # Turn off axis
# Customize plot appearance
atmospheric_image_path = 'atmospheric.webp' # from https://www.astronomy.com/science/the-history-of-cosmic-rays-is-buried-beneath-our-feet/ 
atmospheric_image = mpimg.imread(atmospheric_image_path)

# Add an Axes (subplot) to the figure at a specific position and with specific size
# The values in add_axes are [left, bottom, width, height] in normalized coordinates (0 to 1)
left = 0.62
bottom = 0.47
width = 0.15
height = 0.10
ax_atmospheric = fig.add_axes([left, bottom, width, height])

# Display the image on the subplot
ax_atmospheric.imshow(atmospheric_image)
ax_atmospheric.axis('off')  # Turn off axis

# Customize plot appearance
cosmic_image_path = 'cosmic.jpeg' # from https://www.elisascience.org/articles/lisa-mission/lisa-technology/electromagnetic-universe-and-cosmic-landscape 
cosmic_image = mpimg.imread(cosmic_image_path)

# Add an Axes (subplot) to the figure at a specific position and with specific size
# The values in add_axes are [left, bottom, width, height] in normalized coordinates (0 to 1)
left = 0.76
bottom = 0.54
width = 0.15
height = 0.10
ax_cosmic = fig.add_axes([left, bottom, width, height])

# Display the image on the subplot
ax_cosmic.imshow(cosmic_image)
ax_cosmic.axis('off')  # Turn off axis
#plt.show()
# Save the figure as a PDF
output_path = 'ne_updated.pdf'
plt.savefig(output_path,dpi=600, format='pdf', bbox_inches='tight', pad_inches=0.1)

