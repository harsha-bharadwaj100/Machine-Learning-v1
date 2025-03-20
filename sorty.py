import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

# Input data
input_sizes = np.array([25000, 50000, 75000, 100000, 125000])
selection_sort = np.array([22.12979, 77.19312, 173.96443, 275.54505, 452.49812])
bubble_sort = np.array([11.36401, 34.6602, 75.65809, 128.19612, 201.09087])
merge_sort = np.array([16.2525, 46.12864, 88.48986, 139.15518, 194.94462])
quick_sort = np.array([2.48696, 3.68874, 5.61189, 8.597, 11.84488])

# Create finer x-axis points for smooth interpolation
x_fine = np.linspace(min(input_sizes), max(input_sizes), 200)

# Create interpolation functions (cubic)
f_selection = interp1d(input_sizes, selection_sort, kind="cubic")
f_bubble = interp1d(input_sizes, bubble_sort, kind="cubic")
f_merge = interp1d(input_sizes, merge_sort, kind="cubic")
f_quick = interp1d(input_sizes, quick_sort, kind="cubic")

# Create figure with 4 subplots (2x2 grid)
fig, axs = plt.subplots(2, 2, figsize=(15, 12))  # Increased height from 10 to 12
# fig.suptitle("Sorting Algorithm Performance with Interpolation", fontsize=16)

# Selection Sort Plot
axs[0, 0].plot(input_sizes, selection_sort, "bo")
axs[0, 0].plot(x_fine, f_selection(x_fine), "b-")
axs[0, 0].set_title("Selection Sort")
axs[0, 0].set_xlabel("Input Size")
axs[0, 0].set_ylabel("Time (ms)")
axs[0, 0].grid(True)
axs[0, 0].legend()

# Bubble Sort Plot
axs[0, 1].plot(input_sizes, bubble_sort, "ro")
axs[0, 1].plot(x_fine, f_bubble(x_fine), "r-")
axs[0, 1].set_title("Bubble Sort")
axs[0, 1].set_xlabel("Input Size")
axs[0, 1].set_ylabel("Time (ms)")
axs[0, 1].grid(True)
axs[0, 1].legend()

# Merge Sort Plot
axs[1, 0].plot(input_sizes, merge_sort, "go")
axs[1, 0].plot(x_fine, f_merge(x_fine), "g-")
axs[1, 0].set_title("Merge Sort")
axs[1, 0].set_xlabel("Input Size")
axs[1, 0].set_ylabel("Time (ms)")
axs[1, 0].grid(True)
axs[1, 0].legend()

# Quick Sort Plot
axs[1, 1].plot(input_sizes, quick_sort, "mo")
axs[1, 1].plot(x_fine, f_quick(x_fine), "m-")
axs[1, 1].set_title("Quick Sort")
axs[1, 1].set_xlabel("Input Size")
axs[1, 1].set_ylabel("Time (ms)")
axs[1, 1].grid(True)
axs[1, 1].legend()

# Adjust layout with more space
plt.tight_layout(h_pad=4.0)  # Increased vertical padding between subplots
plt.subplots_adjust(top=0.94)  # Increased space for title (was 0.92)

# Combined plot
# plt.figure(figsize=(10, 7))  # Increased height from 6 to 7
# plt.plot(input_sizes, selection_sort, "bo", label="Selection Data")
# plt.plot(x_fine, f_selection(x_fine), "b-", label="Selection ")
# plt.plot(input_sizes, bubble_sort, "ro", label="Bubble Data")
# plt.plot(x_fine, f_bubble(x_fine), "r-", label="Bubble ")
# plt.plot(input_sizes, merge_sort, "go", label="Merge Data")
# plt.plot(x_fine, f_merge(x_fine), "g-", label="Merge ")
# plt.plot(input_sizes, quick_sort, "mo", label="Quick Data")
# plt.plot(x_fine, f_quick(x_fine), "m-", label="Quick ")
# plt.title("All Sorting Algorithms Comparison with Interpolation")
# plt.xlabel("Input Size")
# plt.ylabel("Time (ms)")
# plt.grid(True)
# plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout(pad=3.0)  # Added padding to combined plot

# Display the plots
plt.show()
