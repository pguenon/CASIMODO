import numpy as np	
import argparse
import MDAnalysis as mda 
from MDAnalysis.analysis import distances
import os
import scipy.stats as stats 
from scipy import signal
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from scipy.stats import t
from math import exp,log
from scipy.signal import find_peaks

import yacare

from scipy.ndimage import uniform_filter1d
from scipy.spatial.distance import pdist, squareform
from sklearn.mixture import GaussianMixture,BayesianGaussianMixture
from sklearn.neighbors import KernelDensity
from statsmodels.nonparametric.kde import KDEUnivariate
from sklearn.neighbors import KernelDensity
from scipy.interpolate import interp1d

import CASIMODO_utils.functions_SBM_casimodo as sbm  

print(dir(sbm))

###################### PRINT LOGO #####################
def print_header():
    with open("CASIMODO_utils/header_casimodo.txt", encoding="utf-8") as f:
        header = f.read()
    print(header)


###################### GENERAL FUNCTIONS #####################
def plot_progress_bar(current, total, previous_progress, bar_length=40):
    """
    Displays a textual progress bar in the console.

    Parameters:
    - current (int): The current progress count.
    - total (int): The total count to reach 100% completion.
    - previous_progress (float): The last recorded progress value.
    - bar_length (int): The length of the progress bar in characters (default is 40).

    Returns:
    - float: The new progress value if updated, otherwise returns previous_progress.

    Note:
    - The progress bar is only updated if:
        - It's the first call (previous_progress == -1),
        - Progress reaches 100%,
        - Or if progress has increased by at least 5% since the last update.
    """
    progress = current / total
    block = int(round(bar_length * progress))
    
    if previous_progress == -1 or progress == 1 or progress - previous_progress >= 0.05:
        text = f"\rProgress: [{'#' * block + '-' * (bar_length - block)}] {progress * 100:.0f}%"
        print(text, end='')
        return progress 
    else:
        return previous_progress   

def open_file(namefile):
    """
    Opens a text file and reads its contents.

    Parameters:
    - namefile (str): The name or path of the file to open.

    Returns:
    - data (list of lists): The file contents split by whitespace into lists of strings.
    - lines_file (list of str): The original lines as strings from the file.

    Note:
    - Each line is split using default whitespace and stored in the `data` list.
    """
    file_opened = open(namefile, 'r')
    lines_file = file_opened.readlines()
    data = []
    for row in lines_file:
        data.append([x for x in row.split()])
    return data, lines_file



def open_data_coordinate(namefile):
    """
    Loads numerical data from a file using NumPy.

    Parameters:
    - namefile (str): The name or path of the file to open.

    Returns:
    - data (numpy.ndarray): A NumPy array containing the numerical data from the file.

    Note:
    - The file is expected to contain whitespace-separated numerical values.
    """
    with open(namefile, 'r') as f:
        data = np.loadtxt(f)
    return data

def load_data_discretization(output_selected_coordinates):
    """
    Loads discretization data from a file and extracts:
    - coordinate names
    - cut points for discretization (minima)
    - corresponding labels for each region

    Parameters:
    - output_selected_coordinates (str): Path to the file containing discretization results.

    Returns:
    - coordinates (list of str): Names of the coordinates.
    - X_cuts (list of list of float): Cut points (e.g., minima) for each coordinate.
    - Labels (list of list of int): Labels corresponding to each region between cuts.
    """

    # Read file content (assumes open_file returns parsed data and raw lines)
    data_discretization, lines_discretization = open_file(output_selected_coordinates)

    coordinates = [row[0] for row in data_discretization]  # Extract coordinate names
    X_cuts = []  # To hold lists of cut points for each coordinate
    Labels = []  # To hold lists of region labels

    for row in data_discretization:
        xcut_i = []
        labels_i = []

        # Process alternating cut-point and label values (starting from column 1)
        for idx in range(1, len(row)):
            value = row[idx]
            if idx % 2 == 0:
                xcut_i.append(float(value))   # Even-indexed
            else:
                labels_i.append(int(value))   # Odd-indexed 

        X_cuts.append(xcut_i)
        Labels.append(labels_i)

    return coordinates, X_cuts, Labels


def get_multiplicities(Discretized_Array):
    # Get the shape of the input array: number of rows (frames) and columns (coordinates/features)
    nframes, ncoord = np.shape(Discretized_Array)
    
    # Initialize an array to hold the multiplicity (number of unique values) for each column
    multiplicities = np.zeros((ncoord), dtype=np.int32)
    
    # Loop over each column (coordinate/feature)
    for i in range(ncoord):
        # Count the number of unique values in column i and store it in the multiplicities array
        multiplicities[i] = len(np.unique(Discretized_Array[:, i]))
    
    # Return the array of multiplicities
    return multiplicities

##################### OPENING TRAJECTORY #####################
def open_trajectory(posfile, trajfile):
    """
    Opens a molecular dynamics trajectory using MDAnalysis.

    Parameters:
    - posfile (str): The topology file (e.g., .psf, .gro, or .pdb) that describes the molecular structure.
    - trajfile (str): The trajectory file (e.g., .dcd, .xtc) containing atomic coordinates over time.

    Returns:
    - u_traj (MDAnalysis.Universe): An MDAnalysis Universe object representing the system and trajectory,
      which can be used for further analysis (e.g., atom selections, RMSD calculations, etc.).
    """
    u_traj = mda.Universe(posfile, trajfile)
    return u_traj



########################## FILTERING TIMES AND INDICES ##################
def filter_times_and_indices(u_traj, time_zero, delta_time, output_dir):
    """
    Filters trajectory frames based on time criteria and saves the results.

    Parameters:
    - u_traj (MDAnalysis.Universe): The trajectory universe object.
    - time_zero (float): The minimum time threshold. Frames before this time are ignored.
    - delta_time (float): The time step interval for selecting frames (e.g., every 100 ps).
    - output_dir (str): Path to the directory where the output files will be saved.

    Returns:
    - times (np.ndarray): Array of selected frame times.
    - times_indices (np.ndarray): Array of corresponding frame indices.

    Behavior:
    - Iterates through the trajectory.
    - Uses a progress bar to show filtering progress in the terminal.
    - Selects frames where the time is greater than or equal to `time_zero`
      and where the time is a multiple of `delta_time`.
    - Saves the selected times and their frame indices as `.npy` files in the specified output directory.
    """
    print("\nFiltering times and indices...")
    times = []
    times_indices = []
    previous_progress = -1

    for ts in u_traj.trajectory:
        # Update progress bar
        previous_progress = plot_progress_bar(ts.frame, len(u_traj.trajectory), previous_progress)
        
        # Apply time filter
        if ts.time >= time_zero and ts.time % delta_time == 0:
            times.append(ts.time)
            times_indices.append(ts.frame)

    # Complete progress bar
    plot_progress_bar(len(u_traj.trajectory), len(u_traj.trajectory), previous_progress)

    # Convert to NumPy arrays and save
    times = np.array(times)
    times_indices = np.array(times_indices)
    np.save(output_dir + 'times.npy', times)
    np.save(output_dir + 'times_indices.npy', times_indices)

    print("\nTimes and indices filtered.")
    return times, times_indices

####################### GET TERMINAL ATOMS #######################
def read_dictionary(dic):
    """
    Reads a dictionary file containing terminal atom definitions.

    Parameters:
    - dic (str): Path to the dictionary text file.

    Returns:
    - terminal_atoms_dic (dict): A dictionary where each key is a residue name (e.g., amino acid),
      and each value is a list of terminal atom names for that residue.
    - amino_acids (list): A list of residue names marked as amino acids using the '@amino_acid' tag.

    Notes:
    - Lines ending with '@amino_acid' are treated specially to distinguish amino acids from other residues.
    - Lines with insufficient data (length <= 1) are skipped and reported.
    """
    terminal_atoms_dic = {}
    amino_acids = []
    with open(dic, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            if len(line) > 1:
                if line[-1] == "@amino_acid":
                    terminal_atoms_dic[line[0]] = line[1:-1]
                    amino_acids.append(line[0])
                else:
                    terminal_atoms_dic[line[0]] = line[1:]
            else:
                print(f"Skipping line: {line}")
    return terminal_atoms_dic, amino_acids

def get_terminal_atoms_MDA(u_traj, terminal_atoms_dic):
    """
    Extracts terminal atom definitions from an MDAnalysis Universe.

    Parameters:
    - u_traj (MDAnalysis.Universe): The trajectory object containing residues and atoms.
    - terminal_atoms_dic (str): Path to the terminal atom dictionary file.

    Returns:
    - terminal_atoms (list): List of terminal atom names (per residue).
    - RESIDS_SELECTED (list): List of residue IDs corresponding to the terminal atoms.
    - RESNAMES_SELECTED (list): List of residue names corresponding to the terminal atoms.
    - indices_aa (list): List of residue IDs that are amino acids (based on the dictionary).
    
    Notes:
    - Residues not found in the dictionary are reported once.
    """
    print("\nGetting terminal atoms...")
    atoms_dic, amino_acids = read_dictionary(terminal_atoms_dic)
    terminal_atoms = []
    RESIDS_SELECTED = []
    RESNAMES_SELECTED = []
    indices_aa = []
    RES_NOT_FOUND = []
    CHAINS_SELECTED = []

    for residue in u_traj.residues:
        resname = residue.resname
        resid = residue.resid
        if resname in atoms_dic:
            terminal_atoms.append(atoms_dic[resname])
            RESIDS_SELECTED.append(resid)
            RESNAMES_SELECTED.append(resname)
            if resname in amino_acids:
                indices_aa.append(resid)
        elif resname not in RES_NOT_FOUND:
            print(f"Residue {resname} not found in terminal_atoms_dic")
            RES_NOT_FOUND.append(resname)
    return terminal_atoms, RESIDS_SELECTED, RESNAMES_SELECTED, indices_aa

def save_terminal_atoms(terminal_atoms, RESIDS_SELECTED, RESNAMES_SELECTED, output_dir):
    """
    Saves terminal atom information to a text file.

    Parameters:
    - terminal_atoms (list): List of terminal atom names per residue.
    - RESIDS_SELECTED (list): List of corresponding residue IDs.
    - RESNAMES_SELECTED (list): List of corresponding residue names.
    - output_dir (str): Directory path where the output file will be saved.

    Output:
    - A text file named 'terminal_atoms.txt' containing:
      <resid>   <resname>   <atom_names>
    """
    print("\nSaving terminal atoms to file...")
    with open(output_dir + 'terminal_atoms.txt', 'w') as f:
        for k in range(len(terminal_atoms)):
            atom = terminal_atoms[k]
            resid = RESIDS_SELECTED[k]
            type_aa = RESNAMES_SELECTED[k]
            f.write(f'{resid}   {type_aa}   {atom}\n')
        f.close()
    print("Terminal atoms saved to file.")


############################## PRECOMPUTE POSITIONS OF ATOMS ##################
def precompute_terminals(u_traj, terminal_atoms, RESIDS_SELECTED, times, times_indices):
    """
    Precomputes the 3D positions of terminal atoms 
    for a selected set of residues across specified frames in a trajectory.

    Parameters:
    - u_traj (MDAnalysis.Universe): The MDAnalysis universe object containing the trajectory.
    - terminal_atoms (list of lists): A list of terminal atom names for each selected residue.
    - RESIDS_SELECTED (list): Residue IDs corresponding to the residues with terminal atoms.
    - times (np.ndarray): Time values for the selected frames (not directly used here but useful contextually).
    - times_indices (np.ndarray): Indices of the frames in the trajectory to process.

    Returns:
    - Positions_terminal_atoms (np.ndarray): A NumPy array of shape (num_atoms, num_frames, 3),
      storing the x, y, z coordinates of each terminal atom across selected frames.

    Behavior:
    - Preselects terminal atoms for all residues to avoid repetitive selection in each frame.
    - Iterates over selected trajectory frames and stores the positions of each terminal atom.
    - Displays a progress bar during processing.
    """
    print("\nPrecomputing terminal atoms positions...")

    num_residues = len(RESIDS_SELECTED)  # Total number of residues with terminal atoms
    num_atoms = np.sum([len(terminal_atoms[i]) for i in range(num_residues)])  # Total number of terminal atoms

    # Pre-select atom groups for each terminal atom in each residue to avoid repeated selections
    terminal_atom_selections = []
    for i in range(num_residues):
        terminal_atom_selections.append([
            u_traj.select_atoms(f"resid {RESIDS_SELECTED[i]} and name {terminal_atoms[i][j]}")
            for j in range(len(terminal_atoms[i]))
        ])

    # Initialize array to store terminal atom positions:
    # Shape: (total terminal atoms, number of selected frames, 3 coordinates)
    Positions_terminal_atoms = np.zeros((num_atoms, len(times_indices), 3))

    # Iterate through selected frames and record positions
    previous_progress = -1
    for k, frame in enumerate(times_indices):
        u_traj.trajectory[frame]  # Move to the specific frame
        previous_progress = plot_progress_bar(k, len(times_indices), previous_progress)
        count_step = 0  # Index for placing atoms in the output array
        for i in range(num_residues):
            for j in range(len(terminal_atoms[i])):
                Positions_terminal_atoms[count_step, k, :] = terminal_atom_selections[i][j].positions
                count_step += 1

    # Complete the progress bar
    plot_progress_bar(len(times_indices), len(times_indices), previous_progress)
    print("\nTerminal atoms positions precomputed.")

    return Positions_terminal_atoms

def precompute_backbone(u_traj, RESIDS_SELECTED, times, times_indices):
    """
    Precomputes the 3D positions of backbone atoms (C, N, and CA) for each selected residue
    across specified trajectory frames.

    Parameters:
    - u_traj (MDAnalysis.Universe): The MDAnalysis universe object containing the trajectory.
    - RESIDS_SELECTED (list): List of residue IDs for which backbone atoms are to be tracked.
    - times (np.ndarray): Array of time values for selected frames (not directly used here).
    - times_indices (np.ndarray): Indices of the trajectory frames to be processed.

    Returns:
    - Positions_atoms_C (np.ndarray): Array of shape (num_residues, num_frames, 3)
      with the 3D positions of the carbon (C) atoms.
    - Positions_atoms_N (np.ndarray): Array of shape (num_residues, num_frames, 3)
      with the 3D positions of the nitrogen (N) atoms.
    - Positions_atoms_CA (np.ndarray): Array of shape (num_residues, num_frames, 3)
      with the 3D positions of the alpha-carbon (CA) atoms.

    Behavior:
    - Selects backbone atoms (C, N, CA) once per residue to avoid repeated lookups.
    - Iterates through specified trajectory frames and stores atom positions for each frame.
    - Shows progress using a progress bar.
    """
    print("\nPrecomputing backbone positions...")

    num_residues = len(RESIDS_SELECTED)

    # Preselect atom groups for each backbone atom type
    atom_C_selections = [
        u_traj.select_atoms(f"resid {RESIDS_SELECTED[i]} and name C")
        for i in range(num_residues)
    ]

    atom_N_selections = [
        u_traj.select_atoms(f"resid {RESIDS_SELECTED[i]} and name N")
        for i in range(num_residues)
    ]

    atom_CA_selections = [
        u_traj.select_atoms(f"resid {RESIDS_SELECTED[i]} and name CA")
        for i in range(num_residues)
    ]

    # Initialize arrays to store positions of backbone atoms over time
    Positions_atoms_C = np.zeros((num_residues, len(times_indices), 3))
    Positions_atoms_N = np.zeros((num_residues, len(times_indices), 3))
    Positions_atoms_CA = np.zeros((num_residues, len(times_indices), 3))

    # Iterate through selected frames and record positions
    previous_progress = -1
    for k, frame in enumerate(times_indices):
        previous_progress = plot_progress_bar(k, len(times_indices), previous_progress)
        u_traj.trajectory[frame]  # Set trajectory to the specific frame

        for i in range(num_residues):
            Positions_atoms_C[i, k, :] = atom_C_selections[i].positions
            Positions_atoms_N[i, k, :] = atom_N_selections[i].positions
            Positions_atoms_CA[i, k, :] = atom_CA_selections[i].positions

    # Complete progress bar
    plot_progress_bar(len(times_indices), len(times_indices), previous_progress)
    print("\nBackbone positions precomputed.")

    return Positions_atoms_C, Positions_atoms_N, Positions_atoms_CA


###########################Save positions###################################
def save_positions(Positions, outname):
    """
    Saves precomputed positions
    """
    np.save(outname, Positions)




############################ Functions for computing average histograms and error bars ########################
def compute_histogram(data, y_min, y_max, delta_y):
    """
    Computes a normalized histogram (probability density) of the data.

    Parameters:
    - data (array-like): Input data to histogram.
    - y_min (float): Minimum bin edge.
    - y_max (float): Maximum bin edge.
    - delta_y (float): Bin width.

    Returns:
    - hist (array): Normalized histogram (PDF).
    - bin_edges (array): Edges of the bins.
    """
    bins = np.arange(y_min, y_max + delta_y, delta_y)
    hist, bin_edges = np.histogram(data, bins=bins, density=True)
    return hist, bin_edges


def compute_hist_tot(times, data, num_blocks, y_min, y_max, delta_y, time_zero_ps, size_block_ps):
    """
    Computes block-averaged histograms over time.

    Parameters:
    - times (array): Time points corresponding to the data.
    - data (array): Coordinate values over time.
    - num_blocks (int): Number of time blocks to divide data into.
    - y_min (float): Minimum bin edge.
    - y_max (float): Maximum bin edge.
    - delta_y (float): Bin width.
    - time_zero_ps (float): Starting time for analysis.
    - size_block_ps (float): Duration of each time block.

    Returns:
    - HIST_TOT (2D array): Histogram for each block.
    - x (array): Bin centers.
    - AVG (array): Average histogram across blocks.
    - STD (array): Standard deviation of histogram across blocks.
    """
    bins = np.arange(y_min, y_max + delta_y, delta_y)
    HIST_TOT = np.zeros((num_blocks, len(bins) - 1))

    for i in range(num_blocks):
        start_time = time_zero_ps + i * size_block_ps
        end_time = start_time + size_block_ps

        # Extract data for the current time block
        block_data = data[(times >= start_time) & (times < end_time)]
        
        # Compute histogram for this block
        hist, bin_edges = compute_histogram(block_data, y_min, y_max, delta_y)
        HIST_TOT[i] = hist

    # Compute bin centers
    x = (bin_edges[:-1] + bin_edges[1:]) / 2
    AVG = np.average(HIST_TOT, axis=0)
    STD = np.std(HIST_TOT, axis=0)

    return HIST_TOT, x, AVG, STD


def compute_error_bars(STD, num_blocks, confidence_level=0.95):
    """
    Computes error bars using the t-distribution for a given confidence level.

    Parameters:
    - STD (array): Standard deviation of histogram across blocks.
    - num_blocks (int): Number of blocks used in averaging.
    - confidence_level (float): Desired confidence level (default is 0.95).

    Returns:
    - error_bars (array): Error bars for each bin.
    """
    degrees_freedom = num_blocks - 1
    t_value = t.ppf((1 + confidence_level) / 2, degrees_freedom)
    return t_value * (STD / np.sqrt(num_blocks))


def get_avg_histogram(times, data, time_zero_ps, size_block_ps, coord_type):
    """
    Computes the average histogram and error bars for a coordinate type (e.g., distance, angle).

    Parameters:
    - times (array): Time points of the trajectory.
    - data (array): Coordinate values.
    - time_zero_ps (float): Starting time for analysis.
    - size_block_ps (float): Size of each time block.
    - coord_type (str): Type of coordinate ('distance' or 'angle').

    Returns:
    - data: Original data (unchanged).
    - filtered_data: Copy of original data (currently same as data).
    - x: Bin centers.
    - AVG: Average histogram across blocks.
    - error_bars: Error bars for each bin.
    - delta_y: Bin width used.
    - coord_type: Coordinate type (for labeling).
    - xlabel: Label for plotting x-axis.
    """
    # Set histogram parameters based on coordinate type
    if coord_type == 'distance':
        xlabel = 'Distance (Angstroms)'
        delta_y = 0.1
    elif coord_type == 'angle':
        xlabel = 'Angle (degrees)'
        delta_y = 2
    else:
        raise ValueError(f"Unsupported coordinate type: {coord_type}")

    # Compute number of blocks
    num_blocks = int((times[-1] - time_zero_ps) / size_block_ps)

    y_max = max(data)
    y_min = min(data)

    # Compute histograms
    HIST_TOT, x, AVG, STD = compute_hist_tot(times, data, num_blocks, y_min, y_max, delta_y,
                                             time_zero_ps, size_block_ps)

    # Compute confidence intervals
    error_bars = compute_error_bars(STD, num_blocks)

    return data, data, x, AVG, error_bars, delta_y, coord_type, xlabel



######################## Functions for discretizing coordinates ########################
def smooth_coordinate(y, delta_y):
    """
    Smooth a 1D distribution using Kernel Density Estimation (KDE)
    with a Gaussian kernel and fixed bandwidth.

    Parameters:
    - y: array-like, 1D input data (e.g., trajectory values).
    - delta_y: float, bandwidth for the KDE and spacing for the evaluation grid.

    Returns:
    - x_smooth: 1D array of x-values (evaluation grid for the KDE).
    - y_smooth: 1D array of corresponding smoothed probability density values.
    """

    # Ensure input is a NumPy array and reshape for sklearn's KDE
    y = np.asarray(y).reshape(-1, 1)

    # Step 1: Fit Gaussian KDE to the input data
    kde = KernelDensity(kernel='gaussian', bandwidth=delta_y)
    kde.fit(y)

    # Step 2: Create an evaluation grid over the range of y
    x_min, x_max = np.min(y), np.max(y)
    x_smooth = np.arange(x_min, x_max, delta_y / 10).reshape(-1, 1)

    # Step 3: Evaluate the log density on the grid
    log_density = kde.score_samples(x_smooth)
    y_smooth = np.exp(log_density)  # Convert from log-density to density

    # Step 4: Normalize the density so it integrates to 1
    y_smooth /= np.trapz(y_smooth, x_smooth.ravel())

    # Return 1D arrays for usability
    return x_smooth.ravel(), y_smooth

def find_minima(x_smooth, y_smooth, size_window):
    """
    Identify local minima in a smoothed distribution using derivative-based detection 
    and filtering based on a window around each candidate.

    Parameters:
    ----------
    x_smooth : np.ndarray
        The x-values corresponding to the smoothed data (must be evenly spaced).
    y_smooth : np.ndarray
        The y-values of the smoothed data (e.g., a KDE or smoothed histogram).
    size_window : float
        The half-width of the window (in x-units) used to validate extrema.

    Returns:
    -------
    filter_minima : list of float
        A list of validated local minima positions (x-values).
    """

    def closest_idx(array, value):
        """Return the index of the element in array closest to value."""
        return np.abs(array - value).argmin()

    # Compute first and second derivatives
    dx = x_smooth[1] - x_smooth[0]
    D_y = np.gradient(y_smooth, dx)
    D2_y = np.gradient(D_y, dx)

    # Find zero-crossings in the first derivative: potential extrema
    zero_crossings = np.where(np.diff(np.sign(D_y)))[0]
    minima, maxima = [], []

    for idx in zero_crossings:
        # Use second derivative to classify extremum
        if D2_y[idx] > 0:
            minima.append(x_smooth[idx])  # local minimum
        elif D2_y[idx] < 0:
            maxima.append(x_smooth[idx])  # local maximum

    # Ensure uniqueness and sorting
    minima = sorted(set(minima))
    maxima = sorted(set(maxima))

    filter_maxima = []  # (not returned here, but logic is preserved)

    for maxi in maxima:
        index_maxi = np.where(x_smooth == maxi)[0][0]
        # Define local window around the maximum
        min_w = maxi - size_window
        max_w = maxi + size_window
        i_min = closest_idx(x_smooth, min_w)
        i_max = closest_idx(x_smooth, max_w)

        i_peak = closest_idx(x_smooth, maxi)
        index_max_window = np.argmax(y_smooth[i_min:i_max+1]) + i_min 

        # Keep maximum only if it's truly the highest point in its window
        if abs(index_maxi-index_max_window)<5 :
            filter_maxima.append(maxi)
    minima_between_modes = []
    for i in range(len(filter_maxima) - 1):
        left = filter_maxima[i]
        right = filter_maxima[i + 1]
        min_in_between = [m for m in minima if left < m < right]
        if min_in_between:
            y_vals = [y_smooth[closest_idx(x_smooth, m)] for m in min_in_between]
            min_absolute = min_in_between[np.argmin(y_vals)]
            minima_between_modes.append(min_absolute)
    
    return minima_between_modes

def filter_significant_minima(x_smooth, y_smooth, minima, proba_cutoff):
    """
    Filters a list of local minima by removing those that do not separate regions
    with significant probability mass (area under the curve).

    For each pair of consecutive minima, the function calculates the integrated
    probability (area under the y_smooth curve). If the area is less than the 
    specified cutoff, it compares the depths of the two bounding minima and 
    removes the less significant one (i.e., the one closer in height to the 
    local maximum between them).

    Parameters:
    - x_smooth: 1D array of smoothed x-values (typically the coordinate range).
    - y_smooth: 1D array of smoothed probability density values corresponding to x_smooth.
    - minima: List of x-values corresponding to detected local minima.
    - proba_cutoff: Minimum probability threshold required to consider the region between minima as significant.

    Returns:
    - selected_minima: List of filtered minima that define significant regions.
    """
    selected_minima = []
    previous = x_smooth[0]  # Start from the leftmost boundary of the distribution

    for next_minimum in minima:
        # Define the region between the current and next minimum
        mask = (x_smooth >= previous) & (x_smooth <= next_minimum)
        area = np.trapz(y_smooth[mask], x_smooth[mask])  # Integrated probability

        if area < proba_cutoff:
            # Skip the first boundary (cannot remove the start)
            if previous == x_smooth[0]:
                continue

            # Compare the "depth" of the two minima relative to the maximum in-between
            y_prev = y_smooth[np.where(x_smooth == previous)[0][0]]
            y_next = y_smooth[np.where(x_smooth == next_minimum)[0][0]]
            max_in_range = max(y_smooth[mask])

            delta_prev = abs(y_prev - max_in_range)
            delta_next = abs(y_next - max_in_range)

            if delta_prev < delta_next:
                # The previous minimum is shallower: replace it with the current one
                if previous in selected_minima:
                    selected_minima.remove(previous)
                selected_minima.append(next_minimum)
                previous = next_minimum
            else:
                # Keep the previous, skip this one
                continue
        else:
            # Area is sufficient to keep the region — keep current minimum
            selected_minima.append(next_minimum)
            previous = next_minimum

    # Check the final region between last minimum and end of curve
    final_mask = (x_smooth >= previous) & (x_smooth <= x_smooth[-1])
    final_area = np.trapz(y_smooth[final_mask], x_smooth[final_mask])
    if final_area < proba_cutoff and previous in selected_minima:
        selected_minima.remove(previous)

    return selected_minima

def get_labels_discretization(minima, x_smooth, y_smooth):
    """
    Assigns labels to discretized regions based on their relative importance (e.g., density peak height).

    Each region is defined by two consecutive minima in the smoothed distribution. The function:
    1. Determines the index of each minimum in the x_smooth array.
    2. Calculates the maximum density value (peak) within each region.
    3. Sorts the regions by peak height (descending).
    4. Assigns labels to each region based on this order (label 0 = highest peak, etc.).

    Parameters:
    - minima: List of x values (positions of selected minima).
    - x_smooth: Array of smoothed x values (e.g., coordinate range).
    - y_smooth: Array of smoothed density values (same length as x_smooth).

    Returns:
    - labels: Array of labels, ranked by peak height within each discretized region.
    """

    # Find indices in x_smooth corresponding to the provided minima
    indexes_minima = [np.where(x_smooth == mini)[0][0] for mini in minima]

    # Define region boundaries: start at 0, go through all minima, end at last index
    all_minima = [0] + indexes_minima + [len(x_smooth) - 1]

    # Compute maximum density in each region (between minima)
    inter_max = []
    for i in range(len(all_minima) - 1):
        max_density = max(y_smooth[all_minima[i]:all_minima[i + 1]])
        inter_max.append(max_density)

    # Sort the regions by their max peak height (descending)
    sorted_indices = np.argsort(inter_max)[::-1]

    # Assign labels based on sorted order
    labels = np.zeros(len(sorted_indices), dtype=int)
    for i in range(len(sorted_indices)):
        labels[sorted_indices[i]] = i  # Highest peak gets label 0, and so on

    return labels

def save_minima(minima, coordinate, labels, name_output):
    """
    Saves the discretization minima and their associated labels to a file.

    The output format for each line is:
    <coordinate_type> <label_0> <minimum_0> <label_1> <minimum_1> ... <label_N>
    
    The last label is repeated at the end, which may represent the label of the final interval.

    Parameters:
    - minima: List of x-values (floats) where local minima were found (used as discretization boundaries).
    - coordinate: String indicating the coordinate type (e.g., "distance", "angle", etc.).
    - labels: List of integers representing the region label associated with each minimum.
    - name_output: Path to the output file where the minima and labels will be appended.
    """

    # Open the output file in append mode
    with open(name_output, 'a') as file_output:
        # Write the coordinate type first
        file_output.write(f'{coordinate} ')
        
        # Write each label-minimum pair
        for i in range(len(minima)):
            file_output.write(f' {labels[i]}')               # Write label
            file_output.write(f' {minima[i]:.3f}')         # Write minimum value with 3 decimal precision

        # Write the final label again (to cover the last interval)
        file_output.write(f' {labels[-1]}\n')  # Newline at the end of the line

def save_coordinate_results(times, distance_to_save, coordinate, output_dir):
    """
    Saves the evolution of a coordinate (e.g., distance) over time to a .dat file.

    Each line in the output file contains a time point and the corresponding coordinate value.

    Parameters:
    - times: 1D array of time points (floats).
    - distance_to_save: 1D array of coordinate values (floats), same length as times.
    - coordinate: String representing the coordinate name (used as filename).
    - output_dir: String path to the output directory (should end with a slash).
    """

    # Stack time and coordinate values into two columns
    Time_evolution = np.column_stack((times, distance_to_save))

    # Construct output file path
    output_file = output_dir + "coordinates_data/" + coordinate + ".dat"

    # Save to file with two decimal places, separated by three spaces
    np.savetxt(output_file, Time_evolution, fmt="%.2f   %.2f")

def plot_histogram(x, AVG, error_bars, x_smooth, y_smooth, delta_y, coord_type, xlabel, coordinate_name, minima, output_dir):
    """
    Plots the histogram of coordinate data with error bars, KDE curve, and vertical lines at selected minima.

    Parameters:
    - x: Array of bin centers for the histogram.
    - AVG: Average histogram values (probability density).
    - error_bars: Error estimates for each bin (e.g., standard error).
    - y_smooth: Smoothed density estimation from Kernel Density Estimation (KDE).
    - x_smooth: x-values corresponding to the KDE curve.
    - delta_y: Bin width or resolution (not directly used here but may be useful for labeling).
    - coord_type: Type of coordinate (unused here but kept for compatibility).
    - xlabel: Label string for the x-axis.
    - coordinate_name: Name of the coordinate (used for plot title and filename).
    - minima: List of x-values representing local minima to highlight on the plot.
    - output_dir: Directory path to save the plot image.

    The function saves the plot as a PNG file and closes the figure to free memory.
    """

    fig, ax = plt.subplots()

    # Plot average histogram as a black line
    ax.plot(x, AVG, color='black', label='Average')

    # Fill between error bars to show variability
    ax.fill_between(x, AVG - error_bars, AVG + error_bars, color='black', alpha=0.3)

    # Plot KDE smoothed curve in red
    ax.plot(x_smooth, y_smooth, color='red', lw=2, label='KDE')

    # Draw vertical dashed blue lines at each minimum position
    for mini in minima:
        ax.axvline(x=mini, color='blue', linestyle='--')

    # Set axis labels and plot title
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Probability density')
    ax.set_title(coordinate_name)

    # Show legend
    ax.legend()

    # Save plot to specified directory with dpi for quality
    plt.savefig(f'{output_dir}coordinates_plots/{coordinate_name}.png', dpi=150)

    # Close the figure to free memory
    plt.close()

def discretize_coordinate(y, delta_y, coordinate_type, times, time_zero, size_block,
                          coordinate_name, output, output_dir):
    """
    Discretizes a continuous coordinate distribution into distinct regions based on local minima 
    identified in the smoothed probability density.

    Workflow:
    1. Smooth the coordinate distribution using KDE.
    2. Compute an average histogram and related statistics from the raw data.
    3. Detect local minima in the smoothed KDE curve.
    4. Generate labels for discretized regions based on detected minima.
    5. Save minima, labels, and coordinate data for downstream use.
    6. Plot and save the histogram with KDE curve and highlighted minima.

    Parameters:
    - y (array-like): 1D array of coordinate values (trajectory over time).
    - delta_y (float): Bin width or resolution for histogram and smoothing.
    - coordinate_type (str): Label describing the coordinate type.
    - times (array-like): 1D array of time points corresponding to the trajectory.
    - time_zero (float): Starting time for analysis.
    - size_block (float): Block size for time-averaging histograms.
    - coordinate_name (str): Identifier used for saving files.
    - output (str or file-like): Path or handle to save minima/label information.
    - output_dir (str): Directory to save coordinate data and plots.

    Returns:
    - None. Results are saved to disk.
    """

    # Step 1: Smooth the coordinate distribution using KDE
    x_smooth, y_smooth = smooth_coordinate(y, delta_y)

    # Step 2: Compute the average histogram and error bars from raw data
    data, filtered_data, x, AVG, error_bars, delta_y, coord_type, xlabel = get_avg_histogram(
        times, y, time_zero, size_block, coordinate_type
    )

    # Step 3: Detect local minima in the smoothed density (robust to noise) and filter them
    minima = find_minima(x_smooth,y_smooth,size_window=10*delta_y)

    # Exit early if no minima are detected
    if len(minima) == 0:
        return
    
    selected_minima = filter_significant_minima(
        x_smooth, y_smooth, minima, proba_cutoff=0.01
    )
    if len(selected_minima) == 0:
        return

    # Step 4: Generate region labels from the minima
    labels = get_labels_discretization(selected_minima, x_smooth, y_smooth)

    # Step 5: Save detected minima and corresponding labels
    save_minima(selected_minima, coordinate_name, labels, output)

    # Step 6: Save the original coordinate data and metadata
    save_coordinate_results(times, y, coordinate_name, output_dir)

    # Step 7: Plot the histogram with KDE and show detected minima
    plot_histogram(
        x, AVG, error_bars,
        x_smooth, y_smooth, delta_y,
        coord_type, xlabel,
        coordinate_name, selected_minima,
        output_dir
    )


############################### Compute minimum distance between terminal atoms ##################
def compute_min_distances(Positions_terminal_atoms, i, j, terminal_atoms, RESIDS_SELECTED):
    """
    Computes the minimum distance between terminal atoms 
    of two residues across all frames in the trajectory.

    Parameters:
    - Positions_terminal_atoms (np.ndarray): A 3D array of shape 
      (total_terminal_atoms, num_frames, 3), storing positions of terminal atoms.
    - i (int): Index of the first residue in RESIDS_SELECTED and terminal_atoms.
    - j (int): Index of the second residue in RESIDS_SELECTED and terminal_atoms.
    - terminal_atoms (list of lists): A list where each entry contains the names 
      of terminal atoms for the corresponding residue.
    - RESIDS_SELECTED (list): List of residue IDs corresponding to the atoms.

    Returns:
    - min_absolute_distance (float): The minimum distance between any pair of terminal 
      atoms from residues i and j, over all frames.
    - distance_to_save (np.ndarray): A 1D array containing the distance values over time 
      for the pair of atoms that gave the minimum distance.
    - atom_i_to_save (str): Name of the atom in residue i involved in the minimum distance.
    - atom_j_to_save (str): Name of the atom in residue j involved in the minimum distance.

    Description:
    - Computes pairwise distances between all terminal atoms of residue i and residue j.
    - For each atom pair, the Euclidean distance is computed across all frames.
    - From all possible atom pair combinations, the one with the smallest minimum 
      distance (across all time steps) is selected and returned.
    """

    # Get number of terminal atoms for each residue
    num_term_i = len(terminal_atoms[i])
    num_term_j = len(terminal_atoms[j])

    # Calculate starting indices of terminal atoms for residues i and j
    ind_term_0_i = sum([len(terminal_atoms[k]) for k in range(i)])
    ind_term_0_j = sum([len(terminal_atoms[k]) for k in range(j)])

    # Extract positions for all terminal atoms of residue i and j
    Positions_i = [Positions_terminal_atoms[ind_term_0_i + k, :, :] for k in range(num_term_i)]
    Positions_j = [Positions_terminal_atoms[ind_term_0_j + k, :, :] for k in range(num_term_j)]

    # Copy the terminal atom names
    atoms_i = terminal_atoms[i].copy()
    atoms_j = terminal_atoms[j].copy()

    # Initialize distance matrix: shape (num_atoms_i, num_atoms_j, num_frames)
    distances = np.zeros((len(atoms_i), len(atoms_j), len(Positions_i[0])))

    # Compute pairwise distances over time
    for k in range(len(atoms_i)):
        for l in range(len(atoms_j)):
            distances[k, l] = np.linalg.norm(Positions_i[k] - Positions_j[l], axis=1)

    # Find the minimal distance for each atom pair across all frames
    minimal_distances = np.zeros((len(atoms_i), len(atoms_j)))
    for k in range(len(atoms_i)):
        for l in range(len(atoms_j)):
            minimal_distances[k, l] = np.min(distances[k, l])

    # Find the atom pair with the absolute minimal distance
    minimal_indexes = np.unravel_index(np.argmin(minimal_distances, axis=None), minimal_distances.shape)

    # Extract values to return
    min_absolute_distance = minimal_distances[minimal_indexes[0], minimal_indexes[1]]
    distance_to_save = distances[minimal_indexes[0], minimal_indexes[1]]
    atom_i_to_save = atoms_i[minimal_indexes[0]]
    atom_j_to_save = atoms_j[minimal_indexes[1]]

    return min_absolute_distance, distance_to_save, atom_i_to_save, atom_j_to_save


############################# process distance pair #############################
def process_distance_pair(i, j, Positions_terminal_atoms, terminal_atoms, RESIDS_SELECTED, times, time_zero, size_block, cutoff_distances, height_cutoff, output,output_dir):
    """
    Processes a pair of residues to compute distances and analyze multimodality.
    """
    min_absolute_distance,distance_to_save,atom_i_to_save,atom_j_to_save = compute_min_distances(
        Positions_terminal_atoms, i, j,terminal_atoms, RESIDS_SELECTED
    )

    if min_absolute_distance > cutoff_distances :
        return
    
    y= distance_to_save
    delta_y=0.1
    coordinate_type= 'distance'
    coordinate_name = f"{RESIDS_SELECTED[i]}_{atom_i_to_save}_{RESIDS_SELECTED[j]}_{atom_j_to_save}"
    
    discretize_coordinate(
        y, delta_y,coordinate_type,times, time_zero, size_block,coordinate_name,output,output_dir
    )


####################### Function to compute distances between terminal atoms for all residue pairs ##########################
def compute_all_distances(u_traj,terminal_atoms,RESIDS_SELECTED,Positions_terminal_atoms,times,time_zero,size_block,delta_resid,cutoff_distances,height_cutoff,output,output_dir):
    """
    Computes pairwise distances between all valid residue pairs based on their terminal atoms,
    and processes each pair using a custom distance analysis function.

    Parameters:
    - u_traj (MDAnalysis.Universe): The MDAnalysis universe object containing the trajectory.
      (Not directly used here but may be needed in `process_distance_pair`.)
    - terminal_atoms (list of lists): Terminal atom names for each residue.
    - RESIDS_SELECTED (list): List of selected residue IDs.
    - Positions_terminal_atoms (np.ndarray): 3D array of precomputed positions of terminal atoms.
      Shape: (total_terminal_atoms, num_frames, 3).
    - times (np.ndarray): Array of times corresponding to selected frames.
    - time_zero (float): Reference time used for distance analysis.
    - size_block (int): Size of blocks used in post-processing (likely temporal).
    - delta_resid (int): Minimum residue index separation; avoids comparing too-close residues.
    - cutoff_distances (float): Distance threshold for further analysis.
    - height_cutoff (float): Height threshold used in filtering results.
    - output (file-like or handle): Destination for saving results.
    - output_dir (str): Directory where output files will be written.

    Returns:
    - None (results are saved to files via `process_distance_pair`)

    Behavior:
    - Iterates over all valid residue index pairs `(i, j)` where `j >= i + delta_resid`.
    - For each pair, calls `process_distance_pair` to compute and process distances.
    - A progress bar is shown during processing.
    """

    num_residues = len(RESIDS_SELECTED)
    total_combinations = num_residues * (num_residues - delta_resid) / 2  # total number of pairs
    count_step = 0

    print("\nComputing distances...")

    # Iterate over all valid residue pairs
    previous_progress = -1  # Initialize progress bar
    for i in range(num_residues - delta_resid):
        for j in range(i + delta_resid, num_residues):
            # Update progress bar
            previous_progress=plot_progress_bar(count_step, total_combinations, previous_progress)
            count_step += 1

            # Process this residue pair
            process_distance_pair(
                i, j,Positions_terminal_atoms,terminal_atoms,RESIDS_SELECTED,times,time_zero,size_block,cutoff_distances,height_cutoff,output,output_dir
            )

    # Finalize progress bar
    plot_progress_bar(total_combinations, total_combinations,previous_progress)
    print("\nDistances computed and saved.")


########################## Function to get the multimodal contacts ################################
def get_contacts(u_traj, terminal_atoms, RESIDS_SELECTED, time_zero, size_block, delta_time, cutoff_distances, delta_resid, height_cutoff, indices_aa, output_dir):
    """
    Main function to compute and process distances (contacts) between terminal atoms
    throughout a molecular dynamics trajectory.

    This function performs the following steps:
    1. Loads pre-filtered time points and frame indices from .npy files.
    2. Precomputes the 3D positions of all terminal atoms across the selected frames.
    3. Saves the computed positions to a file for later use.
    4. Computes distances between all valid residue pairs and processes the results.

    Parameters:
    - u_traj: MDAnalysis Universe object containing the trajectory.
    - terminal_atoms: List of terminal atom names for each selected residue.
    - RESIDS_SELECTED: List of residue IDs to analyze.
    - time_zero: Time reference for analysis start.
    - size_block: Block size used for distance analysis (e.g., time window).
    - delta_time: Time interval used to sample frames (used when loading times).
    - cutoff_distances: Maximum distance threshold to consider a contact.
    - delta_resid: Minimum residue index separation to avoid local contacts.
    - height_cutoff: Cutoff for additional filtering of distance results.
    - indices_aa: List of indices corresponding to amino acid residues (not directly used here).
    - output_dir: Directory path to read inputs and save outputs.

    Returns:
    - None. Results are saved to files.
    """

    # Load time points and frame indices previously filtered and saved
    times = np.load(output_dir + 'times.npy')
    times_indices = np.load(output_dir + 'times_indices.npy')

    # Precompute terminal atom positions across trajectory
    Positions_terminal_atoms = precompute_terminals(u_traj, terminal_atoms, RESIDS_SELECTED, times, times_indices)

    # Save precomputed positions to disk
    save_positions(Positions_terminal_atoms, output_dir + "Positions_npy/Positions_terminal_atoms.npy")

    # Compute and process distances between all valid residue pairs
    compute_all_distances(
        u_traj, terminal_atoms, RESIDS_SELECTED, Positions_terminal_atoms,
        times, time_zero, size_block, delta_resid,
        cutoff_distances, height_cutoff,
        output_dir + "selected_coordinates.txt", output_dir
    )


########################### Functions to process dihedrals for a single residue ##########################
def adjust_angle_data(data, y_min, y_max, delta_y):
    """
    Adjust angle data by "unwrapping" values below the global minimum histogram bin center.
    This is useful for circular data like angles (0-360 degrees), to reduce edge effects
    by shifting low values above the main minimum.

    Parameters:
    - data: 1D array of angle measurements (in degrees).
    - y_min: Minimum angle value for histogram binning.
    - y_max: Maximum angle value for histogram binning.
    - delta_y: Bin width for histogram.

    Returns:
    - adjusted_data: Angle data after adjustment (values below min histogram bin center shifted by +360).
    - new_y_max: Updated max value after adjustment.
    - new_y_min: Updated min value after adjustment.
    """

    # Compute histogram of the input angle data
    hist_all, bin_edges_all = compute_histogram(data, y_min, y_max, delta_y)

    # Compute bin centers from edges
    x_all = (bin_edges_all[:-1] + bin_edges_all[1:]) / 2

    # Find all indices where histogram attains its global minimum
    min_indices = np.where(hist_all == np.min(hist_all))[0]

    # Select the median min bin center if multiple minima, else single minimum bin center
    if len(min_indices) > 1:
        median_index = min_indices[len(min_indices) // 2]
    else:
        median_index = min_indices[0]

    x_min_all = x_all[median_index]

    # Shift all angle values less than the selected minimum bin center by +360 degrees
    adjusted_data = np.where(data < x_min_all, data + 360, data)

    # Update min and max values after adjustment
    new_y_max = adjusted_data.max()
    new_y_min = adjusted_data.min()

    return adjusted_data, new_y_max, new_y_min


def process_dihedral_i(i, Positions_atoms_C, Positions_atoms_N, Positions_atoms_CA, RESIDS_SELECTED, 
                       times, time_zero, size_block, height_cutoff, output, output_dir):
    """
    Processes the i-th residue to compute phi and psi dihedral angles, adjust for angle wrapping,
    and discretize the angle distributions for further analysis.

    Parameters:
    - i: Index of the residue to process.
    - Positions_atoms_C, Positions_atoms_N, Positions_atoms_CA: 3D arrays of atomic positions
      with shape (num_residues, num_timepoints, 3).
    - RESIDS_SELECTED: List or array of residue identifiers.
    - times: 1D array of time points corresponding to frames.
    - time_zero: Start time for analysis.
    - size_block: Block size for time-averaging during discretization.
    - height_cutoff: Threshold used in multimodality detection (not used here, but passed for consistency).
    - output: File handle or path for saving results.
    - output_dir: Directory path for saving outputs.

    Notes:
    - Phi angle is defined only if the previous residue exists and the C-N distance is reasonable.
    - Psi angle is defined only if the next residue exists and the N-C distance is reasonable.
    - Angles are converted from radians to degrees.
    - If the angle range exceeds 180°, the data is "unwrapped" to reduce circular boundary artifacts.
    - Discretization is performed on adjusted angles.
    """

    delta_y = 2  # Bin width for histogram/discretization (degrees)
    coordinate_type = 'angle'

    # Initialize empty arrays (optional, overwritten later)
    phi_angle = np.zeros(len(times))
    psi_angle = np.zeros(len(times))

    # Process phi dihedral if previous residue exists and backbone geometry is valid
    if i > 0:
        distance_C_N = np.linalg.norm(Positions_atoms_C[i - 1, 0, :] - Positions_atoms_N[i, 0, :])
        if distance_C_N < 1.6:
            coordinate_name = f"phi{RESIDS_SELECTED[i]}"
            # Calculate phi dihedral angles (radians) and convert to degrees
            phi_angle = np.rad2deg(mda.lib.distances.calc_dihedrals(Positions_atoms_C[i - 1, :, :],Positions_atoms_N[i, :, :],Positions_atoms_CA[i, :, :],Positions_atoms_C[i, :, :])            )
            # Adjust angles if range spans more than 180 degrees (unwrap circular data)
            if np.ptp(phi_angle) > 180:
                phi_angle, _, _ = adjust_angle_data(phi_angle, np.min(phi_angle), np.max(phi_angle), delta_y)
            
            # Discretize the phi angle data for further analysis
            discretize_coordinate(phi_angle, delta_y, coordinate_type=coordinate_type,
                                  times=times, time_zero=time_zero, size_block=size_block,
                                  coordinate_name=coordinate_name, output=output, output_dir=output_dir)

    # Process psi dihedral if next residue exists and backbone geometry is valid
    if i < len(Positions_atoms_C) - 1:
        distance_N_C = np.linalg.norm(Positions_atoms_N[i + 1, 0, :] - Positions_atoms_C[i, 0, :])
        if distance_N_C < 1.6:
            coordinate_name = f"psi{RESIDS_SELECTED[i]}"
            # Calculate psi dihedral angles (radians) and convert to degrees
            psi_angle = np.rad2deg(mda.lib.distances.calc_dihedrals(Positions_atoms_N[i, :, :],Positions_atoms_CA[i, :, :],Positions_atoms_C[i, :, :],Positions_atoms_N[i + 1, :, :]))
            # Adjust angles if range spans more than 180 degrees (unwrap circular data)
            if np.ptp(psi_angle) > 180:
                psi_angle, _, _ = adjust_angle_data(psi_angle, np.min(psi_angle), np.max(psi_angle), delta_y)
            
            # Discretize the psi angle data for further analysis
            discretize_coordinate(psi_angle, delta_y, coordinate_type=coordinate_type,
                                  times=times, time_zero=time_zero, size_block=size_block,
                                  coordinate_name=coordinate_name, output=output, output_dir=output_dir)
        
    
########################### Function to compute dihedrals for all residues ##########################
def compute_all_dihedrals(u_traj, RESIDS_SELECTED, Positions_atoms_C, Positions_atoms_N, Positions_atoms_CA, times, time_zero, size_block, height_cutoff, output, output_dir):  
    """
    Iterates over all selected residues and computes dihedral angles between them.

    This function processes each selected residue's dihedral angle one by one using
    precomputed backbone atom positions and stores the results for further analysis.

    Parameters:
    - u_traj: MDAnalysis Universe or trajectory object.
    - RESIDS_SELECTED: List of residue indices for which to compute dihedrals.
    - Positions_atoms_C: Precomputed C atom positions for each residue over time.
    - Positions_atoms_N: Precomputed N atom positions for each residue over time.
    - Positions_atoms_CA: Precomputed CA atom positions for each residue over time.
    - times: 1D array of time points (e.g., in ps).
    - time_zero: Time (in ps) to start analysis from.
    - size_block: Block size (in ps) for time-averaging.
    - height_cutoff: Threshold for peak filtering or discretization (used later).
    - output: Path to output file where selected features/labels are written.
    - output_dir: Directory where output data (e.g., plots or processed values) is stored.

    Returns:
    - None. Outputs are saved directly to disk.
    """

    num_residues = len(RESIDS_SELECTED)

    print("\nComputing dihedrals...")
    previous_progress = -1  # Initialize progress bar
    for i in range(num_residues):
        previous_progress=plot_progress_bar(i, num_residues,previous_progress)
        process_dihedral_i(i, Positions_atoms_C, Positions_atoms_N, Positions_atoms_CA, RESIDS_SELECTED, times, time_zero, size_block, height_cutoff, output, output_dir)

    plot_progress_bar(num_residues, num_residues,previous_progress)
    print("\nDihedrals computed and saved.")


########################## Function to get the multimodal dihedrals ################################
def get_dihedrals(u_traj, indices_aa, time_zero, size_block, delta_time, 
                  cutoff_distances, delta_resid, height_cutoff, output_dir):
    """
    Computes and processes dihedral angles for selected amino acid residues in a trajectory.

    Workflow:
    1. Loads simulation time points and their indices.
    2. Checks if at least two amino acids are selected.
    3. Precomputes and saves backbone atom positions (N, C, CA).
    4. Computes all dihedral angles and processes them for further analysis.

    Parameters:
    - u_traj: MDAnalysis Universe or trajectory object.
    - indices_aa: List of amino acid residue indices to analyze.
    - time_zero: Starting time for dihedral analysis.
    - size_block: Size of blocks used for averaging time intervals.
    - delta_time: Temporal resolution (not used directly in this function).
    - cutoff_distances: Distance cutoffs for residue pairing (not used here).
    - delta_resid: Minimum sequence separation between residue pairs (not used here).
    - height_cutoff: Threshold for dihedral peak detection or filtering.
    - output_dir: Directory to save output files.

    Returns:
    - None. Saves dihedral angles and intermediate data to disk.
    """

    # Load time values and their corresponding frame indices
    times = np.load(output_dir + 'times.npy')
    times_indices = np.load(output_dir + 'times_indices.npy')

    # Early exit if fewer than two residues are selected
    if len(indices_aa) < 2:
        print("No amino acids selected for dihedral analysis.")
        return

    # Step 1: Precompute backbone atom positions (N, C, and CA atoms)
    Positions_atoms_C, Positions_atoms_N, Positions_atoms_CA = precompute_backbone(
        u_traj, indices_aa, times, times_indices
    )

    # Step 2: Save backbone atom positions to disk for future use
    save_positions(Positions_atoms_C, output_dir + "Positions_npy/Positions_C_atoms.npy")
    save_positions(Positions_atoms_N, output_dir + "Positions_npy/Positions_N_atoms.npy")
    save_positions(Positions_atoms_CA, output_dir + "Positions_npy/Positions_CA_atoms.npy")

    # Step 3: Compute all dihedral angles and write selected features
    compute_all_dihedrals(u_traj, indices_aa, Positions_atoms_C, Positions_atoms_N, Positions_atoms_CA, times, time_zero, size_block, height_cutoff, output_dir + "selected_coordinates.txt", output_dir)


############################# Function to add new coordinates to the existing discretization ##########################
def add_coordinates(coordinates_to_add, type_coordinates_to_add, output_dir, time_zero, size_block, height_cutoff):
    """
    Adds new coordinates (distance or angle) to an existing discretization setup.

    Arguments:
    coordinates_to_add -- list of file paths to the new coordinate data (.dat files)
    type_coordinates_to_add -- list indicating the type of each coordinate ('distance' or 'angle')
    output_dir -- path to the directory where outputs are stored
    time_zero -- starting time point for block analysis
    size_block -- block size for histogram averaging
    height_cutoff -- threshold for multimodal analysis (not used in this function)

    Notes:
    - Coordinate data must be 2-column files: [time, value]
    - Aligns new data to the reference timeline (from the first existing coordinate)
    - Discretizes the new coordinate and appends it to selected_coordinates.txt
    """

    # Load already discretized coordinates
    coordinates, X_cuts, Labels = load_data_discretization(output_dir + "selected_coordinates.txt")

    # Reference time values from the first known coordinate
    data_zero = open_data_coordinate(output_dir + "coordinates_data/" + coordinates[0] + ".dat")
    times_to_compare = data_zero[:, 0]

    print("\nAdding new coordinates...")
    for i, coord_file in enumerate(coordinates_to_add):
        data_coord_raw = open_data_coordinate(coord_file)
        coord_name = coord_file.split('/')[-1].split('.')[0]
        type_coord = type_coordinates_to_add[i]

        # Set histogram resolution based on coordinate type
        if type_coord == 'distance':
            delta_y = 0.1
        elif type_coord == 'angle':
            delta_y = 2
        else:
            print(f"Unknown coordinate type for {coord_file}, skipping.")
            continue

        # Filter values matching reference times
        y_coord = []
        t_coord = []
        for row in data_coord_raw:
            if row[0] in times_to_compare:
                t_coord.append(row[0])
                y_coord.append(row[1])

        y_coord = np.array(y_coord)
        t_coord = np.array(t_coord)

        # Check for mismatched time alignment
        if len(t_coord) != len(times_to_compare) or not np.allclose(t_coord, times_to_compare):
            print(f"Warning: {coord_file} has different time steps than the reference file. Skipping.")
            continue

        # Fix angle wrapping (e.g., from -180 to 180 or 0 to 360)
        if type_coord == 'angle' and (np.max(y_coord) - np.min(y_coord) > 180):
            y_coord, _, _ = adjust_angle_data(y_coord, np.min(y_coord), np.max(y_coord), delta_y)

        # Discretize and append this coordinate to selected_coordinates.txt
        discretize_coordinate(y_coord, delta_y, coordinate_type=type_coord,
                              times=times_to_compare, time_zero=time_zero, size_block=size_block,
                              coordinate_name=coord_name, output=output_dir + "selected_coordinates.txt",
                              output_dir=output_dir)
    print("New coordinates added and discretized.")


############################ Function to get the discretized array from saved coordinates ##########################
def get_discretized_array(output_dir):
    # Load coordinate names, discretization cutoffs, and corresponding labels
    coordinates, X_cuts, Labels = load_data_discretization(output_dir + "selected_coordinates.txt")

    # Load time information from the first coordinate file (assumes all coordinates share the same time points)
    data_zero = open_data_coordinate(output_dir + "coordinates_data/" + coordinates[0] + ".dat")
    times_to_compare = data_zero[:, 0]  # Extract time column
    nframes = len(times_to_compare)    # Total number of frames (time points)

    # Initialize output array to store discrete labels for each frame and coordinate
    data_discretized = np.zeros((nframes, len(coordinates)), dtype=int)

    print("\nDiscretizing data...")

    # Loop over all selected coordinates
    for i in range(len(coordinates)):
        # Load data for current coordinate
        data_coord = open_data_coordinate(output_dir + "coordinates_data/" + coordinates[i] + ".dat")

        # Loop over all frames
        for f in range(nframes):
            # Compare the current data value to discretization thresholds
            for c in range(len(X_cuts[i])):
                # If the value is less than the current cutoff, assign the corresponding label
                if data_coord[f, 1] < X_cuts[i][c]:
                    data_discretized[f, i] = Labels[i][c]
                    break  # Stop checking more bins once a match is found

                # If value is larger than all cuts, assign the last label
                if c == len(X_cuts[i]) - 1:
                    data_discretized[f, i] = Labels[i][-1]

    print("Discretization completed.")

    # Save the resulting discretized data as a .npy file
    np.save(output_dir + "discretized_array.npy", data_discretized)

########################### Function to compute frequencies of single and double contacts ##########################
def compute_frequencies(Discretized_Array):
    """
    Compute single and double frequencies from a discretized array of coordinates.
    
    Parameters:
    - Discretized_Array (ndarray): shape (nframes, ncoord), with discretized values.
    
    Returns:
    - single_frequencies (ndarray): shape (sum(multiplicities),), 1D frequencies.
    - double_frequencies (ndarray): shape (sum(multiplicities), sum(multiplicities)), 2D joint frequencies.
    """
    nframes, ncoord = Discretized_Array.shape
    multiplicities = get_multiplicities(Discretized_Array)
    max_multiplicity = np.max(multiplicities)
    multiplicity_tot=max_multiplicity * ncoord


    # Allocate frequency arrays
    single_frequencies = np.zeros(multiplicity_tot, dtype=float)
    double_frequencies = np.zeros((multiplicity_tot, multiplicity_tot), dtype=float)

    print("\nComputing single frequencies...")
    previous_progress = -1  # Initialize progress bar
    for i in range(ncoord):
        previous_progress=plot_progress_bar(i, ncoord,previous_progress) 
        col = Discretized_Array[:, i]
        offset=i*max_multiplicity  # Offset for current coordinate in single frequencies
        counts = np.bincount(col, minlength=multiplicities[i])
        single_frequencies[offset:offset + multiplicities[i]] = counts / nframes
    plot_progress_bar(ncoord, ncoord,previous_progress)  # Finalize progress bar    
    print("\nSingle frequencies computed.")

    print("\nComputing double frequencies...")
    total_steps = (ncoord * (ncoord + 1)) // 2
    previous_progress = -1  # Initialize progress bar
    current_step=-1
    for i in range(ncoord):
        
        col_i = Discretized_Array[:, i]
        offset_i = i * max_multiplicity  # Offset for current coordinate in double frequencies
        mult_i = multiplicities[i]

        for j in range(i+1, ncoord):
            current_step+=1
            previous_progress=plot_progress_bar(current_step, total_steps,previous_progress)
            col_j = Discretized_Array[:, j]
            offset_j = j * max_multiplicity  # Offset for current coordinate in double frequencies
            mult_j = multiplicities[j]

            # Fast joint counting
            joint_counts = np.zeros((mult_i, mult_j), dtype=int)
            np.add.at(joint_counts, (col_i, col_j), 1)

            # Normalize to get probabilities
            joint_probs = joint_counts / nframes

            # Store in output matrix
            double_frequencies[offset_i:offset_i + mult_i, offset_j:offset_j + mult_j] = joint_probs
            if i != j:
                double_frequencies[offset_j:offset_j + mult_j, offset_i:offset_i + mult_i] = joint_probs.T

    plot_progress_bar(total_steps, total_steps,previous_progress)  # Finalize progress bar
    print("\nDouble frequencies computed.")

    return single_frequencies, double_frequencies


def get_frequencies(output_dir):
    # Load the discretized array from a .npy file located in the specified output directory
    Discretized_Array = np.load(output_dir + "discretized_array.npy")
    
    # Compute the single and double frequencies using a helper function 
    single_frequencies, double_frequencies = compute_frequencies(Discretized_Array)
    
    # Save the computed single frequencies to a file in the 'frequencies' subdirectory
    np.save(output_dir + 'frequencies/frequencies_single.npy', single_frequencies)
    
    # Save the computed double frequencies to a file in the 'frequencies' subdirectory
    np.save(output_dir + 'frequencies/frequencies_double.npy', double_frequencies)



def compute_couplings_with_SBM(output_dir):
    Discretized_Array = np.load(output_dir + "discretized_array.npy")
    single_frequencies = np.load(output_dir + "frequencies/frequencies_single.npy")
    double_frequencies = np.load(output_dir + "frequencies/frequencies_double.npy")
    multiplicities = get_multiplicities(Discretized_Array)
    ncoord = Discretized_Array.shape[1]

    #print (np.average(single_frequencies), np.average(double_frequencies))

    # Create a toy dataset containing only the first 5 coordinates
    toy_data = Discretized_Array[:, :50]


    single_frequencies, double_frequencies = compute_frequencies(toy_data)

    multiplicities = get_multiplicities(toy_data)
    #print(np.shape(multiplicities))
    ncoord = toy_data.shape[1]

    max_iterations = 20 #iterations for the training
    cutoff_loss = 1e-5
    learning_rate = 0.001
    nsteps = 5000000 #length MC
   
    print("\nComputing couplings with SBM...")
    H,J,loss=sbm.train_coupled_MC (multiplicities,ncoord,max_iterations,cutoff_loss,single_frequencies,double_frequencies,learning_rate,nsteps)
    print("Couplings computed.")
    
    plt.plot(loss)
    plt.show()
    plt.close()
    

    #np.save(output_dir + "frequencies/H_sbm.npy", H)
    #np.save(output_dir + "frequencies/J_sbm.npy", J)
    
    #H= np.load(output_dir + "frequencies/H_sbm.npy")
    #J= np.load(output_dir + "frequencies/J_sbm.npy")
                
    print("\nTesting SBM couplings with Monte Carlo simulation...")
    starting_coords=sbm.get_starting_coords(multiplicities,ncoord,len(H),np.max(multiplicities))
    Traj_MC = sbm.monte_carlo(starting_coords, H, J, nsteps, len(H),np.max(multiplicities), ncoord,multiplicities)
    single_mc = np.zeros(len(H), dtype=np.float64)
    double_mc = np.zeros((len(H), len(H)), dtype=np.float64)
    compute_frequencies(Traj_MC, nsteps, len(H), multiplicities, ncoord, np.max(multiplicities), single_mc, double_mc)

    delta_frequencies_single = single_frequencies - single_MC
    delta_frequencies_double = double_frequencies - double_MC

    print("Single frequencies from SBM:", single_frequencies)
    print("Single frequencies from Monte Carlo:", single_MC)
    print("Double frequencies from SBM:", double_frequencies)
    print("Double frequencies from Monte Carlo:", double_MC)










def get_positions_baricenters(u_traj,output_dir,RESIDS_SELECTED,indices_aa,terminal_atoms,coordinates_to_add,barycenter_coordinates_to_add):
    times_indices=np.load(output_dir+'times_indices.npy')
    coordinates,X_cuts,Labels=load_data_discretization(output_dir+"selected_coordinates.txt")
    ncoord=len(coordinates)
    data_zero=open_data_coordinate(output_dir+"coordinates_data/"+coordinates[0]+".dat")
    times_to_compare=data_zero[:,0]
    nframes=len(times_to_compare)
    if len(indices_aa) >= 2:
        Positions_atoms_CA = np.load(output_dir+"Positions_npy/Positions_CA_atoms.npy")
        Positions_atoms_C = np.load(output_dir+"Positions_npy/Positions_C_atoms.npy")
        Positions_atoms_N = np.load(output_dir+"Positions_npy/Positions_N_atoms.npy")
    Positions_terminal_atoms = np.load(output_dir+"Positions_npy/Positions_terminal_atoms.npy")

    Positions_barycenters=np.zeros((ncoord,nframes,3))

    name_coord_to_add=[]
    for coord_file in coordinates_to_add:
        name_coord_to_add.append(coord_file.split('/')[-1].split('.')[0])
    resids_coord_to_add=[int(barycenter.split('_')[0]) for barycenter in barycenter_coordinates_to_add]
    atoms_coord_to_add=[barycenter.split('_')[1] for barycenter in barycenter_coordinates_to_add]

    print("Computing barycenters...")
    for i in range(len(coordinates)):
        plot_progress_bar(i,len(coordinates))
        coord=coordinates[i]

        if coord[:3]=='phi':
            index_resid=int(coord[3:])
            ind_pos=indices_aa.index(index_resid)
            Positions_barycenters[i]=(Positions_atoms_C[ind_pos-1,:, :]+Positions_atoms_N[ind_pos,:, :]+Positions_atoms_CA[ind_pos , :, :]+Positions_atoms_C[ind_pos, :, :])/4
        elif coord[:3]=='psi':
            index_resid=int(coord[3:])
            ind_pos=indices_aa.index(index_resid)
            Positions_barycenters[i]=(Positions_atoms_N[ind_pos, :, :]+Positions_atoms_CA[ind_pos, :, :]+Positions_atoms_C[ind_pos , :, :]+Positions_atoms_N[ind_pos+1, :, :])/4
        elif coord in name_coord_to_add :
            index_coord=name_coord_to_add.index(coord)
            atom_selection= u_traj.select_atoms(f"resid {resids_coord_to_add[index_coord]} and name {atoms_coord_to_add[index_coord]}")
            for k, frame in enumerate(times_indices):
                plot_progress_bar(k, len(times_indices))
                u_traj.trajectory[frame]
                Positions_barycenters[i, k, :] = atom_selection.positions
        else :
            resid1=int(coord.split('_')[0])
            atom1=coord.split('_')[1]
            resid2=int(coord.split('_')[2])
            atom2=coord.split('_')[3]
            index_term1=-1
            index_term2=-1
            index_CA1=-1
            index_CA2=-1
            if atom1 == 'CA':
                index_CA1=indices_aa.index(resid1)
            else :
                index_resid=RESIDS_SELECTED.index(resid1)
                ind_at=terminal_atoms[index_resid].index(atom1)
                index_term1=int(np.sum([len(terminal_atoms[k]) for k in range(index_resid)])+ind_at)

            if atom2 == 'CA':
                index_CA2=indices_aa.index(resid2)
            else :
                index_resid=RESIDS_SELECTED.index(resid2)
                ind_at=terminal_atoms[index_resid].index(atom2)
                index_term2=int(np.sum([len(terminal_atoms[k]) for k in range(index_resid)])+ind_at)
            
            if index_term1 != -1 and index_term2 != -1:
                Positions_barycenters[i]=(Positions_terminal_atoms[index_term1,:,:]+Positions_terminal_atoms[index_term2,:,:])/2
            elif index_CA1 != -1 and index_term2 != -1:
                Positions_barycenters[i]=(Positions_atoms_CA[index_CA1,:,:]+Positions_terminal_atoms[index_term2,:,:])/2
            elif index_term1 != -1 and index_CA2 != -1:
                Positions_barycenters[i]=(Positions_terminal_atoms[index_term1,:,:]+Positions_atoms_CA[index_CA2,:,:])/2
            elif index_CA1 != -1 and index_CA2 != -1:
                Positions_barycenters[i]=(Positions_atoms_CA[index_CA1,:,:]+Positions_atoms_CA[index_CA2,:,:])/2
    plot_progress_bar(len(coordinates),len(coordinates))
    print("\nBarycenters computed.")
    np.save(output_dir+"Positions_npy/Positions_barycenters.npy",Positions_barycenters)

def get_avg_distances_barycenters(output_dir):
    Positions_barycenters=np.load(output_dir+"Positions_npy/Positions_barycenters.npy")
    ncoord,nframes,dim=Positions_barycenters.shape
    avg_distances=np.zeros((ncoord,ncoord))
    print("Computing average distances...")
    for i in range(ncoord):
        for j in range(i+1,ncoord):
            plot_progress_bar(i*ncoord+j,ncoord*ncoord)
            avg_distances[i,j]=np.mean(np.linalg.norm(Positions_barycenters[i,:,:]-Positions_barycenters[j,:,:],axis=1))
            avg_distances[j,i]=avg_distances[i,j]
    plot_progress_bar(ncoord*ncoord,ncoord*ncoord)
    print("\nAverage distances computed.")
    np.save(output_dir+"analysis/avg_distances_barycenters.npy",avg_distances)



def mutual_information(Discretized_Array,multiplicities,single_frequencies,double_frequencies):
    nframes,ncoord=np.shape(Discretized_Array)
    multiplicities=get_multiplicities(Discretized_Array)
    MI=np.zeros((ncoord,ncoord),dtype=float)

    index_freq_1=0
    for i in range(ncoord):
        for xi in range(multiplicities[i]):
            index_freq_2=0
            for j in range(ncoord):
                for xj in range(multiplicities[j]):
                    probab_xi=single_frequencies[index_freq_1]
                    probab_xj=single_frequencies[index_freq_2]
                    prob_xi_xj=double_frequencies[index_freq_1,index_freq_2]
                    if prob_xi_xj>0:
                        MI[i,j]+=prob_xi_xj*np.log(prob_xi_xj/(probab_xi*probab_xj))
                    index_freq_2+=1
            index_freq_1+=1
    return MI



            
def plot_mutual_information(MI,output_dir,name_out):
    plt.figure(figsize=(10, 6))
    plt.imshow(MI, cmap='jet', interpolation='nearest')
    plt.colorbar(label='Mutual Information')
    plt.title('Mutual Information Matrix')
    plt.xlabel('Coordinate Index')
    plt.ylabel('Coordinate Index')
    plt.savefig(output_dir+'MI_plots/'+name_out+'.png', dpi=200)
    plt.close()

def plot_mutual_information_with_names(labels,MI,output_dir,name_out):
    plt.figure(figsize=(14, 13))
    fonts=min(50*7/len(labels),11)
    plt.imshow(MI, cmap='jet', interpolation='nearest')
    plt.colorbar(label='Mutual Information')
    plt.title('Mutual Information Matrix')
    plt.xticks(range(len(labels)), labels, rotation=90, fontsize=fonts)
    plt.yticks(range(len(labels)), labels, fontsize=fonts)
    plt.xlabel('Coordinate Index')
    plt.ylabel('Coordinate Index')
    plt.savefig(output_dir+'MI_plots/'+name_out+'.png', dpi=200)
    plt.close()

def plot_MI_vs_distance(MI,output_dir,avg_distances_barycenters):
    plt.figure(figsize=(10, 6))
    plt.scatter(avg_distances_barycenters.flatten(), MI.flatten(),marker='x', color='blue', alpha=0.5)
    plt.xlabel('Average Distance (A)')
    plt.ylabel('Mutual Information')
    plt.title('Mutual Information vs Average Distance')
    plt.savefig(output_dir+'MI_plots/MI_vs_distance_plot.png', dpi=200)
    plt.close()

def plot_MI_vs_distance_clusters(MI,output_dir,avg_distances_barycenters,clusters_ndx):
    print('\n')
    print("Plotting MI vs distance for clustered data...")
    plt.figure(figsize=(10, 6))
    print("Plotting noise data...")
    distance_noise=avg_distances_barycenters[np.ix_(clusters_ndx[-1], clusters_ndx[-1])]
    MI_noise=MI[np.ix_(clusters_ndx[-1], clusters_ndx[-1])]
    
    plt.scatter(distance_noise.flatten(), MI_noise.flatten(),marker='x', color='grey', alpha=0.3,label='Noise')
    
    for i in range(len(clusters_ndx)-1):
        print(f"Plotting cluster {i} data...")
        distance_i = avg_distances_barycenters[np.ix_(clusters_ndx[i], clusters_ndx[i])]
        MI_i = MI[np.ix_(clusters_ndx[i], clusters_ndx[i])]
        if len(clusters_ndx) > 2:
            plt.scatter(distance_i.flatten(), MI_i.flatten(),marker='x', alpha=0.8,label=f'Cluster {i}',color=plt.cm.rainbow(i / (len(clusters_ndx)-2)))
        else:
            plt.scatter(distance_i.flatten(), MI_i.flatten(),marker='x', alpha=0.8,label=f'Cluster {i}',color='blue')
    plt.xlabel('Average Distance (A)')
    plt.ylabel('Mutual Information')
    plt.title('Mutual Information vs Average Distance')
    plt.legend()
    plt.savefig(output_dir+'MI_plots/MI_vs_distance_plot_clustered.png', dpi=200)
    plt.close()

def compute_running_avg_and_plot(distances, mi_values, label, color=None):
    distances = distances[~np.eye(distances.shape[0], dtype=bool)].flatten()
    mi_values = mi_values[~np.eye(mi_values.shape[0], dtype=bool)].flatten()
    sorted_indices = np.argsort(distances)
    distances = distances[sorted_indices]
    mi_values = mi_values[sorted_indices]

    window_size = max(1, len(distances) // 10)
    running_avg = uniform_filter1d(mi_values, size=window_size, mode='nearest')
    squared_diff = (mi_values - running_avg) ** 2
    error_bars = np.sqrt(uniform_filter1d(squared_diff, size=window_size, mode='nearest'))

    plt.plot(distances[:len(running_avg)], running_avg, label=label, color=color)
    plt.fill_between(distances[:len(running_avg)], running_avg - error_bars, running_avg + error_bars, alpha=0.3, color=color)

def plot_runningavg_MI_vs_distance_clusters(MI, output_dir, avg_distances_barycenters, clusters_ndx):
    print('\n')
    print("Plotting running average MI vs distance for clustered data...")
    plt.figure(figsize=(10, 6))

    

    print("Plotting noise data...")
    compute_running_avg_and_plot(
        avg_distances_barycenters[np.ix_(clusters_ndx[-1], clusters_ndx[-1])],
        MI[np.ix_(clusters_ndx[-1], clusters_ndx[-1])],
        label='Noise', color='grey'
    )

    for i, cluster in enumerate(clusters_ndx[:-1]):
        print(f"Plotting cluster {i} data...")
        compute_running_avg_and_plot(
            avg_distances_barycenters[np.ix_(cluster, cluster)],
            MI[np.ix_(cluster, cluster)],
            label=f'Cluster {i}',
            color=plt.cm.rainbow(i / (len(clusters_ndx)-2))
        )

    plt.xlabel('Average Distance (A)')
    plt.ylabel('Mutual Information')
    plt.title('Running average of mutual Information vs Average Distance')
    plt.legend()
    plt.savefig(output_dir + 'MI_plots/avg_MI_vs_distance_plot_clustered.png', dpi=200)
    plt.close()

                


def get_mutual_information(output_dir):
    print("Computing mutual information...")
    Discretized_Array=np.load(output_dir+"discretized_array.npy")
    single_frequencies=np.load(output_dir+'frequencies/frequencies_single.npy')
    double_frequencies=np.load(output_dir+'frequencies/frequencies_double.npy')
    avg_distances_barycenters=np.load(output_dir+"analysis/avg_distances_barycenters.npy")
    multiplicities=get_multiplicities(Discretized_Array)
    MI=mutual_information(Discretized_Array,multiplicities,single_frequencies,double_frequencies)
    np.save(output_dir+'analysis/MI.npy', MI)
    print("Mutual information computed.")
    plot_mutual_information(MI,output_dir,'MI_matrix')
    plot_MI_vs_distance(MI,output_dir,avg_distances_barycenters)

def get_entropy(output_dir):
    print("Computing entropy...")
    Discretized_Array=np.load(output_dir+"discretized_array.npy")
    single_frequencies=np.load(output_dir+'frequencies/frequencies_single.npy')
    multiplicities=get_multiplicities(Discretized_Array)
    ncoord=len(multiplicities)
    entropy=np.zeros((ncoord),dtype=float)
    count_index=0
    for i in range(ncoord):
        for xi in range(multiplicities[i]):
            probab_xi=single_frequencies[count_index]
            count_index+=1
            if probab_xi>0:
                entropy[i]-=probab_xi*np.log(probab_xi)

    np.save(output_dir+'analysis/entropy.npy', entropy)
    print("Entropy computed.")
    
    plt.scatter(range(len(entropy)), entropy, color='blue', alpha=0.5)
    plt.xlabel('Index coordinate')
    plt.ylabel('Entropy')
    plt.title('Entropy by Coordinate')
    plt.savefig(output_dir+'analysis/entropy_by_coordinate.png', dpi=200)
    plt.close()

def yacare_clusterization(output_dir,name_cluster_dir,step_to_perform,number_of_coords,distance_matrix,min_size_cluster,function_for_ratio,threshold_variable,amount_of_noise,percentage_moving_square):
    variables=yacare.Variables()
    variables.project_name = name_cluster_dir
    variables.save_images = True
    variables.distance_matrix=distance_matrix
    yacare.perform_first_reordering(variables, percentage_moving_square = percentage_moving_square)
    yacare.find_optimal_cutoff(variables, minimal_size_cluster = min_size_cluster,function_for_ratio=function_for_ratio)
    if step_to_perform != 'all' :
        yacare.choose_if_we_reorder_again(variables)
        yacare.change_proposed_cutoff(variables)
    else :
        yacare.choose_if_we_reorder_again(variables,indices=np.arange(0,number_of_coords))
    yacare.find_final_clusters(variables)
    print("Number of clusters before merging: "+str(variables.number_clusters))
    if variables.number_clusters>1 :
        yacare.compare_clusters(variables, display_stddev = True)
        yacare.propose_list_for_concatenating_clusters(variables, threshold_variable = threshold_variable, choice_merging_clusters=3)
        yacare.concatenate_clusters(variables)
    yacare.expand_clusters(variables, amount_of_noise = amount_of_noise)
    yacare.write_indices(variables)

    os.system('mkdir -p '+output_dir+variables.project_name)
    os.system('mv '+variables.project_name+'* '+output_dir+variables.project_name)

def get_cluster_indexes_from_yacare(output_dir, cluster_dir):
    """
    Extracts cluster indexes from Yacare output.
    """
    print("Extracting cluster indexes from Yacare output...")
    data_yacare, lines_yacare = open_file(output_dir + cluster_dir + '/' + cluster_dir + '_Clustering_Clusters.ndx')
    clusters_ndx = []
    cluster_i = []

    for l in range(len(lines_yacare)):
        line = lines_yacare[l]
        if line[0] == '[':
            if len(cluster_i) >= 1:
                clusters_ndx.append(cluster_i)
            cluster_i = []
            continue
        else:
            for i in range(len(data_yacare[l])):
                index_coord = int(data_yacare[l][i]) - 1
                cluster_i.append(index_coord)

    if len(cluster_i) >= 1:
        clusters_ndx.append(cluster_i)

    print("Cluster indexes extracted.")
    for i in range(len(clusters_ndx)):
        clusters_ndx[i] = sorted(clusters_ndx[i])
    return clusters_ndx

def get_representative_structure_from_yacare(output_dir, cluster_dir):
    """
    Extracts cluster indexes from Yacare output.
    """
    print("Extracting cluster indexes from Yacare output...")
    data_yacare, lines_yacare = open_file(output_dir + cluster_dir + '/' + cluster_dir + '_Clustering_RepresentativeStructures.ndx')
    Representative_structures= []
    cluster_i = []

    for l in range(len(lines_yacare)):
        if len(data_yacare[l])==1:
            Representative_structures.append(int(data_yacare[l][0])-1)
    print("Cluster indexes extracted.")
    return Representative_structures


def write_clusters_to_file(clusters_ndx, coordinates, output_dir, name_output_cluster):
    """
    Writes cluster information to a file.
    """
    print("Writing clusters to file...")
    with open(output_dir + name_output_cluster, 'w') as file_out:
        for i, cluster_i in enumerate(clusters_ndx):
            file_out.write('\n\n')
            if i != len(clusters_ndx) - 1:
                file_out.write(f'[ Cluster{i} ]\n')
            else:
                file_out.write(f'[ Noise ]\n')
            for index_coord in cluster_i:
                file_out.write(f'{coordinates[index_coord]} \n')

    print("Clusters written to file.")
     

def convert_clusters_yacare_to_real_coordinates(output,output_dir,cluster_dir,name_output_cluster):
    print("Converting clusters to real coordinates...")
    coordinates,X_cuts,Labels=load_data_discretization(output)
    
    clusters_ndx=get_cluster_indexes_from_yacare(output_dir,cluster_dir)
    write_clusters_to_file(clusters_ndx, coordinates, output_dir, name_output_cluster)
    return clusters_ndx,coordinates

def get_resids_in_clusters(clusters_ndx,coordinates,name_coordinates_to_add,barycenter_coordinates_to_add,name_output,output_dir):
    print("Getting resids in clusters...")
    file_out=open(output_dir+name_output,'w')
    for i in range (len(clusters_ndx)):
        cluster_i=clusters_ndx[i]
        file_out.write('\n\n')
        if i!=len(clusters_ndx)-1:
            file_out.write(f'[ Cluster{i} ]\n')
        else:
            file_out.write(f'[ Noise ]\n')
        resids_in_cluster_i=[]
        for j in range(len(cluster_i)):
            index_coord=cluster_i[j]
            coord=coordinates[index_coord]
            if coord in name_coordinates_to_add:
                index_coord_to_add=name_coordinates_to_add.index(coord)
                name_atom_to_add=barycenter_coordinates_to_add[index_coord_to_add]
                name_resid_to_add=int(name_atom_to_add.split('_')[0])
                if name_resid_to_add not in resids_in_cluster_i:
                    resids_in_cluster_i.append(name_resid_to_add)
                    
            elif coord[:3]=='phi' or coord[:3]=='psi':
                name_resid_to_add=int(coord[3:])
                if name_resid_to_add not in resids_in_cluster_i:    
                    resids_in_cluster_i.append(name_resid_to_add)
                    
            else:
                name_resid_to_add=int(coord.split('_')[0])
                if name_resid_to_add not in resids_in_cluster_i:
                    resids_in_cluster_i.append(name_resid_to_add)
                    
                
                name_resid_to_add=int(coord.split('_')[2])
                if name_resid_to_add not in resids_in_cluster_i:
                    resids_in_cluster_i.append(name_resid_to_add)
        resids_in_cluster_i.sort()
        for j in range(len(resids_in_cluster_i)):
            file_out.write(f'{resids_in_cluster_i[j]} ')
        
    print("Getting resids in clusters completed.")
    file_out.close()

def MI_map_for_clusters(coordinates,MI,clusters_ndx,output_dir):
    
    if os.path.exists(f'{output_dir}MI_plots/Maps_by_cluster'):
        os.system(f'rm -r {output_dir}MI_plots/Maps_by_cluster')
    os.makedirs(f'{output_dir}MI_plots/Maps_by_cluster', exist_ok=True)
    print('\n')
    print("Creating MI maps for clusters...")
    for i in range(len(clusters_ndx)-1):
        print(f"Creating MI map for cluster {i}...")
        cluster_i=clusters_ndx[i]
        cluster_i_MI=np.zeros((len(cluster_i),len(cluster_i)),dtype=float)
        names_cluster_i=[coordinates[cluster_i[j]] for j in range(len(cluster_i))]
        for j in range(len(cluster_i)):
            for k in range(len(cluster_i)):
                index_coord_1=cluster_i[j]
                index_coord_2=cluster_i[k]
                if j!=k :
                    cluster_i_MI[j,k]=MI[index_coord_1,index_coord_2]
        max_MI=np.max(cluster_i_MI)
        for j in range(len(cluster_i)):
            cluster_i_MI[j,j]=max_MI
        
        plot_mutual_information_with_names(names_cluster_i,cluster_i_MI,output_dir,f'Maps_by_cluster/Cluster_{i}_MI_map')

def MI_map_reordered_by_clusters(coordinates,MI,clusters_ndx,output_dir):
    print('\n')
    print("Creating the reordered MI map from clusters...")
    reordered_MI=np.zeros((len(coordinates),len(coordinates)),dtype=float)
    reordered_indexes = [index for cluster in clusters_ndx for index in cluster]
    for i in range(len(reordered_indexes)):
        for j in range(len(reordered_indexes)):
            reordered_MI[i,j]=MI[reordered_indexes[i],reordered_indexes[j]]
    plot_mutual_information(reordered_MI,output_dir,'MI_matrix_reordered')

def get_times_cluster_states(cluster_i_states,output_dir,selected_states,times,ind_cluster):
    cluster_i_states=cluster_i_states.astype(str)
    file_indices=open(output_dir+f'Clusterize_MI/clusters_states/times_indices_cluster{ind_cluster}.ndx','w')
    file_representative=open(output_dir+f'times_indices_clusters_states_some_structures.txt','a')
    for j in range(len(selected_states)):
        file_indices.write(f'[ State{j} ] \n')
        file_representative.write(f'[ Cluster{ind_cluster}_State{j} ] \n')
        indexes_not_X=np.where(np.array(selected_states[j])!='X')[0]
        cluster_i_states_not_X = cluster_i_states[:, indexes_not_X]
        selected_states_not_X = np.array(selected_states)[j,indexes_not_X] 
        indices_state=np.where(np.all(cluster_i_states_not_X==selected_states_not_X,axis=1))[0]
        indices_from_times=times[indices_state]
        indices_from_times=indices_from_times.astype(int)
        for idx, time in enumerate(indices_from_times):
            file_indices.write(f"{time} ")
            if (idx + 1) % 15 == 0:
                file_indices.write("\n")
        file_indices.write("\n\n")
        random_indices = np.random.choice(indices_from_times, size=min(10, len(indices_from_times)), replace=False)
        random_indices.sort()
        for random_index in random_indices:
            file_representative.write(f"{random_index}\n")
        file_representative.write("\n\n")
    file_indices.close()
        

def get_states_from_clusters(clusters_ndx,output_dir,times_indices,number_of_states_to_show):
    if os.path.exists(f'{output_dir}Clusterize_MI/clusters_states') :
        os.system(f'rm -r {output_dir}Clusterize_MI/clusters_states')
    os.system(f'mkdir -p {output_dir}Clusterize_MI/clusters_states')


    if os.path.exists(f'{output_dir}times_indices_clusters_states_some_structures.txt'):
        os.system(f'rm {output_dir}times_indices_clusters_states_some_structures.txt')

    Discretized_Array=np.load(output_dir+"discretized_array.npy")
    nframes,ncoord=np.shape(Discretized_Array)
    file_out=open(output_dir+'clusters_states.txt','w')
    print('\n')
    print("Getting states from clusters...")
    for i in range(len(clusters_ndx)-1):
        print(f"Getting states from cluster {i}...")
        ind_cluster=0
        file_out.write(f'Cluster {i} states:\n')
        cluster_i=clusters_ndx[i]
        cluster_i_states=np.zeros((nframes,len(cluster_i)),dtype=int)
        for j in range(len(cluster_i)):
            index_coord=cluster_i[j]
            cluster_i_states[:,j]=Discretized_Array[:,index_coord]
        np.save(output_dir+f'Clusterize_MI/clusters_states/cluster_{i}_states.npy',cluster_i_states)
        
        unique_states,count_unique_states=np.unique(cluster_i_states,axis=0,return_counts=True)
        
        probabilities=count_unique_states/nframes
        
        unique_merged_states_with_probabilities = sorted(
            zip(unique_states, probabilities),
            key=lambda x: x[1],
            reverse=True
        )
        sum_probababilities = 0
        selected_states=[]
        for k in range (min(number_of_states_to_show,len(unique_merged_states_with_probabilities))):
            state, probability = unique_merged_states_with_probabilities[k]
            state = state.astype(str)
            
            sum_probababilities += probability
            file_out.write(f"State: ")
            file_out.write(', '.join(state))
            file_out.write(f"   Probability: {probability:.6f} \n")
            selected_states.append(state)
        file_out.write(f"Sum of probabilities: {sum_probababilities:.6f} \n")
        file_out.write('\n\n')
        get_times_cluster_states(cluster_i_states,output_dir,selected_states,times_indices,i)
    
    file_out.close()
    

def clusterize_MI(output_dir,coordinates_to_add,barycenter_coordinates_to_add,step_to_perform,number_of_states_to_show):
    times_indices=np.load(output_dir+"times_indices.npy")
    name_coordinates_to_add=[coord.split('/')[-1].split('.')[0] for coord in coordinates_to_add]
    avg_distances_barycenters=np.load(output_dir+"analysis/avg_distances_barycenters.npy")
    MI=np.load(output_dir+'analysis/MI.npy')
    MI_no_diag=np.copy(MI)
    for i in range (len(MI_no_diag)):
        MI_no_diag[i,i]=0
    ncoord=len(MI)
    max_MI=np.max(MI_no_diag)
    distance_MI=np.zeros((len(MI),len(MI)),dtype=float)
    for i in range(len(MI)):
        for j in range(len(MI)):
            distance_MI[i,j]=-MI_no_diag[i,j]+max_MI
        distance_MI[i,i]=0

    min_size_cluster,function_for_ratio,threshold_variable,amount_of_noise,percentage_moving_square=0.0001,2,1.0,0.3,1
    yacare_clusterization(output_dir,'Clusterize_MI',step_to_perform,ncoord,distance_MI, min_size_cluster,function_for_ratio,threshold_variable,amount_of_noise,percentage_moving_square)
    clusters_ndx,coordinates=convert_clusters_yacare_to_real_coordinates(output_dir+"selected_coordinates.txt",output_dir,'Clusterize_MI','Clusters_of_coordinates_from_MI.txt')
    os.system(f'cp {output_dir}Clusterize_MI/Clusterize_MI_Yacare_11-Matrix-WithNoise.png {output_dir}MI_plots/')
    os.system(f'mv {output_dir}distance_MI.csv {output_dir}Clusterize_MI/')
    get_resids_in_clusters(clusters_ndx,coordinates,name_coordinates_to_add,barycenter_coordinates_to_add,'resids_in_clusters_from_MI.txt',output_dir)
    plot_MI_vs_distance_clusters(MI,output_dir,avg_distances_barycenters,clusters_ndx)
    plot_runningavg_MI_vs_distance_clusters(MI,output_dir,avg_distances_barycenters,clusters_ndx)
    MI_map_reordered_by_clusters(coordinates,MI,clusters_ndx,output_dir)
    MI_map_for_clusters(coordinates,MI,clusters_ndx,output_dir)
    get_states_from_clusters(clusters_ndx,output_dir,times_indices,number_of_states_to_show)

def get_euclidian_distance_between_conformations(array_cluster):
    """
    Computes the Euclidean distance matrix between conformations in the array_cluster.
    Uses scipy's pdist and squareform for efficiency.
    """

    print("Computing Euclidean distance matrix...")
    distance_matrix = squareform(pdist(array_cluster, metric='euclidean'))
    print("Euclidean distance matrix computed.")
    return distance_matrix

def get_representative_frames(unique_states, representative_structures, times_indices, array_cluster):
    """
    Get representative frames for each conformation.
    """
    frames_representative_structures = []
    for i in range(len(representative_structures)):
        frame_index = np.where((array_cluster == unique_states[representative_structures[i]]).all(axis=1))[0][0]
        frames_representative_structures.append(times_indices[frame_index])
    return frames_representative_structures

def calculate_conformation_probabilities(clusters_ndx, probabilities):
    """
    Calculate probabilities for each conformation in a cluster.
    """
    conformation_probabilities = []
    for cluster in clusters_ndx:
        probability = sum(probabilities[ind - 1] for ind in cluster)
        conformation_probabilities.append(probability)
    return conformation_probabilities

def write_conformation_to_file(file_out, conformation_index, representative_structure, frame, probability, coordinates, cluster_coordinates):
    """
    Write details of a conformation to the output file.
    """
    file_out.write(f"Conformation {conformation_index}:\n")
    file_out.write(f"Representative structure: {', '.join(representative_structure)}\n")
    file_out.write(f"Representative structure frame: {frame}\n")
    file_out.write(f"Probability: {probability:.6f}\n")
    file_out.write("Coordinates:\n")
    for coord, value in zip(cluster_coordinates, representative_structure):
        file_out.write(f"{coord}: {value}\n")
    file_out.write("\n")

def get_frames_in_conformation(unique_states, clusters_ndx, times_indices, array_cluster,output_dir,ind_cluster):
    """
    Get frames in each conformation.
    """
    frames_in_conformation = []
    print(len(clusters_ndx))
    for i in range(len(clusters_ndx)):
        frames_in_cluster_i = []
        for j in range(len(clusters_ndx[i])):
            state= clusters_ndx[i][j]
            frame_indices = np.where((array_cluster == unique_states[state]).all(axis=1))[0]
            frames_in_cluster_i+=list(times_indices[frame_indices])
        frames_in_cluster_i.sort()
        frames_in_conformation.append(frames_in_cluster_i)
    file_conf_out=open(output_dir+'Get_conformations_cluster'+str(ind_cluster)+'/cluster'+str(ind_cluster)+'_conformations.ndx','w')
    for i in range(len(frames_in_conformation)-1):
        file_conf_out.write(f'[ Conformation{i} ] \n')
        for j in range(len(frames_in_conformation[i])):
            file_conf_out.write(f"{frames_in_conformation[i][j]} ")
            if (j + 1) % 15 == 0:
                file_conf_out.write("\n")
        file_conf_out.write("\n\n")
    file_conf_out.write(f'[ Noise ] \n')
    for j in range(len(frames_in_conformation[-1])):
        file_conf_out.write(f"{frames_in_conformation[-1][j]} ")
        if (j + 1) % 15 == 0:
            file_conf_out.write("\n")
    file_conf_out.close()
    

def get_proba_conformation(unique_states, probabilities, output_dir, cluster_dir, ind_cluster, clusters_coordinates_ndx, coordinates, times_indices, array_cluster):
    """
    Main function to process conformations and write results to file.
    """
    clusters_ndx = get_cluster_indexes_from_yacare(output_dir, cluster_dir)
    representative_structures = get_representative_structure_from_yacare(output_dir, cluster_dir)
    frames_representative_structures = get_representative_frames(unique_states, representative_structures, times_indices, array_cluster)
    conformation_probabilities = calculate_conformation_probabilities(clusters_ndx, probabilities)
    get_frames_in_conformation(unique_states, clusters_ndx, times_indices, array_cluster,output_dir, ind_cluster)

    cluster_coordinates = [coordinates[ndx] for ndx in clusters_coordinates_ndx[ind_cluster]]
    file_out = open(output_dir + 'clusters_conformations.txt', 'a')
    file_out.write(f"\n#######################################################################\n")
    file_out.write(f"Cluster {ind_cluster} conformations:\n\n")

    total_probabilities = 0
    for i, (probability, frame, representative_structure) in enumerate(zip(conformation_probabilities[:-1], frames_representative_structures, unique_states[representative_structures])):
        representative_structure = representative_structure.astype(str)
        write_conformation_to_file(file_out, i, representative_structure, frame, probability, coordinates, cluster_coordinates)
        total_probabilities += probability

    file_out.write(f"Total probability: {total_probabilities:.6f}\n\n")
    file_out.close()


def cluster_states(output_dir):
    data_clusters,_=open_file(output_dir+'clusters_states.txt')
    Indexes_of_clusters=[]
    file_out=open(output_dir+'clusters_conformations.txt','w')
    file_out.write('Clusters conformations:\n\n')
    file_out.close()
    clusters_coordinates_ndx,coordinates=convert_clusters_yacare_to_real_coordinates(output_dir+"selected_coordinates.txt",output_dir,'Clusterize_MI','Clusters_of_coordinates_from_MI.txt')
    times_indices=np.load(output_dir+"times_indices.npy")
    for i in range(len(data_clusters)):
        if len(data_clusters[i])>1 and data_clusters[i][0]=='Cluster':
            Indexes_of_clusters.append(int(data_clusters[i][1]))
    for i in range(len(Indexes_of_clusters)):

        print('\n')
        print(f"Getting conformations from cluster {i}...")
        Ind_i=Indexes_of_clusters[i]
        array_cluster=np.load(output_dir+f'Clusterize_MI/clusters_states/cluster_{Ind_i}_states.npy')
        unique_states,count_unique_states=np.unique(array_cluster,axis=0,return_counts=True)
        if len(unique_states)>100:
            probabilities=count_unique_states/len(array_cluster)
            distance_matrix=get_euclidian_distance_between_conformations(unique_states)
            print(f"Doing clusterization for cluster {i}...")
            print(Ind_i)
            min_size_cluster,function_for_ratio,threshold_variable,amount_of_noise,percentage_moving_square=1,1,1.0,1.0,2
            yacare_clusterization(output_dir,'Get_conformations_cluster'+str(Ind_i),'Get_conformations_cluster'+str(Ind_i),len(distance_matrix),distance_matrix,min_size_cluster,function_for_ratio,threshold_variable,amount_of_noise,percentage_moving_square)
            get_proba_conformation(unique_states,probabilities,output_dir,'Get_conformations_cluster'+str(Ind_i),Ind_i,clusters_coordinates_ndx,coordinates,times_indices,array_cluster)
        else :
            print("Not enough conformations to clusterize.")
            file_out=open(output_dir+'clusters_conformations.txt','a')
            file_out.write(f'\n####################################################################### \n')
            file_out.write(f'Cluster {i} states:\n')
            for j in range(min(10,len(unique_states))):
                state_j=unique_states[j]
                state_j = state_j.astype(str)
                file_out.write(f"State: ")
                file_out.write(str(state_j))
                file_out.write(f"   Probability: {count_unique_states[j]/len(array_cluster):.6f} \n")
            file_out.write('\n\n')
            file_out.close()
    file_out.close()

    


