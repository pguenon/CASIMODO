import os
import shutil
import logging
from datetime import datetime
import io
import sys
from contextlib import redirect_stdout, redirect_stderr

import numpy as np	

from scipy.stats import t
from scipy.spatial.distance import pdist, squareform

from sklearn.neighbors import KernelDensity

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import MDAnalysis as mda 






###################### INITIATE LOGGING #####################
def initiate_logging(output_dir,step_to_perform):
    """
    Initializes logging to a file in the specified output directory.

    Parameters:
    - output_dir (str): The directory where the log file will be created.
    - step_to_perform (str): The step being performed, used for logging context.
    Returns:
    - None
    """
    now= datetime.now()
    log_file = os.path.join(output_dir, 'casimodo.log')
    if step_to_perform == 'all':
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(message)s',     
            filemode='w' 
        )
        logging.info("Logging initiated. Log file created at: %s", log_file)
        logging.info("Start time: %s", now.strftime("%Y-%m-%d %H:%M:%S"))
    else:
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(message)s',     
            filemode='a' 
        )
        logging.info("\n\n\n\n\n\n\n\n\n")
        logging.info("Logging initiated for step: %s. Log file updated at: %s", step_to_perform, log_file)
        logging.info("Start time: %s", now.strftime("%Y-%m-%d %H:%M:%S"))


###################### PRINT LOGO #####################
def print_header():
    with open("CASIMODO_utils/header_casimodo.txt", encoding="utf-8") as f:
        header = f.read()
    logging.info(header)


####################### PRINT INPUTS #####################
def print_inputs(
    output_dir, 
    step_to_perform, 
    strucfile, trajfile, dic,
    time_zero, delta_time, size_block,
    cutoff_distance,proba_under_cutoff_distance, delta_resid, mode_proba_cutoff,
    method_clustering_coordinates, parameters_clustering_coordinates,
    method_clustering_conformations, parameters_clustering_conformations, cluster_of_coordinates_to_process,
    split_trajectory, cutoff_proba_conformations, cutoff_len_states,
    coordinates_to_add, type_coordinates_to_add,residues_coordinates_to_add):
    """
    Prints the input parameters to the log file.

    Parameters:
    - output_dir (str): The directory where the log file is located.
    - step_to_perform (str): The step being performed, used for logging context.
    - strucfile (str): Path to the position file (e.g., .psf, .gro, or .pdb).
    - trajfile (str): Path to the trajectory file (e.g., .dcd, .xtc).
    - dic (str): Path to the dictionary file containing important atoms.
    - time_zero (float): Minimum time threshold for filtering frames.
    - delta_time (float): Time step interval for selecting frames.
    - size_block (float): Size of each time block for analysis.
    - cutoff_distance (float): Cutoff distance for contacts.
    - delta_resid (int): Delta residue for contact calculations.
    - mode_proba_cutoff (float): Probability cutoff for contacts.
    - method_clustering_coordinates (str): Clustering method for coordinates.
    - parameters_clustering_coordinates (list): Parameters for the clustering method.
    - method_clustering_conformations (str): Clustering method for conformations.
    - parameters_clustering_conformations (list): Parameters for the clustering method.
    - cluster_of_coordinates_to_process (int): Index of the cluster of coordinates to process.
    - split_trajectory (bool): Whether to split the trajectory into blocks.
    - cutoff_proba_conformations (float): Probability cutoff for conformation extraction.
    - cutoff_len_states (int): Cutoff for the number of states to consider in clustering states.
    - coordinates_to_add (list): List of additional coordinate files to include.
    - type_coordinates_to_add (list): List of types for the additional coordinates.
    - residues_coordinates_to_add (list): List of residues to consider for additional coordinates.

    Returns:
    - None
    """
    logging.info("\n\n")
    logging.info("Inputs:")
    logging.info("Step to perform: %s", step_to_perform)
    logging.info("Output directory: %s", output_dir)
    logging.info("Position file: %s", strucfile)
    logging.info("Trajectory file: %s", trajfile)
    logging.info("Dictionary file: %s", dic)
    logging.info("Time zero: %.2f ps", time_zero)
    logging.info("Delta time: %.2f ps", delta_time)
    logging.info("Size block: %.2f ps", size_block)
    logging.info("Cutoff distance: %.2f Angstroms", cutoff_distance)
    logging.info("Delta residue: %d", delta_resid)
    logging.info("Probability cutoff: %.5f", mode_proba_cutoff)
    logging.info("method clustering coordinates: %s", method_clustering_coordinates)
    logging.info("Parameters clustering coordinates: %s", parameters_clustering_coordinates)
    logging.info("method clustering conformations: %s", method_clustering_conformations)
    logging.info("Parameters clustering conformations: %s", parameters_clustering_conformations)
    logging.info("Cluster of coordinates to process: %d", cluster_of_coordinates_to_process)
    logging.info("Split trajectory: %s", split_trajectory)
    logging.info("Cutoff probability for conformations: %.5f", cutoff_proba_conformations)
    logging.info("Cutoff number of states: %d", cutoff_len_states)
    logging.info("Additional coordinates to add: %s", coordinates_to_add)
    logging.info("Types of additional coordinates: %s", type_coordinates_to_add)
    logging.info("Residues for additional coordinates: %s", residues_coordinates_to_add)
    

####################### PRINT ENDING MESSAGE #####################
def print_ending_message(output_dir, step_to_perform):
    """
    Prints a message indicating the completion of the analysis.

    Parameters:
    - output_dir (str): The directory where the results are saved.
    - step_to_perform (str): The step that was performed, used for logging context.

    Returns:
    - None
    """
    now= datetime.now()
    logging.info("\n\n")
    logging.info("Analysis complete at %s", now.strftime("%Y-%m-%d %H:%M:%S"))
    logging.info("Results are saved in: %s", output_dir)
    logging.info("Step performed: %s", step_to_perform)

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
        #text = f"\rProgress: [{'#' * block + '-' * (bar_length - block)}] {progress * 100:.0f}%"

        now = datetime.now()
        text = f"Progress: [{'#' * block + '-' * (bar_length - block)}] {progress * 100:.0f}%  {now.strftime('%Y-%m-%d %H:%M:%S')}"
        logging.info(text) #, end='')
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

def get_multiplicities(discretized_array):
    # Get the shape of the input array: number of rows (frames) and columns (coordinates/features)
    nframes, ncoord = np.shape(discretized_array)
    
    # Initialize an array to hold the multiplicity (number of unique values) for each column
    multiplicities = np.zeros((ncoord), dtype=np.int32)
    
    # Loop over each column (coordinate/feature)
    for i in range(ncoord):
        # Count the number of unique values in column i and store it in the multiplicities array
        multiplicities[i] = len(np.unique(discretized_array[:, i]))
    
    # Return the array of multiplicities
    return multiplicities


##################### OPENING TRAJECTORY #####################
def open_trajectory(strucfile, trajfile):
    """
    Opens a molecular dynamics trajectory using MDAnalysis.

    Parameters:
    - strucfile (str): The topology file (e.g., .psf, .gro, or .pdb) that describes the molecular structure.
    - trajfile (str): The trajectory file (e.g., .dcd, .xtc) containing atomic coordinates over time.

    Returns:
    - u_traj (MDAnalysis.Universe): An MDAnalysis Universe object representing the system and trajectory,
      which can be used for further analysis (e.g., atom selections, RMSD calculations, etc.).
    """
    u_traj = mda.Universe(strucfile, trajfile)
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
    logging.info("\nFiltering times and indices...")
    times = []
    times_indices = []
    previous_progress = -1
    delta_t_traj=u_traj.trajectory.dt
    always_keep=False
    if delta_time < delta_t_traj:
        always_keep=True
    logging.info(f"Trajectory time step: {delta_t_traj} ps")
    for ts in u_traj.trajectory:
        # Update progress bar
        previous_progress = plot_progress_bar(ts.frame, len(u_traj.trajectory), previous_progress)
        time_ts= ts.time
        if always_keep or time_ts% delta_time < delta_t_traj/2:
            times.append(time_ts)
            times_indices.append(ts.frame)
            continue


    # Complete progress bar
    plot_progress_bar(len(u_traj.trajectory), len(u_traj.trajectory), previous_progress)

    # Convert to NumPy arrays and save
    times = np.array(times)
    times_indices = np.array(times_indices)
    np.save(output_dir + 'discretizing_npy/times.npy', times)
    np.save(output_dir + 'discretizing_npy/times_indices.npy', times_indices)

    logging.info("Times and indices filtered.")
    return times, times_indices


####################### GET IMPORTANT ATOMS #######################
def read_dictionary(dic):
    """
    Reads a dictionary file containing definitions of important atoms.

    Parameters:
    - dic (str): Path to the dictionary text file.

    Returns:
    - important_atoms_dic (dict): A dictionary where each key is a residue name (e.g., amino acid),
      and each value is a list of names for important atoms in that residue.
    - amino_acids (list): A list of residue names marked as amino acids using the '@amino_acid' tag.

    Notes:
    - Lines ending with '@amino_acid' are treated specially to distinguish amino acids from other residues.
    - Lines with insufficient data (length <= 1) are skipped and reported.
    """
    important_atoms_dic = {}
    amino_acids = []
    nucleic_acids_pyrimidine = []  
    nucleic_acids_purine = []
    with open(dic, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            if len(line) > 1:
                if line[-1] == "@amino_acid":
                    important_atoms_dic[line[0]] = line[1:-1]
                    amino_acids.append(line[0])
                elif line[-1] == "@nucleic_acid_pyrimidine":
                    important_atoms_dic[line[0]] = line[1:-1]
                    nucleic_acids_pyrimidine.append(line[0])
                elif line[-1] == "@nucleic_acid_purine":
                    important_atoms_dic[line[0]] = line[1:-1]
                    nucleic_acids_purine.append(line[0])
                else:
                    important_atoms_dic[line[0]] = line[1:]
            else:
                logging.info(f"Skipping line: {line}")
    return important_atoms_dic, amino_acids, nucleic_acids_pyrimidine, nucleic_acids_purine

def get_important_atoms_MDA(u_traj, important_atoms_dic,step_to_perform):
    """
    Extracts definitions of important atoms from an MDAnalysis Universe.

    Parameters:
    - u_traj (MDAnalysis.Universe): The trajectory object containing residues and atoms.
    - important_atoms_dic (str): Path to the important atom dictionary file.

    Returns:
    - important_atoms (list): List of important atom names (per residue).
    - selected_resids (list): List of residue IDs corresponding to the important atoms.
    - selected_resnames (list): List of residue names corresponding to the important atoms.
    - indices_aa (list): List of residue IDs that are amino acids (based on the dictionary).
    
    Notes:
    - Residues not found in the dictionary are reported once.
    """
    logging.info("\nGetting important atoms...")
    atoms_dic, amino_acids, nucleic_acids_pyrimidine, nucleic_acids_purine = read_dictionary(important_atoms_dic)
    important_atoms = []
    selected_resids = []
    selected_resnames = []
    indices_aa = []
    indices_na_pyrimidine = []
    indices_na_purine = []

    res_not_found = []

    for residue in u_traj.residues:
        resname = residue.resname
        resid = residue.resid
        if resname in atoms_dic:
            important_atoms.append(atoms_dic[resname])
            selected_resids.append(resid)
            selected_resnames.append(resname)
            if resname in amino_acids:
                indices_aa.append(resid)
            elif resname in nucleic_acids_pyrimidine:
                indices_na_pyrimidine.append(resid)
            elif resname in nucleic_acids_purine:
                indices_na_purine.append(resid)
        elif resname not in res_not_found:
            if step_to_perform == 'all':
                logging.info(f"Residue {resname} not found in {important_atoms_dic}. Skipping it.")
            res_not_found.append(resname)
    if step_to_perform == 'all':
        logging.info("\nSelected residues:")
        for resid, resname in zip(selected_resids, selected_resnames):
            if resid not in indices_aa:
                logging.info(f" {resname} - {resid} ")
            else:
                logging.info(f" {resname} - {resid} (AA) ")

    return important_atoms, selected_resids, selected_resnames, indices_aa, indices_na_pyrimidine, indices_na_purine

def save_important_atoms(important_atoms, selected_resids, selected_resnames, indices_aa, indices_na_pyrimidine, indices_na_purine,output_dir):
    """
    Saves important atoms information to a text file.

    Parameters:
    - important_atoms (list): List of important atom names per residue.
    - selected_resids (list): List of corresponding residue IDs.
    - selected_resnames (list): List of corresponding residue names.
    - indices_aa (list): List of residue IDs that are amino acids.
    - indices_na_pyrimidine (list): List of residue IDs for pyrimidine nucleic acids.
    - indices_na_purine (list): List of residue IDs for purine nucleic acids.
    - output_dir (str): Directory path where the output file will be saved.

    Output:
    - A text file named 'important_atoms.txt' containing:
      <resid>   <resname>   <atom_names> <tag>
    """
    logging.info("\nSaving important atoms to file...")
    with open(output_dir + 'important_atoms.txt', 'w') as f:
        for k in range(len(important_atoms)):
            atom = important_atoms[k]
            resid = selected_resids[k]
            type_res = selected_resnames[k]
            tag = ''
            if resid in indices_aa:
                tag = 'AA'
            elif resid in indices_na_pyrimidine:
                tag = 'NA_pyrimidine'
            elif resid in indices_na_purine:
                tag = 'NA_purine'
            f.write(f'{resid}   {type_res}   {atom}   {tag}\n')
        f.close()
    logging.info("Important atoms saved to file.")


############################## PRECOMPUTE POSITIONS OF ATOMS ##################
def precompute_terminals(u_traj, important_atoms, selected_resids, times_indices):
    """
    Precomputes the 3D positions of important atoms 
    for a selected set of residues across specified frames in a trajectory.

    Parameters:
    - u_traj (MDAnalysis.Universe): The MDAnalysis universe object containing the trajectory.
    - important_atoms (list of lists): A list of important atom names for each selected residue.
    - selected_resids (list): Residue IDs corresponding to the residues with important atoms.
    - times_indices (np.ndarray): Indices of the frames in the trajectory to process.

    Returns:
    - positions_important_atoms (np.ndarray): A NumPy array of shape (num_atoms, num_frames, 3),
      storing the x, y, z coordinates of each important atom across selected frames.

    Behavior:
    - Preselects important atoms for all residues to avoid repetitive selection in each frame.
    - Iterates over selected trajectory frames and stores the positions of each important atom.
    - Displays a progress bar during processing.
    """
    logging.info("\nPrecomputing positions of important atoms...")

    num_residues = len(selected_resids)  # Total number of residues with important atoms
    num_atoms = np.sum([len(important_atoms[i]) for i in range(num_residues)])  # Total number of important atoms

    # Pre-select atom groups for each important atom in each residue to avoid repeated selections
    terminal_atom_selections = []
    for i in range(num_residues):
        terminal_atom_selections.append([
            u_traj.select_atoms(f"resnum {selected_resids[i]} and name {important_atoms[i][j]}")
            for j in range(len(important_atoms[i]))
        ])

    # Initialize array to store important atom positions:
    # Shape: (total important atoms, number of selected frames, 3 coordinates)
    positions_important_atoms = np.zeros((num_atoms, len(times_indices), 3))

    # Iterate through selected frames and record positions
    previous_progress = -1
    for k, frame in enumerate(times_indices):
        u_traj.trajectory[frame]  # Move to the specific frame
        previous_progress = plot_progress_bar(k, len(times_indices), previous_progress)
        count_step = 0  # Index for placing atoms in the output array
        for i in range(num_residues):
            for j in range(len(important_atoms[i])):
                positions_important_atoms[count_step, k, :] = terminal_atom_selections[i][j].positions
                count_step += 1

    # Complete the progress bar
    plot_progress_bar(len(times_indices), len(times_indices), previous_progress)
    logging.info("Positions of important atoms precomputed.")

    return positions_important_atoms

def precompute_backbone_protein(u_traj, indices_aa, times_indices):
    """
    Precomputes the 3D positions of backbone atoms (C, N, and CA) for each selected residue
    across specified trajectory frames.

    Parameters:
    - u_traj (MDAnalysis.Universe): The MDAnalysis universe object containing the trajectory.
    - indices_aa (list): List of residue IDs for which backbone atoms are to be tracked.
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
    logging.info("\nPrecomputing positions of protein backbone atoms...")

    num_residues = len(indices_aa)

    # Preselect atom groups for each backbone atom type
    atom_C_selections = [
        u_traj.select_atoms(f"resnum {indices_aa[i]} and name C")
        for i in range(num_residues)
    ]

    atom_N_selections = [
        u_traj.select_atoms(f"resnum {indices_aa[i]} and name N")
        for i in range(num_residues)
    ]

    atom_CA_selections = [
        u_traj.select_atoms(f"resnum {indices_aa[i]} and name CA")
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
    logging.info("Positions of protein backbone atoms precomputed.")

    return Positions_atoms_C, Positions_atoms_N, Positions_atoms_CA

def precompute_backbone_nucleic_acids(u_traj, indices_na_pyrimidine, indices_na_purine, times_indices):
    """
    Precomputes the 3D positions of backbone atoms for nucleic acids (DNA/RNA) for each selected residue
    across specified trajectory frames.
    Parameters:
    - u_traj (MDAnalysis.Universe): The MDAnalysis universe object containing the trajectory.
    - selected_resids (list): List of residue IDs for which backbone atoms are to be tracked.
    - times_indices (np.ndarray): Indices of the trajectory frames to be processed.
    Returns:
    - Positions_atoms_P (np.ndarray): Array of shape (num_residues, num_frames, 3)
      with the 3D positions of the phosphorus (P) atoms. Warning the first residue has no P atom so the position is (0,0,0).
    - Positions_atoms_O5p (np.ndarray): Array of shape (num_residues, num_frames, 3)
        with the 3D positions of the oxygen-5' (O5') atoms.
    - Positions_atoms_C5p (np.ndarray): Array of shape (num_residues, num_frames, 3)
        with the 3D positions of the carbon-5' (C5') atoms.
    - Positions_atoms_C4p (np.ndarray): Array of shape (num_residues, num_frames, 3)
        with the 3D positions of the carbon-4' (C4') atoms.
    - Positions_atoms_C3p (np.ndarray): Array of shape (num_residues, num_frames, 3)
        with the 3D positions of the carbon-3' (C3') atoms.
    - Positions_atoms_O3p (np.ndarray): Array of shape (num_residues, num_frames, 3)
        with the 3D positions of the oxygen-3' (O3') atoms.
    - Positions_atoms_C1p (np.ndarray): Array of shape (num_residues, num_frames, 3)
        with the 3D positions of the carbon-1' (C1') atoms.
    - Positions_atoms_C2 (np.ndarray): Array of shape (num_residues, num_frames, 3)
        with the 3D positions of the carbon-2 (C2) atoms.
    - Positions_atoms_C4 (np.ndarray): Array of shape (num_residues, num_frames, 3)
        with the 3D positions of the carbon-4 (C4) atoms.

    """
    logging.info("\nPrecomputing positions of nucleic acids backbone atoms...")

    indices_na= np.sort(indices_na_pyrimidine+indices_na_purine)  # Combine and sort indices of nucleic acids

    num_residues = len(indices_na)

    # Preselect atom groups for each backbone atom type
    atom_P_selections = [
        u_traj.select_atoms(f"resnum {indices_na[i]} and name P")
        for i in range(num_residues)
    ]

    atom_O5p_selections = [
        u_traj.select_atoms(f"resnum {indices_na[i]} and name O5'")
        for i in range(num_residues)
    ]

    atom_C5p_selections = [
        u_traj.select_atoms(f"resnum {indices_na[i]} and name C5'")
        for i in range(num_residues)
    ]

    atom_O4p_selections = [
        u_traj.select_atoms(f"resnum {indices_na[i]} and name O4'")
        for i in range(num_residues)
    ]

    atom_C4p_selections = [
        u_traj.select_atoms(f"resnum {indices_na[i]} and name C4'")
        for i in range(num_residues)
    ]

    atom_C3p_selections = [
        u_traj.select_atoms(f"resnum {indices_na[i]} and name C3'")
        for i in range(num_residues)
    ]

    atom_O3p_selections = [
        u_traj.select_atoms(f"resnum {indices_na[i]} and name O3'")
        for i in range(num_residues)
    ]

    atom_C1p_selections = [
        u_traj.select_atoms(f"resnum {indices_na[i]} and name C1'")
        for i in range(num_residues)
    ]

    atom_Nbs_selections = []
    for i in range(num_residues):
        if indices_na[i] in indices_na_pyrimidine:
            atom_Nbs_selections.append(u_traj.select_atoms(f"resnum {indices_na[i]} and name N1"))
        elif indices_na[i] in indices_na_purine:
            atom_Nbs_selections.append(u_traj.select_atoms(f"resnum {indices_na[i]} and name N9"))

    atom_Cbs_selections = []
    for i in range(num_residues):
        if indices_na[i] in indices_na_pyrimidine:
            atom_Cbs_selections.append(u_traj.select_atoms(f"resnum {indices_na[i]} and name C2"))
        elif indices_na[i] in indices_na_purine:
            atom_Cbs_selections.append(u_traj.select_atoms(f"resnum {indices_na[i]} and name C4"))


    # Initialize arrays to store positions of backbone atoms over time
    Positions_atoms_P = np.empty((num_residues, len(times_indices), 3))
    Positions_atoms_P.fill(np.inf)  # Fill with NaN to handle residues without P atom (e.g., first residue)
    Positions_atoms_O5p = np.zeros((num_residues, len(times_indices), 3))
    Positions_atoms_C5p = np.zeros((num_residues, len(times_indices), 3))
    Positions_atoms_O4p = np.zeros((num_residues, len(times_indices), 3))
    Positions_atoms_C4p = np.zeros((num_residues, len(times_indices), 3))
    Positions_atoms_C3p = np.zeros((num_residues, len(times_indices), 3))
    Positions_atoms_O3p = np.zeros((num_residues, len(times_indices), 3))
    Positions_atoms_C1p = np.zeros((num_residues, len(times_indices), 3))
    Positions_atoms_Nbs = np.zeros((num_residues, len(times_indices), 3))
    Positions_atoms_Cbs = np.zeros((num_residues, len(times_indices), 3))





    # Iterate through selected frames and record positions
    previous_progress = -1
    for k, frame in enumerate(times_indices):
        previous_progress = plot_progress_bar(k, len(times_indices), previous_progress)
        u_traj.trajectory[frame]  # Set trajectory to the specific frame

        for i in range(num_residues):
            if len(atom_P_selections[i].positions) != 0:
                Positions_atoms_P[i, k, :] = atom_P_selections[i].positions
            Positions_atoms_O5p[i, k, :] = atom_O5p_selections[i].positions
            Positions_atoms_C5p[i, k, :] = atom_C5p_selections[i].positions
            Positions_atoms_O4p[i, k, :] = atom_O4p_selections[i].positions
            Positions_atoms_C4p[i, k, :] = atom_C4p_selections[i].positions
            Positions_atoms_C3p[i, k, :] = atom_C3p_selections[i].positions
            Positions_atoms_O3p[i, k, :] = atom_O3p_selections[i].positions
            Positions_atoms_C1p[i, k, :] = atom_C1p_selections[i].positions
            Positions_atoms_Nbs[i, k, :] = atom_Nbs_selections[i].positions
            Positions_atoms_Cbs[i, k, :] = atom_Cbs_selections[i].positions

    # Complete progress bar
    plot_progress_bar(len(times_indices), len(times_indices), previous_progress)
    logging.info("Positions of nucleic acids backbone atoms precomputed.")

    return Positions_atoms_P, Positions_atoms_O5p, Positions_atoms_C5p, Positions_atoms_O4p, Positions_atoms_C4p, Positions_atoms_C3p, Positions_atoms_O3p, Positions_atoms_C1p, Positions_atoms_Nbs, Positions_atoms_Cbs


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
    num_blocks = max(1,int((times[-1] - time_zero_ps) / size_block_ps))
    validated_size_block_ps = int((times[-1] - time_zero_ps) / num_blocks)

    y_max = max(data)
    y_min = min(data)

    # Compute histograms
    HIST_TOT, x, AVG, STD = compute_hist_tot(times, data, num_blocks, y_min, y_max, delta_y,
                                             time_zero_ps, validated_size_block_ps)

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

def filter_significant_minima(x_smooth, y_smooth, minima, mode_proba_cutoff):
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
    - mode_proba_cutoff: Minimum probability threshold required to consider the region between minima as significant.

    Returns:
    - selected_minima: List of filtered minima that define significant regions.
    """
    selected_minima = []
    previous = x_smooth[0]  # Start from the leftmost boundary of the distribution

    for next_minimum in minima:
        # Define the region between the current and next minimum
        mask = (x_smooth >= previous) & (x_smooth <= next_minimum)
        area = np.trapz(y_smooth[mask], x_smooth[mask])  # Integrated probability

        if area < mode_proba_cutoff:
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
    if final_area < mode_proba_cutoff and previous in selected_minima:
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

def plot_histogram(x, AVG, error_bars, x_smooth, y_smooth, xlabel, coordinate_name, minima, output_dir):
    """
    Plots the histogram of coordinate data with error bars, KDE curve, and vertical lines at selected minima.

    Parameters:
    - x: Array of bin centers for the histogram.
    - AVG: Average histogram values (probability density).
    - error_bars: Error estimates for each bin (e.g., standard error).
    - y_smooth: Smoothed density estimation from Kernel Density Estimation (KDE).
    - x_smooth: x-values corresponding to the KDE curve.
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
                          coordinate_name,mode_proba_cutoff, output, output_dir):
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
    - mode_proba_cutoff: Minimum probability threshold required to consider the region between minima as significant.
    - output (str or file-like): Path or handle to save minima/label information.
    - output_dir (str): Directory to save coordinate data and plots.

    Returns:
    - None. Results are saved to disk.
    """

    # Step 1: Smooth the coordinate distribution using KDE
    x_smooth, y_smooth = smooth_coordinate(y, delta_y)
    if len(x_smooth)<2:
        return

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
        x_smooth, y_smooth, minima, mode_proba_cutoff
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
        x_smooth, y_smooth, xlabel,
        coordinate_name, selected_minima,
        output_dir
    )


############################### Compute minimum distance between important atoms ##################
def compute_min_distances(positions_important_atoms, i, j, important_atoms):
    """
    Computes the minimum distance between important atoms 
    of two residues across all frames in the trajectory.

    Parameters:
    - positions_important_atoms (np.ndarray): A 3D array of shape 
      (total_important_atoms, num_frames, 3), storing positions of important atoms.
    - i (int): Index of the first residue in selected_resids and important_atoms.
    - j (int): Index of the second residue in selected_resids and important_atoms.
    - important_atoms (list of lists): A list where each entry contains the names 
      of important atoms for the corresponding residue.


    Returns:
    - min_absolute_distance (float): The minimum distance between any pair of terminal 
      atoms from residues i and j, over all frames.
    - distance_to_save (np.ndarray): A 1D array containing the distance values over time 
      for the pair of atoms that gave the minimum distance.
    - atom_i_to_save (str): Name of the atom in residue i involved in the minimum distance.
    - atom_j_to_save (str): Name of the atom in residue j involved in the minimum distance.

    Description:
    - Computes pairwise distances between all important atoms of residue i and residue j.
    - For each atom pair, the Euclidean distance is computed across all frames.
    - From all possible atom pair combinations, the one with the smallest minimum 
      distance (across all time steps) is selected and returned.
    """

    # Get number of important atoms for each residue
    num_term_i = len(important_atoms[i])
    num_term_j = len(important_atoms[j])

    # Calculate starting indices of important atoms for residues i and j
    ind_term_0_i = sum([len(important_atoms[k]) for k in range(i)])
    ind_term_0_j = sum([len(important_atoms[k]) for k in range(j)])

    # Extract positions for all important atoms of residue i and j
    Positions_i = [positions_important_atoms[ind_term_0_i + k, :, :] for k in range(num_term_i)]
    Positions_j = [positions_important_atoms[ind_term_0_j + k, :, :] for k in range(num_term_j)]

    # Copy the important atom names
    atoms_i = important_atoms[i].copy()
    atoms_j = important_atoms[j].copy()

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
def process_distance_pair(i, j, positions_important_atoms, important_atoms, selected_resids, times, time_zero, size_block, cutoff_distance,proba_under_cutoff_distance,mode_proba_cutoff,output,output_dir):
    """
    Processes a pair of selected residues by computing the minimal interatomic distance
    between their important atoms, and discretizes the distance time series if the pair 
    is close enough (below cutoff).

    Parameters:
    -----------
    - i, j (int) :  Indices of the residues in selected_resids to process.
    - positions_important_atoms (np.ndarray): 3D array of precomputed positions of important atoms.
      Shape: (total_important_atoms, num_frames, 3).
    - important_atoms (list of lists): Names of important atoms for each residue.
    - selected_resids (list): List of selected residue IDs.
    - times (np.ndarray): Array of times corresponding to selected frames.
    - time_zero (float): Reference time used for distance analysis.
    - size_block (int): Size of blocks used in post-processing (likely temporal).
    - cutoff_distance (float): Distance threshold for further analysis.
    - mode_proba_cutoff (float): Minimum probability threshold for discretization.
    - output (file-like or handle): Destination for saving results.
    - output_dir (str): Directory where output files will be written.
    """
    
    # Compute the minimal interatomic distance between important atoms of residues i and j
    min_absolute_distance, distance_to_save, atom_i_to_save, atom_j_to_save = compute_min_distances(
        positions_important_atoms, i, j, important_atoms
    )

    # Skip pairs that are too far apart
    if min_absolute_distance > cutoff_distance:
        return
    proba_under = np.mean(distance_to_save < cutoff_distance)
    if proba_under < proba_under_cutoff_distance :
        return

    # Prepare coordinate for discretization
    y = distance_to_save               # Distance time series
    delta_y = 0.1                      # Bin width for discretization
    coordinate_type = 'distance'      # Type of coordinate being discretized

    # Construct a unique name for this distance coordinate
    coordinate_name = f"{selected_resids[i]}_{atom_i_to_save}_{selected_resids[j]}_{atom_j_to_save}"
    
    # Discretize the distance time series and update output data structures
    discretize_coordinate(
        y, delta_y, coordinate_type, times, time_zero, size_block,
        coordinate_name, mode_proba_cutoff, output, output_dir
    )


####################### Function to compute distances between important atoms for all residue pairs ##########################
def compute_all_distances(important_atoms,selected_resids,positions_important_atoms,times,time_zero,size_block,delta_resid,cutoff_distance,proba_under_cutoff_distance,mode_proba_cutoff,output,output_dir):
    """
    Computes pairwise distances between all valid residue pairs based on their important atoms,
    and processes each pair using a custom distance analysis function.

    Parameters:
    - important_atoms (list of lists): Names of important atoms for each residue.
    - selected_resids (list): List of selected residue IDs.
    - positions_important_atoms (np.ndarray): 3D array of precomputed positions of important atoms.
      Shape: (total_important_atoms, num_frames, 3).
    - times (np.ndarray): Array of times corresponding to selected frames.
    - time_zero (float): Reference time used for distance analysis.
    - size_block (int): Size of blocks used in post-processing (likely temporal).
    - delta_resid (int): Minimum residue index separation; avoids comparing too-close residues.
    - cutoff_distance (float): Distance threshold for further analysis.
    - mode_proba_cutoff (float): Minimum probability threshold for discretization.
    - output (file-like or handle): Destination for saving results.
    - output_dir (str): Directory where output files will be written.

    Returns:
    - None (results are saved to files via `process_distance_pair`)

    Behavior:
    - Iterates over all valid residue index pairs `(i, j)` where `j >= i + delta_resid`.
    - For each pair, calls `process_distance_pair` to compute and process distances.
    - A progress bar is shown during processing.
    """

    num_residues = len(selected_resids)
    total_combinations = num_residues * (num_residues - delta_resid) / 2  # total number of pairs
    count_step = 0

    logging.info("\nComputing distances...")

    # Iterate over all valid residue pairs
    previous_progress = -1  # Initialize progress bar
    for i in range(num_residues - delta_resid):
        for j in range(i + delta_resid, num_residues):
            # Update progress bar
            previous_progress=plot_progress_bar(count_step, total_combinations, previous_progress)
            count_step += 1

            # Process this residue pair
            process_distance_pair(
                i, j,positions_important_atoms,important_atoms,selected_resids,times,time_zero,size_block,cutoff_distance,proba_under_cutoff_distance,mode_proba_cutoff,output,output_dir
            )

    # Finalize progress bar
    plot_progress_bar(total_combinations, total_combinations,previous_progress)
    logging.info("Distances computed and saved.")


########################## Function to get the multimodal contacts ################################
def get_contacts(u_traj, important_atoms, selected_resids, time_zero, size_block, cutoff_distance,proba_under_cutoff_distance, delta_resid, mode_proba_cutoff, output_dir):
    """
    Main function to compute and process distances (contacts) between important atoms
    throughout a molecular dynamics trajectory.

    This function performs the following steps:
    1. Loads pre-filtered time points and frame indices from .npy files.
    2. Precomputes the 3D positions of all important atoms across the selected frames.
    3. Saves the computed positions to a file for later use.
    4. Computes distances between all valid residue pairs and processes the results.

    Parameters:
    - u_traj: MDAnalysis Universe object containing the trajectory.
    - important_atoms: List of important atom names for each selected residue.
    - selected_resids: List of residue IDs to analyze.
    - time_zero: Time reference for analysis start.
    - size_block: Block size used for distance analysis (e.g., time window).
    - cutoff_distance: Maximum distance threshold to consider a contact.
    - delta_resid: Minimum residue index separation to avoid local contacts.
    - mode_proba_cutoff: Minimum probability threshold for discretization.
    - output_dir: Directory path to read inputs and save outputs.

    Returns:
    - None. Results are saved to files.
    """

    # Load time points and frame indices previously filtered and saved
    times = np.load(output_dir + 'discretizing_npy/times.npy')
    times_indices = np.load(output_dir + 'discretizing_npy/times_indices.npy')

    # Precompute important atom positions across trajectory
    positions_important_atoms = precompute_terminals(u_traj, important_atoms, selected_resids, times_indices)

    # Save precomputed positions to disk
    save_positions(positions_important_atoms, output_dir + "discretizing_npy/positions_important_atoms.npy")

    # Compute and process distances between all valid residue pairs
    compute_all_distances(
        important_atoms, selected_resids, positions_important_atoms,
        times, time_zero, size_block, delta_resid,
        cutoff_distance,proba_under_cutoff_distance, mode_proba_cutoff,
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

def process_dihedral_i_protein(i, Positions_atoms_C, Positions_atoms_N, Positions_atoms_CA, indices_aa, 
                       times, time_zero, size_block,mode_proba_cutoff, output, output_dir):
    """
    Processes the i-th residue to compute phi and psi dihedral angles, adjust for angle wrapping,
    and discretize the angle distributions for further analysis.

    Parameters:
    - i: Index of the residue to process.
    - Positions_atoms_C, Positions_atoms_N, Positions_atoms_CA: 3D arrays of atomic positions
      with shape (num_residues, num_timepoints, 3).
    - indices_aa: List or array of residue identifiers.
    - times: 1D array of time points corresponding to frames.
    - time_zero: Start time for analysis.
    - size_block: Block size for time-averaging during discretization.
    - mode_proba_cutoff: Minimum probability threshold required to consider the region between minima as significant.
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
        if distance_C_N < 2:
            coordinate_name = f"phi_{indices_aa[i]}"
            # Calculate phi dihedral angles (radians) and convert to degrees
            phi_angle = np.rad2deg(mda.lib.distances.calc_dihedrals(Positions_atoms_C[i - 1, :, :],Positions_atoms_N[i, :, :],Positions_atoms_CA[i, :, :],Positions_atoms_C[i, :, :])            )
            # Adjust angles if range spans more than 180 degrees (unwrap circular data)
            if np.ptp(phi_angle) > 180:
                phi_angle, _, _ = adjust_angle_data(phi_angle, np.min(phi_angle), np.max(phi_angle), delta_y)
            
            # Discretize the phi angle data for further analysis
            discretize_coordinate(phi_angle, delta_y, coordinate_type,
                                  times, time_zero, size_block,
                                  coordinate_name,mode_proba_cutoff, output, output_dir)

    # Process psi dihedral if next residue exists and backbone geometry is valid
    if i < len(Positions_atoms_C) - 1:
        distance_N_C = np.linalg.norm(Positions_atoms_N[i + 1, 0, :] - Positions_atoms_C[i, 0, :])
        if distance_N_C < 2:
            coordinate_name = f"psi_{indices_aa[i]}"
            # Calculate psi dihedral angles (radians) and convert to degrees
            psi_angle = np.rad2deg(mda.lib.distances.calc_dihedrals(Positions_atoms_N[i, :, :],Positions_atoms_CA[i, :, :],Positions_atoms_C[i, :, :],Positions_atoms_N[i + 1, :, :]))
            # Adjust angles if range spans more than 180 degrees (unwrap circular data)
            if np.ptp(psi_angle) > 180:
                psi_angle, _, _ = adjust_angle_data(psi_angle, np.min(psi_angle), np.max(psi_angle), delta_y)
            
            # Discretize the psi angle data for further analysis
            discretize_coordinate(psi_angle, delta_y, coordinate_type,
                                  times, time_zero, size_block,
                                  coordinate_name,mode_proba_cutoff, output, output_dir)
            
def process_dihedral_i_nucleic_acids(i, Positions_atoms_P, Positions_atoms_O5p, Positions_atoms_C5p, Positions_atoms_O4p, Positions_atoms_C4p, Positions_atoms_C3p,
                                     Positions_atoms_O3p, Positions_atoms_C1p, Positions_atoms_Nbs, Positions_atoms_Cbs, indices_na, 
                                    times, time_zero, size_block,mode_proba_cutoff, output, output_dir):
    """
    Processes the i-th residue to compute phi and psi dihedral angles, adjust for angle wrapping,
    and discretize the angle distributions for further analysis.

    Parameters:
    - i: Index of the residue to process.
    - Positions_atoms_P, Positions_atoms_O5p, Positions_atoms_C5p, Positions_atoms_C4p,
      Positions_atoms_C3p, Positions_atoms_O3p, Positions_atoms_C1p, Positions_atoms_Cbs: 3D arrays of atomic positions
      with shape (num_residues, num_timepoints, 3).
    - indices_na: List or array of residue identifiers.
    - times: 1D array of time points corresponding to frames.
    - time_zero: Start time for analysis.
    - size_block: Block size for time-averaging during discretization.
    - mode_proba_cutoff: Minimum probability threshold required to consider the region between minima as significant.
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
    alpha_angle = np.zeros(len(times))
    beta_angle = np.zeros(len(times))
    gamma_angle = np.zeros(len(times))
    delta_angle = np.zeros(len(times))
    epsilon_angle = np.zeros(len(times))
    zeta_angle = np.zeros(len(times))
    chi_angle = np.zeros(len(times))

    # Process phi dihedral if previous residue exists and backbone geometry is valid
    if i > 1 and np.any(np.isinf(Positions_atoms_P[i, :, :]))==False :
        distance_O_P = np.linalg.norm(Positions_atoms_O3p[i - 1, 0, :] - Positions_atoms_P[i, 0, :])
        if distance_O_P < 2:
            coordinate_name = f"alpha_{indices_na[i]}"
            # Calculate alpha dihedral angles (radians) and convert to degrees
            alpha_angle = np.rad2deg(mda.lib.distances.calc_dihedrals(Positions_atoms_O3p[i - 1, :, :],Positions_atoms_P[i, :, :],Positions_atoms_O5p[i, :, :],Positions_atoms_C5p[i, :, :]) )
            # Adjust angles if range spans more than 180 degrees (unwrap circular data)
            if np.ptp(alpha_angle) > 180:
                alpha_angle, _, _ = adjust_angle_data(alpha_angle, np.min(alpha_angle), np.max(alpha_angle), delta_y)
            
            # Discretize the alpha angle data for further analysis
            discretize_coordinate(alpha_angle, delta_y, coordinate_type,
                                  times, time_zero, size_block,
                                  coordinate_name,mode_proba_cutoff, output, output_dir)
            
        coordinate_name = f"beta_{indices_na[i]}"
        # Calculate beta dihedral angles (radians) and convert to degrees
        beta_angle = np.rad2deg(mda.lib.distances.calc_dihedrals(Positions_atoms_P[i, :, :],Positions_atoms_O5p[i, :, :],Positions_atoms_C5p[i, :, :],Positions_atoms_C4p[i, :, :]) )
        # Adjust angles if range spans more than 180 degrees (unwrap circular data)
        if np.ptp(beta_angle) > 180:
            beta_angle, _, _ = adjust_angle_data(beta_angle, np.min(beta_angle), np.max(beta_angle), delta_y)
        
        # Discretize the beta angle data for further analysis
        discretize_coordinate(beta_angle, delta_y, coordinate_type,
                                times, time_zero, size_block,
                                coordinate_name,mode_proba_cutoff, output, output_dir)
    
    coordinate_name = f"gamma_{indices_na[i]}"
    # Calculate gamma dihedral angles (radians) and convert to degrees
    gamma_angle = np.rad2deg(mda.lib.distances.calc_dihedrals(Positions_atoms_O5p[i, :, :],Positions_atoms_C5p[i, :, :],Positions_atoms_C4p[i, :, :],Positions_atoms_C3p[i, :, :]) )
    # Adjust angles if range spans more than 180 degrees (unwrap circular data)
    if np.ptp(gamma_angle) > 180:
        gamma_angle, _, _ = adjust_angle_data(gamma_angle, np.min(gamma_angle), np.max(gamma_angle), delta_y)
    
    # Discretize the gamma angle data for further analysis
    discretize_coordinate(gamma_angle, delta_y, coordinate_type,
                            times, time_zero, size_block,
                            coordinate_name,mode_proba_cutoff, output, output_dir)
    
    coordinate_name = f"delta_{indices_na[i]}"
    # Calculate delta dihedral angles (radians) and convert to degrees
    delta_angle = np.rad2deg(mda.lib.distances.calc_dihedrals(Positions_atoms_C5p[i, :, :],Positions_atoms_C4p[i, :, :],Positions_atoms_C3p[i, :, :],Positions_atoms_O3p[i, :, :]) )
    # Adjust angles if range spans more than 180 degrees (unwrap circular data)
    if np.ptp(delta_angle) > 180:
        delta_angle, _, _ = adjust_angle_data(delta_angle, np.min(delta_angle), np.max(delta_angle), delta_y)
    
    # Discretize the delta angle data for further analysis
    discretize_coordinate(delta_angle, delta_y, coordinate_type,
                            times, time_zero, size_block,
                            coordinate_name,mode_proba_cutoff, output, output_dir)
    
    coordinate_name = f"chi_{indices_na[i]}"
    # Calculate chi dihedral angles (radians) and convert to degrees
    chi_angle = np.rad2deg(mda.lib.distances.calc_dihedrals(Positions_atoms_O4p[i, :, :],Positions_atoms_C1p[i, :, :],Positions_atoms_Nbs[i, :, :],Positions_atoms_Cbs[i, :, :]) )
    # Adjust angles if range spans more than 180 degrees (unwrap circular data)
    if np.ptp(chi_angle) > 180:
        chi_angle, _, _ = adjust_angle_data(chi_angle, np.min(chi_angle), np.max(chi_angle), delta_y)
    
    # Discretize the chi angle data for further analysis
    discretize_coordinate(chi_angle, delta_y, coordinate_type,
                            times, time_zero, size_block,
                            coordinate_name,mode_proba_cutoff, output, output_dir)

    # Process psi dihedral if next residue exists and backbone geometry is valid
    if i < len(Positions_atoms_P) - 1:
        distance_O_P = np.linalg.norm(Positions_atoms_O3p[i, 0, :] - Positions_atoms_P[i+1, 0, :])
        if distance_O_P < 2 and  np.any(np.isinf(Positions_atoms_P[i+1, :, :]))==False:
            coordinate_name = f"epsilon_{indices_na[i]}"
            # Calculate psi dihedral angles (radians) and convert to degrees
            epsilon_angle = np.rad2deg(mda.lib.distances.calc_dihedrals(Positions_atoms_C4p[i, :, :],Positions_atoms_C3p[i, :, :],Positions_atoms_O3p[i, :, :],Positions_atoms_P[i + 1, :, :]))
            # Adjust angles if range spans more than 180 degrees (unwrap circular data)
            if np.ptp(epsilon_angle) > 180:
                epsilon_angle, _, _ = adjust_angle_data(epsilon_angle, np.min(epsilon_angle), np.max(epsilon_angle), delta_y)
            
            # Discretize the psi angle data for further analysis
            discretize_coordinate(epsilon_angle, delta_y, coordinate_type,
                                  times, time_zero, size_block,
                                  coordinate_name,mode_proba_cutoff, output, output_dir)
            
            coordinate_name = f"zeta_{indices_na[i]}"
            # Calculate psi dihedral angles (radians) and convert to degrees
            zeta_angle = np.rad2deg(mda.lib.distances.calc_dihedrals(Positions_atoms_C3p[i, :, :],Positions_atoms_O3p[i, :, :],Positions_atoms_P[i+1, :, :],Positions_atoms_O5p[i + 1, :, :]))
            # Adjust angles if range spans more than 180 degrees (unwrap circular data)
            if np.ptp(zeta_angle) > 180:
                zeta_angle, _, _ = adjust_angle_data(zeta_angle, np.min(zeta_angle), np.max(zeta_angle), delta_y)
            
            # Discretize the psi angle data for further analysis
            discretize_coordinate(zeta_angle, delta_y, coordinate_type,
                                  times, time_zero, size_block,
                                  coordinate_name,mode_proba_cutoff, output, output_dir)
        
    
########################### Functions to compute dihedrals for all residues ##########################
def compute_all_dihedrals_protein(indices_aa, Positions_atoms_C, Positions_atoms_N, Positions_atoms_CA, times, time_zero, size_block, mode_proba_cutoff, output, output_dir):  
    """
    Iterates over all selected residues and computes dihedral angles between them.

    This function processes each selected residue's dihedral angle one by one using
    precomputed backbone atom positions and stores the results for further analysis.

    Parameters:
    - u_traj: MDAnalysis Universe or trajectory object.
    - indices_aa: List of residue indices for which to compute dihedrals.
    - Positions_atoms_C: Precomputed C atom positions for each residue over time.
    - Positions_atoms_N: Precomputed N atom positions for each residue over time.
    - Positions_atoms_CA: Precomputed CA atom positions for each residue over time.
    - times: 1D array of time points (e.g., in ps).
    - time_zero: Time (in ps) to start analysis from.
    - size_block: Block size (in ps) for time-averaging.
    - mode_proba_cutoff -- minimum probability threshold for filtering regions between minima
    - output: Path to output file where selected features/labels are written.
    - output_dir: Directory where output data (e.g., plots or processed values) is stored.

    Returns:
    - None. Outputs are saved directly to disk.
    """

    num_residues = len(indices_aa)

    logging.info("\nComputing dihedrals in protein backbone...")
    previous_progress = -1  # Initialize progress bar
    for i in range(num_residues):
        previous_progress=plot_progress_bar(i, num_residues,previous_progress)
        process_dihedral_i_protein(i, Positions_atoms_C, Positions_atoms_N, Positions_atoms_CA, indices_aa, times, time_zero, size_block, mode_proba_cutoff,output, output_dir)

    plot_progress_bar(num_residues, num_residues,previous_progress)
    logging.info("Dihedrals computed and saved.")

def compute_all_dihedrals_nucleic_acids(indices_na, Positions_atoms_P, Positions_atoms_O5p, Positions_atoms_C5p, Positions_atoms_O4p, Positions_atoms_C4p, Positions_atoms_C3p, Positions_atoms_O3p, Positions_atoms_C1p, Positions_atoms_Nbs, Positions_atoms_Cbs,
                                         times, time_zero, size_block, mode_proba_cutoff, output, output_dir):  
    """
    Iterates over all selected residues and computes dihedral angles between them.

    This function processes each selected residue's dihedral angle one by one using
    precomputed backbone atom positions and stores the results for further analysis.

    Parameters:
    - u_traj: MDAnalysis Universe or trajectory object.
    - indices_na: List of residue indices for which to compute dihedrals.
    - Positions_atoms_P: Precomputed P atom positions for each residue over time.
    - Positions_atoms_O5p: Precomputed O5' atom positions for each residue over time.
    - Positions_atoms_C5p: Precomputed C5' atom positions for each residue over time.
    - Positions_atoms_C4p: Precomputed C4' atom positions for each residue over time.
    - Positions_atoms_C3p: Precomputed C3' atom positions for each residue over time.
    - Positions_atoms_O3p: Precomputed O3' atom positions for each residue over time.
    - Positions_atoms_C1p: Precomputed C1' atom positions for each residue over time.
    - Positions_atoms_Cbs: Precomputed Cb atom positions for each residue over time.
    - times: 1D array of time points (e.g., in ps).
    - time_zero: Time (in ps) to start analysis from.
    - size_block: Block size (in ps) for time-averaging.
    - mode_proba_cutoff -- minimum probability threshold for filtering regions between minima
    - output: Path to output file where selected features/labels are written.
    - output_dir: Directory where output data (e.g., plots or processed values) is stored.

    Returns:
    - None. Outputs are saved directly to disk.
    """

    num_residues = len(indices_na)

    logging.info("\nComputing dihedrals in nucleic acids backbone...")
    previous_progress = -1  # Initialize progress bar
    for i in range(num_residues):
        previous_progress=plot_progress_bar(i, num_residues,previous_progress)
        process_dihedral_i_nucleic_acids(i,  Positions_atoms_P, Positions_atoms_O5p, Positions_atoms_C5p, Positions_atoms_O4p, Positions_atoms_C4p, Positions_atoms_C3p, Positions_atoms_O3p, Positions_atoms_C1p, Positions_atoms_Nbs, Positions_atoms_Cbs, indices_na, times, time_zero, size_block, mode_proba_cutoff,output, output_dir)

    plot_progress_bar(num_residues, num_residues,previous_progress)
    logging.info("Dihedrals computed and saved.")


########################## Function to get the multimodal dihedrals of protein ################################
def get_dihedrals_protein(u_traj, indices_aa, time_zero, size_block, mode_proba_cutoff,output_dir):
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
    - mode_proba_cutoff: Minimum probability threshold for filtering regions between minima.
    - output_dir: Directory to save output files.

    Returns:
    - None. Saves dihedral angles and intermediate data to disk.
    """
    if len(indices_aa) < 2:
        logging.info("Not enough amino acids selected for dihedral analysis. Skipping.")
        return
    
    # Load time values and their corresponding frame indices
    times = np.load(output_dir + 'discretizing_npy/times.npy')
    times_indices = np.load(output_dir + 'discretizing_npy/times_indices.npy')

    # Step 1: Precompute backbone atom positions (N, C, and CA atoms)
    Positions_atoms_C, Positions_atoms_N, Positions_atoms_CA = precompute_backbone_protein(
        u_traj, indices_aa, times_indices
    )

    # Step 2: Save backbone atom positions to disk for future use
    save_positions(Positions_atoms_C, output_dir + "discretizing_npy/Positions_C_atoms.npy")
    save_positions(Positions_atoms_N, output_dir + "discretizing_npy/Positions_N_atoms.npy")
    save_positions(Positions_atoms_CA, output_dir + "discretizing_npy/Positions_CA_atoms.npy")

    # Step 3: Compute all dihedral angles and write selected features
    compute_all_dihedrals_protein(indices_aa, Positions_atoms_C, Positions_atoms_N, Positions_atoms_CA, times, time_zero, size_block, mode_proba_cutoff, output_dir + "selected_coordinates.txt", output_dir)


########################## Function to get the multimodal dihedrals of nucleic acids ################################
def get_dihedrals_nucleic_acids(u_traj, indices_na_pyrimidine,indices_na_purine, time_zero, size_block, mode_proba_cutoff, output_dir):
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
    - mode_proba_cutoff: Minimum probability threshold for filtering regions between minima.
    - output_dir: Directory to save output files.

    Returns:
    - None. Saves dihedral angles and intermediate data to disk.
    """
    if len(indices_na_pyrimidine) < 1 and len(indices_na_purine) < 1:
        logging.info("No nucleic acids selected for dihedral analysis.")
        return
    
    # Load time values and their corresponding frame indices
    times = np.load(output_dir + 'discretizing_npy/times.npy')
    times_indices = np.load(output_dir + 'discretizing_npy/times_indices.npy')

    # Step 1: Precompute backbone atom positions (N, C, and CA atoms)
    Positions_atoms_P, Positions_atoms_O5p, Positions_atoms_C5p, Positions_atoms_O4p, Positions_atoms_C4p, Positions_atoms_C3p, Positions_atoms_O3p, Positions_atoms_C1p, Positions_atoms_Nbs, Positions_atoms_Cbs = precompute_backbone_nucleic_acids(
        u_traj, indices_na_pyrimidine, indices_na_purine, times_indices
    )   

    # Step 2: Save backbone atom positions to disk for future use
    save_positions(Positions_atoms_P, output_dir + "discretizing_npy/Positions_P_atoms.npy")
    save_positions(Positions_atoms_O5p, output_dir + "discretizing_npy/Positions_O5p_atoms.npy")
    save_positions(Positions_atoms_C5p, output_dir + "discretizing_npy/Positions_C5p_atoms.npy")
    save_positions(Positions_atoms_O4p, output_dir + "discretizing_npy/Positions_O4p_atoms.npy")
    save_positions(Positions_atoms_C4p, output_dir + "discretizing_npy/Positions_C4p_atoms.npy")
    save_positions(Positions_atoms_C3p, output_dir + "discretizing_npy/Positions_C3p_atoms.npy")
    save_positions(Positions_atoms_O3p, output_dir + "discretizing_npy/Positions_O3p_atoms.npy")
    save_positions(Positions_atoms_C1p, output_dir + "discretizing_npy/Positions_C1p_atoms.npy")
    save_positions(Positions_atoms_Nbs, output_dir + "discretizing_npy/Positions_Nbs_atoms.npy")
    save_positions(Positions_atoms_Cbs, output_dir + "discretizing_npy/Positions_Cbs_atoms.npy")

    indices_na= np.sort(indices_na_pyrimidine+indices_na_purine)
    # Step 3: Compute all dihedral angles and write selected features
    compute_all_dihedrals_nucleic_acids(indices_na, Positions_atoms_P, Positions_atoms_O5p, Positions_atoms_C5p, Positions_atoms_O4p, Positions_atoms_C4p, Positions_atoms_C3p, Positions_atoms_O3p, Positions_atoms_C1p, Positions_atoms_Nbs, Positions_atoms_Cbs, times, time_zero, size_block, mode_proba_cutoff, output_dir + "selected_coordinates.txt", output_dir)


############################# Function to add new coordinates to the existing discretization ##########################
def add_coordinates(coordinates_to_add, type_coordinates_to_add,size_block,time_zero, mode_proba_cutoff, output_dir ):
    """
    Adds new coordinates (distance or angle) to an existing discretization setup.

    Arguments:
    coordinates_to_add -- list of file paths to the new coordinate data (.dat files)
    size_block -- block size for histogram averaging
    time_zero -- starting time point for block analysis
    type_coordinates_to_add -- list indicating the type of each coordinate ('distance' or 'angle')
    mode_proba_cutoff -- minimum probability threshold for filtering regions between minima
    output_dir -- path to the directory where outputs are stored
    
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

    logging.info("\nAdding new coordinates...")
    for i, coord_file in enumerate(coordinates_to_add):
        data_coord_raw = open_data_coordinate(coord_file)
        coordinate_name = coord_file.split('/')[-1].split('.')[0]
        coordinate_type = type_coordinates_to_add[i]

        # Set histogram resolution based on coordinate type
        if coordinate_type == 'distance':
            delta_y = 0.1
        elif coordinate_type == 'angle':
            delta_y = 2
        else:
            logging.info(f"Unknown coordinate type for {coord_file}, skipping.")
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
            logging.info(f"Warning: {coord_file} has different time steps than the reference file. Skipping.")
            continue

        # Fix angle wrapping (e.g., from -180 to 180 or 0 to 360)
        if coordinate_type == 'angle' and (np.max(y_coord) - np.min(y_coord) > 180):
            y_coord, _, _ = adjust_angle_data(y_coord, np.min(y_coord), np.max(y_coord), delta_y)

        # Discretize and append this coordinate to selected_coordinates.txt
        discretize_coordinate(y_coord, delta_y, coordinate_type,
                              times_to_compare, time_zero, size_block,
                              coordinate_name, mode_proba_cutoff,output_dir + "selected_coordinates.txt",
                              output_dir)
    logging.info("New coordinates added and discretized.")


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

    logging.info("\nDiscretizing data...")

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

    logging.info("Discretization completed.")

    # Save the resulting discretized data as a .npy file
    np.save(output_dir + "discretizing_npy/discretized_array.npy", data_discretized)


########################### Function to compute frequencies of single and double contacts ##########################
def compute_frequencies(discretized_array):
    """
    Compute marginal (single) and joint (double) frequencies for discretized coordinates.

    Parameters
    ----------
    discretized_array : ndarray of shape (n_frames, n_coords)
        The discretized representation of the coordinates.

    Returns
    -------
    single_frequencies : ndarray of shape (sum(multiplicities),)
        The marginal frequencies of each discrete state.

    double_frequencies : ndarray of shape (sum(multiplicities), sum(multiplicities))
        The joint frequencies between all pairs of discrete states.
    """
    n_frames, n_coords = discretized_array.shape
    multiplicities = get_multiplicities(discretized_array)
    total_bins = sum(multiplicities)

    # Precompute flat indices (offsets) for each coordinate
    offsets = np.cumsum([0] + list(multiplicities[:-1]))

    # Allocate output arrays
    single_frequencies = np.zeros(total_bins, dtype=float)
    double_frequencies = np.zeros((total_bins, total_bins), dtype=float)

    logging.info("Computing single frequencies...")
    for i in range(n_coords):
        col = discretized_array[:, i]
        offset = offsets[i]
        counts = np.bincount(col, minlength=multiplicities[i])
        single_frequencies[offset:offset + multiplicities[i]] = counts / n_frames
    logging.info("Single frequencies computed.")

    logging.info("Computing double frequencies...")
    total_steps = (n_coords * (n_coords + 1)) // 2
    step = 0
    prev_progress = -1

    for i in range(n_coords):
        col_i = discretized_array[:, i]
        offset_i = offsets[i]
        mult_i = multiplicities[i]

        for j in range(i, n_coords):  # i ≤ j for symmetry
            prev_progress = plot_progress_bar(step, total_steps, prev_progress)
            col_j = discretized_array[:, j]
            offset_j = offsets[j]
            mult_j = multiplicities[j]

            # Count joint occurrences
            joint_counts = np.zeros((mult_i, mult_j), dtype=int)
            np.add.at(joint_counts, (col_i, col_j), 1)

            joint_probs = joint_counts / n_frames

            # Fill both [i,j] and [j,i] blocks
            double_frequencies[offset_i:offset_i+mult_i, offset_j:offset_j+mult_j] = joint_probs
            if i != j:
                double_frequencies[offset_j:offset_j+mult_j, offset_i:offset_i+mult_i] = joint_probs.T

            step += 1
            

    plot_progress_bar(total_steps, total_steps, prev_progress)
    logging.info("Double frequencies computed.")

    return single_frequencies, double_frequencies

def get_frequencies(output_dir):
    # Load the discretized array from a .npy file located in the specified output directory
    discretized_array = np.load(output_dir + "discretizing_npy/discretized_array.npy")
    
    # Compute the single and double frequencies using a helper function 
    single_frequencies, double_frequencies = compute_frequencies(discretized_array)
    
    # Save the computed single frequencies to a file in the 'frequencies' subdirectory
    np.save(output_dir + 'analysis_npy/frequencies_single.npy', single_frequencies)
    
    # Save the computed double frequencies to a file in the 'frequencies' subdirectory
    np.save(output_dir + 'analysis_npy/frequencies_double.npy', double_frequencies)


########################### Function to plot mutual information matrix ##########################
def plot_information(Information_matrix,output_dir,name_out,label_data=None):
    """
    Plots the mutual information matrix and saves it as an image.
    Parameters:
    - Information_matrix: information matrix (2D numpy array).
    - output_dir: Directory where the plot will be saved.
    - name_out: Name of the output file (without extension).
    """
    plt.figure(figsize=(10, 6))
    plt.imshow(Information_matrix, cmap='magma', interpolation='nearest')
    plt.colorbar(label=label_data)
    plt.title(f'{label_data} Matrix')
    plt.xlabel('Coordinate Index')
    plt.ylabel('Coordinate Index')
    plt.tight_layout()
    plt.savefig(output_dir+name_out+'.png', dpi=200)
    plt.close()

def plot_information_clustered(Information_matrix, reordered_labels, output_dir, name_out, label_data=None,xlabel=None, ylabel=None):
    """
    Plots the mutual information matrix with boxed cluster boundaries.

    Parameters:
    - Information_matrix: 2D numpy array (mutual information matrix).
    - reordered_labels: List or array of cluster labels (in reordered coordinate order).
    - output_dir: Directory to save the plot.
    - name_out: Output file name (without extension).
    - label_data: Optional string for the colorbar label.
    """
    plt.figure(figsize=(10, 8))
    ax = plt.gca()

    # Plot the information matrix
    im = ax.imshow(Information_matrix, cmap='magma', interpolation='nearest')
    plt.colorbar(im, label=label_data)
    plt.title(f'{label_data} Matrix with Cluster Boxes in white and noise in blue' if label_data else "Clustered Information Matrix")

    # Find cluster boundaries
    boundaries = []
    last_label = reordered_labels[0]
    unique_labels= [last_label]  # Initialize with the first label
    start = 0
    for i, label in enumerate(reordered_labels):
        if label != last_label:
            boundaries.append((start, i))
            start = i
            last_label = label
            unique_labels.append(label)  # Add new label to the list

    boundaries.append((start, len(reordered_labels)))  # Add the last block
    # Draw rectangles for each cluster
    countclus=0
    for start, end in boundaries:
        size = end - start
        ecolor='white'
        label_clus=unique_labels[countclus]
        if label_clus == -1:
            ecolor='deepskyblue'
        countclus+=1

        rect = Rectangle(
            (start - 0.5, start - 0.5),  # (x, y) of bottom-left corner
            size,                       # width
            size,                       # height
            linewidth=2,
            edgecolor=ecolor,
            facecolor='none'
        )
        ax.add_patch(rect)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{name_out}.png", dpi=300)
    plt.close()

########################## Function to compute mutual information between coordinates ##########################
def mutual_information(discretized_array, multiplicities, single_frequencies, double_frequencies):
    """
    Compute the pairwise mutual information (MI) matrix for discretized coordinates.

    Parameters:
    -----------
    discretized_array : ndarray (n_frames, n_coords)
        The discretized representation of the coordinates.
    multiplicities : array-like of int, shape (n_coords,)
        Number of discrete states (bins) for each coordinate.
    single_frequencies : ndarray, shape (sum(multiplicities),)
        Marginal probabilities for each discrete state across all coordinates.
    double_frequencies : ndarray, shape (sum(multiplicities), sum(multiplicities))
        Joint probabilities between all discrete state pairs.

    Returns:
    --------
    MI : ndarray, shape (n_coords, n_coords)
        The mutual information matrix in bits.
    """
    n_frames, n_coords = discretized_array.shape
    MI = np.zeros((n_coords, n_coords), dtype=float)
    epsilon = 1e-12  # Small constant to prevent log(0)

    # Precompute offsets for flattened state indexing
    offsets = np.cumsum([0] + list(multiplicities[:-1]))

    for i in range(n_coords):
        for xi in range(multiplicities[i]):
            idx_i = offsets[i] + xi
            p_xi = single_frequencies[idx_i]

            if p_xi < epsilon:
                continue  # skip very low probability states

            for j in range(n_coords):
                for xj in range(multiplicities[j]):
                    idx_j = offsets[j] + xj
                    p_xj = single_frequencies[idx_j]
                    p_xi_xj = double_frequencies[idx_i, idx_j]

                    # Apply mutual information formula only if valid
                    if p_xi_xj > epsilon and p_xj > epsilon:
                        MI[i, j] += p_xi_xj * np.log(p_xi_xj / (p_xi * p_xj))  # in bits

    return MI

def get_mutual_information(output_dir):
    """
    Load discretized data and precomputed frequency tables from disk,
    compute the mutual information (MI) matrix, and save the result.

    Parameters:
    -----------
    output_dir : str
        Path to the directory containing the results.
    """
    logging.info("\nComputing mutual information...")

    # Load discretized coordinate array
    discretized_array = np.load(os.path.join(output_dir, "discretizing_npy/discretized_array.npy"))

    # Load marginal and joint frequencies
    single_frequencies = np.load(os.path.join(output_dir, "analysis_npy", "frequencies_single.npy"))
    double_frequencies = np.load(os.path.join(output_dir, "analysis_npy", "frequencies_double.npy"))

    # Compute multiplicities: number of discrete bins for each coordinate
    multiplicities = get_multiplicities(discretized_array)

    # Compute mutual information matrix
    MI = mutual_information(discretized_array, multiplicities, single_frequencies, double_frequencies)

    # Save the result to output directory
    output_path = os.path.join(output_dir, "analysis_npy")
    os.makedirs(output_path, exist_ok=True)  # Ensure output directory exists
    np.save(os.path.join(output_path, "MI.npy"), MI)

    logging.info("Mutual information computed.")
    # Plot the mutual information matrix
    plot_information(MI, output_dir + 'information_plots/', "MI_matrix", label_data="Mutual Information")


########################## Function to compute entropy  ##########################
def get_entropy(output_dir):
    logging.info("\nComputing entropy...")
    discretized_array=np.load(output_dir+"discretizing_npy/discretized_array.npy")
    single_frequencies=np.load(output_dir+'analysis_npy/frequencies_single.npy')
    multiplicities=get_multiplicities(discretized_array)
    ncoord=len(multiplicities)
    entropy=np.zeros((ncoord),dtype=float)
    count_index=0
    for i in range(ncoord):
        for xi in range(multiplicities[i]):
            probab_xi=single_frequencies[count_index]
            count_index+=1
            if probab_xi>0:
                entropy[i]-=probab_xi*np.log(probab_xi)

    np.save(output_dir+'analysis_npy/entropy.npy', entropy)
    logging.info("Entropy computed.")

    #plot the entropy values
    plt.figure(figsize=(8, 4))
    plt.bar(range(ncoord), entropy, color='blue', alpha=0.7)
    plt.xlabel('Coordinate Index')
    plt.ylabel('Entropy')
    plt.title('Entropy of Coordinates')
    plt.tight_layout()
    plt.savefig(output_dir + 'information_plots/entropy_plot.png', dpi=200)
    plt.close()
    

######################### Function to compute Variation Information ##########################
def get_variation_information(output_dir):
    """
    Computes the Variation Information (VI) matrix from the mutual information (MI) matrix and the entropy.

    The VI matrix is computed as:
    VI(i, j) = H(i) + H(j) - 2 * MI(i, j)
    where H(i) and H(j) are the entropies of coordinates i and j, respectively.

    Parameters:
    -----------
    output_dir : str
        Path to the directory containing the MI matrix and entropy values.
    
    Returns:
    --------
    None. The VI matrix is saved to disk.
    """
    
    logging.info("\nComputing variation information...")

    # Load the mutual information matrix and entropy values
    MI = np.load(os.path.join(output_dir, "analysis_npy", "MI.npy"))
    entropy = np.load(os.path.join(output_dir, "analysis_npy", "entropy.npy"))

    # Compute the Variation Information matrix
    ncoord = len(entropy)
    VI = np.zeros((ncoord, ncoord), dtype=float)

    for i in range(ncoord):
        for j in range(ncoord):
            VI[i, j] = entropy[i] + entropy[j] - 2 * MI[i, j]
    # Ensure VI is non-negative (can happen if MI is too high)
    min_VI = np.min(VI)
    if min_VI < 0:
        VI -= min_VI  # Shift to make minimum zero

    # Ensure the VI matrix is symmetric
    VI = (VI + VI.T) / 2
    # Ensure diagonal elements are zero (no self-information)
    np.fill_diagonal(VI, 0)

    # Save the VI matrix to a file
    np.save(os.path.join(output_dir, "analysis_npy", "VI.npy"), VI)

    # Plot the VI matrix
    plot_information(VI, output_dir + 'information_plots/', "VI_matrix", label_data="Variation Information")

    logging.info("Variation information computed.")


######################### Function to cluster using Advanced Density Peaks ##########################
def density_peaks_clustering(distance_matrix, Z_parameter=1.65, halo_parameter=0):
    """
    Applies Density Peaks Clustering (ADP version) on a precomputed distance matrix.

    Parameters
    ----------
    distance_matrix : np.ndarray of shape (n_samples, n_samples)
        Symmetric pairwise distance matrix between conformations or data points.
    
    Z_parameter : float, default=1.65
        Confidence level for the clustering decision threshold (used in ADP).
        Typical values: 1.65 (≈ 95% confidence), 2.0 (≈ 97.5%), etc.

    halo_parameter : int, default=0
        If set to 1, identifies border points (halo) around clusters.

    Returns
    -------
    cluster_labels : np.ndarray of shape (n_samples,)
        Array of integer cluster labels for each data point.
    """

    # Get the number of data points (states/conformations)
    n_states = np.shape(distance_matrix)[0]

    # Dummy coordinates (not used, but required by DADApy's Data object)
    x_dummy = np.zeros((n_states, 2), dtype=float)

    # Create a buffer to capture stdout
    buf = io.StringIO()

    # Redirect stdout/stderr to the buffer
    with redirect_stdout(buf), redirect_stderr(buf):
        import dadapy
        data = dadapy.Data(coordinates=x_dummy, distances=distance_matrix, verbose=True)
        data.compute_id_2NN()
        data.compute_density_PAk()
        data.compute_clustering_ADP_pure_python(Z=Z_parameter, halo=bool(halo_parameter))



    # Log the captured output
    output = buf.getvalue()
    if output.strip():
        logging.info("[DADApy output]\n" + output.strip())

    # Return cluster labels
    cluster_labels = data.cluster_assignment

    return cluster_labels

def hdbscan_clustering(distance_matrix, min_cluster_size=5, min_samples=5, cluster_selection_epsilon=0.0):
    """
    Applies HDBSCAN clustering on a precomputed distance matrix.

    Parameters
    ----------
    distance_matrix : np.ndarray of shape (n_samples, n_samples)
        Symmetric pairwise distance matrix between conformations or data points.
    
    min_cluster_size : int, default=5
        Minimum size of clusters to be considered valid.
    
    min_samples : int, default=5
        Minimum number of samples in a neighborhood for a point to be considered a core point.

    Returns
    -------
    cluster_labels : np.ndarray of shape (n_samples,)
        Array of integer cluster labels for each data point.
    """
    
    import hdbscan

    # Create HDBSCAN clusterer with specified parameters
    clusterer = hdbscan.HDBSCAN(min_cluster_size=int(min_cluster_size), min_samples=int(min_samples),cluster_selection_epsilon=cluster_selection_epsilon, metric='precomputed')

    # Fit the model to the distance matrix and get cluster labels
    cluster_labels = clusterer.fit_predict(distance_matrix)

    return cluster_labels

def yacare_clustering(distance_matrix,function_for_ratio=2,threshold_variable=0.5,amount_of_noise=0.0,keep_no_noise=1):
    # Create a buffer to capture stdout
    buf = io.StringIO()

    # Redirect stdout/stderr to the buffer
    with redirect_stdout(buf), redirect_stderr(buf):
        import yacare
        
        save_images = False
        show_images = False
        percentage_moving_square = min(25,10*100.0 / distance_matrix.shape[0])  # Percentage of moving square for reordering
        minimal_size_cluster = 0.000001
        choice_merging_clusters = 3
        keep_no_noise = bool(keep_no_noise)  # Convert to boolean

        variables = yacare.Variables()
        variables.distance_matrix = distance_matrix
        variables.project_name = 'temp_yacare_clustering_CASIMODO'
        variables.show_images = show_images
        variables.save_images = save_images
        variables.function_for_ratio = function_for_ratio
        
        yacare.perform_first_reordering(variables, percentage_moving_square = percentage_moving_square, vmax = -1)

        yacare.find_optimal_cutoff(variables, minimal_size_cluster = minimal_size_cluster, use_all_cutoff = True, function_for_ratio = 1)
        
        yacare.find_final_clusters(variables, vmax=-1)
        
        yacare.propose_list_for_concatenating_clusters(variables, threshold_variable = threshold_variable, choice_merging_clusters = choice_merging_clusters)
        
        yacare.concatenate_clusters(variables, vmax = -1)

        yacare.expand_clusters(variables, amount_of_noise = amount_of_noise, keep_no_noise = keep_no_noise, vmax = -1)
        
        yacare.compare_final_clusters(variables, display_stddev = True, display_mean_distances = True)
        
        yacare.write_indices(variables)

        # Extract the data from our clustering. We write a list of list, which contains "index of the data, index of the cluster".
        list_clustered_data = []
        for i in range(variables.number_clusters_write_indices):
            for j in range(len(variables.elements_inside_clusters_write_indices[i])):
                list_clustered_data.append([variables.elements_inside_clusters_write_indices[i][j]+1, i])
        for j in range(len(variables.elements_outside_clusters_write_indices)):
            list_clustered_data.append([variables.elements_outside_clusters_write_indices[j]+1, -1])

        # The list is sorted using the index of the data, i.e. from 1 to N.
        list_clustered_data_sorted = sorted(list_clustered_data, key=lambda x: x[0])
        # We extract the index of the clusters for each data.
    # Log the captured output
    output = buf.getvalue()
    if output.strip():
        logging.info("[DADApy output]\n" + output.strip())
    cluster_labels = np.array([x[1] for x in list_clustered_data_sorted])
    list_sufixes=['_Clustering_Clusters.ndx', '_Clustering_Labels.txt', '_Clustering_Noise.txt','_Clustering_ReorderedElements.txt','_Clustering_RepresentativeStructures.ndx','_Yacare_Summary.txt']
    for sufix in list_sufixes :
        os.remove(variables.project_name + sufix)  # Remove the temporary files created by Yacare
    
    return cluster_labels

def cluster_distances(distance_matrix, method_clustering, parameters_clustering) :
    if method_clustering == 'advanced_density_peaks':
        cluster_labels = density_peaks_clustering(distance_matrix, *parameters_clustering)
    elif method_clustering == 'hdbscan':
        cluster_labels = hdbscan_clustering(distance_matrix, *parameters_clustering)
    elif method_clustering == 'yacare':
        cluster_labels = yacare_clustering(distance_matrix, *parameters_clustering)
    return cluster_labels


############# Function to plot clustering results ##########################
def plot_clustering_results(dist_matrix,cluster_labels, output_dir, output_name, label_data=None, xlabel='X-axis', ylabel='Y-axis'):
    """
    Plots the results of clustering on the mutual information distance matrix.

    This function loads the cluster labels and distance matrix, then generates a scatter plot
    of the coordinates colored by their cluster labels. It also saves the plot to a file.

    Parameters:
    -----------
    output_dir : str
        Path to the directory containing the analysis results.
    
    Returns:
    --------
    None. The plot is saved to disk.
    """
    
    logging.info("\nPlotting clustering results...")

    # Load cluster labels and distance matrix
    unique_labels= np.unique(cluster_labels)
    sorted_indices = []
    for label in unique_labels:
        if label == -1:  # Noise
            continue
        indices = np.where(cluster_labels == label)[0]

        if len(indices) == 0:
            continue
        sub_mi = dist_matrix[np.ix_(indices, indices)]
        mi_sums = sub_mi.sum(axis=1)
        order = indices[np.argsort(mi_sums)]  # descending
        sorted_indices.extend(order)

    # Add noise at the end
    noise_indices = np.where(cluster_labels == -1)[0]
    sorted_indices.extend(noise_indices)

    reordered_labels= cluster_labels[sorted_indices]

    dist_reordered = dist_matrix[sorted_indices, :][:, sorted_indices]

    plot_information_clustered(dist_reordered,reordered_labels, output_dir, output_name, label_data, xlabel=xlabel, ylabel=ylabel)

    
    logging.info("Clustering results plotted and saved.")

    return reordered_labels
    

#################### Function to extract the coordinates in each cluster ##########################
def write_clusters_to_file(clusters_ndx, coordinates, output_dir, name_output_cluster):

    logging.info("\nWriting clusters to file...")
    with open(output_dir + name_output_cluster, 'w') as file_out:
        for i, cluster_i in enumerate(clusters_ndx):
            
            if i != len(clusters_ndx) - 1:
                file_out.write(f'[ Cluster{i} ]\n')
            else:
                file_out.write(f'[ Noise ]\n')
            for index_coord in cluster_i:
                file_out.write(f'{coordinates[index_coord]} \n')
            file_out.write('\n')

    logging.info("Clusters written to file.")

def get_resids_in_clusters(clusters_ndx,coordinates,name_coordinates_to_add,residues_coordinates_to_add,name_output,output_dir):
    logging.info("\nGetting resids in clusters...")
    file_out=open(output_dir+name_output,'w')
    for i in range (len(clusters_ndx)):
        cluster_i=clusters_ndx[i]
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
                name_resid_to_add=int(residues_coordinates_to_add[index_coord_to_add].split('_')[0])
                if name_resid_to_add not in resids_in_cluster_i:
                    resids_in_cluster_i.append(name_resid_to_add)
                    
            elif coord.split('_')[0] in ('phi','psi','alpha','beta','gamma','delta','epsilon','zeta','chi') : 
                name_resid_to_add= int(coord.split('_')[1])
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
        file_out.write('\n\n')
    logging.info("Getting resids in clusters completed.")
    file_out.close()


############ Function to compute information metrics and save results ##############
def compute_information(output_dir):
    """
    Computes mutual information, entropy, and variation information for the discretized coordinates.

    This function loads the discretized data, computes the necessary information metrics,
    and saves the results to disk. It also plots the mutual information matrix.

    Parameters:
    -----------
    output_dir : str
        Path to the directory containing the discretized data.

    Returns:
    --------
    None. The computed metrics are saved to disk.
    """
    
    get_frequencies(output_dir)
    get_mutual_information(output_dir)
    get_entropy(output_dir)
    get_variation_information(output_dir)


############ Function to cluster coordinates based on mutual information distance, using Advanced Density Peaks ##############
def cluster_coordinates(output_dir,coordinates_to_add,residues_coordinates_to_add, method_clustering_coordinates, parameters_clustering_coordinates):
    """
    Clusters coordinates based on mutual information distance using Advanced Density Peaks.

    This function loads the mutual information distance matrix, applies Advanced Density Peaks clustering,
    and saves the resulting cluster labels to a file.

    Parameters:
    -----------
    output_dir : str
        Path to the directory containing the MI distance matrix.
    coordinates_to_add : list of str
        List of file paths to additional coordinates to be clustered.
    residues_coordinates_to_add : list of str
        List of residue identifiers corresponding to the additional coordinates.
    method_clustering_coordinates : str
        Clustering method to use (e.g., 'advanced_density_peaks', 'hdbscan', 'yacare').
    parameters_clustering_coordinates : list
        Parameters for the clustering method, such as Z_parameter and halo_parameter for ADP.

    Returns:
    --------
    None. The cluster labels are saved to disk.
    """

    logging.info("\nClustering coordinates using Advanced Density Peaks...")

    # Load the mutual information distance matrix
    distance_matrix = np.load(os.path.join(output_dir, "analysis_npy", "VI.npy"))
    normalized_distance_matrix = distance_matrix / np.max(distance_matrix)  # Normalize to [0, 1]

    #Apply Density Peaks Clustering
    cluster_labels = cluster_distances(distance_matrix, method_clustering_coordinates, parameters_clustering_coordinates) 

    # Save the cluster labels to a file
    np.save(os.path.join(output_dir, "analysis_npy", "cluster_labels.npy"), cluster_labels)

    logging.info("Clustering completed and labels saved.")

    reordered_labels = plot_clustering_results(normalized_distance_matrix,cluster_labels, output_dir+'information_plots/', "VI_clustering", "Normalized variation of Information",xlabel="Coordinate Index", ylabel="Coordinate Index")

    coordinates,X_cuts,Labels=load_data_discretization(output_dir + "selected_coordinates.txt")

    # Extract clusters and write to file
    clusters_ndx = []
    
    noise_ndx = np.where(cluster_labels == -1)[0]  # Indices of noise points
    for label in np.unique(cluster_labels):
        if label == -1:  # Noise points
            continue
        cluster_indices = np.where(cluster_labels == label)[0]
        clusters_ndx.append(cluster_indices)    
    # Add noise points as a separate cluster
    clusters_ndx.append(noise_ndx)
    # Write clusters to file
    write_clusters_to_file(clusters_ndx, coordinates, output_dir, "clusters_of_coordinates.txt")
    # Get resids in clusters and write to file
    name_coordinates_to_add = [coord.split('/')[-1].split('.')[0] for coord in coordinates_to_add]
    get_resids_in_clusters(clusters_ndx, coordinates, name_coordinates_to_add,residues_coordinates_to_add, "resids_in_clusters.txt", output_dir)


###################### Functions to manipulate states and get conformations ########################
def split_discretized_array_by_clusters(discretized_array, cluster_labels):
    """
    Splits the discretized array into sub-arrays based on cluster labels.

    Parameters:
    -----------
    discretized_array : ndarray
        The discretized representation of the coordinates.
    cluster_labels : ndarray
        The cluster labels for each frame in the discretized array.

    Returns:
    --------
    clusters_data : list of ndarray
        A list where each element is a sub-array corresponding to a unique cluster.
    """
    unique_labels = np.unique(cluster_labels)
    clusters_data = []

    for label in unique_labels:
        if label == -1:  # Skip noise points
            continue
        indices = np.where(cluster_labels == label)[0]
        clusters_data.append(discretized_array[:,indices])

    return clusters_data

def get_unique_states_in_splitted_array(clusters_data,cutoff_len_states,cluster_of_coordinates_to_process):
    """
    Extracts unique states from each cluster's discretized data.

    Parameters:
    -----------
    clusters_data : list of ndarray
        A list where each element is a sub-array corresponding to a unique cluster.

    Returns:
    --------
    unique_states : list of ndarray
        A list containing unique states for each cluster.
    """
    unique_states = []
    probalities_unique_states = []
    for i,cluster_data in enumerate(clusters_data):
        if cluster_of_coordinates_to_process>=0 and i != cluster_of_coordinates_to_process:
            probalities_unique_states.append([])
            unique_states.append([]) 
            continue
        unique_i,count_i= np.unique(cluster_data, axis=0, return_counts=True)
        proba_i= count_i / cluster_data.shape[0]  # Normalize counts to get probabilities
        
        sorted_proba_indices = np.argsort(proba_i)[::-1]  # Sort indices by probability in descending order
        
        unique_i = unique_i[sorted_proba_indices]  # Sort unique states by their probabilities
        proba_i = proba_i[sorted_proba_indices]  # Sort counts accordingly
        
        #cumulative_proba = np.cumsum(proba_i) 
        #cutoff_index = np.where(cumulative_proba >= cutoff_proba_filtering)[0][0]  # Find the index where cumulative probability exceeds the cutoff
        
        unique_i = unique_i[:cutoff_len_states]  # Keep only states up to the cutoff
        proba_i = proba_i[:cutoff_len_states]

        probalities_unique_states.append(proba_i) 
        unique_states.append(unique_i)
        
    return unique_states, probalities_unique_states

def compute_distances_between_states(states,cluster_of_coordinates_to_process):
    """
    Computes pairwise distances between unique states.

    Parameters:
    -----------
    states : list of ndarray
        A list where each element is an array of unique states for a cluster.
    cluster_of_coordinates_to_process : int
        Index of the cluster to process (if > 0, only this cluster is processed).

    Returns:
    --------
    distances : list of ndarray
        A list containing distance matrices for each cluster's unique states.
    """
    distances = []
    for i,state in enumerate(states):
        if cluster_of_coordinates_to_process>=0 and i != cluster_of_coordinates_to_process:
            distances.append([])
            continue

        dist_matrix = squareform(pdist(state, metric='hamming'))
        if np.max(dist_matrix) != 0:
            dist_matrix =dist_matrix/ np.max(dist_matrix)  

        distances.append(dist_matrix)
    return distances

def extract_frames_from_labels(output_dir, clusters_data, unique_states_clusters, all_clusters_labels, times_indices, proba_clusters, cutoff_proba_conformations,cluster_of_coordinates_to_process):
    """
    Extracts the original frame indices corresponding to each conformation
    within each cluster, based on clustering labels of unique states.

    Parameters:
    -----------
    output_dir : str
        Directory where output .ndx files will be saved.
    clusters_data : list of ndarray
        Each element contains the discretized states for a cluster.
    unique_states_clusters : list of ndarray
        Unique states (rows) found in each cluster.
    all_clusters_labels : list of ndarray
        clustering labels of unique states in each cluster.
    times_indices : list or ndarray
        Mapping of indices to the original frame times.
    proba_clusters : list of ndarray
        Probabilities of each conformation in each cluster.
    cutoff_proba_conformations : float
        Probability threshold to filter out low-probability conformations.
    cluster_of_coordinates_to_process : int
        Index of the cluster to process (if > 0, only this cluster is processed).

    Returns:
    --------
    frames_by_clusters : list of list of list of int
        Frame indices for each conformation in each cluster.
        Structure: cluster → conformation → list of frame indices.
    """

    logging.info("Extracting frames from states...")

    frames_by_clusters = []

    for i, cluster_labels in enumerate(all_clusters_labels):
        if cluster_of_coordinates_to_process > 0 and i != cluster_of_coordinates_to_process:
            frames_by_clusters.append([])
            continue
        
               
        unique_labels = np.unique(cluster_labels)
        nb_conformations = len(unique_labels)
      

        # Prepare storage for frames belonging to each conformation
        frames_conformations = [[] for _ in range(nb_conformations)]

        # Map each frame to its conformation based on the state
        for t in range(len(times_indices)):
            state = clusters_data[i][t]  # Discretized state at frame t

            # Find which unique state this frame matches
            if state in unique_states_clusters[i]:
                index_state = np.where((unique_states_clusters[i] == state).all(axis=1))[0][0]

                # Get the clustering label for that unique state
                label_index = list(unique_labels).index(cluster_labels[index_state])

                # Add corresponding time index
                frames_conformations[label_index].append(times_indices[t])

        frames_by_clusters.append(frames_conformations)

        count_large_proba =len(np.where(proba_clusters[i] >= cutoff_proba_conformations)[0])
        if count_large_proba <= 1:
            logging.warning(f"Cluster {i} has no several conformations to process.")
            continue
        
        # Open output file for current cluster
        output_file = open(f"{output_dir}conformations_clustering/frames_conformations_from_cluster_of_CV_{i}.ndx", 'w')

        # Write conformations (excluding noise) to file
        for j in range(nb_conformations):
            proba_conformation= proba_clusters[i][j]
            if unique_labels[j] == -1 or proba_conformation < cutoff_proba_conformations or len(frames_conformations[j]) == 0:
                continue
            output_file.write(f"[ Conformation_{unique_labels[j]} ]\n")
            indexes = frames_conformations[j]
            for k in range(0, len(indexes), 20):
                chunk = indexes[k:k + 20]
                output_file.write(" ".join(map(str, chunk)) + "\n")
            output_file.write("\n")

        output_file.close()
    
    logging.info("Frame extraction completed.")        

    return frames_by_clusters

def split_trajectory_by_conformations(output_dir, u_traj, frames_by_clusters,proba_clusters,cutoff_proba_conformations,all_clusters_labels,strucfile,trajfile,selected_resids,cluster_of_coordinates_to_process):
    """
    Splits the trajectory into separate files for each identified conformation
    in each cluster, based on frame indices.

    Parameters:
    -----------
    output_dir : str
        Path to the base output directory.
    u_traj : MDAnalysis Universe
        MDAnalysis trajectory object containing atom and trajectory data.
    frames_by_clusters : list of list of list of int
        Nested list where each element represents a cluster,
        containing sublists of frame indices for each conformation.
        Structure: cluster → conformation → frame indices.
    """
    extension_struc = strucfile.split('.')[-1]
    extension_traj = trajfile.split('.')[-1]

    logging.info("\nSplitting trajectory by conformations...")

    atoms_selected = u_traj.select_atoms(f"resnum {' '.join(map(str, selected_resids))}")
    atoms_selected.write(output_dir + "conformations_clustering/atoms_selected." + extension_struc)

    for i, frames_conformations in enumerate(frames_by_clusters):
        if cluster_of_coordinates_to_process > 0 and i != cluster_of_coordinates_to_process:
            logging.info(f"Skipping cluster {i} as it is not the one to process.")
            continue
        logging.info(f"Processing cluster {i}...")

        count_large_proba =len(np.where(proba_clusters[i] >= cutoff_proba_conformations)[0])
        if count_large_proba <= 1:
            logging.warning(f"Cluster {i} has no several conformations to process.")
            continue

        # Create directory for storing split trajectories from current cluster
        cluster_output_dir = os.path.join(output_dir, f"conformations_clustering/trajectories_cluster_{i}")
        if os.path.exists(cluster_output_dir):
            shutil.rmtree(cluster_output_dir)  # Remove existing directory            
        os.mkdir(cluster_output_dir)
        unique_labels = np.unique(all_clusters_labels[i])
    
        for j, frames in enumerate(frames_conformations):
            
            proba_conf = proba_clusters[i][j]
            
            if len(frames) == 0 or proba_conf < cutoff_proba_conformations or unique_labels[j] == -1 :
                continue  # Skip empty frames or low-probability conformations or noise
            
            logging.info(f"Writing conformation {unique_labels[j]} in cluster {i} with probability {proba_conf:.2f}...")

            # Define output file path for current conformation
            output_file = os.path.join(
                cluster_output_dir, f"cluster_{i}_conformation_{unique_labels[j]}.{extension_traj}"
            )

            # Write selected frames to new trajectory file
            atoms_selected = u_traj.select_atoms(f"resnum {' '.join(map(str, selected_resids))}")
            atoms_selected.write(output_file, frames=frames)
    logging.info("Trajectory splitting completed.")

def get_most_probable_states(all_clusters_labels, unique_states_clusters, probabilities_unique_states_clusters,cutoff_proba_conformations,cluster_of_coordinates_to_process):
    """
    For each cluster, identify the most probable state (discretized conformation)
    within each sub-cluster (i.e., clustering-labeled conformation).

    Parameters:
    -----------
    all_clusters_labels : list of ndarray
        List of clustering label arrays, one per main cluster.
        Each array gives the label for each unique state within that cluster.
        Example shape: [n_main_clusters][n_states_in_cluster_i]

    unique_states_clusters : list of ndarray
        List of arrays containing all unique discretized states in each main cluster.
        Example shape: [n_main_clusters][n_unique_states_in_cluster_i, n_coords]

    probabilities_unique_states_clusters : list of ndarray
        List of probability arrays corresponding to each unique state in each main cluster.
        Example shape: [n_main_clusters][n_unique_states_in_cluster_i]

    cutoff_proba_conformations : float
        Probability threshold to consider a conformation as valid.
    cluster_of_coordinates_to_process : int
        If greater than 0, only process the specified cluster of coordinates.

    Returns:
    --------
    most_probable_states : list of list of ndarray
        For each cluster, a list of the most probable state in each clustering sub-cluster (i.e., conformation).
        Structure: [n_main_clusters][n_conformations_in_cluster_i]

    proba_most_probable_states : list of list of float
        Corresponding probabilities for each most probable state.
        Structure mirrors `most_probable_states`.
    """
    most_probable_states = []
    proba_most_probable_states = []

    # Loop through each main cluster
    for i, cluster_labels in enumerate(all_clusters_labels):
        if cluster_of_coordinates_to_process > 0 and i != cluster_of_coordinates_to_process:
            most_probable_states.append([])
            proba_most_probable_states.append([])
            continue
        most_probable_states_cluster = []
        proba_most_probable_states_cluster = []

        # Get unique conformation labels in current cluster
        unique_labels = np.unique(cluster_labels)

        # Find indices of unique states that belong to each conformation label
        ind_labels_cluster = [
            np.where(cluster_labels == label)[0] for label in unique_labels
        ]

        # Loop through conformations (clustering sub-clusters)
        for j, ind_labels in enumerate(ind_labels_cluster):
            # Get the probabilities of the states in the current conformation
            proba_cluster_conf_j = probabilities_unique_states_clusters[i][ind_labels]

            # Identify the state with the highest probability
            ind_max_proba = ind_labels[np.argmax(proba_cluster_conf_j)]

            # Save the most probable state and its probability
            most_probable_states_cluster.append(unique_states_clusters[i][ind_max_proba])
            proba_most_probable_states_cluster.append(
                probabilities_unique_states_clusters[i][ind_max_proba]
            )

            if np.sum(proba_cluster_conf_j) > cutoff_proba_conformations :
                # Log the result for tracking
                if unique_labels[j] != -1 :
                    logging.info(
                        f"Most probable state in cluster {i}, conformation {unique_labels[j]}: "
                        f"{unique_states_clusters[i][ind_max_proba]} "
                        f"with probability {probabilities_unique_states_clusters[i][ind_max_proba]}"
                    )
        # Append results for the current cluster
        most_probable_states.append(most_probable_states_cluster)
        proba_most_probable_states.append(proba_most_probable_states_cluster)

    return most_probable_states, proba_most_probable_states

def get_coordinates_in_clusters(output_dir): 
    """
    Parses the 'clusters_of_coordinates.txt' file to extract coordinate groupings for each cluster.

    This function assumes that the file contains sections like:
    [ Cluster 0 ]
    coord_1
    coord_2
    ...
    [ Cluster 1 ]
    ...

    Parameters:
    -----------
    output_dir : str
        Path to the output directory where 'clusters_of_coordinates.txt' is stored.

    Returns:
    --------
    clusters_coords : list of list of str
        Each element is a list of coordinate names (as strings) belonging to a cluster.
        The outer list contains one entry per cluster.
    """
    file_clusters = open(output_dir + "clusters_of_coordinates.txt", 'r')
    clusters_coords = []  # List to hold coordinates per cluster
    current_cluster = []  # Temporarily store coordinates for current cluster

    for line in file_clusters:
        line = line.strip()

        # Start of a new cluster section
        if line.startswith("[ Cluster") or line.startswith("[ Noise"):
            # Save the previous cluster if it had any coordinates
            if len(current_cluster) > 0:
                clusters_coords.append(current_cluster)
                current_cluster = []  # Reset for the next cluster

        elif line:
            # Line contains a coordinate name, add to current cluster
            current_cluster.append(line)

    # Don't forget to append the last cluster if not empty
    if len(current_cluster) > 0:
        clusters_coords.append(current_cluster)

    return clusters_coords

def write_conformations_to_file(all_cluster_labels,most_probable_states, proba_most_probable_states, proba_clusters, output_dir, cutoff_proba_conformations,cluster_of_coordinates_to_process):
    """
    Writes the most probable conformational states of each cluster to a human-readable text file.

    For each cluster (corresponding to a group of coordinates), this function writes:
    - The cluster header.
    - Each most probable conformation and its associated probabilities.
    - The discretized state values (per coordinate) that define each conformation.

    Parameters:
    -----------
    most_probable_states : list of list of ndarray
        Each element corresponds to a cluster and contains the most probable states (as arrays) per conformation.

    proba_most_probable_states : list of list of float
        Probabilities of each most probable state in the corresponding conformation, per cluster.

    proba_clusters : list of ndarray
        Each element is an array containing the total probability of each conformation in the corresponding cluster.

    output_dir : str
        Directory where the output file ("conformations.txt") will be saved.
    """
    clusters_coords = get_coordinates_in_clusters(output_dir)  # Get coordinate names (CVs) associated with each cluster
    logging.info("\nWriting conformations to file...")

    # Open the output file for writing
    # Loop over clusters
    for i, cluster_states in enumerate(most_probable_states):
        if cluster_of_coordinates_to_process > 0 and i != cluster_of_coordinates_to_process:
            continue
        with open(output_dir + f"conformations_clustering/conformations_cluster_{i}.txt", 'w') as file_out:
            
            file_out.write(f"[ Cluster {i} ]\n")
            unique_cluster_labels = np.unique(all_cluster_labels[i])
            # Loop over conformations within the cluster
            for j, state in enumerate(cluster_states):

                if unique_cluster_labels[j]==-1 or proba_clusters[i][j] < cutoff_proba_conformations:
                    continue
                file_out.write(f"Conformation {unique_cluster_labels[j]} - Probability: {proba_clusters[i][j]:.5f}\n")
                file_out.write(f"Most probable state: {state}\n")
                file_out.write(f"Probability of the most probable state: {proba_most_probable_states[i][j]:.5f}\n")
                file_out.write("Discretized values:\n")

                # Write coordinate name and value
                for k, coord in enumerate(state):
                    file_out.write(f"{clusters_coords[i][k]}: {coord}\n")
                file_out.write('\n')  # Blank line between conformations

            file_out.write('\n')  # Blank line between clusters


######################### Function to extract conformations from clusters ##########################
def get_conformations_from_clusters(output_dir, u_traj,
                                     method_clustering_conformations, parameters_clustering_conformations,
                                     split_trajectory,cutoff_proba_conformations,strucfile,trajfile, selected_resids,cutoff_len_states, cluster_of_coordinates_to_process):
    """
    Extracts representative conformations from trajectory data based on hierarchical clustering.
    
    Parameters:
    -----------
    output_dir : str
        Path to output directory containing clustering and discretization data.
    u_traj : MDAnalysis.Universe
        Trajectory universe for accessing conformational frames.
    times_indices : ndarray
        Mapping of trajectory time steps to frame indices.
    Z_parameter_conformations : float
        Z parameter for density peaks clustering of conformations.
    halo_parameter_conformations : int
        Halo parameter for density peaks clustering of conformations.
    split_trajectory : bool
        Whether to save separate trajectory files for each final cluster.
    cutoff_proba_conformations : float
        Minimum probability threshold for a conformation to be considered significant.
    strucfile : str
        Path to the structure file (e.g., PDB) for the trajectory.
    trajfile : str
        Path to the trajectory file (e.g., DCD, XTC).
    selected_resids : list of int
        List of residue IDs to consider for trajectory splitting.
    cutoff_len_states : int
        Maximum number of unique states to consider in each cluster.
    cluster_of_coordinates_to_process : int
        Index of the cluster of coordinates to process (if applicable).
    """
    times_indices = np.load(output_dir + "discretizing_npy/times_indices.npy")  # Load time indices for frames
    # Load top-level cluster assignments
    cluster_labels = np.load(os.path.join(output_dir, "analysis_npy", "cluster_labels.npy"))

    # Load selected coordinates and the discretized representation
    coordinates, X_cuts, Labels = load_data_discretization(output_dir + "selected_coordinates.txt")
    discretized_array = np.load(output_dir + "discretizing_npy/discretized_array.npy")

    logging.info("\nExtracting conformations from clusters...")

    # Split the discretized array based on top-level clustering clustering
    clusters_data = split_discretized_array_by_clusters(discretized_array, cluster_labels)
    logging.info(f"Found {len(clusters_data)} clusters based on clustering labels.")

    # Extract unique conformational states and their probabilities within each cluster
    logging.info("Extracting unique states from clusters...")
    unique_states_clusters, probabilities_unique_states_clusters = get_unique_states_in_splitted_array(clusters_data,cutoff_len_states, cluster_of_coordinates_to_process)

    cumulative_proba = [np.sum(probabilities_unique_states_clusters[i]) for i in range(len(probabilities_unique_states_clusters))]
    cumulative_proba = np.array(cumulative_proba)
    logging.info(f"Total probability of unique states under cutoff_len_states in each cluster: {cumulative_proba}")
    
    # Compute pairwise distances between unique states inside each cluster
    logging.info(f"Computing distances between unique states in each cluster...")
    distances_between_states = compute_distances_between_states(unique_states_clusters,cluster_of_coordinates_to_process)
    
    all_clusters_labels = []
    for i, dist_states in enumerate(distances_between_states):
        if cluster_of_coordinates_to_process>=0 and i != cluster_of_coordinates_to_process:
            all_clusters_labels.append([])
            logging.info(f'Skip cluster {i} as it is not the one to process.')
            continue
        
        logging.info(f"Cluster {i}: Found {len(unique_states_clusters[i])} unique states.")    

        n_unique_states = len(unique_states_clusters[i])
        if n_unique_states <= 2**5:
            cluster_labels = np.array([i for i in range(n_unique_states)])
        else :
            cluster_labels = cluster_distances(dist_states, method_clustering_conformations, parameters_clustering_conformations)

        
        # Plot and save the clustering results for this sub-cluster
        _ = plot_clustering_results(
            dist_states, cluster_labels,
            output_dir + 'conformations_clustering/',
            f"distances_between_states_cluster_{i}",
            label_data="Normalized distance between states",
            xlabel="State Index",
            ylabel="State Index"
        )
        all_clusters_labels.append(cluster_labels)

    # Compute probabilities for each conformation cluster (after second-level clustering)
    proba_clusters = []
    for i, cluster_labels in enumerate(all_clusters_labels):
        
        if cluster_of_coordinates_to_process>=0 and i != cluster_of_coordinates_to_process:
            proba_clusters.append([])
            continue
        unique_labels = np.unique(cluster_labels)
        proba_conformations = np.zeros(len(unique_labels), dtype=float)

        for j, label in enumerate(cluster_labels):
            ind_label = np.where(unique_labels == label)[0][0]
            proba_conformations[ind_label] += probabilities_unique_states_clusters[i][j]
        #select probabilities larger than 0.001
        selected_unique_labels = unique_labels[proba_conformations > cutoff_proba_conformations]
        selected_proba_conformations = proba_conformations[proba_conformations > cutoff_proba_conformations]

        logging.info(f"Conformations in cluster {i}: {selected_unique_labels}        -1 indicates noise")
        
        logging.info("Probabilities of conformations: %s", 
                    ["%.3f" % p for p in selected_proba_conformations])
        logging.info("Total probability: %.3f" % np.sum(selected_proba_conformations))
        proba_clusters.append(proba_conformations)
        
    # Extract the most probable states from each cluster 
    logging.info("\nComputing most probable states in each cluster...")
    most_probable_states, proba_most_probable_states = get_most_probable_states(
        all_clusters_labels, unique_states_clusters, probabilities_unique_states_clusters, cutoff_proba_conformations,cluster_of_coordinates_to_process
    )

    # Write representative conformations to file
    write_conformations_to_file(all_clusters_labels,most_probable_states, proba_most_probable_states, proba_clusters, output_dir, cutoff_proba_conformations,cluster_of_coordinates_to_process)
    logging.info("Conformations written to file.")

    # Extract original frame indices from final conformation labels
    frames_by_clusters = extract_frames_from_labels(
        output_dir, clusters_data, unique_states_clusters, all_clusters_labels, times_indices,proba_clusters,cutoff_proba_conformations,cluster_of_coordinates_to_process
    )

    # Optionally split trajectory files for each conformation cluster
    if split_trajectory:
        split_trajectory_by_conformations(output_dir, u_traj, frames_by_clusters,proba_clusters,cutoff_proba_conformations,all_clusters_labels,strucfile,trajfile,selected_resids,cluster_of_coordinates_to_process)


