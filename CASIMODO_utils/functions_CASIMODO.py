import os
import shutil
import logging
from datetime import datetime
import io
from contextlib import redirect_stdout, redirect_stderr

import numpy as np	

from scipy.stats import t
from scipy.spatial.distance import pdist, squareform

from sklearn.neighbors import KernelDensity

from scipy.sparse.csgraph import connected_components

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap, BoundaryNorm

import MDAnalysis as mda 

import matplotlib
matplotlib.use('Agg')

from matplotlib.colors import LinearSegmentedColormap

###################### INITIATE LOGGING #####################
def initiate_logging(config,basename='casimodo'):
    """
    Initializes logging to a file in the specified output directory.

    Parameters:
    - output_dir (str): The directory where the log file will be created.
    - step_to_perform (str): The step being performed, used for logging context.
    Returns:
    - None
    """
    now= datetime.now()
    step_to_perform = config['step_to_perform']
    output_dir = config['output_dir']
    if 'community_to_process' in config.keys():
        community_to_process = config['community_to_process']
    if step_to_perform == 'all':
        log_file = os.path.join(output_dir, f'{basename}.log')
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(message)s',     
            filemode='w' 
        )
        logging.info("Logging initiated. Log file created at: %s", log_file)
        logging.info("Start time: %s", now.strftime("%Y-%m-%d %H:%M:%S"))

    elif step_to_perform == "discretize_local_variables" :
        log_file = os.path.join(output_dir, f'{basename}_rediscretize_local_variables.log')
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(message)s',     
            filemode='w' 
        )
        logging.info("Logging initiated. Log file created at: %s", log_file)
        logging.info("Start time: %s", now.strftime("%Y-%m-%d %H:%M:%S"))
    
    elif step_to_perform == "cluster_local_variables" :
        log_file = os.path.join(output_dir, f'{basename}_recluster_local_variables.log')
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(message)s',     
            filemode='w' 
        )
        logging.info("Logging initiated. Log file created at: %s", log_file)
        logging.info("Start time: %s", now.strftime("%Y-%m-%d %H:%M:%S"))
    
    elif step_to_perform == "get_conformations" :
        log_file = os.path.join(output_dir, f'{basename}_recluster_conformations.log')
        if community_to_process >=0:
            log_file = os.path.join(output_dir, f'{basename}_recluster_conformations_community_{community_to_process}.log')
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(message)s',     
            filemode='w' 
        )
        logging.info("Logging initiated. Log file created at: %s", log_file)
        logging.info("Start time: %s", now.strftime("%Y-%m-%d %H:%M:%S"))
    
    elif step_to_perform == "plot_conformations_time" :
        log_file = os.path.join(output_dir, f'{basename}_plot_conformations_time.log')
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(message)s',     
            filemode='w' 
        )
        logging.info("Logging initiated for step: %s. Log file updated at: %s", step_to_perform, log_file)
        logging.info("Start time: %s", now.strftime("%Y-%m-%d %H:%M:%S"))

    else :
        log_file = os.path.join(output_dir, f'{basename}.log')
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
def print_header(header_file="CASIMODO_utils/header_casimodo.txt"):
    with open(header_file, encoding="utf-8") as f:
        header = f.read()
    logging.info(header)


####################### PRINT INPUTS #####################
def print_inputs(config):
    logging.info("\n")
    logging.info("========= INPUT PARAMETERS =========")
    for key, value in config.items():
        logging.info(f"{key}: {value}")
    logging.info("===================================")
    

####################### PRINT ENDING MESSAGE #####################
def print_ending_message(config):
    output_dir = config['output_dir']
    step_to_perform = config['step_to_perform']
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
        if row[0] != '#':
            data.append([x for x in row.split()])
    return data, lines_file

def open_data_local_variable(namefile):
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

def load_data_discretization(namefile):

    # Read file content (assumes open_file returns parsed data and raw lines)
    data_discretization, lines_discretization = open_file(namefile)

    local_variables = [row[0] for row in data_discretization if len(row) >= 1 and row[0][0] != '#']
    X_cuts = []  # To hold lists of cut points for each local_variable
    labels = []  # To hold lists of region labels

    for row in data_discretization:
        xcut_i = []
        labels_i = []

        # Process alternating cut-point and label values (starting from column 1)
        if len(row) <= 1:
            X_cuts.append(xcut_i)
            labels.append(labels_i)
            continue  # Skip rows without cut points and labels
        for idx in range(1, len(row)):
            value = row[idx]
            if value[0]=='#':
                break  # Stop processing at comment            
            if idx % 2 == 0:
                xcut_i.append(float(value))   # Even-indexed
            else:
                labels_i.append(int(value))   # Odd-indexed 
            

        X_cuts.append(xcut_i)
        labels.append(labels_i)

    return local_variables, X_cuts, labels

def get_multiplicities(discretized_array):
    # Get the shape of the input array: number of rows (frames) and columns (local_variables/features)
    nframes, ncoord = np.shape(discretized_array)
    
    # Initialize an array to hold the multiplicity (number of unique values) for each column
    multiplicities = np.zeros((ncoord), dtype=np.int32)
    
    # Loop over each column (local_variable/feature)
    for i in range(ncoord):
        # Count the number of unique values in column i and store it in the multiplicities array
        multiplicities[i] = len(np.unique(discretized_array[:, i]))
    
    # Return the array of multiplicities
    return multiplicities


##################### OPENING TRAJECTORY #####################
def open_trajectory(config):
    topolfile = config['topolfile']
    trajfile = config['trajfile']
    u_traj = mda.Universe(topolfile, trajfile)
    return u_traj


########################## FILTERING TIMES AND INDICES ##################
def filter_times_and_indices(u_traj,config):
    logging.info("\nFiltering times and indices...")
    times_selected = []
    frames_selected = []
    previous_progress = -1
    delta_t_traj=u_traj.trajectory.dt
    delta_t_traj=round(delta_t_traj,3)
    always_keep=False
    time_zero = config['time_zero']
    last_time = config['last_time']

    delta_time= config['delta_time']
    output_dir= config['output_dir']
    if delta_time < delta_t_traj:
        always_keep=True
    logging.info(f"Trajectory time step: {delta_t_traj} ps")
    nframes= len(u_traj.trajectory)
    times_from_traj=[delta_t_traj * i for i in range(nframes)]
    for i,ts in enumerate(u_traj.trajectory):
        # Update progress bar
        previous_progress = plot_progress_bar(ts.frame, len(u_traj.trajectory), previous_progress)
        time_ts= times_from_traj[i]
        frame_ts= i

        if (always_keep or min(time_ts% delta_time,abs(delta_time-time_ts% delta_time)) < delta_t_traj/2) and time_ts >= time_zero and (last_time < 0 or time_ts <= last_time):
            times_selected.append(time_ts)
            frames_selected.append(frame_ts)
            continue


    # Complete progress bar
    plot_progress_bar(len(u_traj.trajectory), len(u_traj.trajectory), previous_progress)

    # Convert to NumPy arrays and save
    times_selected = np.array(times_selected)
    frames_selected = np.array(frames_selected)
    np.save(output_dir + 'discretizing_npy/times_selected.npy', times_selected)
    np.save(output_dir + 'discretizing_npy/frames_selected.npy', frames_selected)

    logging.info("Times and indices filtered.")

    times_to_show= times_selected[:10].tolist()+['...']+times_selected[-10:].tolist()
    indices_to_show= frames_selected[:10].tolist()+['...']+frames_selected[-10:].tolist()
    logging.info(f"Filtered times (ps): {times_to_show}")
    logging.info(f"Corresponding frame indices: {indices_to_show}")
    return times_selected, frames_selected


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

def get_important_atoms_MDA(u_traj, config):
    logging.info("\nGetting important atoms...")
    important_atoms_dic = config['dic']
    step_to_perform = config['step_to_perform']
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
        if resid in selected_resids:
            continue  # Skip already processed residues
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

def save_important_atoms(important_atoms, selected_resids, selected_resnames, indices_aa, indices_na_pyrimidine, indices_na_purine,config):
    output_dir = config['output_dir']
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

def load_important_atoms(config):
    output_dir = config['output_dir']
    file_important_atoms = output_dir + 'important_atoms.txt'
    important_atoms = []
    selected_resids = []
    selected_resnames = []
    indices_aa = []
    indices_na_pyrimidine = []
    indices_na_purine = []
    data, lines_file = open_file(file_important_atoms)
    for row in data:
        if len(row) >= 3 :
            resid =int(row[0])
            resname = row[1]
            selected_resids.append(resid)
            selected_resnames.append(resname)
            atoms=[]
            first_column_atom = 2
            last_column_atom = 2
            tag=''
            while last_column_atom < len(row) and row[last_column_atom][-1] != ']':
                last_column_atom += 1
            for i in range(first_column_atom, last_column_atom + 1):
                atom_i = row[i]
                if i > first_column_atom:
                    atom_i = atom_i[1:-2]
                else :
                    atom_i = atom_i[2:-2]
                atoms.append(atom_i)
            
            important_atoms.append(atoms)
            if last_column_atom < len(row) - 1:
                tag = row[last_column_atom + 1]
                if tag == 'AA':
                    indices_aa.append(resid)
                elif tag == 'NA_pyrimidine':
                    indices_na_pyrimidine.append(resid)
                elif tag == 'NA_purine':
                    indices_na_purine.append(resid)
    return important_atoms, selected_resids, selected_resnames, indices_aa, indices_na_pyrimidine, indices_na_purine


############################## PRECOMPUTE POSITIONS OF ATOMS ##################
def precompute_important(u_traj, important_atoms, selected_resids, frames_selected):

    logging.info("\nPrecomputing positions of important atoms...")

    num_residues = len(selected_resids)  # Total number of residues with important atoms
    num_atoms = np.sum([len(important_atoms[i]) for i in range(num_residues)])  # Total number of important atoms

    # Pre-select atom groups for each important atom in each residue to avoid repeated selections
    important_atoms_selection = []
    for i in range(num_residues):
        important_atoms_selection.append([
            u_traj.select_atoms(f"resnum {selected_resids[i]} and name {important_atoms[i][j]}")
            for j in range(len(important_atoms[i]))
        ])

    # Initialize array to store important atom positions:
    # Shape: (total important atoms, number of selected frames, 3 local_variables)
    positions_important_atoms = np.zeros((num_atoms, len(frames_selected), 3))

    # Iterate through selected frames and record positions
    previous_progress = -1
    for k, frame in enumerate(frames_selected):
        u_traj.trajectory[frame]  # Move to the specific frame
        previous_progress = plot_progress_bar(k, len(frames_selected), previous_progress)
        count_step = 0  # Index for placing atoms in the output array
        for i in range(num_residues):
            for j in range(len(important_atoms[i])):
                positions_important_atoms[count_step, k, :] = important_atoms_selection[i][j].positions
                count_step += 1

    # Complete the progress bar
    plot_progress_bar(len(frames_selected), len(frames_selected), previous_progress)
    logging.info("Positions of important atoms precomputed.")

    return positions_important_atoms

def precompute_backbone_protein(u_traj, indices_aa, frames_selected):
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
    Positions_atoms_C = np.zeros((num_residues, len(frames_selected), 3))
    Positions_atoms_N = np.zeros((num_residues, len(frames_selected), 3))
    Positions_atoms_CA = np.zeros((num_residues, len(frames_selected), 3))

    # Iterate through selected frames and record positions
    previous_progress = -1
    for k, frame in enumerate(frames_selected):
        previous_progress = plot_progress_bar(k, len(frames_selected), previous_progress)
        u_traj.trajectory[frame]  # Set trajectory to the specific frame

        for i in range(num_residues):
            Positions_atoms_C[i, k, :] = atom_C_selections[i].positions
            Positions_atoms_N[i, k, :] = atom_N_selections[i].positions
            Positions_atoms_CA[i, k, :] = atom_CA_selections[i].positions

    # Complete progress bar
    plot_progress_bar(len(frames_selected), len(frames_selected), previous_progress)
    logging.info("Positions of protein backbone atoms precomputed.")

    return Positions_atoms_C, Positions_atoms_N, Positions_atoms_CA

def precompute_backbone_nucleic_acids(u_traj, indices_na_pyrimidine, indices_na_purine, frames_selected):
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
    Positions_atoms_P = np.empty((num_residues, len(frames_selected), 3))
    Positions_atoms_P.fill(np.inf)  # Fill with NaN to handle residues without P atom (e.g., first residue)
    Positions_atoms_O5p = np.zeros((num_residues, len(frames_selected), 3))
    Positions_atoms_C5p = np.zeros((num_residues, len(frames_selected), 3))
    Positions_atoms_O4p = np.zeros((num_residues, len(frames_selected), 3))
    Positions_atoms_C4p = np.zeros((num_residues, len(frames_selected), 3))
    Positions_atoms_C3p = np.zeros((num_residues, len(frames_selected), 3))
    Positions_atoms_O3p = np.zeros((num_residues, len(frames_selected), 3))
    Positions_atoms_C1p = np.zeros((num_residues, len(frames_selected), 3))
    Positions_atoms_Nbs = np.zeros((num_residues, len(frames_selected), 3))
    Positions_atoms_Cbs = np.zeros((num_residues, len(frames_selected), 3))





    # Iterate through selected frames and record positions
    previous_progress = -1
    for k, frame in enumerate(frames_selected):
        previous_progress = plot_progress_bar(k, len(frames_selected), previous_progress)
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
    plot_progress_bar(len(frames_selected), len(frames_selected), previous_progress)
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

def get_histogram(times, data, coord_type,delta_y):
    """
    Computes the histogram.

    Parameters:
    - times (array): Time points of the trajectory.
    - data (array): Coordinate values.
    - coord_type (str): Type of local_variable ('distance' or 'angle').

    Returns:

    """
    # Set histogram parameters based on local_variable type
    if coord_type == 'distance':
        xlabel = 'Distance (Å)'
    elif coord_type == 'angle':
        xlabel = 'Angle (°)'
    else:
        raise ValueError(f"Unsupported local_variable type: {coord_type}")

    y_max = max(data)
    y_min = min(data)

    hist, bin_edges = compute_histogram(data, y_min, y_max, delta_y)
    x = (bin_edges[:-1] + bin_edges[1:]) / 2

    return x, hist, xlabel


######################## Functions for discretizing local_variables ########################

def smooth_local_variable(y, delta_y,config):
    smooth_factor = config['smooth_factor']

    # Ensure input is a NumPy array and reshape for sklearn's KDE
    y = np.asarray(y).reshape(-1, 1)

    # Step 1: Fit Gaussian KDE to the input data
    kde = KernelDensity(kernel='gaussian', bandwidth=delta_y)
    kde.fit(y)
    # Step 2: Create an evaluation grid over the range of y
    x_min, x_max = np.min(y), np.max(y)
    x_smooth = np.arange(x_min, x_max, delta_y / smooth_factor).reshape(-1, 1)

    # Step 3: Evaluate the log density on the grid
    log_density = kde.score_samples(x_smooth)
    y_smooth = np.exp(log_density)  # Convert from log-density to density

    # Step 4: Normalize the density so it integrates to 1
    y_smooth /= np.trapz(y_smooth, x_smooth.ravel())
    
    # Return 1D arrays for usability
    return x_smooth.ravel(), y_smooth


def find_minima(x_smooth,y_smooth,config) :
    # === Step 1: Derivative-based extrema detection ===
    dx = x_smooth[1] - x_smooth[0]
    dy = np.gradient(y_smooth, dx)
    d2y = np.gradient(dy, dx)

    # Zero-crossings in the first derivative → potential extrema
    zero_crossings = np.where(np.diff(np.sign(dy)))[0]
    if zero_crossings.size == 0:
        return []

    # Classify extrema based on the sign of the second derivative
    minima_idx = zero_crossings[d2y[zero_crossings] > 0]
    maxima_idx = zero_crossings[d2y[zero_crossings] < 0]

    # Must have both minima and maxima for meaningful regions
    if minima_idx.size == 0 or maxima_idx.size == 0:
        return []

    # === Step 2: Define zones where y_smooth varies slowly ===
    prominence = config['prominence']
    val_cutoff = prominence * np.max(y_smooth)
    zones = []
    start = 0
    for i in range(1, len(y_smooth)):
        if abs(y_smooth[i] - y_smooth[start]) > val_cutoff:
            zones.append((start, i - 1))
            start = i
    if start < len(y_smooth) - 1:
        zones.append((start, len(y_smooth) - 1))

    # === Step 3: Pick the lowest minimum in each zone ===
    selected_idx = []
    for z_start, z_end in zones:
        mask = (minima_idx >= z_start) & (minima_idx <= z_end)
        if not np.any(mask):
            continue
        local_idx = minima_idx[mask]
        best_idx = local_idx[np.argmin(y_smooth[local_idx])]
        selected_idx.append(best_idx)

    if not selected_idx:
        return []

    selected_idx = np.array(selected_idx)

    # === Step 4: Merge close minima separated by shallow barriers ===
    n = len(selected_idx)
    merge_matrix = np.eye(n, dtype=int)

    for i in range(n):
        ind_i = selected_idx[i]
        val_i = y_smooth[ind_i]
        for j in range(i + 1, n):
            ind_j = selected_idx[j]
            val_j = y_smooth[ind_j]
            max_between = np.max(y_smooth[ind_i:ind_j + 1])
            # Merge if barrier between minima is shallow
            if (max_between - val_i < val_cutoff) and (max_between - val_j < val_cutoff):
                merge_matrix[i, j] = merge_matrix[j, i] = 1

    # Find connected minima clusters
    n_comp, labels = connected_components(merge_matrix)
    merged_minima_idx = []
    for comp in range(n_comp):
        group = np.where(labels == comp)[0]
        if len(group) == 1:
            merged_minima_idx.append(selected_idx[group[0]])
        else:
            group_vals = y_smooth[selected_idx[group]]
            merged_minima_idx.append(selected_idx[group[np.argmin(group_vals)]])

    merged_minima_idx = np.array(sorted(merged_minima_idx))

    # === Step 5: Final filtering — keep only sufficiently deep minima ===
    final_minima = []
    for i, idx_min in enumerate(merged_minima_idx):
        idx_before = merged_minima_idx[i - 1] if i > 0 else 0
        idx_after = merged_minima_idx[i + 1] if i < len(merged_minima_idx) - 1 else len(y_smooth) - 1

        max_before = np.max(y_smooth[idx_before:idx_min]) if idx_min > 0 else y_smooth[idx_min]
        max_after = np.max(y_smooth[idx_min:idx_after]) if idx_min < len(y_smooth) - 1 else y_smooth[idx_min]

        depth = min(max_before, max_after) - y_smooth[idx_min]
        if depth >= val_cutoff:
            final_minima.append(x_smooth[idx_min])

    return final_minima

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
    - x_smooth: Array of smoothed x values (e.g., local_variable range).
    - y_smooth: Array of smoothed density values (same length as x_smooth).

    Returns:
    - labels: Array of labels, ranked by peak height within each discretized region.
    """

    # Find indices in x_smooth corresponding to the provided minima
    indexes_minima = [np.where(x_smooth == mini)[0][0] for mini in minima]

    # Define region boundaries: start at 0, go through all minima, end at last index
    all_minima = [0] + indexes_minima + [len(x_smooth) - 1]
    all_minima=sorted(np.unique(all_minima))
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

def save_minima(minima, local_variable, labels, config):
    name_output = config['output_dir'] + "selected_local_variables.txt"

    # Open the output file in append mode
    with open(name_output, 'a') as file_output:
        # Write the local_variable type first
        file_output.write(f'{local_variable} ')
        
        # Write each label-minimum pair
        for i in range(len(minima)):
            file_output.write(f' {labels[i]}')               # Write label
            file_output.write(f' {minima[i]:.3f}')         # Write minimum value with 3 decimal precision

        # Write the final label again (to cover the last interval)
        file_output.write(f' {labels[-1]}\n')  # Newline at the end of the line

def save_local_variable_results(times, distance_to_save, local_variable, config):
    output_dir = config['output_dir']

    # Stack time and local_variable values into two columns
    Time_evolution = np.column_stack((times, distance_to_save))

    # Construct output file path
    output_file = output_dir + "local_variables_data/" + local_variable + ".dat"

    # Save to file with two decimal places, separated by three spaces
    np.savetxt(output_file, Time_evolution, fmt="%.2f   %.2f")


def plot_histogram(x, hist, x_smooth, y_smooth, xlabel, local_variable_name, minima, config):

    fig, ax = plt.subplots()

    # Plot average histogram as a black line
    ax.plot(x, hist, color='black', label='Data')

    # Plot KDE smoothed curve in red
    ax.plot(x_smooth, y_smooth, color='green', lw=2, label='KDE Smoothed')

    # Draw vertical dashed blue lines at each minimum position
    if len(minima)!=0:
        for mini in minima:
            if mini==minima[0]:
                ax.axvline(x=mini, color='blue', linestyle='--', label='Discretization minima')
            else:
                ax.axvline(x=mini, color='blue', linestyle='--')

    # Set axis labels and plot title
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Probability density')
    ax.set_title(local_variable_name)

    # Show legend
    ax.legend()
    output_dir = config['output_dir']
    extension_plots = config['extension_plots']
    resolution_plots = config['resolution_plots']
    # Save plot to specified directory with dpi for quality
    plt.savefig(f'{output_dir}local_variables_plots/{local_variable_name}.{extension_plots}', dpi=resolution_plots)

    # Close the figure to free memory
    plt.close()

def discretize_local_variable(y, local_variable_type, times,local_variable_name, config, labels=None, selected_minima=None):
    
    cutoff_npoints_discretization= config['cutoff_npoints_discretization']
    n_points_per_bin= config['n_points_per_bin']
    min_bin_size_distances= config['min_bin_size_distances']
    min_bin_size_angles= config['min_bin_size_angles']

    # Step 1: Smooth the local_variable distribution using KDE
    y_subset=y.copy()
    times_subset=times.copy()
    n_points=len(y_subset)
    

    if n_points>cutoff_npoints_discretization : 
        subset_indexes=np.linspace(0,n_points-1,cutoff_npoints_discretization).astype(int)
        y_subset= y_subset[subset_indexes]
        times_subset= times_subset[subset_indexes]
    n_points_subset=len(y_subset)

    n_bins=n_points_subset//n_points_per_bin
    delta_y=(max(y_subset)-min(y_subset))/(n_bins)
    if local_variable_type=='distance' and delta_y<min_bin_size_distances:
        delta_y=min_bin_size_distances
    elif local_variable_type=='angle' and delta_y<min_bin_size_angles:
        delta_y=min_bin_size_angles

    x_smooth, y_smooth = smooth_local_variable(y_subset, delta_y,config)


    x,hist, xlabel = get_histogram(
        times_subset, y_subset, local_variable_type, delta_y
    )

    if labels is None and selected_minima is None:
        # Step 3: Detect local minima in the smoothed density (robust to noise) and filter them
        selected_minima = find_minima(x_smooth,y_smooth,config)

        if len(selected_minima) !=0 :
            # Step 4: Generate region labels from the minima
            labels = get_labels_discretization(selected_minima, x_smooth, y_smooth)
    output_dir= config['output_dir']
    output = output_dir + "selected_local_variables.txt"
    save_data= config['save_data']
    save_all_plots= config['save_all_plots']
    
    if len(selected_minima) !=0 :
        # Step 5: Save detected minima and corresponding labels
        save_minima(selected_minima, local_variable_name, labels, config)
        if save_data:
            # Step 6: Save the original local_variable data and metadata
            save_local_variable_results(times, y, local_variable_name, config)

        # Step 7: Plot the histogram with KDE and show detected minima
        plot_histogram(
            x, hist,
            x_smooth, y_smooth,xlabel,
            local_variable_name, selected_minima,
            config
        )
    elif save_all_plots:
        # Even if no minima found, save the plot for reference
        plot_histogram(
            x, hist,
            x_smooth, y_smooth,xlabel,
            local_variable_name, selected_minima,
            config
        )


############################### Compute minimum distance between important atoms ##################
def compute_min_distances(positions_important_atoms, i, j, important_atoms,config):
    

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

    cutoff_distance = config['cutoff_distance']
    proba_under_cutoff_distance = config['proba_under_cutoff_distance']
    # Find the minimal distance for each atom pair across all frames
    proba_under_cutoff= np.zeros((len(atoms_i), len(atoms_j)))
    for k in range(len(atoms_i)):
        for l in range(len(atoms_j)):
            proba_under_cutoff[k, l] = np.mean(distances[k, l] < cutoff_distance)
    # Filter pairs based on the probability of being under the cutoff distance
    valid_pairs = proba_under_cutoff >= proba_under_cutoff_distance
    if not np.any(valid_pairs):
        # If no pairs meet the cutoff probability, return default values
        return [],[],[]
    
    list_distances_to_save = []
    list_atoms_i_to_save = []
    list_atoms_j_to_save = []
    list_proba_under_cutoff_to_save = []
    for k in range(len(atoms_i)):
        for l in range(len(atoms_j)):
            if valid_pairs[k, l]:
                list_distances_to_save.append(distances[k, l])
                list_atoms_i_to_save.append(atoms_i[k])
                list_atoms_j_to_save.append(atoms_j[l])
                list_proba_under_cutoff_to_save.append(proba_under_cutoff[k, l])
    list_distances_to_save = np.array(list_distances_to_save)
    list_atoms_i_to_save = np.array(list_atoms_i_to_save)
    list_atoms_j_to_save = np.array(list_atoms_j_to_save)
    indexes_sorted = np.argsort(list_proba_under_cutoff_to_save)[::-1]  # Sort by probability (descending)
    list_distances_to_save = list_distances_to_save[indexes_sorted]
    list_atoms_i_to_save = list_atoms_i_to_save[indexes_sorted]
    list_atoms_j_to_save = list_atoms_j_to_save[indexes_sorted]

    return list_distances_to_save, list_atoms_i_to_save, list_atoms_j_to_save


############################# process distance pair #############################
def process_distance_pair(i, j, positions_important_atoms, important_atoms, selected_resids, times,config):    
    # Compute the minimal interatomic distance between important atoms of residues i and j
    distances_to_save, atoms_i_to_save, atoms_j_to_save = compute_min_distances(positions_important_atoms, i, j, important_atoms,config)
    if len(distances_to_save)==0:
        return

    for k, y in enumerate(distances_to_save):
        atom_i=atoms_i_to_save[k]
        atom_j=atoms_j_to_save[k]

        local_variable_type = 'distance'      # Type of local_variable being discretized

        # Construct a unique name for this distance local_variable
        local_variable_name = f"{selected_resids[i]}_{atom_i}_{selected_resids[j]}_{atom_j}"
    
        # Discretize the distance time series and update output data structures
        discretize_local_variable(y, local_variable_type, times, local_variable_name,config)
    

####################### Function to compute distances between important atoms for all residue pairs ##########################
def compute_all_distances(important_atoms,selected_resids,positions_important_atoms,times,config):
    num_residues = len(selected_resids)
    total_combinations = num_residues * (num_residues - 1) / 2  # total number of pairs
    count_step = 0

    logging.info("\nComputing distances...")

    # Iterate over all valid residue pairs
    previous_progress = -1  # Initialize progress bar
    for i in range(num_residues - 1):
        for j in range(i + 1, num_residues):
            # Update progress bar
            previous_progress=plot_progress_bar(count_step, total_combinations, previous_progress)
            count_step += 1

            # Process this residue pair
            process_distance_pair(
                i, j,positions_important_atoms,important_atoms,selected_resids,times,config)

    # Finalize progress bar
    plot_progress_bar(total_combinations, total_combinations,previous_progress)
    logging.info("Distances computed and saved.")

########################### Precompute important atom positions ##############################
def precompute_all_positions(u_traj, config):
    output_dir = config['output_dir']
    # Load time points and frame indices previously filtered and saved
    times = np.load(output_dir + 'discretizing_npy/times_selected.npy')
    frames_selected = np.load(output_dir + 'discretizing_npy/frames_selected.npy')

    important_atoms, selected_resids, selected_resnames, indices_aa, indices_na_pyrimidine, indices_na_purine = get_important_atoms_MDA(u_traj, config)
    save_important_atoms(important_atoms, selected_resids, selected_resnames, indices_aa, indices_na_pyrimidine, indices_na_purine, config)

    # Precompute important atom positions across trajectory
    positions_important_atoms = precompute_important(u_traj, important_atoms, selected_resids, frames_selected)

    # Save precomputed positions to disk
    save_positions(positions_important_atoms, output_dir + "discretizing_npy/positions_important_atoms.npy")

    if len(indices_aa)!=0:
        # Precompute backbone atom positions (N, C, and CA atoms)
        Positions_atoms_C, Positions_atoms_N, Positions_atoms_CA = precompute_backbone_protein(
            u_traj, indices_aa, frames_selected
        )

        # Save backbone atom positions to disk for future use
        save_positions(Positions_atoms_C, output_dir + "discretizing_npy/Positions_C_atoms.npy")
        save_positions(Positions_atoms_N, output_dir + "discretizing_npy/Positions_N_atoms.npy")
        save_positions(Positions_atoms_CA, output_dir + "discretizing_npy/Positions_CA_atoms.npy")

    
    if len(indices_na_pyrimidine) != 0 or len(indices_na_purine) != 0 : 
        # Precompute backbone atom positions (N, C, and CA atoms)
        Positions_atoms_P, Positions_atoms_O5p, Positions_atoms_C5p, Positions_atoms_O4p, Positions_atoms_C4p, Positions_atoms_C3p, Positions_atoms_O3p, Positions_atoms_C1p, Positions_atoms_Nbs, Positions_atoms_Cbs = precompute_backbone_nucleic_acids(
            u_traj, indices_na_pyrimidine, indices_na_purine, frames_selected
        )   

        # Save backbone atom positions to disk for future use
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




########################## Function to get the multimodal contacts ################################
def get_contacts(u_traj, config):
    output_dir = config['output_dir']

    important_atoms, selected_resids, selected_resnames, indices_aa, indices_na_pyrimidine, indices_na_purine = load_important_atoms(config)

    # Load time points and frame indices previously filtered and saved
    times = np.load(output_dir + 'discretizing_npy/times_selected.npy')
    frames_selected = np.load(output_dir + 'discretizing_npy/frames_selected.npy')

    # Positions of important atoms
    positions_important_atoms = np.load(output_dir + "discretizing_npy/positions_important_atoms.npy")

    # Compute and process distances between all valid residue pairs
    compute_all_distances(important_atoms, selected_resids, positions_important_atoms,times, config)


########################### Functions to process dihedrals for a single residue ##########################
def load_adjustments_angles(output_dir):
    output_data_adjustement = output_dir + 'angles_adjustments.txt'
    data_adjustement,lines_file_adjustement= open_file(output_data_adjustement)
    angle_names_adjusted=[]
    angle_values_adjusted=[]
    cycle_corrections=[]
    for i in range(len(data_adjustement)):
        if len(data_adjustement[i])<1 or data_adjustement[i][0].startswith('#'):
            continue
        coord_adjusted= data_adjustement[i][0]
        angle_adjusted= float(data_adjustement[i][1])
        cycle_correction = int(data_adjustement[i][2])
        angle_names_adjusted.append(coord_adjusted)
        angle_values_adjusted.append(angle_adjusted)
        cycle_corrections.append(cycle_correction)
    return angle_names_adjusted,angle_values_adjusted,cycle_corrections
    

def adjust_angle_data(name_angle, data, y_min, y_max, delta_y, config):
    output_dir = config['output_dir']
    output_data_adjustement = output_dir + 'angles_adjustments.txt'
    if not os.path.exists(output_data_adjustement):
        with open(output_data_adjustement, 'w') as f:
            f.write('#Angle_name  Angle_to_put_periodicity 2Pi_to_add\n')
    angle_names_adjusted,angle_values_adjusted,cycle_corrections = load_adjustments_angles(output_dir)
    cycle_correction=0
    
    if name_angle in angle_names_adjusted:
        index_adjusted= angle_names_adjusted.index(name_angle)
        angle_to_adjust= angle_values_adjusted[index_adjusted]
        cycle_correction= cycle_corrections[index_adjusted]

    else : 
        # Wrap data to [0, 360)
        data_wrapped = np.asarray(data) % 360
        # Sort angles once — O(N log N)
        sorted_data = np.sort(data_wrapped)
        n = len(data_wrapped)
        # Compute circular gaps between consecutive sorted values (including wrap-around)
        gaps = np.diff(np.r_[sorted_data, sorted_data[0] + 360])
        # The largest gap indicates the "empty" region (best place to cut)
        i_max_gap = np.argmax(gaps)
        angle_to_adjust = (sorted_data[i_max_gap] + gaps[i_max_gap] / 2.0) % 360
        if angle_to_adjust < -180 :
            angle_to_adjust += 360
        if angle_to_adjust > 180 :
            angle_to_adjust -= 360
        # Adjust data: shift values below cutoff by +360
        adjusted_data = np.where(data < angle_to_adjust, data + 360, data)
        if np.min(adjusted_data) > 180 or np.max(adjusted_data) > 360:
            cycle_correction = -1
        if np.max(adjusted_data) < -180 or np.min(adjusted_data) < -360:
            cycle_correction = 1
            # Save adjustment info
        with open(output_data_adjustement, 'a') as f:
            f.write(f'{name_angle}  {angle_to_adjust}  {cycle_correction}\n')
        
    adjusted_data = np.where(data < angle_to_adjust, data + 360, data) + 360*cycle_correction
    return adjusted_data, adjusted_data.max(), adjusted_data.min()

    

def process_dihedral_i_protein(i, Positions_atoms_C, Positions_atoms_N, Positions_atoms_CA, indices_aa,times, config):
    local_variable_type = 'angle'

    # Initialize empty arrays (optional, overwritten later)
    phi_angle = np.zeros(len(times))
    psi_angle = np.zeros(len(times))

    # Process phi dihedral if previous residue exists and backbone geometry is valid
    if i > 0:
        distance_C_N = np.linalg.norm(Positions_atoms_C[i - 1, 0, :] - Positions_atoms_N[i, 0, :])
        if distance_C_N < 2:
            local_variable_name = f"phi_{indices_aa[i]}"
            # Calculate phi dihedral angles (radians) and convert to degrees
            phi_angle = np.rad2deg(mda.lib.distances.calc_dihedrals(Positions_atoms_C[i - 1, :, :],Positions_atoms_N[i, :, :],Positions_atoms_CA[i, :, :],Positions_atoms_C[i, :, :])            )
            # Adjust angles if range spans more than 180 degrees (unwrap circular data)
            phi_angle, _, _ = adjust_angle_data(local_variable_name,phi_angle, np.min(phi_angle), np.max(phi_angle), 4,config)
            # Discretize the phi angle data for further analysis
            discretize_local_variable(phi_angle, local_variable_type,times, local_variable_name,config)

    # Process psi dihedral if next residue exists and backbone geometry is valid
    if i < len(Positions_atoms_C) - 1:
        distance_N_C = np.linalg.norm(Positions_atoms_N[i + 1, 0, :] - Positions_atoms_C[i, 0, :])
        if distance_N_C < 2:
            local_variable_name = f"psi_{indices_aa[i]}"
            # Calculate psi dihedral angles (radians) and convert to degrees
            psi_angle = np.rad2deg(mda.lib.distances.calc_dihedrals(Positions_atoms_N[i, :, :],Positions_atoms_CA[i, :, :],Positions_atoms_C[i, :, :],Positions_atoms_N[i + 1, :, :]))
            # Adjust angles if range spans more than 180 degrees (unwrap circular data)
            psi_angle, _, _ = adjust_angle_data(local_variable_name,psi_angle, np.min(psi_angle), np.max(psi_angle), 4,config)
            
            # Discretize the psi angle data for further analysis
            discretize_local_variable(psi_angle, local_variable_type,times, local_variable_name,config)
            
def process_dihedral_i_nucleic_acids(i, Positions_atoms_P, Positions_atoms_O5p, Positions_atoms_C5p, Positions_atoms_O4p, Positions_atoms_C4p, Positions_atoms_C3p,
                                     Positions_atoms_O3p, Positions_atoms_C1p, Positions_atoms_Nbs, Positions_atoms_Cbs, indices_na, 
                                    times, config):
    local_variable_type = 'angle'

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
            local_variable_name = f"alpha_{indices_na[i]}"
            # Calculate alpha dihedral angles (radians) and convert to degrees
            alpha_angle = np.rad2deg(mda.lib.distances.calc_dihedrals(Positions_atoms_O3p[i - 1, :, :],Positions_atoms_P[i, :, :],Positions_atoms_O5p[i, :, :],Positions_atoms_C5p[i, :, :]) )
            # Adjust angles if range spans more than 180 degrees (unwrap circular data)
            alpha_angle, _, _ = adjust_angle_data(local_variable_name,alpha_angle, np.min(alpha_angle), np.max(alpha_angle), 4,config)
            
            # Discretize the alpha angle data for further analysis
            discretize_local_variable(alpha_angle, local_variable_type,
                                  times, local_variable_name,config)
            
        local_variable_name = f"beta_{indices_na[i]}"
        # Calculate beta dihedral angles (radians) and convert to degrees
        beta_angle = np.rad2deg(mda.lib.distances.calc_dihedrals(Positions_atoms_P[i, :, :],Positions_atoms_O5p[i, :, :],Positions_atoms_C5p[i, :, :],Positions_atoms_C4p[i, :, :]) )
        # Adjust angles if range spans more than 180 degrees (unwrap circular data)
        beta_angle, _, _ = adjust_angle_data(local_variable_name,beta_angle, np.min(beta_angle), np.max(beta_angle), 4,config)
        
        # Discretize the beta angle data for further analysis
        discretize_local_variable(beta_angle, local_variable_type,times, local_variable_name,config)
    
    local_variable_name = f"gamma_{indices_na[i]}"
    # Calculate gamma dihedral angles (radians) and convert to degrees
    gamma_angle = np.rad2deg(mda.lib.distances.calc_dihedrals(Positions_atoms_O5p[i, :, :],Positions_atoms_C5p[i, :, :],Positions_atoms_C4p[i, :, :],Positions_atoms_C3p[i, :, :]) )
    # Adjust angles if range spans more than 180 degrees (unwrap circular data)
    gamma_angle, _, _ = adjust_angle_data(local_variable_name,gamma_angle, np.min(gamma_angle), np.max(gamma_angle), 4,config)
    
    # Discretize the gamma angle data for further analysis
    discretize_local_variable(gamma_angle, local_variable_type,times, local_variable_name,config)
    
    local_variable_name = f"delta_{indices_na[i]}"
    # Calculate delta dihedral angles (radians) and convert to degrees
    delta_angle = np.rad2deg(mda.lib.distances.calc_dihedrals(Positions_atoms_C5p[i, :, :],Positions_atoms_C4p[i, :, :],Positions_atoms_C3p[i, :, :],Positions_atoms_O3p[i, :, :]) )
    # Adjust angles if range spans more than 180 degrees (unwrap circular data)
    delta_angle, _, _ = adjust_angle_data(local_variable_name,delta_angle, np.min(delta_angle), np.max(delta_angle), 4,config)
    
    # Discretize the delta angle data for further analysis
    discretize_local_variable(delta_angle, local_variable_type, times, local_variable_name,config)
    
    local_variable_name = f"chi_{indices_na[i]}"
    # Calculate chi dihedral angles (radians) and convert to degrees
    chi_angle = np.rad2deg(mda.lib.distances.calc_dihedrals(Positions_atoms_O4p[i, :, :],Positions_atoms_C1p[i, :, :],Positions_atoms_Nbs[i, :, :],Positions_atoms_Cbs[i, :, :]) )
    # Adjust angles if range spans more than 180 degrees (unwrap circular data)
    chi_angle, _, _ = adjust_angle_data(local_variable_name,chi_angle, np.min(chi_angle), np.max(chi_angle), 4,config)
    
    # Discretize the chi angle data for further analysis
    discretize_local_variable(chi_angle, local_variable_type, times, local_variable_name,config)

    # Process psi dihedral if next residue exists and backbone geometry is valid
    if i < len(Positions_atoms_P) - 1:
        distance_O_P = np.linalg.norm(Positions_atoms_O3p[i, 0, :] - Positions_atoms_P[i+1, 0, :])
        if distance_O_P < 2 and  np.any(np.isinf(Positions_atoms_P[i+1, :, :]))==False:
            local_variable_name = f"epsilon_{indices_na[i]}"
            # Calculate psi dihedral angles (radians) and convert to degrees
            epsilon_angle = np.rad2deg(mda.lib.distances.calc_dihedrals(Positions_atoms_C4p[i, :, :],Positions_atoms_C3p[i, :, :],Positions_atoms_O3p[i, :, :],Positions_atoms_P[i + 1, :, :]))
            # Adjust angles if range spans more than 180 degrees (unwrap circular data)
            epsilon_angle, _, _ = adjust_angle_data(local_variable_name,epsilon_angle, np.min(epsilon_angle), np.max(epsilon_angle), 4,config)
            
            # Discretize the psi angle data for further analysis
            discretize_local_variable(epsilon_angle, local_variable_type,times, local_variable_name,config)
            
            local_variable_name = f"zeta_{indices_na[i]}"
            # Calculate psi dihedral angles (radians) and convert to degrees
            zeta_angle = np.rad2deg(mda.lib.distances.calc_dihedrals(Positions_atoms_C3p[i, :, :],Positions_atoms_O3p[i, :, :],Positions_atoms_P[i+1, :, :],Positions_atoms_O5p[i + 1, :, :]))
            # Adjust angles if range spans more than 180 degrees (unwrap circular data)
            zeta_angle, _, _ = adjust_angle_data(local_variable_name,zeta_angle, np.min(zeta_angle), np.max(zeta_angle), 4,config)
            
            # Discretize the psi angle data for further analysis
            discretize_local_variable(zeta_angle, local_variable_type, times,local_variable_name,config)
        
    
########################### Functions to compute dihedrals for all residues ##########################
def compute_all_dihedrals_protein(indices_aa, Positions_atoms_C, Positions_atoms_N, Positions_atoms_CA, times, config):  

    num_residues = len(indices_aa)

    logging.info("\nComputing dihedrals in protein backbone...")
    previous_progress = -1  # Initialize progress bar
    for i in range(num_residues):
        previous_progress=plot_progress_bar(i, num_residues,previous_progress)
        process_dihedral_i_protein(i, Positions_atoms_C, Positions_atoms_N, Positions_atoms_CA, indices_aa, times, config)

    plot_progress_bar(num_residues, num_residues,previous_progress)
    logging.info("Dihedrals computed and saved.")

def compute_all_dihedrals_nucleic_acids(indices_na, Positions_atoms_P, Positions_atoms_O5p, Positions_atoms_C5p, Positions_atoms_O4p, Positions_atoms_C4p, Positions_atoms_C3p, Positions_atoms_O3p, Positions_atoms_C1p, Positions_atoms_Nbs, Positions_atoms_Cbs,times, config):  
    num_residues = len(indices_na)

    logging.info("\nComputing dihedrals in nucleic acids backbone...")
    previous_progress = -1  # Initialize progress bar
    for i in range(num_residues):
        previous_progress=plot_progress_bar(i, num_residues,previous_progress)
        process_dihedral_i_nucleic_acids(i,  Positions_atoms_P, Positions_atoms_O5p, Positions_atoms_C5p, Positions_atoms_O4p, Positions_atoms_C4p, Positions_atoms_C3p, Positions_atoms_O3p, Positions_atoms_C1p, Positions_atoms_Nbs, Positions_atoms_Cbs, indices_na, times, config)

    plot_progress_bar(num_residues, num_residues,previous_progress)
    logging.info("Dihedrals computed and saved.")


########################## Function to get the multimodal dihedrals of protein ################################
def get_dihedrals_protein(u_traj, config):

    important_atoms, selected_resids, selected_resnames, indices_aa, indices_na_pyrimidine, indices_na_purine = load_important_atoms(config)
    if len(indices_aa) < 2:
        logging.info("Not enough amino acids selected for dihedral analysis. Skipping.")
        return
    output_dir = config['output_dir']
    # Load time values and their corresponding frame indices
    times = np.load(output_dir + 'discretizing_npy/times_selected.npy')
    frames_selected = np.load(output_dir + 'discretizing_npy/frames_selected.npy')

    # Load precomputed backbone atom positions (N, C, and CA atoms)
    Positions_atoms_C =np.load( output_dir + "discretizing_npy/Positions_C_atoms.npy")
    Positions_atoms_N =np.load( output_dir + "discretizing_npy/Positions_N_atoms.npy")
    Positions_atoms_CA =np.load( output_dir + "discretizing_npy/Positions_CA_atoms.npy")

    # Step 3: Compute all dihedral angles and write selected features
    compute_all_dihedrals_protein(indices_aa, Positions_atoms_C, Positions_atoms_N, Positions_atoms_CA, times, config)


########################## Function to get the multimodal dihedrals of nucleic acids ################################
def get_dihedrals_nucleic_acids(u_traj, config):

    important_atoms, selected_resids, selected_resnames, indices_aa, indices_na_pyrimidine, indices_na_purine = load_important_atoms(config)
    output_dir = config['output_dir']
    if len(indices_na_pyrimidine) < 1 and len(indices_na_purine) < 1:
        logging.info("No nucleic acids selected for dihedral analysis.")
        return
    
    # Load time values and their corresponding frame indices
    times = np.load(output_dir + 'discretizing_npy/times_selected.npy')
    frames_selected = np.load(output_dir + 'discretizing_npy/frames_selected.npy')

    # Load precomputed backbone atom positions
    Positions_atoms_P = np.load( output_dir + "discretizing_npy/Positions_P_atoms.npy")
    Positions_atoms_O5p = np.load( output_dir + "discretizing_npy/Positions_O5p_atoms.npy")
    Positions_atoms_C5p = np.load( output_dir + "discretizing_npy/Positions_C5p_atoms.npy")
    Positions_atoms_O4p = np.load( output_dir + "discretizing_npy/Positions_O4p_atoms.npy")
    Positions_atoms_C4p = np.load( output_dir + "discretizing_npy/Positions_C4p_atoms.npy")
    Positions_atoms_C3p = np.load( output_dir + "discretizing_npy/Positions_C3p_atoms.npy")
    Positions_atoms_O3p = np.load( output_dir + "discretizing_npy/Positions_O3p_atoms.npy")
    Positions_atoms_C1p = np.load( output_dir + "discretizing_npy/Positions_C1p_atoms.npy")
    Positions_atoms_Nbs = np.load( output_dir + "discretizing_npy/Positions_Nbs_atoms.npy")
    Positions_atoms_Cbs = np.load( output_dir + "discretizing_npy/Positions_Cbs_atoms.npy")


    indices_na= np.sort(indices_na_pyrimidine+indices_na_purine)
    # Step 3: Compute all dihedral angles and write selected features
    compute_all_dihedrals_nucleic_acids(indices_na, Positions_atoms_P, Positions_atoms_O5p, Positions_atoms_C5p, Positions_atoms_O4p, Positions_atoms_C4p, Positions_atoms_C3p, Positions_atoms_O3p, Positions_atoms_C1p, Positions_atoms_Nbs, Positions_atoms_Cbs, times, config)


############################# Function to add new local_variables to the existing discretization ##########################
def add_local_variables(config):
    output_dir = config['output_dir']
    local_variables_to_add = config['local_variables_to_add']
    type_local_variables_to_add = config['type_local_variables_to_add']
    # Load already discretized local_variables
    local_variables, X_cuts, Labels = load_data_discretization(output_dir + "selected_local_variables.txt")

    # Reference time values from the first known local_variable
    data_zero = open_data_local_variable(output_dir + "local_variables_data/" + local_variables[0] + ".dat")
    times_to_compare = data_zero[:, 0]

    logging.info("\nAdding new local variables...")
    for i, coord_file in enumerate(local_variables_to_add):
        data_coord_raw = open_data_local_variable(coord_file)
        local_variable_name = coord_file.split('/')[-1].split('.')[0]
        local_variable_type = type_local_variables_to_add[i]

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
        if local_variable_type == 'angle' :
            y_coord, _, _ = adjust_angle_data(local_variable_name,y_coord, np.min(y_coord), np.max(y_coord), 4,output_dir)

        # Discretize and append this local_variable to selected_local_variables.txt
        discretize_local_variable(y_coord, local_variable_type,times_to_compare, local_variable_name, config)
    logging.info("New local variables added and discretized.")


############################ Function to get the discretized array from saved local_variables ##########################
def get_discretized_array(config):
    # Load local_variable names, discretization cutoffs, and corresponding labels
    output_dir = config['output_dir']
    local_variables, X_cuts, Labels = load_data_discretization(output_dir + "selected_local_variables.txt")

    # Load time information from the first local_variable file (assumes all local_variables share the same time points)
    frames_selected = np.load(output_dir + 'discretizing_npy/frames_selected.npy')

    nframes_to_save = len(frames_selected)
    # Initialize output array to store discrete labels for each frame and local_variable
    data_discretized = np.zeros((nframes_to_save, len(local_variables)), dtype=int)

    logging.info("\nDiscretizing data...")

    # Loop over all selected local_variables
    for i in range(len(local_variables)):
        # Load data for current local_variable
        data_coord = open_data_local_variable(output_dir + "local_variables_data/" + local_variables[i] + ".dat")

        # Loop over all frames
        for f in range (len(frames_selected)):
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
    Compute marginal (single) and joint (double) frequencies for discretized local_variables.

    Parameters
    ----------
    discretized_array : ndarray of shape (n_frames, n_coords)
        The discretized representation of the local_variables.

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

    # Precompute flat indices (offsets) for each local_variable
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

def get_frequencies(config):
    # Load the discretized array from a .npy file located in the specified output directory
    output_dir = config['output_dir']
    discretized_array = np.load(output_dir + "discretizing_npy/discretized_array.npy")
    
    # Compute the single and double frequencies using a helper function 
    single_frequencies, double_frequencies = compute_frequencies(discretized_array)
    
    # Save the computed single frequencies to a file in the 'frequencies' subdirectory
    np.save(output_dir + 'analysis_npy/frequencies_single.npy', single_frequencies)
    
    # Save the computed double frequencies to a file in the 'frequencies' subdirectory
    np.save(output_dir + 'analysis_npy/frequencies_double.npy', double_frequencies)


########################### Function to plot mutual information matrix ##########################
def plot_information(config,Information_matrix,output_dir,name_out,label_data=None):
    """
    Plots the mutual information matrix and saves it as an image.
    Parameters:
    - Information_matrix: information matrix (2D numpy array).
    - output_dir: Directory where the plot will be saved.
    - name_out: Name of the output file (without extension).
    """
    extension_plots = config['extension_plots']
    resolution_plots = config['resolution_plots']
    plt.figure(figsize=(10, 6))
    plt.imshow(Information_matrix, cmap='magma', interpolation='nearest')
    plt.colorbar(label=label_data)
    plt.title(f'{label_data} Matrix')
    plt.xlabel('Local Variable Index')
    plt.ylabel('Local Variable Index')
    plt.tight_layout()
    plt.savefig(output_dir+name_out+f'.{extension_plots}', dpi=resolution_plots)
    plt.close()

def plot_information_clustered(config,Information_matrix, reordered_labels, output_dir, name_out, label_data=None,xlabel=None, ylabel=None):
    """
    Plots the mutual information matrix with boxed cluster boundaries.

    Parameters:
    - Information_matrix: 2D numpy array (mutual information matrix).
    - reordered_labels: List or array of cluster labels (in reordered local_variable order).
    - output_dir: Directory to save the plot.
    - name_out: Output file name (without extension).
    - label_data: Optional string for the colorbar label.
    """
    plt.figure(figsize=(10, 8))
    ax = plt.gca()

    # Plot the information matrix
    im = ax.imshow(Information_matrix, cmap='magma', interpolation='nearest')
    plt.colorbar(im, label=label_data)
    plt.title(f'{label_data} Matrix with Cluster Boxes in green and noise in blue' if label_data else "Clustered Information Matrix")

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
        ecolor='lawngreen'
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

    extension_plots = config['extension_plots']
    resolution_plots = config['resolution_plots']
    plt.savefig(f"{output_dir}/{name_out}.{extension_plots}", dpi=resolution_plots)
    plt.close()

########################## Function to compute mutual information between local_variables ##########################
def get_B_correction(output_dir):
    """
    Compute the bias correction for mutual information estimates.
    Parameters:
    discretized_array : ndarray (n_frames, n_coords)
        The discretized representation of the local_variables.
    multiplicities : array-like of int, shape (n_coords,)
        Number of discrete states (bins) for each local_variable.
    Returns:
    correction : ndarray, shape (n_coords, n_coords)
        The bias correction matrix for mutual information.
    """
    discretized_array=np.load(output_dir+"discretizing_npy/discretized_array.npy")
    single_frequencies=np.load(output_dir+'analysis_npy/frequencies_single.npy')
    double_frequencies=np.load(output_dir+'analysis_npy/frequencies_double.npy')
    multiplicities=get_multiplicities(discretized_array)
    n_frames, n_coords = discretized_array.shape
    B_single = np.zeros((n_coords), dtype=float)
    B_coupled = np.zeros((n_coords, n_coords), dtype=float)
    offsets = np.cumsum([0] + list(multiplicities[:-1]))
    epsilon = 1e-12  # Small constant to prevent log(0)
    
    logging.info("Computing low data bias correction...")
    for i in range(n_coords):
        for xi in range(multiplicities[i]):
            idx_i= offsets[i] + xi
            pi= single_frequencies[idx_i]
            if pi>epsilon :
                B_single[i]+= 1
            for j in range(i, n_coords):
                for xj in range(multiplicities[j]):
                    idx_j= offsets[j] + xj
                    pij= double_frequencies[idx_i, idx_j]
                    if pij>epsilon :
                        B_coupled[i,j]+= 1
            B_coupled[j,i]=B_coupled[i,j]
    logging.info("Low data bias correction computed.")
    return B_single, B_coupled, n_frames

########################## Function to compute entropy  ##########################
def get_entropy(config):
    logging.info("\nComputing entropy...")
    output_dir = config['output_dir']
    discretized_array=np.load(output_dir+"discretizing_npy/discretized_array.npy")
    single_frequencies=np.load(output_dir+'analysis_npy/frequencies_single.npy')
    double_frequencies=np.load(output_dir+'analysis_npy/frequencies_double.npy')
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

    extension_plots = config['extension_plots']
    resolution_plots = config['resolution_plots']
    #plot the entropy values
    plt.figure(figsize=(8, 4))
    plt.bar(range(ncoord), entropy, color='blue', alpha=0.7)
    plt.title('Entropy of Local Variables')
    plt.xlabel('Local Variable Index')
    plt.ylabel('Entropy')
    plt.tight_layout()
    plt.savefig(output_dir + f'information_plots/entropy.{extension_plots}', dpi=resolution_plots)
    plt.close()
    logging.info('\nComputing coupled entropy...')
    offsets = np.cumsum([0] + list(multiplicities[:-1]))
    coupled_entropy=np.zeros((ncoord,ncoord),dtype=float)
    for i in range(ncoord):
        for xi in range(multiplicities[i]):
            idx_i= offsets[i] + xi
            for j in range(ncoord):
                for xj in range(multiplicities[j]):
                    idx_j= offsets[j] + xj
                    proba_xixj=double_frequencies[idx_i, idx_j]
                    if proba_xixj>0:
                        coupled_entropy[i,j]-=proba_xixj*np.log(proba_xixj)
    np.save(output_dir+'analysis_npy/coupled_entropy.npy', coupled_entropy)
    plot_information(config,coupled_entropy, output_dir + 'information_plots/', "coupled_entropy", label_data="Coupled Entropy")   
    logging.info("Coupled Entropy computed.")


######################## Function to compute Rajski distance ##########################
def get_rajski_distance(config):
    output_dir = config['output_dir']    
    logging.info("\nComputing Rajski distance...")
    entropy= np.load(os.path.join(output_dir, "analysis_npy", "entropy.npy"))
    coupled_entropy= np.load(os.path.join(output_dir, "analysis_npy", "coupled_entropy.npy"))
    single_frequencies=np.load(os.path.join(output_dir, "analysis_npy", "frequencies_single.npy"))
    double_frequencies=np.load(os.path.join(output_dir, "analysis_npy", "frequencies_double.npy"))
    B_single, B_coupled, n_frames = get_B_correction(output_dir)
    

    ncoord=len(entropy)
    rajski_distance=np.zeros((ncoord,ncoord),dtype=float)
    for i in range(ncoord):
        entropy_i=entropy[i]+(B_single[i]-1)/(2*n_frames)
        for j in range(i,ncoord):
            entropy_j=entropy[j]+(B_single[j]-1)/(2*n_frames)
            coupled_entropy_ij=coupled_entropy[i,j]+(B_coupled[i,j]-1)/(2*n_frames)
            mutual_information_ij=entropy_i+entropy_j-coupled_entropy_ij
            if coupled_entropy_ij>0:
                rajski_distance[i,j]=1.0 - mutual_information_ij/coupled_entropy_ij
                if rajski_distance[i,j]<0.0:
                    rajski_distance[i,j]=0.0
                if rajski_distance[i,j]>1.0:
                    rajski_distance[i,j]=1.0
            else:
                rajski_distance[i,j]=1.0
            rajski_distance[j,i]=rajski_distance[i,j]
    np.save(os.path.join(output_dir, "analysis_npy", "Rajski_distance.npy"), rajski_distance)

    plot_information(config,rajski_distance, output_dir + 'information_plots/', "rajski_distance",label_data="Rajski Distance between Local Variables")

    logging.info("Rajski distance computed.")


######################### Function to cluster using Advanced Density Peaks ##########################
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

def yacare_clustering(distance_matrix,minimal_size_cluster=10, threshold_variable=0.5,amount_of_noise=0.0,keep_no_noise=1,size_moving_square=10.0):
    # Create a buffer to capture stdout
    buf = io.StringIO()

    # Redirect stdout/stderr to the buffer
    with redirect_stdout(buf), redirect_stderr(buf):
        import yacare
        
        save_images = False
        show_images = False
        percentage_moving_square = size_moving_square*100.0 / distance_matrix.shape[0]  # Percentage of moving square for reordering
        choice_merging_clusters = 3
        keep_no_noise = bool(keep_no_noise)  # Convert to boolean

        variables = yacare.Variables()
        variables.distance_matrix = distance_matrix
        variables.project_name = 'temp_yacare_clustering_CASIMODO'
        variables.show_images = show_images
        variables.save_images = save_images
        variables.function_for_ratio = 2
        
        yacare.perform_first_reordering(variables, percentage_moving_square = percentage_moving_square, vmax = -1)

        yacare.find_optimal_cutoff(variables, minimal_size_cluster = minimal_size_cluster, use_all_cutoff = True, function_for_ratio = 2)
        
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
        logging.info("[YACARE output]\n" + output.strip())
    cluster_labels = np.array([x[1] for x in list_clustered_data_sorted])
    list_sufixes=['_Clustering_Clusters.ndx', '_Clustering_Labels.txt', '_Clustering_Noise.txt','_Clustering_ReorderedElements.txt','_Clustering_RepresentativeStructures.ndx','_Yacare_Summary.txt']
    for sufix in list_sufixes :
        os.remove(variables.project_name + sufix)  # Remove the temporary files created by Yacare
    
    return cluster_labels

def ward_clustering(distance_matrix, max_d=1.0):
    """
    Applies Ward hierarchical clustering on a precomputed distance matrix.
    Parameters
    ----------
    distance_matrix : np.ndarray of shape (n_samples, n_samples)
        Symmetric pairwise distance matrix between conformations or data points.
    max_d : float, default=1.0
        The maximum distance threshold to cut the dendrogram for forming flat clusters.
    Returns
    -------
    cluster_labels : np.ndarray of shape (n_samples,)
        Array of integer cluster labels for each data point.
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    # Ensure condensed distance format
    if distance_matrix.ndim == 2:
        if not np.allclose(np.diag(distance_matrix), 0):
            raise ValueError("Distance matrix diagonal must be zero.")
        distance_matrix = squareform(distance_matrix, checks=False)

    Z = linkage(distance_matrix, method='ward')
    cluster_labels = fcluster(Z, max_d, criterion='distance')

    return cluster_labels

def kmeans_clustering(data, n_clusters=3):
    """
    Applies KMeans clustering on the given data.
    Parameters
    ----------
    data : np.ndarray of shape (n_samples, n_features)
        The input data to cluster.
    n_clusters : int, default=3
        The number of clusters to form.
    random_state : int, default=0
        Random seed for reproducibility.
    Returns
    -------
    cluster_labels : np.ndarray of shape (n_samples,)
        Array of integer cluster labels for each data point.
    """
    from sklearn.cluster import KMeans

    n_clusters = int(n_clusters)
    random_state = 0  # Fixed random state for reproducibility
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    kmeans.fit(data)
      
    return kmeans.labels_

def cluster_distances(distance_matrix, method_clustering, parameters_clustering) :
    if method_clustering == 'hdbscan':
        if len(parameters_clustering) != 3:
            logging.info("HDBSCAN clustering requires exactly three parameters (min_cluster_size, min_samples, cluster_selection_epsilon). No clustering will be performed.")
            cluster_labels = np.arange(distance_matrix.shape[0])  # Assign each point to its own cluster (no clustering)
        else:
            cluster_labels = hdbscan_clustering(distance_matrix, *parameters_clustering)


    elif method_clustering == 'yacare':
        if len(parameters_clustering) != 5:
            logging.info("YACARE clustering requires exactly five parameters (min_cluster_size, threshold_variable, amount_of_noise, keep_no_noise, size_moving_square). No clustering will be performed.")
            cluster_labels = np.arange(distance_matrix.shape[0])  # Assign each point to its own cluster (no clustering)
        else:
            cluster_labels = yacare_clustering(distance_matrix, *parameters_clustering)

    elif method_clustering == 'ward':
        if len(parameters_clustering) != 1:
            logging.info("Ward clustering requires exactly one parameter (threshold). No clustering will be performed.")
            cluster_labels = np.arange(distance_matrix.shape[0])  # Assign each point to its own cluster (no clustering)
        else:
            cluster_labels = ward_clustering(distance_matrix, *parameters_clustering)
    
    elif method_clustering == 'k-means':
        if len(parameters_clustering) != 1:
            logging.info("K-means clustering requires exactly one parameter (n_clusters). No clustering will be performed.")
            cluster_labels = np.arange(distance_matrix.shape[0])  # Assign each point to its own cluster (no clustering)
        else:
            cluster_labels = kmeans_clustering(distance_matrix, *parameters_clustering)

    else:
        logging.info(f"Clustering method '{method_clustering}' not recognized. No clustering will be performed.")
        cluster_labels = np.arange(distance_matrix.shape[0])  # Assign each point to its own cluster (no clustering)

    if type(cluster_labels) == list:
        cluster_labels = np.array(cluster_labels)
    if np.min(cluster_labels)==1 : 
        cluster_labels-=1
    return cluster_labels


############# Function to plot clustering results ##########################
def plot_clustering_results(config,dist_matrix,cluster_labels, output_dir, output_name, label_data=None, xlabel='X-axis', ylabel='Y-axis'):

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
        first_index = np.argmin(mi_sums)
        order= [first_index]
        while len(order) < len(indices):
            last_index = order[-1]
            remaining_indices = list(set(range(len(indices))) - set(order))
            distances_to_last = sub_mi[last_index, remaining_indices]
            next_index = remaining_indices[np.argmin(distances_to_last)]
            order.append(next_index)
        order = indices[order]
        #order = indices[np.argsort(mi_sums)]  # descending
        sorted_indices.extend(order)

    # Add noise at the end
    noise_indices = np.where(cluster_labels == -1)[0]
    sorted_indices.extend(noise_indices)

    reordered_labels= cluster_labels[sorted_indices]

    dist_reordered = dist_matrix[sorted_indices, :][:, sorted_indices]


    plot_information_clustered(config,dist_reordered,reordered_labels, output_dir, output_name, label_data, xlabel=xlabel, ylabel=ylabel)

    
    logging.info("Clustering results plotted and saved.")

    return reordered_labels
    

#################### Function to extract the local_variables in each cluster ##########################
def write_communities_to_file(clusters_ndx,corresponding_labels, local_variables, output_dir, name_output_cluster):

    logging.info("\nWriting communities to file...")
    with open(output_dir + name_output_cluster, 'w') as file_out:
        for i, cluster_i in enumerate(clusters_ndx):
            label_i=corresponding_labels[i]
            if label_i != -1:
                file_out.write(f'[ Community_{i} ]\n')
            else:
                file_out.write(f'[ Noise ]\n')
            for index_coord in cluster_i:
                file_out.write(f'{local_variables[index_coord]} \n')
            file_out.write('\n')

    logging.info("Communities written to file.")

def get_resids_in_communities(clusters_ndx,local_variables,name_local_variables_to_add,name_output,config):
    logging.info("\nGetting resids in communities...")
    output_dir = config['output_dir']
    residues_local_variables_to_add=config['residues_local_variables_to_add']
    file_out=open(output_dir+name_output,'w')
    for i in range (len(clusters_ndx)):
        cluster_i=clusters_ndx[i]
        if i!=len(clusters_ndx)-1:
            file_out.write(f'[ Community_{i} ]\n')
        else:
            file_out.write(f'[ Noise ]\n')
        resids_in_cluster_i=[]
        for j in range(len(cluster_i)):
            index_coord=cluster_i[j]
            coord=local_variables[index_coord]
            if coord in name_local_variables_to_add:
                index_coord_to_add=name_local_variables_to_add.index(coord)
                name_resid_to_add=int(residues_local_variables_to_add[index_coord_to_add].split('_')[0])
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
    logging.info("Getting resids in communities completed.")
    file_out.close()


############ Function to compute information metrics and save results ##############
def compute_information(config):  
    get_frequencies(config)
    get_entropy(config)
    get_rajski_distance(config)


############ Function to cluster local_variables based on mutual information distance, using Advanced Density Peaks ##############
def cluster_local_variables(config):
    output_dir = config['output_dir']
    method_clustering_local_variables = config['method_clustering_local_variables']
    minimal_size_to_cluster = config['minimal_size_to_cluster']
    parameters_clustering_local_variables = config['parameters_clustering_local_variables']


    logging.info(f"\nClustering local variables using {method_clustering_local_variables}...")

    # Load the mutual information distance matrix
    rajski_distance = np.load(os.path.join(output_dir, "analysis_npy", "Rajski_distance.npy"))
    n_local_variables= rajski_distance.shape[0]
    #keep all in one cluster if not enough local_variables to cluster
    if n_local_variables<minimal_size_to_cluster:
        cluster_labels=np.array([0 for j in range(n_local_variables)])
    
    #Apply clustering
    else:
        cluster_labels = cluster_distances(rajski_distance, method_clustering_local_variables, parameters_clustering_local_variables) 

    # Save the cluster labels to a file
    np.save(os.path.join(output_dir, "analysis_npy", "community_labels.npy"), cluster_labels)

    logging.info("Clustering completed and labels saved.")

    reordered_labels = plot_clustering_results(config,rajski_distance,cluster_labels, output_dir+'information_plots/', "rajski_distance_clustering", "Rajski distance between Local Variables",xlabel="Local Variable Index", ylabel="Local Variable Index")
    local_variables,X_cuts,Labels=load_data_discretization(output_dir + "selected_local_variables.txt")

    # Extract clusters and write to file
    clusters_ndx = []
    corresponding_labels=[]

    noise_ndx = np.where(cluster_labels == -1)[0]  # Indices of noise points
    for label in np.unique(cluster_labels):
        if label == -1:  # Noise points
            continue
        cluster_indices = np.where(cluster_labels == label)[0]
        clusters_ndx.append(cluster_indices)    
        corresponding_labels.append(label)
    # Add noise points as a separate cluster
    if len(noise_ndx) > 0:
        clusters_ndx.append(noise_ndx)
        corresponding_labels.append(-1)
    # Write clusters to file
    write_communities_to_file(clusters_ndx,corresponding_labels, local_variables, output_dir, "communities_of_local_variables.txt")
    # Get resids in clusters and write to file
    local_variables_to_add=config['local_variables_to_add']
    name_local_variables_to_add = [coord.split('/')[-1].split('.')[0] for coord in local_variables_to_add]
    get_resids_in_communities(clusters_ndx, local_variables, name_local_variables_to_add, "resids_in_communities_of_LVs.txt",config)


###################### Functions to manipulate states and get conformations ########################
def splt_discretized_array_by_communities(discretized_array, cluster_labels):
    """
    Splits the discretized array into sub-arrays based on cluster labels.

    Parameters:
    -----------
    discretized_array : ndarray
        The discretized representation of the local_variables.
    cluster_labels : ndarray
        The cluster labels for each frame in the discretized array.

    Returns:
    --------
    communities_data : list of ndarray
        A list where each element is a sub-array corresponding to a unique cluster.
    """
    unique_labels = np.unique(cluster_labels)
    communities_data = []

    for label in unique_labels:
        if label == -1:  # Skip noise points
            continue
        indices = np.where(cluster_labels == label)[0]
        communities_data.append(discretized_array[:,indices])

    return communities_data



def get_unique_configurations_in_splitted_array(communities_data,config):
    """
    Extracts unique states from each cluster's discretized data.

    Parameters:
    -----------
    communities_data : list of ndarray
        A list where each element is a sub-array corresponding to a unique cluster.

    Returns:
    --------
    unique_states : list of ndarray
        A list containing unique states for each cluster.
    """
    unique_states = []
    probalities_unique_states = []
    dic_merged_states= []
    cutoff_n_configurations = config['cutoff_n_configurations']
    community_to_process = config['community_to_process']
    for i,cluster_data in enumerate(communities_data):
        if community_to_process>=0 and i != community_to_process:
            probalities_unique_states.append([])
            unique_states.append([]) 
            dic_merged_states.append({})
            continue
        unique_i,count_i= np.unique(cluster_data, axis=0, return_counts=True)
        proba_i= count_i / cluster_data.shape[0]  # Normalize counts to get probabilities
        
        sorted_proba_indices = np.argsort(proba_i)[::-1]  # Sort indices by probability in descending order
        
        unique_i = unique_i[sorted_proba_indices]  # Sort unique states by their probabilities
        proba_i = proba_i[sorted_proba_indices]  # Sort counts accordingly
        
        unique_i_selected = unique_i[:cutoff_n_configurations]  # Keep only states up to the cutoff
        proba_i_selected = proba_i[:cutoff_n_configurations]

        dic_configurations_selected = {tuple(state): [tuple(state)] for state in unique_i_selected}

        
        # Merge non-selected states into closest selected states
        n_non_selected_states = len(unique_i) - cutoff_n_configurations
        if n_non_selected_states > 0:
            logging.info(f"Cluster {i}: Merging {n_non_selected_states} low-probability configurations into closest selected configurations.")
            previous_progress = -1
            for j in range(cutoff_n_configurations, len(unique_i)):
                previous_progress = plot_progress_bar(j - cutoff_n_configurations, n_non_selected_states, previous_progress)
                state_to_assign = unique_i[j]
                dists = np.sum(state_to_assign != unique_i_selected, axis=1)  # Hamming distances to selected states
                closest_index = np.argmin(dists)
                proba_i_selected[closest_index] += proba_i[j]  # Merge probability into closest selected state
                dic_configurations_selected[tuple(unique_i_selected[closest_index])].append(tuple(state_to_assign))  # Merge configuration into closest selected state
            previous_progress = plot_progress_bar(n_non_selected_states, n_non_selected_states, previous_progress)

        probalities_unique_states.append(proba_i_selected) 
        unique_states.append(unique_i_selected)
        dic_merged_states.append(dic_configurations_selected)
    return unique_states, probalities_unique_states,dic_merged_states

def compute_distances_between_configurations(states,config):
    """
    Computes pairwise distances between unique states.

    Parameters:
    -----------
    states : list of ndarray
        A list where each element is an array of unique states for a cluster.
    community_to_process : int
        Index of the cluster to process (if > 0, only this cluster is processed).

    Returns:
    --------
    distances : list of ndarray
        A list containing distance matrices for each cluster's unique states.
    """
    distances = []
    community_to_process = config['community_to_process']
    for i,state in enumerate(states):
        if community_to_process>=0 and i != community_to_process:
            distances.append([])
            continue

        dist_matrix = squareform(pdist(state, metric='hamming'))
        if np.max(dist_matrix) != 0:
            dist_matrix =dist_matrix/ np.max(dist_matrix)  

        distances.append(dist_matrix)
    return distances

def extract_frames_from_labels(clusters_data, unique_configurations, all_clusters_labels, frames_selected, proba_clusters, config,dic_merged_states):
    logging.info("Extracting frames for conformational states...")

    frames_by_clusters = []
    community_to_process = config['community_to_process']
    cutoff_proba_conformations = config['cutoff_proba_conformations']
    output_dir = config['output_dir']
    for i, cluster_labels in enumerate(all_clusters_labels):
        
        if community_to_process >= 0 and i != community_to_process:
            frames_by_clusters.append([])
            continue
        
        logging.info(f"Processing conformations for community {i}...")     
        unique_labels = np.unique(cluster_labels)
        nb_conformations = len(unique_labels)
        label_map = {label: idx for idx, label in enumerate(unique_labels)}

        config_to_index = {
            tuple(state): idx for idx, state in enumerate(unique_configurations[i])
        }

        frames_conformations = [[] for _ in range(len(unique_labels))]
        dict_i = dic_merged_states[i]

        # Build reverse mapping: state -> key
        state_to_key = {
            state: key
            for key, states in dict_i.items()
            for state in states
        }

        for t, state in enumerate(clusters_data[i]):
            key_state = state_to_key.get(tuple(state))
            if key_state is None:
                continue

            index_state = config_to_index.get(key_state)
            if index_state is None:
                continue

            label_index = label_map[cluster_labels[index_state]]
            frames_conformations[label_index].append(frames_selected[t])
        

        frames_by_clusters.append(frames_conformations)
        
         # Check if there are enough conformations with high probability
        count_large_proba =len(np.where(proba_clusters[i] >= cutoff_proba_conformations)[0])
        if count_large_proba <= 1:
            logging.warning(f"Community {i} has no several conformations to process.")
            continue
        
        # Open output file for current cluster
        output_file = open(f"{output_dir}conformational_states_clustering/frames_conformations_from_community_{i}.ndx", 'w')

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

def split_trajectory_by_conformations(u_traj, frames_by_clusters,proba_clusters,all_clusters_labels,selected_resids, config):
    
    output_dir = config['output_dir']
    community_to_process = config['community_to_process']
    cutoff_proba_conformations = config['cutoff_proba_conformations']
    topolfile = config['topolfile']
    trajfile = config['trajfile']
    extension_topol = topolfile.split('.')[-1]
    extension_traj = trajfile.split('.')[-1]

    logging.info("\nSplitting trajectory by conformations...")

    atoms_selected = u_traj.select_atoms(f"resnum {' '.join(map(str, selected_resids))}")
    atoms_selected.write(output_dir + "conformational_states_clustering/atoms_selected." + extension_topol)

    for i, frames_conformations in enumerate(frames_by_clusters):
        if community_to_process >= 0 and i != community_to_process:
            logging.info(f"Skipping community {i} as it is not the one to process.")
            continue
        logging.info(f"Processing community {i}...")

        count_large_proba =len(np.where(proba_clusters[i] >= cutoff_proba_conformations)[0])
        if count_large_proba <= 1:
            logging.warning(f"Community {i} has no several conformations to process.")
            continue

        # Create directory for storing split trajectories from current cluster
        cluster_output_dir = os.path.join(output_dir, f"conformational_states_clustering/splitted_trajectory_community_{i}")
        if os.path.exists(cluster_output_dir):
            shutil.rmtree(cluster_output_dir)  # Remove existing directory            
        os.mkdir(cluster_output_dir)
        unique_labels = np.unique(all_clusters_labels[i])
    
        for j, frames in enumerate(frames_conformations):
            
            proba_conf = proba_clusters[i][j]
            
            if len(frames) == 0 or proba_conf < cutoff_proba_conformations or unique_labels[j] == -1 :
                continue  # Skip empty frames or low-probability conformations or noise
            
            logging.info(f"Writing conformation {unique_labels[j]} for community {i} with probability {proba_conf:.2f}...")

            # Define output file path for current conformation
            output_file = os.path.join(
                cluster_output_dir, f"community_{i}_conformation_{unique_labels[j]}.{extension_traj}"
            )

            # Write selected frames to new trajectory file
            atoms_selected = u_traj.select_atoms(f"resnum {' '.join(map(str, selected_resids))}")
            atoms_selected.write(output_file, frames=frames)
    logging.info("Trajectory splitting completed.")

def get_most_probable_configurations(all_clusters_labels, unique_configurations, probabilities_unique_configurations,config):
    most_probable_configurations = []
    proba_most_probable_configurations = []
    cutoff_proba_conformations = config['cutoff_proba_conformations']
    community_to_process = config['community_to_process']
    # Loop through each main cluster
    for i, cluster_labels in enumerate(all_clusters_labels):
        if community_to_process >= 0 and i != community_to_process:
            most_probable_configurations.append([])
            proba_most_probable_configurations.append([])
            continue
        most_probable_configurations_cluster = []
        proba_most_probable_configurations_cluster = []

        # Get unique conformation labels in current cluster
        unique_labels = np.unique(cluster_labels)

        # Find indices of unique states that belong to each conformation label
        ind_labels_cluster = [
            np.where(cluster_labels == label)[0] for label in unique_labels
        ]

        # Loop through conformations (clustering sub-clusters)
        for j, ind_labels in enumerate(ind_labels_cluster):
            # Get the probabilities of the states in the current conformation
            proba_cluster_conf_j = probabilities_unique_configurations[i][ind_labels]

            # Identify the state with the highest probability
            ind_max_proba = ind_labels[np.argmax(proba_cluster_conf_j)]

            # Save the most probable state and its probability
            most_probable_configurations_cluster.append(unique_configurations[i][ind_max_proba])
            proba_most_probable_configurations_cluster.append(
                probabilities_unique_configurations[i][ind_max_proba]
            )

            if np.sum(proba_cluster_conf_j) > cutoff_proba_conformations :
                # Log the result for tracking
                if unique_labels[j] != -1 :
                    logging.info(
                        f"Most probable configuration in community {i}, conformation {unique_labels[j]}: "
                        f"{unique_configurations[i][ind_max_proba]} "
                        f"with probability {probabilities_unique_configurations[i][ind_max_proba]}"
                    )
        # Append results for the current cluster
        most_probable_configurations.append(most_probable_configurations_cluster)
        proba_most_probable_configurations.append(proba_most_probable_configurations_cluster)

    return most_probable_configurations, proba_most_probable_configurations

def get_local_variables_in_clusters(config): 
    output_dir = config['output_dir']
    file_clusters = open(output_dir + "communities_of_local_variables.txt", 'r')
    local_variables_communities = []  # List to hold local_variables per cluster
    current_cluster = []  # Temporarily store local_variables for current cluster

    for line in file_clusters:
        line = line.strip()

        # Start of a new cluster section
        if line.startswith("[ Community") or line.startswith("[ Noise"):
            # Save the previous cluster if it had any local_variables
            if len(current_cluster) > 0:
                local_variables_communities.append(current_cluster)
                current_cluster = []  # Reset for the next cluster

        elif line:
            # Line contains a local_variable name, add to current cluster
            current_cluster.append(line)

    # Don't forget to append the last cluster if not empty
    if len(current_cluster) > 0:
        local_variables_communities.append(current_cluster)

    return local_variables_communities

def write_conformations_to_file(all_cluster_labels,most_probable_configurations, proba_most_probable_configurations, proba_clusters, config):
    local_variables_communities = get_local_variables_in_clusters(config)  # Get local_variable names (CVs) associated with each cluster
    logging.info("\nWriting conformations to file...")

    # Open the output file for writing
    # Loop over clusters
    community_to_process = config['community_to_process']
    output_dir = config['output_dir']
    cutoff_proba_conformations = config['cutoff_proba_conformations']
    for i, community_configurations in enumerate(most_probable_configurations):
        if community_to_process >= 0 and i != community_to_process:
            continue
        with open(output_dir + f"conformational_states_clustering/conformations_community_{i}.txt", 'w') as file_out:
            
            file_out.write(f"[ Community_{i} ]\n\n")
            unique_cluster_labels = np.unique(all_cluster_labels[i])
            # Loop over conformations within the cluster
            for j, state in enumerate(community_configurations):

                if unique_cluster_labels[j]==-1 or proba_clusters[i][j] < cutoff_proba_conformations:
                    continue
                file_out.write(f"Conformation {unique_cluster_labels[j]} - Probability: {proba_clusters[i][j]:.5f}\n")
                file_out.write(f"Most probable configuration: {state}\n")
                file_out.write(f"Probability of the most probable configuration: {proba_most_probable_configurations[i][j]:.5f}\n")
                file_out.write("Discretized values:\n")

                # Write local_variable name and value
                for k, coord in enumerate(state):
                    file_out.write(f"{local_variables_communities[i][k]}: {coord}\n")
                file_out.write('\n')  # Blank line between conformations

            file_out.write('\n')  # Blank line between clusters


######################### Function to extract conformations from clusters ##########################
def get_conformations_for_communities(u_traj,config):
    output_dir = config['output_dir']

    method_clustering_conformations = config['method_clustering_conformations']
    parameters_clustering_conformations = config['parameters_clustering_conformations']
    community_to_process = config['community_to_process']
    minimal_size_to_cluster = config['minimal_size_to_cluster']
    cutoff_proba_conformations = config['cutoff_proba_conformations']
    split_trajectory = config['split_trajectory']

    frames_selected = np.load(output_dir + "discretizing_npy/frames_selected.npy")  # Load time indices for frames

    # Load top-level cluster assignments
    cluster_labels = np.load(os.path.join(output_dir, "analysis_npy", "community_labels.npy"))

    # Load selected local_variables and the discretized representation
    local_variables, X_cuts, Labels = load_data_discretization(output_dir + "selected_local_variables.txt")
    discretized_array = np.load(output_dir + "discretizing_npy/discretized_array.npy")

    logging.info("\nExtracting conformations for communities...")

    # Split the discretized array based on top-level clustering 
    communities_data = splt_discretized_array_by_communities(discretized_array, cluster_labels)
    logging.info(f"Found {len(communities_data)} communities based on clustering labels.")

    # Extract unique conformational states and their probabilities within each cluster
    logging.info("Extracting unique configurations for communities...")
    unique_configurations, probabilities_unique_configurations,dic_merged_states = get_unique_configurations_in_splitted_array(communities_data,config)
    cumulative_proba = [np.sum(probabilities_unique_configurations[i]) for i in range(len(probabilities_unique_configurations))]
    cumulative_proba = np.array(cumulative_proba)
    logging.info(f"Total probability of unique configurations under cutoff_n_configurations in each cluster: {cumulative_proba}")
    
    # Compute pairwise distances between unique states inside each cluster
    logging.info(f"Computing distances between unique configurations in each community...")
    distances_between_configurations = compute_distances_between_configurations(unique_configurations,config)
    
    all_clusters_labels = []
    for i, dist_states in enumerate(distances_between_configurations):
        if community_to_process>=0 and i != community_to_process:
            all_clusters_labels.append([])
            logging.info(f'Skip cluster {i} as it is not the one to process.')
            continue
        
        logging.info(f"Cluster {i}: Found {len(unique_configurations[i])} unique configurations.")    

        n_unique_states = len(unique_configurations[i])
        # split into each state if not enough states to cluster
        if n_unique_states < minimal_size_to_cluster :
            cluster_labels = np.array([j for j in range(n_unique_states)])
        # Apply clustering to the distance matrix of unique states
        else :
            cluster_labels = cluster_distances(dist_states, method_clustering_conformations, parameters_clustering_conformations)

        
        # Plot and save the clustering results for this sub-cluster
        _ = plot_clustering_results(config,
            dist_states, cluster_labels,
            output_dir + 'conformational_states_clustering/',
            f"distances_between_configurations_community_{i}",
            label_data="Normalized Hamming distance between configurations",
            xlabel="Unique Configuration Index",
            ylabel="Unique Configuration Index"
        )
        all_clusters_labels.append(cluster_labels)
    
    # Compute probabilities for each conformation cluster (after second-level clustering)
    proba_clusters = []
    for i, cluster_labels in enumerate(all_clusters_labels):
        
        if community_to_process>=0 and i != community_to_process:
            proba_clusters.append([])
            continue
        unique_labels = np.unique(cluster_labels)
        proba_conformations = np.zeros(len(unique_labels), dtype=float)

        for j, label in enumerate(cluster_labels):
            ind_label = np.where(unique_labels == label)[0][0]
            proba_conformations[ind_label] += probabilities_unique_configurations[i][j]
        #select probabilities larger than 0.001
        selected_unique_labels = unique_labels[proba_conformations > cutoff_proba_conformations]
        selected_proba_conformations = proba_conformations[proba_conformations > cutoff_proba_conformations]

        logging.info(f"Conformations for community {i}: {selected_unique_labels}        -1 indicates noise")
        
        logging.info("Probabilities of conformations: %s", 
                    ["%.3f" % p for p in selected_proba_conformations])
        logging.info("Total probability: %.3f" % np.sum(selected_proba_conformations))
        proba_clusters.append(proba_conformations)
        
    # Extract the most probable states from each cluster 
    logging.info("\nComputing most probable configuration for each community...")
    most_probable_configurations, proba_most_probable_configurations = get_most_probable_configurations(all_clusters_labels, unique_configurations, probabilities_unique_configurations,config)

    # Write representative conformations to file
    write_conformations_to_file(all_clusters_labels,most_probable_configurations, proba_most_probable_configurations, proba_clusters, config)
    logging.info("Conformations written to file.")

    # Extract original frame indices from final conformation labels
    frames_by_clusters = extract_frames_from_labels(communities_data, unique_configurations, all_clusters_labels, frames_selected,proba_clusters,config,dic_merged_states)

    # Optionally split trajectory files for each conformation cluster
    if split_trajectory:
        important_atoms, selected_resids, selected_resnames, indices_aa, indices_na_pyrimidine, indices_na_purine = load_important_atoms(config)
        split_trajectory_by_conformations(u_traj, frames_by_clusters,proba_clusters,all_clusters_labels,selected_resids, config)

################### Function to plot conformations as function of time ##########################
def load_conformation_by_frame_from_ndx(ndx_file,n_frames,frames_selected):
    data_ndx,lines_ndx =open_file(ndx_file)
    conformation_by_frame = np.zeros(n_frames) -1
    index_conformation = -1
    for j,l in enumerate(lines_ndx) :
        if l[0]=='[' : 
            index_conformation = int(data_ndx[j][1].split('_')[1])
        elif len(data_ndx[j])>0 :
            for f in data_ndx[j] :
                if int(f) in frames_selected:
                    conformation_by_frame[int(f)-frames_selected[0]] = index_conformation
    return conformation_by_frame

def plot_conformations_as_function_of_time(config):
    logging.info("\nPlotting conformations as a function of time...")
    output_dir = config['output_dir']
    extension_plots = config['extension_plots']
    resolution_plots = config['resolution_plots']
    times_selected = np.load(output_dir + "discretizing_npy/times_selected.npy")  # Load time points
    frames_selected = np.load(output_dir + "discretizing_npy/frames_selected.npy")  # Load frame indices corresponding to time points

    n_frames = len(times_selected)
    cluster_labels = np.load(os.path.join(output_dir, "analysis_npy", "community_labels.npy"))
    unique_labels = np.unique(cluster_labels)
    unique_labels = unique_labels[unique_labels != -1]  # Exclude noise label (-1)
    conformations_for_community = []

    comumunities_to_plot=[]
    for i in unique_labels:
        ndx_file = output_dir + f"conformational_states_clustering/frames_conformations_from_community_{i}.ndx"
        if not os.path.exists(ndx_file):
            logging.warning(f"Ndx file for community {i} not found. Skipping plot.")
            continue
        comumunities_to_plot.append(i)
        conformation_by_frame = load_conformation_by_frame_from_ndx(ndx_file, n_frames, frames_selected)        
        conformations_for_community.append(conformation_by_frame)

    # Plotting
    num_clusters = len(conformations_for_community)
    if num_clusters == 0:
        logging.warning("No conformations found for any community. Skipping plot.")
        return
    max_number_of_colors = max([int(np.max(conf)) for conf in conformations_for_community if len(conf) > 0]) + 1
    #create a colormap with enough colors
    base_colors = plt.cm.magma(np.linspace(0, 1, max_number_of_colors))
    cmap = ListedColormap(base_colors[:max_number_of_colors])
    # Discrete boundaries centered on integer indices
    bounds = np.arange(-0.5, max_number_of_colors + 0.5, 1)
    norm = BoundaryNorm(bounds, cmap.N)

    conformations_for_community_colored =np.copy(conformations_for_community)
    # X-axis uses real times
    extent = [float(times_selected[0]), float(times_selected[-1]), 0, num_clusters]
    values_by_conformation = {}
    for i in range(num_clusters):
        values_by_conformation[i] = {}
        labels_cluster_i = np.unique(conformations_for_community[i])
        n_conformations = len(labels_cluster_i)
        delta_colors = max_number_of_colors / (n_conformations - 1) if n_conformations > 1 else 0
        for j in range(n_conformations):
            values_by_conformation[i][j] = round(delta_colors * j) 
            conformations_for_community_colored[i][conformations_for_community[i] ==labels_cluster_i[j] ] = values_by_conformation[i][j]
    
    plt.figure(figsize=(10, 6))
    for i, cluster_data in enumerate(conformations_for_community_colored):
        # Normalize for this cluster (conformation indices within cluster)
        labels_cluster_i = np.unique(cluster_data)
        labels_cluster_i = labels_cluster_i[labels_cluster_i != -1]  # remove -1
        n_conformations = len(labels_cluster_i)
        if n_conformations > 1:
            delta_colors = 1.0 / (n_conformations - 1)
        else:
            delta_colors = 1.0
        norm = plt.Normalize(vmin=0, vmax=max(n_conformations - 1, 1))
        
        #get a color from tab10 for this cluster
        base_color= plt.cm.tab10(i % 10)
        cmap = LinearSegmentedColormap.from_list(f'custom_{base_color}', ['white', base_color,'black'])
        cmap = ListedColormap(cmap(np.linspace(0.2, 0.8, n_conformations)))
        
        # Map conformation indices to normalized color values
        colored_row = np.copy(cluster_data)
        for j, label in enumerate(labels_cluster_i):
            colored_row[cluster_data == label] = j
        
        # Plot this row with its own colormap
        plt.imshow(
            colored_row[np.newaxis, :],  # Make it 2D: 1 row
            cmap=cmap,
            norm=norm,
            aspect='auto',
            extent=[times_selected[0], times_selected[-1], i, i+1],
            origin='lower'
        )

    # Draw horizontal lines between clusters
    for i in range(len(conformations_for_community)-1):
        plt.axhline(y=i+1, color='k', linestyle='-', linewidth=0.5)

    plt.yticks(np.arange(0.5, num_clusters + 0.5), comumunities_to_plot)
    plt.title('Conformational States for Communities as a Function of Time')
    plt.xlabel('Time (in ps)')
    plt.ylabel('Community of LVs Index')
    plt.tight_layout()
    plt.savefig(output_dir + f"conformational_states_clustering/conformational_states_as_function_of_time.{extension_plots}", dpi=resolution_plots)
    plt.close()
    logging.info("Conformational time plot saved.")

    correlation_matrix = np.corrcoef(conformations_for_community)
    correlation_matrix = np.nan_to_num(correlation_matrix)  # Replace NaNs with 0 for plotting
    correlation_matrix =np.abs(correlation_matrix)

    matrix_to_text = np.array2string(correlation_matrix, formatter={'float_kind': lambda x: f"{x:.2f}"})

    logging.info("Absolute Pearson correlation matrix between clusters of local_variables:")
    logging.info(matrix_to_text)
    plt.figure(figsize=(8, 6))
    plt.imshow(correlation_matrix, cmap='magma', aspect='equal')
    cbar = plt.colorbar(label='Absolute Pearson Correlation Coefficient')
    plt.clim(0, 1)
    plt.xticks(np.arange(num_clusters), comumunities_to_plot)
    plt.yticks(np.arange(num_clusters), comumunities_to_plot)
    plt.title('Correlation of Conformational States Between Communities of LVs')
    plt.xlabel('Community of LVs Index')
    plt.ylabel('Community of LVs Index')
    plt.tight_layout()
    plt.savefig(output_dir + f"conformational_states_clustering/correlation_conformations_between_communities.{extension_plots}", dpi=resolution_plots)
    plt.close()

