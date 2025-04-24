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



def plot_progress_bar(current, total, bar_length=40):
    progress = current / total
    block = int(round(bar_length * progress))
    text = f"\rProgress: [{'#' * block + '-' * (bar_length - block)}] {progress * 100:.2f}%"
    print(text, end='')
        

def read_dictionary(dic):
    """
    Reads the terminal atoms dictionary file and returns a dictionary of terminal atoms.
    """
    terminal_atoms_dic = {}
    amino_acids=[]
    with open(dic, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            if len(line) > 1:
                if line[-1] == "@amino_acid" :
                    terminal_atoms_dic[line[0]] = line[1:-1]
                    amino_acids.append(line[0])
                else :
                    terminal_atoms_dic[line[0]] = line[1:]
            else:
                print(f"Skipping line: {line}")
    return terminal_atoms_dic,amino_acids


def open_file (namefile) :
    file_opened=open(namefile,'r')
    lines_file=file_opened.readlines()
    data=[]
    for row in lines_file :
     #   print(row)
        data.append([x for x in row.split()])
    return data,lines_file

def open_data_coordinate (namefile) :
    with open(namefile, 'r') as f:
        data = np.loadtxt(f)
    return data


def get_terminal_atoms_MDA(u_traj, terminal_atoms_dic):
    """
    Get terminal atoms from MDAnalysis universe.
    """
    atoms_dic,amino_acids = read_dictionary(terminal_atoms_dic)
    terminal_atoms = []
    RESIDS_SELECTED = []
    RESNAMES_SELECTED = []
    indices_aa = []
    RES_NOT_FOUND=[]
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
    return terminal_atoms, RESIDS_SELECTED, RESNAMES_SELECTED,indices_aa

def save_terminal_atoms(terminal_atoms, RESIDS_SELECTED, RESNAMES_SELECTED,output_dir):
    with open(output_dir+'terminal_atoms.txt', 'w') as f:
        for k in range(len(terminal_atoms)):
            atom=terminal_atoms[k]
            resid=RESIDS_SELECTED[k]
            type_aa=RESNAMES_SELECTED[k]
            f.write(f'{resid}   {type_aa}   {atom}\n')
        f.close()

def open_trajectory(grofile,trajfile):
    """
    Opens a trajectory file using MDAnalysis and returns the universe object.
    """
    u_traj = mda.Universe(grofile,trajfile)
    return u_traj


def compute_distances_CA_from_gro(u_gro, RESIDS_SELECTED):
    """
    Computes the distances between terminal atoms and all other atoms in the universe.
    """
    number_of_resids= len(RESIDS_SELECTED)

    distances_CA = np.zeros((number_of_resids,number_of_resids))
    for i in range(number_of_resids-1):
        for j in range(i+1,number_of_resids):
            if i != j:
                atom1 = u_gro.select_atoms(f"resid {RESIDS_SELECTED[i]} and name CA")
                atom2 = u_gro.select_atoms(f"resid {RESIDS_SELECTED[j]} and name CA")
                distances_CA[i, j] = np.linalg.norm(atom1.positions - atom2.positions)
                distances_CA[j, i] = distances_CA[i, j]
    
    return distances_CA

def first_selection_on_CA(u_gro, RESIDS_SELECTED, cutoff_CA):
    """
    Selects the first selection of residues based on the distances between CA atoms.
    """
    distances_CA = compute_distances_CA_from_gro(u_gro, RESIDS_SELECTED)
    number_of_resids= len(RESIDS_SELECTED)
    dic_selection_CA=np.zeros((number_of_resids,number_of_resids))
    for i in range(number_of_resids-1):
        for j in range(i+1,number_of_resids):
            if distances_CA[i, j] < cutoff_CA:
                dic_selection_CA[i, j] = 1
                dic_selection_CA[j, i] = 1

    return dic_selection_CA

def precompute_CA_and_terminals(u_traj, terminal_atoms, RESIDS_SELECTED, times, times_indices,indices_aa):
    """
    Optimized version of precomputing positions of terminal and CA atoms for all residues over the trajectory.
    """
    print("Precomputing positions...")
    num_residues = len(RESIDS_SELECTED)
    num_atoms = np.sum([len(terminal_atoms[i]) for i in range(num_residues)])

    num_amino_acids = len(indices_aa)
   
    # Pre-select atoms for terminal and CA
    atom_terminal_selections = []
    for i in range(num_residues):
        atom_terminal_selections.append(
            [u_traj.select_atoms(f"resid {RESIDS_SELECTED[i]} and name {terminal_atoms[i][j]}") for j in range(len(terminal_atoms[i]))]
        )
    atom_CA_selections = [
        u_traj.select_atoms(f"resid {indices_aa[i]} and name CA")
        for i in range(num_amino_acids)
    ]

    # Initialize arrays for positions
    Positions_atoms_terminal = np.zeros((num_atoms, len(times_indices), 3))
    Positions_atoms_CA = np.zeros((num_amino_acids, len(times_indices), 3))

    # Iterate over frames and precompute positions
    for k, frame in enumerate(times_indices):
        u_traj.trajectory[frame]
        plot_progress_bar(k, len(times_indices))
        count_step=0
        count_aa=0
        for i in range(num_residues):

            for j in range(len(terminal_atoms[i])):
                Positions_atoms_terminal[count_step, k, :] = atom_terminal_selections[i][j].positions
                if RESIDS_SELECTED[i] in indices_aa :
                    if terminal_atoms[i][j]== 'CA':
                        Positions_atoms_CA[count_aa, k, :] = Positions_atoms_terminal[count_step, k, :]
                    count_aa += 1
                count_step += 1
        for i in range(num_amino_acids):
            if 'CA' not in terminal_atoms[i] :
                Positions_atoms_CA[i, k, :] = atom_CA_selections[i].positions
    plot_progress_bar(len(times_indices), len(times_indices))
    print("\nPositions precomputed.")
    return Positions_atoms_terminal, Positions_atoms_CA

def precompute_C_and_N(u_traj, RESIDS_SELECTED, times, times_indices):
    """
    Optimized version of precomputing positions of C and N atoms for all residues over the trajectory.
    """
    print("Precomputing positions...")
    num_residues = len(RESIDS_SELECTED)

    atom_C_selections = [
        u_traj.select_atoms(f"resid {RESIDS_SELECTED[i]} and name C")
        for i in range(num_residues)
    ]

    atom_N_selections = [
        u_traj.select_atoms(f"resid {RESIDS_SELECTED[i]} and name N")
        for i in range(num_residues)
    ]

    # Initialize arrays for positions
    Positions_atoms_C = np.zeros((num_residues, len(times_indices), 3))
    Positions_atoms_N = np.zeros((num_residues, len(times_indices), 3))

    # Iterate over frames and precompute positions
    for k, frame in enumerate(times_indices):
        plot_progress_bar(k, len(times_indices))
        u_traj.trajectory[frame]
        for i in range(num_residues):
            Positions_atoms_C[i, k, :] = atom_C_selections[i].positions
            Positions_atoms_N[i, k, :] = atom_N_selections[i].positions
    plot_progress_bar(len(times_indices), len(times_indices))
    print("\nPositions precomputed.")
    return Positions_atoms_C, Positions_atoms_N

def compute_min_distances(Positions_atoms_terminal, Positions_atoms_CA, i, j,indices_aa,terminal_atoms, RESIDS_SELECTED):
    """
    Computes the minimum distances between terminal and CA atoms for two residues.
    """
    num_term_i= len(terminal_atoms[i])
    num_term_j= len(terminal_atoms[j])
    
    ind_term_0_i=sum([len(terminal_atoms[k]) for k in range(i)])
    ind_term_0_j=sum([len(terminal_atoms[k]) for k in range(j)])

    Positions_i=[Positions_atoms_terminal[ind_term_0_i+k,:,:] for k in range(num_term_i)]
    Positions_j=[Positions_atoms_terminal[ind_term_0_j+k,:,:] for k in range(num_term_j)]
    
    atoms_i=terminal_atoms[i].copy()
    atoms_j=terminal_atoms[j].copy()
    

    if RESIDS_SELECTED[i] in indices_aa:
        ind_aa=indices_aa.index(RESIDS_SELECTED[i])
        Positions_i.append(Positions_atoms_CA[i,:,:])
        atoms_i.append('CA')
    if RESIDS_SELECTED[j] in indices_aa:
        ind_aa=indices_aa.index(RESIDS_SELECTED[j])
        Positions_j.append(Positions_atoms_CA[j,:,:])
        atoms_j.append('CA')
    
    distances= np.zeros((len(atoms_i),len(atoms_j),len(Positions_i[0])))
    for k in range(len(atoms_i)):
        for l in range(len(atoms_j)):
            distances[k,l]=np.linalg.norm(Positions_i[k]-Positions_j[l],axis=1)
    minimal_distances=np.zeros((len(atoms_i),len(atoms_j)))
    for k in range(len(atoms_i)):
        for l in range(len(atoms_j)):
            minimal_distances[k,l]=np.min(distances[k,l])
    minimal_indexes = np.unravel_index(np.argmin(minimal_distances, axis=None), minimal_distances.shape)
    
    
    min_absolute_distance = minimal_distances[minimal_indexes[0], minimal_indexes[1]]
    distance_to_save = distances[minimal_indexes[0], minimal_indexes[1]]
    atom_i_to_save= atoms_i[minimal_indexes[0]]
    atom_j_to_save= atoms_j[minimal_indexes[1]]
    return min_absolute_distance,distance_to_save,atom_i_to_save,atom_j_to_save


def save_coordinate_results(times, distance_to_save, coordinate,output_dir):
    """
    Saves the distance results to a file.
    """
    Time_evolution = np.column_stack((times, distance_to_save))
    output_file = output_dir+"coordinates_data/"+coordinate+".dat"
    np.savetxt(output_file, Time_evolution, fmt="%.2f   %.2f")


def perform_kde(data, delta_y, bandwidth=None):
    # Convert data to numpy array
    data = np.asarray(data)

    delta_y_smooth = delta_y/5
    # Adaptive bandwidth selection if not provided
    if bandwidth is None:
        silverman_bw = 1.06 * np.std(data) * len(data) ** (-1 / 5)
        scott_bw = np.power(len(data), -1 / (data.ndim + 4))
        bandwidth = min(silverman_bw, scott_bw)  # Use the more conservative estimate

    # Create a grid for KDE
    x_smooth = np.arange(np.min(data), np.max(data), delta_y_smooth)

    # Perform KDE
    kde = gaussian_kde(data, bw_method=bandwidth)
    H_kde = kde(x_smooth)

    # Improve peak detection by dynamically adjusting height and prominence
    peaks, properties = find_peaks(H_kde, height=np.max(H_kde) * 0.1, prominence=np.max(H_kde) * 0.05)

    if len(peaks) > 1:
        return True, H_kde, x_smooth
    else:
        return False, None, None
    
def compute_histogram(data, y_min, y_max, delta_y):
    return np.histogram(data, bins=np.arange(y_min, y_max + delta_y, delta_y), density=True)

def compute_hist_tot(times,data, num_blocks, y_min, y_max, delta_y, time_zero_ps, size_block_ps):
    HIST_TOT = np.zeros((num_blocks, len(np.arange(y_min, y_max + delta_y, delta_y)) - 1))
    for i in range(num_blocks):
        start_time = time_zero_ps + i * size_block_ps
        end_time = start_time + size_block_ps
        block_data = data[(times >= start_time) & (times < end_time)]
        hist, bin_edges = compute_histogram(block_data, y_min, y_max, delta_y)
        HIST_TOT[i] = hist
        x = (bin_edges[:-1] + bin_edges[1:]) / 2
        AVG = np.average(HIST_TOT, axis=0)
        STD = np.std(HIST_TOT, axis=0)
    return HIST_TOT, x, AVG, STD

def compute_error_bars(STD, num_blocks, confidence_level=0.95):
    degrees_freedom = num_blocks - 1
    t_value = t.ppf((1 + confidence_level) / 2, degrees_freedom)
    return t_value * (STD / np.sqrt(num_blocks))



def adjust_angle_data(data, y_min, y_max, delta_y):
    hist_all, bin_edges_all = compute_histogram(data, y_min, y_max, delta_y)
    x_all = (bin_edges_all[:-1] + bin_edges_all[1:]) / 2
    min_indices = np.where(hist_all == np.min(hist_all))[0]
    x_min_all = x_all[min_indices[len(min_indices) // 2]] if len(min_indices) > 1 else x_all[min_indices[0]]
    data = np.where(data < x_min_all, data + 360, data)
    y_max = max(data)
    y_min = min(data)
    return data, y_max, y_min

def get_avg_histogram(times,data,time_zero_ps,size_block_ps,coord_type):

    if coord_type == 'distance':
        xlabel = 'Distance (Angstroms)'
        delta_y=0.1
    elif coord_type == 'angle':
        xlabel = 'Angle (degrees)'
        delta_y = 2
    
    num_blocks = int((times[-1] - time_zero_ps) / size_block_ps)
    y_max = max(data)
    y_min = min(data)


    discretized_data = np.zeros_like(data)

    HIST_TOT, x, AVG, STD = compute_hist_tot(times,data, num_blocks, y_min, y_max, delta_y, time_zero_ps, size_block_ps)
    error_bars = compute_error_bars(STD, num_blocks)
    return data,data,x,AVG,error_bars,delta_y,coord_type,xlabel

def find_minimums(x_smooth,H_kde):
    D_kde = np.gradient(H_kde, x_smooth[1] - x_smooth[0])
    D2_kde = np.gradient(D_kde, x_smooth[1] - x_smooth[0])
    
    zero_crossings = np.where(np.diff(np.sign(D_kde)))[0]
    minimums = []
    for idx in zero_crossings:
        if D2_kde[idx] > 0:
            minimums.append(x_smooth[idx])
    return minimums

def filter_minimums_KDE(minimums,x_smooth,H_kde,cutoff_value_kde,cutoff_value_x):
    filtered_minimums = []
    val_0=0

    indexes_minimums=[np.where(x_smooth == minimums[i])[0][0] for i in range(len(minimums))]


    for i in range(len(indexes_minimums)):
        ind_mini=indexes_minimums[i]
        mini=x_smooth[ind_mini]
        if len(filtered_minimums)>0:
            ind_before = np.where(x_smooth == filtered_minimums[-1])[0][0]
            
        else:
            ind_before = 0
        if ind_before < ind_mini:
            max_before = max(H_kde[ind_before:ind_mini])
            x_before = x_smooth[ind_before+np.argmax(H_kde[ind_before:ind_mini])]
        else:
            max_before = 0
            x_before = 0
        if i < len(indexes_minimums) - 1:
            ind_after = indexes_minimums[i+1]
            if ind_mini < ind_after:
                max_after = max(H_kde[ind_mini:ind_after])
                x_after = x_smooth[ind_mini+np.argmax(H_kde[ind_mini:ind_after])]
            else:
                max_after = 0
                x_after = 0
        else:
            max_after = max(H_kde[ind_mini:])
            x_after = x_smooth[ind_mini+np.argmax(H_kde[ind_mini:])]
        val_0 = H_kde[ind_mini]
        x_mini=x_smooth[ind_mini]
        delta_val = max_before - val_0
        delta_val2 = max_after - val_0  
        delta_x = x_after - x_before
        
        if delta_val > cutoff_value_kde and delta_val2 > cutoff_value_kde and delta_x > cutoff_value_x :
            filtered_minimums.append(mini)
    return filtered_minimums

def plot_histogram(x, AVG, error_bars, H_kde, x_smooth, delta_y, coord_type, xlabel, coordinate,minimums,output_dir):
    fig, ax = plt.subplots()
    ax.plot(x, AVG, color='black', label='Average')
    ax.fill_between(x, AVG - error_bars, AVG + error_bars, color='black', alpha=0.3)
    ax.plot(x_smooth, H_kde, color='red', lw=2, label='KDE')
    for mini in minimums:
        ax.axvline(x=mini, color='blue', linestyle='--')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Probability density')
    ax.set_title(coordinate)
    ax.legend()
    plt.savefig(f'{output_dir}coordinates_plots/{coordinate}.png', dpi=150)
    plt.close()


def get_labels_discretization_kde(minimums,x_smooth,H_kde):
    indexes_minimums = [np.where(x_smooth==mini)[0][0] for mini in minimums]
    all_minimums=[0]+indexes_minimums+[len(x_smooth)-1]
    inter_max=[]
    for i in range(len(all_minimums)-1):
        inter_max.append(max(H_kde[all_minimums[i]:all_minimums[i+1]]))
    sorted_indices = np.argsort(inter_max)[::-1]
    labels=np.zeros(len(sorted_indices),dtype=int)

    for i in range(len(sorted_indices)):
        labels[sorted_indices[i]]=i
    return labels

def save_minimums(minimums,coordinate,labels,name_output):
    

    file_output=open(name_output,'a')
    file_output.write(f'{coordinate} ')
    for i in range(len(minimums)):
        file_output.write(f' {labels[i]}')
        file_output.write(f' {minimums[i]:.3f}')
    file_output.write(f' {labels[-1]}')
    file_output.write('\n')
    file_output.close()

def filter_times_and_indices(u_traj, time_zero, delta_time,output_dir):
    """
    Filters trajectory times and indices based on the given time_zero and delta_time.
    """
    print("Filtering times and indices...")
    times=[]
    times_indices=[]
    for ts in u_traj.trajectory:
        plot_progress_bar(ts.frame, len(u_traj.trajectory))
        if ts.time >= time_zero and ts.time % delta_time == 0:
            times.append(ts.time)
            times_indices.append(ts.frame)
    plot_progress_bar(len(u_traj.trajectory), len(u_traj.trajectory))
    times = np.array(times)
    times_indices = np.array(times_indices)
    np.save(output_dir+'times.npy', times)
    np.save(output_dir+'times_indices.npy', times_indices)
    print("\nTimes and indices filtered.")
    return times,  times_indices

def save_positions(Positions, outname):
    """
    Saves precomputed positions
    """
    np.save(outname, Positions)
    

def process_distance_pair(i, j, Positions_atoms_terminal, Positions_atoms_CA, terminal_atoms, RESIDS_SELECTED, times, time_zero, size_block, cutoff_distances, height_cutoff, output,indices_aa,output_dir):
    """
    Processes a pair of residues to compute distances and analyze multimodality.
    """
    min_absolute_distance,distance_to_save,atom_i_to_save,atom_j_to_save = compute_min_distances(
        Positions_atoms_terminal, Positions_atoms_CA, i, j,indices_aa,terminal_atoms, RESIDS_SELECTED
    )
   
    delta_y=0.1

    delta_distance = max(distance_to_save) - min(distance_to_save)
          
    if min_absolute_distance > cutoff_distances or delta_distance < delta_y*20:
        return

    
    
    multimodality, H_kde, x_smooth = perform_kde(distance_to_save, delta_y)
    if multimodality:
        data, filtered_data, x, AVG, error_bars, delta_y, coord_type, xlabel = get_avg_histogram(times, distance_to_save, time_zero, size_block, 'distance')
        cutoff_value_kde = max(H_kde) * height_cutoff / 100
        cutoff_value_x= delta_y*5
        minimums = find_minimums(x_smooth, H_kde)
        if len(minimums) > 0:
            minimums = filter_minimums_KDE(minimums, x_smooth, H_kde, cutoff_value_kde,cutoff_value_x)
            if len(minimums) > 0:
                coordinate = f"{RESIDS_SELECTED[i]}_{atom_i_to_save}_{RESIDS_SELECTED[j]}_{atom_j_to_save}"
                plot_histogram(x, AVG, error_bars, H_kde, x_smooth, delta_y, coord_type, xlabel, coordinate, minimums,output_dir)
                labels=get_labels_discretization_kde(minimums,x_smooth,H_kde)
                save_minimums(minimums, coordinate, labels, output)
                save_coordinate_results(times, distance_to_save, coordinate,output_dir)

def process_dihedral_i(i, Positions_atoms_C, Positions_atoms_N, Positions_atoms_CA, RESIDS_SELECTED, times, time_zero, size_block, height_cutoff, output,output_dir):
    """
    Processes a residue to compute dihedrals and analyze multimodality.
    """
    phi_angle=np.zeros(len(times))
    psi_angle=np.zeros(len(times))
    delta_y = 2
    
    if i>0 and np.linalg.norm(Positions_atoms_C[i-1,0, :]-Positions_atoms_N[i,0, :]) < 1.6 :
        phi_angle = np.rad2deg(mda.lib.distances.calc_dihedrals(Positions_atoms_C[i-1,:, :], Positions_atoms_N[i,:, :], Positions_atoms_CA[i , :, :], Positions_atoms_C[i, :, :]))
        if max(phi_angle)-min(phi_angle) > 180:
            phi_angle, y_max_phi, y_min_phi = adjust_angle_data(phi_angle, min(phi_angle), max(phi_angle), delta_y)
        delta_angle= max(phi_angle) - min(phi_angle)
        multimodality_phi, H_kde_phi, x_smooth_phi = perform_kde(phi_angle, delta_y)
        if multimodality_phi and delta_angle > delta_y*20:
            data, filtered_data, x, AVG, error_bars, delta_y, coord_type, xlabel = get_avg_histogram(times, phi_angle, time_zero, size_block,'angle')
            cutoff_value = max(AVG) * height_cutoff / 100
            cutoff_value_x= delta_y*5
            minimums = find_minimums(x_smooth_phi, H_kde_phi)
            if len(minimums) > 0:
                minimums = filter_minimums_KDE(minimums, x_smooth_phi, H_kde_phi, cutoff_value,cutoff_value_x)
                if len(minimums) > 0:
                    coordinate = f"phi{RESIDS_SELECTED[i]}"
                    plot_histogram(x, AVG, error_bars,H_kde_phi,x_smooth_phi,delta_y,'angle',xlabel,coordinate,minimums,output_dir)
                    labels=get_labels_discretization_kde(minimums,x_smooth_phi,H_kde_phi)
                    save_minimums(minimums,coordinate,labels,output)
                    save_coordinate_results(times, phi_angle,coordinate,output_dir)

            
    if i<len(Positions_atoms_C)-1 and np.linalg.norm(Positions_atoms_N[i+1,0, :]-Positions_atoms_C[i,0, :]) < 1.6 :
        psi_angle = np.rad2deg(mda.lib.distances.calc_dihedrals(Positions_atoms_N[i, :, :], Positions_atoms_CA[i, :, :], Positions_atoms_C[i , :, :], Positions_atoms_N[i+1, :, :]))
        if max(psi_angle)-min(psi_angle) > 180:
            psi_angle, y_max_psi, y_min_psi = adjust_angle_data(psi_angle, min(psi_angle), max(psi_angle), delta_y)
        delta_angle= max(psi_angle) - min(psi_angle)
        multimodality_psi, H_kde_psi, x_smooth_psi = perform_kde(psi_angle,delta_y)
        if multimodality_psi and delta_angle > delta_y*20:
            data, filtered_data, x, AVG, error_bars, delta_y, coord_type, xlabel = get_avg_histogram(times, psi_angle, time_zero, size_block,'angle')
            cutoff_value = max(AVG) * height_cutoff / 100
            cutoff_value_x= delta_y*5
            minimums = find_minimums(x_smooth_psi, H_kde_psi)
            
            if len(minimums) > 0:
                minimums = filter_minimums_KDE(minimums, x_smooth_psi, H_kde_psi, cutoff_value,cutoff_value_x)
                if len(minimums) > 0:
                    coordinate = f"psi{RESIDS_SELECTED[i]}"
                    plot_histogram(x, AVG, error_bars,H_kde_psi,x_smooth_psi,delta_y,'angle',xlabel,coordinate,minimums,output_dir)
                    labels=get_labels_discretization_kde(minimums,x_smooth_psi,H_kde_psi)
                    save_minimums(minimums,coordinate,labels,output)
                    save_coordinate_results(times, psi_angle,coordinate,output_dir)
    
    

def compute_all_distances(u_traj, terminal_atoms, RESIDS_SELECTED, Positions_atoms_terminal, Positions_atoms_CA, times, time_zero, size_block, delta_resid, cutoff_distances, height_cutoff, output,indices_aa,output_dir):
    """
    Computes distances for all residue pairs and processes them.
    """
    num_residues = len(RESIDS_SELECTED)
    total_combinations = num_residues * (num_residues - delta_resid) / 2
    count_step = 0
    print("Computing distances...")
    for i in range(num_residues - delta_resid):
        for j in range(i + delta_resid, num_residues):
            plot_progress_bar(count_step, total_combinations)
            count_step += 1
            process_distance_pair(i, j, Positions_atoms_terminal, Positions_atoms_CA, terminal_atoms, RESIDS_SELECTED, times, time_zero, size_block, cutoff_distances, height_cutoff, output,indices_aa,output_dir)

    plot_progress_bar(total_combinations, total_combinations)
    print("\nDistances computed and saved.")



def compute_all_dihedrals(u_traj, RESIDS_SELECTED, Positions_atoms_C, Positions_atoms_N, Positions_atoms_CA, times, time_zero, size_block, height_cutoff, output,output_dir):  
    """
    Computes dihedrals for all residue pairs and processes them.
    """
    num_residues = len(RESIDS_SELECTED)
    print("Computing dihedrals...")

    for i in range(num_residues ):
        plot_progress_bar(i,num_residues)
        process_dihedral_i(i, Positions_atoms_C, Positions_atoms_N, Positions_atoms_CA, RESIDS_SELECTED, times, time_zero, size_block, height_cutoff, output,output_dir)

    plot_progress_bar(num_residues, num_residues)
    print("\nDihedrals computed and saved.")

def get_contacts(u_traj, terminal_atoms, RESIDS_SELECTED, time_zero, size_block, delta_time, cutoff_distances, delta_resid, height_cutoff,indices_aa,output_dir):
    """
    Main function to compute distances between terminal atoms over the trajectory.
    """
    times= np.load(output_dir+'times.npy')
    times_indices= np.load(output_dir+'times_indices.npy')
    Positions_atoms_terminal, Positions_atoms_CA = precompute_CA_and_terminals(
        u_traj, terminal_atoms, RESIDS_SELECTED, times, times_indices,indices_aa
    )
    save_positions(Positions_atoms_terminal, output_dir+"Positions_npy/Positions_terminal_atoms.npy")
    if len(indices_aa) > 2:
        save_positions(Positions_atoms_CA, output_dir+"Positions_npy/Positions_CA_atoms.npy")

    compute_all_distances(
        u_traj, terminal_atoms, RESIDS_SELECTED, Positions_atoms_terminal, Positions_atoms_CA,
        times, time_zero, size_block, delta_resid, cutoff_distances, height_cutoff, output_dir+"selected_coordinates.txt",indices_aa,output_dir
    )

def get_dihedrals(u_traj, indices_aa, time_zero, size_block, delta_time, cutoff_distances, delta_resid, height_cutoff,output_dir):

    """
    Computes dihedrals for all residue pairs and processes them.
    """
    times= np.load(output_dir+'times.npy')
    times_indices= np.load(output_dir+'times_indices.npy')
    if len(indices_aa) < 2:
        print("No amino acids selected for dihedral analysis.")
        return
    Positions_atoms_C, Positions_atoms_N = precompute_C_and_N(
        u_traj, indices_aa, times, times_indices
    )
    save_positions(Positions_atoms_C, output_dir+"Positions_npy/Positions_C_atoms.npy")
    save_positions(Positions_atoms_N, output_dir+"Positions_npy/Positions_N_atoms.npy")
    Positions_atoms_CA = np.load(output_dir+"Positions_npy/Positions_CA_atoms.npy")
    
    compute_all_dihedrals(
        u_traj, indices_aa, Positions_atoms_C,Positions_atoms_N, Positions_atoms_CA,
        times, time_zero, size_block, height_cutoff, output_dir+"selected_coordinates.txt",output_dir
    )

def load_data_discretization(output_selected_coordinates):
    data_discretization,lines_discretization=open_file(output_selected_coordinates)
    coordinates=[data_discretization[i][0] for i in range(len(data_discretization))]
    X_cuts=[]
    Labels=[]
    for i in range(len(data_discretization)):
        data_i=data_discretization[i]
        xcut_i=[]
        labels_i=[]
        for c in range(1,len(data_i)):
            if c%2==0:
                xcut_i.append(float(data_i[c]))
            if c%2==1:
                labels_i.append(int(data_i[c]))
        X_cuts.append(xcut_i)
        Labels.append(labels_i)
    return coordinates,X_cuts,Labels

def add_coordinates(coordinates_to_add,type_coordinates_to_add,output_dir,time_zero,size_block,height_cutoff):
    coordinates,X_cuts,Labels=load_data_discretization(output_dir+"selected_coordinates.txt")
    
    data_zero=open_data_coordinate(output_dir+"coordinates_data/"+coordinates[0]+".dat")
    times_to_compare=data_zero[:,0]
    delta_y=0
    for coord_file in coordinates_to_add:
        data_coord_raw=open_data_coordinate(coord_file)
        coord_name=coord_file.split('/')[-1].split('.')[0]
        type_coord=type_coordinates_to_add[coordinates_to_add.index(coord_file)]
        if type_coord == 'distance':
            delta_y=0.1
        elif type_coord == 'angle':
            delta_y=2
        y_coord=[]
        t_coord=[]
        for i in range(len(data_coord_raw)):
            if data_coord_raw[i][0] in times_to_compare:
                y_coord.append(data_coord_raw[i][1])
                t_coord.append(data_coord_raw[i][0])
        y_coord=np.array(y_coord)
        t_coord=np.array(t_coord)
        if len(t_coord)!=len(times_to_compare) or max(times_to_compare-t_coord)!=0 :
            print(f"Warning: {coord_file} has different steps than the reference file.")
            continue
        if type_coord == 'angle' and max(y_coord)-min(y_coord) > 180:
            y_coord,y_max,y_min=adjust_angle_data(y_coord,min(y_coord),max(y_coord),delta_y)
        multimodality, H_kde, x_smooth = perform_kde(y_coord,delta_y)
        delta_coord= max(y_coord) - min(y_coord)
        if multimodality and delta_coord > delta_y*20:
            data, filtered_data, x, AVG, error_bars, delta_y, coord_type, xlabel = get_avg_histogram(t_coord, y_coord, time_zero, size_block,type_coord)
            cutoff_value = max(AVG) * height_cutoff / 100
            cutoff_value_x= delta_y*5
            minimums = find_minimums(x_smooth, H_kde)
            
            if len(minimums) > 0:
                minimums = filter_minimums_KDE(minimums, x_smooth, H_kde, cutoff_value,cutoff_value_x)
                if len(minimums) > 0:
                    plot_histogram(x, AVG, error_bars,H_kde,x_smooth,delta_y,type_coord,xlabel,coord_name,minimums,output_dir)
                    labels=get_labels_discretization_kde(minimums,x_smooth,H_kde)
                    save_minimums(minimums,coord_name,labels,output_dir+"selected_coordinates.txt")
                    save_coordinate_results(t_coord, y_coord,coord_name,output_dir)
        
        
def get_discretized_array(output_dir):
    coordinates,X_cuts,Labels=load_data_discretization(output_dir+"selected_coordinates.txt")
    data_zero=open_data_coordinate(output_dir+"coordinates_data/"+coordinates[0]+".dat")
    times_to_compare=data_zero[:,0]
    nframes=len(times_to_compare)
    data_discretized=np.zeros((nframes,len(coordinates)),dtype=int)
    print("Discretizing data...")
    for i in range(len(coordinates)):
        plot_progress_bar(i,len(coordinates))
        data_coord=open_data_coordinate(output_dir+"coordinates_data/"+coordinates[i]+".dat")
        for f in range(nframes):
            for c in range(len(X_cuts[i])):
                if data_coord[f,1]<X_cuts[i][c]:
                    data_discretized[f,i]=Labels[i][c]
                    break
                if c==len(X_cuts[i])-1:
                    data_discretized[f,i]=Labels[i][-1]
    plot_progress_bar(len(coordinates),len(coordinates))
    print("\nDiscretization completed.")
    np.save(output_dir+"discretized_array.npy",data_discretized)


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
    Positions_atoms_terminal = np.load(output_dir+"Positions_npy/Positions_terminal_atoms.npy")

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
            ind_pos=RESIDS_SELECTED.index(index_resid)
            Positions_barycenters[i]=(Positions_atoms_C[ind_pos-1,:, :]+Positions_atoms_N[ind_pos,:, :]+Positions_atoms_CA[ind_pos , :, :]+Positions_atoms_C[ind_pos, :, :])/4
        elif coord[:3]=='psi':
            index_resid=int(coord[3:])
            ind_pos=RESIDS_SELECTED.index(index_resid)
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
                Positions_barycenters[i]=(Positions_atoms_terminal[index_term1,:,:]+Positions_atoms_terminal[index_term2,:,:])/2
            elif index_CA1 != -1 and index_term2 != -1:
                Positions_barycenters[i]=(Positions_atoms_CA[index_CA1,:,:]+Positions_atoms_terminal[index_term2,:,:])/2
            elif index_term1 != -1 and index_CA2 != -1:
                Positions_barycenters[i]=(Positions_atoms_terminal[index_term1,:,:]+Positions_atoms_CA[index_CA2,:,:])/2
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

def get_multiplicities(Discretized_Array):
    nframes,ncoord=np.shape(Discretized_Array)
    multiplicities=np.zeros((ncoord),dtype=int)
    for i in range(ncoord):
        multiplicities[i]=len(np.unique(Discretized_Array[:,i]))
    return multiplicities

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
def compute_frequencies(Discretized_Array):
    nframes, ncoord = Discretized_Array.shape
    multiplicities = get_multiplicities(Discretized_Array)
    multiplicity_tot = np.sum(multiplicities)

    single_frequencies = np.zeros(multiplicity_tot, dtype=float)
    double_frequencies = np.zeros((multiplicity_tot, multiplicity_tot), dtype=float)

    print("Computing single frequencies...")
    index_freq = np.cumsum([0] + list(multiplicities[:-1]))
    for i in range(ncoord):
        plot_progress_bar(i, ncoord)
        unique, counts = np.unique(Discretized_Array[:, i], return_counts=True)
        single_frequencies[index_freq[i] + unique] = counts / nframes
    plot_progress_bar(ncoord, ncoord)
    print("\nSingle frequencies computed.")

    print("Computing double frequencies...")
    for i in range(ncoord):
        for j in range(i, ncoord):
            plot_progress_bar(i * ncoord + j, ncoord * ncoord)
            joint_counts = np.zeros((multiplicities[i], multiplicities[j]), dtype=int)
            for f in range(nframes):
                joint_counts[Discretized_Array[f, i], Discretized_Array[f, j]] += 1
            joint_probs = joint_counts / nframes
            idx_i = index_freq[i]
            idx_j = index_freq[j]
            double_frequencies[idx_i:idx_i + multiplicities[i], idx_j:idx_j + multiplicities[j]] = joint_probs
            double_frequencies[idx_j:idx_j + multiplicities[j], idx_i:idx_i + multiplicities[i]] = joint_probs.T
    plot_progress_bar(ncoord * ncoord, ncoord * ncoord)
    print("\nDouble frequencies computed.")
    

    return single_frequencies, double_frequencies

def compute_frequencies_slow(Discretized_Array):
    nframes,ncoord=np.shape(Discretized_Array)
    multiplicities=get_multiplicities(Discretized_Array)
    multiplicity_tot=np.sum(multiplicities)
    
    single_frequencies=np.zeros((multiplicity_tot),dtype=float)
    index_freq=0
    print("Computing single frequencies...")
    for i in range(ncoord):
        plot_progress_bar(i,ncoord)
        for xi in range(multiplicities[i]):
            probab_xi=0
            for f in range(nframes):
                if Discretized_Array[f,i]==xi:
                    probab_xi+=1/nframes
            single_frequencies[index_freq]=probab_xi
            index_freq+=1
    plot_progress_bar(ncoord,ncoord)
    print("\nSingle frequencies computed.")
    print("Computing double frequencies...")
    double_frequencies=np.ones((multiplicity_tot,multiplicity_tot),dtype=float)
    index_freq_1=0
    count_step=0
    for i in range(ncoord):
        for xi in range(multiplicities[i]):
            index_freq_2=0
            for j in range(i,ncoord):
                for xj in range(multiplicities[j]):
                    plot_progress_bar(count_step,multiplicity_tot*multiplicity_tot)
                    prob_xi_xj=0
                    for f in range(nframes):
                        if Discretized_Array[f,i]==xi and Discretized_Array[f,j]==xj:
                            prob_xi_xj+=1/nframes
                    double_frequencies[index_freq_1,index_freq_2]=prob_xi_xj
                    double_frequencies[index_freq_2,index_freq_1]=prob_xi_xj
                    index_freq_2+=1
                    count_step+=1
            index_freq_1+=1
    plot_progress_bar(multiplicity_tot*multiplicity_tot,multiplicity_tot*multiplicity_tot)
    print("\nDouble frequencies computed.")
            
    return single_frequencies, double_frequencies
            
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
        plt.scatter(distance_i.flatten(), MI_i.flatten(),marker='x', alpha=0.8,label=f'Cluster {i}',color=plt.cm.rainbow(i / (len(clusters_ndx)-2)))
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

                
def get_frequencies(output_dir) :
    Discretized_Array=np.load(output_dir+"discretized_array.npy")
    single_frequencies, double_frequencies=compute_frequencies(Discretized_Array)
    np.save(output_dir+'frequencies/frequencies_single.npy', single_frequencies)
    np.save(output_dir+'frequencies/frequencies_double.npy', double_frequencies)

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
    print("Number of clusters before finale merging: "+str(variables.number_clusters_extend_data))
    if variables.number_clusters_extend_data>1 :
        yacare.compare_final_clusters(variables)
    yacare.find_final_clusters(variables)
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

    


