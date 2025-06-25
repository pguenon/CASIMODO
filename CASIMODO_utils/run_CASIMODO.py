import os
import sys
import argparse

sys.path.append(os.getcwd()) # Add current directory to path for module imports

from CASIMODO_utils.functions_CASIMODO import *

#######################################
#         ARGUMENT PARSING            #
#######################################

def parse_arguments():
    parser = argparse.ArgumentParser(description='CASIMODO - Conformational Information & Dynamics Entropy Reader')
    
    parser.add_argument('-struc', type=str, required=True, help='Path to GRO structure file')
    parser.add_argument('-trj', type=str, required=True, help='Path to trajector/y file')
    parser.add_argument('-dic', type=str, default='dic_terminal_atoms_protein.txt', help='Path to terminal atoms dictionary')
    parser.add_argument('--o_dir', type=str, default='results/', help='Output directory')
    
    parser.add_argument('--cutoff_distances', type=int, default=5, help='Distance cutoff (Å) to define contact')
    parser.add_argument('--delta_resid', type=int, default=3, help='Residue separation threshold for contact filtering')
    parser.add_argument('--time_zero', type=int, default=150000, help='Time (ps) to start analysis')
    parser.add_argument('--size_block', type=int, default=50000, help='Size of block (ps) for analysis')
    parser.add_argument('-dt', '--delta_time', type=int, default=1, help='Time (ps) between frames to consider')
    parser.add_argument('--height_cutoff', type=float, default=5.0, help='Minimum height difference (%) to define peaks')
    
    parser.add_argument('--coordinates_to_add', nargs='*', default=[], help='List of additional coordinate files')
    parser.add_argument('--type_coordinates_to_add', nargs='*', default=[], help='List of coordinate types (same order)')
    parser.add_argument('--barycenter_coordinates_to_add', nargs='*', default=[], help='List of atoms used as barycenters')
    
    parser.add_argument('--step_to_perform', type=str, default='all', help='Step to perform in the pipeline')
    parser.add_argument('--number_of_states_to_show', type=int, default=10, help='Number of states to show after clustering')
    
    return parser.parse_args()


args = parse_arguments()

#######################################
#        VARIABLE INITIALIZATION      #
#######################################

strucfile = args.struc
trajfile = args.trj
dic = args.dic
output_dir = args.o_dir.rstrip('/') + '/'

cutoff_distances = args.cutoff_distances
delta_resid = args.delta_resid
time_zero = args.time_zero
size_block = args.size_block
delta_time = args.delta_time
height_cutoff = args.height_cutoff

coordinates_to_add = args.coordinates_to_add
type_coordinates_to_add = args.type_coordinates_to_add
barycenter_coordinates_to_add = args.barycenter_coordinates_to_add

step_to_perform = args.step_to_perform
number_of_states_to_show = args.number_of_states_to_show


#######################################
#     CHECK INPUT FILE EXISTENCE      #
#######################################

for path in [strucfile, trajfile, dic]:
    if not os.path.exists(path):
        print(f'Error: File "{path}" does not exist.')
        exit(1)

#######################################
#      CREATE OUTPUT DIRECTORIES      #
#######################################

subdirs = [
    'coordinates_data',
    'coordinates_plots',
    'Positions_npy',
    'analysis',
    'MI_plots',
    'frequencies'
]

for subdir in subdirs:
    os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)


########################################
#           PRINT HEADER               #
########################################
print_header()


#######################################
#       OPEN TRAJECTORY (if needed)   #
#######################################

if step_to_perform in ['all', 'discretize_conformations', 'get_distances_between_coordinates']:
    u_traj = open_trajectory(strucfile, trajfile)

#######################################
#         TIME FILTERING              #
#######################################

if step_to_perform == 'all':
    times, times_indices = filter_times_and_indices(u_traj, time_zero, delta_time, output_dir)

#######################################
#        GET TERMINAL ATOMS           #
#######################################

if step_to_perform in ['all', 'discretize_conformations', 'get_distances_between_coordinates']:
    terminal_atoms_file = os.path.join(output_dir, 'terminal_atoms.txt')
    if os.path.exists(terminal_atoms_file):
        os.remove(terminal_atoms_file)
    
    terminal_atoms, RESIDS_SELECTED, RESNAMES_SELECTED, indices_aa = get_terminal_atoms_MDA(u_traj, dic)
    save_terminal_atoms(terminal_atoms, RESIDS_SELECTED, RESNAMES_SELECTED, output_dir)

    print("\nSelected residues:")
    for resid, resname in zip(RESIDS_SELECTED, RESNAMES_SELECTED):
        if resid not in indices_aa:
            print(f" {resname} - {resid} ")
        else:
            print(f" {resname} - {resid} (AA) ")


#######################################
#     DISCRETIZE CONFORMATIONS        #
#######################################

if step_to_perform in ['all', 'discretize_conformations']:
    selected_coordinates_file = os.path.join(output_dir, 'selected_coordinates.txt')
    if os.path.exists(selected_coordinates_file):
        os.remove(selected_coordinates_file)

    get_contacts(
        u_traj, terminal_atoms, RESIDS_SELECTED, time_zero, size_block,
        delta_time, cutoff_distances, delta_resid, height_cutoff, indices_aa, output_dir
    )

    get_dihedrals(
        u_traj, indices_aa, time_zero, size_block,
        delta_time, cutoff_distances, delta_resid, height_cutoff, output_dir
    )

    add_coordinates(
        coordinates_to_add, type_coordinates_to_add,
        output_dir, time_zero, size_block, height_cutoff
    )

    get_discretized_array(output_dir)

#######################################
#    ANALYSIS OF CONFORMATIONS        #
#######################################

if step_to_perform in ['all', 'get_frequencies']:
    get_frequencies(output_dir)


if step_to_perform in ['all', 'run_sbm']:
    #compute_couplings_with_SBM(output_dir)  
    extract_couplings_between_residues(output_dir)  

if step_to_perform in [ 'get_distances_between_coordinates']:
    get_positions_baricenters(
        u_traj, output_dir, RESIDS_SELECTED, indices_aa,
        terminal_atoms, coordinates_to_add, barycenter_coordinates_to_add
    )
    get_avg_distances_barycenters(output_dir)



if step_to_perform in [ 'get_mutual_information']:
    get_mutual_information(output_dir)

if step_to_perform in [ 'get_entropy']:
    get_entropy(output_dir)

#######################################
#           CLUSTERING STEP           #
#######################################

if step_to_perform == 'clusterize_MI':
    for f in ['Clusters_of_coordinate_from_MI.txt', 'resids_in_cluster_from_MI.txt']:
        f_path = os.path.join(output_dir, f)
        if os.path.exists(f_path):
            os.remove(f_path)

    clusterize_MI(
        output_dir,
        coordinates_to_add,
        barycenter_coordinates_to_add,
        step_to_perform,
        number_of_states_to_show
    )

#######################################
#     EXTRACT REPRESENTATIVE FRAMES   #
#######################################

if step_to_perform == 'extract_conformations':
    cluster_states(output_dir)
