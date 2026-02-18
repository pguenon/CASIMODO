import os
import sys
import argparse
import shutil

sys.path.append(os.getcwd()) # Add current directory to path for module imports

from CASIMODO_utils.functions_CASIMODO import *

#######################################
#         ARGUMENT PARSING            #
#######################################

def parse_arguments():
    parser = argparse.ArgumentParser(description='CASIMODO - Conformational Analysis via Statistical Inference of MOlecular Dynamics Observables')
    
    parser.add_argument('--step_to_perform', type=str, default='all', choices=['all','discretize_coordinates','cluster_coordinates','get_conformations','plot_conformations_time'] , help='Step to perform in the pipeline')

    parser.add_argument('--topol_file', type=str, required=True, help='Path to topology file')
    parser.add_argument('--trj_file', type=str, required=True, help='Path to trajector/y file')
    parser.add_argument('--dic_file', type=str, default='dic_important_atoms_protein.txt', help='Path to important atoms dictionary')
    parser.add_argument('--output_directory', type=str, default='results/', help='Output directory')
    

    parser.add_argument('--time_zero', type=float, default=0., help='Time (ps) to start analysis')
    parser.add_argument('-dt', '--delta_time', type=float, default=1.0, help='Time (ps) between frames to consider')

    parser.add_argument('--cutoff_distance', type=int, default=5, help='Distance cutoff (A) to define contact')
    parser.add_argument('--proba_under_cutoff_distance', type=float, default=0.01, help='Probability cutoff for filtering contacts')
    
    parser.add_argument('--prominence', type=float, default=0.01, help='Prominence for minima detection in discretization')
    parser.add_argument('--smooth_factor', type=float, default=10.0, help='Smoothing factor for discretization larger values mean KDE closer to histogram')
    parser.add_argument('--n_points_per_bin', type=int, default=1000, help='Number of points per bin for histogram discretization')
    parser.add_argument('--min_bin_size_distances', type=float, default=0.05, help='Minimum distance between bins in discretization in Angstroms')
    parser.add_argument('--min_bin_size_angles', type=float, default=0.5, help='Minimum distance between bins in discretization in degrees')
    
    parser.add_argument('--cutoff_npoints_discretization', type=int, default=100000, help='Maximum number of points to use for discretization')

    parser.add_argument('--save_data', type=int, default=1, choices=[0, 1], help='Whether to save data (1 for True, 0 for False)')
    parser.add_argument('--save_all_plots', type=int, default=0, choices=[0, 1], help='Whether to save all plots (1 for True, 0 for False)')
    parser.add_argument('--extension_plots', type=str, default='png', choices=['png', 'pdf', 'svg'], help='File extension for saved plots')
    parser.add_argument('--resolution_plots', type=int, default=200, help='Resolution (dpi) for saved plots')

    parser.add_argument('--method_clustering_coordinates', type=str, default='advanced_density_peaks', choices=['advanced_density_peaks', 'hdbscan', 'yacare','ward','k-means'], help='Clustering method for coordinates')
    parser.add_argument('--parameters_clustering_coordinates', nargs='*', type=float, default=[3.0, 1], help='Parameters for clustering coordinates (e.g., Z_parameter and halo_parameter)')

    parser.add_argument('--method_clustering_conformations', type=str, default='advanced_density_peaks', choices=['advanced_density_peaks', 'hdbscan', 'yacare','ward','k-means'], help='Clustering method for conformations')
    parser.add_argument('--parameters_clustering_conformations', nargs='*', type=float, default=[3.0, 0], help='Parameters for clustering conformations (e.g., Z_parameter and halo_parameter)')
    parser.add_argument('--cluster_of_coordinates_to_process', type=int, default=-1, help='Index of the cluster of coordinates to process (default: -1 for all clusters)')

    parser.add_argument('--minimal_size_to_cluster', type=int, default=20, help='Minimal size to perform clustering (if number of states is below this value, no clustering is performed)')

    parser.add_argument('--cutoff_len_states', type=int, default=100000, help='Cutoff for the number of states to consider in clustering states')

    parser.add_argument('--cutoff_proba_conformations', type=float, default=0.001, help='Probability cutoff for conformations extraction')
    parser.add_argument('--split_trajectory', type=int, default=1, choices=[0, 1], help='Whether to split the trajectory by conformations (1 for True, 0 for False)')

    parser.add_argument('--coordinates_to_add', nargs='*', default=[], help='List of additional coordinate files')
    parser.add_argument('--type_coordinates_to_add', nargs='*', default=[], help='List of coordinate types (same order)')
    parser.add_argument('--residues_coordinates_to_add', nargs='*', default=[], help='List of residues to consider for additional coordinates (e.g., 161_162)')



    return parser.parse_args()


args = parse_arguments()

#######################################
#        VARIABLE INITIALIZATION      #
#######################################

output_dir = args.output_directory
if len(output_dir)>0 and output_dir[-1] != '/':
    output_dir += '/'

split_trajectory_int = args.split_trajectory
# Convert split_trajectory to boolean
if split_trajectory_int not in [0, 1]:
    raise ValueError("split_trajectory must be 0 (False) or 1 (True).") 
split_trajectory = bool(split_trajectory_int)

save_data = bool(args.save_data)
save_all_plots = bool(args.save_all_plots)

config={
    'step_to_perform': args.step_to_perform,
    'topolfile': args.topol_file,
    'trajfile': args.trj_file,
    'dic': args.dic_file,
    'output_dir': output_dir,
    'time_zero': args.time_zero,
    'delta_time': args.delta_time,
    'cutoff_distance': args.cutoff_distance,
    'proba_under_cutoff_distance': args.proba_under_cutoff_distance,
    'prominence': args.prominence,
    'smooth_factor': args.smooth_factor,
    'n_points_per_bin': args.n_points_per_bin,
    'min_bin_size_distances': args.min_bin_size_distances,
    'min_bin_size_angles': args.min_bin_size_angles,
    'cutoff_npoints_discretization': args.cutoff_npoints_discretization,
    'save_data': save_data,
    'save_all_plots': save_all_plots,
    'extension_plots': args.extension_plots,
    'resolution_plots': args.resolution_plots,
    'method_clustering_coordinates': args.method_clustering_coordinates,
    'parameters_clustering_coordinates': args.parameters_clustering_coordinates,
    'method_clustering_conformations': args.method_clustering_conformations,
    'parameters_clustering_conformations': args.parameters_clustering_conformations,
    'cluster_of_coordinates_to_process': args.cluster_of_coordinates_to_process,
    'minimal_size_to_cluster': args.minimal_size_to_cluster,
    'cutoff_len_states': args.cutoff_len_states,
    'cutoff_proba_conformations': args.cutoff_proba_conformations,
    'split_trajectory': split_trajectory,
    'coordinates_to_add': args.coordinates_to_add,
    'type_coordinates_to_add': args.type_coordinates_to_add,
    'residues_coordinates_to_add': args.residues_coordinates_to_add
}

step_to_perform = config['step_to_perform']
topolfile = config['topolfile']
trajfile = config['trajfile']
dic = config['dic']
coordinates_to_add = config['coordinates_to_add']
cluster_of_coordinates_to_process = config['cluster_of_coordinates_to_process']

#######################################
#     CHECK INPUT FILE EXISTENCE      #
#######################################

for path in [topolfile, trajfile, dic]:
    if not os.path.exists(path):
        print(f"Error: File '{path}' does not exist.")
        exit(1)

########################################
#       CREATE OUTPUT DIRECTORY        #    
########################################
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

########################################
#           INITIATE LOGGING            #
########################################
initiate_logging(config)

########################################
#           PRINT HEADER               #
########################################
print_header()

#######################################
#         PRINT INPUTS                #
#######################################
print_inputs(config)


#######################################
#       OPEN TRAJECTORY (if needed)   #
#######################################

if step_to_perform in ['all', 'discretize_coordinates','get_conformations','precompute_positions']:
    u_traj = open_trajectory(config)

#######################################
#         TIME FILTERING              #
#######################################

if step_to_perform == 'all' :
    subdirs = [
    'discretizing_npy',
    ]
    for subdir in subdirs:
        if os.path.exists(os.path.join(output_dir, subdir)):
            shutil.rmtree(os.path.join(output_dir, subdir))  # Remove existing directory  
        os.mkdir(os.path.join(output_dir, subdir))
    times, times_indices = filter_times_and_indices(u_traj,config)

#######################################
#        GET IMPORTANT ATOMS          #
#######################################

if step_to_perform in ['all', 'discretize_coordinates', 'get_conformations', 'precompute_positions']:
    important_atoms_file = os.path.join(output_dir, 'important_atoms.txt')
    if os.path.exists(important_atoms_file):
        os.remove(important_atoms_file)
    
    important_atoms, selected_resids, selected_resnames, indices_aa, indices_na_pyrimidine, indices_na_purine = get_important_atoms_MDA(u_traj, config)
    save_important_atoms(important_atoms, selected_resids, selected_resnames, indices_aa, indices_na_pyrimidine, indices_na_purine, config)

if step_to_perform in ['all', 'precompute_positions']:
    precompute_all_positions(u_traj, important_atoms, selected_resids,indices_aa,indices_na_pyrimidine,indices_na_purine, config)

#######################################
#     DISCRETIZE CONFORMATIONS        #
#######################################

if step_to_perform in ['all', 'discretize_coordinates']:
    subdirs = [
    'coordinates_plots',
    'analysis_npy',
    'information_plots'   
    ]

    for subdir in subdirs:
        if os.path.exists(os.path.join(output_dir, subdir)):
            shutil.rmtree(os.path.join(output_dir, subdir))  # Remove existing directory  
        os.mkdir(os.path.join(output_dir, subdir))
    if save_data :
        subdir = 'coordinates_data' 
        if os.path.exists(os.path.join(output_dir, subdir)):
            shutil.rmtree(os.path.join(output_dir, subdir))  # Remove existing directory  
        os.mkdir(os.path.join(output_dir, subdir))

    selected_coordinates_file = os.path.join(output_dir, 'selected_coordinates.txt')
    if os.path.exists(selected_coordinates_file):
        os.remove(selected_coordinates_file)

    get_contacts(u_traj, important_atoms, selected_resids, config)
    
    if len(indices_aa)!=0 :
        get_dihedrals_protein(u_traj, indices_aa, config)

    if len(indices_na_pyrimidine) != 0 or len(indices_na_purine) != 0:
        get_dihedrals_nucleic_acids(u_traj, indices_na_pyrimidine, indices_na_purine, config)
        
    if len(coordinates_to_add) != 0:
        add_coordinates(config)

    get_discretized_array(config)
    compute_information(config)

    

#######################################
#           CLUSTERING STEPS           #
#######################################

if step_to_perform in ['all','cluster_coordinates']:
    cluster_coordinates(config)
    
if step_to_perform in ['all','get_conformations']:
    subdirs = [
    'conformations_clustering'
    ]
    if cluster_of_coordinates_to_process == -1 :
        for subdir in subdirs:
            if os.path.exists(os.path.join(output_dir, subdir)):
                shutil.rmtree(os.path.join(output_dir, subdir))  # Remove existing directory  
            os.mkdir(os.path.join(output_dir, subdir))
    else:
        subdir = 'conformations_clustering/trajectories_cluster_' + str(cluster_of_coordinates_to_process)
        if os.path.exists(os.path.join(output_dir, subdir)):
            shutil.rmtree(os.path.join(output_dir, subdir))
        file_png = 'conformations_clustering/distances_between_states_cluster_' + str(cluster_of_coordinates_to_process) + '.png'
        file_ndx = 'conformations_clustering/frames_conformations_from_cluster_of_CV_' + str(cluster_of_coordinates_to_process) + '.ndx'
        if os.path.exists(os.path.join(output_dir, file_png)):
            os.remove(os.path.join(output_dir, file_png))
        if os.path.exists(os.path.join(output_dir, file_ndx)):
            os.remove(os.path.join(output_dir, file_ndx))


    get_conformations_from_clusters(u_traj,selected_resids,config)

if step_to_perform in ['all','plot_conformations_time']:
    plot_conformations_as_function_of_time(config)
    
print_ending_message(config)
