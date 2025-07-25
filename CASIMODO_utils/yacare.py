# yacare.py

"""
YACARE
==========

This module implements the YACARE algorithm described in the article "Clustering and noise detection through optimal reordering and contextual analysis" by Axel Descamps et al.
Eventhough it can be used in command line, it is advised to use it with the provided Jupyter notebook.
This module has been written by Nicolas Chéron and Axel Descamps.
"""

import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import cluster
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import homogeneity_completeness_v_measure
from sklearn.metrics.cluster import fowlkes_mallows_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import davies_bouldin_score
from sklearn.metrics import silhouette_score
from datetime import datetime
import itertools
import copy
import sys

print(f"We are using Python {sys.version[0:6]}, numpy {np.version.version}, sci-kit learn {sklearn.__version__}, yacare 0.99e from 2025-07-24.")
print("For help on a given function, type for example help(yacare.perform_first_reordering).")

class Variables:
    # Variables that will be defined during the execution and kept fixed.
    project_name = "_"
    file_name = "_"
    save_images = False
    show_images = True
    raw_data = []
    distance_matrix = []
    num_elements = 0
    num_coords = 0
    size_moving_square = 0
    raw_data_is_distance_matrix = False

    # Variables to keep track of what has been done.
    reordering_has_been_done = False
    merging_has_been_done = False
    extending_data_has_been_done = False
    writing_indices_has_been_done = False

    # Variables used during the execution, and then use in other functions.
    #...in perform_first_reordering
    integral_deltaD = 0
    delta_diagonal = []
    reordered_matrix = []
    indices_by_closest_element = []
    
    #...in choose_if_we_reorder_again
    delta_diagonal_new_ordering = []
    reordered_matrix_new_ordering = []
    indices_by_closest_element_new_ordering = []

    #...in find_optimal_cutoff
    minimal_size_cluster = 0
    delta_diagonal_findCutoff = []
    selected_cutoff = 0
    selected_cutoff_automatic = 0
    ratios = 0
    cutoff_list = 0
    cluster_borders = 0

    #...in find_final_clusters
    borders = []
    number_clusters = 0
    size_clusters = []
    elements_inside_clusters = []
    elements_outside_clusters = []
    representative_structures_in_original_index = []

    #...in compare_clusters
    distance_inside_matrix_final = []
    stddev_inside_matrix_final = []

    #...in propose_list_for_concatenating_clusters
    clusters_to_merge = []

    #...in concatenate_clusters
    number_clusters_with_merging = 0
    reordered_matrix_with_merging = []
    elements_inside_clusters_with_merging = []
    representative_structures_in_original_index_with_merging = []
    borders_with_merging = []
    
    #...in expand_clusters
    number_clusters_extend_data = 0
    elements_outside_clusters_extend_data = []
    reordered_matrix_with_noise = []
    elements_inside_clusters_with_noise = []
    representative_structures_in_original_index_with_noise = []
    borders_with_noise = []
    
    #...in write_indices
    number_clusters_write_indices = 0
    elements_inside_clusters_write_indices = []
    elements_outside_clusters_write_indices = []

    #...in confusion_matrix
    labels_yacare = []
    labels_hdbscan = []
    labels_optics = []
    labels_kmeans = []
    labels_density_peaks = []
        
########################################################### Useful functions

def get_delta_diagonal(matrix, size_moving_square):
    """
    Compute the delta_diagonal by analyzing the matrix.

    Parameters:
    matrix (numpy.ndarray): The input matrix to analyze.
    size_moving_square (int): The size of the moving square used for calculations.

    Returns:
    list: A list containing the delta_diagonal values.
    """
    
    delta_diagonal = []
    
    #Due to the stencil, we can't start from 0 and and can't stop at the end of the matrix.
    for i in range(size_moving_square, len(matrix) - size_moving_square):
        # Define sub-matrices for calculations.
        submatrix_before      = matrix[(i - size_moving_square):i, (i - size_moving_square):i]
        submatrix_after       = matrix[i:(i + size_moving_square), i:(i + size_moving_square)]
        submatrix_out_diagonal = matrix[(i - size_moving_square):i, i:(i + size_moving_square)]
        # Append the calculated mean to delta_diagonal.
        delta_diagonal.append(2 * np.mean(submatrix_out_diagonal) - np.mean(submatrix_before) - np.mean(submatrix_after))
    return delta_diagonal

###########################################################

def reorder_matrix(matrix, starting_point):
    """
    Reorder a given matrix starting from a specified element and store the reordered indices in a list.

    Parameters:
    matrix (numpy.ndarray): The input matrix to reorder.
    starting_point (int): The index of the element to start the reordering from.

    Returns:
    tuple: A tuple containing the reordered matrix and a list of reordered indices.
    """
    
    # Initialize the list;
    indices_by_closest_element = []
    
    # Find the maximal distance in the matrix to set a high value for the diagonal elements.
    # This ensures that the diagonal elements are not chosen during the reordering process.
    max_distance = np.max(matrix)

    # Initialize matrices for reordering.
    # The distance_matrix has 0s on the diagonal, and we will use a working_matrix similar to the distance_matrix
    # but with values higher than max_distance on the diagonal (we choose 2*max_distance+1). The idea is to be able
    # to easily look for the minimum in a row without picking the value from the diagonal (which would otherwise be 0).
    diagonal_value = int(2 * max_distance + 1)
    working_matrix = matrix.copy()
    np.fill_diagonal(working_matrix, diagonal_value)

    # Initialize the new list.
    a = starting_point
    indices_by_closest_element.append(a)
        
    # Reorder elements based on minimum distance.
    # We extract the row with index a and look for the element with the lowest distance to a. The column with index a is
    # then changed to 2*max_distance+1, so that when we will start from the new reference a' we are sure that we will
    # not pick again the previous value a on the row. This avoids deleting lines and columns, and make it easier to handle indices.
    for _ in range(len(matrix) - 1):
        row_reference = working_matrix[a]
        index_closest_element = np.argmin(row_reference)
        working_matrix[:, a] = diagonal_value
        a = index_closest_element
        indices_by_closest_element.append(index_closest_element)
    
    # Reorder the distance matrix according to the new order of structures (first the lines, then the columns).
    reordered_matrix = matrix[indices_by_closest_element, :]
    reordered_matrix = reordered_matrix[:, indices_by_closest_element]

    # Conclude.
    return reordered_matrix, indices_by_closest_element

###########################################################

def get_reorder_parameter(num_elements):
    """
    Prompt the user to choose a reordering parameter for the matrix.

    Parameters:
    num_elements (int): The number of elements in the matrix.

    Returns:
    tuple: A tuple containing the chosen reordering type and an optional number of extra points or starting index.
    """
    
    print("You will have to choose how you want to reorder the distance matrix. The options are the following ones:")
    print("  1. None       : this option is not recommended, we will use the matrix above.")
    print("  2. Centroid   : we will reorder by using the centroids of clusters found during the first reordering.")
    print("  3. Random     : we will reorder by using elements randomly choosen (you will provide the number of elements).")
    print("  4. Evenly     : we will reorder by using elements evenly distributed (you will provide the number of elements).")
    print("  5. All        : we will reorder by using all elements (this is the recommended choice).")
    print("  6. Predecided : we will reorder by using a given starting element that you provide, usefull to restart from a previous reordering.")
    
    # Choose how to reorder the matrix.
    while True:
        try:
            choice = input("Enter the number corresponding to your choice and press Enter (and not Maj+Enter): ")
            if choice == '1':
                result = 'None'
            elif choice == '2':
                result = 'Centroid'
            elif choice == '3':
                result = 'Random'
            elif choice == '4':
                result = 'Evenly'
            elif choice == '5':
                result = 'All'
            elif choice == '6':
                result = 'Predecided'
            else:
                print("Invalid choice. Please choose again.")
                continue
            print(f'The chosen type of reordering is: {result}')
            if result in ['Random', 'Evenly', 'Predecided']:
                while True:
                    try:
                        number_extra_points = int(input(f"Enter the number of points for {result.lower()} reordering (must be strictly lower than {num_elements}): "))
                        if 0 <= number_extra_points < num_elements:
                            return result, number_extra_points
                        else:
                            print(f"Invalid input. Please enter a value lower than {num_elements}.")
                    except ValueError:
                        print("Invalid input. Please enter an integer value.")
            else:
                return result, None
        except EOFError:
            print("\nInput stream closed. Exiting.")
            sys.exit(1)
        
###########################################################

def get_clusters(matrix, indices_by_closest_element, min_percentage_deltaD, max_percentage_deltaD, minimal_size_cluster, size_moving_square, subdivision_deltaD, function_for_ratio):    
    """
    Identify clusters in the matrix based on delta_diagonal values and compute the sum of ratios for different cut-off values. We will also get the list
    of representative structures (used for reordering with Centroid). The indices that are in the outputs start at 0.

    Parameters:
    matrix (numpy.ndarray): The input matrix to analyze.
    indices_by_closest_element (list): List of indices ordered by closest element.
    min_percentage_deltaD (float): Minimum percentage of delta_diagonal to consider for cut-off.
    max_percentage_deltaD (float): Maximum percentage of delta_diagonal to consider for cut-off.
    minimal_size_cluster (int): Minimum size of a cluster as a percentage of the total number of elements.
    size_moving_square (int): The size of the moving square used for calculations.
    subdivision_deltaD (int): Number of subdivisions for delta_diagonal percentage range.
    function_for_ratio (int): Function to compute the ratio (1 for length/variance, 2 for inverse of variance).

    Returns:
    tuple: A tuple containing lists of ratios, cutoff values, cluster borders, and indices to try for reordering.
    """

    # Initialize lists.
    ratios = []
    cutoff_list = []
    cluster_borders = []
    indices_to_try = []

    # Compute delta_diagonal by iterating over the matrix.
    delta_diagonal = get_delta_diagonal(matrix, size_moving_square)
    
    # Iterate over cutoff values. Inputs are percentages. We will try 'subdivision_deltaD' values (200 by default, i.e. we want to do it every 0.5% of the range).
    for i in range(subdivision_deltaD + 1):
        # Get the percentage of cutoff to look at.
        percentage_cutoff = min_percentage_deltaD + (max_percentage_deltaD - min_percentage_deltaD) * i / subdivision_deltaD
        # Get the cutoff value for the current iteration.
        cutoff = np.min(delta_diagonal) + (np.max(delta_diagonal) - np.min(delta_diagonal)) * (percentage_cutoff / 100)
        cutoff_list.append(cutoff)       
        
        # Identify indices below the cutoff.
        indices_below_cutoff = [  k + size_moving_square for k in range(len(delta_diagonal)) if delta_diagonal[k] <= cutoff  ]

        # Group consecutive indices from the reordered matrix into clusters.
        clusters_indices = []
        cluster_indiv = []
        diff = np.diff(indices_below_cutoff)
        for j in range(len(indices_below_cutoff) - 1):
            # If two values in indices_below_cutoff follow each other, they are from the same cluster, and we add one to cluster_indiv.
            if diff[j] == 1:
                cluster_indiv.append(indices_below_cutoff[j])
            # Else, add the last index to the cluster, add the list of indices from cluster_indiv to clusters_indices and reset cluster_indiv.
            else:
                cluster_indiv.append(indices_below_cutoff[j])
                clusters_indices.append(np.array(cluster_indiv))
                cluster_indiv = []
        # Add the last cluster to the list.
        clusters_indices.append(np.array(cluster_indiv))
    
        # Define limits for each cluster.
        borders = [  [clust[0], clust[-1]] for clust in clusters_indices if len(clust) != 0 and (clust[-1] - clust[0] + 1) >= int((minimal_size_cluster / 100) * len(matrix))  ]
        cluster_borders.append(borders)
        
        # Find the representative structure for each cluster.
        for border in borders:
            temporary_matrix = np.array(matrix[border[0]:border[1]+1, border[0]:border[1]+1])    #The +1 is here to include the last index
            mean_row_from_temporary_matrix = np.mean(temporary_matrix, axis=0)
            representative_structure_for_cluster = border[0] + np.argmin(mean_row_from_temporary_matrix)
            indices_to_try.append(indices_by_closest_element[representative_structure_for_cluster])
       
        # Extract matrices for all clusters, based on boundaries.
        clusters = []
        for border in borders:
            cluster_temp = np.array(matrix[border[0]:border[1]+1, border[0]:border[1]+1])        #The +1 is here to include the last index
            clusters.append(cluster_temp)
        
        # Calculate the "ratio value" for each cluster. It can be defined in two different ways:
        # * either the ratio for each cluster is the length of the cluster divided by the variance (i.e. square of the standard deviation) in it;
        # * or the ratio for each cluster is the inverse of the variance.
        cluster_ratios = []
        for clust in clusters:
            # Taking the standard variance doesn't make sense because it would take into account the diagonal which is made of 0s.
            # Thus, we change it with NewVariance=OldVariance*(l*l)/(l*l-l)=OldVariance*l/(l-1) (where l is the size of the cluster).
            variance = np.std(clust)**2
            variance = variance * len(clust[0]) / (len(clust[0]) - 1)
            # Add a condition on the variance which is mainly useful for clusters with 2 elements.
            if variance != 0:
                if function_for_ratio == 1:
                    cluster_ratios.append(len(clust[0]) / variance)
                elif function_for_ratio == 2:
                    cluster_ratios.append(1 / variance)
        
        # Calculate the sum of ratio values for all clusters at this cutoff.
        if len(cluster_ratios) != 0:
            ratio_value = np.sum(cluster_ratios)
        else:
            ratio_value = 0

        # Add ratio_value to the the list storing all ratios.
        ratios.append(ratio_value)

    # Remove duplicates from indices_to_try since we have looked for clusters for a bunch of values of cutoff.
    indices_to_try = np.unique(indices_to_try)

    # Conclude.
    return ratios, cutoff_list, cluster_borders, indices_to_try

###########################################################

def find_nearest(array, value):
    """
    Find the nearest value to a given value in an array.

    Parameters:
    array (numpy.ndarray): The array to search.
    value (float): The value to find the nearest to.

    Returns:
    tuple: A tuple containing the index of the nearest value and the nearest value itself.
    """
    
    array = np.asarray(array)
    index = (np.abs(array - value)).argmin()
    return index, array[index]

###########################################################

# Between numpy 1 and 2 the function to integrate has changed its name.
major_numpy_version = int(np.__version__.split('.')[0])
if major_numpy_version <= 1:
    integrate = np.trapz
elif major_numpy_version > 1:
    integrate = np.trapezoid

########################################################### Functions below are called in the notebook.

def load_data(variables, delimiter=",", comments=('#', '@'), dtype=np.float32, usecols=None):
    """
    Load data from a file and determine if it is a distance matrix or a set of features.

    Parameters:
    variables (Variables): An instance of the Variables class to store the loaded data.
    delimiter (str): The string used to separate values. Default is ','.
    comments (tuple): Characters used to indicate the start of a comment. Default is ('#', '@').
    dtype (data-type): The data type of the resulting array. Default is np.float32.
    usecols (int or sequence, optional): Which columns to read, with 0 being the first. Default is None.

    Returns:
    None

    Comments:
    Before running Yacare, you must prepare your data. You can load two kind of files:
    * either a N*N matrix which contains the distance between each pair of datapoint (here N is the number of datapoint),
    * or a file with N lines, where each line contains M descriptors of each datapoint. The N*N matrix will then be computed below.
    With this function, you can load one of these two files. If the data have the same amount of lines than columns and if the diagonal is made with 0s, then we assume you want to load a distance matrix. Otherwise, we assume you want to load a set of features, and we will compute the distance matrix with a euclidean norm.
    * By default, the delimeter between data is ",", lines with comments start with '@' or '#', and we load 32-bit floats. You can change it with the delimeter, comments and dtype parameters (by default, in the function we have: delimiter=",", comments=['@', '#'], dtype=np.float32).
    * By default, all columns are loaded. To load a text file using columns 2 to N (if the label of each datapoint is in the first column for example), add "usecols=range(1, N-1)" as a parameter (because the first column is labelled 0).
    * Once the distance matrix is loaded or computed, you can save it in a binary file with: np.save(variables.ProjectName + "_DistanceMatrix.npy", variables.distance_matrix)
    * If you want to load a binary file: variables.distance_matrix = np.load(variables.FileName)
    Note that the more data you have, the more RAM you need. If the distance matrix needs to be computed, extra memory are needed. Loading a 20000*20000 matrix takes a couple of minutes usually. If you are low on RAM because of a huge file, check the user manual. For more precision, you can load 64-bit floats.

    """

    # Load the data.
    raw_data = np.loadtxt(variables.file_name, delimiter=delimiter, comments=comments, dtype=dtype, usecols=usecols)
    variables.raw_data = raw_data

    # Determine if the data is a distance matrix.
    raw_data_is_distance_matrix = False
    if raw_data.shape[0] == raw_data.shape[1]:
        if np.allclose(np.diag(raw_data), 0):
            raw_data_is_distance_matrix = True

    # Store the distances.
    if raw_data_is_distance_matrix == True:
        print("We are using a distance matrix.")
        variables.raw_data_is_distance_matrix = True
        variables.distance_matrix = raw_data      # Change to '1 - raw_data' if you have a similarity index
    else:
        print("We are using a set of data with features, and we will compute the distance matrix.")
        variables.raw_data_is_distance_matrix = False
        # Normalize the data.
        raw_data = StandardScaler().fit_transform(raw_data)
        # If you have enough RAM, this should be the fastest. If you are low on RAW, check the user manual for other way of computing the distance matrix.
        diffs = raw_data[:, np.newaxis, :] - raw_data[np.newaxis, :, :]
        variables.distance_matrix = np.linalg.norm(diffs, axis=2)

###########################################################

def perform_first_reordering(variables, percentage_moving_square, vmax=-1):
    """
    Perform the first reordering of the distance matrix and compute the delta_diagonal.

    Parameters:
    variables (Variables): An instance of the Variables class to store the reordered data.
    percentage_moving_square (float): The percentage of the data to use for the moving square.
    vmax (int, optional): The maximum value for the color scale in the plots. Default is -1.

    Returns:
    None

    Comments:
    * This function reorders the data starting from the first frame.
    * The "percentage_moving_square" variable can be changed by the user; it represents the size of the stencil moving along the diagonal. This value is a percentage of the data, we propose 1.0% by default. If the plot of Delta_d appears noisy, increase the value to 2.0%.
    * If you have outliers that pollutes the presentation of the matrices and you want to restrict the range of data in the colorbar, you can add the parameter "vmax=..." in the function. By default, vmax is the maximal value in the distance matrix. This parameter can be added to all the functions that display a distance matrix.

    """
    
    # Correct the default value of vmax.
    vmax = vmax if vmax != -1 else np.max(variables.distance_matrix)
        
    # Reset the variables to False because otherwise, when the percentage_moving_square is changed, the program will still remember the old choices.
    variables.reordering_has_been_done = False
    variables.merging_has_been_done = False
    variables.extending_data_has_been_done = False
    
    # Check the matrix.
    if not np.allclose(np.diag(variables.distance_matrix), 0):
        print(f"WARNING: Your distance matrix don't have 0s on the diagonal. Please check your distance matrix. If you used a similarity index, set it to 1 - distance_matrix.")
    
    # Get shape of the input matrix.
    variables.num_elements, variables.num_coords = np.shape(variables.distance_matrix)
    if variables.num_elements != variables.num_coords:
        print("WARNING: your matrix is not squared")
        sys.exit(1)

    # Define number of data for the square moving along the diagonal. We divide by 2 because there will be size_moving_square data before and after the middle of the square.
    variables.size_moving_square = int((percentage_moving_square / 2) * variables.num_elements // 100)

    # If we have choosen a too low percentage_moving_square, we must be sure that size_moving_square is not 0.
    if variables.size_moving_square == 0:
        variables.size_moving_square = 1

    # Print some data.
    print(f'Number of elements: {variables.num_elements}')
    print(f"Data are going from {np.min(variables.distance_matrix)} to {np.max(variables.distance_matrix)}.")

    # Print in the output file.
    summary_file = open(variables.project_name + "_Yacare_Summary.txt", "w")
    summary_file.write("===== Summary file for the clustering with Yacare =====\n")
    # From https://asciiart.cc/view/11873
    summary_file.write("                                            ___.-----.______                  \n")
    summary_file.write("                                  ___.-----'::::::::::::::::`---.___          \n")
    summary_file.write("               _.--._            (:::;,-----'~~~~~`----::::::::::.. `-.       \n")         
    summary_file.write("  _          .'_---. `--.__       `~~'                 `~`--.:::::`..  `..    \n")
    summary_file.write(" ; `-.____.-' ' {0} ` `--._`---.____                         `:::::::: : ::   \n")
    summary_file.write(":_^              ~   `--.___ `----.__`----.____                ~::::::.`;':   \n")
    summary_file.write(" :`--.__,-----.___(         `---.___ `---.___  `----.___         ~|;:,' : |   \n")
    summary_file.write("  `-.___,---.____     _,        ._  `----.____ `----.__ `-----.___;--'  ; :   \n")
    summary_file.write("                 `---' `.  `._    `))  ,  , , `----.____.----.____   --' :|   \n") 
    summary_file.write("                       / `,--.\    `.` `  ` ` ,   ,  ,     _.--   `-----'|'   \n")
    summary_file.write("_.~~~~~~._____    __./'_/'     :   .:----.___ ` ` ` ``  .-'      , ,  :::'    \n")
    summary_file.write("                ///--\;  ____  :   :'    ____`---.___.--::     , ` ` ::'      \n")
    summary_file.write("                `'           _.'   (    /______     (   `-._   `-._,-'        \n")
    summary_file.write("                          .-' __.-//     /_______---'       `-._   `.         \n") 
    summary_file.write("              ~~~        /////    `'       ~~~~~~      ~~ ______;   ::.       \n")
    summary_file.write("                         `'`'                            /_______   _.'       \n")
    summary_file.write("                   ~~~                  ~~~~~~~~           /___.---'  --__    \n")
    summary_file.write("                                                            ~~~               \n")
    summary_file.write("                                                                              \n")
    summary_file.write(f"***** Started on {datetime.today().strftime('%Y-%m-%d %H:%M:%S')} *****\n")
    summary_file.write(f"°°°°° The project name is {variables.project_name} °°°°°\n")
    summary_file.write(f"^^^^^ We are working with the file {variables.file_name} ^^^^^\n")
    summary_file.write(f"Number of elements: {variables.num_elements}\n")
    summary_file.write(f"The size of the stencil (i.e. the moving square) is {percentage_moving_square} % of the data.\n")

    # Reorder the matrix starting from the first element (0) and get a list that will store ordered indices.
    variables.reordered_matrix, variables.indices_by_closest_element = reorder_matrix(variables.distance_matrix, starting_point=0)

    # Compute delta_diagonal for the reordered matrix.
    variables.delta_diagonal = get_delta_diagonal(variables.reordered_matrix, variables.size_moving_square)
        
    # Get integral of delta_diagonal.
    variables.integral_deltaD = integrate(variables.delta_diagonal)
    print(f"First integral of delta_diagonal: {round(variables.integral_deltaD, 3)}")
    summary_file.write(f"First integral of delta_diagonal: {round(variables.integral_deltaD, 3)}\n")
    summary_file.write("\n")

    # Close the output file.
    summary_file.close()

    # Plot the original and reordered matrices. 
    plt.figure(figsize=(24, 12))
    plt.subplot(1, 2, 1)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title('Original distance matrix', size=20)
    plt.imshow(variables.distance_matrix, cmap='terrain', vmax=vmax)
    cbar = plt.colorbar(shrink=0.75)
    cbar.ax.tick_params(labelsize=16)

    plt.subplot(1, 2, 2)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title('Reordered distance matrix', size=20)
    plt.imshow(variables.reordered_matrix, cmap='terrain', vmax=vmax)
    cbar = plt.colorbar(shrink=0.75)
    cbar.ax.tick_params(labelsize=16)
    if variables.save_images == True:
        plt.savefig(variables.project_name + "_Yacare_1-Matrix-FirstReordering.png", bbox_inches='tight', pad_inches=0.1, dpi=150)
    if variables.show_images == True:
        plt.show()
    plt.close()

    # Plot delta_D.
    plt.figure(figsize=(24, 6))
    plt.subplot(1,1,1)
    plt.plot(range(variables.size_moving_square, variables.num_elements - variables.size_moving_square), variables.delta_diagonal)
    plt.xlabel('Index', size=18)
    plt.ylabel(r'$\Delta_d$', size=18)
    plt.xlim(0, variables.num_elements)
    #plt.ylim(0,)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    if variables.save_images == True:
        plt.savefig(variables.project_name + "_Yacare_2-DeltaD.png", bbox_inches='tight', pad_inches=0.1, dpi=150)
    if variables.show_images == True:
        plt.show()
    plt.close()

###########################################################

def choose_if_we_reorder_again(variables, indices=[], vmax=-1):
    """
    Choose whether to reorder the distance matrix again based on user input or on provided indices.

    Parameters:
    variables (Variables): An instance of the Variables class to store the reordered data.
    indices (list, optional): List of indices to use for reordering. Default is an empty list.
    vmax (int, optional): The maximum value for the color scale in the plots. Default is -1.

    Returns:
    None

    Comments:
    This function allow to choose how you want to reorder your data. Several options are possible, that are described once you run the cell.
    * Usually, option 2 (Centroid) is enough and quite fast.
    * If you want to dig more and try more starting points for the reordering, you can try options 3, 4 or 5.
    * If you have less than ~2000 datapoints, you can try all the starting points (option 5), it should only take a few minutes.
    * With more data, you can still try all the starting points but we recommand to do it in command line and parallelize it. This can be done by adding in the function the parameter "indices=list(range(0,2000))" to try as starting points indices from 0 to 1999 (e.g.).
    * If you already know what is the best starting point, choose option 6.
    Some details on the 'Centroid' option: we divide the range of Delta_d from its minimal value to 50% of its maximal value in 200 parts (this can be increased to 100%, but we didn't see a benefit of doing so).
    For each value of Delta_d, we look for clusters and search the centroid of each cluster (i.e. the representative structure). The list of all centroids from all values of Delta_d are then merged.
    We remove redundant data, and use this list as new starting points to test for reordering.

    """

    # Correct the default value of vmax.
    vmax = vmax if vmax != -1 else np.max(variables.distance_matrix)
        
    # Reset some variables to False because otherwise, when reordering is done with another option, the program will still remember the old choices.
    variables.reordering_has_been_done = True
    variables.merging_has_been_done = False
    variables.extending_data_has_been_done = False
    
    summary_file = open(variables.project_name + "_Yacare_Summary.txt", "a")
        
    # If the indices array was not specified, ask a question and proceed.
    if len(indices) == 0:
        # Ask what to do.
        reorder_option, number_extra_points = get_reorder_parameter(variables.num_elements)
        if reorder_option not in ['None', 'Centroid', 'Random', 'Evenly', 'All', 'Predecided']:
            print("Invalid reordering option returned. Exiting.")
            sys.exit(1)
        if reorder_option in ['Random', 'Evenly', 'Predecided']:
            if number_extra_points is None or number_extra_points < 0 or number_extra_points >= variables.num_elements:
                print("Invalid number of extra points returned. Exiting.")
                sys.exit(1)
    
        # Print in the output file.
        summary_file.write(f'The chosen type of reordering is: {reorder_option}' + "\n")

        ########################### Manage the chosen option.

        if reorder_option == 'None':
            indices_to_try = []

        elif reorder_option == 'Centroid':
            # Get the clusters and the list of indices to try. We start with the lowest value of delta_D (min_percentage_deltaD=0) and cut the range in 200 parts
            # (every 0.5%). We set max_percentage_deltaD at 50 because taking the first 50% of cutoff is usually enough (it is only used here to find new starting
            # points to reorder the distance matrix). The minimal cluster size is set at 3% for this stage of reordering. This choice does not really matter, it is
            # only used to propose other starting points for the reordering. Another choice of size will be done later for actually finding clusters. The
            # function_for_ratio is set to its default value at 1.
            ratios, cutoff_list, borders_of_clusters, indices_to_try = get_clusters(matrix = variables.reordered_matrix,
                indices_by_closest_element = variables.indices_by_closest_element, min_percentage_deltaD = 0, max_percentage_deltaD = 50,
                minimal_size_cluster = 3.0, size_moving_square = variables.size_moving_square, subdivision_deltaD = 200, function_for_ratio = 1)

            print(f"List of indices that will be tried as starting points: {indices_to_try}")
            summary_file.write(f"List of indices that will be tried as starting points: {indices_to_try}\n")

        elif reorder_option == 'Random':
            # Generate a list of random integers within the range of num_elements.
            indices_to_try = np.random.choice(variables.num_elements, number_extra_points, replace=False)
            indices_to_try = np.unique(indices_to_try)

            print(f'Number of points for randomly reordering: {number_extra_points}')
            summary_file.write(f'Number of points for randomly reordering: {number_extra_points}' + "\n")
            print(f"List of indices that will be tried as starting points: {indices_to_try}")
            summary_file.write(f"List of indices that will be tried as starting points: {indices_to_try}\n")

        elif reorder_option == 'Evenly':
            # Generate a list of evenly separated integers within the range of num_elements.
            indices_to_try = np.arange(0, variables.num_elements, variables.num_elements // number_extra_points)

            print(f'Number of points for evenly reordering: {number_extra_points}')
            summary_file.write(f'Number of points for evenly reordering: {number_extra_points}' + "\n")
            print(f"List of indices that will be tried as starting points: {indices_to_try}")
            summary_file.write(f"List of indices that will be tried as starting points: {indices_to_try}\n")

        elif reorder_option == 'All':
            indices_to_try = np.arange(0, variables.num_elements)

            print(f"List of indices that will be tried as starting points: {indices_to_try}")
            summary_file.write(f"List of indices that will be tried as starting points: {indices_to_try}\n")

        elif reorder_option == 'Predecided':
            # In this case the variable has a wrong name, but we keep it like that to make it simple.
            indices_to_try = [number_extra_points]

            print(f"We will reorder with the starting point: {indices_to_try}")
            summary_file.write(f"We will reorder with the starting point: {indices_to_try}\n")
    else:
        indices_to_try = indices
        print(f"List of indices that will be tried as starting points: {indices_to_try}")
        summary_file.write(f"List of indices that will be tried as starting points: {indices_to_try}\n")
        
    ########################### Reorder the matrix according to the list of new starting points.
    
    # Start by initializing data.
    integral_deltaD_reference = copy.deepcopy(variables.integral_deltaD)
    variables.delta_diagonal_new_ordering = copy.deepcopy(variables.delta_diagonal)
    variables.indices_by_closest_element_new_ordering = copy.deepcopy(variables.indices_by_closest_element)
    index_starting_element_new_ordering = 0

    counter = 0     
    for k in range(len(indices_to_try)):
        # Display information regarding advances. This is approximate due to roundings, and we force the counter to stop at 100%.
        if len(indices_to_try) > 20:
            if k % int(len(indices_to_try)/20) == 0 and counter <= 100:
                print(f"        -------- {counter}% done --------")
                counter = counter + 5

        # Reorder the matrix and get a list that will store ordered indices.
        reordered_matrix_temp, indices_by_closest_element_temp = reorder_matrix(variables.distance_matrix, indices_to_try[k])

        # Compute delta_D by iterating over the reordered matrix.
        delta_diagonal_temp = get_delta_diagonal(reordered_matrix_temp, variables.size_moving_square)

        # Check if the current reordering is better than the previous best one.
        integral_deltaD_temp = integrate(delta_diagonal_temp)
        if integral_deltaD_temp < integral_deltaD_reference:
            if integral_deltaD_reference == variables.integral_deltaD:
                print(f"Starting index {indices_to_try[k]} is better than the previous one (0) since the integral of delta_D is lower: {round(integral_deltaD_temp, 3)} < {round(integral_deltaD_reference, 3)}")
                summary_file.write(f"Starting index {indices_to_try[k]} is better than the previous one (0) since the integral of delta_D is lower: {round(integral_deltaD_temp, 3)} < {round(integral_deltaD_reference, 3)}\n")
            else:
                print(f"Starting index {indices_to_try[k]} is better than the previous one ({indices_to_try[index_starting_element_new_ordering]}) since the integral of delta_D is lower: {round(integral_deltaD_temp, 3)} < {round(integral_deltaD_reference, 3)}")
                summary_file.write(f"Starting index {indices_to_try[k]} is better than the previous one ({indices_to_try[index_starting_element_new_ordering]}) since the integral of delta_D is lower: {round(integral_deltaD_temp, 3)} < {round(integral_deltaD_reference, 3)}\n")
            integral_deltaD_reference = integral_deltaD_temp
            index_starting_element_new_ordering = k
            variables.indices_by_closest_element_new_ordering = indices_by_closest_element_temp
            variables.delta_diagonal_new_ordering = delta_diagonal_temp

    if len(indices_to_try) != 0:
        best_reordering_index = indices_to_try[index_starting_element_new_ordering]
        print(f"The best reordering was obtained with the index {best_reordering_index}.")
        summary_file.write(f"The best reordering was obtained with the index {best_reordering_index}.\n")

    # In case no reordering was asked or done.
    if len(indices_to_try) == 0:
        print("No extra reordering is asked.")
        summary_file.write("No extra reordering is asked.\n")
    if len(indices_to_try) != 0 and best_reordering_index == 0:
        print("No extra reordering has been done.")
        summary_file.write("No extra reordering has been done.\n")

    # Apply the best reordering to the matrix.
    best_reordered_matrix = variables.distance_matrix[variables.indices_by_closest_element_new_ordering, :]
    variables.reordered_matrix_new_ordering = best_reordered_matrix[:, variables.indices_by_closest_element_new_ordering]

    # Plot the original and reordered matrices.
    plt.figure(figsize=(24, 12))
    plt.subplot(1, 2, 1)
    plt.imshow(variables.reordered_matrix, cmap='terrain', vmax=vmax)
    cbar = plt.colorbar(shrink=0.75)
    cbar.ax.tick_params(labelsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title('First reordered distance matrix', size=20)

    plt.subplot(1, 2, 2)
    plt.imshow(variables.reordered_matrix_new_ordering, cmap='terrain', vmax=vmax)
    cbar = plt.colorbar(shrink=0.75)
    cbar.ax.tick_params(labelsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title('Best reordered distance matrix', size=20)
    if variables.save_images == True:
        plt.savefig(variables.project_name + "_Yacare_3-Matrix-BestReordered.png", bbox_inches='tight', pad_inches=0.1, dpi=150)
    if variables.show_images == True:
        plt.show()
    plt.close()

    # Plot delta_D.
    plt.figure(figsize=(24, 6))
    plt.subplot(1,1,1)
    plt.plot(range(variables.size_moving_square, variables.num_elements - variables.size_moving_square), variables.delta_diagonal_new_ordering)
    plt.xlabel('Index', size=18)
    plt.ylabel(r'$\Delta_d$', size=18)
    plt.xlim(0, variables.num_elements)
    #plt.ylim(0,)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    if variables.save_images == True:
        plt.savefig(variables.project_name + "_Yacare_4-DeltaD-BestReordered.png", bbox_inches='tight', pad_inches=0.1, dpi=150)
    if variables.show_images == True:
        plt.show()
    plt.close()

    # Close the output file.
    summary_file.write("\n")
    summary_file.close()

###########################################################

def find_optimal_cutoff(variables, minimal_size_cluster, use_all_cutoff=True, function_for_ratio=1):
    """
    Find the optimal cutoff for clustering based on the delta_diagonal values.

    Parameters:
    variables (Variables): An instance of the Variables class to store the clustering data.
    minimal_size_cluster (float): Minimum size of a cluster as a percentage of the total number of elements.
    use_all_cutoff (bool): Whether to use all cutoff values or exclude the first 5%. Default is True.
    function_for_ratio (int): Function to compute the ratio (1 for length/variance, 2 for inverse of variance). Default is 1.
    
    Returns:
    None

    Comments:
    From the reordered matrix, we will look for a value of Delta_d that is the best to separate clusters.
    * The "minimal_size_cluster" variable can be changed by the user.
       ° A cluster must be at least this percentage of data points to be kept. To start, we recommand a value of 2.0%. Use a lower value (0.5%, e.g.) if you want to identify more clusters, or a larger value (4.0%, e.g.) if you want to identify fewer and larger clusters.
       ° Another possibility is to use the smallest possible value to identify as much clusters as possible, and then merge them. In that case, choose a very low value (0.01%, e.g.). If this a value leads to clusters that would be smaller than 2 points, the program will detect it and will use a value that is high enough to avoid issues (and it will warn you).
    * The "use_all_cutoff" parameter can be used to rescrict the range of Delta_d.
      ° When we compare value of Delta_d to find the optimal cut-off, by default we compare all cut-offs, i.e. all values of Delta_d between the minimal and maximal values per step of 0.5% (i.e. 200 values).
      ° If from the figure above you see that the lower values of Delta_d are noisy, we can skip the first 10% of values. To do this, add "use_all_cutoff = False" to the find_optimal_cutoff function (by default, it is set to True).
      ° If you don't know what to do, use True, i.e. don't add anything.
    * The "function_for_ratio" parameter can be used to change the way cut-offs are compared.
      ° Two mathematical functions are possible to find the optimal cutoff. Either Sum (length/variance) for all clusters at a given cut-off, or Sum (1/variance). By default, the fist one is used i.e. "function_for_ratio = 1".
      ° However, sometimes the size of clusters starts to dominate and we end up with a curve of "Sum of ratios" vs "Cut-off" that looks like a sigmoid with no well-defined peak. In such cases, we advice to use the second function.
      ° Using the second function will change the way the "Sum of ratios" is computed and will often solve problems. To do this, add "function_for_ratio = 2" in the find_optimal_cutoff function (by default, the choice is 1).

    """
 
    # Reset some variables to False because otherwise, when the minimal_size_cluster is changed, the program will still remember the old choices.
    variables.merging_has_been_done = False
    variables.extending_data_has_been_done = False
    
    variables.minimal_size_cluster = minimal_size_cluster
    # If minimal_size_cluster is too low, adapt it so that a cluster is made of at least two elements. There is no error on the equation, and if
    # we remove one "* 100" and the "/ 100" we don't get the same result.
    if int((minimal_size_cluster / 100) * variables.num_elements) < 2:
        variables.minimal_size_cluster = ((2 * 100 * 100 // variables.num_elements) + 1 ) / 100
        print(f"WARNING: we have changed your value of minimal_size_cluster because it was too low. We are using the value {variables.minimal_size_cluster}.")

    # Set the value of minimal_cutoff based on the 'use_all_cutoff' variable. If set to 'False', it removes the first 5% (=10/200).
    minimal_cutoff = 0 if use_all_cutoff == True else 10

    # Choose which data to work on.
    if variables.reordering_has_been_done == True:
        variables.delta_diagonal_find_cutoff = copy.deepcopy(variables.delta_diagonal_new_ordering)
        reordered_matrix_find_cutoff = variables.reordered_matrix_new_ordering
        indices_by_closest_element_find_cutoff = copy.deepcopy(variables.indices_by_closest_element_new_ordering)
    else:
        variables.delta_diagonal_find_cutoff = copy.deepcopy(variables.delta_diagonal)
        reordered_matrix_find_cutoff = variables.reordered_matrix
        indices_by_closest_element_find_cutoff = copy.deepcopy(variables.indices_by_closest_element)

    # Get clusters and the sum of ratios to then find the optimal cutoff.
    variables.ratios, variables.cutoff_list, variables.borders_of_clusters, indices_to_try = get_clusters(matrix = reordered_matrix_find_cutoff,
        indices_by_closest_element = indices_by_closest_element_find_cutoff, min_percentage_deltaD = minimal_cutoff, max_percentage_deltaD = 100,
        minimal_size_cluster = variables.minimal_size_cluster, size_moving_square = variables.size_moving_square, subdivision_deltaD = 200,
        function_for_ratio = function_for_ratio)

    # Keep the cutoff with the highest value for the sum of ratios.
    variables.selected_cutoff = variables.cutoff_list[np.argmax(variables.ratios)]
    variables.selected_cutoff_automatic = variables.selected_cutoff
    print(f'The automated selected cut-off is {round(variables.selected_cutoff, 5)}. At this level, there are {len(variables.borders_of_clusters[np.argmax(variables.ratios)])} clusters.')

    # Print in the output file.
    summary_file = open(variables.project_name + "_Yacare_Summary.txt", "a")
    summary_file.write(f'The minimal cluster size is {variables.minimal_size_cluster}%.\n')
    summary_file.write(f'The function to compute the sum of ratios was the function number {function_for_ratio}.\n')
    summary_file.write(f'The automated selected cut-off is {round(variables.selected_cutoff, 5)}. At this level, there are {len(variables.borders_of_clusters[np.argmax(variables.ratios)])} clusters.\n')
    summary_file.write('\n')
    summary_file.close()

    # Plot all ratios.
    plt.figure(figsize=(24, 6))
    plt.subplot(1,1,1)
    plt.scatter(variables.cutoff_list, variables.ratios)
    plt.xlabel('Cut-off', size=18)
    plt.ylabel('Sum of ratios', size=18)
    #plt.xlim(0,)
    #plt.ylim(0,)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    if variables.save_images == True:
        plt.savefig(variables.project_name + "_Yacare_5-BestRatios.png", bbox_inches='tight', pad_inches=0.1, dpi=150)
    if variables.show_images == True:
        plt.show()
    plt.close()

    # Plot delta_D with cutoff.
    color = itertools.cycle(('tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'))
    plt.figure(figsize=(24, 6))
    plt.subplot(1,1,1)
    plt.plot(range(variables.size_moving_square, variables.num_elements - variables.size_moving_square), variables.delta_diagonal_find_cutoff)
    plt.axhline(variables.selected_cutoff, color='gray', label='Cut-off', linewidth=0.5)
    for i in range(0, len(variables.borders_of_clusters[np.argmax(variables.ratios)])):
        xmin = variables.borders_of_clusters[np.argmax(variables.ratios)][i][0] / variables.num_elements
        xmax = variables.borders_of_clusters[np.argmax(variables.ratios)][i][1] / variables.num_elements
        plt.axhline(xmin=xmin, xmax=xmax, y=variables.selected_cutoff, color=next(color), linewidth=3)
    plt.xlabel('Index', size=18)
    plt.ylabel(r'$\Delta_d$', size=18)
    plt.legend(loc='upper left', fontsize=14)
    plt.xlim(0, variables.num_elements)
    plt.ylim(0, None)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    if variables.save_images == True:
        plt.savefig(variables.project_name + "_Yacare_6-DeltaD-WithClusters.png", bbox_inches='tight', pad_inches=0.1, dpi=150)
    if variables.show_images == True:
        plt.show()
    plt.close()

###########################################################

def change_proposed_cutoff(variables):
    """
    Change the proposed cutoff value for clustering based on user input.

    Parameters:
    variables (Variables): An instance of the Variables class to store the clustering data.

    Returns:
    None

    Comments:
    * The Delta_d graph is shown, and you can decide to increase or decrease the proposed cut-off to either separate a large cluster in two, or to merge two clusters that are close.
    * Merging at this stage should be used with care, it is not because two clusters are separated and just below a peak that they should be merged by increasing to cut-off the go above the peak.
    * When this cell is ran, you will have to manually write the value of Delta_d that you want to use as a cut-off.

    """

    # Reset some variables to False because otherwise, if we change the cutoff, the program will still remember the old choices.
    variables.merging_has_been_done = False
    variables.extending_data_has_been_done = False
    
    # Recall some information.
    print(f'The automated selected cut-off is {round(variables.selected_cutoff_automatic, 6)}. At this level, there are {len(variables.borders_of_clusters[np.argmax(variables.ratios)])} clusters.')

    # Print in the output file.
    summary_file = open(variables.project_name + "_Yacare_Summary.txt", "a")

    # Choose the new cut-off.
    while True:
        try:
            variables.selected_cutoff = float(input("Please enter a value for the cut-off and press Enter (and not Maj+Enter): "))
            if variables.selected_cutoff < 0 or variables.selected_cutoff > max(variables.delta_diagonal_find_cutoff):
                print(f"Invalid input. Please enter a value between 0 and {max(variables.delta_diagonal_find_cutoff)}.")
            else:
                break
        except ValueError:
            print("Invalid input. Please enter a valid float value.")

    closest_index, closest_value = find_nearest(variables.cutoff_list, variables.selected_cutoff)
    print(f'The closest value from the pre-tested cut-offs is {round(closest_value, 3)} and at this value there were {len(variables.borders_of_clusters[closest_index])} clusters.')
    summary_file.write(f'You have chosen {variables.selected_cutoff} for the cut-off. The closest value of pre-tested cut-off is {round(closest_value, 3)} and at this value there are {len(variables.borders_of_clusters[closest_index])} clusters.\n')

    # Plot delta_D with cutoff.
    plt.figure(figsize=(24, 6))
    plt.subplot(1,1,1)
    plt.plot(range(variables.size_moving_square, variables.num_elements - variables.size_moving_square), variables.delta_diagonal_find_cutoff)
    plt.axhline(variables.selected_cutoff_automatic, color='gray', label='Proposed cut-off', linewidth=0.6)
    color = itertools.cycle(('tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'))
    for i in range(0, len(variables.borders_of_clusters[np.argmax(variables.ratios)])):
        plt.axhline(xmin=(variables.borders_of_clusters[np.argmax(variables.ratios)][i][0])/variables.num_elements, xmax=(variables.borders_of_clusters[np.argmax(variables.ratios)][i][1])/variables.num_elements, y=variables.selected_cutoff_automatic, color=next(color), linewidth=3)
    plt.axhline(variables.selected_cutoff, color='darkgray', label='New cut-off', linewidth=0.6, linestyle='--')
    color = itertools.cycle(('tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'))
    for i in range(0, len(variables.borders_of_clusters[closest_index])):
        plt.axhline(xmin=(variables.borders_of_clusters[closest_index][i][0])/variables.num_elements, xmax=(variables.borders_of_clusters[closest_index][i][1])/variables.num_elements, y=variables.selected_cutoff, color=next(color), linewidth=3, linestyle='--')
    plt.xlabel('Index', size=18)
    plt.ylabel(r'$\Delta_d$', size=18)
    plt.legend(loc='upper left', fontsize=14)
    plt.xlim(0, variables.num_elements)
    plt.ylim(0, None)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    if variables.save_images == True:
        plt.savefig(variables.project_name + "_Yacare_6-DeltaD-WithClusters-ManuallyChanged.png", bbox_inches='tight', pad_inches=0.1, dpi=150)
    if variables.show_images == True:
        plt.show()
    plt.close()

    # Close the output file.
    summary_file.write("\n")
    summary_file.close()

###########################################################
   
def find_final_clusters(variables, vmax=-1):
    """
    Identify final clusters in the reordered matrix based on the cutoff value, and display them.

    Parameters:
    variables (Variables): An instance of the Variables class to store the clustering data.
    vmax (int, optional): The maximum value for the color scale in the plots. Default is -1.

    Returns:
    None
    """
    
    # Correct the default value of vmax.
    vmax = vmax if vmax != -1 else np.max(variables.distance_matrix)
 
    # Choose which data to work on.
    if variables.reordering_has_been_done == True:
        delta_diagonal_find_clusters = copy.deepcopy(variables.delta_diagonal_new_ordering)
        reordered_matrix_find_clusters = variables.reordered_matrix_new_ordering
        indices_by_closest_element_find_clusters = copy.deepcopy(variables.indices_by_closest_element_new_ordering)
    else:
        delta_diagonal_find_clusters = copy.deepcopy(variables.delta_diagonal)
        reordered_matrix_find_clusters = variables.reordered_matrix
        indices_by_closest_element_find_clusters = copy.deepcopy(variables.indices_by_closest_element)

    # Identify indices below the cutoff.
    indices_below_cutoff = [  i + variables.size_moving_square for i in range(len(delta_diagonal_find_clusters)) if delta_diagonal_find_clusters[i] <= variables.selected_cutoff  ]

    # Group consecutive indices from the reordered matrix into clusters.
    clusters_indices = []
    cluster_indiv = []
    diff = np.diff(indices_below_cutoff)
    for j in range(len(indices_below_cutoff) - 1):
        # If two values in indices_below_cutoff follow each other, they are from the same cluster, and we add one to cluster_indiv.
        if diff[j] == 1:
            cluster_indiv.append(indices_below_cutoff[j])
        # Else, add the last index to the cluster, add the list of indices from cluster_indiv to clusters_indices and reset cluster_indiv.
        else:
            cluster_indiv.append(indices_below_cutoff[j])
            clusters_indices.append(np.array(cluster_indiv))
            cluster_indiv = []
    # Add the last cluster to the list.
    clusters_indices.append(np.array(cluster_indiv))

    # Keep indices of the kept clusters and those outside the clusters.
    # Values in all_kept_clusters_indices / outside_clusters_indices are the indices from the reordered matrix.
    all_kept_clusters_indices = []
    outside_clusters_indices = []
    variables.borders = []
    for clust in clusters_indices:
        if len(clust) != 0 and (clust[-1] - clust[0] + 1) >= int((variables.minimal_size_cluster / 100) * variables.num_elements):
            all_kept_clusters_indices.append(clust)
            variables.borders.append([clust[0], clust[-1]])
    for i in range(variables.num_elements):
        if i not in np.concatenate(all_kept_clusters_indices).ravel().tolist():
            outside_clusters_indices.append(i)
    variables.number_clusters = len(all_kept_clusters_indices)

    # Get indices of elements inside and outside clusters, using indices from the reordered matrix.
    # Values in elements_inside_clusters / elements_outside_clusters are the indices in indices_by_closest_element_find_clusters, i.e. from the raw data.
    variables.elements_inside_clusters  = []
    variables.elements_outside_clusters = []
    for i in range(variables.number_clusters):
        cluster_temp = []
        for j in range(len(all_kept_clusters_indices[i])):
             cluster_temp.append(indices_by_closest_element_find_clusters[all_kept_clusters_indices[i][j]])
        variables.elements_inside_clusters.append(cluster_temp)
    for i in range(variables.num_elements):
        if i not in np.concatenate(variables.elements_inside_clusters).ravel().tolist():
            variables.elements_outside_clusters.append(i)

    # Extract matrices for all clusters, based on boundaries.
    clusters = []
    for brdr in variables.borders:
        cluster_temp = np.array(reordered_matrix_find_clusters[brdr[0]:brdr[1]+1, brdr[0]:brdr[1]+1])
        clusters.append(cluster_temp)

    # Initialize lists to store mean row values, size of clusters, the representative structure indices, and the representative structure indices from the raw data.
    mean_distance_on_row = []
    variables.size_clusters = []
    representative_structures = []
    variables.representative_structures_in_original_index = []

    # Loop over clusters to find the size for each cluster.
    for i in range(variables.number_clusters):
        variables.size_clusters.append(len(variables.elements_inside_clusters[i]))

    # Loop over clusters to calculate the mean value of distance of each row.
    for clust in clusters:
        mean_distance_on_row.append(np.mean(clust, axis=0))

    # Loop over mean_distance_on_row to determine representative structure indices based on mean row values.
    for i in range(len(mean_distance_on_row)):
        representative_structures.append(variables.borders[i][0] + np.argmin(mean_distance_on_row[i]))

    # Loop over representative_structures to map representative structure indices to original indices.
    for i in representative_structures:
        variables.representative_structures_in_original_index.append(indices_by_closest_element_find_clusters[i]+1)
 
    # Plot the original and reordered matrices.
    plt.figure(figsize=(24, 12))
    plt.subplot(1, 2, 1)
    plt.imshow(variables.distance_matrix, cmap='terrain', vmax=vmax)
    cbar = plt.colorbar(shrink=0.75)
    cbar.ax.tick_params(labelsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title('Original distance matrix', size=20)

    plt.subplot(1, 2, 2)
    plt.imshow(reordered_matrix_find_clusters, cmap='terrain', vmax=vmax)
    cbar = plt.colorbar(shrink=0.75)
    cbar.ax.tick_params(labelsize=16)

    for i in range(len(variables.borders)):
        x0 = variables.borders[i][0]
        x1 = variables.borders[i][1]
        plt.axvline(x=x0, ymin=1-x0/variables.num_elements, ymax=1-x1/variables.num_elements, color='red')
        plt.axvline(x=x1, ymin=1-x0/variables.num_elements, ymax=1-x1/variables.num_elements, color='red')
        plt.axhline(y=x0, xmin=x0/variables.num_elements, xmax=x1/variables.num_elements, color='red')
        plt.axhline(y=x1, xmin=x0/variables.num_elements, xmax=x1/variables.num_elements, color='red')
        for j in range(len(variables.borders)):
            if j != i:
                x2 = variables.borders[j][0]
                x3 = variables.borders[j][1]
                plt.axvline(x=x0, ymin=1-x2/variables.num_elements, ymax=1-x3/variables.num_elements, color='purple', ls='--', lw='0.5')
                plt.axvline(x=x1, ymin=1-x2/variables.num_elements, ymax=1-x3/variables.num_elements, color='purple', ls='--', lw='0.5')
                plt.axhline(y=x2, xmin=x0/variables.num_elements, xmax=x1/variables.num_elements, color='purple', ls='--', lw='0.5')
                plt.axhline(y=x3, xmin=x0/variables.num_elements, xmax=x1/variables.num_elements, color='purple', ls='--', lw='0.5')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title('Reordered distance matrix with clusters', size=20)
    if variables.save_images == True:
        plt.savefig(variables.project_name + "_Yacare_7-Matrix-ReorderedWithClusters.png", bbox_inches='tight', pad_inches=0.1, dpi=150)
    if variables.show_images == True:
        plt.show()
    plt.close()

    print(f'At the cut-off {round(variables.selected_cutoff, 3)}, clusters include {round(float((100*np.sum(variables.size_clusters)/variables.num_elements)), 1)}% of the trajectory.')
    print(f'We have found {variables.number_clusters} clusters and their representative structures are (indices start at 1): {variables.representative_structures_in_original_index}.')
    print(f'The sizes of the clusters are respectively {variables.size_clusters}.')
    print('The out-of-diagonal zones in dashed purple are the ones that will be compared on the next cell.')

    # Print in the output file.
    summary_file = open(variables.project_name + "_Yacare_Summary.txt", "a")
    summary_file.write(f'At the cut-off {round(variables.selected_cutoff, 3)}, clusters include {round(float((100*np.sum(variables.size_clusters)/variables.num_elements)), 1)}% of the trajectory.\n')
    summary_file.write(f'We have found {variables.number_clusters} clusters and their representative structures are (indices start at 1): {variables.representative_structures_in_original_index}.\n')
    summary_file.write(f'The sizes of the clusters are respectively {variables.size_clusters}.\n')
    summary_file.write("\n")
    summary_file.close()

###########################################################

def compare_clusters(variables, display_stddev = False, display_mean_distances = False):
    """
    Compare clusters by plotting the mean distance and standard deviation within and between clusters.

    Parameters:
    variables (Variables): An instance of the Variables class containing clustering data.
    display_stddev (bool): Whether to display the standard deviation in the plot. Default is False.
    display_mean_distances (bool): Whether to display the mean distances in the plot. Default is False.

    Returns:
    None

    Comments:
    * This function displays a figure that helps to compare the clusters. This is not mandatory, it is only a visual help.
    * By default, the values of the mean distance and standard deviation in each zone are not displayed.
    * To display the standard deviation in each square, add "display_stddev = True" in the function. To display the mean distance, add "display_mean_distances = True". By default, both parameters are set to False.

    """
    
    # Choose the size of each square for the image. The value must be adapted for each case because it will depend on the system.
    # ~30 lines below we have tried to propose an automated way to compute size_scaling. If the image is ugly, you must comment the
    # size_scaling line that is ~40 lines below and manually pick one. Note that the displayed size of an zone in the matrix is
    # the square root of its zone (all zones will be displayed as squares).
    #size_scaling = 0.0025

    # Choose which data to work on.
    if variables.reordering_has_been_done == True:
        reordered_matrix_compare_clusters = variables.reordered_matrix_new_ordering
    else:
        reordered_matrix_compare_clusters = variables.reordered_matrix

    # Compute the mean value of the distance in each cluster (in the diagonal) and in each out-of-diagonal rectangle.
    distance_inside_matrix = []
    stddev_inside_matrix = []
    for i in range(variables.number_clusters):
        for j in range(variables.number_clusters):
            distance_inside_matrix.append(np.mean(reordered_matrix_compare_clusters[variables.borders[i][0]:variables.borders[i][1], variables.borders[j][0]:variables.borders[j][1]]))
            stddev_inside_matrix.append(np.std(reordered_matrix_compare_clusters[variables.borders[i][0]:variables.borders[i][1], variables.borders[j][0]:variables.borders[j][1]]))
    variables.distance_inside_matrix_final = np.array(distance_inside_matrix).reshape(variables.number_clusters, variables.number_clusters)
    variables.stddev_inside_matrix_final = np.array(stddev_inside_matrix).reshape(variables.number_clusters, variables.number_clusters)
    # For the clusters, taking the full mean and full stddev doesn't make sense because it would take into account the diagonal which is made of 0s.
    # The current mean of clusters is {Sum_i Sum_j (d_ij)} / {l*l}, whereas we want {Sum_i Sum_j (d_ij)} / {l*l-l} (where l is the size of the cluster).
    # Thus, we change the values with NewMean=OldMean*(l*l)/(l*l-l)=OldMean*l/(l-1). We do the same for the standard deviation.
    for i in range(variables.number_clusters):
        cluster_size = variables.borders[i][1]-variables.borders[i][0]+1
        variables.distance_inside_matrix_final[i][i] = variables.distance_inside_matrix_final[i][i]*cluster_size/(cluster_size-1)
        variables.stddev_inside_matrix_final[i][i] = np.sqrt((variables.stddev_inside_matrix_final[i][i]**2)*cluster_size/(cluster_size-1))

    # Get the size of zones from the distance matrix.
    zone_sizes = []
    zone_all_sizes = []
    # Compute the length of each zone.
    for i in range(variables.number_clusters):
        zone_sizes.append(variables.borders[i][1]-variables.borders[i][0])
    # Compute the size of each zone (cluster or off-diagonal part).
    for i in range(variables.number_clusters):
        for j in range(variables.number_clusters):
            zone_all_sizes.append(zone_sizes[i]*zone_sizes[j])
    # Reshape in a 2D array.
    zone_all_sizes_final = np.array(zone_all_sizes).reshape(variables.number_clusters, variables.number_clusters)

    # Try to define automatically the scaling_factor.
    # The idea is that the size occupied by the largest cluster is size_scaling*max(zone_all_sizes_final.flatten()).
    # This size is made of M^2 points. Since the figure will be 12*12 inches ("figsize=(12, 12)"), each cluster will have
    # at most 12/(num_clusters+1) inches (the +1 is to give some space around) for itself. Each point is markersize*1/72 inches,
    # and my understanding is that by defaut markersize=1.66. Thus, 12/(num_clusters+1) = M*1.66/72. So we can have access to M.
    # We add a 0.9 term to scale down a little bit the squares to have some space.
    size_scaling = 0.9*((12/(variables.number_clusters+1))/(1.66/72))**2/max(zone_all_sizes_final.flatten())

    # Shape of the matrix for the size of clusters.
    rows, cols = zone_all_sizes_final.shape

    # Coordinates for each element, and flatten the matrices for scatter.
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    x = x.flatten()
    y = y.flatten()

    # Plot.
    plt.figure(figsize=(12, 12))
    plt.scatter(x, y, s=size_scaling*zone_all_sizes_final.flatten(), c=variables.distance_inside_matrix_final.flatten(), alpha=0.6, marker='s')
    plt.xlim(-1, rows)
    plt.ylim(-1, cols)
    plt.xticks(range(rows), size=16)
    plt.yticks(range(cols), size=16)

    # Invert y axis to match the orientation.
    plt.gca().invert_yaxis()

    # Add text in the matrix.
    for i in range(variables.number_clusters):
        for j in range(variables.number_clusters):
            # Display the distances between clusters and/or the standard deviation.
            if display_stddev == True and display_mean_distances == True:
                    plt.text(i, j, "{:.3f}".format(variables.distance_inside_matrix_final[i, j]) + "\n" + "{:.3f}".format(variables.stddev_inside_matrix_final[i, j]), ha='center', va='center', color='black', size=10)
            elif display_stddev == True and display_mean_distances == False:
                    plt.text(i, j, "{:.3f}".format(variables.stddev_inside_matrix_final[i, j]), ha='center', va='center', color='black', size=10)
            elif display_stddev == False and display_mean_distances == True:
                    plt.text(i, j, "{:.3f}".format(variables.distance_inside_matrix_final[i, j]), ha='center', va='center', color='black', size=10)

    # Add title and colorbar. The title depends on the choosen options.
    if display_stddev == True and display_mean_distances == True:
        plt.title("size of a square = size of the corresponding zone in the matrix \n color of a square = mean value of the distances in the zone \n numbers in a square = mean value and standard deviation of the distances in the zone", size=10)
    elif display_stddev == True and display_mean_distances == False:
        plt.title("size of a square = size of the corresponding zone in the matrix \n color of a square = mean value of the distances in the zone \n number in a square = standard deviation of the distances in the zone", size=10)
    elif display_stddev == False and display_mean_distances == True:
        plt.title("size of a square = size of the corresponding zone in the matrix \n color of a square = mean value of the distances in the zone \n number in a square = mean value of the distances in the zone", size=10)
    else:
        plt.title("size of a square = size of the corresponding zone in the matrix \n color of a square = mean value of the distances in the zone", size=10)
    colorbar = plt.colorbar()
    colorbar.set_label('Mean distance in the cluster', fontsize=16)
    colorbar.ax.tick_params(labelsize=16)
    if variables.save_images == True:
        plt.savefig(variables.project_name + "_Yacare_8-CompareClusters.png", bbox_inches='tight', pad_inches=0.1, dpi=150)
    if variables.show_images == True:
        plt.show()
    plt.close()
    
    average_mu_clusters = 0
    average_mu_out_of_diagonal = 0
    for i in range(variables.number_clusters):
        average_mu_clusters += variables.distance_inside_matrix_final[i][i]
        for j in range(i+1, variables.number_clusters):
            average_mu_out_of_diagonal += variables.distance_inside_matrix_final[i][j]
    average_mu_clusters = average_mu_clusters / variables.number_clusters
    average_mu_out_of_diagonal = average_mu_out_of_diagonal / ((variables.number_clusters*variables.number_clusters-variables.number_clusters)/2)
    print(f"The mean of the average of distances inside clusters (i.e. from the diagonal) is {round(average_mu_clusters, 3)}, and the mean of the average of distances out of diagonal is {round(average_mu_out_of_diagonal, 3)}.")
    
###########################################################
           
def propose_list_for_concatenating_clusters(variables, threshold_variable, choice_merging_clusters=0):
    """
    Propose a list of clusters to be concatenated based on a threshold variable and a chosen merging strategy.

    Parameters:
    variables (Variables): An instance of the Variables class containing clustering data.
    threshold_variable (float): The threshold value used to determine if clusters should be merged.
    choice_merging_clusters (int, optional): The strategy for merging clusters. Default is 0.
        1 - Start from the smallest cluster.
        2 - Start from the largest cluster.
        3 - Start from the cluster with the highest number of neighbours.
        4 - Start from the cluster with the fewest number of neighbours.

    Returns:
    None

    Comments:
    Clusters can end up being separated in several smaller clusters. We propose here an automatic way to identify when clusters should be merged.
    * We will here create a list than you can later modify if needed, using your analysis of the distance matrices.
    * The merging of clusters is actually made in the next cell, in this cell we will only propose a list.
    * To build the list, you must decide how you want to proceed and a question will be asked when you run the cell (see below). There is no better choice than the others since it depends on your data and on the question that is asked. Nonetheless, option "3" is probably the most common and is inspired by the gromos clustering method; otherwise, options "1" and "2" can also be useful.
    * You can skip the question that will be asked by adding the parameter "choice_merging_clusters=3" (for example) to the function.
    * Some details on the construction of the list:
      ° From the matrix above, we compute the mean distance within the out-of-diagonal rectangle (in dashed purple) between clusters A and B, and we compare this mean distance to a threshold which is "mu+threshold_variable*sigma" where mu and sigma are the mean distance and standard deviation from the data in cluster A (in the diagonal).
      ° Cluster B will be merged with cluster A if the out-of-diagonal mean distance is below the threshold of distances found from A. Note that the problem is not symmetric, since it is possible to accept to merge B with A but not A with B, because mu and sigma are different for A and B, and the threshold distances are thus different.
      ° Since the problem is not symmetric, you must choose how you want to start merging clusters (the question will be asked when you run the cell):
          "1" to start from the smallest one
          "2" to start from the largest one
          "3" to start from the one with the highest number of neighbours
          "4" to start from the one with the fewest number of neighbours
      ° A decision matrix will be displayed. It must be analyzed by lines: on line i, if the column j is in green it means that cluster j is close enough to cluster i and they will be merged; otherwise, it is in red.
      ° "threshold_variable" is called alpha in the article describing YACARE and this variable can be changed by the user. We propose 1.0 by default, with the idea that for a normal distribution, within one standard deviation there are 68% of the data, and within 2 standard deviations there are 95% of the data.

    """
    
    prompt = ('Choose how you want to start merging clusters:\n'
           '1 - Start from the smallest one\n'
           '2 - Start from the largest one\n'
           '3 - Start from the one with the highest number of neighbours\n'
           '4 - Start from the one with the fewest number of neighbours\n'
           'Enter your choice: ')
                
    # Choose how we want to start merging clusters: from the smallest one, from the largest one, from the one with the highest number of neighbours, or from the one with the fewest number of neighbours.
    if choice_merging_clusters not in [1, 2, 3, 4]:
        while True:
            try:
                choice_merging_clusters = int(input(prompt))
                if choice_merging_clusters in [1, 2, 3, 4]:
                    print(f"You have choosen the option {choice_merging_clusters}.")
                    break
                else:
                    print("Invalid input. Please enter 1, 2, 3 or 4.")
            except ValueError:
                print("Invalid input. Please enter an integer value.")
   
    # Choose which data to work on.
    if variables.reordering_has_been_done == True:
        reordered_matrix_compare_clusters = variables.reordered_matrix_new_ordering
    else:
        reordered_matrix_compare_clusters = variables.reordered_matrix

    # We recompute the variables distance_inside_matrix_final and stddev_inside_matrix_final even if they were computed before, because it is fast and to avoid any issues.
    # Compute the mean value of the distance in each cluster (in the diagonal) and in each out-of-diagonal rectangle.
    distance_inside_matrix = []
    stddev_inside_matrix = []
    for i in range(variables.number_clusters):
        for j in range(variables.number_clusters):
            distance_inside_matrix.append(np.mean(reordered_matrix_compare_clusters[variables.borders[i][0]:variables.borders[i][1], variables.borders[j][0]:variables.borders[j][1]]))
            stddev_inside_matrix.append(np.std(reordered_matrix_compare_clusters[variables.borders[i][0]:variables.borders[i][1], variables.borders[j][0]:variables.borders[j][1]]))
    variables.distance_inside_matrix_final = np.array(distance_inside_matrix).reshape(variables.number_clusters, variables.number_clusters)
    variables.stddev_inside_matrix_final = np.array(stddev_inside_matrix).reshape(variables.number_clusters, variables.number_clusters)
    # For the clusters, taking the full mean and full stddev doesn't make sense because it would take into account the diagonal which is made of 0s.
    # The current mean of clusters is {Sum_i Sum_j (d_ij)} / {l*l}, whereas we want {Sum_i Sum_j (d_ij)} / {l*l-l} (where l is the size of the cluster).
    # Thus, we change the values with NewMean=OldMean*(l*l)/(l*l-l)=OldMean*l/(l-1). We do the same for the standard deviation.
    for i in range(variables.number_clusters):
        cluster_size = variables.borders[i][1]-variables.borders[i][0]+1
        variables.distance_inside_matrix_final[i][i] = variables.distance_inside_matrix_final[i][i]*cluster_size/(cluster_size-1)
        variables.stddev_inside_matrix_final[i][i] = np.sqrt((variables.stddev_inside_matrix_final[i][i]**2)*cluster_size/(cluster_size-1))
   
    # Create a decision matrix (filled with booleans) to know if two clusters are close to each others.
    # We work by line, and check if the out-of-diagonal values (of the mean distance in the zone) are within the threshold to the value from the diagonal on this line.
    # The following 8 lines can be improved to avoid nested loops (see next line), however, since this is done only once and on small matrices, we prefer to keep it like this to improve readibility.
    # decision_matrix_final = variables.distance_inside_matrix_final <= (variables.distance_inside_matrix_final.diagonal()[:, None] + threshold_variable * variables.stddev_inside_matrix_final.diagonal()[:, None])
    decision_matrix = []
    for i in range(variables.number_clusters):
        for j in range(variables.number_clusters):
            if variables.distance_inside_matrix_final[i][j] <= variables.distance_inside_matrix_final[i][i] + threshold_variable*variables.stddev_inside_matrix_final[i][i]:
                decision_matrix.append(1)
            else:
                decision_matrix.append(0)
    decision_matrix_final = np.array(decision_matrix).reshape(variables.number_clusters, variables.number_clusters)
  
    # Compute the number of neighbours of each cluster.
    cluster_neighbors = []
    for k in range(len(variables.size_clusters)):
        cluster_neighbors.append(np.sum(decision_matrix_final[k]))

    # Display the decision matrix indicating cluster proximity.
    print("The decision matrix is displayed below. It must be analyzed by lines: on line i, if the column j is in green it means that cluster j is close enough to cluster i, otherwise it is in red. Note that the matrix is not always symmetric.")
    plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1)
    plt.imshow(decision_matrix_final, cmap='RdYlGn', alpha=0.6)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title('Decision matrix to merge clusters', size=20)
    if variables.save_images == True:
        plt.savefig(variables.project_name + "_Yacare_9-DecisionMatrixToMergeClusters.png", bbox_inches='tight', pad_inches=0.1, dpi=150)
    if variables.show_images == True:
        plt.show()
    plt.close()

    # Look into the decision_matrix for each line, check when there are 1's without looking at already found clusters. Then remove empty lists.
    variables.clusters_to_merge = []
    clusters_to_merge_flat = []

    # If we start merging from the smallest cluster.
    if choice_merging_clusters == 1:
        for k in range(len(variables.size_clusters)):
            trial_list = []
            for u in range(len(variables.size_clusters)):
                if decision_matrix_final[np.argsort(variables.size_clusters)[k]][u] == 1:
                    if u not in clusters_to_merge_flat:
                        trial_list.append(u)
                        clusters_to_merge_flat.append(u)
            variables.clusters_to_merge.append(trial_list)
        variables.clusters_to_merge = [list_clusters for list_clusters in variables.clusters_to_merge if len(list_clusters) != 0]

    # If we start merging from the largest cluster.
    if choice_merging_clusters == 2:
        for k in range(len(variables.size_clusters)):
            trial_list = []
            for u in range(len(variables.size_clusters)):
                if decision_matrix_final[np.argsort(variables.size_clusters)[::-1][k]][u] == 1:
                    if u not in clusters_to_merge_flat:
                        trial_list.append(u)
                        clusters_to_merge_flat.append(u)
            variables.clusters_to_merge.append(trial_list)
        variables.clusters_to_merge = [list_clusters for list_clusters in variables.clusters_to_merge if len(list_clusters) != 0]

    # If we start merging from the cluster with the highest number of neighbours.
    if choice_merging_clusters == 3:
        for k in range(len(variables.size_clusters)):
            trial_list = []
            for u in range(len(variables.size_clusters)):
                if decision_matrix_final[np.argsort(cluster_neighbors)[::-1][k]][u] == 1:
                    if u not in clusters_to_merge_flat:
                        trial_list.append(u)
                        clusters_to_merge_flat.append(u)
            variables.clusters_to_merge.append(trial_list)
        variables.clusters_to_merge = [list_clusters for list_clusters in variables.clusters_to_merge if len(list_clusters) != 0]

    # If we start merging from the cluster with the fewest number of neighbours.
    if choice_merging_clusters == 4:
        for k in range(len(variables.size_clusters)):
            trial_list = []
            for u in range(len(variables.size_clusters)):
                if decision_matrix_final[np.argsort(cluster_neighbors)[k]][u] == 1:
                    if u not in clusters_to_merge_flat:
                        trial_list.append(u)
                        clusters_to_merge_flat.append(u)
            variables.clusters_to_merge.append(trial_list)
        variables.clusters_to_merge = [list_clusters for list_clusters in variables.clusters_to_merge if len(list_clusters) != 0]

    # Print the proposed list of clusters to merge.
    print(f"We propose to merge the clusters according to the following list: {variables.clusters_to_merge}")
    print(f"There will be {len(variables.clusters_to_merge)} clusters after merging.")

###########################################################

def concatenate_clusters(variables, vmax=-1):
    """
    Concatenate clusters based on the provided merging list and update the reordered matrix.

    Parameters:
    variables (Variables): An instance of the Variables class containing clustering data.
    vmax (int, optional): The maximum value for the color scale in the plots. Default is -1.

    Returns:
    None

    Comments:
    * If you are happy with the list that was proposed in the previous cell, you can directly run this cell.
    * If you want to change the list, add the clusters you want to merge in the list "list_clusters_to_merge" below and uncomment the line. The list is empty by default, and if the list is empty we will use the previously proposed list.
    * The clusters that must be merged should be listed as [i,j,k]. For example, to merge clusters 4/5/8 on one hand and 1/7 on the other hand, write "variables.list_clusters_to_merge=[[4, 5, 8], [1, 7]]". Clusters which are not given (0, 2, 3, 6 in this case) will be kept and not merged with anyone.
    * Note that even if you don't merge clusters, it may be interesting to merge a single cluster with itself ("variables.list_clusters_to_merge=[[0]]"), since this will put all the noise at the end.
    * Note that the numbering of clusters starts at 0 here.

    """
    
    variables.merging_has_been_done = True

    # Correct the default value of vmax.
    vmax = vmax if vmax != -1 else np.max(variables.distance_matrix)

    # Check if there is an error.
    if len(variables.clusters_to_merge[0]) == 0:
        print("WARNING: the list of clusters to merge is empty.")
        sys.exit(1)

    # Print in the output file.
    summary_file = open(variables.project_name + "_Yacare_Summary.txt", "a")

    # Print current status.
    print(f"There are currently (before merging) {variables.number_clusters} clusters.")
    summary_file.write(f"There are currently (before merging) {variables.number_clusters} clusters.\n")

    for group in variables.clusters_to_merge:
        print(f"   We will merge clusters {group}.")
        summary_file.write(f"   We will merge clusters {group}.\n")

    # Choose which data to work on.
    if variables.reordering_has_been_done == True:
        reordered_matrix_concatenate_clusters = variables.reordered_matrix_new_ordering
    else:
        reordered_matrix_concatenate_clusters = variables.reordered_matrix

    # Define variables.
    elements_temp = copy.deepcopy(variables.elements_inside_clusters)
    variables.elements_inside_clusters_with_merging = []

    # Group the clusters that were asked to be merged.
    for group in variables.clusters_to_merge:
        elements_with_merging = []
        group.sort()
        # Merge the indices from the reordered matrix, and replace the merged cluster with an empty list to avoid a mess with the order.
        for index_to_merge in group:
            elements_with_merging.append(elements_temp[index_to_merge])
            elements_temp[index_to_merge] = []
        variables.elements_inside_clusters_with_merging.append(np.concatenate(elements_with_merging).ravel().tolist())

    # Remove empty clusters from the temporary list of indices.
    elements_temp = [list_elements for list_elements in elements_temp if len(list_elements) != 0]

    # We now group the merged clusters and the untouched clusters.
    for list_elements in elements_temp:
        variables.elements_inside_clusters_with_merging.append(list_elements)

    # We now add the noise (data which are not in a cluster).
    elements_all_indices_with_merging = copy.deepcopy(variables.elements_inside_clusters_with_merging)
    elements_all_indices_with_merging.append(variables.elements_outside_clusters)

    # Reorder the matrix to merge clusters (first the lines, then the columns).
    elements_all_indices_with_merging_list = [int(a) for a in np.concatenate(elements_all_indices_with_merging).ravel().tolist()]
    variables.reordered_matrix_with_merging = variables.distance_matrix[elements_all_indices_with_merging_list, :]
    variables.reordered_matrix_with_merging = variables.reordered_matrix_with_merging[:, elements_all_indices_with_merging_list]

    # Get new borders of clusters. We have a -1 to avoid looking at the noise.
    variables.borders_with_merging = []
    k = 0
    for i in range(len(variables.elements_inside_clusters_with_merging)):
        clust = variables.elements_inside_clusters_with_merging[i]
        variables.borders_with_merging.append([k, k+len(clust)-1])
        k = k + len(clust)
    variables.number_clusters_with_merging = len(variables.borders_with_merging)

    # Extract matrices for all clusters, based on boundaries.
    clusters_with_merging = []
    for brdr in variables.borders_with_merging:
        cluster_temp = np.array(variables.reordered_matrix_with_merging[brdr[0]:brdr[1], brdr[0]:brdr[1]])
        clusters_with_merging.append(cluster_temp)

    # Initialize lists to store mean row values, size of clusters, the representative structure indices, and the representative structure indices from the raw data.
    mean_distance_on_row_with_merging = []
    size_clusters_with_merging = []
    representative_structures_with_merging = []
    variables.representative_structures_in_original_index_with_merging = []

    # Loop over clusters to find the size for each cluster.
    for i in range(variables.number_clusters_with_merging):
        size_clusters_with_merging.append(len(variables.elements_inside_clusters_with_merging[i]))

    # Loop over clusters to calculate the mean value of each row and the size for each cluster matrix.
    for clust in clusters_with_merging:
        mean_distance_on_row_with_merging.append(np.mean(clust, axis=0))

    # Loop over mean_distance_on_row to determine representative structure indices based on mean row values.
    for i in range(len(mean_distance_on_row_with_merging)):
        representative_structures_with_merging.append(variables.borders_with_merging[i][0] + np.argmin(mean_distance_on_row_with_merging[i]))

    # Loop over representative_structures to map representative structure indices to original indices.
    for i in representative_structures_with_merging:
        variables.representative_structures_in_original_index_with_merging.append(np.concatenate(variables.elements_inside_clusters_with_merging).ravel().tolist()[i]+1)

    print(f'There are now {variables.number_clusters_with_merging} clusters and their representative structures are (indices start at 1): {variables.representative_structures_in_original_index_with_merging}.')
    print(f'The size of the clusters is respectively {size_clusters_with_merging}.')
    print('More noise appears on the lower right corner for the new matrix because previously noise was between clusters, and now clusters are touching each others and all the noise is at the end.')
    print('The out-of-diagonal zones in dashed purple are the ones that will be compared on the next cell.')
    summary_file.write(f'There are now {variables.number_clusters_with_merging} clusters and their representative structures are (indices start at 1): {variables.representative_structures_in_original_index_with_merging}.\n')
    summary_file.write(f'The size of the clusters is respectively {size_clusters_with_merging}.\n')

    # Close the output file.
    summary_file.write("\n")
    summary_file.close()

    #############################

    # Plot the original and reordered matrices.
    plt.figure(figsize=(24, 12))
    plt.subplot(1, 2, 1)
    plt.imshow(reordered_matrix_concatenate_clusters, cmap='terrain', vmax=vmax)
    cbar = plt.colorbar(shrink=0.75)
    cbar.ax.tick_params(labelsize=16)
    for i in range(len(variables.borders)):
        x0 = variables.borders[i][0]
        x1 = variables.borders[i][1]
        plt.axvline(x=x0, ymin=1-x0/variables.num_elements, ymax=1-x1/variables.num_elements, color='red')
        plt.axvline(x=x1, ymin=1-x0/variables.num_elements, ymax=1-x1/variables.num_elements, color='red')
        plt.axhline(y=x0, xmin=x0/variables.num_elements,   xmax=x1/variables.num_elements,   color='red')
        plt.axhline(y=x1, xmin=x0/variables.num_elements,   xmax=x1/variables.num_elements,   color='red')
        for j in range(len(variables.borders)):
            if j != i:
                x2 = variables.borders[j][0]
                x3 = variables.borders[j][1]
                plt.axvline(x=x0, ymin=1-x2/variables.num_elements, ymax=1-x3/variables.num_elements, color='purple', ls='--', lw='0.5')
                plt.axvline(x=x1, ymin=1-x2/variables.num_elements, ymax=1-x3/variables.num_elements, color='purple', ls='--', lw='0.5')
                plt.axhline(y=x2, xmin=x0/variables.num_elements, xmax=x1/variables.num_elements, color='purple', ls='--', lw='0.5')
                plt.axhline(y=x3, xmin=x0/variables.num_elements, xmax=x1/variables.num_elements, color='purple', ls='--', lw='0.5')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title('Reordered distance matrix with clusters', size=20)

    plt.subplot(1, 2, 2)
    plt.imshow(variables.reordered_matrix_with_merging, cmap='terrain', vmax=vmax)
    cbar = plt.colorbar(shrink=0.75)
    cbar.ax.tick_params(labelsize=16)
    plt.xlim(0, variables.num_elements)
    plt.ylim(variables.num_elements, 0)
    for i in range(len(variables.borders_with_merging)):
        x0 = variables.borders_with_merging[i][0]
        x1 = variables.borders_with_merging[i][1]
        plt.axvline(x=x0, ymin=1-x0/variables.num_elements, ymax=1-x1/variables.num_elements, color='red')
        plt.axvline(x=x1, ymin=1-x0/variables.num_elements, ymax=1-x1/variables.num_elements, color='red')
        plt.axhline(y=x0, xmin=x0/variables.num_elements,   xmax=x1/variables.num_elements,   color='red')
        plt.axhline(y=x1, xmin=x0/variables.num_elements,   xmax=x1/variables.num_elements,   color='red')
        for j in range(len(variables.borders_with_merging)):
            if j != i:
                x2 = variables.borders_with_merging[j][0]
                x3 = variables.borders_with_merging[j][1]
                plt.axvline(x=x0, ymin=1-x2/variables.num_elements, ymax=1-x3/variables.num_elements, color='purple', ls='--', lw='0.5')
                plt.axvline(x=x1, ymin=1-x2/variables.num_elements, ymax=1-x3/variables.num_elements, color='purple', ls='--', lw='0.5')
                plt.axhline(y=x2, xmin=x0/variables.num_elements, xmax=x1/variables.num_elements, color='purple', ls='--', lw='0.5')
                plt.axhline(y=x3, xmin=x0/variables.num_elements, xmax=x1/variables.num_elements, color='purple', ls='--', lw='0.5')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title('Reordered distance matrix with merged clusters', size=20)
    if variables.save_images == True:
        plt.savefig(variables.project_name + "_Yacare_10-Matrix-MergedClusters.png", bbox_inches='tight', pad_inches=0.1, dpi=150)
    if variables.show_images == True:
        plt.show()
    plt.close()

###########################################################

def expand_clusters(variables, amount_of_noise, keep_no_noise=False, vmax=-1):
    """
    Expand clusters by rescuing data points from noise based on a specified threshold.

    Parameters:
    variables (Variables): An instance of the Variables class containing clustering data.
    amount_of_noise (float): The threshold value used to determine if data points should be rescued from noise.
    keep_no_noise (bool, optional): Whether to keep all data points as noise. Default is False.
    vmax (int, optional): The maximum value for the color scale in the plots. Default is -1.

    Returns:
    None

    Comments:
    We will try here to expand cluster and "rescue" data.
    * For each data in the noise, we will look to which cluster it is the closest (by comparing the distances between data from the noise and the centroid of clusters).
    * We will then compare the distance between the element from the noise and its closest cluster to a threshold which is "mu+amount_of_noise*sigma" where mu and sigma are the mean distance and standard deviation from closest cluster.
    * "amount_of_noise" is called beta in the article describing YACARE and this variable can be changed by the user. We propose 1.0 by default.
    * To be strict on the quality of the cluster, you can avoid this step or choose amount_of_noise = 0.0, but you will keep a significant amount of noise. To put more data in clusters, choose a higher value for amount_of_noise (typically from 1.0 to 10.0).
    * If you want to assign all the data in clusters because you know you have no noise, add "keep_no_noise = True" in the function (by default this parameter is set to False): the value of amount_of_noise will be ignored and this will provide the same result as setting it to infinity.

    """
    
    # Correct the default value of vmax.
    vmax = vmax if vmax != -1 else np.max(variables.distance_matrix)

    # Choose which data to work on.
    if variables.merging_has_been_done == True:
        variables.number_clusters_extend_data = copy.deepcopy(variables.number_clusters_with_merging)
        elements_inside_clusters_extend_data = copy.deepcopy(variables.elements_inside_clusters_with_merging)
        representative_structures_extend_data = copy.deepcopy(variables.representative_structures_in_original_index_with_merging)
        reordered_matrix_extend_data = variables.reordered_matrix_with_merging
        border_extend_data = copy.deepcopy(variables.borders_with_merging)
    else:
        variables.number_clusters_extend_data = copy.deepcopy(variables.number_clusters)
        elements_inside_clusters_extend_data = copy.deepcopy(variables.elements_inside_clusters)
        representative_structures_extend_data = copy.deepcopy(variables.representative_structures_in_original_index)
        border_extend_data = copy.deepcopy(variables.borders)
        if variables.reordering_has_been_done == True:
            reordered_matrix_extend_data = variables.reordered_matrix_new_ordering
        else:
            reordered_matrix_extend_data = variables.reordered_matrix
    variables.extending_data_has_been_done = True

    # Compute the mean value of the distance in each cluster (in the diagonal) and in each out-of-diagonal rectangle. We need to do it again if merging was done.
    distance_inside_matrix = []
    stddev_inside_matrix = []
    for i in range(variables.number_clusters_extend_data):
        for j in range(variables.number_clusters_extend_data):
            distance_inside_matrix.append(np.mean(reordered_matrix_extend_data[border_extend_data[i][0]:border_extend_data[i][1], border_extend_data[j][0]:border_extend_data[j][1]]))
            stddev_inside_matrix.append(np.std(reordered_matrix_extend_data[border_extend_data[i][0]:border_extend_data[i][1], border_extend_data[j][0]:border_extend_data[j][1]]))
    distance_inside_matrix_final = np.array(distance_inside_matrix).reshape(variables.number_clusters_extend_data, variables.number_clusters_extend_data)
    stddev_inside_matrix_final = np.array(stddev_inside_matrix).reshape(variables.number_clusters_extend_data, variables.number_clusters_extend_data)
    # For the clusters, taking the full mean and full stddev doesn't make sense because it would take into account the diagonal which is made of 0s.
    # The current mean of clusters is {Sum_i Sum_j (d_ij)} / {l*l}, whereas we want {Sum_i Sum_j (d_ij)} / {l*l-l} (where l is the size of the cluster).
    # Thus, we change the values with NewMean=OldMean*(l*l)/(l*l-l)=OldMean*l/(l-1). We do the same for the standard deviation.
    for i in range(variables.number_clusters_extend_data):
        cluster_size = border_extend_data[i][1]-border_extend_data[i][0]+1
        distance_inside_matrix_final[i][i] = distance_inside_matrix_final[i][i]*cluster_size/(cluster_size-1)
        stddev_inside_matrix_final[i][i] = np.sqrt((stddev_inside_matrix_final[i][i]**2)*cluster_size/(cluster_size-1))

    # Print in the output file.
    summary_file = open(variables.project_name + "_Yacare_Summary.txt", "a")

    # Print current status.
    print(f"There are currently (before expanding clusters) {len(np.sort(variables.elements_outside_clusters))} elements in noise, i.e. {round(float((100*len(variables.elements_outside_clusters)/variables.num_elements)), 1)}% of the data.")
    summary_file.write(f"There are currently (before expanding clusters) {len(np.sort(variables.elements_outside_clusters))} elements in noise, i.e. {round(float((100*len(variables.elements_outside_clusters)/variables.num_elements)), 1)}% of the data.\n")
    summary_file.write(f"The chosen amount of noise for expanding clusters is {amount_of_noise}.\n")

    # Make copies of noise.
    variables.elements_outside_clusters_extend_data = copy.deepcopy(variables.elements_outside_clusters)

    # Check if elements that are not in clusters have a distance to the closest representative structure that is lower than mean+amount_of_noise*stddev from this cluster.
    elements_to_remove_from_noise = []
    for i in variables.elements_outside_clusters_extend_data:
        # For each point in noise, get the distance to the representative structure of all clusters.
        distance_noise_point_to_centroid = []
        for j in range(variables.number_clusters_extend_data):
            distance_noise_point_to_centroid.append(variables.distance_matrix[representative_structures_extend_data[j]-1][i])
        # Look if the lowest distance (to the closest cluster) is smaller than mean+amount_of_noise*stddev, and if yes store the value of the noise point.
        closest_cluster = np.argmin(distance_noise_point_to_centroid)
        lowest_distance = np.min(distance_noise_point_to_centroid)
        cluster_mean = distance_inside_matrix_final[closest_cluster][closest_cluster]
        cluster_stddev = stddev_inside_matrix_final[closest_cluster][closest_cluster]
        threshold = cluster_mean + amount_of_noise * cluster_stddev
        if keep_no_noise == True:
            elements_inside_clusters_extend_data[closest_cluster].append(i)
            elements_to_remove_from_noise.append(i)
        elif lowest_distance < threshold:
            elements_inside_clusters_extend_data[closest_cluster].append(i)
            elements_to_remove_from_noise.append(i)
    # Create a new clean list, and update the variable.
    temp_list = [i for i in variables.elements_outside_clusters_extend_data if i not in elements_to_remove_from_noise]
    variables.elements_outside_clusters_extend_data = temp_list

    # New name of list.
    variables.elements_inside_clusters_with_noise = elements_inside_clusters_extend_data

    # Add the noise i.e. data which are not in a cluster.
    elements_all_indices_with_noise = copy.deepcopy(variables.elements_inside_clusters_with_noise)
    elements_all_indices_with_noise.append(variables.elements_outside_clusters_extend_data)

    # Reorder the matrix to make clusters with some data extracted from noise (first the lines, then the columns).
    # Was before: elements_all_indices_with_noise_list = [int(a) for a in np.concatenate(elements_all_indices_with_noise).ravel().tolist()]
    elements_all_indices_with_noise_list = np.concatenate(elements_all_indices_with_noise).ravel().astype(int).tolist()
    variables.reordered_matrix_with_noise = variables.distance_matrix[elements_all_indices_with_noise_list, :]
    variables.reordered_matrix_with_noise = variables.reordered_matrix_with_noise[:, elements_all_indices_with_noise_list]

    # Get new borders of clusters. We have a -1 to avoid looking at the noise.
    variables.borders_with_noise = []
    k = 0
    for i in range(len(variables.elements_inside_clusters_with_noise)):
        clust = variables.elements_inside_clusters_with_noise[i]
        variables.borders_with_noise.append([k, k+len(clust)-1])
        k = k + len(clust)

    # Extract matrices for all clusters, based on boundaries.
    clusters_with_noise = []
    for brdr in variables.borders_with_noise:
        cluster_temp = np.array(variables.reordered_matrix_with_noise[brdr[0]:brdr[1], brdr[0]:brdr[1]])
        clusters_with_noise.append(cluster_temp)

    # Initialize lists to store mean row values, size of clusters, the representative structure indices, and the representative structure indices from the raw data.
    mean_distance_on_row_with_noise = []
    size_clusters_with_noise = []
    representative_structures_with_noise = []
    variables.representative_structures_in_original_index_with_noise = []

    # Loop over clusters to find the size for each cluster.
    for i in range(variables.number_clusters_extend_data):
        size_clusters_with_noise.append(len(variables.elements_inside_clusters_with_noise[i]))

    # Loop over clusters to calculate the mean value of each row and the size for each cluster matrix.
    for clust in clusters_with_noise:
        mean_distance_on_row_with_noise.append(np.mean(clust, axis=0))

    # Loop over mean_distance_on_row to determine representative structure indices based on mean row values.
    for i in range(len(mean_distance_on_row_with_noise)):
        representative_structures_with_noise.append(variables.borders_with_noise[i][0] + np.argmin(mean_distance_on_row_with_noise[i]))

    # Loop over representative_structures to map representative structure indices to original indices.
    for i in representative_structures_with_noise:
        variables.representative_structures_in_original_index_with_noise.append(np.concatenate(variables.elements_inside_clusters_with_noise).ravel().tolist()[i]+1)

    # Compute the final delta_diagonal by iterating over the reordered matrix.
    delta_diagonal_final = get_delta_diagonal(variables.reordered_matrix_with_noise, variables.size_moving_square)

    #############################

    # Plot the original and reordered matrices.
    plt.figure(figsize=(24, 12))
    plt.subplot(1, 2, 1)
    plt.imshow(reordered_matrix_extend_data, cmap='terrain', vmax=vmax)
    cbar = plt.colorbar(shrink=0.75)
    cbar.ax.tick_params(labelsize=16)
    for i in range(len(border_extend_data)):
        x0 = border_extend_data[i][0]
        x1 = border_extend_data[i][1]
        plt.axvline(x=x0, ymin=1-x0/variables.num_elements, ymax=1-x1/variables.num_elements, color='red')
        plt.axvline(x=x1, ymin=1-x0/variables.num_elements, ymax=1-x1/variables.num_elements, color='red')
        plt.axhline(y=x0, xmin=x0/variables.num_elements, xmax=x1/variables.num_elements, color='red')
        plt.axhline(y=x1, xmin=x0/variables.num_elements, xmax=x1/variables.num_elements, color='red')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title('Previous distance matrix', size=20)

    plt.subplot(1, 2, 2)
    plt.imshow(variables.reordered_matrix_with_noise, cmap='terrain', vmax=vmax)
    cbar = plt.colorbar(shrink=0.75)
    cbar.ax.tick_params(labelsize=16)
    for i in range(len(variables.borders_with_noise)):
        x0 = variables.borders_with_noise[i][0]
        x1 = variables.borders_with_noise[i][1]
        plt.axvline(x=x0, ymin=1-x0/variables.num_elements, ymax=1-x1/variables.num_elements, color='red')
        plt.axvline(x=x1, ymin=1-x0/variables.num_elements, ymax=1-x1/variables.num_elements, color='red')
        plt.axhline(y=x0, xmin=x0/variables.num_elements, xmax=x1/variables.num_elements, color='red')
        plt.axhline(y=x1, xmin=x0/variables.num_elements, xmax=x1/variables.num_elements, color='red')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title('Reordered distance matrix with data from noise', size=20)
    if variables.save_images == True:
        plt.savefig(variables.project_name + "_Yacare_11-Matrix-WithNoise.png", bbox_inches='tight', pad_inches=0.1, dpi=150)
    if variables.show_images == True:
        plt.show()
    plt.close()

    # Plot delta_D with cutoff.
    color = itertools.cycle(('tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'))
    plt.figure(figsize=(24, 6))
    plt.subplot(1,1,1)
    plt.plot(range(variables.size_moving_square, variables.num_elements - variables.size_moving_square), delta_diagonal_final)
    plt.axhline(variables.selected_cutoff, color='gray', label='Cut-off', linewidth=0.5)
    for i in range(0, len(variables.borders_with_noise)):
        plt.axhline(xmin=(variables.borders_with_noise[i][0])/variables.num_elements, xmax=(variables.borders_with_noise[i][1])/variables.num_elements, y=variables.selected_cutoff, color=next(color), linewidth=3)
    plt.xlabel('Index', size=18)
    plt.ylabel(r'$\Delta_d$', size=18)
    plt.legend(loc='upper left', fontsize=14)
    plt.xlim(0, variables.num_elements)
    plt.ylim(0,)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    if variables.save_images == True:
        plt.savefig(variables.project_name + "_Yacare_12-DeltaD-WithNoise.png", bbox_inches='tight', pad_inches=0.1, dpi=150)
    if variables.show_images == True:
        plt.show()
    plt.close()

    # Print new status.
    print(f"After expansion, there are now {len(np.sort(variables.elements_outside_clusters_extend_data))} elements in noise, i.e. {round(float((100*len(variables.elements_outside_clusters_extend_data)/variables.num_elements)), 1)}% of the data.")
    print(f'The new representative structures are (indices start at 1): {variables.representative_structures_in_original_index_with_noise}.')
    print(f'The size of the clusters is respectively {size_clusters_with_noise}.')
    summary_file.write(f"After expansion, there are now {len(np.sort(variables.elements_outside_clusters_extend_data))} elements in noise, i.e. {round(float((100*len(variables.elements_outside_clusters_extend_data)/variables.num_elements)), 1)}% of the data.\n")
    summary_file.write(f'The new representative structures are (indices start at 1): {variables.representative_structures_in_original_index_with_noise}.\n')
    summary_file.write(f'The size of the clusters is respectively {size_clusters_with_noise}.\n')

    # Close the output file.
    summary_file.write("\n")
    summary_file.close()

###########################################################

def compare_final_clusters(variables, display_stddev = False, display_mean_distances = False):
    """
    Compare final clusters by plotting the mean distance and standard deviation within and between clusters.

    Parameters:
    variables (Variables): An instance of the Variables class containing clustering data.
    display_stddev (bool): Whether to display the standard deviation in the plot. Default is False.
    display_mean_distances (bool): Whether to display the mean distances in the plot. Default is False.

    Returns:
    None
    """
    
    # Choose the size of each square for the image. The value must be adapted for each case because it will depend on the system.
    # ~50 lines below we have tried to propose an automated way to compute size_scaling. If the image is ugly, you must comment the
    # size_scaling line that is ~60 lines below and manually pick one here. Note that the displayed size of an zone in the matrix is
    # the square root of its zone (all zones will be displayed as squares).
    #size_scaling = 0.0025

    # Choose which data to work on.
    if variables.extending_data_has_been_done == True:
        number_clusters_compare_final = copy.deepcopy(variables.number_clusters_extend_data)
        elements_inside_clusters_compare_final = copy.deepcopy(variables.elements_inside_clusters_with_noise)
        representative_structures_compare_final = copy.deepcopy(variables.representative_structures_in_original_index_with_noise)
        reordered_matrix_compare_final = variables.reordered_matrix_with_noise
        border_compare_final = copy.deepcopy(variables.borders_with_noise) 
    elif variables.merging_has_been_done == True:
        number_clusters_compare_final = copy.deepcopy(variables.number_clusters_with_merging)
        elements_inside_clusters_compare_final = copy.deepcopy(variables.elements_inside_clusters_with_merging)
        representative_structures_compare_final = copy.deepcopy(variables.representative_structures_in_original_index_with_merging)
        reordered_matrix_compare_final = variables.reordered_matrix_with_merging
        border_compare_final = copy.deepcopy(variables.borders_with_merging)
    else:
        number_clusters_compare_final = copy.deepcopy(variables.number_clusters)
        elements_inside_clusters_compare_final = copy.deepcopy(variables.elements_inside_clusters)
        representative_structures_compare_final = copy.deepcopy(variables.representative_structures_in_original_index)
        border_compare_final = copy.deepcopy(variables.borders)
        if variables.reordering_has_been_done == True:
            reordered_matrix_compare_final = variables.reordered_matrix_new_ordering
        else:
            reordered_matrix_compare_final = variables.reordered_matrix

    if number_clusters_compare_final <  2 :
        return

    # Compute the mean value of the distance in each cluster (in the diagonal) and in each out-of-diagonal rectangle.
    distance_inside_matrix = []
    stddev_inside_matrix = []
    for i in range(number_clusters_compare_final):
        for j in range(number_clusters_compare_final):
            distance_inside_matrix.append(np.mean(reordered_matrix_compare_final[border_compare_final[i][0]:border_compare_final[i][1], border_compare_final[j][0]:border_compare_final[j][1]]))
            stddev_inside_matrix.append(np.std(reordered_matrix_compare_final[border_compare_final[i][0]:border_compare_final[i][1], border_compare_final[j][0]:border_compare_final[j][1]]))
    distance_inside_matrix_final = np.array(distance_inside_matrix).reshape(number_clusters_compare_final, number_clusters_compare_final)
    stddev_inside_matrix_final = np.array(stddev_inside_matrix).reshape(number_clusters_compare_final, number_clusters_compare_final)
    # For the clusters, taking the full mean and full stddev doesn't make sense because it would take into account the diagonal which is made of 0s.
    # The current mean of clusters is {Sum_i Sum_j (d_ij)} / {l*l}, whereas we want {Sum_i Sum_j (d_ij)} / {l*l-l} (where l is the size of the cluster).
    # Thus, we change the values with NewMean=OldMean*(l*l)/(l*l-l)=OldMean*l/(l-1). We do the same for the standard deviation.
    for i in range(number_clusters_compare_final):
        cluster_size = border_compare_final[i][1]-border_compare_final[i][0]+1
        distance_inside_matrix_final[i][i] = distance_inside_matrix_final[i][i]*cluster_size/(cluster_size-1)
        stddev_inside_matrix_final[i][i] = np.sqrt((stddev_inside_matrix_final[i][i]**2)*cluster_size/(cluster_size-1))

    # Get the size of zones from the distance matrix.
    zone_sizes = []
    zone_all_sizes = []
    # Compute the length of each zone.
    for i in range(number_clusters_compare_final):
        zone_sizes.append(border_compare_final[i][1]-border_compare_final[i][0])
    # Compute the size of each zone (cluster or off-diagonal part).
    for i in range(number_clusters_compare_final):
        for j in range(number_clusters_compare_final):
            zone_all_sizes.append(zone_sizes[i]*zone_sizes[j])
    # Reshape in a 2D array.
    zone_all_sizes_final = np.array(zone_all_sizes).reshape(number_clusters_compare_final, number_clusters_compare_final)

    # Try to define automatically the scaling_factor.
    # The idea is that the size occupied by the largest cluster is size_scaling*max(zone_all_sizes_final.flatten()).
    # This size is made of M^2 points. Since the figure will be 12*12 inches ("figsize=(12, 12)"), each cluster will have
    # at most 12/(num_clusters+1) inches (the +1 is to give some space around) for itself. Each point is markersize*1/72 inches,
    # and my understanding is that by defaut markersize=1.66. Thus, 12/(num_clusters+1) = M*1.66/72. So we can have access to M.
    # We add a 0.9 term to scale down a little bit the squares to have some space.
    size_scaling = 0.9*((12/(number_clusters_compare_final+1))/(1.66/72))**2/max(zone_all_sizes_final.flatten())

    # Shape of the matrix for the size of zones.
    rows, cols = zone_all_sizes_final.shape

    # Coordinates for each element, and flatten the matrices for scatter.
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    x = x.flatten()
    y = y.flatten()

    # Plot.
    plt.figure(figsize=(12, 12))
    plt.scatter(x, y, s=size_scaling*zone_all_sizes_final.flatten(), c=distance_inside_matrix_final.flatten(), alpha=0.6, marker='s')
    plt.xlim(-1, rows)
    plt.ylim(-1, cols)
    plt.xticks(range(rows), size=16)
    plt.yticks(range(cols), size=16)

    # Invert y axis to match the orientation.
    plt.gca().invert_yaxis()

    # Add text in the matrix.
    for i in range(number_clusters_compare_final):
        for j in range(number_clusters_compare_final):
            # Display the distances between clusters and/or the standard deviation.
            if display_stddev == True and display_mean_distances == True:
                    plt.text(i, j, "{:.3f}".format(distance_inside_matrix_final[i, j]) + "\n" + "{:.3f}".format(stddev_inside_matrix_final[i, j]), ha='center', va='center', color='black', size=10)
            elif display_stddev == True and display_mean_distances == False:
                    plt.text(i, j, "{:.3f}".format(stddev_inside_matrix_final[i, j]), ha='center', va='center', color='black', size=10)
            elif display_stddev == False and display_mean_distances == True:
                    plt.text(i, j, "{:.3f}".format(distance_inside_matrix_final[i, j]), ha='center', va='center', color='black', size=10)        
        
    # Add title and colorbar. The title depends on the choosen options.
    if display_stddev == True and display_mean_distances == True:
        plt.title("size of a square = size of the corresponding zone in the matrix \n color of a square = mean value of the distances in the zone \n numbers in a square = mean value and standard deviation of the distances in the zone", size=10)
    elif display_stddev == True and display_mean_distances == False:
        plt.title("size of a square = size of the corresponding zone in the matrix \n color of a square = mean value of the distances in the zone \n number in a square = standard deviation of the distances in the zone", size=10)
    elif display_stddev == False and display_mean_distances == True:
        plt.title("size of a square = size of the corresponding zone in the matrix \n color of a square = mean value of the distances in the zone \n number in a square = mean value of the distances in the zone", size=10)
    else:
        plt.title("size of a square = size of the corresponding zone in the matrix \n color of a square = mean value of the distances in the zone", size=10)
    colorbar = plt.colorbar()
    colorbar.set_label('Mean distance in the cluster', fontsize=16)
    colorbar.ax.tick_params(labelsize=16)
    if variables.save_images == True:
        plt.savefig(variables.project_name + "_Yacare_13-CompareLastClusters.png", bbox_inches='tight', pad_inches=0.1, dpi=150)
    if variables.show_images == True:
        plt.show()
    plt.close()
    
    average_mu_clusters = 0
    average_mu_out_of_diagonal = 0
    for i in range(number_clusters_compare_final):
        average_mu_clusters += distance_inside_matrix_final[i][i]
        for j in range(i+1, number_clusters_compare_final):
            average_mu_out_of_diagonal += distance_inside_matrix_final[i][j]
    average_mu_clusters = average_mu_clusters / number_clusters_compare_final
    average_mu_out_of_diagonal = average_mu_out_of_diagonal / ((number_clusters_compare_final*number_clusters_compare_final-number_clusters_compare_final)/2)
    print(f"The mean of the average of distances inside clusters (i.e. from the diagonal) is {round(average_mu_clusters, 3)}, and the mean of the average of distances out of diagonal is {round(average_mu_out_of_diagonal, 3)}.")

###########################################################

def write_indices(variables):
    """
    Write the indices of clusters, representative structures, labels, and noise to files.

    This function writes the indices of elements inside clusters, the representative structures, the label for each element, and the noise elements into separate files.
    It also updates the variables object with the necessary data for writing.

    Parameters:
    variables (Variables): An instance of the Variables class containing clustering data.

    Returns:
    None
    """
        
    variables.writing_indices_has_been_done = True
 
    # Choose which data to work on.
    if variables.extending_data_has_been_done == True:
        variables.number_clusters_write_indices = copy.deepcopy(variables.number_clusters_extend_data)
        variables.elements_inside_clusters_write_indices = copy.deepcopy(variables.elements_inside_clusters_with_noise)
        representative_structures_write_indices = copy.deepcopy(variables.representative_structures_in_original_index_with_noise)
        reordered_matrix_write_indices = variables.reordered_matrix_with_noise
        border_write_indices = copy.deepcopy(variables.borders_with_noise)
        variables.elements_outside_clusters_write_indices = copy.deepcopy(variables.elements_outside_clusters_extend_data)
    elif variables.merging_has_been_done == True:
        variables.number_clusters_write_indices = copy.deepcopy(variables.number_clusters_with_merging)
        variables.elements_inside_clusters_write_indices = copy.deepcopy(variables.elements_inside_clusters_with_merging)
        representative_structures_write_indices = copy.deepcopy(variables.representative_structures_in_original_index_with_merging)
        reordered_matrix_write_indices = variables.reordered_matrix_with_merging
        border_write_indices = copy.deepcopy(variables.borders_with_merging)
        variables.elements_outside_clusters_write_indices = copy.deepcopy(variables.elements_outside_clusters)
    else:
        variables.number_clusters_write_indices = copy.deepcopy(variables.number_clusters)
        variables.elements_inside_clusters_write_indices = copy.deepcopy(variables.elements_inside_clusters)
        representative_structures_write_indices = copy.deepcopy(variables.representative_structures_in_original_index)
        border_write_indices = copy.deepcopy(variables.borders)
        variables.elements_outside_clusters_write_indices = copy.deepcopy(variables.elements_outside_clusters)
        if variables.reordering_has_been_done == True:
            reordered_matrix_write_indices = variables.reordered_matrix_new_ordering
        else:
            reordered_matrix_write_indices = variables.reordered_matrix

    # Write the representative structure of clusters.
    # Note: The "+1" shift to convert from 0-based to 1-based indexing was added earlier when representative_structures_write_indices was created.
    structure_file = open(variables.project_name + "_Clustering_RepresentativeStructures.ndx", "w")
    for i in range(variables.number_clusters_write_indices):
        structure_file.write(f"[ Cluster{i+1}_Centroid ]\n")
        structure_file.write(f"{representative_structures_write_indices[i]}\n")
    structure_file.close()

    # Write the clusters (the +1 is here to start indices at 1 and not at 0).
    index_file = open(variables.project_name + "_Clustering_Clusters.ndx", "w")
    # Loop through indices for clusters, write 10 per line, and the remaining ones on another line.
    # If there are 82 data in a cluster, len(elements_inside_clusters[i])//10 with give us 8, so the first loop is from 0 to 80 (excluding 80)
    for i in range(variables.number_clusters_write_indices):
        index_file.write(f"[ Cluster{i+1} ]\n")
        for j in range(0, 10*(len(variables.elements_inside_clusters_write_indices[i])//10), 10):
            for k in range(10):
                index_file.write(f"{variables.elements_inside_clusters_write_indices[i][j+k]+1} ")
            index_file.write("\n")
        for j in range(10*(len(variables.elements_inside_clusters_write_indices[i])//10), len(variables.elements_inside_clusters_write_indices[i])):
            index_file.write(f"{variables.elements_inside_clusters_write_indices[i][j] + 1} ")
        # Create a new line if the size of the cluster is not a multiple of 10.
        if 10*(len(variables.elements_inside_clusters_write_indices[i])//10) != len(variables.elements_inside_clusters_write_indices[i]):
            index_file.write("\n")

    # Write indices for the noise.
    index_file.write("[ Noise ]\n")
    for j in range(0, 10*(len(variables.elements_outside_clusters_write_indices)//10), 10):
        for k in range(10):
            index_file.write(f"{variables.elements_outside_clusters_write_indices[j+k] + 1} ")
        index_file.write("\n")
    for j in range(10*(len(variables.elements_outside_clusters_write_indices)//10), len(variables.elements_outside_clusters_write_indices)):
        index_file.write(f"{variables.elements_outside_clusters_write_indices[j] + 1} ")
    index_file.write("\n")
    index_file.close()

    # Write the list of reordered elements in a separate file.
    reordered_file = open(variables.project_name + "_Clustering_ReorderedElements.txt", "w")
    for i in range(variables.number_clusters_write_indices):   
        for j in range(0, len(variables.elements_inside_clusters_write_indices[i])):
            reordered_file.write(str(variables.elements_inside_clusters_write_indices[i][j]+1))
            reordered_file.write("\n")
    reordered_file.close()

    noise_file = open(variables.project_name + "_Clustering_Noise.txt", "w")
    for j in range(0, len(variables.elements_outside_clusters_write_indices)):
        noise_file.write(str(variables.elements_outside_clusters_write_indices[j]+1))
        noise_file.write("\n")
    noise_file.close()
    
    # Write a file with the index of cluster for each element.
    labels_clusters_file  = open(variables.project_name + "_Clustering_Labels.txt", "w")
    list_clustered_data = []
    for i in range(variables.number_clusters_write_indices):
        for j in range(len(variables.elements_inside_clusters_write_indices[i])):
            list_clustered_data.append([variables.elements_inside_clusters_write_indices[i][j]+1, i+1])
    for j in range(len(variables.elements_outside_clusters_write_indices)):
        list_clustered_data.append([variables.elements_outside_clusters_write_indices[j]+1, variables.number_clusters_write_indices+1])
    # The list is sorted using the index of the data, i.e. from 1 to N.
    list_clustered_data_sorted = sorted(list_clustered_data, key=lambda x: x[0])
    # We extract the index of the clusters for each data.
    labels_clusters = [x[1] for x in list_clustered_data_sorted]
    for j in range(0, len(labels_clusters)):
        labels_clusters_file.write(str(labels_clusters[j]))
        labels_clusters_file.write("\n")
    labels_clusters_file.close()

    print("Files successfully written.")

###########################################################

def plot_confusion_matrix(variables, labels_true, transformation = {}, auto_reorder_columns=False):
    """
    Plot the confusion matrix to compare clustering results from YACARE with true labels.

    This function computes and plots the confusion matrix to compare the clustering results obtained from the YACARE algorithm with the true labels.
    It also calculates various clustering metrics such as Adjusted Rand Index, Adjusted Mutual Information, Normalized Mutual Information, Homogeneity, Completeness, V-measure, Fowlkes-Mallows, Davies-Bouldin and Silhouette scores.

    Parameters:
    variables (Variables): An instance of the Variables class containing clustering data.
    labels_true (list): A list of true labels for the data points.
    transformation (dict, optional): A dictionary to transform cluster labels. Default is an empty dictionary.
    auto_reorder_columns (bool, optional): Whether to automatically reorder the confusion matrix for better visualization. Default is False.

    Returns:
    None
    """
    
    import seaborn as sns
    
    # Proceed only if we have written indices.
    if variables.writing_indices_has_been_done == 0:
        raise RuntimeError("You must have activated the cell where indices are written because we use variables that will be defined there.")

    # True values should start at 1.
    if 0 in labels_true:
        raise RuntimeError("Your true values should start at 1. You can for example do: transformation_truth = { 0:1, 1:2, 2:3} then labels_true = [transformation_truth.get(x, x) for x in labels_true].")
    
    # Extract the data from our clustering. We write a list of list, which contains "index of the data, index of the cluster".
    list_clustered_data = []
    for i in range(variables.number_clusters_write_indices):
        for j in range(len(variables.elements_inside_clusters_write_indices[i])):
            list_clustered_data.append([variables.elements_inside_clusters_write_indices[i][j]+1, i+1])
    for j in range(len(variables.elements_outside_clusters_write_indices)):
        list_clustered_data.append([variables.elements_outside_clusters_write_indices[j]+1, variables.number_clusters_write_indices+1])
    # The list is sorted using the index of the data, i.e. from 1 to N.
    list_clustered_data_sorted = sorted(list_clustered_data, key=lambda x: x[0])
    # We extract the index of the clusters for each data.
    labels_clusters = [x[1] for x in list_clustered_data_sorted]

    # Define a variable to know if there is noise.
    if len(variables.elements_outside_clusters_write_indices) == 0:
        there_is_noise = False
    else:
        there_is_noise = True

    # Get some parameters (in number_predicted_clusters, we include the noise).
    number_true_clusters = len(np.unique(labels_true))
    if there_is_noise == True:
        number_predicted_clusters = variables.number_clusters_write_indices + 1
    else:
        number_predicted_clusters = variables.number_clusters_write_indices
    
    # Apply the transformation to all elements in the list. If transformation is empty, nothing is done.
    labels_clusters_transformed = [transformation.get(x, x) for x in labels_clusters]
    
    # Compute the confusion matrix.
    cm = confusion_matrix(labels_true, labels_clusters_transformed)
    
    # Save the labels.
    variables.labels_yacare = labels_clusters_transformed
    
    #Get the score of clustering.
    print(f"Adjusted Rand Index score = {adjusted_rand_score(labels_true, labels_clusters_transformed):.4f}")
    print(f"Adjusted Mutual Information score = {adjusted_mutual_info_score(labels_true, labels_clusters_transformed):.4f}")
    print(f"Normalized Mutual Information score = {normalized_mutual_info_score(labels_true, labels_clusters_transformed):.4f}")
    print(f"Homogeneity, completeness and v-measure scores = {homogeneity_completeness_v_measure(labels_true, labels_clusters_transformed)[0]:.4f}, {homogeneity_completeness_v_measure(labels_true, labels_clusters_transformed)[1]:.4f}, {homogeneity_completeness_v_measure(labels_true, labels_clusters_transformed)[2]:.4f}")
    print(f"Fowlkes-Mallows score = {fowlkes_mallows_score(labels_true, labels_clusters_transformed):.4f}")
    if variables.raw_data_is_distance_matrix == False:
        print(f"Davies-Bouldin score = {davies_bouldin_score(variables.raw_data, labels_clusters_transformed):.4f}")
        print(f"Silhouette score = {silhouette_score(variables.raw_data, labels_clusters_transformed):.4f}")

    # Restrict the number of lines and columns that are displayed.
    cm_filtered = cm[:number_true_clusters, :]
    cm_filtered = cm_filtered[:, :number_predicted_clusters]

    # If we have choosen to automatically reorder.
    if auto_reorder_columns == True:
        #Prepare the working array. First case is if there is no noise, second case is if there is noise.
        if there_is_noise == False:
            cm_prepared = cm_filtered
        else:
            cm_prepared = [row[:-1] for row in cm_filtered]
            last_column = [row[-1] for row in cm_filtered]
        
        # Convert in numpy array.
        cm_prepared = np.array(cm_prepared)
        num_rows, num_columns = cm_prepared.shape
        # Find the indices of lines with the maximal values for each column.
        max_indices = [np.argmax(cm_prepared[:, j]) for j in range(num_columns)]
        # Match each column with its max index and its value.
        column_order = sorted( range(num_columns), key=lambda col: (max_indices[col], -cm_prepared[max_indices[col], col]) )
        # Reorder the matrix.
        cm_new = cm_prepared[:, column_order].tolist()
        
        # Add back the last column which contains noise if there was noise.
        if there_is_noise == True:
            for i, row in enumerate(cm_new):
                row.append(last_column[i])

        # Display what was done.
        transformation_dict = {j+1: i+1 for i, j in enumerate(column_order)}
        transformation_dict[len(column_order)+1] = len(column_order)+1
        print(f"We will apply this: transformation = {transformation_dict}")      

        # Reset the variable.
        cm_filtered = cm_new
    
    # Convert to np array.
    cm_filtered = np.array(cm_filtered)
    
    # Prepare lists for the labels.
    xticks = [x + 0.5 for x in list(range(0, number_predicted_clusters))]
    yticks = [x + 0.5 for x in list(range(0, number_true_clusters))]
    ylabels = [str(i) for i in range(1, number_true_clusters+1)]
    if there_is_noise == False:    
        xlabels = [str(i) for i in range(1, number_predicted_clusters+1)]
    else:
        xlabels = [str(i) for i in range(1, number_predicted_clusters)]
        xlabels.append("Noise")
    
    plt.figure(figsize=(12, 10))
    plt.subplot(1, 1, 1)
    ax = sns.heatmap(cm_filtered, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 12}, cbar=False)
    plt.xlabel('Prediction', fontsize=16)
    plt.ylabel('Truth', fontsize=16)
    plt.xticks(xticks, xlabels, fontsize=12)
    plt.yticks(yticks, ylabels, fontsize=12)
    #plt.title('Confusion matrix', size=20)
    if variables.save_images == True:
        plt.savefig(variables.project_name + "_Yacare_14-ConfusionMatrix.png", bbox_inches='tight', pad_inches=0.1, dpi=150)
    if variables.show_images == True:
        plt.show()
    plt.close()   

    # Count the number of pure and uncut clusters. First condition is for the case with no noise, second condition is for the other case.
    if there_is_noise == False:
        print("'Pure clusters' are clusters found by YACARE with data coming from a single true cluster: in the column there is a single value which is different than 0.")
        print("'Uncut clusters' are original clusters for which all data were found in a single YACARE cluster: in the line there is a single value (other than noise) which is different than 0.")
        print(" ")
        # Count the number of pure clusters.
        pure_clusters = np.sum(cm_filtered != 0, axis=0)      
        print(f"In the {cm_filtered.shape[1]} found clusters, there are {np.sum(pure_clusters == 1)} pure clusters.")
        # Count the number of uncut clusters.
        non_zero_per_row = np.sum(cm_filtered != 0, axis=1)
        print(f"There are {np.sum(non_zero_per_row == 1)} uncut clusters.")
    else:
        print("'Pure clusters' are clusters found by YACARE with data coming from a single true cluster: in the column there is a single value which is different than 0.")
        print("'Uncut clusters' are original clusters for which all data were found in a single YACARE cluster: in the line there is a single value (other than noise) which is different than 0.")
        # Count the number of pure clusters.
        cm_filtered_no_noise = cm_filtered[:, :-1]
        pure_clusters = np.sum(cm_filtered_no_noise != 0, axis=0)
        print(f"In the {cm_filtered_no_noise.shape[1]} found clusters, there are {np.sum(pure_clusters == 1)} pure clusters.")
        # Count the number of uncut clusters.
        non_zero_per_row = np.sum(cm_filtered_no_noise != 0, axis=1)
        print(f"There are {np.sum(non_zero_per_row == 1)} uncut clusters.")

###########################################################

def plot_confusion_matrix_HDBSCAN(variables, labels_true, transformation = {}, auto_reorder_columns=False):
    """
    Plot the confusion matrix to compare clustering results from HDBSCAN with true labels.

    This function computes and plots the confusion matrix to compare the clustering results obtained from the HDBSCAN algorithm with the true labels.
    It also calculates various clustering metrics such as Adjusted Rand Index, Adjusted Mutual Information, Normalized Mutual Information, Homogeneity, Completeness, V-measure, Fowlkes-Mallows, Davies-Bouldin and Silhouette scores.

    Parameters:
    variables (Variables): An instance of the Variables class containing clustering data.
    labels_true (list): A list of true labels for the data points.
    transformation (dict, optional): A dictionary to transform cluster labels. Default is an empty dictionary.
    auto_reorder_columns (bool, optional): Whether to automatically reorder the confusion matrix for better visualization. Default is False.

    Returns:
    None
    """
    
    import seaborn as sns
    
    # Get the same parameter as YACARE for the minimal number of elements in a cluster.
    min_cluster_size = int(variables.minimal_size_cluster*variables.num_elements/100)
    if variables.raw_data_is_distance_matrix == False:
        hdbscan = cluster.HDBSCAN(min_cluster_size=min_cluster_size)
        hdbscan.fit(variables.raw_data)
    else:
        hdbscan = cluster.HDBSCAN(metric='precomputed', min_cluster_size=min_cluster_size)
        hdbscan.fit(variables.distance_matrix)
    
    # Transform the noise from being labelled -1 and become last cluster, and start indices of clusters at 1 and not at 0.
    labels_hdbscan_new = [max(hdbscan.labels_)+2 if x == -1 else x+1 for x in hdbscan.labels_]

    # Define a variable to know if there is noise.
    there_is_noise = True if -1 in hdbscan.labels_ else False
    if there_is_noise == True:
        print("Noise was found by HDBSCAN.")

    # Apply the transformation to all elements in the list. If transformation is empty, nothing is done.
    labels_hdbscan_transformed = [transformation.get(x, x) for x in labels_hdbscan_new]

    # Get some parameters.
    number_true_clusters = len(np.unique(labels_true))
    number_predicted_clusters = len(np.unique(labels_hdbscan_transformed))

    # Compute the confusion matrix.
    cm = confusion_matrix(labels_true, labels_hdbscan_transformed)

    # Save the labels.
    variables.labels_hdbscan = labels_hdbscan_transformed
        
    #Get the score of clustering.
    print(f"Adjusted Rand Index score = {adjusted_rand_score(labels_true, labels_hdbscan_transformed):.4f}")
    print(f"Adjusted Mutual Information score = {adjusted_mutual_info_score(labels_true, labels_hdbscan_transformed):.4f}")
    print(f"Normalized Mutual Information score = {normalized_mutual_info_score(labels_true, labels_hdbscan_transformed):.4f}")
    print(f"Homogeneity, completeness and v-measure scores = {homogeneity_completeness_v_measure(labels_true, labels_hdbscan_transformed)[0]:.4f}, {homogeneity_completeness_v_measure(labels_true, labels_hdbscan_transformed)[1]:.4f}, {homogeneity_completeness_v_measure(labels_true, labels_hdbscan_transformed)[2]:.4f}")
    print(f"Fowlkes-Mallows score = {fowlkes_mallows_score(labels_true, labels_hdbscan_transformed):.4f}")
    if variables.raw_data_is_distance_matrix == False:
        print(f"Davies-Bouldin score = {davies_bouldin_score(variables.raw_data, labels_hdbscan_transformed):.4f}")
        print(f"Silhouette score = {silhouette_score(variables.raw_data, labels_hdbscan_transformed):.4f}")
        
    # Restrict the number of lines and columns that are displayed.
    cm_filtered = cm[:number_true_clusters, :]
    cm_filtered = cm_filtered[:, :number_predicted_clusters]

    # If we have choosen to automatically reorder.
    if auto_reorder_columns == True:
        #Prepare the working array. First case is if there is no noise, second case is if there is noise.
        if there_is_noise == False:
            cm_prepared = cm_filtered
        else:
            cm_prepared = [row[:-1] for row in cm_filtered]
            last_column = [row[-1] for row in cm_filtered]
        
        # Convert in numpy array.
        cm_prepared = np.array(cm_prepared)
        num_rows, num_columns = cm_prepared.shape
        # Find the indices of lines with the maximal values for each column.
        max_indices = [np.argmax(cm_prepared[:, j]) for j in range(num_columns)]
        # Match each column with its max index and its value.
        column_order = sorted( range(num_columns), key=lambda col: (max_indices[col], -cm_prepared[max_indices[col], col]) )
        # Reorder the matrix.
        cm_new = cm_prepared[:, column_order].tolist()
        
        # Add back the last column which contains noise if there was noise.
        if there_is_noise == True:
            for i, row in enumerate(cm_new):
                row.append(last_column[i])

        # Display what was done.
        transformation_dict = {j+1: i+1 for i, j in enumerate(column_order)}
        transformation_dict[len(column_order)+1] = len(column_order)+1
        print(f"We will apply this: transformation = {transformation_dict}")      

        # Reset the variable.
        cm_filtered = cm_new
    
    # Convert to np array.
    cm_filtered = np.array(cm_filtered)    

    # Prepare lists for the labels.
    xticks = [x + 0.5 for x in list(range(0, number_predicted_clusters))]
    yticks = [x + 0.5 for x in list(range(0, number_true_clusters))]
    ylabels = [str(i) for i in range(1, number_true_clusters+1)]
    if there_is_noise == False:    
        xlabels = [str(i) for i in range(1, number_predicted_clusters+1)]
    else:
        xlabels = [str(i) for i in range(1, number_predicted_clusters)]
        xlabels.append("Noise")
    
    plt.figure(figsize=(12, 10))
    plt.subplot(1, 1, 1)
    ax = sns.heatmap(cm_filtered, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 12}, cbar=False)
    plt.xlabel('Prediction', fontsize=16)
    plt.ylabel('Truth', fontsize=16)
    plt.xticks(xticks, xlabels, fontsize=12) #rotation=90
    plt.yticks(yticks, ylabels, fontsize=12)
    plt.title('Confusion matrix from HDBSCAN', size=20)
    if variables.save_images == True:
        plt.savefig(variables.project_name + "_HDBSCAN_ConfusionMatrix.png", bbox_inches='tight', pad_inches=0.1, dpi=150)
    if variables.show_images == True:
        plt.show()
    plt.close()   

    # Count the number of pure and uncut clusters. First condition is for the case with no noise, second condition is for the other case.
    if there_is_noise == False:
        # Count the number of pure clusters.
        pure_clusters = np.sum(cm_filtered != 0, axis=0)
        print(f"In the {cm_filtered.shape[1]} found clusters, there are {np.sum(pure_clusters == 1)} pure clusters.")
        # Count the number of uncut clusters.
        non_zero_per_row = np.sum(cm_filtered != 0, axis=1)
        print(f"There are {np.sum(non_zero_per_row == 1)} uncut clusters.")
    else:
        # Count the number of pure clusters.
        cm_filtered_no_noise = cm_filtered[:, :-1]
        pure_clusters = np.sum(cm_filtered_no_noise != 0, axis=0)
        print(f"In the {cm_filtered_no_noise.shape[1]} found clusters, there are {np.sum(pure_clusters == 1)} pure clusters.")
        # Count the number of uncut clusters.
        non_zero_per_row = np.sum(cm_filtered_no_noise != 0, axis=1)
        print(f"There are {np.sum(non_zero_per_row == 1)} uncut clusters.")
        print("There are", str(round(np.sum(hdbscan.labels_ == -1) / len(hdbscan.labels_) * 100, 1)), "% of noise.")

###########################################################
 
def plot_confusion_matrix_OPTICS(variables, labels_true, transformation = {}, auto_reorder_columns=False):
    """
    Plot the confusion matrix to compare clustering results from OPTICS with true labels.

    This function computes and plots the confusion matrix to compare the clustering results obtained from the OPTICS algorithm with the true labels.
    It also calculates various clustering metrics such as Adjusted Rand Index, Adjusted Mutual Information, Normalized Mutual Information, Homogeneity, Completeness, V-measure, Fowlkes-Mallows, Davies-Bouldin and Silhouette scores.

    Parameters:
    variables (Variables): An instance of the Variables class containing clustering data.
    labels_true (list): A list of true labels for the data points.
    transformation (dict, optional): A dictionary to transform cluster labels. Default is an empty dictionary.
    auto_reorder_columns (bool, optional): Whether to automatically reorder the confusion matrix for better visualization. Default is False.

    Returns:
    None
    """
    
    import seaborn as sns

    # Get the same parameter as YACARE for the minimal number of elements in a cluster.
    min_cluster_size = int(variables.minimal_size_cluster*variables.num_elements/100)
    if variables.raw_data_is_distance_matrix == False:
        optics = cluster.OPTICS(min_cluster_size=min_cluster_size)
        labels_optics = optics.fit_predict(variables.raw_data)
    else:
        optics = cluster.OPTICS(metric='precomputed', min_cluster_size=min_cluster_size)
        labels_optics = optics.fit_predict(variables.distance_matrix)

    # Transform the noise from being labelled -1 and become last cluster, and start indices of clusters at 1 and not at 0.
    labels_optics_new = [max(labels_optics)+2 if x == -1 else x+1 for x in labels_optics]
    
    # Define a variable to know if there is noise.
    there_is_noise = True if -1 in optics.labels_ else False
    if there_is_noise == True:
        print("Noise was found by OPTICS.")
        
    # Apply the transformation to all elements in the list.
    labels_optics_transformed = [transformation.get(x, x) for x in labels_optics_new]

    # Get some parameters.
    number_true_clusters = len(np.unique(labels_true))
    number_predicted_clusters = len(np.unique(labels_optics_transformed))

    # Compute the confusion matrix.
    cm = confusion_matrix(labels_true, labels_optics_transformed)

    # Save the labels.
    variables.labels_optics = labels_optics_transformed
    
    #Get the score of clustering
    print(f"Adjusted Rand Index score = {adjusted_rand_score(labels_true, labels_optics_transformed):.4f}")
    print(f"Adjusted Mutual Information score = {adjusted_mutual_info_score(labels_true, labels_optics_transformed):.4f}")
    print(f"Normalized Mutual Information score = {normalized_mutual_info_score(labels_true, labels_optics_transformed):.4f}")
    print(f"Homogeneity, completeness and v-measure scores = {homogeneity_completeness_v_measure(labels_true, labels_optics_transformed)[0]:.4f}, {homogeneity_completeness_v_measure(labels_true, labels_optics_transformed)[1]:.4f}, {homogeneity_completeness_v_measure(labels_true, labels_optics_transformed)[2]:.4f}")
    print(f"Fowlkes-Mallows score = {fowlkes_mallows_score(labels_true, labels_optics_transformed):.4f}")
    if variables.raw_data_is_distance_matrix == False:
        print(f"Davies-Bouldin score = {davies_bouldin_score(variables.raw_data, labels_optics_transformed):.4f}")
        print(f"Silhouette score = {silhouette_score(variables.raw_data, labels_optics_transformed):.4f}")
        
    # Restrict the number of lines and columns that are displayed.
    cm_filtered = cm[:number_true_clusters, :]
    cm_filtered = cm_filtered[:, :number_predicted_clusters]

    # If we have choosen to automatically reorder.
    if auto_reorder_columns == True:
        #Prepare the working array. First case is if there is no noise, second case is if there is noise.
        if there_is_noise == False:
            cm_prepared = cm_filtered
        else:
            cm_prepared = [row[:-1] for row in cm_filtered]
            last_column = [row[-1] for row in cm_filtered]
        
        # Convert in numpy array.
        cm_prepared = np.array(cm_prepared)
        num_rows, num_columns = cm_prepared.shape
        # Find the indices of lines with the maximal values for each column.
        max_indices = [np.argmax(cm_prepared[:, j]) for j in range(num_columns)]
        # Match each column with its max index and its value.
        column_order = sorted( range(num_columns), key=lambda col: (max_indices[col], -cm_prepared[max_indices[col], col]) )
        # Reorder the matrix.
        cm_new = cm_prepared[:, column_order].tolist()
        
        # Add back the last column which contains noise if there was noise.
        if there_is_noise == True:
            for i, row in enumerate(cm_new):
                row.append(last_column[i])

        # Display what was done.
        transformation_dict = {j+1: i+1 for i, j in enumerate(column_order)}
        transformation_dict[len(column_order)+1] = len(column_order)+1
        print(f"We will apply this: transformation = {transformation_dict}")      

        # Reset the variable.
        cm_filtered = cm_new
    
    # Convert to np array.
    cm_filtered = np.array(cm_filtered)  
    
    # Prepare lists for the labels.
    xticks = [x + 0.5 for x in list(range(0, number_predicted_clusters))]
    yticks = [x + 0.5 for x in list(range(0, number_true_clusters))]
    ylabels = [str(i) for i in range(1, number_true_clusters+1)]
    if there_is_noise == False:    
        xlabels = [str(i) for i in range(1, number_predicted_clusters+1)]
    else:
        xlabels = [str(i) for i in range(1, number_predicted_clusters)]
        xlabels.append("Noise")
    
    plt.figure(figsize=(12, 10))
    plt.subplot(1, 1, 1)
    ax = sns.heatmap(cm_filtered, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 12}, cbar=False)
    plt.xlabel('Prediction', fontsize=16)
    plt.ylabel('Truth', fontsize=16)
    plt.xticks(xticks, xlabels, fontsize=12) #rotation=90
    plt.yticks(yticks, ylabels, fontsize=12)
    plt.title('Confusion matrix from OPTICS', size=20)
    if variables.save_images == True:
        plt.savefig(variables.project_name + "_OPTICS_ConfusionMatrix.png", bbox_inches='tight', pad_inches=0.1, dpi=150)
    if variables.show_images == True:
        plt.show()
    plt.close()
    
    # Count the number of pure and uncut clusters. First condition is for the case with no noise, second condition is for the other case.
    if there_is_noise == False:
        # Count the number of pure clusters.
        pure_clusters = np.sum(cm_filtered != 0, axis=0)
        print(f"In the {cm_filtered.shape[1]} found clusters, there are {np.sum(pure_clusters == 1)} pure clusters.")
        # Count the number of uncut clusters.
        non_zero_per_row = np.sum(cm_filtered != 0, axis=1)
        print(f"There are {np.sum(non_zero_per_row == 1)} uncut clusters.")
    else:
        # Count the number of pure clusters.
        cm_filtered_no_noise = cm_filtered[:, :-1]
        pure_clusters = np.sum(cm_filtered_no_noise != 0, axis=0)
        print(f"In the {cm_filtered_no_noise.shape[1]} found clusters, there are {np.sum(pure_clusters == 1)} pure clusters.")
        # Count the number of uncut clusters.
        non_zero_per_row = np.sum(cm_filtered_no_noise != 0, axis=1)
        print(f"There are {np.sum(non_zero_per_row == 1)} uncut clusters.")
        print("There are", str(round(np.sum(optics.labels_ == -1) / len(optics.labels_) * 100, 1)), "% of noise.")
        
###########################################################

def plot_confusion_matrix_kmeans(variables, labels_true, transformation = {}, auto_reorder_columns=False, n_components=2):
    """
    Plot the confusion matrix to compare clustering results from k-Means with true labels.

    This function computes and plots the confusion matrix to compare the clustering results obtained from the k-Means algorithm with the true labels.
    It also calculates various clustering metrics such as Adjusted Rand Index, Adjusted Mutual Information, Normalized Mutual Information, Homogeneity, Completeness, V-measure, Fowlkes-Mallows, Davies-Bouldin and Silhouette scores.

    Parameters:
    variables (Variables): An instance of the Variables class containing clustering data.
    labels_true (list): A list of true labels for the data points.
    transformation (dict, optional): A dictionary to transform cluster labels. Default is an empty dictionary.
    auto_reorder_columns (bool, optional): Whether to automatically reorder the confusion matrix for better visualization. Default is False.

    Returns:
    None
    """
        
    from sklearn.manifold import MDS     
    from sklearn.cluster import KMeans   
    import seaborn as sns
    
    # Reduce the data to an euclidean space (of dimension n_components) if we started directly from a distance matrix. Otherwise, use directly the data.
    if variables.raw_data_is_distance_matrix == True:
        mds = MDS(n_components=n_components, dissimilarity="precomputed", random_state=42)
        dataToProcess = mds.fit_transform(variables.raw_data)
    else:
        dataToProcess = variables.raw_data
    
    # Cluster with k-means, using the same number of clusters as in the truth.
    kmeans = KMeans(n_clusters=len(np.unique(labels_true)), random_state=42)
    labels = kmeans.fit_predict(dataToProcess)
    labels_kmeans_transformed = [x+1 for x in labels]            # To start from 1 and not 0

    # Get some parameters (in number_predicted_clusters, we include the noise).
    number_true_clusters = len(np.unique(labels_true))
    number_predicted_clusters = max(labels_kmeans_transformed)
    
    # Compute the confusion matrix.
    cm = confusion_matrix(labels_true, labels_kmeans_transformed)
    
    # Save the labels.
    variables.labels_kmeans = labels_kmeans_transformed
    
    #Get the score of clustering.
    print(f"Adjusted Rand Index score = {adjusted_rand_score(labels_true, labels_kmeans_transformed):.4f}")
    print(f"Adjusted Mutual Information score = {adjusted_mutual_info_score(labels_true, labels_kmeans_transformed):.4f}")
    print(f"Normalized Mutual Information score = {normalized_mutual_info_score(labels_true, labels_kmeans_transformed):.4f}")
    print(f"Homogeneity, completeness and v-measure scores = {homogeneity_completeness_v_measure(labels_true, labels_kmeans_transformed)[0]:.4f}, {homogeneity_completeness_v_measure(labels_true, labels_kmeans_transformed)[1]:.4f}, {homogeneity_completeness_v_measure(labels_true, labels_kmeans_transformed)[2]:.4f}")
    print(f"Fowlkes-Mallows score = {fowlkes_mallows_score(labels_true, labels_kmeans_transformed):.4f}")
    if variables.raw_data_is_distance_matrix == False:
        print(f"Davies-Bouldin score = {davies_bouldin_score(variables.raw_data, labels_kmeans_transformed):.4f}")
        print(f"Silhouette score = {silhouette_score(variables.raw_data, labels_kmeans_transformed):.4f}")

    # Restrict the number of lines and columns that are displayed.
    cm_filtered = cm[:number_true_clusters, :]
    cm_filtered = cm_filtered[:, :number_predicted_clusters]

    # If we have choosen to automatically reorder.
    if auto_reorder_columns == True:
        # Convert in numpy array.
        cm_prepared = np.array(cm_filtered)
        num_rows, num_columns = cm_prepared.shape
        # Find the indices of lines with the maximal values for each column.
        max_indices = [np.argmax(cm_prepared[:, j]) for j in range(num_columns)]
        # Match each column with its max index and its value.
        column_order = sorted( range(num_columns), key=lambda col: (max_indices[col], -cm_prepared[max_indices[col], col]) )
        # Reorder the matrix.
        cm_new = cm_prepared[:, column_order].tolist()

        # Display what was done.
        transformation_dict = {j+1: i+1 for i, j in enumerate(column_order)}
        transformation_dict[len(column_order)+1] = len(column_order)+1
        print(f"We will apply this: transformation = {transformation_dict}")      

        # Reset the variable.
        cm_filtered = cm_new

    # Convert to np array.
    cm_filtered = np.array(cm_filtered)
    
    # Prepare lists for the labels
    xticks = [x + 0.5 for x in list(range(0, number_predicted_clusters))]
    yticks = [x + 0.5 for x in list(range(0, number_true_clusters))]
    xlabels = [str(i) for i in range(1, number_predicted_clusters+1)]
    ylabels = [str(i) for i in range(1, number_true_clusters+1)]

    plt.figure(figsize=(12, 10))
    plt.subplot(1, 1, 1)
    ax = sns.heatmap(cm_filtered, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 12}, cbar=False)
    plt.xlabel('Prediction', fontsize=16)
    plt.ylabel('Truth', fontsize=16)
    plt.xticks(xticks, xlabels, fontsize=12)
    plt.yticks(yticks, ylabels, fontsize=12)
    plt.title('Confusion matrix from k-Means', size=20)
    if variables.save_images == True:
        plt.savefig(variables.project_name + "_kMeans_ConfusionMatrix.png", bbox_inches='tight', pad_inches=0.1, dpi=150)
    if variables.show_images == True:
        plt.show()
    plt.close()
    
    # Count the number of pure clusters. 
    pure_clusters = np.sum(cm_filtered != 0, axis=0)
    print(f"In the {cm_filtered.shape[1]} found clusters, there are {np.sum(pure_clusters == 1)} pure clusters.")
    # Count the number of uncut clusters
    non_zero_per_row = np.sum(cm_filtered != 0, axis=1)
    print(f"There are {np.sum(non_zero_per_row == 1)} uncut clusters.")

###########################################################

def define_functions_for_density_peaks():
    """We define here the functions that are needed for density peaks clustering. We do it inside a function for clarity during development."""

    def create_new_format(dist_matrix):
        """Create a new format for the distance matrix."""
        num_points = dist_matrix.shape[0]
        new_format = []
        for i in range(num_points):
            for j in range(i + 1, num_points):
                new_format.append([i, j, dist_matrix[i, j]])
        return np.array(new_format)

    def compute_rho(dist_matrix, dc):
        """Compute the local density (rho) using a Gaussian kernel."""
        num_points = dist_matrix.shape[0]
        rho = np.zeros(num_points)
        for i in range(num_points - 1):
            for j in range(i + 1, num_points):
                rho[i] += np.exp(-(dist_matrix[i, j] / dc) ** 2)
                rho[j] += np.exp(-(dist_matrix[i, j] / dc) ** 2)
        return rho

    def compute_delta(dist_matrix, rho):
        """Compute the delta and nearest neighbors."""
        num_points = dist_matrix.shape[0]
        sorted_indices = np.argsort(-rho)
        delta = np.full(num_points, np.max(dist_matrix))
        nneigh = np.zeros(num_points)

        for i in range(1, num_points):
            for j in range(i):
                if dist_matrix[sorted_indices[i], sorted_indices[j]] < delta[sorted_indices[i]]:
                    delta[sorted_indices[i]] = dist_matrix[sorted_indices[i], sorted_indices[j]]
                    nneigh[sorted_indices[i]] = sorted_indices[j]

        delta[sorted_indices[0]] = np.max(delta)
        return delta, nneigh

    def assign_clusters(rho, delta, rhomin, deltamin, nneigh):
        """Assign clusters based on rho and delta values."""
        ND = len(rho)
        NCLUST = 0
        cl = -np.ones(ND, dtype=int)
        icl = np.zeros(ND, dtype=int)

        # Determine initial cluster centers.
        for i in range(ND):
            if rho[i] > rhomin and delta[i] > deltamin:
                NCLUST += 1
                cl[i] = NCLUST
                icl[NCLUST - 1] = i + 1

        print(f'NUMBER OF CLUSTERS: {NCLUST}')
        print('Performing assignment')

        # Assign points to clusters.
        ordrho = np.argsort(-rho)
        for i in range(ND):
            if cl[ordrho[i]] == -1:
                cl[ordrho[i]] = cl[int(nneigh[ordrho[i]])]

        return cl, icl, NCLUST

    def identify_halo(cl, rho, dist, dc, NCLUST):
        """Identify halo regions within the clusters."""
        ND = len(rho)
        halo = cl.copy()

        if NCLUST > 1:
            bord_rho = np.zeros(NCLUST)
            for i in range(ND - 1):
                for j in range(i + 1, ND):
                    if cl[i] != cl[j] and dist[i, j] <= dc:
                        rho_aver = (rho[i] + rho[j]) / 2.0
                        if rho_aver > bord_rho[int(cl[i]) - 1]:
                            bord_rho[int(cl[i]) - 1] = rho_aver
                        if rho_aver > bord_rho[int(cl[j]) - 1]:
                            bord_rho[int(cl[j]) - 1] = rho_aver

            for i in range(ND):
                if rho[i] < bord_rho[int(cl[i]) - 1]:
                    halo[i] = 0

        return halo

    def extract_clusters(data, cl, halo, NCLUST):
        """Extract clusters and halo points from data."""
        List_New_TRJ_All = []
        List_New_TRJ_Noise = []

        for i in range(NCLUST):
            Clust_all = data[cl == i + 1]
            Clust_noise = data[halo == i + 1]
            List_New_TRJ_All.append(Clust_all)
            List_New_TRJ_Noise.append(Clust_noise)

        return List_New_TRJ_All, List_New_TRJ_Noise

    return create_new_format, compute_rho, compute_delta, assign_clusters, identify_halo, extract_clusters

###########################################################

def plot_confusion_matrix_density_peaks_decision_graph(variables, percent=0.5):
    """
    Plot the decision graph for density peaks clustering.
    
    Parameters:
    variables (Variables): An instance of the Variables class containing clustering data.
    percent (float, optional): A value of the percentage of the ordered distances list to choose cutoff. Default is 0.5.

    Returns:
    None
    """
    
    global dc, rho, delta, nneigh
    create_new_format, compute_rho, compute_delta, assign_clusters, identify_halo, extract_clusters = define_functions_for_density_peaks()
    
    # Create the new format for the distance matrix.
    new_format = create_new_format(variables.distance_matrix)

    # Determine the cutoff distance (dc).
    num_distances = new_format.shape[0]
    sorted_distances = np.sort(new_format[:, 2])
    position = round(num_distances * percent / 100)
    dc = sorted_distances[position - 1]
    
    print(f'Average percentage of neighbours: {percent}')
    print(f'Computing rho with gaussian kernel of radius: {dc}')

    # Compute local density (rho).
    rho = compute_rho(variables.distance_matrix, dc)

    # Compute delta and nearest neighbors.
    delta, nneigh = compute_delta(variables.distance_matrix, rho)

    # Plot the decision graph.
    plt.figure(figsize=(8, 8))
    plt.scatter(rho, delta, c='k', s=15)
    plt.minorticks_on()
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.grid(which='major', linestyle='-', linewidth=0.75, color='black')
    plt.grid(which='minor', linestyle=':', linewidth=0.5, color='gray')
    #plt.axhline(2, color='green', linewidth=0.5)
    #plt.axvline(48, color='green', linewidth=0.5)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title('Decision graph', fontsize=20)
    plt.xlabel(r'$\rho$', fontsize=16)
    plt.ylabel(r'$\delta$', fontsize=16)
    if variables.save_images == True:
        plt.savefig(variables.project_name + "_DensityPeaks_DecisionGraph.png", bbox_inches='tight', pad_inches=0.1, dpi=150)
    if variables.show_images == True:
        plt.show()
    plt.close()

###########################################################

def plot_confusion_matrix_density_peaks(variables, labels_true, rhomin, deltamin, transformation = {}, auto_reorder_columns=False):
    """
    Plot the confusion matrix to compare clustering results from density peaks with true labels.

    This function computes and plots the confusion matrix to compare the clustering results obtained from the density peaks algorithm with the true labels.
    It also calculates various clustering metrics such as Adjusted Rand Index, Adjusted Mutual Information, Normalized Mutual Information, Homogeneity, Completeness, V-measure, Fowlkes-Mallows, Davies-Bouldin and Silhouette scores.

    Parameters:
    variables (Variables): An instance of the Variables class containing clustering data.
    labels_true (list): A list of true labels for the data points.
    rhomin (float): The minimal value of rho to define a centroid. This must be read in the decision graph.
    deltamin (float): The minimal value of delta to define a centroid. This must be read in the decision graph.
    transformation (dict, optional): A dictionary to transform cluster labels. Default is an empty dictionary.
    auto_reorder_columns (bool, optional): Whether to automatically reorder the confusion matrix for better visualization. Default is False.

    Returns:
    None
    """

    import seaborn as sns
    create_new_format, compute_rho, compute_delta, assign_clusters, identify_halo, extract_clusters = define_functions_for_density_peaks()
    
    # Assign clusters and identify halos.
    cl, icl, NCLUST = assign_clusters(rho, delta, rhomin, deltamin, nneigh)
    halo = identify_halo(cl, rho, variables.distance_matrix, dc, NCLUST)
    
    # Extract clusters and halo points.
    clusters, halos = extract_clusters(variables.distance_matrix, cl, halo, NCLUST)
    
    # Print cluster information.
    for i in range(NCLUST):
        nc = np.sum(cl == i + 1)
        nh = np.sum(halo == i + 1)
        print(f'CLUSTER: {i + 1}, CENTER: {icl[i]}, ELEMENTS: {nc}, CORE: {nh}, HALO: {nc - nh}')

    # Check if there is noise
    there_is_noise = True
    if len(cl) == variables.distance_matrix.shape[0]:
        print("All points were assigned to a cluster, there is no noise.")
        there_is_noise = False
    else:
        print("Not all points were assigned to a cluster, there is noise.")
        
    # Get some parameters (in number_predicted_clusters, we include the noise).
    number_true_clusters = len(np.unique(labels_true))
    number_predicted_clusters = max(cl)
    
    # Compute the confusion matrix.
    cm = confusion_matrix(labels_true, cl)

    # Save the labels.
    variables.labels_density_peaks = cl
    
    #Get the score of clustering.
    print(f"Adjusted Rand Index score = {adjusted_rand_score(labels_true, cl):.4f}")
    print(f"Adjusted Mutual Information score = {adjusted_mutual_info_score(labels_true, cl):.4f}")
    print(f"Normalized Mutual Information score = {normalized_mutual_info_score(labels_true, cl):.4f}")
    print(f"Homogeneity, completeness and v-measure scores = {homogeneity_completeness_v_measure(labels_true, cl)[0]:.4f}, {homogeneity_completeness_v_measure(labels_true, cl)[1]:.4f}, {homogeneity_completeness_v_measure(labels_true, cl)[2]:.4f}")
    print(f"Fowlkes-Mallows score = {fowlkes_mallows_score(labels_true, cl):.4f}")
    if variables.raw_data_is_distance_matrix == False:
        print(f"Davies-Bouldin score = {davies_bouldin_score(variables.raw_data, cl):.4f}")
        print(f"Silhouette score = {silhouette_score(variables.raw_data, cl):.4f}")
        
    # Restrict the number of lines and columns that are displayed.
    cm_filtered = cm[:number_true_clusters, :]
    cm_filtered = cm_filtered[:, :number_predicted_clusters]
        
    # If we have choosen to automatically reorder.
    if auto_reorder_columns == True:
        #Prepare the working array. First case is if there is no noise, second case is if there is noise.
        if there_is_noise == False:
            cm_prepared = cm_filtered
        else:
            cm_prepared = [row[:-1] for row in cm_filtered]
            last_column = [row[-1] for row in cm_filtered]

        # Convert in numpy array.
        cm_prepared = np.array(cm_prepared)
        num_rows, num_columns = cm_prepared.shape
        # Find the indices of lines with the maximal values for each column.
        max_indices = [np.argmax(cm_prepared[:, j]) for j in range(num_columns)]
        # Match each column with its max index and its value.
        column_order = sorted( range(num_columns), key=lambda col: (max_indices[col], -cm_prepared[max_indices[col], col]) )
        # Reorder the matrix.
        cm_new = cm_prepared[:, column_order].tolist()
        
        # Add back the last column which contains noise if there was noise.
        if there_is_noise == True:
            for i, row in enumerate(cm_new):
                row.append(last_column[i])

        # Display what was done.
        transformation_dict = {j+1: i+1 for i, j in enumerate(column_order)}
        transformation_dict[len(column_order)+1] = len(column_order)+1
        print(f"We will apply this: transformation = {transformation_dict}")      

        # Reset the variable.
        cm_filtered = np.array(cm_new)

    # Prepare lists for the labels.
    xticks = [x + 0.5 for x in list(range(0, number_predicted_clusters))]
    yticks = [x + 0.5 for x in list(range(0, number_true_clusters))]
    ylabels = [str(i) for i in range(1, number_true_clusters+1)]
    if there_is_noise == False:    
        xlabels = [str(i) for i in range(1, number_predicted_clusters+1)]
    else:
        xlabels = [str(i) for i in range(1, number_predicted_clusters)]
        xlabels.append("Noise")

    plt.figure(figsize=(12, 10))
    plt.subplot(1, 1, 1)
    ax = sns.heatmap(cm_filtered, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 12}, cbar=False)
    plt.xlabel('Prediction', fontsize=16)
    plt.ylabel('Truth', fontsize=16)
    plt.xticks(xticks, xlabels, fontsize=12) #rotation=90
    plt.yticks(yticks, ylabels, fontsize=12)
    plt.title('Confusion matrix from density peaks', size=20)
    if variables.save_images == True:
        plt.savefig(variables.project_name + "_DensityPeaks_ConfusionMatrix.png", bbox_inches='tight', pad_inches=0.1, dpi=150)
    if variables.show_images == True:
        plt.show()
    plt.close()

    # Count the number of pure and uncut clusters. First condition is for the case with no noise, second condition is for the other case.
    if there_is_noise == False:
        # Count the number of pure clusters.
        pure_clusters = np.sum(cm_filtered != 0, axis=0)
        print(f"In the {cm_filtered.shape[1]} found clusters, there are {np.sum(pure_clusters == 1)} pure clusters.")
        # Count the number of uncut clusters.
        non_zero_per_row = np.sum(cm_filtered != 0, axis=1)
        print(f"There are {np.sum(non_zero_per_row == 1)} uncut clusters.")
    else:
        # Count the number of pure clusters.
        cm_filtered_no_noise = cm_filtered[:, :-1]
        pure_clusters = np.sum(cm_filtered_no_noise != 0, axis=0)
        print(f"In the {cm_filtered_no_noise.shape[1]} found clusters, there are {np.sum(pure_clusters == 1)} pure clusters.")
        # Count the number of uncut clusters.
        non_zero_per_row = np.sum(cm_filtered_no_noise != 0, axis=1)
        print(f"There are {np.sum(non_zero_per_row == 1)} uncut clusters.")
        
###########################################################

def autopilot(variables, project_name, file_name, save_images = False, show_images = True, percentage_moving_square = 1.0, indices=[], minimal_size_cluster = 2.0, function_for_ratio = 1, threshold_variable = 1.0, choice_merging_clusters = 3, amount_of_noise = 1.0, keep_no_noise = False):
    """
    Do everything automatically. For now we don't include as parameters: delimiter, comments, dtype, usecols, vmax, use_all_cutoff, display_stddev, display_mean_distances.
    
    Parameters:
    variables (Variables): An instance of the Variables class containing clustering data.
    project_name (string): The name of the project.
    file_name (string): The file that will be loaded.
    same_images (bool, optional): Whether to save images or not. Default is False.
    percentage_moving_square (float, optional): The size of the moving stencil in percentage of the dataset size. Default is 1.0.
    indices (list, optional): The list of starting points that will be tried for reordering. Default is empty list, and all elements will be tried.
    minimal_size_cluster (float, optional): The minimal size that a cluster must have to be kept in percentage of the dataset size. Default is 2.0%.
    function_for_ratio (int, optional): Function to compute the ratio (1 for length/variance, 2 for inverse of variance). Default is 1.
    threshold_variable (float, optional): The threshold value used to determine if clusters should be merged (alpha in the article). Default is 1.0.
    choice_merging_clusters (int, optional): The strategy for merging clusters. Default is 3.
    amount_of_noise (float, optional): The threshold value used to determine if data points should be rescued from noise. Default is 1.0.
    keep_no_noise (bool, optional): Whether to keep all data points as noise. Default is False.
    
    Returns:
    None
    
    """
    
    variables.project_name = project_name
    variables.file_name = file_name
    variables.save_images = save_images
    variables.show_images = show_images
        
    load_data(variables, delimiter=",", comments=('#', '@'), dtype=np.float32, usecols=None)

    print(f"percentage_moving_square is set to {percentage_moving_square}.")
    perform_first_reordering(variables, percentage_moving_square = percentage_moving_square, vmax = -1)
    
    indices = indices if len(indices) != 0 else list(range(0, variables.num_elements))        
    choose_if_we_reorder_again(variables, indices = indices, vmax = -1)

    print(f"minimal_size_cluster is set to {minimal_size_cluster}.")
    find_optimal_cutoff(variables, minimal_size_cluster = minimal_size_cluster, use_all_cutoff = True, function_for_ratio = 1)
    
    find_final_clusters(variables, vmax=-1)
    
    print(f"threshold_variable is set to {threshold_variable}.")
    print(f"choice_merging_clusters is set to {choice_merging_clusters}.")
    propose_list_for_concatenating_clusters(variables, threshold_variable = threshold_variable, choice_merging_clusters = choice_merging_clusters)
    
    concatenate_clusters(variables, vmax = -1)

    print(f"amount_of_noise is set to {amount_of_noise}.")
    expand_clusters(variables, amount_of_noise = amount_of_noise, keep_no_noise = keep_no_noise, vmax = -1)
    
    compare_final_clusters(variables, display_stddev = True, display_mean_distances = True)
    
    write_indices(variables)

###########################################################
