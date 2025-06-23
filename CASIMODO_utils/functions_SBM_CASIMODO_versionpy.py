import numpy as np
from numba import njit
# --- MC sampler (non-uniform multiplicities, local energy diff) ---




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


def initiate_H_and_J(multiplicities,ncoord):
    max_multiplicity = np.max(multiplicities)
    n_states = max_multiplicity*ncoord
    H = np.zeros((n_states), dtype=np.float64)
    J = np.zeros((n_states, n_states), dtype=np.float64)
    for i in range(ncoord):
        for m in range(max_multiplicity):
            idx_state= i * max_multiplicity + m
            if m < multiplicities[i]:
                H[idx_state] = -np.random.uniform(0, 1)
            else:
                H[idx_state] = np.inf

    for i in range(ncoord):
        for j in range(ncoord):
            if i != j:
                for mi in range(max_multiplicity):
                    for mj in range(max_multiplicity):
                        idx_state_i = i * max_multiplicity + mi
                        idx_state_j = j * max_multiplicity + mj
                        if mi < multiplicities[i] and mj < multiplicities[j]:
                            J[idx_state_i, idx_state_j] = -np.random.uniform(-1, 1)
                            J[idx_state_j, idx_state_i] = J[idx_state_i, idx_state_j]
                        else:
                            J[idx_state_i, idx_state_j] = np.inf
                            J[idx_state_j, idx_state_i] = np.inf
            else :
                for mi in range(max_multiplicity):
                    for mj in range(max_multiplicity):
                        idx_state_i = i * max_multiplicity + mi
                        idx_state_j = j * max_multiplicity + mj
                        if mi==mj:
                            J[idx_state_i, idx_state_j] = 0
                        else:
                            J[idx_state_i, idx_state_j] = np.inf
                            J[idx_state_j, idx_state_i] = np.inf
    return H, J


@njit(fastmath=True)
def compute_frequencies(Traj_MC, nsteps, tot_mult, multiplicities, ncoord, max_mult):
    single_frequencies = np.zeros(tot_mult, dtype=np.float64)
    double_frequencies = np.zeros((tot_mult, tot_mult), dtype=np.float64)
    active_states = np.empty(ncoord, dtype=np.int32)

    inv_nsteps = 1.0 / nsteps

    for step in range(nsteps):  # nsteps+1 rows, but use only first nsteps
        for i in range(ncoord):
            base_idx = i * max_mult
            for j in range(multiplicities[i]):
                if Traj_MC[step, base_idx + j] == 1:
                    active_states[i] = j
                    single_frequencies[base_idx + j] += inv_nsteps
                    break

        for i in range(ncoord):
            s_i = active_states[i]
            idx_i = i * max_mult + s_i
            for j in range(i + 1, ncoord):
                s_j = active_states[j]
                idx_j = j * max_mult + s_j
                double_frequencies[idx_i, idx_j] += inv_nsteps
                double_frequencies[idx_j, idx_i] += inv_nsteps  # symmetric

    return single_frequencies, double_frequencies

@njit
def get_delta_E(number_of_changes, coords_to_change, old_val, new_val,
                Traj_MC, i, max_mult, multiplicities, ncoord, H, J):
    deltaE = 0.0
    is_changed = np.zeros(ncoord, dtype=np.uint8)
    val_map_old = np.full(ncoord, -1, dtype=np.int32)
    val_map_new = np.full(ncoord, -1, dtype=np.int32)

    for j in range(number_of_changes):
        coord = coords_to_change[j]
        is_changed[coord] = 1
        val_map_old[coord] = old_val[j]
        val_map_new[coord] = new_val[j]

    for j in range(number_of_changes):
        spin = coords_to_change[j]
        idx_old = spin * max_mult + val_map_old[spin]
        idx_new = spin * max_mult + val_map_new[spin]
        deltaE += H[idx_new] - H[idx_old]

        for k in range(ncoord):
            if k == spin:
                continue
            if not is_changed[k]:
                start_k = k * max_mult
                mult_k = multiplicities[k]
                for t in range(mult_k):
                    if Traj_MC[i - 1, start_k + t] == 1:
                        idx_k = start_k + t
                        break
                deltaE += J[idx_old, idx_k] - J[idx_new, idx_k]
            elif k > spin:
                idx_k_old = k * max_mult + val_map_old[k]
                idx_k_new = k * max_mult + val_map_new[k]
                deltaE += J[idx_new, idx_k_new] - J[idx_old, idx_k_old]

    return deltaE

    
@njit
def monte_carlo(starting_coords, H, J, nsteps, tot_mult, max_mult, ncoord,multiplicities):

    Traj_MC = np.zeros((nsteps + 1, tot_mult), dtype=np.uint8)
    Traj_MC[0, :] = starting_coords
    max_number_changes = max(1, ncoord // 5)

    for step in range(1, nsteps + 1):
        number_of_changes = np.random.randint(1, max_number_changes + 1)
        coords_to_change = np.empty(number_of_changes, dtype=np.int32)
        old_val = np.empty(number_of_changes, dtype=np.int32)
        new_val = np.empty(number_of_changes, dtype=np.int32)
        is_changed = np.zeros(ncoord, dtype=np.uint8)
        val_map_old = np.full(ncoord, -1, dtype=np.int32)
        val_map_new = np.full(ncoord, -1, dtype=np.int32)

        for j in range(number_of_changes):
            spin = np.random.randint(0, ncoord)
            coords_to_change[j] = spin
            is_changed[spin] = 1
            start = spin * max_mult
            mult = multiplicities[spin]
            for k in range(mult):
                if Traj_MC[step - 1, start + k] == 1:
                    old_val[j] = k
                    break
            val_map_old[spin] = old_val[j]
            new_val[j] = (old_val[j] + np.random.randint(1, mult)) % mult
            val_map_new[spin] = new_val[j]

        deltaE = get_delta_E(number_of_changes, coords_to_change, old_val, new_val,
                             Traj_MC, step, max_mult, multiplicities, ncoord, H, J)

        if deltaE < 0 or np.random.rand() < np.exp(-deltaE):
            Traj_MC[step, :] = Traj_MC[step - 1, :]
            for j in range(number_of_changes):
                spin = coords_to_change[j]
                Traj_MC[step, spin * max_mult + val_map_old[spin]] = 0
                Traj_MC[step, spin * max_mult + val_map_new[spin]] = 1
        else:
            Traj_MC[step, :] = Traj_MC[step - 1, :]

    return Traj_MC


def get_starting_coords(multiplicities,ncoord,tot_mult, max_mult):
    starting_coords=np.zeros(tot_mult) 
    
    for c in range(ncoord):
        multiplicity = multiplicities[c]
        random_pos=np.random.randint(0,multiplicity)
        starting_coords[c * max_mult + random_pos] = 1
    return starting_coords

def get_gradient_coupled_MC (H,J,ncoord,multiplicities,single_frequencies,double_frequencies,tot_mult,max_mult,nsteps) :
    gradient_H = np.zeros(tot_mult)
    gradient_J = np.zeros((tot_mult,tot_mult))

    starting_coords=get_starting_coords(multiplicities,ncoord,tot_mult,max_mult)

    Traj_MC=monte_carlo(starting_coords, H, J, nsteps, tot_mult, max_mult, ncoord,multiplicities)
    single_MC, double_MC = compute_frequencies(Traj_MC, nsteps, tot_mult, multiplicities, ncoord, max_mult)
    gradient_H = single_frequencies-single_MC
    gradient_J = double_frequencies - double_MC
    return gradient_H,gradient_J

def update_field_coupling(H, J, gradient_H, gradient_J, learning_rate):
    H=H+learning_rate*gradient_H
    J=J+learning_rate*gradient_J
    return H,J

def train_coupled_MC (multiplicities,ncoord,max_iterations,cutoff_loss,single_frequencies,double_frequencies,learning_rate,nsteps):
    H,J=initiate_H_and_J(multiplicities,ncoord)
    tot_mult=len(H)
    max_mult= np.max(multiplicities)
    #previous_progress = -1.0  # Initialize progress tracking
    for i in range(max_iterations):
        #previous_progress = plot_progress_bar(i, max_iterations, previous_progress)
        gradient_H,gradient_J=get_gradient_coupled_MC(H,J,ncoord,multiplicities,single_frequencies,double_frequencies,tot_mult,max_mult,nsteps)
        H,J=update_field_coupling(H, J, gradient_H, gradient_J, learning_rate)

        loss_function = np.sum(np.abs(gradient_H)) + np.sum(np.abs(gradient_J))
        print(loss_function)
        if loss_function < cutoff_loss:
            print(f"Convergence reached at iteration {i+1} with loss {loss_function:.6f}")
            break
    #plot_progress_bar(max_iterations, max_iterations, previous_progress)  # Finalize progress bar
    return H,J,loss_function
#####################