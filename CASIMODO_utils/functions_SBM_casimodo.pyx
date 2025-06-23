# cython: boundscheck=False, wraparound=False, nonecheck=False, language_level=3

import numpy as np
cimport numpy as cnp
from cython cimport boundscheck, wraparound, cdivision

from libc.math cimport exp
from cython cimport inline
from libc.stdlib cimport rand, srand, RAND_MAX, malloc, free
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from cython.parallel import prange
from libc.string cimport memcpy

ctypedef cnp.float64_t DTYPE_t
ctypedef cnp.int32_t ITYPE_t
ctypedef cnp.uint8_t UTYPE_t


def plot_progress_bar(int current, int total, int previous_percent, int bar_length=40):
    if total == 0:
        return previous_percent
    cdef float progress = current / <float>total  # ensure float division
    cdef int percent = <int>(progress * 100)
    cdef int block = <int>(bar_length * progress)
    if percent == 100 or percent - previous_percent >= 1 or previous_percent == -1:
        print(f"\rProgress: [{'#' * block}{'-' * (bar_length - block)}] {percent}%", end='', flush=True)
        return percent
    return previous_percent



def initiate_H_and_J(int[:] multiplicities, int ncoord):
    cdef int max_mult = np.max(multiplicities)
    cdef int N = max_mult * ncoord
    cdef cnp.ndarray[DTYPE_t, ndim=1] H = np.empty(N, dtype=np.float64)
    cdef cnp.ndarray[DTYPE_t, ndim=2] J = np.empty((N, N), dtype=np.float64)
    H.fill(np.inf)
    J.fill(np.inf)
    cdef int i, j, mi, mj, idx_i, idx_j
    cdef double val

    for i in range(ncoord):
        for mi in range(multiplicities[i]):
            idx_i = i * max_mult + mi
            H[idx_i] = -rand() / <double>RAND_MAX

    for i in range(ncoord):
        for j in range(i + 1, ncoord):
            for mi in range(multiplicities[i]):
                idx_i = i * max_mult + mi
                for mj in range(multiplicities[j]):
                    idx_j = j * max_mult + mj
                    val = -2.0 * rand() / <double>RAND_MAX + 1.0
                    J[idx_i, idx_j] = val
                    J[idx_j, idx_i] = val
        for mi in range(multiplicities[i]):
            idx_i = i * max_mult + mi
            J[idx_i, idx_i] = 0.0

    return H, J



@boundscheck(False)
@wraparound(False)
@cdivision(True)
cpdef void compute_frequencies(
    int[:, :] Traj_MC,
    int nsteps,
    int tot_mult,
    int[:] multiplicities,
    int ncoord,
    int max_mult,
    DTYPE_t[:] single_frequencies,
    DTYPE_t[:, :] double_frequencies
):
    cdef int step, i, j
    cdef int s_i, s_j
    cdef int idx_i, idx_j
    cdef DTYPE_t inv_n = 1.0 / nsteps

    for step in range(nsteps):
        # Update single frequencies
        for i in range(ncoord):
            s_i = Traj_MC[step, i]
            idx_i = i * max_mult + s_i
            single_frequencies[idx_i] += inv_n

        # Update double frequencies (upper triangle + symmetric)
        for i in range(ncoord):
            s_i = Traj_MC[step, i]
            idx_i = i * max_mult + s_i
            for j in range(i + 1, ncoord):
                s_j = Traj_MC[step, j]
                idx_j = j * max_mult + s_j
                double_frequencies[idx_i, idx_j] += inv_n
                double_frequencies[idx_j, idx_i] += inv_n





@boundscheck(False)
@wraparound(False)
@cdivision(True)
cpdef int[:, :] monte_carlo(
    int[:] starting_state,  # now holds actual spin values per coordinate
    double[:] H,
    double[:, :] J,
    int nsteps,
    int tot_mult,
    int max_mult,
    int ncoord,
    int[:] multiplicities
):
    cdef int[:, :] Traj_MC = np.empty((nsteps + 1, ncoord), dtype=np.int32)
    cdef int[:] current_state = np.copy(starting_state)

    cdef int step, spin, old_val, new_val
    cdef int i, idx_old, idx_new, idx_k
    cdef double deltaE
    cdef bint accept

    # Store initial state
    for i in range(ncoord):
        Traj_MC[0, i] = current_state[i]

    for step in range(1, nsteps + 1):
        spin = rand() % ncoord
        old_val = current_state[spin]
        new_val = (old_val + 1 + rand() % (multiplicities[spin] - 1)) % multiplicities[spin]

        idx_old = spin * max_mult + old_val
        idx_new = spin * max_mult + new_val

        deltaE = H[idx_new] - H[idx_old]

        # Interaction energy
        for i in range(ncoord):
            if i == spin:
                continue
            idx_k = i * max_mult + current_state[i]
            deltaE += J[idx_new, idx_k] - J[idx_old, idx_k]

        accept = deltaE < 0.0 or (rand() / <double>RAND_MAX) < exp(-deltaE)

        # Store state
        if accept:
            current_state[spin] = new_val

        for i in range(ncoord):
            Traj_MC[step, i] = current_state[i]
    return Traj_MC


def get_starting_coords(int[:] multiplicities, int ncoord):
    """
    Returns an array of shape (ncoord,) with the active state index (0-based)
    for each coordinate, using random initialization.
    """
    cdef int[:] coords = np.empty(ncoord, dtype=np.int32)
    cdef int c, mult
    for c in range(ncoord):
        mult = multiplicities[c]
        coords[c] = rand() % mult
    return coords

def get_gradient_coupled_MC(
    cnp.ndarray[DTYPE_t, ndim=1] H,
    cnp.ndarray[DTYPE_t, ndim=2] J,
    int ncoord,
    int[:] multiplicities,
    DTYPE_t[:] single_freq,
    DTYPE_t[:, :] double_freq,
    int tot_mult,
    int max_mult,
    int nsteps,
    int[:] starting
):
    cdef cnp.ndarray[DTYPE_t, ndim=1] grad_H = np.zeros_like(H)
    cdef cnp.ndarray[DTYPE_t, ndim=2] grad_J = np.zeros_like(J)
    cdef int[:, :] traj
    cdef DTYPE_t[:] single_mc = np.zeros(tot_mult, dtype=np.float64)
    cdef DTYPE_t[:, :] double_mc = np.zeros((tot_mult, tot_mult), dtype=np.float64)
    cdef int i

    # Generate trajectory using integer-based spin representation
    traj = monte_carlo(starting, H, J, nsteps, tot_mult, max_mult, ncoord, multiplicities)

    # Update starting to the last state
    for i in range(ncoord):
        starting[i] = traj[nsteps - 1, i]

    # Compute MC-estimated frequencies (uses integer-based traj)
    compute_frequencies(traj, nsteps, tot_mult, multiplicities, ncoord, max_mult, single_mc, double_mc)

    # Compute gradients
    for i in range(tot_mult):
        grad_H[i] = single_freq[i] - single_mc[i]

    for i in range(tot_mult):
        for j in range(tot_mult):
            grad_J[i, j] = double_freq[i, j] - double_mc[i, j]

    return grad_H, grad_J, starting


def update_field_coupling(cnp.ndarray[DTYPE_t, ndim=1] H,
                          cnp.ndarray[DTYPE_t, ndim=2] J,
                          cnp.ndarray[DTYPE_t, ndim=1] grad_H,
                          cnp.ndarray[DTYPE_t, ndim=2] grad_J,
                          float lr):
    H -= lr * grad_H
    J -= lr * grad_J
    return H, J


def train_coupled_MC(int[:] multiplicities,
                     int ncoord, int max_iters,
                     float cutoff_loss,
                     cnp.ndarray[DTYPE_t, ndim=1] single_freq,
                     cnp.ndarray[DTYPE_t, ndim=2] double_freq,
                     float lr, int nsteps):
    cdef cnp.ndarray[DTYPE_t, ndim=1] H
    cdef cnp.ndarray[DTYPE_t, ndim=2] J
    cdef int tot_mult, max_mult, i, prev = -1
    cdef float loss
    cdef list loss_list = []
    cdef float learning_rate = 0.1

    H, J = initiate_H_and_J(multiplicities, ncoord)
    tot_mult = H.shape[0]
    max_mult = int(np.max(multiplicities))

    starting = get_starting_coords(multiplicities, ncoord)

    for i in range(max_iters):
        #if i%10==0 : 
        #    learning_rate/=10
        prev = plot_progress_bar(i, max_iters, prev)
        grad_H, grad_J,starting = get_gradient_coupled_MC(H, J, ncoord, multiplicities,
                                                 single_freq, double_freq,
                                                 tot_mult, max_mult, nsteps,starting)
        H, J = update_field_coupling(H, J, grad_H, grad_J, learning_rate)
        loss = np.sum(np.abs(grad_H)) + np.sum(np.abs(grad_J))
        loss_list.append(loss)
        if loss < cutoff_loss:
            prev = plot_progress_bar(max_iters, max_iters, prev)
            print(f"\nConvergence reached at iteration {i+1} with loss {loss:.6f}")
            break
    else:
        prev = plot_progress_bar(max_iters, max_iters, prev)
        print(f"\nWarning: Max iterations reached. Final loss: {loss:.6f}")


    return H, J, loss_list
