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
from libc.math cimport isfinite

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

@cdivision(True)
cdef double compute_energy(int[:] state, double[:] H, double[:, :] J, int max_mult, int ncoord) nogil:
    cdef int i, j, idx_i, idx_j
    cdef double E = 0.0
    for i in range(ncoord):
        idx_i = i * max_mult + state[i]
        E += H[idx_i]
        for j in range(i + 1, ncoord):
            idx_j = j * max_mult + state[j]
            E += J[idx_i, idx_j]
    return E


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
cpdef int monte_carlo(
    int[:] starting_state,
    double[:] H,
    double[:, :] J,
    int nsteps,
    int tot_mult,
    int max_mult,
    int ncoord,
    int[:] multiplicities,
    DTYPE_t[:] single_frequencies,
    DTYPE_t[:, :] double_frequencies,
    int[:, :] saved_states, 
    double[:] saved_energies
):
    # copy while GIL is held
    cdef int[:] current_state = np.copy(starting_state)

    # call inner worker with nogil
    return monte_carlo_worker(
        current_state, H, J, nsteps, tot_mult, max_mult,
        ncoord, multiplicities, single_frequencies, double_frequencies,saved_states,saved_energies
    )

cdef int monte_carlo_worker(
    int[:] current_state,
    double[:] H,
    double[:, :] J,
    int nsteps,
    int tot_mult,
    int max_mult,
    int ncoord,
    int[:] multiplicities,
    DTYPE_t[:] single_frequencies,
    DTYPE_t[:, :] double_frequencies,
    int[:, :] saved_states, 
    double[:] saved_energies
) nogil:
    cdef int step, spin, old_val, new_val
    cdef int i, j, idx_old, idx_new, idx_k
    cdef double deltaE, prob
    cdef bint accept
    cdef int n_accepts = 0
    cdef int idx_i, idx_j, s_i, s_j

    # Record initial state
    for i in range(ncoord):
        s_i = current_state[i]
        idx_i = i * max_mult + s_i
        single_frequencies[idx_i] += 1.0
        for j in range(i + 1, ncoord):
            s_j = current_state[j]
            idx_j = j * max_mult + s_j
            double_frequencies[idx_i, idx_j] += 1.0
            double_frequencies[idx_j, idx_i] += 1.0

    for step in range(1, nsteps * 10):
        spin = rand() % ncoord
        old_val = current_state[spin]
        new_val = (old_val + 1 + rand() % (multiplicities[spin] - 1)) % multiplicities[spin]

        idx_old = spin * max_mult + old_val
        idx_new = spin * max_mult + new_val

        deltaE = H[idx_new] - H[idx_old]

        for i in range(ncoord):
            if i == spin:
                continue
            idx_k = i * max_mult + current_state[i]
            deltaE += J[idx_new, idx_k] - J[idx_old, idx_k]

        if deltaE < 0.0:
            accept = True
        else:
            prob = exp(-deltaE / 10.0)
            accept = (rand() / <double>RAND_MAX) < prob

        if accept:
            current_state[spin] = new_val
            n_accepts += 1

            # Record frequencies
            for i in range(ncoord):
                s_i = current_state[i]
                idx_i = i * max_mult + s_i
                single_frequencies[idx_i] += 1.0

            for i in range(ncoord):
                s_i = current_state[i]
                idx_i = i * max_mult + s_i
                for j in range(i + 1, ncoord):
                    s_j = current_state[j]
                    idx_j = j * max_mult + s_j
                    double_frequencies[idx_i, idx_j] += 1.0
                    double_frequencies[idx_j, idx_i] += 1.0
            for i in range(ncoord):
                saved_states[n_accepts - 1, i] = current_state[i]
            saved_energies[n_accepts - 1] = compute_energy(current_state, H, J, max_mult, ncoord)

        if n_accepts >= nsteps:
            break

    return n_accepts



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

@boundscheck(False)
@wraparound(False)
@cdivision(True)
cpdef tuple get_gradient_coupled_MC(
    cnp.ndarray[DTYPE_t, ndim=1] H,
    cnp.ndarray[DTYPE_t, ndim=2] J,
    int ncoord,
    int[:] multiplicities,
    DTYPE_t[:] single_freq,
    DTYPE_t[:, :] double_freq,
    int tot_mult,
    int max_mult,
    int nsteps,
    int[:, :] saved_states, 
    double[:] saved_energies,
    tuple runMC
):
    cdef:
        cnp.ndarray[DTYPE_t, ndim=1] grad_H = np.zeros_like(H)
        cnp.ndarray[DTYPE_t, ndim=2] grad_J = np.zeros_like(J)
        DTYPE_t[:] single_mc = np.zeros(tot_mult, dtype=np.float64)
        DTYPE_t[:, :] double_mc = np.zeros((tot_mult, tot_mult), dtype=np.float64)
        int i, j, n_effective
        DTYPE_t norm, lambda_H = 0.1, lambda_J = 0.1
        int[:] start_tmp = get_starting_coords(multiplicities, ncoord)
        double[:] new_energies = np.empty(n_effective, dtype=np.float64)
        double[:] weights = np.empty(n_effective, dtype=np.float64)
        double max_weight = -1e100  
        double weight_sum = 0.0      


    if runMC:
        # Run MC sampling
        n_effective = monte_carlo(start_tmp, H, J, nsteps, tot_mult, max_mult, ncoord,
                                multiplicities, single_mc, double_mc,saved_states,saved_energies)

        # Early exit if nothing was accepted
        if n_effective == 0:
            return grad_H, grad_J

        norm = 1.0 / n_effective

        # Normalize sampled frequencies
        for i in range(tot_mult):
            single_mc[i] *= norm

        for i in range(tot_mult):
            for j in range(tot_mult):
                double_mc[i, j] *= norm

        
    else :
        n_effective = saved_states.shape[0]
        for i in range(n_effective):
            new_energies[i] = compute_energy(saved_states[i], H, J, max_mult, ncoord)
            weights[i] = - (new_energies[i] - saved_energies[i])
            if weights[i] > max_weight:
                max_weight = weights[i]
            for i in range(n_effective):
                weights[i] = exp(weights[i] - max_weight)
                weight_sum += weights[i]

        for i in range(n_effective):
            w = weights[i] / weight_sum
            for a in range(ncoord):
                idx = a * max_mult + saved_states[i, a]
                single_mc[idx] += w
            for a in range(ncoord):
                idx_a = a * max_mult + saved_states[i, a]
                for b in range(a + 1, ncoord):
                    idx_b = b * max_mult + saved_states[i, b]
                    double_mc[idx_a, idx_b] += w
                    double_mc[idx_b, idx_a] += w

    # Gradient of H with L2 regularization, skipping invalid
    for i in range(tot_mult):
        if isfinite(H[i]) :
            grad_H[i] = single_freq[i] - single_mc[i] + 2.0 * lambda_H * H[i]

    # Gradient of J with L2 regularization, skipping invalid
    for i in range(tot_mult):
        for j in range(tot_mult):
            if isfinite(J[i, j]) :
                grad_J[i, j] = double_freq[i, j] - double_mc[i, j] + 2.0 * lambda_J * J[i, j]

    return grad_H, grad_J

            


@boundscheck(False)
@wraparound(False)
@cdivision(True)
cpdef void update_field_coupling(
    cnp.ndarray[DTYPE_t, ndim=1] H,
    cnp.ndarray[DTYPE_t, ndim=2] J,
    cnp.ndarray[DTYPE_t, ndim=1] grad_H,
    cnp.ndarray[DTYPE_t, ndim=2] grad_J,
    DTYPE_t lr
):
    cdef int i, j
    cdef int size_H = H.shape[0]
    cdef int size_J0 = J.shape[0]
    cdef int size_J1 = J.shape[1]

    # Update H
    for i in range(size_H):
        if isfinite(H[i]) :
            H[i] = H[i]- lr * grad_H[i]

    # Update J
    for i in range(size_J0):
        for j in range(size_J1):
            if isfinite(J[i,j]):
                J[i, j] = J[i,j] - lr * grad_J[i, j]


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
    cdef int[:, :] saved_states = np.empty((nsteps, ncoord), dtype=np.int32)
    cdef double[:] saved_energies = np.empty(nsteps, dtype=np.float64)
    cdef tuple runMC=True

    H, J = initiate_H_and_J(multiplicities, ncoord)
    tot_mult = H.shape[0]
    max_mult = int(np.max(multiplicities))

    for i in range(max_iters):
        if i%10==0 : 
            runMC=True
        else:
            runMC=False
        prev = plot_progress_bar(i, max_iters, prev)
        grad_H, grad_J = get_gradient_coupled_MC(H, J, ncoord, multiplicities,
                                                 single_freq, double_freq,
                                                 tot_mult, max_mult, nsteps,saved_states,saved_energies,runMC)
        update_field_coupling(H, J, grad_H, grad_J, learning_rate)
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
