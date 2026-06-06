# Calculate the functional-connectivity matrix of PEC(Phase-based Edge Coupling)
import numpy as np
import scipy.signal as signal

def phase_diff_time_series(eeg_ts):
    '''
    calculate the phase diff time series, the preliminary result of edge time series
    
    input: EEG time series, shape: [time points, number of nodes(electrodes)]

    output: phase diff time series of every node pair, shape: [time points, number of edges]
    '''
    # T, N, M = time points, number of nodes, number of edges
    Phase = np.angle(signal.hilbert(eeg_ts, axis=0))
    T,N = Phase.shape
    assert N > 1
    M = N*(N-1)//2

    # phase diff time series, shape is T*M
    phase_diff_ts = np.zeros([T,M])
    k = 0
    
    for i in range(1,N):
        for j in range(i):
            # phase difference of every node pair
            phase_diff_ts[:,k] = Phase[:,i]-Phase[:,j]
            k += 1
    
    return phase_diff_ts

def pec_matrix(eeg_ts):
    '''
    calculate the PEC matrix

    input: EEG time series, shape: [time points, number of nodes(electrodes)]

    output: PEC matrix, shape: [number of edges, number of edges]
    '''
    phase_diff_ts = phase_diff_time_series(eeg_ts)
    l, edge_num = phase_diff_ts.shape[0], phase_diff_ts.shape[1]
    pec_mat = np.zeros([edge_num, edge_num])

    for i in range(edge_num):
        for j in range(edge_num):
            nip1 = np.abs(np.sum(np.exp(1j * (phase_diff_ts[:,i] - phase_diff_ts[:,j]))))/l
            nip2 = np.abs(np.sum(np.exp(1j * (phase_diff_ts[:,i] + phase_diff_ts[:,j]))))/l
            pec_mat[i,j] = max(nip1, nip2)

    return pec_mat