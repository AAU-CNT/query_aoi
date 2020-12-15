import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib

filename = 'results.npz'

data = np.load(filename)
eps, p_q, p_e, p_b, M, B, gamma, aoi_pq, qaoi_pq, aoi_qapa, qaoi_qapa, av_aoi_pq, av_qaoi_pq, av_aoi_qapa, av_qaoi_qapa, pq_history, qapa_history, pq_query_history, qapa_query_history = data.values()

def plot_average(error, av_aoi_pq, av_qaoi_pq, av_aoi_qapa, av_qaoi_qapa):
    plt.figure('Average age')
    plt.xlabel('Error prob.')
    plt.ylabel('Average age')
    plt.plot(error,av_aoi_pq,label='PQ, AoI')
    plt.plot(error,av_qaoi_pq,label='PQ, QAoI')
    plt.plot(error,av_aoi_qapa,label='QAPA, AoI')
    plt.plot(error,av_qaoi_qapa,label='QAPA, QAoI')
    tikzplotlib.save("AverageAge.tex")
    plt.legend()
    plt.show()
    
def plot_ccdf(index, aoi_pq, qaoi_pq, aoi_qapa, qaoi_qapa, M, T_q):
    plt.figure('CCDF')
    plt.xlabel('AoI/QAoI')
    plt.ylabel('1-CDF')
    plt.plot(range(1, 1 + M * T_q), np.log10(1 - np.cumsum(aoi_pq[index, :])), label = 'PQ, AoI', color='tab:blue', linestyle='solid')
    plt.plot(range(1, 1 + M * T_q), np.log10(1 - np.cumsum(qaoi_pq[index, :])), label = 'PQ, QAoI', color='tab:orange', linestyle='solid')
    plt.plot(range(1, 1 + M * T_q), np.log10(1 - np.cumsum(aoi_qapa[index, :])), label = 'QAPA, AoI', color='tab:blue', linestyle='dashed')
    plt.plot(range(1, 1 + M * T_q), np.log10(1 - np.cumsum(qaoi_qapa[index, :])), label = 'QAPA, QAoI',color='tab:orange', linestyle='dashed')
    plt.legend()
    figname = 'ccdf_{}'.format(index)
    tikzplotlib.save(figname)
    plt.show()

def plot_imshow(vect, figfilename, title, T_q, maxaoi=None):
    # Compute distribution for each number of slots until query
    if maxaoi is None:
        maxaoi = np.max(vect)
    hist_bins = np.arange(1, maxaoi+1)
    dist = np.zeros((T_q, len(hist_bins)-1))
    for tq in range(T_q):
        tq_vect = vect[tq::T_q]
        dist[np.mod(tq+1, T_q), :], _edges = np.histogram(tq_vect, bins=hist_bins, density=True)
    plt.figure(figsize=(10,10))
    plt.imshow(dist.T, interpolation='none', origin='lower', aspect='auto')
    plt.title(title)
    plt.savefig(figfilename)
    
    # Generate csv
    matrixdata = []
    for i in range(dist.T.shape[0]):
        for j in range(dist.T.shape[1]):
            matrixdata.append((j, i+1, dist.T[i,j]))
    np.savetxt('{}.csv'.format(figfilename), matrixdata, fmt='%.5f')

def plot_cdf(vect, maxaoi=None):
    # Compute distribution for each number of slots until query
    if maxaoi is None:
        maxaoi = np.max(vect)
    hist_bins = np.arange(1, maxaoi+1)
    hist, _bins = np.histogram(vect, bins=hist_bins, density=True)
    plt.plot(hist_bins[:-1], np.cumsum(hist))
    #plt.savefig(figfilename)

initial_remove = 0

plot_average(eps, av_aoi_pq, av_qaoi_pq, av_aoi_qapa, av_qaoi_qapa)

for error_index in range(len(eps)):
    plot_ccdf(error_index, aoi_pq, qaoi_pq, aoi_qapa, qaoi_qapa, M, len(p_q))
    plot_imshow(pq_history[error_index,initial_remove:], 'aoi_pdf_pq_{}'.format(error_index), len(p_q), len(p_q))
    plot_imshow(qapa_history[error_index,initial_remove:], 'aoi_pdf_qapa_{}'.format(error_index), len(p_q), len(p_q))
