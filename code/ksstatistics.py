import numpy as np
import scipy.stats as stats

class StatisticalTester:

    def __init__(self, standardSamples):
        np.random.seed(123456)
        self.standardSamples = []
        for sample in standardSamples:
            # self.standardSamples.append(sample)
            # self.standardSamples.append(sample - np.mean(sample))
            # print('sample.shape = ' + str(sample.shape))
            # print('sample = ' + str(sample))
            self.standardSamples.append(stats.zscore(sample))

    # Kolmogorov Smirnov 2-sample test
    def ks_test(self, x):
        d_L = []
        pval_L = []
        chi2_L = []
        for y in self.standardSamples:
            # print('stats.zscore(x).shape = ' + str(stats.zscore(x).shape[0]) + ', y.shape = ' + str(y[:x.shape[0]].shape[0]))
            d, pval, chi2 = self.ks_2samp_modified(stats.zscore(x), y)
            # d, pval, chi2 = self.ks_2samp_modified(stats.zscore(x), y[:x.shape[0]])
            d_L.append(d)
            pval_L.append(pval)
            chi2_L.append(chi2)
        return np.array(d_L), np.array(pval_L), np.array(chi2_L)

    '''
     below copied from scipy github and modified
        https://github.com/scipy/scipy/blob/master/scipy/stats/stats.py
    '''
    def ks_2samp_modified(self, data1, data2):
        data1 = np.sort(data1)
        data2 = np.sort(data2)
        n1 = data1.shape[0]
        n2 = data2.shape[0]
        data_all = np.concatenate([data1, data2])
        cdf1 = np.searchsorted(data1, data_all, side='right') / n1
        cdf2 = np.searchsorted(data2, data_all, side='right') / n2
        d = np.max(np.absolute(cdf1 - cdf2))
        # Note: d absolute not signed distance
        # below added to scipy.stats.ks_2samp by Taro Tezuka
        chi2 = 4 * (d ** 2) * (n1 * n2) / (n1 + n2)
        # pval = 1 - stats.chi2.cdf(chi2, 1)
        en = np.sqrt(n1 * n2 / (n1 + n2))
        try:
            pval = stats.distributions.kstwobign.sf((en + 0.12 + 0.11 / en) * d)
        except Exception:
            warnings.warn('This should not happen! Please open an issue at '
                        'https://github.com/scipy/scipy/issues and provide the code '
                        'you used to trigger this warning.\n')
            pval = 1.0
        return d, pval, chi2
