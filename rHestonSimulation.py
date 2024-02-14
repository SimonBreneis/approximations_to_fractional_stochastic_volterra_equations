import rHestonQESimulation
import rHestonMarkovSimulation
import numpy as np
import psutil


class SampleGenerator:
    def __init__(self, lambda_, nu, theta, V_0, T, rho, S_0, r):
        self.lambda_ = lambda_
        self.nu = nu
        self.theta = theta
        self.V_0 = V_0
        self.T = T
        self.rho = rho
        self.S_0 = S_0
        self.r = r

    def generate(self, m, N_time, sample_paths=False, return_times=None, vol_only=False, stock_only=False, euler=False,
                 qmc=True, rng=None, rv_shift=False, verbose=0):
        pass

    def eur(self, K, m, N_time, n_maturities=None, euler=False, qmc=True, payoff='call', implied_vol=False,
            qmc_error_estimators=25, verbose=0):
        pass


class SampleGeneratorHQE(SampleGenerator):
    def __init__(self, lambda_, nu, theta, V_0, T, rho, S_0, r, H):
        SampleGenerator.__init__(self, lambda_=lambda_, nu=nu, theta=theta, V_0=V_0, T=T, rho=rho, S_0=S_0, r=r)
        self.H = H
        self.dict_of_kernels = {}

    def generate(self, m, N_time, sample_paths=False, return_times=None, vol_only=False, stock_only=False, euler=False,
                 qmc=True, rng=None, rv_shift=False, verbose=0):
        samples, self.dict_of_kernels[N_time], rng_ = \
            rHestonQESimulation.samples(H=self.H, lambda_=self.lambda_, nu=self.nu, theta=self.theta, V_0=self.V_0,
                                        T=self.T, rho=self.rho, S_0=self.S_0, r=self.r, m=m, N_time=N_time,
                                        sample_paths=sample_paths, qmc=qmc, rng=rng, return_times=return_times,
                                        rv_shift=rv_shift, kernel_dict=self.dict_of_kernels.get(N_time),
                                        verbose=verbose)
        return samples, rng_

    def eur(self, K, m, N_time, n_maturities=None, euler=False, qmc=True, payoff='call', implied_vol=False,
            qmc_error_estimators=25, verbose=0):
        n_batches, m_batch = number_batches(N_time=1 if n_maturities is None else n_maturities, m=m)
        if qmc:
            pass
        else:
            est, low, upp = np.empty(())


class SampleGeneratorMarkov(SampleGenerator):
    def __init__(self, lambda_, nu, theta, V_0, T, rho, S_0, r, nodes, weights):
        SampleGenerator.__init__(self, lambda_=lambda_, nu=nu, theta=theta, V_0=V_0, T=T, rho=rho, S_0=S_0, r=r)
        self.nodes = nodes
        self.weights = weights

    def generate(self, m, N_time, sample_paths=False, return_times=None, vol_only=False, stock_only=False, euler=False,
                 qmc=True, rng=None, rv_shift=False, verbose=0):
        return rHestonMarkovSimulation.samples(lambda_=self.lambda_, nu=self.nu, theta=self.theta, V_0=self.V_0,
                                               T=self.T, nodes=self.nodes, weights=self.weights, rho=self.rho,
                                               S_0=self.S_0, r=self.r, m=m, N_time=N_time, sample_paths=sample_paths,
                                               return_times=return_times, vol_only=vol_only, stock_only=stock_only,
                                               euler=euler, qmc=qmc, rng=rng, rv_shift=rv_shift, verbose=verbose)


def number_batches(N_time, m):
    available_memory = np.sqrt(psutil.virtual_memory().available)
    necessary_memory_per_path = 3 * np.sqrt(N_time + 1) * np.sqrt(np.array([0.]).nbytes)
    max_m_per_batch = (available_memory / necessary_memory_per_path) ** 2 / 10
    if max_m_per_batch < 1:
        raise MemoryError(f'Not enough memory to simulate sample paths of the rough Heston model with {N_time} '
                          f'time points and {m} sample paths. Roughly {necessary_memory_per_path * np.sqrt(10)}**2 '
                          f'bytes needed, while only {available_memory}**2 bytes are available.')
    n_batches = int(np.floor(m / max_m_per_batch))
    return n_batches, int(np.ceil(m / n_batches))
