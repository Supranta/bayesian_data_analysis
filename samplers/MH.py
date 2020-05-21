import numpy as np

class MH_sampler():
    """
    A class for doing Metroolis-Hastings Sampling
    :param ndim: Dimension of the parameter space being sampled
    :param lnprob: The log-posterior function.
    :param step_size: Initial Step size
    :param lnprob_args: (optional) extra positional arguments for the function lnprob
    """

    def __init__(self, ndim, lnprob, lnprob_args=[]):
        self.ndim = ndim
        self.lnprob = lnprob
        self.lnprob_args = lnprob_args

    def sample(self, init_pos, step_size_init, N_ITERATIONS=1000):
        """
        Samples the parameter space 'N_ITERATIONS' times
        :param init_pos: (1D Array of size ndim) The initial guess from where the chain is started
        :param step_size_init: (1D Array of size ndim) Initial guess for the size of a step-out
        :param N_ITERATIONS: Number of interations of sampling. Default is 1000.
        """
        assert init_pos.shape[0] == self.ndim, "init_pos must be an 1-D array of size n_dim"
        assert step_size_init.shape[0] == self.ndim, "step_size_init must be an 1-D array of size n_dim"

        self._chain = np.zeros((N_ITERATIONS, self.ndim))
        self._posterior = np.zeros(N_ITERATIONS)
        self._chain[0, :] = init_pos
        self.num_iterations = N_ITERATIONS
        self.step_cov = step_size_init**2
        self._accepted = 0

        x = init_pos

        for i in range(N_ITERATIONS):
            x = self.MH_one_step(x)
            self._chain[i] = x
            self._posterior[i] = self.get_lnprob(x)

    def MH_one_step(self, x):
        """
        """
        # Propose a new point with a Gaussian step
        x_proposed = x + np.random.multivariate_normal(np.zeros(2),cov=np.diag(self.step_cov))

        # Get the value of the posterior at the old point and the proposed point. Also their difference
        ln_prob_old = self.get_lnprob(x)
        ln_prob_new = self.get_lnprob(x_proposed)

        delta_ln_prob = ln_prob_new - ln_prob_old

        # accept or reject the proposed point according to the scheme mentioned above
        if(delta_ln_prob > 0):
            x_new = x_proposed
            self._accepted += 1
        else:
            u = np.random.uniform()
            if(np.log(u) > delta_ln_prob):
                x_new = x
            else:
                x_new = x_proposed
                self._accepted += 1
        return x_new

    @property
    def chain(self):
        """
        Return the chain of the sampler.
        """
        return self._chain

    @property
    def acceptance_fraction(self):
        """
        Return the chain of the sampler.
        """
        return float(self._accepted / self.num_iterations)

    @property
    def posterior(self):
        """
        Return the posterior of the sampler.
        """
        return self._posterior

    def get_lnprob(self, x):
        """Return lnprob at the given position."""
        return self.lnprob(x, *self.lnprob_args)
