import numpy as np
from kmeans import KMeans

class GMM():
    '''
        Fits a Gausian Mixture model to the data.

        attrs:
            n_cluster : Number of mixtures (Int)
            e : error tolerance (Float)
            max_iter : maximum number of updates (Int)
            init : initialization of means and variance
                Can be 'random' or 'kmeans'
            means : means of Gaussian mixtures (n_cluster X D numpy array)
            variances : variance of Gaussian mixtures (n_cluster X D X D numpy array)
            pi_k : mixture probabilities of different component ((n_cluster,) size numpy array)
    '''

    def __init__(self, n_cluster, init='k_means', max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.e = e
        self.max_iter = max_iter
        self.init = init
        self.means = None
        self.variances = None
        self.pi_k = None

    def fit(self, x):
        '''
            Fits a GMM to x.

            x: is a NXD size numpy array
            updates:
                self.means
                self.variances
                self.pi_k
        '''
        assert len(x.shape) == 2, 'x can only be 2 dimensional'

        np.random.seed(42)
        N, D = x.shape

        if (self.init == 'k_means'):
            # TODO
            # - comment/remove the exception
            # - initialize means using k-means clustering
            # - compute variance and pi_k (see P4.pdf)


            (means, memships, iters) = KMeans(self.n_cluster, self.max_iter,
                self.e).fit(x)
            self.means = means
            self.pi_k = np.zeros(self.n_cluster)
            self.variances = np.zeros((self.n_cluster,D,D))
            for i in np.arange(self.n_cluster):
                mask = (memships == i)
                total = np.sum(mask)
                self.pi_k[i] = (total/N)
                x_relevant = x[mask,:]
                covar = 1/total * np.matmul(np.transpose(x_relevant), x_relevant)
                self.variances[i,:,:] = covar

            # DONOT MODIFY CODE BELOW THIS LINE

        elif (self.init == 'random'):
            # TODO
            # - comment/remove the exception
            # - initialize means randomly
            # - initialize variance to be identity and pi_k to be uniform

            self.variances = np.zeros((self.n_cluster,D,D))

            idxes = np.random.choice(N, self.n_cluster)
            self.means = x[idxes,:]
            self.pi_k = (1/self.n_cluster)*np.ones(self.n_cluster)
            for i in np.arange(self.n_cluster):
                self.variances[i,:,:] = np.identity(D)

            # DONOT MODIFY CODE BELOW THIS LINE

        else:
            raise Exception('Invalid initialization provided')

        # TODO
        # - comment/remove the exception
        # - Use EM to learn the means, variances, and pi_k and assign them to self
        # - Update until convergence or until you have made self.max_iter updates.
        # - Return the number of E/M-Steps executed (Int)
        # Hint: Try to separate E & M step for clarity

        iternum = 0
        oldlik = 0
        newlik = self.compute_log_likelihood(x, self.means, self.variances, self.pi_k)
        while((self.max_iter > iternum) and (np.abs(newlik - oldlik) > 0.1)):

            # Expectation

            gamma = np.zeros((N,self.n_cluster))
            for k in np.arange(self.n_cluster):
                mean = self.means[k,:]
                covar = self.variances[k,:,:]
                pik = self.pi_k[k]
                for i in np.arange(N):
                    gamma[i,k] = pik * Gaussian_pdf(mean, covar).getLikelihood(x[i,:])
            gamma = np.divide(gamma, np.sum(gamma, axis=1)[:, None]) #broadcast by first dimension

            # Maximization

            Nks = np.sum(gamma, axis=0)
            self.pi_k = (1/N) * Nks

            for k in np.arange(self.n_cluster):
                self.means[k,:] = np.sum(np.multiply(x, (gamma[:, k])[:, None]), axis=0) / Nks[k]
                self.variances[k,:,:] = np.matmul(np.transpose(np.subtract(x,self.means[k,:])), np.multiply((gamma[:,k])[:, None], np.subtract(x,self.means[k,:]))) / Nks[k]

            iternum = iternum+1
            oldlik = newlik.copy()
            newlik = self.compute_log_likelihood(x, self.means, self.variances, self.pi_k)
            #print(newlik)

        return iternum

        # DONOT MODIFY CODE BELOW THIS LINE


    def sample(self, N):
        '''
        sample from the GMM model

        N is a positive integer
        return : NXD array of samples

        '''
        assert type(N) == int and N > 0, 'N should be a positive integer'
        np.random.seed(42)
        if (self.means is None):
            raise Exception('Train GMM before sampling')

        # TODO
        # - comment/remove the exception
        # - generate samples from the GMM
        # - return the samples

        D = self.means.shape[1]
        samples = np.zeros((N,D))

        for i in np.arange(N):
            Gauss_idx = np.random.choice(self.n_cluster, p=self.pi_k)
            Gauss_mean = self.means[Gauss_idx]
            Gauss_covar = self.variances[Gauss_idx,:,:]
            samples[i,:] = np.random.multivariate_normal(Gauss_mean, Gauss_covar)

        # DONOT MODIFY CODE BELOW THIS LINE
        return samples

    def compute_log_likelihood(self, x, means=None, variances=None, pi_k=None):
        '''
            Return log-likelihood for the data

            x is a NXD matrix
            return : a float number which is the log-likelihood of data
        '''
        assert len(x.shape) == 2,  'x can only be 2 dimensional'
        if means is None:
            means = self.means
        if variances is None:
            variances = self.variances
        if pi_k is None:
            pi_k = self.pi_k
        # TODO
        # - comment/remove the exception
        # - calculate log-likelihood using means, variances and pi_k attr in self
        # - return the log-likelihood (Float)
        # Note: you can call this function in fit function (if required)

        N,D = x.shape

        loglik = 0
        for i in np.arange(N):
            p = 0
            for k in np.arange(self.n_cluster):
                p = p + pi_k[k] * Gaussian_pdf(means[k,:], variances[k,:,:]).getLikelihood(x[i,:])
            loglik = loglik + np.log(p)

        # DONOT MODIFY CODE BELOW THIS LINE
        return loglik

class Gaussian_pdf():
    def __init__(self,mean,variance):
        self.mean = mean
        self.variance = variance
        self.c = None
        self.inv = None
        '''
            Input:
                Means: A 1 X D numpy array of the Gaussian mean
                Variance: A D X D numpy array of the Gaussian covariance matrix
            Output:
                None:
        '''
        # TODO
        # - comment/remove the exception
        # - Set self.inv equal to the inverse the variance matrix (after ensuring it is full rank - see P4.pdf)
        # - Set self.c equal to ((2pi)^D) * det(variance) (after ensuring the variance matrix is full rank)
        # Note you can call this class in compute_log_likelihood and fit

        D = variance.shape[0]

        def invertmat(mat, D):
            if np.linalg.cond(mat) < 1/3*np.exp(16):
                self.c = 2*np.pi*D*np.linalg.det(mat)
                self.inv = np.linalg.inv(mat)
            else:
                mat = mat + 10**(-3) * np.identity(D)
                invertmat(mat, D)

        invertmat(variance, D)

        # DONOT MODIFY CODE BELOW THIS LINE

    def getLikelihood(self,x):
        '''
            Input:
                x: a 1 X D numpy array representing a sample
            Output:
                p: a numpy float, the likelihood sample x was generated by this Gaussian
            Hint:
                p = e^(-0.5(x-mean)*(inv(variance))*(x-mean)')/sqrt(c)
                where ' is transpose and * is matrix multiplication
        '''
        #TODO
        # - Comment/remove the exception
        # - Calculate the likelihood of sample x generated by this Gaussian
        # Note: use the described implementation of a Gaussian to ensure compatibility with the solutions
        p = np.exp(-0.5*(np.dot(np.dot(np.subtract(x,self.mean),self.inv),(np.subtract(x,self.mean)).T)))/np.sqrt(self.c)

        # DONOT MODIFY CODE BELOW THIS LINE
        return p
