
import numpy as np

class RandomProjection:



    def train(self, X, K=100, type='regular'):
        '''Train on X, (N_components, N samples)

        Parameters
        ----------

        X : train data, array with size (N_components, N samples)

        K : number of compression dimensions

        type : 'regular' or 'sparse'
            'regular' : components of random vector are selected from Normal distribution with parameters mean = 0, scale = 1.0
            'sparse' : components of random vector is +1 with probability (1/6), 0 with probability (4/6) and -1 with probability (1/6)

        '''

        # D = Number of components (aka features) of X
        X = np.array(X)

        D, N = X.shape

        if K >= D:
            raise ValueError('Original number of dimensions should be more than number of reduced components (D > K)')

        print(f'Number of components: {D}')
        print(f'Number of samples: {N}')
        print(f'Number of reduced dimensions: {K}')
        
        # Set up Gaussian random projections
        # Each dimension is drawn from N(0, 1)
        if type == 'regular':
            R = np.random.normal(loc=0.0, scale=1.0, size=(K, D))

        if type == 'sparse':
            # components of random vector is +1 with probability (1/6), 0 with probability (4/6) and -1 with probability (1/6)
            r = np.random.random((K, D))
            R = np.zeros((K, D))
            R[r < (1/6)] = np.sqrt(3)
            R[r > (5/6)] = -np.sqrt(3)

        for d in range(D):
            # Each column is a unit vector

            # Warning! if K is low, it is possible that np.linalg.norm(R[:, d]) = 0 for sparse construction of R. Simply re-sample until we get a non-zero value for the norm of the column. This is not possible for random draws from normal distribution
            while True:
                if np.linalg.norm(R[:, d]) != 0:
                    break
                
                temp = np.random.random(K)
                R[:, d] = 0
                R[temp < (1/6), d] = np.sqrt(3)
                R[temp > (5/6), d] = -np.sqrt(3)


            R[:,d] = R[:,d]/np.linalg.norm(R[:, d])

        # Calculate original euclidean distance between sample 0 and sample 1-100 using R (100 data points were used in the paper by Bingham and Mannila )
        if N <= 100: 
            raise ValueError('Data set must have at least 101 samples')

        distance_orig = np.zeros(100)
        distance_reduced = np.zeros(100)
        for n in range(0, 100):
            distance_orig[n] = np.linalg.norm(X[:, 0] - X[:, n+1])
            distance_reduced[n] = np.sqrt(D/K) * np.linalg.norm(np.matmul(R,X[:, 0]) - np.matmul(R, X[:, n+1]))

        return distance_orig, distance_reduced


if __name__ == "__main__":

    RP = RandomProjection()

    # data set (n_components (D), n_samples (N))
    X = np.random.rand(1000, 500) 

    # K = 100, Random Projection (RP)
    RPdistance_orig, RPdistance_reduced = RP.train(X, 100, 'regular')

    # K = 100, Sparse Random projection (SRP)
    RPdistance_orig, RPdistance_reduced = RP.train(X, 100, 'sparse')

    pass