import warnings

import numpy as np
from scipy import spatial
import scipy.sparse as sp

from sklearn.cluster import KMeans
from sklearn.cluster.k_means_ import (
    _init_centroids,
    _labels_inertia,
    _tolerance,
    _validate_center_shape,
)
from sklearn.utils import (
    check_array,
    check_random_state,
    as_float_array,
)
from sklearn.cluster import _k_means
from sklearn.preprocessing import normalize
from sklearn.externals.joblib import Parallel, delayed
from sklearn.utils.extmath import row_norms, squared_norm
from sklearn.metrics import pairwise_distances_argmin, pairwise_distances

# def normalize(v):
#     v = np.around(v, decimals=7)
#     norm = np.linalg.norm(v, ord=2, axis=1).reshape(-1,1)
#     if len(np.where(norm==0)[0]):
#         norm[np.where(norm==0)[0]] = 1
#     return np.around(v / norm, decimals = 7)

# def _centers_dense(X, labels, n_clusters, distances):
#     """M step of the K-means EM algorithm
#     Computation of cluster centers / means.
#     Parameters
#     ----------
#     X : array-like, shape (n_samples, n_features)
#     labels : array of integers, shape (n_samples)
#         Current label assignment
#     n_clusters : int
#         Number of desired clusters
#     distances : array-like, shape (n_samples)
#         Distance to closest cluster for each sample.
#     Returns
#     -------
#     centers : array, shape (n_clusters, n_features)
#         The resulting centers
#     """
#     ## TODO: add support for CSR input
#     n_samples = X.shape[0]
#     n_features = X.shape[1]
#     centers = np.zeros((n_clusters, n_features), dtype=np.float)

#     n_samples_in_cluster = np.bincount(labels, minlength=n_clusters)
#     empty_clusters = np.where(n_samples_in_cluster == 0)[0]
#     # maybe also relocate small clusters?

#     if len(empty_clusters):
#         # find points to reassign empty clusters to
#         far_from_centers = distances.argsort()#[::-1]

#         for i, cluster_id in enumerate(empty_clusters):
#             # XXX two relocated clusters could be close to each other
#             new_center = X[far_from_centers[i]]
#             centers[cluster_id] = new_center
#             n_samples_in_cluster[cluster_id] = 1

#     for i in range(n_samples):
#         for j in range(n_features):
#             centers[labels[i], j] += X[i, j]

#     centers /= n_samples_in_cluster[:, np.newaxis]

#     return centers

def _spherical_kmeans_single_lloyd(X, n_clusters, max_iter=300,
                                   init='k-means++', verbose=True,
                                   x_squared_norms=None,
                                   random_state=None, tol=1e-4,
                                   precompute_distances=True):
    '''
    Modified from sklearn.cluster.k_means_.k_means_single_lloyd.
    '''
    random_state = check_random_state(random_state)

    best_labels, best_inertia, best_centers = None, None, None

    # init
    centers = _init_centroids(X, n_clusters, init, random_state=random_state,
                              x_squared_norms=x_squared_norms)
    if verbose:
        print("Initialization complete")

    # Allocate memory to store the distances for each sample to its
    # closer center for reallocation in case of ties
    distances = np.zeros(shape=(X.shape[0],), dtype=X.dtype)

    # iterations
    for i in range(max_iter):
        centers_old = centers.copy()

        # labels assignment
        # TODO: _labels_inertia should be done with cosine distance
        #       since ||a - b|| = 2(1 - cos(a,b)) when a,b are unit normalized
        #       this doesn't really matter.
        labels, inertia = \
           _labels_inertia(X, x_squared_norms, centers,
                           precompute_distances=precompute_distances,
                           distances=distances)

        # S = np.dot(X, centers.T)
        # labels = np.argmax(S, axis=1).astype(np.int32).reshape(-1)
        # inertia = np.zeros([n_clusters])
        # for ii in np.unique(labels):
        #     inertia[ii] = np.sum(S[labels==ii, ii])
        # inertia = np.sum(inertia)

        # for ii in range(X.shape[0]):
        #     distances[ii] = S[ii, labels[ii]]
        
        # computation of the means
        if sp.issparse(X):
            centers = _k_means._centers_sparse(X, labels, n_clusters,
                                               distances)
        else:
            centers = _k_means._centers_dense(X, labels, n_clusters, distances)

        # l2-normalize centers (this is the main contibution here)
        centers = normalize(centers)

        if verbose:
            print("Iteration %2d, inertia %.3f" % (i, inertia))

        if best_inertia is None or inertia < best_inertia:
            best_labels = labels.copy()
            best_centers = centers.copy()
            best_inertia = inertia

        center_shift_total = squared_norm(centers_old - centers)
        if center_shift_total <= tol:
            if verbose:
                print("Converged at iteration %d: "
                      "center shift %e within tolerance %e"
                      % (i, center_shift_total, tol))
            break
    
    #print(center_shift_total)
    if center_shift_total > 0:
        # rerun E-step in case of non-convergence so that predicted labels
        # match cluster centers
        best_labels, best_inertia = \
            _labels_inertia(X, x_squared_norms, best_centers,
                            precompute_distances=precompute_distances,
                            distances=distances)
        
        # S = np.dot(X, best_centers.T)
        # best_labels = np.argmax(S, axis=1).astype(np.int32)
        # best_inertia = np.zeros([n_clusters])
        # for ii in np.unique(labels):
        #     best_inertia[ii] = np.sum(S[labels==ii, ii])
        # best_inertia = np.sum(best_inertia)

        # for ii in range(X.shape[0]):
        #     distances[ii] = S[ii, best_labels[ii]]

    return best_labels, best_inertia, best_centers, i + 1, distances


def spherical_k_means(X, n_clusters, init='k-means++', n_init=10,
                      max_iter=300, verbose=True, tol=1e-4, random_state=None,
                      copy_x=True, n_jobs=1, algorithm="auto",
                      return_n_iter=False):
    """Modified from sklearn.cluster.k_means_.k_means.
    """
    if n_init <= 0:
        raise ValueError("Invalid number of initializations."
                         " n_init=%d must be bigger than zero." % n_init)
    random_state = check_random_state(random_state)

    if max_iter <= 0:
        raise ValueError('Number of iterations should be a positive number,'
                         ' got %d instead' % max_iter)

    best_inertia = np.infty
    X = as_float_array(X, copy=copy_x)
    tol = _tolerance(X, tol)

    if hasattr(init, '__array__'):
        init = check_array(init, dtype=X.dtype.type, copy=True)
        _validate_center_shape(X, n_clusters, init)

        if n_init != 1:
            warnings.warn(
                'Explicit initial center position passed: '
                'performing only one init in k-means instead of n_init=%d'
                % n_init, RuntimeWarning, stacklevel=2)
            n_init = 1

    # precompute squared norms of data points
    x_squared_norms = row_norms(X, squared=True)
    #print(np.unique(x_squared_norms))

    if n_jobs == 1:
        # For a single thread, less memory is needed if we just store one set
        # of the best results (as opposed to one set per run per thread).
        for it in range(n_init):
            # run a k-means once
            labels, inertia, centers, n_iter_, distances = _spherical_kmeans_single_lloyd(
                X, n_clusters, max_iter=max_iter, init=init, verbose=verbose,
                tol=tol, x_squared_norms=x_squared_norms,
                random_state=random_state)

            # determine if these results are the best so far
            if best_inertia is None or inertia < best_inertia:
                best_labels = labels.copy()
                best_centers = centers.copy()
                best_inertia = inertia
                best_n_iter = n_iter_
                best_dist = distances.copy()
            #print(best_n_iter)
    else:
        # parallelisation of k-means runs
        seeds = random_state.randint(np.iinfo(np.int32).max, size=n_init)
        results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(_spherical_kmeans_single_lloyd)(
                X,
                n_clusters,
                max_iter=max_iter, init=init,
                verbose=verbose, tol=tol,
                x_squared_norms=x_squared_norms,
                # Change seed to ensure variety
                random_state=seed
            )
            for seed in seeds)

        # Get results with the lowest inertia
        labels, inertia, centers, n_iters, distances = zip(*results)
        best = np.argmin(inertia)
        best_labels = labels[best]
        best_inertia = inertia[best]
        best_centers = centers[best]
        best_n_iter = n_iters[best]
        best_dist = distances[best]

    if return_n_iter:
        return best_centers, best_labels, best_inertia, best_n_iter, best_dist
    else:
        return best_centers, best_labels, best_inertia, best_dist


class SphericalKMeans(KMeans):
    """Spherical K-Means clustering

    Modfication of sklearn.cluster.KMeans where cluster centers are normalized
    (projected onto the sphere) in each iteration.

    Parameters
    ----------

    n_clusters : int, optional, default: 8
        The number of clusters to form as well as the number of
        centroids to generate.

    max_iter : int, default: 300
        Maximum number of iterations of the k-means algorithm for a
        single run.

    n_init : int, default: 10
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.

    init : {'k-means++', 'random' or an ndarray}
        Method for initialization, defaults to 'k-means++':
        'k-means++' : selects initial cluster centers for k-mean
        clustering in a smart way to speed up convergence. See section
        Notes in k_init for more details.
        'random': choose k observations (rows) at random from data for
        the initial centroids.
        If an ndarray is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.

    tol : float, default: 1e-4
        Relative tolerance with regards to inertia to declare convergence

    n_jobs : int
        The number of jobs to use for the computation. This works by computing
        each of the n_init runs in parallel.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.

    random_state : integer or numpy.RandomState, optional
        The generator used to initialize the centers. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.

    verbose : int, default 0
        Verbosity mode.

    copy_x : boolean, default True
        When pre-computing distances it is more numerically accurate to center
        the data first.  If copy_x is True, then the original data is not
        modified.  If False, the original data is modified, and put back before
        the function returns, but small numerical differences may be introduced
        by subtracting and then adding the data mean.

    normalize : boolean, default True
        Normalize the input to have unnit norm.

    Attributes
    ----------

    cluster_centers_ : array, [n_clusters, n_features]
        Coordinates of cluster centers

    labels_ :
        Labels of each point

    inertia_ : float
        Sum of distances of samples to their closest cluster center.
    """
    def __init__(self, n_clusters=8, init='k-means++', n_init=10,
                 max_iter=300, tol=1e-4, n_jobs=1,
                 verbose=0, random_state=None, copy_x=True, normalize=True, centers_old=None, dist_old=None, check_dup=False):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.verbose = verbose
        self.random_state = random_state
        self.copy_x = copy_x
        self.n_jobs = n_jobs
        self.normalize = normalize
        self.centers_old = centers_old
        self.dist_old = dist_old
        self.check_dup = check_dup

    def fit(self, X, y=None):
        """Compute k-means clustering.

        Parameters
        ----------

        X : array-like or sparse matrix, shape=(n_samples, n_features)
        """
        if self.normalize:
            X = normalize(X)

        random_state = check_random_state(self.random_state)
        X = self._check_fit_data(X)

        self.cluster_centers_, self.labels_, self.inertia_, self.n_iter_ = \
            spherical_k_means(
                X, n_clusters=self.n_clusters, init=self.init,
                n_init=self.n_init, max_iter=self.max_iter,
                verbose=self.verbose,
                tol=self.tol, random_state=random_state, copy_x=self.copy_x,
                n_jobs=self.n_jobs,
                return_n_iter=True
            )
    
    def fit_transform(self, X, y=None):
        # check same points: precision 1e-8
        if self.normalize:
            X = normalize(X)
        
        self.data_ = X
        random_state = check_random_state(self.random_state)
        self.cluster_centers_, self.labels_, self.inertia_, self.n_iter_, self.best_dist = \
            spherical_k_means(
                X, n_clusters=self.n_clusters, init=self.init,
                n_init=self.n_init, max_iter=self.max_iter,
                verbose=self.verbose,
                tol=self.tol, random_state=random_state, copy_x=self.copy_x,
                n_jobs=self.n_jobs,
                return_n_iter=True
            )

        if self.centers_old is not None:
            index = np.where((self.best_dist - self.dist_old) > 0)[0]
            self.labels_[index] = self.n_clusters
            self.best_dist[index] = self.dist_old[index]
            for i in range(self.n_clusters):
                if i in self.labels_:
                    new_center = np.mean(self.data_[self.labels_==i,:], axis=0).reshape(1,-1)
                    #print(new_center)
                    new_center = normalize(new_center)
                    self.cluster_centers_[i,:] = new_center
            self.cluster_centers_ = np.concatenate((self.cluster_centers_, self.centers_old.reshape(1,-1)), axis=0)
            
            #self.labels_ = pairwise_distances_argmin(X, self.cluster_centers_)
            #self.labels_ = np.argmax(np.dot(X, self.cluster_centers_.T), axis=1).astype(np.int32)

        # if len(labels_.shape)>1:
        #     self.labels_ = np.array(labels_).squeeze(axis=1)
        #     print(type(labels_))
        # else:
        #     self.labels_ = labels_
        # print(self.labels_.shape)
        #weight = np.zeros([self.cluster_centers_.shape[0]])
        #for i in range(self.cluster_centers_.shape[0]):
        #    if len(np.where(self.labels_==i)[0]):
        #        weight[i] = np.mean(np.dot(X[self.labels_==i,:], self.cluster_centers_[i,:].reshape(-1,1)))
        #    else:
        #        weight[i] = 0
        #
        #return weight