import numpy as np
import sys
import time

from sklearn.model_selection import KFold
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator
from scipy.linalg import cholesky, solve_triangular, svd
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator

from gaussian_process import GaussianProcess
from pygrid import PyGrid

def train_job(train_i, test_i, gamma, alpha, y_, dist):
    K_train = -gamma * dist[np.ix_(train_i, train_i)]
    np.exp(K_train, K_train)
    K_test = -gamma * dist[np.ix_(test_i, train_i)]
    np.exp(K_test, K_test)
    K_train.flat[::K_train.shape[0] + 1] += alpha
    try:
        L_ = cholesky(K_train, lower=True)
        x = solve_triangular(L_, y_[train_i], lower=True)
        dual_coef_ = solve_triangular(L_.T, x)
        pred_mean = np.dot(K_test, dual_coef_)
        e = np.mean((pred_mean - y_[test_i]) ** 2, 0)
    except np.linalg.LinAlgError:
        e = np.inf
    return e

def get_alpha_add(n_basis, n_grid, delta, v):
    alpha_add = np.pi * ((np.arange(n_basis / 2) / (n_grid * delta))**2 + v**2)/v
    alpha_add = np.repeat(alpha_add, 2)
    return alpha_add

class MultivariateGaussianProcessCV(BaseEstimator):
    def __init__(self, krr_param_grid=None, cv=5, n_components=None, single_combo=False,
                 verbose=0, copy_X=True, n_jobs=None, cluster_params=[],
                 v=None, n_basis=None, n_grid=None, delta=None):
        self.krr_param_grid = krr_param_grid
        self.verbose = verbose
        self.cv = cv
        self.n_components = n_components
        self.single_combo = single_combo
        self.copy_X = copy_X
        self.n_jobs = n_jobs
        self.cluster_params = cluster_params
        self.n_grid = n_grid
        self.delta = delta
        self.n_basis = n_basis
        if 'v' in self.krr_param_grid is not None and not single_combo:
            raise ValueError('Can only add to alpha if single_combo=True')
    
    def score(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def fit(self, X, y, labels=None, dist=None):
        t = time.time()

        if y.ndim < 2:
            y = y.reshape(-1, 1)

        if self.n_components is not None:
            if self.verbose > 0:
                elapsed = time.time() - t
                print('PCA [%dmin %dsec]' % (int(elapsed / 60),
                                             int(elapsed % 60)))
            sys.stdout.flush()
            self.pca = PCA(n_components=self.n_components, svd_solver='arpack')
            y_ = self.pca.fit_transform(y)
            if self.verbose > 0:
                print('Lost %.1f%% information ' % (self.pca.noise_variance_) +
                      '[%dmin %dsec]' % (int(elapsed / 60), int(elapsed % 60)))
                elapsed = time.time() - t
        else:
            y_ = y

        if labels is not None:
                raise RuntimeError('Not implemented.')

        if type(self.cv) == int:
            kfold = KFold(n_splits=self.cv)
            cv_folds = kfold.split(X)
            n_cv_folds = kfold.get_n_splits()
        elif hasattr(self.cv, '__iter__'):
            cv_folds = self.cv
            n_cv_folds = len(self.cv)
        else:
            cv_folds = self.cv.split(X)
            n_cv_folds = self.cv.get_n_splits()


        if self.verbose > 0:
            elapsed = time.time() - t
            print('Computing distance matrix [%dmin %dsec]' % (
                int(elapsed / 60), int(elapsed % 60)))
            sys.stdout.flush()

        if dist is None:
            dist = euclidean_distances(X, None, squared=True)

        errors = []
        if self.n_jobs is not None:
            run_params = []
            n_folds = 0
            for fold_i, (train_i, test_i) in enumerate(cv_folds):
                n_folds += 1
                for gamma_i, gamma in enumerate(self.krr_param_grid['gamma']):
                    for alpha_i, alpha in enumerate(self.krr_param_grid['alpha']):
                        run_params.append({'train_i': train_i, 'test_i': test_i,
                                           'gamma': gamma, 'alpha': alpha})
            if self.verbose > 0:
                elapsed = time.time() - t
                print('Starting PyGrid jobs [%dmin %dsec]' % (
                    int(elapsed / 60), int(elapsed % 60)))
                sys.stdout.flush()
            pygrid = PyGrid(id='1', temp_folder="/home/fbrockherde/dft", always_cleanup=True)
            pygrid.map('train_job', args={'dist': dist, 'y_': y_}, cargs=run_params, use_cluster=True,
                       njobs=self.n_jobs, cluster_params=self.cluster_params)
            l_old = None
            while True:
                l = len(pygrid.still_running())
                if l == 0:
                    break
                elif l_old is None or l_old != l:
                    if self.verbose > 0:
                        elapsed = time.time() - t
                        print('Waiting for %s of %s jobs. ' % (l, len(run_params)) +
                              '[%dmin %dsec]' % (int(elapsed / 60), int(elapsed % 60)))
                        sys.stdout.flush()
                    l_old = l
                    time.sleep(5)
                    if l <= 10:
                        print(pygrid.still_running())
            if self.verbose > 0:
                elapsed = time.time() - t
                print('Getting results [%dmin %dsec]' % (int(elapsed / 60), int(elapsed % 60)))
                sys.stdout.flush()
            res = pygrid.get_results()

            foi = 0 
            for fold_i in range(n_folds):
                fold_errors = np.empty((len(self.krr_param_grid['gamma']),
                                        len(self.krr_param_grid['alpha']), y_.shape[1]))
                for gamma_i, gamma in enumerate(self.krr_param_grid['gamma']):
                    for alpha_i, alpha in enumerate(self.krr_param_grid['alpha']):
                        fold_errors[gamma_i, alpha_i] = res[foi]
                        foi += 1
                errors.append(fold_errors)
            errors = np.array(errors)
            errors = np.mean(errors, 0)  # average over folds
        elif 'v' in self.krr_param_grid:
            for fold_i, (train_i, test_i) in enumerate(cv_folds):
                fold_errors = np.empty((len(self.krr_param_grid['v']),
                                        len(self.krr_param_grid['gamma']),
                                        len(self.krr_param_grid['alpha']), y_.shape[1]))
                if self.verbose > 0:
                    elapsed = time.time() - t
                    print('CV %d of %d [%dmin %dsec]' % (fold_i + 1,
                                                         n_cv_folds,
                                                         int(elapsed / 60),
                                                         int(elapsed % 60)))
                    sys.stdout.flush()
                for v_i, v in enumerate(self.krr_param_grid['v']):
                    for gamma_i, gamma in enumerate(self.krr_param_grid['gamma']):
                        if self.verbose > 0:
                            sys.stdout.write('.')
                            sys.stdout.flush()
                        K_train = -gamma * dist[np.ix_(train_i, train_i)]
                        np.exp(K_train, K_train)
                        K_test = -gamma * dist[np.ix_(test_i, train_i)]
                        np.exp(K_test, K_test)
                        for alpha_i, alpha in enumerate(self.krr_param_grid['alpha']):
                            if self.verbose > 0:
                                sys.stdout.write(',')
                                sys.stdout.flush()
                            for y_i in np.arange(y_.shape[1]):
                                K_train_ = K_train.copy()
                                alpha_add = get_alpha_add(self.n_basis, self.n_grid, self.delta, v)
                                K_train_.flat[::K_train_.shape[0] + 1] += alpha * alpha_add[y_i]
                                try:
                                    L_ = cholesky(K_train_, lower=True)
                                    x = solve_triangular(L_, y_[train_i, y_i], lower=True)
                                    dual_coef_ = solve_triangular(L_.T, x)
                                    pred_mean = np.dot(K_test, dual_coef_)
                                    e = np.mean((pred_mean - y_[test_i, y_i]) ** 2, 0)
                                except np.linalg.LinAlgError:
                                    e = np.inf
                                fold_errors[v_i, gamma_i, alpha_i, y_i] = e
                if self.verbose > 0:
                    sys.stdout.write('\n')
                    sys.stdout.flush()
                errors.append(fold_errors)
            errors = np.array(errors)
            errors = np.mean(errors, 0)  # average over folds
        else:
            for fold_i, (train_i, test_i) in enumerate(cv_folds):
                fold_errors = np.empty((len(self.krr_param_grid['gamma']),
                                        len(self.krr_param_grid['alpha']), y_.shape[1]))
                if self.verbose > 0:
                    elapsed = time.time() - t
                    print('CV %d of %d [%dmin %dsec]' % (fold_i + 1,
                                                         n_cv_folds,
                                                         int(elapsed / 60),
                                                         int(elapsed % 60)))
                    sys.stdout.flush()
                for gamma_i, gamma in enumerate(self.krr_param_grid['gamma']):
                    if self.verbose > 0:
                        sys.stdout.write('.')
                        sys.stdout.flush()
                    K_train = -gamma * dist[np.ix_(train_i, train_i)]
                    np.exp(K_train, K_train)
                    K_test = -gamma * dist[np.ix_(test_i, train_i)]
                    np.exp(K_test, K_test)
                    for alpha_i, alpha in enumerate(self.krr_param_grid['alpha']):
                        if self.verbose > 0:
                            sys.stdout.write(',')
                            sys.stdout.flush()
                        K_train_ = K_train.copy()
                        K_train_.flat[::K_train_.shape[0] + 1] += alpha
                        try:
                            L_ = cholesky(K_train_, lower=True)
                            x = solve_triangular(L_, y_[train_i], lower=True)
                            dual_coef_ = solve_triangular(L_.T, x)
                            pred_mean = np.dot(K_test, dual_coef_)
                            e = np.mean((pred_mean - y_[test_i]) ** 2, 0)
                        except np.linalg.LinAlgError:
                            e = np.inf
                        fold_errors[gamma_i, alpha_i] = e
                if self.verbose > 0:
                    sys.stdout.write('\n')
                    sys.stdout.flush()
                errors.append(fold_errors)
            errors = np.array(errors)
            errors = np.mean(errors, 0)  # average over folds

        self.dual_coefs_ = np.empty((y_.shape[1], X.shape[0]))
        self.alphas_ = np.empty(y_.shape[1])
        self.gammas_ = np.empty(y_.shape[1])
        if self.verbose > 0:
            elapsed = time.time() - t
            print('Refit [%dmin %dsec]' % (int(elapsed / 60),
                                           int(elapsed % 60)))
            sys.stdout.flush()
        print_count = 0

        if not self.single_combo:
            for i in range(y_.shape[1]):
                min_params = np.argsort(errors[:, :, i], axis=None)
                lin_alg_errors = 0
                gamma_i, alpha_i = np.unravel_index(min_params[0],
                                                    errors.shape[:2])
                gamma = self.krr_param_grid['gamma'][gamma_i]
                alpha = self.krr_param_grid['alpha'][alpha_i]
                self.alphas_[i] = alpha
                self.gammas_[i] = gamma

                if (gamma_i in (0, len(self.krr_param_grid['gamma']) - 1) or
                        alpha_i in (0, len(self.krr_param_grid['alpha']) - 1)):
                    if print_count <= 200:
                        fmtstr = '%d: gamma=%g\talpha=%g\terror=%g\tmean=%g'
                        print(fmtstr % (i, gamma, alpha,
                                        errors[gamma_i, alpha_i, i],
                                        errors[gamma_i, alpha_i, i] /
                                            np.mean(np.abs(y_[:, i]))))
                        print_count += 1
        else:
            errors = np.mean(errors, -1)  # average over outputs
            min_params = np.argsort(errors, axis=None)
            if 'v' in self.krr_param_grid:
                v_i, gamma_i, alpha_i = np.unravel_index(min_params[0],
                                                errors.shape)
            else:
                gamma_i, alpha_i = np.unravel_index(min_params[0],
                                                errors.shape)
            if 'v' in self.krr_param_grid:
                v = self.krr_param_grid['v'][v_i]
                print('v=', v)
            gamma = self.krr_param_grid['gamma'][gamma_i]
            alpha = self.krr_param_grid['alpha'][alpha_i]

            if 'v' in self.krr_param_grid:
                    if v == self.krr_param_grid['v'][0]:
                        print('v at lower edge.')
                    if v == self.krr_param_grid['v'][-1]:
                        print('v at upper edge.')
            if gamma == self.krr_param_grid['gamma'][0]:
                print('Gamma at lower edge.')
            if gamma == self.krr_param_grid['gamma'][-1]:
                print('Gamma at upper edge.')
            if alpha == self.krr_param_grid['alpha'][0]:
                print('Alpha at lower edge.')
            if alpha == self.krr_param_grid['alpha'][-1]:
                print('Alpha at upper edge.')
            self.alphas_[:] = alpha
            self.gammas_[:] = gamma

            if 'v' in self.krr_param_grid:
                alpha_add = get_alpha_add(self.n_basis, self.n_grid, self.delta, v)
                self.alphas_ *= alpha_add

        combos = list(zip(self.alphas_, self.gammas_))
        n_unique_combos = len(set(combos))
        for i, (alpha, gamma) in enumerate(set(combos)):
            if self.verbose > 0:
                elapsed = time.time() - t
                print('Parameter combinations ' +
                      '%d of %d [%dmin %dsec]' % (i+1, n_unique_combos,
                                                  int(elapsed / 60),
                                                  int(elapsed % 60)))
                sys.stdout.flush()
            y_list = [i for i in range(y_.shape[1]) if
                      self.alphas_[i] == alpha and self.gammas_[i] == gamma]

            K = -gamma * dist
            np.exp(K, K)
            K.flat[::K.shape[0] + 1] += alpha
            try:
                L_ = cholesky(K, lower=True)
                x = solve_triangular(L_, y_[:, y_list], lower=True)
                dual_coef_ = solve_triangular(L_.T, x)
                self.dual_coefs_[y_list] = dual_coef_.T.copy()
            except np.linalg.LinAlgError:
                raise
        if self.copy_X:
            self.X_fit_ = X.copy()
        else:
            self.X_fit_ = X
        self.errors = errors

        if self.verbose:
            elapsed = time.time() - t
            print('Done [%dmin %dsec]' % (int(elapsed / 60), int(elapsed % 60)))
            sys.stdout.flush()

    def predict(self, X, verbose=None):
        t = time.time()
        
        if verbose is None:
            verbose = self.verbose

        y_ = np.empty(shape=(X.shape[0], len(self.alphas_)))
        if verbose > 0:
            elapsed = time.time() - t
            print('Computing distance matrix [%dmin %dsec]' % (
                int(elapsed / 60), int(elapsed % 60)))
            sys.stdout.flush()
        dist = euclidean_distances(X, self.X_fit_, squared=True)

        for i, gamma in enumerate(np.unique(self.gammas_)):
            if verbose > 0:
                print('Gamma %d of %d [%dmin %dsec]' % (i + 1,
                    len(np.unique(self.gammas_)), int(elapsed / 60),
                    int(elapsed % 60)))
                sys.stdout.flush()

            y_list = [i for i in range(len(self.gammas_)) if
                      self.gammas_[i] == gamma]
            K_test = -gamma * dist
            np.exp(K_test, K_test)
            y_[:, y_list] = np.dot(K_test, self.dual_coefs_[y_list].T)

        if self.n_components is not None:
            y = self.pca.inverse_transform(y_)
        else:
            y = y_

        if y.shape[1] == 1:
            y = y.flatten()

        if verbose:
            elapsed = time.time() - t
            print('Done [%dmin %dsec]' % (int(elapsed / 60), int(elapsed % 60)))
            sys.stdout.flush()
        return y

    def pred_variance(self, X):
        t = time.time()
        if self.verbose > 0:
            elapsed = time.time() - t
            print('Train distances [%dmin %dsec]' % (int(elapsed / 60),
                                                     int(elapsed % 60)))
            sys.stdout.flush()
        dist_train = euclidean_distances(self.X_fit_, self.X_fit_, squared=True)
        if self.verbose > 0:
            elapsed = time.time() - t
            print('Test train distances [%dmin %dsec]' % (int(elapsed / 60),
                                                          int(elapsed % 60)))
            sys.stdout.flush()
        dist_test_train = euclidean_distances(X, self.X_fit_, squared=True)
        if self.verbose > 0:
            elapsed = time.time() - t
            print('Test distances [%dmin %dsec]' % (int(elapsed / 60),
                                                    int(elapsed % 60)))
            sys.stdout.flush()
        dist_test = euclidean_distances(X, X, squared=True)
        combos = list(zip(self.alphas_, self.gammas_))
        n_unique_combos = len(set(combos))

        pred_var = np.zeros(X.shape[0])
        for i, (alpha, gamma) in enumerate(set(combos)):
            if self.verbose > 0:
                elapsed = time.time() - t
                print('Parameter combinations ' +
                      '%d of %d [%dmin %dsec]' % (i+1, n_unique_combos,
                                                  int(elapsed / 60),
                                                  int(elapsed % 60)))
                sys.stdout.flush()
            K_train = -gamma * dist_train
            np.exp(K_train, K_train)
            K_train.flat[::K_train.shape[0] + 1] += alpha
            L_ = cholesky(K_train, lower=True)
            K_test_train = -gamma * dist_test_train
            np.exp(K_test_train, K_test_train)
            V = solve_triangular(L_, K_test_train.T, lower=True)
            v = np.sum(V * V, axis=0)
            K_test = -gamma * dist_test
            np.exp(K_test, K_test)
            pred_var_ = K_test.flat[::X.shape[0] + 1] - v
            n_outputs = len([i for i in range(len(self.gammas_)) if
                             self.alphas_[i] == alpha and self.gammas_[i] == gamma])
            pred_var += pred_var_ * n_outputs / len(self.gammas_)
        if self.verbose > 0:
            elapsed = time.time() - t
            print('Done [%dmin %dsec]' % (int(elapsed / 60),
                                          int(elapsed % 60)))
            sys.stdout.flush()
        return pred_var
