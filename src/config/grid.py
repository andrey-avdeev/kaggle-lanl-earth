import numpy as np

from src.config.common import RANDOM_STATE, N_JOBS, VERBOSE, FOLDS

SCORING_FUNCTION = 'neg_mean_absolute_error'

NORMALIZE = False


class GridSpace:
    ard_regression = [{
        'alpha_1': np.linspace(1e-8, 0.01, 100),
        'alpha_2': np.linspace(1.e-8, 0.01, 100),
        'lambda_1': np.linspace(1.e-8, 0.01, 100),
        'threshold_lambda': np.linspace(1.e+3, 1.e+5, 10),
        'normalize': [NORMALIZE],
        'verbose': [VERBOSE]
    }]

    bayesian_ridge = [{
        'alpha_1': np.linspace(1e-8, 0.01, 100),
        'alpha_2': np.linspace(1.e-8, 0.01, 100),
        'lambda_1': np.linspace(1.e-8, 0.01, 100),
        'threshold_lambda': np.linspace(1.e+3, 1.e+5, 100),
        'normalize': [NORMALIZE]
    }]

    elastic_net = [{
        'alpha': np.linspace(0.1, 1.0, 10),
        'l1_ratio': np.linspace(0.1, 1.0, 10),
        'random_state': [RANDOM_STATE],
        'normalize': [NORMALIZE]
    }]

    elastic_net_cv = [{
        'l1_ratio': np.linspace(0.1, 1.0, 10),
        'eps': np.linspace(1e-4, 1e-2, 10),
        'n_alphas': np.linspace(1, 10, 10),
        'random_state': [RANDOM_STATE],
        'n_jobs': [N_JOBS],
        'normalize': [NORMALIZE]
    }]

    huber_regressor = [{
        'epsilon': np.linspace(1.1, 1.7, 100),
        'alpha': np.linspace(0.0001, 0.1, 100)
    }]

    lars = [{
        'eps': np.linspace(np.finfo(np.float).eps, 1e-3, 100),
        'n_nonzero_coefs': np.linspace(1, 1e+4, 100),
        'normalize': [NORMALIZE]
    }]

    lars_cv = [{
        'eps': np.linspace(np.finfo(np.float).eps, 1e-3, 100),
        'max_n_alphas': np.linspace(100, 10000, 100),
        'n_jobs': [N_JOBS],
        'normalize': [NORMALIZE]
    }]

    lasso = [{
        'alpha': np.linspace(0.01, 0.99, 10),
        'normalize': [NORMALIZE],
        'random_state': [RANDOM_STATE],
    }]

    lasso_cv = [{
        'eps': np.linspace(np.finfo(np.float).eps, 1e-3, 100),
        'n_alphas': np.linspace(100, 10000, 100),
        'normalize': [NORMALIZE],
        'n_jobs': [N_JOBS],
        'random_state': [RANDOM_STATE]
    }]

    lasso_lars = [{
        'alpha': np.linspace(0.01, 0.99, 10),
        'normalize': [NORMALIZE],
        'eps': np.linspace(np.finfo(np.float).eps, 1e-3, 100)
    }]

    lasso_lars_cv = [{
        'normalize': [NORMALIZE],
        'max_n_alphas': np.linspace(100, 10000, 100),
        'n_jobs': [N_JOBS],
        'eps': np.linspace(np.finfo(np.float).eps, 1e-3, 100)
    }]

    lasso_lars_ic = [{
        'criterion': ['bic', 'aic'],
        'normalize': [NORMALIZE],
        'eps': np.linspace(np.finfo(np.float).eps, 1e-3, 100)
    }]

    linear_regression = [{
        'normalize': [NORMALIZE],
        # 'n_jobs': [N_JOBS]
    }]

    multitask_lasso = [{
        'alpha': np.linspace(0.01, 0.99, 10),
        'normalize': [NORMALIZE],
        'random_state': [RANDOM_STATE]
    }]

    multitask_lasso_cv = [{
        'eps': np.linspace(np.finfo(np.float).eps, 1e-3, 100),
        'n_alphas': np.linspace(100, 10000, 100),
        'normalize': [NORMALIZE],
        'n_jobs': [N_JOBS],
        'random_state': [RANDOM_STATE]
    }]

    multitask_elastic_net = [{
        'alpha': np.linspace(0.01, 0.99, 10),
        'l1_ratio': np.linspace(0.01, 0.99, 10),
        'normalize': [NORMALIZE],
        'random_state': [RANDOM_STATE]
    }]

    multitask_elastic_net_cv = [{
        'l1_ratio': np.linspace(0.01, 0.99, 10),
        'eps': np.linspace(np.finfo(np.float).eps, 1e-3, 100),
        'n_alphas': np.linspace(100, 10000, 100),
        'normalize': [NORMALIZE],
        'n_jobs': [N_JOBS],
        'random_state': [RANDOM_STATE]
    }]

    ortogonal_matching_pursuit = [{
        'normalize': [NORMALIZE],
    }]

    ortogonal_matching_pursuit_cv = [{
        'normalize': [NORMALIZE],
        'n_jobs': [N_JOBS],
    }]

    passive_aggressive_regressor = [{
        'C': np.linspace(0.1, 10, 10),
        'eps': np.linspace(np.finfo(np.float).eps, 1e-3, 100),
        'random_state': [RANDOM_STATE]
    }]

    ransac_regressor = [{

    }]

    ridge = [{
        'alpha': np.concatenate([np.linspace(0.001, 1, 100), np.linspace(1, 200, 1000)]),
        'random_state': [RANDOM_STATE]
    }]

    ridge_cv = [{
        'normalize': [NORMALIZE],

    }]

    sgd_regressor = [{
        'loss ': ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
        'penalty': ['none', 'l2', 'l1', 'elasticnet'],
        'alpha': np.linspace(0.00001, 0.001, 10),
        'l1_ratio ': np.linspace(0.01, 0.4, 10),
        'random_state': [RANDOM_STATE]

    }]

    theil_sen_regressor = [{
        'random_state': [RANDOM_STATE],
        'n_jobs': [N_JOBS],

    }]

    kernel_ridge = [{
        'gamma': np.linspace(1e-8, 0.1, 10),
        'alpha': np.concatenate([np.linspace(0.001, 1, 100), np.linspace(1, 200, 1000)])
    }]
