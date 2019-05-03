import logging
import time
from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import GridSearchCV

from src.config.grid import FOLDS, SCORING_FUNCTION

log = logging.getLogger(__name__)


def grid_search(estimator,
                grid: List[dict],
                X: pd.DataFrame,
                y: pd.Series) -> Tuple[dict, float, float]:
    log.debug('grid_search.best_parameters.call')
    start = time.time()

    cv = GridSearchCV(estimator=estimator,
                      param_grid=grid,
                      cv=FOLDS,
                      scoring=SCORING_FUNCTION)

    cv.fit(X, y)

    time_left = (time.time() - start) / 60
    log.info(
        f'estimator={type(estimator).__name__},'
        f'score={-cv.best_score_},'
        f'minutes_left={time_left},'
        f'best_parameters={cv.best_params_}')

    return cv.best_params_, -cv.best_score_, time_left
