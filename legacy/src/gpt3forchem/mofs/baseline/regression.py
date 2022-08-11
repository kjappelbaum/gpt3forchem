from xgboost import XGBRegressor
from optuna.samplers import TPESampler
from optuna.integration import XGBoostPruningCallback
from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import mean_squared_error
from optuna import create_study
from gpt3forchem.baseline import BaseLineModel


class XGBRegressionBaseline(BaseLineModel):
    def __init__(self, seed, num_trials=100) -> None:
        self.seed = seed
        self.num_trials = num_trials
        self.model = XGBRegressor()

    def tune(self, X_train, y_train):
        def objective(
            trial,
            X,
            y,
            random_state=22,
            n_splits=3,
            n_jobs=1,
            early_stopping_rounds=50,
        ):
            # XGBoost parameters
            params = {
                "verbosity": 0,  # 0 (silent) - 3 (debug)
                "objective": "reg:squarederror",
                "n_estimators": 10000,
                "max_depth": trial.suggest_int("max_depth", 4, 12),
                "learning_rate": trial.suggest_loguniform("learning_rate", 0.005, 0.05),
                "colsample_bytree": trial.suggest_loguniform("colsample_bytree", 0.2, 0.6),
                "subsample": trial.suggest_loguniform("subsample", 0.4, 0.8),
                "alpha": trial.suggest_loguniform("alpha", 0.01, 10.0),
                "lambda": trial.suggest_loguniform("lambda", 1e-8, 10.0),
                "gamma": trial.suggest_loguniform("lambda", 1e-8, 10.0),
                "min_child_weight": trial.suggest_loguniform("min_child_weight", 10, 1000),
                "seed": random_state,
                "n_jobs": n_jobs,
            }

            model = XGBRegressor(**params)
            pruning_callback = XGBoostPruningCallback(trial, "validation_0-rmse")
            kf = KFold(n_splits=n_splits)
            X_values = X.values
            y_values = y.values
            scores = []
            for train_index, test_index in kf.split(X_values):
                X_A, X_B = X_values[train_index, :], X_values[test_index, :]
                y_A, y_B = y_values[train_index], y_values[test_index]
                model.fit(
                    X_A,
                    y_A,
                    eval_set=[(X_B, y_B)],
                    eval_metric="rmse",
                    verbose=0,
                    callbacks=[pruning_callback],
                    early_stopping_rounds=early_stopping_rounds,
                )
                y_pred = model.predict(X_B)
                scores.append(mean_squared_error(y_pred, y_B))
            return np.mean(scores)

        sampler = TPESampler(seed=self.seed)
        study = create_study(direction="minimize", sampler=sampler)
        study.optimize(
            lambda trial: objective(
                trial,
                X_train,
                y_train,
                random_state=self.seed,
                n_splits=5,
                n_jobs=8,
                early_stopping_rounds=100,
            ),
            n_trials=self.num_trials,
            n_jobs=1,
        )

        self.model = XGBRegressor(**study.best_params)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)
