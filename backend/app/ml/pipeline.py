import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report

from app.ml.interpretability import (
    generate_confusion_matrix,
    generate_feature_importance
)

class MLPipeline:
    def __init__(self, df: pd.DataFrame, target_column: str):
        self.df = df
        self.target_column = target_column
        self.X = df.drop(columns=[target_column])
        self.y = df[target_column]
        self.problem_type = self._detect_problem_type()
        self.models = self._define_models()
        self.params = self._define_param_grid()

    def _detect_problem_type(self):
        if self.y.nunique() <= 10 and self.y.dtype != 'float':
            return 'classification'
        else:
            raise ValueError("Apenas problemas de classificação com até 10 classes são suportados nesta versão.")

    def _define_models(self):
        return {
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "RandomForest": RandomForestClassifier(),
            "SVC": SVC()
        }

    def _define_param_grid(self):
        return {
            "LogisticRegression": {"classifier__C": [0.1, 1, 10]},
            "RandomForest": {"classifier__n_estimators": [50, 100]},
            "SVC": {"classifier__C": [0.1, 1], "classifier__kernel": ["linear", "rbf"]},
        }

    def _preprocess(self):
        num_cols = self.X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        cat_cols = self.X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, num_cols),
                ("cat", categorical_transformer, cat_cols)
            ]
        )

        self.feature_names = num_cols + cat_cols
        return preprocessor

    def run(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, stratify=self.y, test_size=0.2, random_state=42
        )

        preprocessor = self._preprocess()

        best_score = 0
        best_model_name = None
        best_report = None
        best_predictions = None
        best_conf_matrix = None
        best_feature_importance = None

        for name, model in self.models.items():
            pipe = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("classifier", model)
            ])

            grid = GridSearchCV(pipe, param_grid=self.params[name], cv=3, scoring="accuracy")
            grid.fit(X_train, y_train)

            score = grid.score(X_test, y_test)

            if score > best_score:
                best_score = score
                best_model_name = name
                y_pred = grid.predict(X_test)
                best_report = classification_report(y_test, y_pred, output_dict=True)
                best_predictions = pd.DataFrame({"actual": y_test, "predicted": y_pred})
                best_conf_matrix = generate_confusion_matrix(y_test, y_pred)
                feature_names = grid.best_estimator_.named_steps["preprocessor"].get_feature_names_out()
                best_feature_importance = generate_feature_importance(
                    grid.best_estimator_.named_steps["classifier"], feature_names
                )

        return {
            "problem_type": self.problem_type,
            "best_model": best_model_name,
            "accuracy": best_score,
            "report": best_report,
            "confusion_matrix_image": best_conf_matrix,
            "feature_importance_image": best_feature_importance,
            "predictions": best_predictions.to_dict(orient="records")
        }
