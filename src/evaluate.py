import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

def nasa_score(y_true, y_pred):

    diff = y_pred.flatten() - y_true

    return np.sum(
        np.where(
            diff < 0,
            np.exp(-diff/13)-1,
            np.exp(diff/10)-1
        )
    )

def evaluate_model(y_true, y_pred):

    print(
        f"RMSE : "
        f"{np.sqrt(mean_squared_error(y_true,y_pred)):.2f}"
    )

    print(
        f"MAE : "
        f"{mean_absolute_error(y_true,y_pred):.2f}"
    )

    print(
        f"R² Score : "
        f"{r2_score(y_true,y_pred):.3f}"
    )

    print(
        f"NASA Score : "
        f"{nasa_score(y_true,y_pred):.1f}"
    )