import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df['Residuals'] = df['Actual Free Energy'] - df['Module Y Pred']
    return df


def train_ridge_model(df, features, label, alpha=1.0):
    df = df.dropna(subset=features + ['Residuals'])
    X = df[features]
    y = df['Residuals']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"Training Ridge model on '{label}' pairs")
    print(f"Number of training samples: {len(X_train)}")
    print("=" * 40)
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")
    print("=" * 40)

    return model, scaler


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def main():
    df = load_data("residual_predictor/training_data/revised_rr_predictions.csv")

    # Filter to just the joint module
    joint_df = df[df['Module'] == 'y-joint']

    features = ['Joint Uncertainty', 'Joint Dominance Ratio', 'Joint Effective Neighbors']
    label = "y-joint"

    Path("residual_predictor/models").mkdir(exist_ok=True)

    model, scaler = train_ridge_model(
        joint_df,
        features,
        label=label,
        alpha=1.0
    )

    save_pickle(model, "residual_predictor/models/joint_ridge_model.pkl")
    save_pickle(scaler, "residual_predictor/models/joint_scaler.pkl")

    print("Joint model and scaler saved in 'residual_predictor/models/' folder.")


if __name__ == "__main__":
    main()

