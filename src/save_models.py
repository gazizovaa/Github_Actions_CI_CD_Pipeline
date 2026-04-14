import joblib, os
import prepare_data, train_model
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

def save_model(pipeline, filename):
    os.makedirs("models", exist_ok=True)
    path = f"models/{filename}"
    joblib.dump(pipeline, path)
    print(f"Saved model: {path}")

def load_model(filename):
    return joblib.load(f"models/{filename}")

if __name__ == "__main__":
    df = prepare_data.load_data()
    df = prepare_data.clean_data(df)
    X_train, X_test, y_train, y_test = prepare_data.split_data(df)

    lr = train_model.build_pipeline(X_train, LinearRegression())
    lr = train_model.train_model(lr, X_train, y_train)
    save_model(lr, "linear_regression.pkl")

    fine_tuned_rf = train_model.tune_model(X_train, y_train)
    save_model(fine_tuned_rf, "random_forest_fine_tuned.pkl")