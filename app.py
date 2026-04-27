import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Configure the main page
st.set_page_config(page_title="Medical Insurance Costs Prediction", layout="wide",
                   initial_sidebar_state="expanded")

# Custom CSS style
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');
 
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
h1, h2, h3 {
    font-family: 'DM Serif Display', serif;
}
[data-testid="stSidebar"] {
    background: #0f1923;
}
[data-testid="stSidebar"] * {
    color: #e8e0d4 !important;
}
.sidebar-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.3rem;
    color: #f0c96b !important;
    padding: 0.5rem 0 1rem 0;
    border-bottom: 1px solid #2a3a4a;
    margin-bottom: 1rem;
}
.metric-card {
    background: #f7f3ee;
    border-left: 4px solid #c0392b;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin: 0.4rem 0;
}
.metric-card h4 { margin: 0 0 4px 0; font-size: 0.8rem; color: #888; text-transform: uppercase; letter-spacing: 1px; }
.metric-card p  { margin: 0; font-size: 1.6rem; font-weight: 600; color: #1a1a2e; }
.section-tag {
    display: inline-block;
    background: #c0392b;
    color: white;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 3px;
    margin-bottom: 0.5rem;
}
.stButton > button {
    background: #c0392b;
    color: white;
    border: none;
    border-radius: 6px;
    font-weight: 600;
    padding: 0.5rem 1.5rem;
    transition: background 0.2s;
}
.stButton > button:hover { background: #a93226; }
</style>
""", unsafe_allow_html=True)

# data helpers
data_path = "src/data/medical-charges.csv"
models_dir = "models"

@st.cache_data
def load_data():
    df = pd.read_csv(data_path)
    df = df.drop_duplicates()
    return df

@st.cache_data
def split_data(df):
    X = df.drop("charges", axis=1)
    y = np.log1p(df["charges"])
    return train_test_split(X, y, test_size=0.2, random_state=42)
 
def build_pipeline(X_train, model):
    num_features = X_train.select_dtypes(include=[np.number]).columns
    cat_features = X_train.select_dtypes(exclude=[np.number]).columns
    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                         ("onehot", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))])
    transformer = ColumnTransformer([("num", num_pipe, num_features), ("cat", cat_pipe, cat_features)], remainder="passthrough")
    return Pipeline([("preprocessing", transformer), ("model", model)])
 
def get_metrics(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    return {
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R²":   r2_score(y_test, y_pred),
    }
    
# Sidebar Navigation
with st.sidebar:
    page = st.radio(
        "Navigation",
        ["Dataset Overview", "Visualizations", "Model Training", "Fine-Tuning",
         "Evaluation", "Prediction"],
        label_visibility="collapsed"
    )

# Dataset Overview 
if page == "Dataset Overview":
    st.title("Dataset Overview")
    try:
        df = load_data()
    except FileNotFoundError:
        st.error(f"Dataset not found at `{data_path}`. Please place `medical-charges.csv` in the `data/` folder.")
        st.stop()
        
    overview_df = pd.DataFrame({
    "Metric": ["Total Rows", "Features", "Duplicates Removed", "Missing Values"],
    "Value": [
        df.shape[0],
        df.shape[1],
        df.duplicated().sum(),
        df.isnull().sum().sum()
    ]
})

    st.dataframe(overview_df.set_index("Metric"), use_container_width=True)
 
    st.markdown("### Raw Data Sample")
    st.dataframe(df.head(10), use_container_width=True)
    st.markdown("### Statistical Summary")
    st.dataframe(df.describe().T.style.format("{:.2f}"), use_container_width=True)
    st.markdown("### Column Types")
    dtype_df = pd.DataFrame({"Column": df.columns, "Type": df.dtypes.astype(str).values, "Non-Null": df.notnull().sum().values})
    st.dataframe(dtype_df, use_container_width=True)
    
# Visualizations
elif page == "Visualizations":
    st.title("Visualizations & Correlation")
    df = load_data()
    sns.set_style("whitegrid")
    palette = {"yes": "#c0392b", "no": "#2980b9"}
    tab1, tab2, tab3, tab4 = st.tabs(["Charges Distribution", "BMI and Charges", "Age and Charges", "Correlation Matrix"])
 
    with tab1:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        sns.histplot(df["charges"], bins=35, color="#c0392b", kde=True, ax=axes[0])
        axes[0].set_title("Distribution of Charges")
        sns.histplot(np.log1p(df["charges"]), bins=35, color="#2980b9", kde=True, ax=axes[1])
        axes[1].set_title("Distribution of Charges with Log Transformation")
        plt.tight_layout()
        st.pyplot(fig)
    with tab2:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.scatterplot(data=df, x="bmi", y="charges", hue="smoker", palette=palette, alpha=0.7, ax=ax)
        ax.set_title("BMI vs Charges")
        plt.tight_layout()
        st.pyplot(fig)
    with tab3:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.scatterplot(data=df, x="age", y="charges", hue="smoker", palette=palette, alpha=0.7, ax=ax)
        ax.set_title("Age vs Charges")
        plt.tight_layout()
        st.pyplot(fig)
    with tab4:
        fig, ax = plt.subplots(figsize=(8, 5))
        numeric_df = df.select_dtypes(include=[np.number])
        sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Matrix")
        plt.tight_layout()
        st.pyplot(fig)
    
# Model Training
elif page == "Model Training":
    st.title("Regression Model Training")
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    st.markdown("""
    Two regression models are trained on the preprocessed training set:
    - **Linear Regression** — baseline model
    - **Random Forest Regression** — ensemble tree model
    
    The pipeline includes: `SimpleImputer → StandardScaler / OneHotEncoder → Model`
    """)
 
    if st.button("Train Both Models"):
        with st.spinner("Training Linear Regression..."):
            lr = build_pipeline(X_train, LinearRegression())
            lr.fit(X_train, y_train)
        with st.spinner("Training Random Forest..."):
            rf = build_pipeline(X_train, RandomForestRegressor(random_state=42))
            rf.fit(X_train, y_train)
        st.session_state["lr"]  = lr
        st.session_state["rf"]  = rf
        st.session_state["X_test"]  = X_test
        st.session_state["y_test"]  = y_test
        st.success("Both models trained successfully!")
 
        results = []
        for name, model in [("Linear Regression", lr), ("Random Forest Regression", rf)]:
            m = get_metrics(model, X_test, y_test)
            m["Model"] = name
            results.append(m)
 
        res_df = pd.DataFrame(results)[["Model", "R²", "MAE", "RMSE", "MSE"]]
        st.markdown("### Quick Metrics Comparison")
        st.dataframe(res_df.style.format({"R²": "{:.4f}", "MAE": "{:.4f}", "RMSE": "{:.4f}", "MSE": "{:.4f}"}), use_container_width=True)
 
        # Actual vs Predicted plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, (name, m) in zip(axes, [("Linear Regression", lr), ("Random Forest", rf)]):
            y_pred = m.predict(X_test)
            ax.scatter(y_test, y_pred, alpha=0.4, color="#c0392b", s=15)
            lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
            ax.plot(lims, lims, "b--", linewidth=1.5)
            ax.set_xlabel("Actual log(charges)")
            ax.set_ylabel("Predicted log(charges)")
            ax.set_title(f"{name}: Actual vs Predicted")
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Click the button above to train the models.")
        
# Fine-Tuning 
elif page == "Fine-Tuning":
    st.title("Hyperparameter Fine-Tuning")
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    st.markdown("""
    **GridSearchCV** is used to find optimal hyperparameters for `RandomForestRegressor`.  
    Search space:
    """)
    param_df = pd.DataFrame({
        "Parameter": ["n_estimators", "max_depth", "min_samples_split"],
        "Values": ["[100, 200]", "[None, 10, 20]", "[2, 5]"],
    })
    st.dataframe(param_df, use_container_width=True)
    st.warning("Fine-tuning runs GridSearchCV with cv=5. This may take 1–3 minutes.")
 
    if st.button("Run Fine-Tuning"):
        model = RandomForestRegressor(random_state=42)
        pipeline = build_pipeline(X_train, model)
        param_grid = {
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5],
        }
        with st.spinner("Running GridSearchCV (cv=5)..."):
            gs = GridSearchCV(pipeline, param_grid, cv=5, scoring="r2", n_jobs=-1)
            gs.fit(X_train, y_train)
        tuned = gs.best_estimator_
        st.session_state["tuned_rf"] = tuned
        st.session_state["X_test"]   = X_test
        st.session_state["y_test"]   = y_test
        st.success("Fine-tuning complete!")
        st.markdown("### Best Parameters")
        best_params = {k.replace("model__", ""): v for k, v in gs.best_params_.items()}
        st.json(best_params)
 
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Best CV R²", f"{gs.best_score_:.4f}")
        with col2:
            m = get_metrics(tuned, X_test, y_test)
            st.metric("Test R²", f"{m['R²']:.4f}")
 
        st.markdown("### Cross-Validation Results Table")
        cv_df = pd.DataFrame(gs.cv_results_)[["param_model__n_estimators", "param_model__max_depth",
                                               "param_model__min_samples_split", "mean_test_score", "rank_test_score"]]
        cv_df.columns = ["n_estimators", "max_depth", "min_samples_split", "Mean CV R²", "Rank"]
        st.dataframe(cv_df.sort_values("Rank").style.format({"Mean CV R²": "{:.4f}"}), use_container_width=True)
    else:
        st.info("Click the button above to start fine-tuning.")

# Evaluation
elif page == "Evaluation":
    st.title("Model Evaluation & Metrics")
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    if st.button("Evaluate All Models"):
        models_info = {}
        progress = st.progress(0)
 
        with st.spinner("Training LR..."):
            lr = build_pipeline(X_train, LinearRegression())
            lr.fit(X_train, y_train)
            models_info["Linear Regression"] = lr
        progress.progress(33)
        with st.spinner("Training RF..."):
            rf = build_pipeline(X_train, RandomForestRegressor(random_state=42))
            rf.fit(X_train, y_train)
            models_info["Random Forest"] = rf
        progress.progress(66)
        with st.spinner("Fine-tuning RF..."):
            pipeline = build_pipeline(X_train, RandomForestRegressor(random_state=42))
            param_grid = {
                "model__n_estimators": [100, 200],
                "model__max_depth": [None, 10, 20],
                "model__min_samples_split": [2, 5],
            }
            gs = GridSearchCV(pipeline, param_grid, cv=5, scoring="r2", n_jobs=-1)
            gs.fit(X_train, y_train)
            models_info["Fine-tuned RF"] = gs.best_estimator_
        progress.progress(100)
        st.success("Evaluation complete!")
 
        rows = []
        for name, m in models_info.items():
            metrics = get_metrics(m, X_test, y_test)
            metrics["Model"] = name
            rows.append(metrics)
        eval_df = pd.DataFrame(rows)[["Model", "R²", "MAE", "RMSE", "MSE"]]
 
        st.markdown("### Metrics Table")
        st.dataframe(
            eval_df.style
                .format({"R²": "{:.4f}", "MAE": "{:.4f}", "RMSE": "{:.4f}", "MSE": "{:.4f}"})
                .highlight_max(
                    subset=["R²"],
                    props="background-color: #0E1117; color: white;"
                )
                .highlight_min(
                    subset=["MAE", "RMSE", "MSE"],
                    props="background-color: #0E1117; color: white;"
                ),
            use_container_width=True)
 
        # Actual and Predicted
        colors = ["#c0392b", "#2980b9", "#27ae60"]
        st.markdown("### Actual and Predicted (log scale)")
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, (name, m), c in zip(axes, models_info.items(), colors):
            y_pred = m.predict(X_test)
            ax.scatter(y_test, y_pred, alpha=0.4, color=c, s=15)
            lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
            ax.plot(lims, lims, "k--", linewidth=1)
            ax.set_title(name)
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Click the button to evaluate all three models.")

# Prediction
elif page == "Prediction":
    st.title("Insurance Cost Prediction")

    @st.cache_resource
    def load_or_train_model():
        path = f"{models_dir}/model.pkl"
        if os.path.exists(path):
            return joblib.load(path)
        
        df = load_data()
        X_train, X_test, y_train, y_test = split_data(df)
        pipeline = build_pipeline(X_train, RandomForestRegressor(random_state=42))
        pipeline.fit(X_train, y_train)
        os.makedirs(models_dir, exist_ok=True)
        joblib.dump(pipeline, path)
        return pipeline

    model = load_or_train_model()
    st.success("Model is ready!")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 1, 100, 30)
        sex = st.selectbox("Sex", ["male", "female"])
        bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
    with col2:
        children = st.number_input("Number of Children", 0, 10, 0)
        smoker = st.selectbox("Smoker?", ["no", "yes"])
        region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

    if st.button("Predict Insurance Cost"):
        input_df = pd.DataFrame({"age": [age], "sex": [sex], "bmi": [bmi],
                                 "children": [children], "smoker": [smoker], "region": [region]})
        log_pred = model.predict(input_df)
        cost = np.expm1(log_pred)[0]
        st.markdown(f"""
        <div style="background:#0f1923;border-radius:12px;padding:2rem;text-align:center;margin-top:1rem;">
            <p style="color:#aaa;font-size:0.9rem;letter-spacing:2px;text-transform:uppercase;margin:0">Estimated Annual Cost</p>
            <p style="color:#f0c96b;font-size:3rem;font-weight:700;margin:0.5rem 0">${cost:,.2f}</p>
        </div>
        """, unsafe_allow_html=True)