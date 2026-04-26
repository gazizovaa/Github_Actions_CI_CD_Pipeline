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
        ["Dataset Overview", "Visualizations", "Data Splitting", "Model Training", "Fine-Tuning",
         "Evaluation", "Save Models", "Prediction"],
        label_visibility="collapsed"
    )

# Dataset Overview 
if page == "Dataset Overview":
    st.markdown('<span class="section-tag">Step 01</span>', unsafe_allow_html=True)
    st.title("Dataset Overview")
    try:
        df = load_data()
    except FileNotFoundError:
        st.error(f"Dataset not found at `{data_path}`. Please place `medical-charges.csv` in the `data/` folder.")
        st.stop()
        
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><h4>Total Rows</h4><p>{df.shape[0]:,}</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><h4>Features</h4><p>{df.shape[1]}</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><h4>Duplicates Removed</h4><p>{df.duplicated().sum()}</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><h4>Missing Values</h4><p>{df.isnull().sum().sum()}</p></div>', unsafe_allow_html=True)
 
    st.markdown("### Raw Data Sample")
    st.dataframe(df.head(10), use_container_width=True)
    st.markdown("### Statistical Summary")
    st.dataframe(df.describe().T.style.format("{:.2f}"), use_container_width=True)
    st.markdown("### Column Types")
    dtype_df = pd.DataFrame({"Column": df.columns, "Type": df.dtypes.astype(str).values, "Non-Null": df.notnull().sum().values})
    st.dataframe(dtype_df, use_container_width=True)
    
# Visualizations
elif page == "Visualizations":
    st.markdown('<span class="section-tag">Step 02</span>', unsafe_allow_html=True)
    st.title("Visualizations & Correlation")
    df = load_data()
    sns.set_style("whitegrid")
    palette = {"yes": "#c0392b", "no": "#2980b9"}
    tab1, tab2, tab3, tab4 = st.tabs(["Charges Distribution", "BMI vs Charges", "Age vs Charges", "Correlation Matrix"])
 
    with tab1:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        sns.histplot(df["charges"], bins=35, color="#c0392b", kde=True, ax=axes[0])
        axes[0].set_title("Distribution of Charges")
        sns.histplot(np.log1p(df["charges"]), bins=35, color="#2980b9", kde=True, ax=axes[1])
        axes[1].set_title("Distribution of log(Charges)")
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
 
        st.markdown("### Correlation Table")
        st.dataframe(numeric_df.corr().style.format("{:.3f}").background_gradient(cmap="coolwarm"), use_container_width=True)

# Data Splitting 
elif page == "Data Splitting":
    st.markdown('<span class="section-tag">Step 03</span>', unsafe_allow_html=True)
    st.title("Feature Separation & Train/Test Split")
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    st.markdown("### Target vs Features")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("**Target (y):** `charges` → log-transformed with `np.log1p`")
        st.dataframe(pd.DataFrame({"y_log_train (sample)": y_train.values[:8]}), use_container_width=True)
    with col2:
        st.info(f"**Features (X):** {list(df.drop('charges', axis=1).columns)}")
        st.dataframe(X_train.head(8), use_container_width=True)
 
    st.markdown("### Split Summary")
    split_info = pd.DataFrame({
        "Set": ["X_train", "X_test", "y_train", "y_test"],
        "Shape": [str(X_train.shape), str(X_test.shape), str(y_train.shape), str(y_test.shape)],
        "Rows": [len(X_train), len(X_test), len(y_train), len(y_test)],
        "% of Total": [
            f"{len(X_train)/len(df)*100:.1f}%",
            f"{len(X_test)/len(df)*100:.1f}%",
            f"{len(y_train)/len(df)*100:.1f}%",
            f"{len(y_test)/len(df)*100:.1f}%",
        ]
    })
    st.dataframe(split_info, use_container_width=True)
 
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.pie([len(X_train), len(X_test)], labels=["Train (80%)", "Test (20%)"],
           colors=["#c0392b", "#2980b9"], autopct="%1.1f%%", startangle=90)
    ax.set_title("Train / Test Split")
    st.pyplot(fig)
    
# Model Training
elif page == "Model Training":
    st.markdown('<span class="section-tag">Step 04</span>', unsafe_allow_html=True)
    st.title("Regression Model Training")
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    st.markdown("""
    Two regression models are trained on the preprocessed training set:
    - **Linear Regression** — baseline model
    - **Random Forest Regressor** — ensemble tree model
    
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
        for name, model in [("Linear Regression", lr), ("Random Forest", rf)]:
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
    st.markdown('<span class="section-tag">Step 05</span>', unsafe_allow_html=True)
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
    st.markdown('<span class="section-tag">Step 06</span>', unsafe_allow_html=True)
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
            eval_df.style.format({"R²": "{:.4f}", "MAE": "{:.4f}", "RMSE": "{:.4f}", "MSE": "{:.4f}"})
                   .highlight_max(subset=["R²"], color="#d4edda")
                   .highlight_min(subset=["MAE", "RMSE", "MSE"], color="#d4edda"),
            use_container_width=True
        )
 
        # Save report
        os.makedirs("reports", exist_ok=True)
        with open("reports/evaluation_report.txt", "w") as f:
            f.write("=== Model Evaluation Report ===\n\n")
            for _, row in eval_df.iterrows():
                f.write(f"{row['Model']}: R²={row['R²']:.4f}, MAE={row['MAE']:.4f}, RMSE={row['RMSE']:.4f}\n")
        st.success("Report saved to `reports/evaluation_report.txt`")
 
        # Bar chart comparison
        st.markdown("### R² Score Comparison")
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ["#c0392b", "#2980b9", "#27ae60"]
        ax.barh(eval_df["Model"], eval_df["R²"], color=colors)
        ax.set_xlabel("R² Score")
        ax.set_title("Model R² Comparison")
        ax.set_xlim(0, 1)
        for i, v in enumerate(eval_df["R²"]):
            ax.text(v + 0.005, i, f"{v:.4f}", va="center", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)
 
        # Actual vs Predicted
        st.markdown("### Actual vs Predicted (log scale)")
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
        
# Save Models
elif page == "Save Models":
    st.markdown('<span class="section-tag">Step 07</span>', unsafe_allow_html=True)
    st.title("Save & Load Models")
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    if st.button("Train & Save All Models"):
        os.makedirs(models_dir, exist_ok=True)
        saved = []
 
        with st.spinner("Training and saving Linear Regression..."):
            lr = build_pipeline(X_train, LinearRegression())
            lr.fit(X_train, y_train)
            joblib.dump(lr, f"{models_dir}/linear_regression.pkl")
            saved.append(("linear_regression.pkl", get_metrics(lr, X_test, y_test)["R²"]))
        with st.spinner("Training and saving Fine-tuned Random Forest..."):
            pipeline = build_pipeline(X_train, RandomForestRegressor(random_state=42))
            param_grid = {
                "model__n_estimators": [100, 200],
                "model__max_depth": [None, 10, 20],
                "model__min_samples_split": [2, 5],
            }
            gs = GridSearchCV(pipeline, param_grid, cv=5, scoring="r2", n_jobs=-1)
            gs.fit(X_train, y_train)
            best_rf = gs.best_estimator_
            joblib.dump(best_rf, f"{models_dir}/model.pkl")
            saved.append(("model.pkl (fine-tuned RF)", get_metrics(best_rf, X_test, y_test)["R²"]))
        st.success("Models saved to `models/` folder!")
        save_df = pd.DataFrame(saved, columns=["File", "Test R²"])
        save_df["Test R²"] = save_df["Test R²"].map("{:.4f}".format)
        st.dataframe(save_df, use_container_width=True)
 
    st.markdown("### Saved Model Files")
    if os.path.exists(models_dir):
        files = os.listdir(models_dir)
        if files:
            for f in files:
                path = os.path.join(models_dir, f)
                size_kb = os.path.getsize(path) / 1024
                st.markdown(f"- `{f}` — {size_kb:.1f} KB")
        else:
            st.info("No saved models yet. Click the button above.")
    else:
        st.info("No `models/` directory yet.")

# Prediction
elif page == "Prediction":
    st.markdown('<span class="section-tag">Step 08</span>', unsafe_allow_html=True)
    st.title("Insurance Cost Prediction")
 
    @st.cache_resource
    def load_model():
        path = f"{models_dir}/model.pkl"
        if os.path.exists(path):
            return joblib.load(path)
        return None
    model = load_model()
 
    if model is None:
        st.warning("No saved model found. Please go to **Save Models** first.")
    else:
        st.success("Model loaded from `models/model.pkl`")
        col1, col2 = st.columns(2)
        with col1:
            age      = st.number_input("Age", 1, 100, 30)
            sex      = st.selectbox("Sex", ["male", "female"])
            bmi      = st.number_input("BMI", 10.0, 60.0, 25.0)
        with col2:
            children = st.number_input("Number of Children", 0, 10, 0)
            smoker   = st.selectbox("Smoker?", ["no", "yes"])
            region   = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])
 
        if st.button("Predict Insurance Cost"):
            input_df = pd.DataFrame({"age": [age], "sex": [sex], "bmi": [bmi],
                                     "children": [children], "smoker": [smoker], "region": [region]})
            log_pred = model.predict(input_df)
            cost     = np.expm1(log_pred)[0]
            st.markdown(f"""
            <div style="background:#0f1923;border-radius:12px;padding:2rem;text-align:center;margin-top:1rem;">
                <p style="color:#aaa;font-size:0.9rem;letter-spacing:2px;text-transform:uppercase;margin:0">Estimated Annual Cost</p>
                <p style="color:#f0c96b;font-size:3rem;font-weight:700;margin:0.5rem 0">${cost:,.2f}</p>
            </div>
            """, unsafe_allow_html=True)
 