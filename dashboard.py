import streamlit as st
import pandas as pd
import plotly.express as px

# Page Config
st.set_page_config(
    page_title="Student Performance Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load datasets
# Original dataset for Home page charts
df = pd.read_csv("student-scores.csv")
df.columns = [col.strip().lower() for col in df.columns]

# Enhanced dataset for engineered features
df_enhanced = pd.read_csv("student-scores-enhanced.csv")
df_enhanced.columns = [col.strip().lower() for col in df_enhanced.columns]

# Merge engineered features into df
engineered_features = ['engagement_score', 'performance_consistency']
for feature in engineered_features:
    if feature in df_enhanced.columns:
        df[feature] = df_enhanced[feature]

# Initialize session state

if "page" not in st.session_state:
    st.session_state.page = "Home"

# Sidebar Styles
st.sidebar.markdown("""
<style>
/* Style all Streamlit sidebar buttons */
.stButton > button {
    background-color: #222938;
    color: white;
    border-radius: 12px;
    border: 1px solid #333;
    width: 200px;       
    height: 60px;       
    text-align: left;
    padding-left: 12px;
    margin-bottom: 5px;
    font-size: 16px;
    font-weight: 600;
    display: flex;
    align-items: center;
    transition: 0.2s ease;
}
.stButton > button:hover {
    transform: scale(1.03);
    background-color: #2e3648;
    cursor: pointer;
}

/* Selected page style */
.stButton > button.selected {
    background-color: #EB9DA7 !important;
    color: black !important;
    border: 1px solid #c77d86;
}

/* Icon inside button */
.stButton > button span.icon {
    font-size: 20px;
    margin-right: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------- Sidebar Navigation ----------
st.sidebar.markdown("""
<h2 style='color:#ffffff; text-align:center; margin-bottom:10px; margin-top:10px;'>Dashboard Navigation</h2>
<hr style='margin-top:5px; margin-bottom:15px;'>  <!-- increased bottom margin -->
""", unsafe_allow_html=True)


# ---------- Function to create sidebar card ----------
def sidebar_card(label, page_name, icon):
    if st.session_state.page == page_name:
        # Add a 'selected' class for CSS
        button = st.sidebar.button(f"{icon} {label}", key=page_name)
        # Apply class by JS hack (works in Streamlit)
        st.sidebar.markdown(f"""
            <script>
            const btn = window.parent.document.querySelectorAll('button[kind="secondary"]')[{list(st.session_state.keys()).index(page_name)}];
            if(btn){{btn.classList.add('selected')}}
            </script>
        """, unsafe_allow_html=True)
    else:
        button = st.sidebar.button(f"{icon} {label}", key=page_name)

    if button:
        st.session_state.page = page_name


# ---------- Cards ----------
sidebar_card("Home", "Home", "🏠")
sidebar_card("Data Overview", "Data Overview", "📊")
sidebar_card("Visualizations", "Visualizations", "📈")
sidebar_card("Predictions", "Predictions", "🔮")



# Mini-card CSS with icons
card_style = """
<style>
.metric-card {
    background: #222938;
    color: white;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
    margin-bottom: 10px;
    transition: transform 0.2s;
}
.metric-card:hover {
    transform: scale(1.05);
    cursor: pointer;
}
.metric-title {
    font-size: 16px;
    font-weight: bold;
    margin-bottom: 5px;
}
.metric-value {
    font-size: 28px;
    font-weight: bold;
}
.metric-icon {
    font-size: 24px;
    margin-bottom: 5px;
}
</style>
"""

st.markdown(card_style, unsafe_allow_html=True)

# Home Page
if st.session_state.page == "Home":
    st.title("Welcome to the Student Performance Dashboard!")
    st.write("This page provides general insights into the student population.")
    st.write("### Key Insights")

    # Metrics for mini-cards
    total_students = df.shape[0]
    avg_absence = df['absence_days'].mean() if 'absence_days' in df.columns else 0
    avg_self_study = df['weekly_self_study_hours'].mean() if 'weekly_self_study_hours' in df.columns else 0

    # Display metrics in mini-cards with icons and tooltips
    col1, col2, col3 = st.columns(3)

    col1.markdown(f"""
    <div class="metric-card" title="Total number of students in the dataset">
        <div class="metric-icon">👨‍🎓</div>
        <div class="metric-title">Total Students</div>
        <div class="metric-value">{total_students}</div>
    </div>
    """, unsafe_allow_html=True)

    col2.markdown(f"""
    <div class="metric-card" title="Average number of absence days per student">
        <div class="metric-icon">📅</div>
        <div class="metric-title">Average Absence Days</div>
        <div class="metric-value">{round(avg_absence, 2)}</div>
    </div>
    """, unsafe_allow_html=True)

    col3.markdown(f"""
    <div class="metric-card" title="Average weekly self-study hours per student">
        <div class="metric-icon">📚</div>
        <div class="metric-title">Avg Weekly Self-Study Hours</div>
        <div class="metric-value">{round(avg_self_study, 2)}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Pie charts and bar charts
    # Gender Distribution
    if 'gender' in df.columns:
        gender_counts = df['gender'].fillna('Unknown').value_counts().reset_index()
        gender_counts.columns = ['gender', 'count']
        fig_gender = px.pie(
            gender_counts, names='gender', values='count',
            color='gender', color_discrete_map={'male':'#3498db', 'female':'#e74c3c', 'Unknown':'#95a5a6'},
            title="Gender Distribution"
        )
    else:
        fig_gender = None

    # Part-time Job
    if 'part_time_job' in df.columns:
        job_counts = df['part_time_job'].value_counts().reset_index()
        job_counts.columns = ['part_time_job', 'count']
        fig_job = px.pie(
            job_counts, names='part_time_job', values='count',
            color='part_time_job', color_discrete_sequence=px.colors.qualitative.Set2,
            title="Part-Time Job Participation"
        )
    else:
        fig_job = None

    # Extracurricular Activities
    if 'extracurricular_activities' in df.columns:
        extra_counts = df['extracurricular_activities'].value_counts().reset_index()
        extra_counts.columns = ['extracurricular', 'count']
        fig_extra = px.pie(
            extra_counts, names='extracurricular', values='count',
            color='extracurricular', color_discrete_sequence=px.colors.qualitative.Pastel,
            title="Extracurricular Activities"
        )
    else:
        fig_extra = None

    # Average Scores
    subjects = ['math_score', 'history_score', 'physics_score', 'chemistry_score',
                'biology_score', 'english_score', 'geography_score']
    existing_subjects = [s for s in subjects if s in df.columns]
    if existing_subjects:
        avg_scores = df[existing_subjects].mean().reset_index()
        avg_scores.columns = ['subject', 'avg_score']
        fig_scores = px.bar(
            avg_scores, x='subject', y='avg_score', color='subject',
            title="Average Scores by Subject", text='avg_score'
        )
    else:
        fig_scores = None

    # Career Aspirations
    if 'career_aspiration' in df.columns:
        top_careers = df['career_aspiration'].value_counts().nlargest(5).reset_index()
        top_careers.columns = ['career', 'count']
        fig_career = px.bar(
            top_careers, x='career', y='count', color='career',
            title="Top 5 Career Aspirations"
        )
    else:
        fig_career = None

    # Display charts
    col1, col2, col3 = st.columns(3)
    if fig_gender:
        col1.plotly_chart(fig_gender, use_container_width=True)
    if fig_job:
        col2.plotly_chart(fig_job, use_container_width=True)
    if fig_extra:
        col3.plotly_chart(fig_extra, use_container_width=True)

    if fig_scores:
        st.plotly_chart(fig_scores, use_container_width=True)

    if fig_career:
        st.plotly_chart(fig_career, use_container_width=True)

# Data Overview Page
elif st.session_state.page == "Data Overview":
    st.header("📊 Data Overview")
    st.write("This section provides a summary and insights of the dataset.")

    df_overview = df.drop(columns=['first_name', 'last_name', 'email'], errors='ignore')

    # Basic Metrics
    total_students = df_overview.shape[0]
    total_features = df_overview.shape[1]
    missing_values_count = df_overview.isnull().sum().sum()
    numeric_columns = df_overview.select_dtypes(include=['float64', 'int64']).shape[1]
    categorical_columns = df_overview.select_dtypes(include=['object']).shape[1]
    duplicate_rows = df_overview.duplicated().sum()

    col1, col2, col3, col4 = st.columns(4)

    col1.markdown(f"""
    <div class="metric-card" title="Total number of students in the dataset">
        <div class="metric-icon">👨‍🎓</div>
        <div class="metric-title">Total Students</div>
        <div class="metric-value">{total_students}</div>
    </div>
    """, unsafe_allow_html=True)

    col2.markdown(f"""
    <div class="metric-card" title="Total number of features/columns in the dataset">
        <div class="metric-icon">📊</div>
        <div class="metric-title">Total Features</div>
        <div class="metric-value">{total_features}</div>
    </div>
    """, unsafe_allow_html=True)

    col3.markdown(f"""
    <div class="metric-card" title="Total numeric columns in the dataset">
        <div class="metric-icon">🔢</div>
        <div class="metric-title">Numeric Columns</div>
        <div class="metric-value">{numeric_columns}</div>
    </div>
    """, unsafe_allow_html=True)

    col4.markdown(f"""
    <div class="metric-card" title="Total categorical columns in the dataset">
        <div class="metric-icon">🗂️</div>
        <div class="metric-title">Categorical Columns</div>
        <div class="metric-value">{categorical_columns}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Dataset Health Summary
    st.subheader("Dataset Summary")
    st.markdown(f"""
    - **Total Students (Rows):** {total_students}  
    - **Total Features (Columns):** {total_features}  
    - **Missing Values:** {missing_values_count}  
    - **Duplicate Rows:** {duplicate_rows}  
    """)

    st.markdown("---")

    # Columns and Data Types (Original Features)
    st.subheader("Columns and Data Types (Original Features)")
    dtype_df = pd.DataFrame(df_overview.dtypes, columns=['Data Type']).reset_index()
    dtype_df.rename(columns={'index': 'Column'}, inplace=True)
    st.dataframe(dtype_df)

    st.markdown("---")

    # Columns with Missing Values
    missing = df_overview.isnull().sum()
    missing = missing[missing > 0]
    st.subheader("Columns with Missing Values")
    if not missing.empty:
        st.dataframe(missing)
    else:
        st.write("There are no missing values in the dataset.")

    st.markdown("---")

    # Summary Statistics
    st.subheader("Summary Statistics (Numeric Columns)")
    st.dataframe(df_overview.describe().T)

    st.markdown("---")

    # Performance Category Distribution
    if 'performance_category' in df.columns:
        st.subheader("Performance Category Distribution")
        perf_counts = df['performance_category'].value_counts().reset_index()
        perf_counts.columns = ['Category', 'Count']
        fig_perf = px.pie(
            perf_counts,
            names='Category',
            values='Count',
            title="Distribution of Student Performance Categories",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_perf, use_container_width=True)

    st.markdown("---")

    # Correlation Heatmap for Numeric Features
    st.subheader("Correlation Heatmap (Numeric Features)")
    numeric_df = df_overview.select_dtypes(include=['float64', 'int64'])
    if numeric_df.shape[1] > 1:
        corr = numeric_df.corr()
        fig_corr = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale='Viridis',
            aspect="auto",
            title="Correlation Heatmap of Numeric Features"
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    # Distributions of Numeric Features
    st.subheader("Distributions of Numeric Features")
    for col in numeric_df.columns:
        fig = px.histogram(
            df_overview,
            x=col,
            nbins=20,
            title=f"Distribution of {col}",
            marginal="box"  # optional: shows mini box plot on top
        )
        st.plotly_chart(fig, use_container_width=True)

    # Engineered Features
    existing_engineered = [f for f in engineered_features if f in df.columns]

    if existing_engineered:
        st.subheader("Engineered Features Overview")
        total_engineered = len(existing_engineered)
        engineered_missing = df[existing_engineered].isnull().sum().sum()
        numeric_engineered = df[existing_engineered].select_dtypes(include=['int64', 'float64']).shape[1]
        categorical_engineered = df[existing_engineered].select_dtypes(include=['object']).shape[1]

        col1, col2, col3, col4 = st.columns(4)

        col1.markdown(f"""
        <div class="metric-card" title="Number of engineered features">
            <div class="metric-icon">🛠️</div>
            <div class="metric-title">Engineered Features</div>
            <div class="metric-value">{total_engineered}</div>
        </div>
        """, unsafe_allow_html=True)

        col2.markdown(f"""
        <div class="metric-card" title="Number of missing values in engineered features">
            <div class="metric-icon">⚠️</div>
            <div class="metric-title">Missing Values</div>
            <div class="metric-value">{engineered_missing}</div>
        </div>
        """, unsafe_allow_html=True)

        col3.markdown(f"""
        <div class="metric-card" title="Number of numeric engineered features">
            <div class="metric-icon">🔢</div>
            <div class="metric-title">Numeric Columns</div>
            <div class="metric-value">{numeric_engineered}</div>
        </div>
        """, unsafe_allow_html=True)

        col4.markdown(f"""
        <div class="metric-card" title="Number of categorical engineered features">
            <div class="metric-icon">🗂️</div>
            <div class="metric-title">Categorical Columns</div>
            <div class="metric-value">{categorical_engineered}</div>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("Sample Data (Engineered Features)")
        st.dataframe(df[existing_engineered].head(10))


# Visualizations Page
elif st.session_state.page == "Visualizations":
    st.header("📈 Visualizations")
    st.write("Insights from selected and engineered features, plus predictive insights from the trained SVM model.")

    # Load selected features dataset
    df_viz = pd.read_csv("student_scores_selected_features.csv")
    df_viz.columns = [col.strip().lower() for col in df_viz.columns]

    # Cleaning 
    # Strip string columns
    str_cols = df_viz.select_dtypes(include='object').columns
    for col in str_cols:
        df_viz[col] = df_viz[col].astype(str).str.strip()

    # Fix corrupted performance_category entries
    if 'performance_category' in df_viz.columns:
        df_viz['performance_category'] = df_viz['performance_category'].replace(
            {
                'Average cgeck for me': 'Average',
                '': 'Unknown'
            }
        )

    # Ensure numeric columns are numeric
    numeric_cols = [
        'absence_days','weekly_self_study_hours','math_score','history_score',
        'physics_score','chemistry_score','biology_score','english_score','geography_score',
        'science_avg','humanities_avg','engagement_score','performance_consistency','average_score'
    ]
    for col in numeric_cols:
        if col in df_viz.columns:
            df_viz[col] = pd.to_numeric(df_viz[col], errors='coerce')

    # Drop rows with missing critical numeric values for plotting
    df_viz = df_viz.dropna(subset=['weekly_self_study_hours','average_score'])

    # 1. Study vs Performance using Scatter Plot
    st.subheader("Weekly Self-Study Hours vs Average Score vs Engagement")
    if 'weekly_self_study_hours' in df_viz.columns and 'average_score' in df_viz.columns:
        size_col = df_viz['engagement_score'].clip(lower=0) if 'engagement_score' in df_viz.columns else None
        fig_study = px.scatter(
            df_viz,
            x='weekly_self_study_hours',
            y='average_score',
            size=size_col,
            color='performance_category' if 'performance_category' in df_viz.columns else None,
            hover_data=['absence_days', 'performance_consistency'],
            title="Weekly Self-Study Hours vs Average Score vs Engagement"
        )
        st.plotly_chart(fig_study, use_container_width=True)


    # 2. Average Score by Category 
    st.subheader("Average Score by Performance Category")
    if 'performance_category' in df_viz.columns and 'average_score' in df_viz.columns:
        agg = df_viz.groupby('performance_category')['average_score'].agg(['mean', 'std']).reset_index()
        fig_bar = px.bar(
            agg,
            x='performance_category',
            y='mean',
            error_y='std',
            color='performance_category',
            title="Average Score ± Std by Performance Category"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # 3. Simplified Correlation Heatmap
    st.subheader("Correlation Heatmap of Core Scores")
    core_scores = [c for c in [
        'average_score','math_score','history_score','physics_score',
        'chemistry_score','biology_score','english_score','geography_score'
    ] if c in df_viz.columns]
    if len(core_scores) > 1:
        corr = df_viz[core_scores].corr()
        fig_corr = px.imshow(
            corr, text_auto=True,
            color_continuous_scale='Viridis',
            aspect="auto",
            title="Correlation Heatmap of Core Scores"
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    # 4. Distribution of High Scorers
    st.subheader("Distribution of High Scorers")
    high_scorers = df_viz[df_viz['average_score'] >= df_viz['average_score'].mean()]
    if not high_scorers.empty:
        fig_high = px.histogram(
            high_scorers,
            x='average_score',
            nbins=10,
            title="Distribution of Students Above Average Score",
            color='performance_category' if 'performance_category' in df_viz.columns else None
        )
        st.plotly_chart(fig_high, use_container_width=True)

    # 5. Career Aspirations
    st.subheader("Career Aspirations Distribution")
    career_cols = [c for c in df_viz.columns if c.startswith('career_aspiration_')]
    if career_cols:
        career_counts = df_viz[career_cols].sum().reset_index()
        career_counts.columns = ['career', 'count']
        fig_career = px.bar(career_counts, x='career', y='count', color='career', title="Career Aspirations")
        st.plotly_chart(fig_career, use_container_width=True)

    # 6. Performance Category Pie
    st.subheader("Performance Category Distribution")
    if 'performance_category' in df_viz.columns:
        perf_counts = df_viz['performance_category'].value_counts().reset_index()
        perf_counts.columns = ['category', 'count']
        fig_perf = px.pie(perf_counts, names='category', values='count', title="Performance Category Distribution")
        st.plotly_chart(fig_perf, use_container_width=True)

    # 7. Feature Importance based on SVM Model
    import joblib
    import numpy as np
    from sklearn.inspection import permutation_importance

    st.subheader("Feature Importance from Trained SVM Model")

    try:
        # Load model bundle
        model_bundle = joblib.load("svm_model.pkl")

        if isinstance(model_bundle, tuple) and len(model_bundle) == 4:
            svm, scaler, features, label_encoder = model_bundle
        else:
            # If you saved them separately
            svm = joblib.load("svm_model.pkl")
            scaler = joblib.load("scaler.pkl")
            features = joblib.load("svm_features.pkl")
            label_encoder = None

        # Copy visualization data
        X_viz = df_viz.copy()

        # Add any missing columns as zeros to prevent "not in index" errors
        for col in features:
            if col not in X_viz.columns:
                X_viz[col] = 0

        # Reorder to match model training order
        X_viz = X_viz[features]

        # Scale features
        X_scaled = scaler.transform(X_viz)

        # Create dummy target 
        y_dummy = df_viz['performance_category'].astype('category').cat.codes

        # Compute permutation-based importance
        result = permutation_importance(svm, X_scaled, y_dummy, n_repeats=5, random_state=42, n_jobs=-1)
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': result.importances_mean
        }).sort_values(by='Importance', ascending=False)

        # Plot top 3 features
        fig_imp = px.bar(
            importance_df.head(3),
            x='Importance',
            y='Feature',
            orientation='h',
            title="Top 3 Feature Importances (SVM Model)"
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    except Exception as e:
        st.info(f"Feature importance will appear once the SVM model is saved and available. Error: {e}")

    # 8. Filter by Category and Display Data
    st.subheader("Explore Students by Category")
    if 'performance_category' in df_viz.columns:
        category_filter = st.selectbox("Select Performance Category", df_viz['performance_category'].unique())
        filtered = df_viz[df_viz['performance_category'] == category_filter]
        display_cols = ['weekly_self_study_hours','absence_days','engagement_score','performance_consistency','average_score']
        filtered_cols = [c for c in display_cols if c in filtered.columns]
        st.dataframe(filtered[filtered_cols])




elif st.session_state.page == "Predictions":
    st.header("🔮 Predictions Dashboard")
    st.write("Predict student performance or at-risk status using your trained models.")

    import joblib
    import numpy as np
    import pandas as pd
    import plotly.express as px

     # Mapping numeric performance labels to categories
    performance_mapping = {
        0: "Low",
        1: "Average",
        2: "High",
        3: "Excellent"
    }

    # Tabs for model selection
    tab1, tab2 = st.tabs(["🎯 SVM Classification", "⚠️ XGBoost At-Risk"])

    # ================= SVM Performance Classification =================
    with tab1:
        st.subheader("🎯 Student Performance Classification (SVM)")

        try:
            model_bundle = joblib.load("svm_model.pkl")
            if isinstance(model_bundle, tuple):
                svm_model, scaler, features, label_encoder = model_bundle
            else:
                svm_model = joblib.load("svm_model.pkl")
                scaler = joblib.load("scaler.pkl")
                features = joblib.load("svm_features.pkl")
                label_encoder = None
        except Exception as e:
            st.error(f"Could not load SVM model: {e}")
            svm_model, scaler, features, label_encoder = None, None, None, None

        if svm_model:
            mode = st.radio("Choose input mode:", ["Manual Input", "Upload CSV"], horizontal=True, key="svm_mode")

            # ---------------- Manual Input ----------------
            if mode == "Manual Input":
                st.write("Enter student details below:")
                input_features = [f for f in features if "career_aspiration" not in f.lower()]
                feature_limits = {
                    "average_score": (0, 100),
                    "science_avg": (0, 100),
                    "humanities_avg": (0, 100),
                    "performance_consistency": (0, 100),
                    "weekly_self_study_hours": (0, 168),
                    "engagement_score": (0, 100),
                    "absence_days": (0, 365),
                }

                user_input = {}
                for feat in input_features:
                    min_val, max_val = feature_limits.get(feat, (0, 100))
                    user_input[feat] = st.number_input(
                        f"{feat.replace('_',' ').title()}",
                        value=min_val, min_value=min_val, max_value=max_val,
                        key=f"svm_manual_{feat}"
                    )

                if st.button("Predict Performance Category", key="svm_manual_predict"):
                    X_input = pd.DataFrame([user_input])
                    for feat in features:
                        if feat not in X_input.columns:
                            X_input[feat] = 0
                    X_input = X_input[features]
                    X_scaled = scaler.transform(X_input)
                    pred = svm_model.predict(X_scaled)
                    try:
                        pred_label = label_encoder.inverse_transform(pred)[0]
                    except:
                        pred_label = int(pred[0])

                    pred_label = performance_mapping.get(pred_label, str(pred_label))
                    st.success(f"Predicted Performance Category: **{pred_label}**")

            # ---------------- CSV Upload ----------------
            elif mode == "Upload CSV":
                uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"], key="svm_file")
                st.info("📌 Please upload a CSV that contains ALL required model features.")
                
                st.download_button(
                    "📄 Download Template CSV (Correct Columns)",
                    open("student_scores_selected_features.csv", "rb").read(),
                    "template_student_features.csv",
                    "text/csv",
                    key="svm_template_download"
                )

                if uploaded_file:
                    df_input = pd.read_csv(uploaded_file)
                    st.success("File uploaded successfully!")
                    st.dataframe(df_input.head())

                    column_mapping = {
                        "avg_score": "average_score",
                        "science_average": "science_avg",
                        "humanities_average": "humanities_avg",
                        "consistency": "performance_consistency",
                        "career_BusinessOwner": "career_aspiration_Business Owner",
                        "career_Doctor": "career_aspiration_Doctor",
                    }
                    df_input.rename(columns={k: v for k, v in column_mapping.items() if k in df_input.columns}, inplace=True)

                    overlap = [col for col in features if col in df_input.columns]
                    if len(overlap) == 0:
                        st.error("❌ Uploaded CSV does not contain recognized model features.")
                        st.stop()

                    for col in features:
                        if col not in df_input.columns:
                            df_input[col] = 0

                    X_scaled = scaler.transform(df_input[features])
                    preds = svm_model.predict(X_scaled)
                    try:
                        pred_labels = label_encoder.inverse_transform(preds)
                    except:
                        pred_labels = [str(p) for p in preds]
                    # If label_encoder exists, decode first
                    try:
                        pred_labels = label_encoder.inverse_transform(preds)
                    except:
                        pred_labels = preds  # numeric labels if no encoder

                    # Map numeric labels to categories
                    df_input["Predicted_Performance"] = [performance_mapping.get(int(p), str(p)) for p in pred_labels]


                    # ---------- Visualizations ----------
                    fig_count = px.histogram(df_input, x="Predicted_Performance", color="Predicted_Performance",
                                             title="Distribution of Predicted Performance Categories", text_auto=True)
                    st.plotly_chart(fig_count, use_container_width=True, key="svm_fig_count")

                    if len(features) > 1:
                        corr_df = df_input[features].corr()
                        fig_corr = px.imshow(corr_df, text_auto=True, color_continuous_scale='Viridis',
                                             title="Feature Correlation Heatmap")
                        st.plotly_chart(fig_corr, use_container_width=True, key="svm_fig_corr")

                    num_feats = [f for f in features if df_input[f].dtype in [np.float64, np.int64]]
                    if len(num_feats) >= 2:
                        fig_scatter = px.scatter(df_input, x=num_feats[0], y=num_feats[1], color="Predicted_Performance",
                                                 title=f"{num_feats[0]} vs {num_feats[1]} by Predicted Performance",
                                                 hover_data=features)
                        st.plotly_chart(fig_scatter, use_container_width=True, key="svm_fig_scatter")

                    st.success("✔ Predictions completed!")
                    st.dataframe(df_input.head())

                    csv = df_input.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Download Predictions",
                        data=csv,
                        file_name="svm_predictions.csv",
                        mime="text/csv",
                        key="svm_download"
                    )

    # ================= XGBoost At-Risk Prediction =================
    with tab2:
        st.subheader("⚠️ Predict At-Risk Students (XGBoost)")

        try:
            xgb_model = joblib.load("xgb_model.pkl")
            xgb_scaler = joblib.load("xgb_scaler.pkl")
            xgb_features = joblib.load("xgb_features.pkl")
        except Exception as e:
            st.error(f"Could not load XGBoost model: {e}")
            xgb_model, xgb_scaler, xgb_features = None, None, None

        if xgb_model:
            mode = st.radio("Choose input mode:", ["Manual Input", "Upload CSV"], horizontal=True, key="xgb_mode")

            # Manual Input
            if mode == "Manual Input":
                st.write("Enter student behavioral details:")
                behavioral_limits = {
                    "weekly_self_study_hours": (0, 168),
                    "engagement_score": (0, 100),
                    "performance_consistency": (0, 100),
                    "science_avg": (0, 100),
                    "humanities_avg": (0, 100),
                }
                xgb_input = {}
                for i, f in enumerate(xgb_features):
                    min_val, max_val = behavioral_limits.get(f, (0, 100))
                    xgb_input[f] = st.number_input(f"{f.replace('_',' ').title()}",
                                                   value=min_val, min_value=min_val, max_value=max_val,
                                                   key=f"xgb_manual_{f}")

                if st.button("Predict At-Risk Status", key="xgb_manual_predict"):
                    X_input = pd.DataFrame([xgb_input])
                    X_scaled = xgb_scaler.transform(X_input)
                    pred = xgb_model.predict(X_scaled)[0]
                    pred_proba = xgb_model.predict_proba(X_scaled)[0][1]
                    status = "At Risk" if pred == 1 else "Not At Risk"
                    st.success(f"Prediction: **{status}** ({pred_proba:.2%} probability)")

            # CSV Upload
            elif mode == "Upload CSV":
                uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"], key="xgb_file_upload")
                st.info("📌 Please upload a CSV that contains ALL required model features.")

                st.download_button(
                    "📄 Download Template CSV (Correct Columns)",
                    open("student_scores_selected_features.csv", "rb").read(),
                    "template_student_features.csv",
                    "text/csv",
                    key="xgb_template_download"
                )

                if uploaded_file:
                    user_df = pd.read_csv(uploaded_file)
                    st.success("File uploaded successfully!")
                    st.dataframe(user_df.head())

                    column_mapping = {
                        "avg_score": "average_score",
                        "science_average": "science_avg",
                        "humanities_average": "humanities_avg",
                        "consistency": "performance_consistency",
                        "career_BusinessOwner": "career_aspiration_Business Owner",
                        "career_Doctor": "career_aspiration_Doctor",
                    }
                    user_df.rename(columns={k: v for k, v in column_mapping.items() if k in user_df.columns}, inplace=True)

                    overlap = [col for col in xgb_features if col in user_df.columns]
                    if len(overlap) == 0:
                        st.error("❌ Uploaded CSV does not contain recognized features.")
                        st.stop()

                    for col in xgb_features:
                        if col not in user_df.columns:
                            user_df[col] = 0

                    X_input = user_df[xgb_features]
                    X_scaled = xgb_scaler.transform(X_input)
                    preds = xgb_model.predict(X_scaled)
                    pred_probas = xgb_model.predict_proba(X_scaled)[:, 1]
                    user_df['Predicted_Status'] = np.where(preds == 1, "At Risk", "Not At Risk")
                    user_df['At-Risk Probability'] = pred_probas

                    st.success("✔ Prediction complete!")
                    st.dataframe(user_df[['Predicted_Status', 'At-Risk Probability'] + xgb_features].head())

                    # ---------- Visualizations ----------
                    fig_count = px.histogram(user_df, x="Predicted_Status", color="Predicted_Status",
                                             title="Distribution of Predicted At-Risk Status", text_auto=True)
                    st.plotly_chart(fig_count, use_container_width=True, key="xgb_fig_count")

                    fig_prob = px.histogram(user_df, x="At-Risk Probability", nbins=20,
                                            title="Distribution of At-Risk Probabilities", color_discrete_sequence=["crimson"])
                    st.plotly_chart(fig_prob, use_container_width=True, key="xgb_fig_prob")

                    num_feats = [f for f in xgb_features if user_df[f].dtype in [np.float64, np.int64]]
                    if len(num_feats) >= 2:
                        fig_scatter = px.scatter(user_df, x=num_feats[0], y=num_feats[1], color="At-Risk Probability",
                                                 color_continuous_scale='RdYlGn_r',
                                                 title=f"{num_feats[0]} vs {num_feats[1]} by At-Risk Probability",
                                                 hover_data=xgb_features)
                        st.plotly_chart(fig_scatter, use_container_width=True, key="xgb_fig_scatter")

                    csv = user_df.to_csv(index=False).encode('utf-8')
                    st.download_button("⬇️ Download Predictions", csv,
                                       "xgb_at_risk_predictions.csv", "text/csv", key="xgb_download")


