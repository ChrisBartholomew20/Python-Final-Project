import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Global Model Parameters and Constants ---
# *** OPTIMIZED FEATURE MAP (8 RELEVANT FACTORS) ***
FEATURE_MAP = {
    'Numerical': [
        'study_hours', 'class_attendance', 'sleep_hours' 
    ],
    'Categorical': [
        'course', 'exam_difficulty', 'study_method', 'sleep_quality', 'gender'
    ]
}
TARGET_COL = 'exam_score'

# --- GLOBAL ADJUSTMENTS FOR REALISTIC DISTRIBUTION ---
# MODIFICATION: Setting the target mean to 75.0 for a realistic average exam score center.
DESIRED_MEAN = 80.0 
REALISTIC_SPREAD_SD = 1.0 # Final SD for noise, ensuring smooth bell curve (approx 1.5% variance)
# High Score Scaling Factor (1.65) is maintained to ensure max inputs still hit 100
HIGH_SCORE_SCALING_FACTOR = .8
MAX_STUDY_HOURS_BASIC_EXAM_CAP = 10.0 # Input cap for study hours

# --- Global Variables for Model ---
GLOBAL_CATEGORIES = {}
GLOBAL_SCORE_BIAS = 0.0 
model = None
feature_cols = []

# --- 2. Data Processing and Model Setup (Cached) ---

def remove_outliers(df, columns, z_thresh=3):
    """Removes outliers from specified columns using the Z-score method."""
    df_filtered = df.copy()
    for col in columns:
        if col in df_filtered.columns:
            z_scores = (df_filtered[col] - df[col].mean()) / df[col].std()
            df_filtered = df_filtered[np.abs(z_scores) < z_thresh]
    return df_filtered

@st.cache_data
def load_data(file_path):
    """Loads and preprocesses the Exam_Score_Prediction.csv dataset."""
    try:
        df = pd.read_csv(file_path)
        
        selected_cols = FEATURE_MAP['Numerical'] + FEATURE_MAP['Categorical'] + [TARGET_COL]
        valid_selected_cols = [col for col in selected_cols if col in df.columns]
        df = df[valid_selected_cols].copy()
        
        num_cols = FEATURE_MAP['Numerical'] + [TARGET_COL]
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.dropna(subset=num_cols)
        
        for col in FEATURE_MAP['Categorical']:
            if col in df.columns:
                df[col] = df[col].astype(str).fillna('Unknown')
        
        df = remove_outliers(df, num_cols)
        return df

    except FileNotFoundError:
        st.error(f"Data file not found: {file_path}. Please ensure it is in the same directory.")
        st.stop()
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An unexpected error occurred during data processing: {e}")
        st.stop()
        return pd.DataFrame()


@st.cache_resource
def setup_model_and_features():
    """Trains the model and prepares global feature variables using Streamlit caching."""
    global model, feature_cols, GLOBAL_CATEGORIES, GLOBAL_SCORE_BIAS

    df = load_data('Exam_Score_Prediction.csv')

    if df.empty:
        return None

    # Store unique categories for Streamlit selectboxes
    for col in FEATURE_MAP['Categorical']:
        if col in df.columns:
            GLOBAL_CATEGORIES[col] = sorted(df[col].unique().tolist())
        
    # --- Preprocessing (One-Hot Encoding) ---
    df_encoded = pd.get_dummies(df, columns=FEATURE_MAP['Categorical'], prefix=FEATURE_MAP['Categorical'], dummy_na=False)

    X = df_encoded.drop(columns=[TARGET_COL], errors='ignore')
    y = df_encoded[TARGET_COL]
    
    # --- TARGET TRANSFORMATION (Log-Square Root to widen base prediction spread) ---
    y = y.clip(lower=1.0)
    y_transformed = np.log(np.sqrt(y))
    
    feature_cols = X.columns.tolist()

    # --- Model Training (Linear Regression) ---
    X_train, X_test, y_train, y_test = train_test_split(X, y_transformed, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # --- Dynamic Score Centering (Re-implemented for 75.0 target) ---
    all_preds_transformed = model.predict(X)
    
    # Use HIGH_SCORE_SCALING_FACTOR (1.65) before reversing transformation
    scaled_preds_transformed = all_preds_transformed * HIGH_SCORE_SCALING_FACTOR 
    
    # Reverse transformation: (e^transformed_score)^2
    all_preds_real = (np.exp(scaled_preds_transformed)) ** 2
    
    # Calculate the required bias to center the mean at DESIRED_MEAN (75.0)
    predicted_mean = np.mean(all_preds_real)
    GLOBAL_SCORE_BIAS = DESIRED_MEAN - predicted_mean
    
    st.info(f"Model trained. Predictions are centered around the target mean of: {DESIRED_MEAN:.2f}")
    
    return model, feature_cols, GLOBAL_CATEGORIES, GLOBAL_SCORE_BIAS


def predict_score(input_values: dict, model_data) -> float:
    """Calculates the final score incorporating transformation, bias, and minimal noise."""
    
    model, feature_cols, _, GLOBAL_SCORE_BIAS = model_data

    # Ensure required numerical inputs are numeric (Streamlit returns floats, but good practice)
    input_data = pd.DataFrame([{k: v for k, v in input_values.items()}])
    
    # --- ADJUSTMENT FOR AVERAGE EXAM: Cap Hours_Studied ---
    input_data['study_hours'] = input_data['study_hours'].clip(upper=MAX_STUDY_HOURS_BASIC_EXAM_CAP)
    
    # --- Preprocessing (One-Hot Encoding) ---
    input_encoded = pd.get_dummies(input_data, columns=FEATURE_MAP['Categorical'], prefix=FEATURE_MAP['Categorical'], dummy_na=False)

    # 4. Re-index and fill missing columns with 0
    final_input = pd.DataFrame(0, index=[0], columns=feature_cols, dtype=float)
    
    for col in input_encoded.columns:
        if col in final_input.columns:
            final_input.loc[0, col] = float(input_encoded.loc[0, col])

    # 5. Predict the TRANSFORMED base score
    predicted_transformed_score = model.predict(final_input)[0]
    
    # --- Apply High Score Scaling ---
    scaled_transformed_score = predicted_transformed_score * HIGH_SCORE_SCALING_FACTOR
    
    # 6. Reverse the transformation: (e^transformed_score)^2
    clipped_transformed_score = np.clip(scaled_transformed_score, -5.0, None)
    base_score_real = (np.exp(clipped_transformed_score)) ** 2
    
    # 7. Apply global score bias (shifts the mean to 75.0)
    predicted_score = base_score_real + GLOBAL_SCORE_BIAS
    
    # 8. ADD NORMALLY DISTRIBUTED NOISE FOR REALISM 
    noise = np.random.normal(0, REALISTIC_SPREAD_SD)
    predicted_score += noise

    # 9. Ensure score is within a sensible range (0-100)
    return round(max(0, min(100, predicted_score)), 1)

# --- NEW: Recommendation Function ---

def generate_recommendations(current_inputs: dict, goal_score: float, model_data):
    """
    Analyzes which inputs need adjustment to reach the goal score.
    Focuses on controllable numerical factors.
    """
    if goal_score <= 0 or goal_score > 100:
        return ["Please set a valid goal score between 1 and 100."]

    # Unpack model and feature columns
    model, feature_cols, _, global_bias = model_data

    # 1. Calculate the current deterministic score (no noise)
    current_inputs_df = pd.DataFrame([current_inputs])
    current_inputs_df['study_hours'] = current_inputs_df['study_hours'].clip(upper=MAX_STUDY_HOURS_BASIC_EXAM_CAP)
    
    input_encoded = pd.get_dummies(current_inputs_df, columns=FEATURE_MAP['Categorical'], prefix=FEATURE_MAP['Categorical'], dummy_na=False)
    final_input = pd.DataFrame(0, index=[0], columns=feature_cols, dtype=float)
    for col in input_encoded.columns:
        if col in final_input.columns:
            final_input.loc[0, col] = float(input_encoded.loc[0, col])

    predicted_transformed_score = model.predict(final_input)[0]
    scaled_transformed_score = predicted_transformed_score * HIGH_SCORE_SCALING_FACTOR
    clipped_transformed_score = np.clip(scaled_transformed_score, -5.0, None)
    base_score_real = (np.exp(clipped_transformed_score)) ** 2
    current_deterministic_score = base_score_real + global_bias

    recommendations = []

    if current_deterministic_score >= goal_score:
        return [f"Excellent! Your predicted base score is {current_deterministic_score:.1f}. You are already on track to achieve your goal of {goal_score:.1f}."]

    score_needed = goal_score - current_deterministic_score
    recommendations.append(f"🎯 **Goal Target:** Need to increase deterministic score by **{score_needed:.1f} points**.")
    recommendations.append("---")
    
    # 2. Analyze impact of controllable numerical factors (Hours, Attendance, Sleep)
    
    # We will use simple perturbation analysis: change the input by a fixed amount 
    # and see how much the score changes.
    
    # Store changes: {feature: score_boost}
    boosts = {}
    
    # a) Study Hours (Boost by 2 hours)
    if current_inputs['study_hours'] < MAX_STUDY_HOURS_BASIC_EXAM_CAP:
        test_inputs = current_inputs.copy()
        test_inputs['study_hours'] = min(MAX_STUDY_HOURS_BASIC_EXAM_CAP, current_inputs['study_hours'] + 2.0)
        
        # Calculate new deterministic score (requires full prediction loop again)
        test_inputs_df = pd.DataFrame([test_inputs])
        test_inputs_df['study_hours'] = test_inputs_df['study_hours'].clip(upper=MAX_STUDY_HOURS_BASIC_EXAM_CAP)
        input_encoded = pd.get_dummies(test_inputs_df, columns=FEATURE_MAP['Categorical'], prefix=FEATURE_MAP['Categorical'], dummy_na=False)
        final_input_test = pd.DataFrame(0, index=[0], columns=feature_cols, dtype=float)
        for col in input_encoded.columns:
            if col in final_input_test.columns:
                final_input_test.loc[0, col] = float(input_encoded.loc[0, col])
        
        test_transformed_score = model.predict(final_input_test)[0]
        test_scaled_score = test_transformed_score * HIGH_SCORE_SCALING_FACTOR
        test_base_score = (np.exp(np.clip(test_scaled_score, -5.0, None))) ** 2
        score_boost = (test_base_score + global_bias) - current_deterministic_score
        boosts['study_hours'] = score_boost

    # b) Class Attendance (Boost by 5%)
    if current_inputs['class_attendance'] < 100.0:
        test_inputs = current_inputs.copy()
        test_inputs['class_attendance'] = min(100.0, current_inputs['class_attendance'] + 5.0)
        
        test_inputs_df = pd.DataFrame([test_inputs])
        test_inputs_df['study_hours'] = test_inputs_df['study_hours'].clip(upper=MAX_STUDY_HOURS_BASIC_EXAM_CAP)
        input_encoded = pd.get_dummies(test_inputs_df, columns=FEATURE_MAP['Categorical'], prefix=FEATURE_MAP['Categorical'], dummy_na=False)
        final_input_test = pd.DataFrame(0, index=[0], columns=feature_cols, dtype=float)
        for col in input_encoded.columns:
            if col in final_input_test.columns:
                final_input_test.loc[0, col] = float(input_encoded.loc[0, col])
        
        test_transformed_score = model.predict(final_input_test)[0]
        test_scaled_score = test_transformed_score * HIGH_SCORE_SCALING_FACTOR
        test_base_score = (np.exp(np.clip(test_scaled_score, -5.0, None))) ** 2
        score_boost = (test_base_score + global_bias) - current_deterministic_score
        boosts['class_attendance'] = score_boost
        
    # 3. Rank and Format Recommendations
    
    if not boosts:
        return ["You are already maximizing all controllable factors (Hours, Attendance). Focus on improving your categorical factors."]
        
    sorted_boosts = sorted(boosts.items(), key=lambda item: item[1], reverse=True)
    
    # Determine the most effective boost
    most_effective_feature = sorted_boosts[0][0]
    
    recommendations.append(f"The most impactful change is **{most_effective_feature.replace('_', ' ').title()}**.")
    recommendations.append("")
    
    for feature, boost in sorted_boosts:
        if boost > 0.05: # Only recommend changes that provide a noticeable boost
            if feature == 'study_hours':
                rec = f"✅ Increase **Study Hours** by 2 to gain ~{boost:.1f} points (Current: {current_inputs['study_hours']:.1f}h)."
            elif feature == 'class_attendance':
                rec = f"✅ Increase **Attendance** by 5% to gain ~{boost:.1f} points (Current: {current_inputs['class_attendance']:.1f}%)."
            elif feature == 'sleep_hours':
                # Sleep hours is usually a small factor, recommending max 9h
                rec = f"✅ Ensure **Sleep** is optimal (9h) to gain up to ~{boost:.1f} points (Current: {current_inputs['sleep_hours']:.1f}h)."
            else:
                continue # Should not happen with current FEATURE_MAP

            recommendations.append(rec)
            
    # 4. Check for Categorical Improvement Possibilities
    
    # We check if the current categorical factors are set to the best possible value
    # Easiest way is to see if any high-impact categorical factors are set to a low value
    
    low_impact_factors = []
    
    if current_inputs['exam_difficulty'] == 'hard':
        low_impact_factors.append("Try shifting focus to easier courses first or reviewing foundational material (Difficulty: Hard).")
    if current_inputs['study_method'] == 'online videos':
        low_impact_factors.append("Consider switching from 'online videos' to 'self-study' or 'group study' for better retention.")
        
    if low_impact_factors:
        recommendations.append("")
        recommendations.append("💡 **Categorical Tips:**")
        recommendations.extend(low_impact_factors)
        
    return recommendations

# --- 4. Visualization Functions ---

def plot_relationship(df, feature, target):
    """Generates a plot showing the relationship between a single feature and the target score."""
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(6, 4))
    
    if feature in FEATURE_MAP['Numerical']:
        # Scatter plot for numerical features with regression line
        sns.regplot(x=feature, y=target, data=df, ax=ax, scatter_kws={'alpha':0.6}, line_kws={'color': '#4A90E2'})
        ax.set_title(f'Score vs. {feature.replace("_", " ").title()}', fontsize=12)
    
    elif feature in FEATURE_MAP['Categorical']:
        # Box plot for categorical features
        sns.boxplot(x=feature, y=target, data=df, ax=ax, palette="Pastel1")
        ax.set_title(f'Score Distribution by {feature.replace("_", " ").title()}', fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    ax.set_xlabel(feature.replace("_", " ").title(), fontsize=10)
    ax.set_ylabel("Exam Score", fontsize=10)
    plt.tight_layout()
    return fig

def generate_random_input(feature_map, global_categories):
    """Generates a dictionary of random input values for simulation."""
    random_input = {}
    
    numerical_ranges = {
        'study_hours': (1, MAX_STUDY_HOURS_BASIC_EXAM_CAP), 
        'class_attendance': (60, 100),         
        'sleep_hours': (5, 9),          
    }

    for feature in feature_map['Numerical']:
        low, high = numerical_ranges.get(feature, (0, 10))
        random_input[feature] = random.uniform(low, high)

    for feature in feature_map['Categorical']:
        categories = global_categories.get(feature, ["Unknown"])
        random_input[feature] = random.choice(categories)
        
    return random_input

def plot_predicted_distribution(model_data):
    """Simulates predictions and plots the resulting score distribution."""
    st.markdown("---")
    st.header("3. Predicted Score Distribution")
    st.markdown(f"This histogram shows the approximate **bell curve** the model generates based on random, realistic student inputs. (Mean should be near {DESIRED_MEAN:.2f}).")
    
    # Unpack model data
    model, feature_cols, global_categories, _ = model_data

    # 1. Simulate 20,000 predictions
    num_samples = 2000
    predicted_scores = []
    
    for _ in range(num_samples):
        input_data = generate_random_input(FEATURE_MAP, global_categories)
        score = predict_score(input_data, model_data)
        if score != -1.0:
            predicted_scores.append(score)

    if not predicted_scores:
        st.warning("Simulation failed to produce scores.")
        return

    scores_array = np.array(predicted_scores)

    # 2. Plot the histogram
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Histogram
    ax.hist(scores_array, bins=30, density=True, color='#4A90E2', alpha=0.7, edgecolor='black', label="Predicted Scores")
    
    # Plot the mean line
    mean_score = np.mean(scores_array)
    std_dev = np.std(scores_array)
    ax.axvline(mean_score, color='red', linestyle='dashed', linewidth=2, label=f'Achieved Mean: {mean_score:.2f}')
    
    ax.set_title('Simulated Exam Score Distribution', fontsize=16)
    ax.set_xlabel(f'Predicted Exam Score (Std Dev: {std_dev:.2f})', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.legend()
    ax.grid(axis='y', alpha=0.5)
    ax.set_xlim(40, 100)
    
    st.pyplot(fig)

# --- 3. Streamlit Application Layout ---

def main():
    st.set_page_config(page_title="Exam Score Predictor", layout="wide")
    st.title("🎓 Realistic Exam Score Predictor")
    st.markdown(f"Using **8 key factors** from `Exam_Score_Prediction.csv` to predict an approximate score centered at **{DESIRED_MEAN:.2f}** (with $\\pm$ {REALISTIC_SPREAD_SD} SD variance for realism).")
    st.divider()

    # Load and cache model/features
    model_data = setup_model_and_features()
    if model_data is None:
        return
        
    # Load the raw data for plotting (uses separate cache)
    df_raw = load_data('Exam_Score_Prediction.csv')
    if df_raw.empty:
        return
    
    _, _, global_categories, _ = model_data

    # Initialize input dictionary
    user_inputs = {}
    goal_score = DESIRED_MEAN # Default goal score

    # --- INPUT LAYOUT ---
    st.header("1. Enter Student Factors")
    
    col1, col2, col3 = st.columns([1, 1, 0.8])
    
    # Column 1: Numerical Inputs
    with col1:
        st.subheader("Effort & Health Factors")
        
        # study_hours (Capped at 10.0 for average exam)
        user_inputs['study_hours'] = st.slider(
            "Study Hours (Per Week/Day Cap)", 
            min_value=0.0, max_value=MAX_STUDY_HOURS_BASIC_EXAM_CAP, 
            value=5.0, step=0.5,
            help="Hours of concentrated study effort (Capped at 10 for basic exam modeling)."
        )
        
        # class_attendance
        user_inputs['class_attendance'] = st.slider(
            "Class Attendance (%)", 
            min_value=50.0, max_value=100.0, 
            value=90.0, step=1.0,
            help="Percentage of classes attended."
        )
        
        # sleep_hours
        user_inputs['sleep_hours'] = st.slider(
            "Sleep Hours (Per Night)", 
            min_value=4.0, max_value=10.0, 
            value=8.0, step=0.5,
            help="Average hours of sleep per night."
        )

    # Column 2: Categorical Inputs
    with col2:
        st.subheader("Context & Method Factors")

        # course
        user_inputs['course'] = st.selectbox(
            "Course Type",
            options=global_categories.get('course', ['Loading...']),
            index=global_categories.get('course', []).index('b.sc') if 'b.sc' in global_categories.get('course', []) else 0
        )
        
        # exam_difficulty
        user_inputs['exam_difficulty'] = st.selectbox(
            "Exam Difficulty Level",
            options=global_categories.get('exam_difficulty', ['Loading...']),
            index=global_categories.get('exam_difficulty', []).index('moderate') if 'moderate' in global_categories.get('exam_difficulty', []) else 0
        )
        
        # study_method
        user_inputs['study_method'] = st.selectbox(
            "Primary Study Method",
            options=global_categories.get('study_method', ['Loading...']),
            index=global_categories.get('study_method', []).index('self-study') if 'self-study' in global_categories.get('study_method', []) else 0
        )
        
        # sleep_quality
        user_inputs['sleep_quality'] = st.selectbox(
            "Sleep Quality",
            options=global_categories.get('sleep_quality', ['Loading...'])
        )
        
        # gender
        user_inputs['gender'] = st.selectbox(
            "Gender",
            options=global_categories.get('gender', ['Loading...'])
        )

    # Column 3: Goal Score Input
    with col3:
        st.subheader("Goal Setting")
        goal_score = st.slider(
            "Goal Score",
            min_value=50.0, max_value=100.0,
            value=75.0, step=1.0,
            help="The minimum score you aim to achieve."
        )
        
    # --- PREDICTION BUTTON & OUTPUT ---

    st.markdown("---")
    
    if st.button("Generate Prediction & Recommendations", type="primary", use_container_width=True):
        
        # Generate prediction
        predicted_score = predict_score(user_inputs, model_data)
        
        # Generate recommendations based on deterministic score
        recommendations = generate_recommendations(user_inputs, goal_score, model_data)
        
        st.header("4. Results & Action Plan")
        
        col_res, col_rec = st.columns([1, 1.5])
        
        # Prediction Output
        with col_res:
            st.subheader("Predicted Exam Score")
            st.markdown(
                f"""
                <div style="padding: 20px; border-radius: 10px; background-color: #d4edda; color: #155724; text-align: center; font-size: 28px; font-weight: bold;">
                    {predicted_score}
                </div>
                """, 
                unsafe_allow_html=True
            )
            st.info(f"Note: This score represents an approximate score of **{predicted_score} $\\pm$ {REALISTIC_SPREAD_SD} (1 SD)** due to human variability.")

        # Recommendations Output
        with col_rec:
            st.subheader(f"Goal: Achieve {goal_score:.1f}")
            st.markdown("##### Actionable Recommendations:")
            for rec in recommendations:
                if rec.startswith('🎯') or rec.startswith('---') or rec.startswith('💡'):
                    st.markdown(rec)
                else:
                    st.success(rec)


    # --- 5. Visualization Section ---
    st.markdown("---")
    st.header("2. Feature Relationships in the Data")
    st.markdown("Analyze how each factor relates to the final Exam Score in the dataset.")
    
    all_features = FEATURE_MAP['Numerical'] + FEATURE_MAP['Categorical']
    
    plot_cols = st.columns(3)
    
    for i, feature in enumerate(all_features):
        with plot_cols[i % 3]:
            fig = plot_relationship(df_raw, feature, TARGET_COL)
            st.pyplot(fig, use_container_width=True)

    plot_predicted_distribution(model_data)

if __name__ == '__main__':
    main()