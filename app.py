import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

# --- Load data ---
@st.cache_data
def load_data():
    # Load from pickle file
    df = pd.read_pickle('df_clean.pkl')
    return df

df = load_data()

# --- App Title ---
st.title("🎬 Pixar Box Office Prediction")

# --- Show Data ---
if st.checkbox("Show raw data"):
    st.write(df)

# --- Model Training ---
st.header("Model Training & Evaluation")

# Define features & target
features = ['budget', 'run_time', 'rotten_tomatoes_score', 'metacritic_score', 'imdb_score']
target = 'box_office_worldwide'

# Drop missing data
df_model = df[features + [target]].dropna()

# Train/test split
if df_model.shape[0] > 1:
    X = df_model[features]
    y = df_model[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    st.subheader("Performance Metrics")
    st.write(f"**R-squared:** {r2:.3f}")
    st.write(f"**Root Mean Squared Error:** {rmse:,.0f}")
    
    # --- Feature Importance Plot ---
    importances = model.feature_importances_
    indices = np.argsort(importances)
    
    st.subheader("Feature Importance")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(len(indices)), importances[indices], color='skyblue')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([features[i] for i in indices])
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.set_title("Feature Importance in Random Forest")
    st.pyplot(fig)

else:
    st.warning("Not enough data to train and evaluate the model.")

# --- Prediction Form ---
st.header("Try a Prediction!")

user_input = {}
for feature in features:
    user_input[feature] = st.number_input(
        f"Input {feature}", min_value=0.0, value=100.0, step=10.0
    )

if st.button("Predict Box Office"):
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]
    st.success(f"🎉 Estimated Box Office Worldwide: ${prediction:,.0f}")

# --- Footer ---
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit.")

