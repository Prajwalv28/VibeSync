import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random

# ---------------------------- Page Configuration ----------------------------
st.set_page_config(
    page_title="🎧 VibeSync - Music Recommender", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------- Custom CSS ----------------------------
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .stApp {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    }
    
    .title-container {
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4, #ffeaa7);
        background-size: 300% 300%;
        animation: gradientShift 8s ease infinite;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .song-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .song-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.2);
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        margin: 0.5rem;
    }
    
    .mood-badge {
        background: linear-gradient(45deg, #ff6b6b, #feca57);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        font-weight: bold;
        margin: 0.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .refresh-btn {
        background: linear-gradient(45deg, #4ecdc4, #44a08d);
        border: none;
        color: white;
        padding: 0.8rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .refresh-btn:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 20px rgba(0,0,0,0.3);
    }
    
    .watermark {
        position: fixed;
        bottom: 20px;
        right: 20px;
        font-size: 4rem;
        font-weight: bold;
        opacity: 0.4;
        color: white;
        z-index: -1;
        font-family: 'Arial Black', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# Watermark
st.markdown('<div class="watermark">PV</div>', unsafe_allow_html=True)

# ---------------------------- Load Model (with caching) ----------------------------
@st.cache_resource
def load_emotion_model():
    model_name = "j-hartmann/emotion-english-distilroberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

try:
    tokenizer, model = load_emotion_model()
    model_loaded = True
except:
    st.error("⚠️ Could not load emotion detection model. Using fallback emotion detection.")
    model_loaded = False

# ---------------------------- Load Dataset (with caching) ----------------------------
@st.cache_data
def load_data():
    try:
        df_songs = pd.read_csv("light_spotify_dataset.csv")
        
        # Clean column names
        df_songs.columns = df_songs.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Fix release_date formatting
        if 'release_date' in df_songs.columns:
            # Extract year from release_date if it's in timestamp format
            df_songs['release_year'] = pd.to_numeric(df_songs['release_date'], errors='coerce')
            df_songs['release_date'] = pd.to_datetime(df_songs['release_year'], format='%Y', errors='coerce')
        else:
            # If no release_date column, create from year
            df_songs['release_date'] = pd.to_datetime(df_songs.get('year', 2020), format='%Y', errors='coerce')
            df_songs['release_year'] = df_songs['release_date'].dt.year
        
        # Add decade and year_month for analysis
        df_songs['decade'] = (df_songs['release_year'] // 10) * 10
        df_songs['year_month'] = df_songs['release_date'].dt.to_period('M')
        
        # If no emotion column exists, create a sample one
        if 'emotion' not in df_songs.columns:
            emotions = ['joy', 'sadness', 'anger', 'love', 'fear', 'surprise']
            df_songs['emotion'] = np.random.choice(emotions, size=len(df_songs))
        
        return df_songs
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

df_songs = load_data()

if df_songs is None:
    st.error("Could not load the dataset. Please check if 'light_spotify_dataset.csv' exists.")
    st.stop()

# ---------------------------- Feature Preparation ----------------------------
@st.cache_data
def prepare_features(df):
    feature_cols = ['variance', 'energy', 'danceability', 'acousticness', 'tempo']
    
    # Use available columns if the exact ones don't exist
    available_cols = []
    for col in feature_cols:
        if col in df.columns:
            available_cols.append(col)
    
    if len(available_cols) < 3:
        # Create dummy features if not enough columns
        for i, col in enumerate(feature_cols):
            if col not in df.columns:
                df[col] = np.random.uniform(0, 1, len(df))
        available_cols = feature_cols
    
    df_clean = df.dropna(subset=available_cols).copy()
    
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(df_clean[available_cols])
    
    # Add popularity weighting
    if 'popularity' in df_clean.columns:
        df_clean['weighted_popularity'] = MinMaxScaler().fit_transform(df_clean[['popularity']]).flatten()
    else:
        df_clean['weighted_popularity'] = np.random.uniform(0, 1, len(df_clean))
    
    nn_model = NearestNeighbors(n_neighbors=min(50, len(df_clean)), metric='euclidean')
    nn_model.fit(scaled_features)
    
    return df_clean, scaled_features, nn_model, available_cols, scaler

df_songs, scaled_features, nn_model, feature_cols, scaler = prepare_features(df_songs)

# ---------------------------- Enhanced Emotion Detection ----------------------------
def get_emotion_label(text):
    if not model_loaded:
        # Fallback emotion detection based on keywords
        emotion_keywords = {
            'joy': ['happy', 'excited', 'great', 'amazing', 'wonderful', 'fantastic', 'good', 'excellent'],
            'sadness': ['sad', 'depressed', 'down', 'blue', 'lonely', 'gloomy', 'melancholy'],
            'anger': ['angry', 'mad', 'furious', 'annoyed', 'frustrated', 'irritated'],
            'love': ['love', 'romantic', 'heart', 'valentine', 'relationship', 'crush'],
            'fear': ['scared', 'afraid', 'worried', 'anxious', 'nervous', 'terrified'],
            'surprise': ['surprised', 'shocked', 'amazed', 'astonished', 'unexpected']
        }
        
        text_lower = text.lower()
        for emotion, keywords in emotion_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return emotion
        
        # Special case for weather-related inputs
        if any(word in text_lower for word in ['rain', 'raining', 'cloudy', 'storm']):
            return 'sadness'  # This matches your observation
        
        return 'joy'  # Default emotion
    
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]
        labels = model.config.id2label
        predicted_emotion = labels[torch.argmax(probs).item()]
        return predicted_emotion
    except:
        return 'joy'

# ---------------------------- Emotion Vector Mapping ----------------------------
def map_emotion_to_feature_vector(emotion, feature_cols):
    # Adjusted mapping based on your feature columns
    base_mapping = {
        'joy': [0.8, 0.9, 0.8, 0.3, 0.7],
        'sadness': [0.2, 0.3, 0.3, 0.8, 0.3],
        'anger': [0.3, 0.9, 0.4, 0.2, 0.8],
        'love': [0.9, 0.6, 0.7, 0.4, 0.5],
        'fear': [0.1, 0.2, 0.2, 0.9, 0.3],
        'surprise': [0.7, 0.8, 0.6, 0.4, 0.6]
    }
    
    vector = base_mapping.get(emotion.lower(), [0.5] * 5)
    return np.array(vector[:len(feature_cols)] + [0.5] * max(0, len(feature_cols) - len(vector)))

# ---------------------------- Recommendation Function ----------------------------
def recommend_songs_by_emotion(emotion, top_k=5, random_seed=None):
    if random_seed:
        np.random.seed(random_seed)
    
    emotion_vector = map_emotion_to_feature_vector(emotion, feature_cols).reshape(1, -1)
    
    # Get more neighbors to allow for randomization
    n_neighbors = min(100, len(df_songs))
    distances, indices = nn_model.kneighbors(emotion_vector, n_neighbors=n_neighbors)
    
    results = df_songs.iloc[indices[0]].copy()
    results['similarity_score'] = 1 / (1 + distances[0])
    
    # Calculate trend boost
    results['trend_boost'] = results['release_year'].apply(
        lambda y: get_emotion_trend_boost(emotion, y) if pd.notnull(y) else 0.1
    )
    
    # Calculate final score
    results['final_score'] = (
        0.6 * results['similarity_score'] +
        0.3 * results['weighted_popularity'] +
        0.1 * results['trend_boost']
    )
    
    # Add some randomization to prevent same results
    if random_seed:
        results['final_score'] += np.random.uniform(-0.05, 0.05, len(results))
    
    return results.nlargest(top_k, 'final_score')

def recommend_similar_songs(song_name, top_k=5):
    song_matches = df_songs[df_songs['song'].str.lower().str.contains(song_name.lower(), na=False)]
    
    if song_matches.empty:
        return None
    
    # Use the first match
    song_idx = song_matches.index[0]
    song_features = scaled_features[df_songs.index.get_loc(song_idx)].reshape(1, -1)
    
    distances, indices = nn_model.kneighbors(song_features, n_neighbors=top_k+1)
    
    # Exclude the original song
    similar_indices = indices[0][1:]
    similar_distances = distances[0][1:]
    
    results = df_songs.iloc[similar_indices].copy()
    results['similarity_score'] = 1 / (1 + similar_distances)
    
    # Add trend boost
    results['trend_boost'] = results['release_year'].apply(
        lambda y: 0.1 if pd.isnull(y) else 0.1
    )
    
    results['final_score'] = (
        0.7 * results['similarity_score'] +
        0.3 * results['weighted_popularity']
    )
    
    return results.sort_values('final_score', ascending=False)

# ---------------------------- Trend Analysis ----------------------------
def get_emotion_trend_boost(emotion, release_year):
    try:
        decade = (release_year // 10) * 10
        # Simple trend boost calculation
        recent_boost = max(0, (release_year - 1980) / (2025 - 1980)) * 0.2
        return 0.1 + recent_boost
    except:
        return 0.1

# ---------------------------- Main UI ----------------------------
# Title with animated gradient
st.markdown("""
<div class="title-container">
    <h1 style='margin: 0; font-size: 3rem; font-weight: bold;'>🎧 VibeSync</h1>
    <p style='margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;'>
        AI-Powered Music Recommendation System
    </p>
</div>
""", unsafe_allow_html=True)

# Initialize session state for refresh functionality
if 'refresh_seed' not in st.session_state:
    st.session_state.refresh_seed = 42

# Create tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["🎯 Mood Detection ", "🔍 Similar Songs ", "📊 Music Analytics "])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 💬 Tell us about your mood")
        user_input = st.text_area(
            "Describe how you're feeling:",
            "I am feeling great today!",
            height=100,
            help="Describe your current mood or feelings, and we'll recommend songs that match!"
        )
        
        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            detect_btn = st.button("🎯 Detect Mood & Recommend", use_container_width=True)
        with col_btn2:
            refresh_btn = st.button("🔄 Get Different Songs", use_container_width=True)
    
    with col2:
        if 'current_mood' in st.session_state:
            st.markdown(f"### Current Mood")
            st.markdown(f'<div class="mood-badge">{st.session_state.current_mood.title()}</div>', 
                       unsafe_allow_html=True)
    
    if detect_btn or refresh_btn:
        if refresh_btn and 'current_mood' not in st.session_state:
            st.warning("Please detect mood first!")
        else:
            if detect_btn:
                mood = get_emotion_label(user_input)
                st.session_state.current_mood = mood
                st.session_state.refresh_seed = 42
            else:
                mood = st.session_state.current_mood
                st.session_state.refresh_seed = random.randint(1, 1000)
            
            st.markdown(f"""
            <div style='background: linear-gradient(45deg, #4ecdc4, #44a08d); 
                        color: white; padding: 1rem; border-radius: 10px; 
                        text-align: center; margin: 1rem 0; 
                        box-shadow: 0 4px 15px rgba(0,0,0,0.2);'>
                <h3 style='margin: 0;'>🎭 Detected Mood: <strong>{mood.title()}</strong></h3>
            </div>
            """, unsafe_allow_html=True)
            
            with st.spinner("Finding perfect songs for your mood..."):
                recommendations = recommend_songs_by_emotion(
                    mood, top_k=5, random_seed=st.session_state.refresh_seed
                )
                
                if not recommendations.empty:
                    st.markdown("### 🎵 Recommended Songs")
                    
                    for idx, (_, song) in enumerate(recommendations.iterrows()):
                        with st.container():
                            st.markdown(f"""
                            <div class="song-card">
                                <h4>🎵 {song.get('song', 'Unknown Song')}</h4>
                                <p><strong>Artist:</strong> {song.get('artist', 'Unknown Artist')}</p>
                                <p><strong>Popularity:</strong> {song.get('popularity', 0):.0f}/100</p>
                                <p><strong>Year:</strong> {song.get('release_year', 'Unknown')}</p>
                                <p><strong>Match Score:</strong> {song.get('final_score', 0):.2f}</p>
                            </div>
                            """, unsafe_allow_html=True)

with tab2:
    st.markdown("### 🔍 Find Similar Songs")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        song_name = st.text_input(
            "Enter a song name:",
            placeholder="e.g., Shape of You, Bohemian Rhapsody",
            help="Enter part of a song name to find similar tracks"
        )
    
    with col2:
        similar_btn = st.button("🎵 Find Similar", use_container_width=True)
    
    if similar_btn and song_name:
        with st.spinner("Searching for similar songs..."):
            similar_songs = recommend_similar_songs(song_name, top_k=5)
            
            if similar_songs is not None and not similar_songs.empty:
                st.success(f"Found songs similar to: **{song_name}**")
                
                for idx, (_, song) in enumerate(similar_songs.iterrows()):
                    st.markdown(f"""
                    <div class="song-card">
                        <h4>🎵 {song.get('song', 'Unknown Song')}</h4>
                        <p><strong>Artist:</strong> {song.get('artist', 'Unknown Artist')}</p>
                        <p><strong>Popularity:</strong> {song.get('popularity', 0):.0f}/100</p>
                        <p><strong>Year:</strong> {song.get('release_year', 'Unknown')}</p>
                        <p><strong>Similarity:</strong> {song.get('similarity_score', 0):.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.error(f"❌ No songs found matching '{song_name}'. Try a different search term.")

with tab3:
    st.markdown("### 📊 Music Analytics Dashboard")
    
    # Popularity over time (Line chart)
    if 'year_month' in df_songs.columns and 'popularity' in df_songs.columns:
        st.markdown("#### 📈 Average Popularity Over Time")
        
        monthly_data = df_songs.groupby('year_month')['popularity'].mean().reset_index()
        monthly_data['year_month_str'] = monthly_data['year_month'].astype(str)
        
        fig_popularity = px.line(
            monthly_data, 
            x='year_month_str', 
            y='popularity',
            title='Average Song Popularity Trend',
            color_discrete_sequence=['#00d4aa']
        )
        fig_popularity.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            xaxis_title="Year-Month",
            yaxis_title="Average Popularity"
        )
        fig_popularity.update_traces(line=dict(width=3))
        st.plotly_chart(fig_popularity, use_container_width=True)
    
    # Emotion trends
    if 'emotion' in df_songs.columns and 'release_year' in df_songs.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎭 Emotion Distribution Over Years")
            
            # Create emotion trends data
            emotion_data = df_songs.groupby(['release_year', 'emotion']).size().unstack(fill_value=0)
            emotion_data_reset = emotion_data.reset_index()
            
            # Convert to long format for plotly
            emotion_long = pd.melt(
                emotion_data_reset, 
                id_vars=['release_year'], 
                var_name='emotion', 
                value_name='count'
            )
            
            fig_emotions = px.line(
                emotion_long, 
                x='release_year', 
                y='count',
                color='emotion',
                title='Songs per Emotion Over Time'
            )
            fig_emotions.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig_emotions, use_container_width=True)
        
        with col2:
            st.markdown("#### 📊 Emotion Percentage Trends")
            
            # Percentage trends
            emotion_percent = emotion_data.div(emotion_data.sum(axis=1), axis=0) * 100
            emotion_percent_reset = emotion_percent.reset_index()
            emotion_percent_long = pd.melt(
                emotion_percent_reset,
                id_vars=['release_year'],
                var_name='emotion',
                value_name='percentage'
            )
            
            fig_percent = px.area(
                emotion_percent_long,
                x='release_year',
                y='percentage',
                color='emotion',
                title='Emotion Percentage Distribution'
            )
            fig_percent.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig_percent, use_container_width=True)

# ---------------------------- Sidebar Info ----------------------------
with st.sidebar:
    st.markdown("### 📊 Dataset Info")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Songs", f"{len(df_songs):,}")
    with col2:
        if 'emotion' in df_songs.columns:
            st.metric("Emotions", df_songs['emotion'].nunique())
    
    if 'release_year' in df_songs.columns:
        year_range = f"{df_songs['release_year'].min():.0f} - {df_songs['release_year'].max():.0f}"
        st.metric("Year Range", year_range)
    
    st.markdown("### 🎭 Available Emotions")
    if 'emotion' in df_songs.columns:
        for emotion in sorted(df_songs['emotion'].unique()):
            count = len(df_songs[df_songs['emotion'] == emotion])
            st.write(f"**{emotion.title()}**: {count:,} songs")
    
    st.markdown("---")
    st.markdown("### ℹ️ How it works")
    st.markdown("""
    1. **Mood Detection**: AI analyzes your text to detect emotions
    2. **Feature Matching**: Maps emotions to audio features
    3. **Smart Ranking**: Uses similarity + popularity + trends
    4. **Fresh Results**: Refresh button gives different songs
    """)

# ---------------------------- Footer ----------------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 2rem; opacity: 0.8;'>
    <p>🎵 Made with ❤️ by Prajwal </p>
</div>
""", unsafe_allow_html=True)