# 🎵 VibeSync: Mood-Driven Music Recommender
[![Streamlit App](https://img.shields.io/badge/Launch-VibeSync%20App-ff4b4b?logo=streamlit)](https://vibesyncai.streamlit.app)


> *Let your mood choose the music – intelligently, beautifully, and interactively.*  
> Powered by NLP, time-series emotion trends, and Spotify-inspired vibes.

---

## Live Demo  
🎯 [Click here to try the live app](https://vibesyncai.streamlit.app)

---

## 🚀 What is VibeSync?

**VibeSync** is a Streamlit-powered, AI-backed music recommender system that personalizes songs based on your mood or a song you love. It blends:
- 💬 Natural Language Emotion Detection using Hugging Face Transformers
- 🎵 Combines Spotify API data with emotion-based logic to suggest personalized tracks.
- 🎯 Content-based filtering via audio features
- 📈 Time-Series Trend Boosting for emotion-aware recommendations
- 🌐 Interactive Streamlit UI with audio previews, album art and Music analytics

---

## 🧠 How It Works

| Feature | Description |
|--------|-------------|
| **Mood Input** | Type how you're feeling, and the model predicts your emotion |
| **Song-Based Search** | Find songs similar to any track you enter |
| **Emotion Trends** | Explore emotion distribution over years |
| **Popularity Trends** | See how average popularity evolved over time |
| **Live Recommendations** | Refresh suggestions with one click |
| **Visuals & Media** | Album art, audio snippets, and Spotify-style UI |

---

## 🛠️ Tech Stack

-  Python, Pandas, NumPy
-  Spotify Web API (via `Spotipy` for Oauth)
-  HuggingFace Transformers (for emotion detection)
-  Streamlit (for UI)
-  Matplotlib / Plotly (for mood graphs)

---

## 📊 Models & Data

- **Emotion Detection**: `joeddav/distilbert-base-uncased-go-emotions-student` from HuggingFace
- **Dataset**: Curated Spotify dataset (CSV), cleaned & enriched with emotion labels
- **Similarity Engine**: KNN on scaled audio features (`valence`, `energy`, etc.)

---

## 📂 Folder Structure

```
📁 VibesyncRecommender/
├── vibesync.py        # Main Streamlit app
├── light_spotify_dataset.csv
├── requirements.txt
├── Vibesync_Music_Recommendation_System.ipynb	
└── README.md
```

---

## Streamlit app snaps

![29EC018A-14F0-4359-970D-D8134C02389A_1_201_a](https://github.com/user-attachments/assets/a975bead-59a3-4901-adc7-3070970781ac)

![AF53C3AA-00B9-479E-BBB4-17377A7EAA3B_1_201_a](https://github.com/user-attachments/assets/ced27155-37d1-45d7-936b-a2d5df9dbcf4)

![465B7222-1EC7-4945-A460-C1B644EC2F8D_1_201_a](https://github.com/user-attachments/assets/7b2d0ac3-974b-4797-837d-e432c4e66cb2)

![1E494F97-4F6C-44AE-9AE8-1382506C7DEC_1_201_a](https://github.com/user-attachments/assets/a6dde8c9-0e4c-4d91-84b6-32ad31b7ccb2)

![FFF4774F-16AF-4CD9-BF2A-46DFC7D96243_1_201_a](https://github.com/user-attachments/assets/131bca25-8468-4041-a105-f0282db12198)

---

## ⚙️ Setup & Run Locally

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run vibesync.py
```

---

## 🔥 Future Enhancements

- 🖼️ Use CLIP model for image-based emotion sync
- 📊 Metrics dashboard: Precision@K, recall, listening time
- 🌈 Theme customization with user profiles

---

## 💡 Name Rationale: VibeSync

> **"Vibe"** – because it understands your mood.  
> **"Sync"** – because it syncs your emotions with music.  

---

## 🧑‍💻 Author

**Prajwal Venkat**  
💼 Data Scientist | Music Explorer | AI Builder

📧 prajwalvenkatv@gmail.com
Find me on [LinkedIn](https://www.linkedin.com/in/prajwal-venkat-v-9654a5180)
