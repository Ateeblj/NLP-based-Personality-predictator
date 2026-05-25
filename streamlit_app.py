import streamlit as st
import requests
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Personality Prediction", layout="wide")
st.title("🧠 Personality Prediction (Introvert/Extrovert)")

st.markdown("Enter behavioral traits to predict personality type.")

# Load training data categories dynamically
@st.cache_data
def load_categories():
    try:
        train_df = pd.read_csv('train.csv')
        stage_fear_cats = sorted(train_df['Stage_fear'].dropna().unique().tolist())
        drained_cats = sorted(train_df['Drained_after_socializing'].dropna().unique().tolist())
        return stage_fear_cats, drained_cats
    except:
        return [], []

stage_fear_options, drained_options = load_categories()

# Sidebar - Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    api_url = st.text_input("Flask API URL", value="http://localhost:5000")
    
    try:
        health = requests.get(f"{api_url}/health", timeout=2)
        if health.status_code == 200:
            st.success("✅ Backend connected")
        else:
            st.error("❌ Backend error")
    except:
        st.warning("⚠️ Cannot reach backend. Start Flask: `python flask_app.py`")

# Main content
st.markdown("### 📝 Enter Your Behavioral Traits")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
        <div class="card">
        <h3>📊 Numerical Features</h3>
        <p style="color: #666; font-size: 0.95em;">Rate each on a scale of 1-5</p>
        </div>
        """, unsafe_allow_html=True)
    
    Time_spent_Alone = st.slider(
        "⏰ Time Spent Alone",
        1.0, 5.0, 3.0, 0.1,
        help="How much time do you spend alone?"
    )
    
    Social_event_attendance = st.slider(
        "🎉 Social Event Attendance",
        1.0, 5.0, 3.0, 0.1,
        help="How often do you attend social events?"
    )
    
    Going_outside = st.slider(
        "🚶 Going Outside",
        1.0, 5.0, 3.0, 0.1,
        help="How often do you go outside?"
    )
    
    Friends_circle_size = st.slider(
        "👥 Friends Circle Size",
        1.0, 5.0, 3.0, 0.1,
        help="How large is your friend circle?"
    )
    
    Post_frequency = st.slider(
        "📱 Post Frequency",
        1.0, 5.0, 3.0, 0.1,
        help="How often do you post on social media?"
    )

with col2:
    st.markdown("""
        <div class="card">
        <h3>✅ Categorical Features</h3>
        <p style="color: #666; font-size: 0.95em;">Select from available options</p>
        </div>
        """, unsafe_allow_html=True)
    
    Stage_fear = st.selectbox(
        "😰 Stage Fear",
        stage_fear_options if stage_fear_options else ["Loading..."],
        help="How much do you fear performing on stage?"
    )
    
    Drained_after_socializing = st.selectbox(
        "🔋 Drained After Socializing",
        drained_options if drained_options else ["Loading..."],
        help="Do you feel drained after social interaction?"
    )
    
    st.markdown("---")
    st.markdown("")  # Spacing
    st.markdown("")  # Spacing
    st.markdown("")  # Spacing

# Predict button
col_pred = st.columns([1, 1, 1])
with col_pred[1]:
    predict_clicked = st.button("🔮 Predict My Personality", use_container_width=True)

if predict_clicked:
    payload = {
        "instances": [
            {
                "Time_spent_Alone": float(Time_spent_Alone),
                "Social_event_attendance": float(Social_event_attendance),
                "Going_outside": float(Going_outside),
                "Friends_circle_size": float(Friends_circle_size),
                "Post_frequency": float(Post_frequency),
                "Stage_fear": Stage_fear,
                "Drained_after_socializing": Drained_after_socializing
            }
        ]
    }
    
    try:
        with st.spinner("🔄 Analyzing your traits..."):
            resp = requests.post(f"{api_url}/predict", json=payload, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        
        if 'predictions' in data and data['predictions']:
            pred = data['predictions'][0]
            label = pred['predicted_label']
            prob = pred['predicted_prob']
            
            st.markdown("---")
            
            # Result display
            col_result1, col_result2 = st.columns([1, 1])
            
            with col_result1:
                st.markdown("""
                    <div class="result-container">
                    <h4 style="margin: 0 0 15px 0;">🎯 Your Personality</h4>
                    """, unsafe_allow_html=True)
                
                if label == "Introvert":
                    st.markdown(f"""
                        <div class="result-label" style="color: #4ade80;">
                        {label} 🔷
                        </div>
                        """, unsafe_allow_html=True)
                    st.markdown("You tend to be **reserved**, prefer **intimate settings**, and gain energy from **solitude**.", 
                               unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="result-label" style="color: #fbbf24;">
                        {label} 🔶
                        </div>
                        """, unsafe_allow_html=True)
                    st.markdown("You tend to be **outgoing**, enjoy **social events**, and gain energy from **people**.", 
                               unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col_result2:
                st.markdown("""
                    <div class="card" style="border-left: 5px solid #667eea; background: rgba(255, 255, 255, 0.98);">
                    <h4 style="color: #667eea; margin-top: 0;">📊 Your Input Summary</h4>
                    """, unsafe_allow_html=True)
                
                summary_df = pd.DataFrame([pred['input']]).T
                summary_df.columns = ['Value']
                
                for key, value in pred['input'].items():
                    if isinstance(value, float):
                        st.markdown(f"**{key.replace('_', ' ')}**: {value:.1f}")
                    else:
                        st.markdown(f"**{key.replace('_', ' ')}**: {value}")
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Personality insights
            st.markdown("---")
            with st.expander("💡 Personality Insights"):
                if label == "Introvert":
                    st.markdown("""
                    ### Introvert Traits
                    - 🤐 Prefer deep, meaningful conversations
                    - 📚 Enjoy solitary activities
                    - 🧠 Tend to be reflective and thoughtful
                    - ⚡ Recharge through alone time
                    - 🎯 Often have focused attention
                    """)
                else:
                    st.markdown("""
                    ### Extrovert Traits
                    - 🗣️ Love meeting new people
                    - 🎊 Thrive in social situations
                    - ⚡ Get energized by group activities
                    - 🌟 Often natural leaders
                    - 🎯 Broad social networks
                    """)
        else:
            st.error(f"❌ Unexpected response: {data}")
    
    except requests.exceptions.ConnectionError:
        st.error("❌ **Connection Failed**")
        st.info(f"Cannot reach Flask backend at {api_url}. Please ensure it's running.")
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ **HTTP Error {e.response.status_code}**")
        try:
            st.write(e.response.json())
        except:
            st.write(e.response.text)
    except Exception as e:
        st.error(f"❌ **Error**: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #888; font-size: 0.9em; padding: 20px;">
    <p>🧠 Personality Prediction System | Powered by Flask + Streamlit + Neural Networks</p>
    </div>
    """, unsafe_allow_html=True)
