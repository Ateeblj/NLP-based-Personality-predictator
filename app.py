import streamlit as st
import pickle
import re
import string
import numpy as np
import time
import random
from nltk.corpus import stopwords
import nltk

# Download stopwords once
nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words('english'))

# =============================================================================
# MBTI DESCRIPTIONS (Full names + letter meanings + Pros/Cons + Famous examples)
# =============================================================================

# Full type names and easy descriptions
TYPE_NAMES = {
    "INFP": "The Mediator",
    "INTP": "The Logician", 
    "INFJ": "The Advocate",
    "INTJ": "The Architect",
    "ISFP": "The Adventurer",
    "ISTP": "The Virtuoso",
    "ISFJ": "The Defender",
    "ISTJ": "The Logistician",
    "ENFP": "The Campaigner",
    "ENTP": "The Debater",
    "ENFJ": "The Protagonist",
    "ENTJ": "The Commander",
    "ESFP": "The Entertainer",
    "ESTP": "The Entrepreneur",
    "ESFJ": "The Consul",
    "ESTJ": "The Executive"
}

# Easy-to-understand descriptions for each type
TYPE_DESCRIPTIONS = {
    "INFP": "You're an idealist with strong values. You care deeply about authenticity and making the world better. Creative and compassionate, you seek meaning in everything.",
    "INTP": "You're a logical thinker who loves exploring ideas and theories. Curious and analytical, you enjoy understanding how things work.",
    "INFJ": "You're a creative visionary who cares about helping others. Intuitive and decisive, you have a gift for understanding complex emotions.",
    "INTJ": "You're a strategic planner who loves solving complex problems. Independent and determined, you turn ideas into reality.",
    "ISFP": "You're an artistic soul who lives in the moment. Gentle and sensitive, you express yourself through action and creativity.",
    "ISTP": "You're a practical problem-solver who loves hands-on challenges. Calm under pressure, you fix things with logic and skill.",
    "ISFJ": "You're a loyal helper who remembers every detail. Warm and responsible, you quietly support those you love.",
    "ISTJ": "You're a dependable realist who values order. Hardworking and honest, you get things done right the first time.",
    "ENFP": "You're an enthusiastic idea person who loves possibilities. Energetic and creative, you inspire others with your passion.",
    "ENTP": "You're a clever debater who loves intellectual challenges. Quick-witted and innovative, you thrive on new ideas.",
    "ENFJ": "You're a charismatic leader who brings people together. Warm and persuasive, you help others reach their potential.",
    "ENTJ": "You're a natural commander who gets things done. Bold and strategic, you lead with confidence and vision.",
    "ESFP": "You're a spontaneous entertainer who loves life. Playful and energetic, you make every moment fun.",
    "ESTP": "You're a bold risk-taker who lives for action. Smart and energetic, you're great at thinking on your feet.",
    "ESFJ": "You're a caring organizer who builds community. Friendly and helpful, you bring people together.",
    "ESTJ": "You're an efficient leader who values tradition. Practical and decisive, you get results through organization."
}

# Pros and Cons for each type
PROS_CONS = {
    "INFP": {
        "pros": ["Deep care for others", "Creative and imaginative", "Strong personal values", "Great listeners"],
        "cons": ["Can be too idealistic", "Struggle with criticism", "Overthink simple things", "Avoid conflict too much"]
    },
    "INTP": {
        "pros": ["Excellent problem solvers", "Very logical and fair", "Curious and knowledgeable", "Open-minded"],
        "cons": ["Can seem distant or cold", "Overcomplicate simple things", "Messy and disorganized", "Struggle with emotions"]
    },
    "INFJ": {
        "pros": ["Deeply empathetic", "Visionary and creative", "Strong moral compass", "Excellent at advice"],
        "cons": ["Burn out easily helping others", "Too sensitive to criticism", "Perfectionist tendencies", "Hard to get to know"]
    },
    "INTJ": {
        "pros": ["Strategic masterminds", "Highly independent", "Confident in decisions", "Always learning"],
        "cons": ["Can be arrogant", "Struggle with emotions", "Impatient with incompetence", "Too focused on the future"]
    },
    "ISFP": {
        "pros": ["Artistic and creative", "Loyal to loved ones", "Live in the moment", "Gentle with others"],
        "cons": ["Avoid long-term planning", "Easily stressed by conflict", "Struggle with criticism", "Can be too reserved"]
    },
    "ISTP": {
        "pros": ["Practical problem solvers", "Calm under pressure", "Adventurous and bold", "Quick learners"],
        "cons": ["Risk-taking can backfire", "Difficulty expressing feelings", "Rebel against rules", "Can seem uncaring"]
    },
    "ISFJ": {
        "pros": ["Incredibly loyal", "Remember everything important", "Great practical helpers", "Patient and kind"],
        "cons": ["Take things too personally", "Struggle to say no", "Avoid change too much", "Overwork themselves"]
    },
    "ISTJ": {
        "pros": ["Extremely dependable", "Detail-oriented", "Honest and fair", "Respect traditions"],
        "cons": ["Can be rigid and stubborn", "Struggle with new ideas", "Too serious sometimes", "Appear cold"]
    },
    "ENFP": {
        "pros": ["Infectious enthusiasm", "Creative brainstormer", "Great with people", "Loves possibilities"],
        "cons": ["Get bored easily", "Struggle with follow-through", "Too emotional sometimes", "Easily distracted"]
    },
    "ENTP": {
        "pros": ["Brilliant debater", "Innovative thinker", "Quick on their feet", "Love learning"],
        "cons": ["Argumentative", "Can be insensitive", "Struggle with routine", "Forget important details"]
    },
    "ENFJ": {
        "pros": ["Natural leader", "Great at inspiring others", "Deeply caring", "Good at resolving fights"],
        "cons": ["Need constant approval", "Too hard on themselves", "Can manipulate without trying", "Burn out helping"]
    },
    "ENTJ": {
        "pros": ["Born to lead", "Efficient executor", "Self-confident", "Strategic genius"],
        "cons": ["Can be intimidating", "Impatient with emotions", "Workaholic tendencies", "Too focused on winning"]
    },
    "ESFP": {
        "pros": ["Life of the party", "Great at making friends", "Fun and spontaneous", "Practical doer"],
        "cons": ["Avoid deep conversations", "Struggle with commitment", "Can be impulsive", "Seek constant excitement"]
    },
    "ESTP": {
        "pros": ["Action hero energy", "Great under pressure", "Smart risk-taker", "Charming and persuasive"],
        "cons": ["Reckless at times", "Avoid thinking long-term", "Can hurt feelings without knowing", "Get bored easily"]
    },
    "ESFJ": {
        "pros": ["Super helpful", "Great at organizing events", "Loyal friend", "Caring and warm"],
        "cons": ["Need approval from others", "Take things too personally", "Can be controlling", "Struggle with change"]
    },
    "ESTJ": {
        "pros": ["Get things done", "Natural leader", "Hardworking and honest", "Great at making plans"],
        "cons": ["Can be bossy", "Stubborn about routines", "Insensitive to feelings", "Work too much"]
    }
}

# Famous people
FAMOUS_PEOPLE = {
    "INFP": ["J.K. Rowling", "Johnny Depp", "William Shakespeare", "Björk"],
    "INTP": ["Albert Einstein", "Charles Darwin", "Marie Curie", "Tina Fey"],
    "INFJ": ["Martin Luther King Jr.", "Oprah Winfrey", "Mother Teresa", "Lady Gaga"],
    "INTJ": ["Elon Musk", "Mark Zuckerberg", "Jane Austen", "Stephen Hawking"],
    "ISFP": ["Michael Jackson", "Britney Spears", "Bob Dylan", "Frida Kahlo"],
    "ISTP": ["Bruce Lee", "Clint Eastwood", "Tom Cruise", "Harrison Ford"],
    "ISFJ": ["Selena Gomez", "Rosa Parks", "Kate Middleton", "King Charles III"],
    "ISTJ": ["Queen Elizabeth II", "George Washington", "Warren Buffett", "Angela Merkel"],
    "ENFP": ["Robin Williams", "Ellen DeGeneres", "Robert Downey Jr.", "Mark Twain"],
    "ENTP": ["Leonardo da Vinci", "Steve Jobs", "Walt Disney", "Jim Carrey"],
    "ENFJ": ["Barack Obama", "Martin Luther King", "Oprah Winfrey", "Maya Angelou"],
    "ENTJ": ["Steve Jobs", "Margaret Thatcher", "Franklin D. Roosevelt", "Taylor Swift"],
    "ESFP": ["Elvis Presley", "Marilyn Monroe", "Justin Bieber", "Will Smith"],
    "ESTP": ["Ernest Hemingway", "Donald Trump", "Madonna", "John F. Kennedy"],
    "ESFJ": ["Taylor Swift", "Bill Clinton", "Hugh Jackman", "Jennifer Lopez"],
    "ESTJ": ["Judge Judy", "Michelle Obama", "George Washington", "Hillary Clinton"]
}

# Career suggestions
CAREERS = {
    "INFP": ["Writer", "Psychologist", "Artist", "Counselor", "Social Worker"],
    "INTP": ["Software Developer", "Scientist", "Architect", "Professor", "Strategist"],
    "INFJ": ["Life Coach", "Writer", "Psychologist", "Social Worker", "Human Resources"],
    "INTJ": ["Executive", "Engineer", "Scientist", "Lawyer", "Consultant"],
    "ISFP": ["Artist", "Musician", "Veterinarian", "Chef", "Fashion Designer"],
    "ISTP": ["Mechanic", "Pilot", "Forensic Scientist", "Engineer", "Athlete"],
    "ISFJ": ["Nurse", "Librarian", "Teacher", "Accountant", "Administrator"],
    "ISTJ": ["Accountant", "Lawyer", "Doctor", "Police Officer", "Data Analyst"],
    "ENFP": ["Actor", "Entrepreneur", "Journalist", "Public Speaker", "Sales"],
    "ENTP": ["Entrepreneur", "Lawyer", "Marketing", "Engineer", "Comedian"],
    "ENFJ": ["HR Manager", "Teacher", "Politician", "Event Planner", "Clergy"],
    "ENTJ": ["CEO", "Military Officer", "Judge", "Project Manager", "Entrepreneur"],
    "ESFP": ["Performer", "Sales", "Tour Guide", "Recreation Worker", "Barista"],
    "ESTP": ["Entrepreneur", "Salesperson", "Paramedic", "Athlete", "Police Officer"],
    "ESFJ": ["Nurse", "Social Worker", "Event Planner", "Real Estate Agent", "Administrator"],
    "ESTJ": ["Manager", "Judge", "Police Officer", "Accountant", "Project Manager"]
}

# Letter meanings (easy wording)
LETTER_DESCRIPTIONS = {
    "I": {"name": "Introverted", "desc": "🌙 You gain energy from quiet time alone. You prefer deep conversations with few close friends rather than large parties."},
    "E": {"name": "Extraverted", "desc": "☀️ You gain energy from being around people. You love social activities and meeting new friends."},
    "N": {"name": "Intuitive", "desc": "🔮 You focus on possibilities, patterns, and the big picture. You enjoy thinking about the future and new ideas."},
    "S": {"name": "Observant", "desc": "📖 You focus on concrete facts, details, and what's real right now. You trust experience and practical things."},
    "T": {"name": "Thinking", "desc": "⚖️ You make decisions using logic and objective analysis. You value fairness and truth over personal feelings."},
    "F": {"name": "Feeling", "desc": "❤️ You make decisions based on values and how they affect people. You prioritize harmony and empathy."},
    "J": {"name": "Judging", "desc": "📅 You prefer structure, plans, and decisions. You like knowing what's going to happen next."},
    "P": {"name": "Perceiving", "desc": "🌀 You prefer flexibility, spontaneity, and keeping options open. You adapt easily to new situations."}
}

# =============================================================================
# LOAD MODEL
# =============================================================================
@st.cache_resource
def load_model():
    try:
        with open('mbti_model.pkl', 'rb') as f:
            artifacts = pickle.load(f)
        return artifacts
    except FileNotFoundError:
        st.error("Model file 'mbti_model.pkl' not found. Please make sure it's in the same directory.")
        st.stop()

artifacts = load_model()
vectorizer = artifacts['vectorizer']
label_encoder = artifacts['label_encoder']
lr_model = artifacts['logistic_regression']
xgb_model = artifacts.get('xgboost')

# =============================================================================
# PREPROCESS & PREDICT
# =============================================================================
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.replace("|||", " ").lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [w for w in words if w not in STOPWORDS and len(w) > 2]
    return ' '.join(words)

def predict(text):
    cleaned = preprocess_text(text)
    X = vectorizer.transform([cleaned])
    proba_list = []
    if lr_model is not None:
        proba_list.append(lr_model.predict_proba(X))
    if xgb_model is not None:
        proba_list.append(xgb_model.predict_proba(X))
    final_proba = np.average(proba_list, axis=0)[0]
    pred_idx = np.argmax(final_proba)
    mbti_type = label_encoder.classes_[pred_idx]
    
    # Get top 3 with probabilities
    top3_indices = np.argsort(final_proba)[::-1][:3]
    top3_types = [(label_encoder.classes_[i], final_proba[i]) for i in top3_indices]
    
    return mbti_type, top3_types

# =============================================================================
# CUSTOM CSS FOR DARK THEME & ANIMATIONS
# =============================================================================
def apply_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Inter', 'Poppins', sans-serif;
    }
    
    /* Dark elegant background */
    .stApp {
        background: linear-gradient(145deg, #1a1a2e 0%, #16213e 50%, #1a1a2e 100%);
        background-attachment: fixed;
    }
    
    /* Main container cards */
    .main-header {
        text-align: center;
        padding: 2rem;
        background: rgba(255,255,255,0.08);
        backdrop-filter: blur(12px);
        border-radius: 32px;
        border: 1px solid rgba(255,255,255,0.15);
        margin-bottom: 2rem;
        animation: fadeInDown 0.8s ease-out;
    }
    
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Result cards */
    .result-card {
        animation: fadeInUp 0.6s ease-out forwards;
        background: rgba(30,30,46,0.85);
        backdrop-filter: blur(8px);
        border-radius: 24px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 15px 35px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
        color: #f1f1f1;
    }
    
    .result-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.4);
        border-color: rgba(100,150,255,0.3);
    }
    
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Input text area - dark theme */
    .stTextArea textarea {
        background: #0f0f1a !important;
        color: #ffffff !important;
        border-radius: 20px !important;
        border: 1px solid #2a2a3e !important;
        font-size: 1rem !important;
        line-height: 1.6 !important;
        transition: all 0.3s ease;
    }
    
    .stTextArea textarea:focus {
        border-color: #6c8cff !important;
        box-shadow: 0 0 0 2px rgba(108,140,255,0.3) !important;
        background: #141422 !important;
    }
    
    /* Input label */
    .stTextArea label {
        color: #e0e0ff !important;
        font-weight: 500 !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6c8cff 0%, #8a6cff 100%);
        color: white;
        border: none;
        border-radius: 40px;
        padding: 0.7rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(108,140,255,0.3);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 8px 25px rgba(108,140,255,0.5);
        background: linear-gradient(135deg, #7a9cff 0%, #9a7cff 100%);
    }
    
    /* Badge */
    .type-badge {
        display: inline-block;
        background: linear-gradient(135deg, #6c8cff 0%, #8a6cff 100%);
        color: white;
        padding: 0.3rem 1.2rem;
        border-radius: 60px;
        font-weight: bold;
        font-size: 1.2rem;
        letter-spacing: 1px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    
    /* Pros & Cons cards inside result */
    .pro-card, .con-card {
        background: rgba(20,20,35,0.7);
        border-radius: 16px;
        padding: 0.8rem;
        margin: 0.6rem 0;
        transition: transform 0.2s ease;
        backdrop-filter: blur(4px);
    }
    
    .pro-card {
        border-left: 4px solid #4cd964;
    }
    
    .con-card {
        border-left: 4px solid #ff5e5e;
    }
    
    .pro-card:hover, .con-card:hover {
        transform: translateX(5px);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.05) !important;
        border-radius: 20px !important;
        color: #ddd !important;
    }
    
    /* Info boxes */
    .stAlert {
        background: rgba(0,0,0,0.5) !important;
        border-radius: 16px !important;
        border-left: 4px solid #6c8cff !important;
    }
    
    /* Code block (share section) */
    .stCodeBlock {
        background: #0a0a12 !important;
        border-radius: 16px !important;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #6c8cff !important;
    }
    
    /* Captions & misc text */
    .stMarkdown, .stCaption, p, li, span {
        color: #f0f0f0 !important;
    }
    
    /* Warning / info colors */
    .stWarning {
        background: rgba(255,200,100,0.15) !important;
        border-left-color: #ffaa44 !important;
    }
    
    /* Expander content */
    .streamlit-expanderContent {
        background: rgba(0,0,0,0.2) !important;
        border-radius: 20px;
        padding: 1rem;
    }
    
    /* Floating animation for header */
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    .floating {
        animation: float 5s ease-in-out infinite;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .type-badge { font-size: 1rem; }
        .result-card { padding: 1rem; }
    }
    </style>
    """, unsafe_allow_html=True)

# =============================================================================
# DISPLAY RESULT DETAILS
# =============================================================================
def show_type_details(mbti_type, confidence_percent):
    st.markdown(f"""
    <div class="result-card">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <span class="type-badge" style="font-size: 2rem; padding: 0.5rem 1.5rem;">{mbti_type}</span>
                <h2 style="margin-top: 0.5rem; margin-bottom: 0; color: white;">{TYPE_NAMES.get(mbti_type, 'Unique Mind')}</h2>
                <p style="color: #aaa; margin-top: 0.25rem;">Confidence: {confidence_percent:.1f}% match with your text</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="result-card">
        <h3>📝 What This Means For You</h3>
        <p style="font-size: 1.1rem; line-height: 1.6; color: #f0f0f0;">{TYPE_DESCRIPTIONS.get(mbti_type, "You have a unique personality that combines these traits in special ways.")}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown("### 🔍 Breaking Down Your Type")
    st.markdown("Each letter tells something special about how you think and act:")
    cols = st.columns(4)
    for i, letter in enumerate(mbti_type):
        info = LETTER_DESCRIPTIONS.get(letter, {})
        name = info.get("name", letter)
        desc = info.get("desc", "")
        with cols[i]:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #2a2a40 0%, #1a1a30 100%); border-radius: 12px; padding: 0.75rem; text-align: center; height: 100%; border: 1px solid rgba(255,255,255,0.1);">
                <span class="type-badge" style="background: #6c8cff; font-size: 1.5rem; margin-bottom: 0.5rem;">{letter}</span>
                <p style="color: white;"><strong>{name}</strong></p>
                <p style="font-size: 0.85rem; color: #ccc;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown("### ⚡ Strengths & Weaknesses")
    pros = PROS_CONS.get(mbti_type, {}).get("pros", ["Caring", "Creative", "Loyal", "Hardworking"])
    cons = PROS_CONS.get(mbti_type, {}).get("cons", ["Can be sensitive", "Avoids conflict", "Overthinks", "Struggles with criticism"])
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### ✅ Your Superpowers")
        for pro in pros:
            st.markdown(f'<div class="pro-card">✨ {pro}</div>', unsafe_allow_html=True)
    with col2:
        st.markdown("#### 🌱 Areas to Grow")
        for con in cons:
            st.markdown(f'<div class="con-card">💪 {con}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🌟 Famous People Like You")
        famous = FAMOUS_PEOPLE.get(mbti_type, ["Many amazing people!", "Creative thinkers", "Inspirational leaders"])
        for person in famous[:3]:
            st.markdown(f"🎭 {person}")
    with col2:
        st.markdown("### 💼 Career Paths That Fit")
        careers = CAREERS.get(mbti_type, ["Many fields!", "Where your strengths shine", "What makes you happy"])
        for career in careers[:3]:
            st.markdown(f"📌 {career}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown("### 💡 Did You Know?")
    facts = {
        "INFP": "INFPs make up only 4% of the population, but they're often writers and artists who change the world with their ideas!",
        "INTP": "INTPs are sometimes called 'the architects of ideas' - Einstein and Darwin were both INTPs!",
        "INFJ": "INFJs are the rarest personality type - only 1-2% of people share your gift of deep intuition!",
        "INTJ": "INTJs love strategy so much that many become master chess players or successful entrepreneurs!",
        "ISFP": "ISFPs see beauty everywhere - many famous musicians and painters share your artistic soul!",
        "ISTP": "ISTPs are natural heroes - they stay calm in emergencies and become amazing firefighters or pilots!",
        "ISFJ": "ISFJs remember every birthday and special moment - you're the friend everyone wants!",
        "ISTJ": "ISTJs are the backbone of society - you're reliable and make the world run smoothly!",
        "ENFP": "ENFPs have magical energy - you can make anyone feel special and inspired!",
        "ENTP": "ENTPs never stop learning - you could debate both sides of any argument and still enjoy it!",
        "ENFJ": "ENFJs are born leaders - people naturally trust and follow your warm guidance!",
        "ENTJ": "ENTJs see the future - you're 5 steps ahead of everyone and that's your superpower!",
        "ESFP": "ESFPs light up every room - you make boring moments fun and unforgettable!",
        "ESTP": "ESTPs live life to the fullest - you're the person everyone wants on their adventure team!",
        "ESFJ": "ESFJs build communities - you're the reason friend groups and families stay connected!",
        "ESTJ": "ESTJs get things done - while others dream, you're already making it happen!"
    }
    st.info(facts.get(mbti_type, "Your personality type has unique gifts that make you special. Embrace who you are!"))
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# MAIN UI
# =============================================================================
def main():
    apply_custom_css()   # <-- apply dark theme
    
    st.markdown("""
    <div class="main-header floating">
        <h1 style="color: white; margin: 0; font-size: 2.5rem;">🌟 MBTI Personality Analyzer</h1>
        <p style="color: rgba(255,255,255,0.9); font-size: 1.2rem; margin-top: 0.5rem;">
            Write naturally - AI reads between the lines
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 1rem; color: white;">
        <span style="background: rgba(255,255,255,0.2); border-radius: 50px; padding: 0.25rem 1rem; margin: 0 0.25rem;">✨ Just describe yourself</span>
        <span style="background: rgba(255,255,255,0.2); border-radius: 50px; padding: 0.25rem 1rem; margin: 0 0.25rem;">📝 No tests, no pressure</span>
        <span style="background: rgba(255,255,255,0.2); border-radius: 50px; padding: 0.25rem 1rem; margin: 0 0.25rem;">🔮 Get insights instantly</span>
    </div>
    """, unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div style="background: rgba(255,255,255,0.08); border-radius: 20px; padding: 1rem;">', unsafe_allow_html=True)
        text_input = st.text_area(
            "✍️ **Tell me about yourself**", 
            height=150,
            placeholder="Example: I love spending time thinking about deep ideas. I prefer small gatherings with close friends. I'm creative and often get lost in my imagination. I value harmony and try to help others whenever I can...",
            help="Write naturally like you're talking to a friend. Include your thoughts, feelings, and how you act in different situations."
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with st.expander("💡 Tips for better results"):
        st.markdown("""
        - **Be honest** - Write what's really true for you
        - **Be detailed** - 50-200 words works best
        - **Include examples** - How do you handle stress? What makes you happy?
        - **Write naturally** - Don't overthink it, just express yourself
        """)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_clicked = st.button("🔮 Discover My Personality Type", use_container_width=True)
    
    if analyze_clicked:
        if text_input.strip():
            with st.spinner("📖 Reading your words..."):
                time.sleep(0.5)
                mbti_type, top3 = predict(text_input)
                st.success("✨ Analysis complete! Here's what your words reveal...")
                st.balloons()
                confidence = top3[0][1] * 100 if len(top3) > 0 else 70
                show_type_details(mbti_type, confidence)
                
                if len(top3) > 1:
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    st.markdown("### 🔄 Other Types That Match You")
                    st.markdown("Personality is complex - you might also relate to:")
                    alt_cols = st.columns(len(top3)-1)
                    for idx, (type_name, prob) in enumerate(top3[1:3]):
                        with alt_cols[idx]:
                            st.markdown(f"""
                            <div style="text-align: center; background: rgba(0,0,0,0.4); border-radius: 15px; padding: 1rem;">
                                <span class="type-badge">{type_name}</span>
                                <p style="font-size: 0.85rem; margin-top: 0.5rem; color: white;">{TYPE_NAMES.get(type_name, '')}</p>
                                <small style="color: #aaa;">{(prob*100):.1f}% match</small>
                            </div>
                            """, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.markdown("### 📤 Share Your Result")
                share_text = f"I discovered my MBTI type: {mbti_type} - {TYPE_NAMES.get(mbti_type)}! Find your personality for free ✨"
                st.code(f"{share_text}\n\n👉 Try it yourself: [Your App URL]", language="text")
                st.caption("💙 Remember: This is just a fun reflection, not a final label. You're wonderfully unique!")
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("Please write a few sentences about yourself first. The more you write, the better the insight!")
    
    st.markdown("""
    <div style="text-align: center; margin-top: 3rem; color: rgba(255,255,255,0.6); font-size: 0.8rem;">
        <p>✨ Based on your writing style using AI analysis ✨</p>
        <p>🔮 No personality test is 100% accurate - use this as a fun guide to understand yourself better</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()