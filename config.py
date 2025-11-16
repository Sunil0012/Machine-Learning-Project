"""
Configuration file for Emotional Support Chatbot
Author: Khethavath Sunil Naik
Project: Enhancing Emotional Support Chatbots with ML
"""

import os
import torch

import os

# Main project folders
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Required for evaluation.py
OUTPUT_DIR = r"C:\Users\TL1\Desktop\Machine Learning\models"

# Optional but recommended for Ollama based generator
USE_OLLAMA_CLI = False
OLLAMA_API_URL = "http://localhost:11434/api/generate"



class Config:
    """Global configuration for the project"""
    
    # Project Settings
    PROJECT_NAME = "Emotional_Support_Chatbot"
    RANDOM_SEED = 42
    
    # Device Configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data Paths
    DATA_DIR = "data"
    RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
    
    # Model Paths
    MODEL_DIR = "models"
    BERT_MODEL_PATH = os.path.join(MODEL_DIR, "bert_emotion_classifier")
    GPT2_MODEL_PATH = os.path.join(MODEL_DIR, "gpt2_response_generator")
    CLASSICAL_ML_DIR = os.path.join(MODEL_DIR, "classical_ml")
    
    # Results Paths
    RESULTS_DIR = "results"
    PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
    LOGS_DIR = os.path.join(RESULTS_DIR, "logs")
    
    # Dataset URLs
    EMPATHETIC_DIALOGUES_URL = "https://github.com/facebookresearch/EmpatheticDialogues"
    
    # 32 Emotion Categories
    EMOTIONS = [
        'afraid', 'angry', 'annoyed', 'anticipating', 'anxious', 'apprehensive',
        'ashamed', 'caring', 'confident', 'content', 'devastated', 'disappointed',
        'disgusted', 'embarrassed', 'excited', 'faithful', 'furious', 'grateful',
        'guilty', 'hopeful', 'impressed', 'jealous', 'joyful', 'lonely', 'nostalgic',
        'prepared', 'proud', 'sad', 'sentimental', 'surprised', 'terrified', 'trusting'
    ]
    
    # BERT Configuration
    BERT_MODEL_NAME = "bert-base-uncased"
    BERT_MAX_LENGTH = 128
    BERT_BATCH_SIZE = 16
    BERT_LEARNING_RATE = 2e-5
    BERT_EPOCHS = 3
    BERT_DROPOUT = 0.3
    
    # GPT-2 Configuration
    GPT2_MODEL_NAME = "gpt2"
    GPT2_MAX_LENGTH = 256
    GPT2_BATCH_SIZE = 8
    GPT2_LEARNING_RATE = 5e-5
    GPT2_EPOCHS = 2
    GPT2_TEMPERATURE = 0.7
    GPT2_TOP_K = 50
    GPT2_TOP_P = 0.95
    
    # Classical ML Configuration
    TFIDF_MAX_FEATURES = 3000
    TFIDF_NGRAM_RANGE = (1, 2)
    TFIDF_MIN_DF = 3
    TFIDF_MAX_DF = 0.9
    
    # Semi-Supervised Learning
    PSEUDO_LABEL_CONFIDENCE = 0.85
    UNLABELED_RATIO = 0.5  # 50% of validation data
    
    # Empathy Templates
    EMPATHY_TEMPLATES = {
        'sad': "I'm sorry you're feeling down. ",
        'anxious': "That sounds stressful. ",
        'angry': "I hear that you're upset. ",
        'lonely': "You're not alone. ",
        'joyful': "That's wonderful! ",
        'grateful': "Gratitude is beautiful. ",
        'afraid': "It's okay to feel scared. ",
        'excited': "That's so exciting! ",
        'devastated': "I'm so sorry you're going through this. ",
        'proud': "You should be proud of yourself! ",
        'disappointed': "Disappointment is tough to handle. ",
        'guilty': "It's okay to make mistakes. ",
        'ashamed': "You're being too hard on yourself. ",
        'jealous': "Those feelings are valid. ",
        'annoyed': "I understand your frustration. ",
        'furious': "Take a deep breath. ",
        'terrified': "Fear can be overwhelming. ",
        'surprised': "Wow, that's unexpected! ",
        'nostalgic': "Memories can be bittersweet. ",
        'hopeful': "Hope is a beautiful thing. ",
        'caring': "Your compassion shows. ",
        'confident': "Your confidence is inspiring! ",
        'content': "Peace of mind is precious. ",
        'trusting': "Trust is valuable. ",
        'faithful': "Your faith is strong. ",
        'impressed': "That's impressive! ",
        'anticipating': "Anticipation can be exciting! ",
        'apprehensive': "Uncertainty can be unsettling. ",
        'embarrassed': "We all have awkward moments. ",
        'disgusted': "That's understandably upsetting. ",
        'prepared': "Preparation shows dedication! ",
        'sentimental': "Emotions run deep sometimes. "
    }
    
    # LIWC-Style Categories
    LIWC_CATEGORIES = {
        'emotion': ['positive_emotion', 'negative_emotion', 'anxiety', 'anger', 'sadness', 'fear', 'disgust', 'joy'],
        'social': ['social', 'family', 'friends', 'communication', 'help', 'sympathy'],
        'cognitive': ['thinking', 'insight', 'causation', 'discrepancy'],
        'personal': ['achievement', 'power']
    }
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        dirs = [
            cls.DATA_DIR, cls.RAW_DATA_DIR, cls.PROCESSED_DATA_DIR,
            cls.MODEL_DIR, cls.BERT_MODEL_PATH, cls.GPT2_MODEL_PATH,
            cls.CLASSICAL_ML_DIR, cls.RESULTS_DIR, cls.PLOTS_DIR, cls.LOGS_DIR
        ]
        for directory in dirs:
            os.makedirs(directory, exist_ok=True)
            
    @classmethod
    def get_emotion_id_mappings(cls):
        """Get emotion to ID and ID to emotion mappings"""
        emotion2id = {emotion: idx for idx, emotion in enumerate(sorted(cls.EMOTIONS))}
        id2emotion = {idx: emotion for emotion, idx in emotion2id.items()}
        return emotion2id, id2emotion