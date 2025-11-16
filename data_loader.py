# Corrected DataLoader with fixed get_statistics and duplicate method removed

import pandas as pd
import numpy as np
import re
import json
import os
from sklearn.model_selection import train_test_split
from textblob import TextBlob
from config import Config
import spacy
from tqdm import tqdm

class DataLoader:
    """Load and preprocess Empathetic Dialogues dataset"""
    
    def __init__(self):
        self.config = Config()
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            print("Downloading spacy model...")
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
    
    def load_csv_data(self, train_path, val_path, test_path):
        """Load data from CSV files"""
        print("Loading datasets...")
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
        
        print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        return train_df, val_df, test_df
    
    def clean_text(self, text):
        """Clean text data"""
        if not isinstance(text, str): return ""

        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'\@\w+|\#', '', text)
        text = re.sub(r'[^a-zA-Z\s\.\,\!\?]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def lemmatize_text(self, text):
        if not text: return ""
        doc = self.nlp(text)
        lemmas = [token.lemma_ for token in doc 
                  if not token.is_stop and token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']]
        return ' '.join(lemmas)

    def extract_sentiment(self, text):
        if not isinstance(text, str) or not text:
            return 0.0, 0.0
        blob = TextBlob(text)
        return blob.sentiment.polarity, blob.sentiment.subjectivity

    def preprocess_dataframe(self, df):
        print("Preprocessing data...")

        df['prompt'] = df['prompt'].fillna('')
        df['utterance'] = df['utterance'].fillna('')
        df['context'] = df['context'].fillna('neutral')

        df['text'] = df['prompt'] + ' ' + df['utterance']

        tqdm.pandas(desc="Cleaning text")
        df['text_clean'] = df['text'].progress_apply(self.clean_text)

        print("Lemmatizing text (this may take a while)...")
        tqdm.pandas(desc="Lemmatizing")
        df['text_lemmatized'] = df['text_clean'].progress_apply(self.lemmatize_text)

        tqdm.pandas(desc="Extracting sentiment")
        df[['polarity', 'subjectivity']] = df['text'].progress_apply(
            lambda x: pd.Series(self.extract_sentiment(x))
        )

        df['text_length'] = df['text'].astype(str).apply(lambda x: len(x.split()))

        emotion2id, id2emotion = self.config.get_emotion_id_mappings()
        df['emotion_label'] = df['context'].map(emotion2id)
        df = df.dropna(subset=['emotion_label'])
        df['emotion_label'] = df['emotion_label'].astype(int)

        return df

    def save_mappings(self, save_dir):
        emotion2id, id2emotion = self.config.get_emotion_id_mappings()
        with open(os.path.join(save_dir, 'emotion2id.json'), 'w') as f:
            json.dump(emotion2id, f, indent=2)
        with open(os.path.join(save_dir, 'id2emotion.json'), 'w') as f:
            json.dump(id2emotion, f, indent=2)

    def save_processed_data(self, train_df, val_df, test_df, save_dir):
        print(f"Saving processed data to {save_dir}...")
        train_df.to_csv(os.path.join(save_dir, 'train_preprocessed.csv'), index=False)
        val_df.to_csv(os.path.join(save_dir, 'val_preprocessed.csv'), index=False)
        test_df.to_csv(os.path.join(save_dir, 'test_preprocessed.csv'), index=False)
        self.save_mappings(save_dir)
        print("Data saved successfully!")

    def get_statistics(self, df, name="Dataset"):
        """Print dataset statistics safely (auto-create text_length)"""
        print(f"{'='*50}")
        print(f"{name} Statistics")
        print(f"{'='*50}")

        # Ensure text_length exists
        if 'text_length' not in df.columns:
            if 'text' in df.columns:
                df['text_length'] = df['text'].astype(str).apply(lambda x: len(x.split()))
            elif 'utterance' in df.columns:
                df['text_length'] = df['utterance'].astype(str).apply(lambda x: len(x.split()))
            else:
                print("Warning: No text column found, skipping text length stats.")
                df['text_length'] = 0

        print("Text length statistics:")
        print(df['text_length'].describe())

        print(f"\nTotal samples: {len(df)}")
        print(f"Unique emotions: {df['context'].nunique()}")

        print(f"\nEmotion distribution:")
        print(df['context'].value_counts())

        print(f"\nMissing values:")
        print(df.isnull().sum())


def main():
    Config.create_directories()
    loader = DataLoader()

    train_path = r"C:\Users\TL1\Desktop\Machine Learning\empathetic_dialogues_train.csv"
    val_path = r"C:\Users\TL1\Desktop\Machine Learning\empathetic_dialogues_validation.csv"
    test_path = r"C:\Users\TL1\Desktop\Machine Learning\empathetic_dialogues_test.csv"

    if not all(os.path.exists(p) for p in [train_path, val_path, test_path]):
        print("Error: Data files not found!")
        return

    train_df, val_df, test_df = loader.load_csv_data(train_path, val_path, test_path)

    # Stats before preprocess
    loader.get_statistics(train_df, "Training Set (Raw)")

    train_df = loader.preprocess_dataframe(train_df)
    val_df = loader.preprocess_dataframe(val_df)
    test_df = loader.preprocess_dataframe(test_df)

    loader.save_processed_data(train_df, val_df, test_df, Config.PROCESSED_DATA_DIR)

    print("\nPreprocessing complete!")

if __name__ == "__main__":
    main()
