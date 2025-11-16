"""
Semi-Supervised Learning with Pseudo-Labeling
"""

import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from config import Config
from bert_classifier import BERTTrainer, EmotionDataset



class SemiSupervisedLearning:
    """Pseudo-labeling for semi-supervised learning"""
    
    def __init__(self, trained_model_path):
        self.config = Config()
        self.device = self.config.DEVICE
        
        # Load trained BERT model
        self.trainer = BERTTrainer(n_classes=32)
        self.trainer.load_model(trained_model_path)
        
    def generate_pseudo_labels(self, texts, confidence_threshold=None):
        """Generate pseudo-labels for unlabeled data"""
        if confidence_threshold is None:
            confidence_threshold = self.config.PSEUDO_LABEL_CONFIDENCE
        
        print(f"Generating pseudo-labels with confidence threshold: {confidence_threshold}")
        
        # Create dataset without labels (use dummy labels)
        dataset = EmotionDataset(
            texts,
            np.zeros(len(texts)),
            self.trainer.tokenizer,
            self.config.BERT_MAX_LENGTH
        )
        
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        pseudo_labeled = []
        self.trainer.model.eval()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(loader, desc="Generating pseudo-labels")):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.trainer.model(input_ids, attention_mask)
                probs = torch.softmax(outputs, dim=1)
                max_probs, pseudo_labels = torch.max(probs, dim=1)
                
                # Filter by confidence
                for j in range(len(max_probs)):
                    if max_probs[j].item() >= confidence_threshold:
                        global_idx = batch_idx * 32 + j
                        if global_idx < len(texts):
                            text_val = texts.iloc[global_idx] if hasattr(texts, 'iloc') else texts[global_idx]
                            pseudo_labeled.append({
                                'text': text_val,
                                'pseudo_label': pseudo_labels[j].item(),
                                'confidence': max_probs[j].item()
                            })
        
        print(f"Generated {len(pseudo_labeled)} pseudo-labels from {len(texts)} samples")
        print(f"Retention rate: {len(pseudo_labeled)/len(texts)*100:.2f}%")
        
        return pd.DataFrame(pseudo_labeled)
    
    def create_augmented_dataset(self, train_df, pseudo_df):
        """Combine original training data with pseudo-labeled data"""
        print("\nCreating augmented dataset...")
        
        # Prepare pseudo-labeled data
        pseudo_train = pd.DataFrame({
            'text': pseudo_df['text'],
            'emotion_label': pseudo_df['pseudo_label']
        })
        
        # Combine datasets
        augmented_df = pd.concat([
            train_df[['text', 'emotion_label']],
            pseudo_train
        ], ignore_index=True)
        
        print(f"Original training size: {len(train_df)}")
        print(f"Pseudo-labeled size: {len(pseudo_df)}")
        print(f"Augmented size: {len(augmented_df)}")
        print(f"Increase: {(len(augmented_df) - len(train_df)) / len(train_df) * 100:.2f}%")
        
        return augmented_df
    
    def analyze_pseudo_labels(self, pseudo_df):
        """Analyze pseudo-label distribution and confidence"""
        print("\n" + "="*50)
        print("Pseudo-Label Analysis")
        print("="*50)
        
        # Confidence statistics
        print("\nConfidence Statistics:")
        print(pseudo_df['confidence'].describe())
        
        # Label distribution
        print("\nTop 10 Pseudo-Labeled Emotions:")
        _, id2emotion = self.config.get_emotion_id_mappings()
        pseudo_df['emotion'] = pseudo_df['pseudo_label'].map(id2emotion)
        print(pseudo_df['emotion'].value_counts().head(10))
        
        # Save analysis
        pseudo_df.to_csv(
            os.path.join(self.config.RESULTS_DIR, 'pseudo_labels.csv'),
            index=False
        )


def main():
    """Main semi-supervised learning pipeline"""
    
    # Load preprocessed data
    train_df = pd.read_csv(os.path.join(Config.PROCESSED_DATA_DIR, 'train_preprocessed.csv'))
    val_df = pd.read_csv(os.path.join(Config.PROCESSED_DATA_DIR, 'val_preprocessed.csv'))
    
    # Split validation set (50% for pseudo-labeling, 50% for actual validation)
    val_unlabeled = val_df.sample(frac=0.5, random_state=Config.RANDOM_SEED)
    val_labeled = val_df.drop(val_unlabeled.index)
    
    print(f"Training set: {len(train_df)}")
    print(f"Unlabeled set (for pseudo-labeling): {len(val_unlabeled)}")
    print(f"Validation set: {len(val_labeled)}")
    
    # Initialize semi-supervised learner
    ssl = SemiSupervisedLearning(Config.BERT_MODEL_PATH)
    
    # Generate pseudo-labels
    pseudo_df = ssl.generate_pseudo_labels(
        val_unlabeled['text'],
        confidence_threshold=Config.PSEUDO_LABEL_CONFIDENCE
    )
    
    # Analyze pseudo-labels
    ssl.analyze_pseudo_labels(pseudo_df)
    
    # Create augmented dataset
    augmented_df = ssl.create_augmented_dataset(train_df, pseudo_df)
    
    # Save augmented dataset
    augmented_df.to_csv(
        os.path.join(Config.PROCESSED_DATA_DIR, 'train_augmented.csv'),
        index=False
    )
    val_labeled.to_csv(
        os.path.join(Config.PROCESSED_DATA_DIR, 'val_reduced.csv'),
        index=False
    )
    
    print("\n" + "="*50)
    print("Retraining BERT with Augmented Data")
    print("="*50)
    
    # Retrain BERT with augmented data
    trainer = BERTTrainer(n_classes=32)
    history = trainer.train(augmented_df, val_labeled, epochs=2)
    
    # Save retrained model
    trainer.save_model(os.path.join(Config.MODEL_DIR, 'bert_semisupervised'))
    
    # Save history
    history_df = pd.DataFrame(history)
    history_df.to_csv(
        os.path.join(Config.RESULTS_DIR, 'bert_semisupervised_history.csv'),
        index=False
    )
    
    print("\nSemi-supervised learning complete!")
    print(f"Final validation accuracy: {history[-1]['val_acc']:.4f}")


if __name__ == "__main__":
    main()