"""
BERT-based Emotion Classifier
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW

from transformers import BertModel, BertTokenizer
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from tqdm import tqdm
import os
from config import Config

class EmotionDataset(Dataset):
    """Dataset for BERT emotion classification"""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx]) if hasattr(self.texts, 'iloc') else str(self.texts[idx])
        label = self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


class BERTEmotionClassifier(nn.Module):
    """BERT model for emotion classification"""
    
    def __init__(self, n_classes, dropout=0.3):
        super(BERTEmotionClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(Config.BERT_MODEL_NAME)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, n_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [CLS] token
        output = self.dropout(pooled_output)
        return self.classifier(output)


class BERTTrainer:
    """Trainer for BERT emotion classifier"""
    
    def __init__(self, n_classes=32):
        self.config = Config()
        self.device = self.config.DEVICE
        self.n_classes = n_classes
        
        # Initialize model and tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(self.config.BERT_MODEL_NAME)
        self.model = BERTEmotionClassifier(n_classes, self.config.BERT_DROPOUT).to(self.device)
        
        print(f"Using device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def create_data_loaders(self, train_df, val_df, batch_size=None):
        """Create data loaders"""
        if batch_size is None:
            batch_size = self.config.BERT_BATCH_SIZE
        
        train_dataset = EmotionDataset(
            train_df['text'].values,
            train_df['emotion_label'].values,
            self.tokenizer,
            self.config.BERT_MAX_LENGTH
        )
        
        val_dataset = EmotionDataset(
            val_df['text'].values,
            val_df['emotion_label'].values,
            self.tokenizer,
            self.config.BERT_MAX_LENGTH
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def train_epoch(self, data_loader, optimizer, criterion):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(data_loader, desc='Training')
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            optimizer.zero_grad()
            
            outputs = self.model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            progress_bar.set_postfix({'loss': loss.item(), 'acc': correct/total})
        
        return total_loss / len(data_loader), correct / total
    
    def evaluate(self, data_loader):
        """Evaluate model"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc='Evaluating'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        
        return {
            'loss': total_loss / len(data_loader),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def train(self, train_df, val_df, epochs=None, save_best=True):
        """Full training pipeline"""
        if epochs is None:
            epochs = self.config.BERT_EPOCHS
        
        print(f"\nTraining BERT for {epochs} epochs...")
        
        # Create data loaders
        train_loader, val_loader = self.create_data_loaders(train_df, val_df)
        
        # Setup optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=self.config.BERT_LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()
        
        best_val_acc = 0
        history = []
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            
            # Evaluate
            val_metrics = self.evaluate(val_loader)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
            print(f"Val F1: {val_metrics['f1_score']:.4f}")
            
            history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_metrics['loss'],
                'val_acc': val_metrics['accuracy'],
                'val_f1': val_metrics['f1_score']
            })
            
            # Save best model
            if save_best and val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                self.save_model(self.config.BERT_MODEL_PATH)
                print(f"✓ Best model saved with accuracy: {best_val_acc:.4f}")
        
        return history
    
    def save_model(self, save_path):
        """Save model and tokenizer"""
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(save_path, 'model.pt'))
        self.tokenizer.save_pretrained(save_path)
    
    def load_model(self, load_path):
        """Load model"""
        self.model.load_state_dict(torch.load(os.path.join(load_path, 'model.pt')))
        self.tokenizer = BertTokenizer.from_pretrained(load_path)
        self.model.to(self.device)
        print(f"Model loaded from {load_path}")


def main():
    """Main training pipeline for BERT"""
    
    # Load data
    train_df = pd.read_csv(os.path.join(Config.PROCESSED_DATA_DIR, 'train_preprocessed.csv'))
    val_df = pd.read_csv(os.path.join(Config.PROCESSED_DATA_DIR, 'val_preprocessed.csv'))
    
    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")
    print(f"Number of emotion classes: {train_df['emotion_label'].nunique()}")
    
    # Initialize trainer
    trainer = BERTTrainer(n_classes=32)
    
    # Train
    history = trainer.train(train_df, val_df, epochs=Config.BERT_EPOCHS)
    
    # Save history
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(Config.RESULTS_DIR, 'bert_training_history.csv'), index=False)
    
    print("\n" + "="*50)
    print("BERT Training Complete!")
    print("="*50)
    print(f"Best Validation Accuracy: {max([h['val_acc'] for h in history]):.4f}")


if __name__ == "__main__":
    main()