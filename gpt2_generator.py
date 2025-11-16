"""
GPT-2 Response Generator for Empathetic Responses
"""


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer

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

class ResponseDataset(Dataset):
    """Dataset for response generation"""
    
    def __init__(self, contexts, responses, tokenizer, max_length=256):
        self.contexts = contexts
        self.responses = responses
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.contexts)
    
    def __getitem__(self, idx):
        context = str(self.contexts.iloc[idx]) if hasattr(self.contexts, 'iloc') else str(self.contexts[idx])
        response = str(self.responses.iloc[idx]) if hasattr(self.responses, 'iloc') else str(self.responses[idx])
        
        # Format: "Context: {context} Response: {response}"
        full_text = f"Context: {context} Response: {response}"
        
        encoding = self.tokenizer.encode_plus(
            full_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }


class GPT2ResponseGenerator:
    """GPT-2 based response generator"""
    
    def __init__(self):
        self.config = Config()
        self.device = self.config.DEVICE
        
        # Initialize model and tokenizer
        self.model = GPT2LMHeadModel.from_pretrained(self.config.GPT2_MODEL_NAME).to(self.device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.config.GPT2_MODEL_NAME)
        
        # Set pad token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.eos_token_id
        
        print(f"Using device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def create_data_loader(self, df, batch_size=None):
        """Create data loader"""
        if batch_size is None:
            batch_size = self.config.GPT2_BATCH_SIZE
        
        # Use prompt as context and utterance as response
        dataset = ResponseDataset(
            df['prompt'].fillna(''),
            df['utterance'].fillna(''),
            self.tokenizer,
            self.config.GPT2_MAX_LENGTH
        )
        
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return loader
    
    def train_epoch(self, data_loader, optimizer):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(data_loader, desc='Training')
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            optimizer.zero_grad()
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(data_loader)
        perplexity = np.exp(avg_loss)
        
        return avg_loss, perplexity
    
    def train(self, train_df, epochs=None, save_best=True):
        """Full training pipeline"""
        if epochs is None:
            epochs = self.config.GPT2_EPOCHS
        
        print(f"\nTraining GPT-2 for {epochs} epochs...")
        
        # Create data loader
        train_loader = self.create_data_loader(train_df)
        
        # Setup optimizer
        optimizer = AdamW(self.model.parameters(), lr=self.config.GPT2_LEARNING_RATE)
        
        history = []
        best_loss = float('inf')
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            # Train
            train_loss, perplexity = self.train_epoch(train_loader, optimizer)
            
            print(f"Train Loss: {train_loss:.4f}, Perplexity: {perplexity:.2f}")
            
            history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'perplexity': perplexity
            })
            
            # Save best model
            if save_best and train_loss < best_loss:
                best_loss = train_loss
                self.save_model(self.config.GPT2_MODEL_PATH)
                print(f"✓ Best model saved with loss: {best_loss:.4f}")
        
        return history
    
    def generate_response(self, context, emotion=None, max_length=150):
        """Generate empathetic response"""
        self.model.eval()
        
        # Add empathy template if emotion provided
        empathy_start = ""
        if emotion and emotion in self.config.EMPATHY_TEMPLATES:
            empathy_start = self.config.EMPATHY_TEMPLATES[emotion]
        
        # Create prompt
        if emotion:
            prompt = f"Context: {context} Emotion: {emotion} Response: {empathy_start}"
        else:
            prompt = f"Context: {context} Response: {empathy_start}"
        
        # Encode
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        # Generate
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_length=max_length,
                min_length=20,
                temperature=self.config.GPT2_TEMPERATURE,
                top_k=self.config.GPT2_TOP_K,
                top_p=self.config.GPT2_TOP_P,
                do_sample=True,
                no_repeat_ngram_size=3,
                pad_token_id=self.tokenizer.eos_token_id,
                early_stopping=True
            )
        
        # Decode
        generated = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Extract response part
        if "Response:" in generated:
            response = generated.split("Response:")[-1].strip()
        else:
            response = generated
        
        return response
    
    def save_model(self, save_path):
        """Save model and tokenizer"""
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"Model saved to {save_path}")
    
    def load_model(self, load_path):
        """Load model"""
        self.model = GPT2LMHeadModel.from_pretrained(load_path).to(self.device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(load_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"Model loaded from {load_path}")


def test_generation(generator):
    """Test response generation with examples"""
    print("\n" + "="*50)
    print("Testing Response Generation")
    print("="*50)
    
    test_cases = [
        ("I'm feeling really anxious about my exam tomorrow", "anxious"),
        ("I feel so lonely and sad today", "sad"),
        ("I'm so happy! I got the job I applied for!", "joyful"),
        ("I'm really angry about what happened", "angry"),
        ("I'm scared of the future", "afraid")
    ]
    
    for context, emotion in test_cases:
        print(f"\nContext: {context}")
        print(f"Emotion: {emotion}")
        response = generator.generate_response(context, emotion)
        print(f"Response: {response}")
        print("-" * 50)


def main():
    """Main training pipeline for GPT-2"""
    
    # Load data
    train_df = pd.read_csv(os.path.join(Config.PROCESSED_DATA_DIR, 'train_preprocessed.csv'))
    
    print(f"Training size: {len(train_df)}")
    
    # Initialize generator
    generator = GPT2ResponseGenerator()
    
    # Train
    history = generator.train(train_df, epochs=Config.GPT2_EPOCHS)
    
    # Save history
    history_df = pd.DataFrame(history)
    history_df.to_csv(
        os.path.join(Config.RESULTS_DIR, 'gpt2_training_history.csv'),
        index=False
    )
    
    print("\n" + "="*50)
    print("GPT-2 Training Complete!")
    print("="*50)
    
    # Test generation
    test_generation(generator)


if __name__ == "__main__":
    main()