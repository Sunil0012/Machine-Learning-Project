"""
Complete Emotional Support Chatbot System
Integrates BERT emotion detection, GPT-2 response generation, and linguistic analysis
"""

import torch
import json
from datetime import datetime
from collections import Counter
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from empath import Empath
from config import Config
from bert_classifier import BERTTrainer
# from gpt2_generator import GPT2ResponseGenerator
from ollama_generator import OllamaResponseGenerator


class EmotionalSupportChatbot:
    """Complete emotional support chatbot"""
    
    def __init__(self, bert_model_path, gpt2_model_path):
        self.config = Config()
        
        # Load emotion mappings
        with open('data/processed/emotion2id.json', 'r') as f:
            self.emotion2id = json.load(f)
        with open('data/processed/id2emotion.json', 'r') as f:
            # Convert string keys to int
            self.id2emotion = {int(k): v for k, v in json.load(f).items()}
        
        # Initialize models
        print("Loading BERT emotion classifier...")
        self.bert_trainer = BERTTrainer(n_classes=32)
        self.bert_trainer.load_model(bert_model_path)
        self.bert_trainer.model.eval()
        
        print("Loading GPT-2 response generator...")
        # self.gpt2_generator = GPT2ResponseGenerator()
        # self.gpt2_generator.load_model(gpt2_model_path)
        self.gpt2_generator = OllamaResponseGenerator(model_name="llama3:8b")


        
        # Initialize analyzers
        self.sia = SentimentIntensityAnalyzer()
        self.lexicon = Empath()
        
        # Conversation history
        self.conversation_history = []
        
        print("Chatbot initialized successfully!")
    
    def detect_emotion(self, text):
        """Detect emotion from text using BERT"""
        # Tokenize
        encoding = self.bert_trainer.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.bert_trainer.device)
        attention_mask = encoding['attention_mask'].to(self.bert_trainer.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.bert_trainer.model(input_ids, attention_mask)
            probs = torch.softmax(outputs, dim=1)
            top_probs, top_idx = torch.topk(probs, k=3)
        
        # Return top 3 emotions
        emotions = []
        for prob, idx in zip(top_probs[0], top_idx[0]):
            emotions.append({
                'emotion': self.id2emotion[idx.item()],
                'confidence': prob.item()
            })
        
        return emotions
    
    def analyze_sentiment(self, text):
        """Analyze sentiment using VADER"""
        if not text:
            return {'compound': 0.0, 'pos': 0.0, 'neg': 0.0, 'neu': 1.0}
        return self.sia.polarity_scores(text)
    
    def analyze_empathy(self, text):
        """Compute empathy score using LIWC-style analysis"""
        if not text:
            return {'empathy_score': 0.0}
        
        analysis = self.lexicon.analyze(text, normalize=True)
        
        empathy_categories = [
            'positive_emotion', 'affection', 'sympathy', 
            'help', 'social', 'communication'
        ]
        
        empathy_score = sum(analysis.get(cat, 0) for cat in empathy_categories)
        
        return {
            'empathy_score': empathy_score,
            'positive_emotion': analysis.get('positive_emotion', 0),
            'help': analysis.get('help', 0),
            'sympathy': analysis.get('sympathy', 0),
            'social': analysis.get('social', 0)
        }
    
    def generate_response(self, user_input, detected_emotion):
        """Generate empathetic response"""
        emotion_name = detected_emotion[0]['emotion']
        
        # Generate response using GPT-2
        response = self.gpt2_generator.generate_response(
            user_input, 
            emotion_name
        )
        
        return response
    
    def chat(self, user_input):
        """Process user input and generate response"""
        if not user_input or not user_input.strip():
            return {
                'response': "I'm here to listen. How are you feeling today?",
                'emotion': None,
                'confidence': 0.0
            }
        
        # Detect emotion
        emotions = self.detect_emotion(user_input)
        
        # Analyze sentiment
        user_sentiment = self.analyze_sentiment(user_input)
        
        # Generate response
        response = self.generate_response(user_input, emotions)
        
        # Analyze response quality
        response_empathy = self.analyze_empathy(response)
        response_sentiment = self.analyze_sentiment(response)
        
        # Store interaction
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input,
            'bot_response': response,
            'detected_emotion': emotions[0]['emotion'],
            'emotion_confidence': emotions[0]['confidence'],
            'top_3_emotions': emotions,
            'user_sentiment': user_sentiment['compound'],
            'response_empathy_score': response_empathy['empathy_score'],
            'response_sentiment': response_sentiment['compound']
        }
        
        self.conversation_history.append(interaction)
        
        return {
            'response': response,
            'emotion': emotions[0]['emotion'],
            'confidence': emotions[0]['confidence'],
            'top_3_emotions': emotions,
            'analysis': interaction
        }
    
    def get_conversation_summary(self):
        """Get summary of conversation"""
        if not self.conversation_history:
            return None
        
        emotions = [h['detected_emotion'] for h in self.conversation_history]
        
        return {
            'total_interactions': len(self.conversation_history),
            'emotions_detected': dict(Counter(emotions)),
            'avg_confidence': np.mean([h['emotion_confidence'] for h in self.conversation_history]),
            'avg_user_sentiment': np.mean([h['user_sentiment'] for h in self.conversation_history]),
            'avg_response_empathy': np.mean([h['response_empathy_score'] for h in self.conversation_history]),
            'avg_response_sentiment': np.mean([h['response_sentiment'] for h in self.conversation_history]),
            'conversation_history': self.conversation_history
        }
    
    def reset_conversation(self):
        """Reset conversation history"""
        self.conversation_history = []
    
    def save_conversation(self, filepath):
        """Save conversation to file"""
        summary = self.get_conversation_summary()
        if summary:
            with open(filepath, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"Conversation saved to {filepath}")


def interactive_chat():
    """Interactive chat session"""
    
    # Initialize chatbot
    print("\nInitializing Emotional Support Chatbot...")
    print("="*50)
    
    chatbot = EmotionalSupportChatbot(
        bert_model_path=Config.BERT_MODEL_PATH,
        gpt2_model_path=Config.GPT2_MODEL_PATH
    )
    
    print("\n" + "="*50)
    print("Emotional Support Chatbot Ready!")
    print("="*50)
    print("Type 'quit' to exit")
    print("Type 'summary' to see conversation statistics")
    print("Type 'reset' to start a new conversation")
    print("="*50 + "\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() == 'quit':
            # Save conversation
            if chatbot.conversation_history:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                chatbot.save_conversation(f'results/conversation_{timestamp}.json')
            print("\nGoodbye! Take care of yourself. 💙")
            break
        
        elif user_input.lower() == 'summary':
            summary = chatbot.get_conversation_summary()
            if summary:
                print("\n" + "="*50)
                print("Conversation Summary")
                print("="*50)
                print(f"Total Interactions: {summary['total_interactions']}")
                print(f"Emotions Detected: {summary['emotions_detected']}")
                print(f"Avg Confidence: {summary['avg_confidence']:.2f}")
                print(f"Avg User Sentiment: {summary['avg_user_sentiment']:.2f}")
                print(f"Avg Response Empathy: {summary['avg_response_empathy']:.2f}")
                print("="*50 + "\n")
            else:
                print("No conversation history yet.\n")
            continue
        
        elif user_input.lower() == 'reset':
            chatbot.reset_conversation()
            print("Conversation reset. Starting fresh!\n")
            continue
        
        elif not user_input:
            continue
        
        # Get response
        result = chatbot.chat(user_input)
        
        print(f"\nBot: {result['response']}")
        print(f"[Detected: {result['emotion']} ({result['confidence']:.2f})]\n")


def batch_test():
    """Test chatbot with predefined examples"""
    
    chatbot = EmotionalSupportChatbot(
        bert_model_path=Config.BERT_MODEL_PATH,
        gpt2_model_path=Config.GPT2_MODEL_PATH
    )
    
    test_cases = [
        "I'm feeling really anxious about my exam tomorrow",
        "I feel so lonely and sad today",
        "I'm so happy! I got the job I applied for!",
        "I'm really angry about what happened at work",
        "I'm scared about the future",
        "I'm proud of myself for completing this project",
        "I feel disappointed with the results",
        "I'm grateful for all the support I've received"
    ]
    
    print("\n" + "="*50)
    print("Batch Testing Chatbot")
    print("="*50 + "\n")
    
    for i, test_input in enumerate(test_cases, 1):
        print(f"Test {i}/{len(test_cases)}")
        print(f"User: {test_input}")
        
        result = chatbot.chat(test_input)
        
        print(f"Bot: {result['response']}")
        print(f"Emotion: {result['emotion']} (confidence: {result['confidence']:.3f})")
        top3 = ", ".join([f"{e['emotion']}({e['confidence']:.2f})" for e in result['top_3_emotions']])
        print(f"Top 3: {top3}")

        print("-" * 50 + "\n")
    
    # Show summary
    summary = chatbot.get_conversation_summary()
    print("="*50)
    print("Test Summary")
    print("="*50)
    print(f"Total Tests: {summary['total_interactions']}")
    print(f"Emotions: {summary['emotions_detected']}")
    print(f"Avg Empathy Score: {summary['avg_response_empathy']:.3f}")
    print("="*50)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        batch_test()
    else:
        interactive_chat()