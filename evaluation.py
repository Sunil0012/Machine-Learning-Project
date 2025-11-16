"""
Comprehensive Evaluation Framework
Includes BLEU, ROUGE, BERTScore, and custom empathy metrics
"""

import pandas as pd
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from empath import Empath
from tqdm import tqdm
import os
from config import Config
from chatbot_system import EmotionalSupportChatbot

class ChatbotEvaluator:
    """Evaluate chatbot performance"""
    
    def __init__(self, chatbot):
        self.chatbot = chatbot
        self.config = Config()
        self.sia = SentimentIntensityAnalyzer()
        self.lexicon = Empath()
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
    def compute_bleu(self, reference, candidate):
        """Compute BLEU scores"""
        reference_tokens = [reference.split()]
        candidate_tokens = candidate.split()
        
        smoothie = SmoothingFunction().method4
        
        bleu1 = sentence_bleu(reference_tokens, candidate_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothie)
        bleu2 = sentence_bleu(reference_tokens, candidate_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
        bleu3 = sentence_bleu(reference_tokens, candidate_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie)
        bleu4 = sentence_bleu(reference_tokens, candidate_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
        
        return {
            'bleu1': bleu1,
            'bleu2': bleu2,
            'bleu3': bleu3,
            'bleu4': bleu4
        }
    
    def compute_rouge(self, reference, candidate):
        """Compute ROUGE scores"""
        scores = self.rouge_scorer.score(reference, candidate)
        
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }
    
    def compute_bert_score(self, references, candidates):
        """Compute BERTScore (batch operation)"""
        P, R, F1 = bert_score(candidates, references, lang='en', verbose=False)
        
        return {
            'precision': P.mean().item(),
            'recall': R.mean().item(),
            'f1': F1.mean().item()
        }
    
    def compute_empathy_score(self, text):
        """Compute empathy score"""
        analysis = self.lexicon.analyze(text, normalize=True)
        
        empathy_categories = [
            'positive_emotion', 'affection', 'sympathy',
            'help', 'social', 'communication'
        ]
        
        return sum(analysis.get(cat, 0) for cat in empathy_categories)
    
    def evaluate_on_test_set(self, test_df, num_samples=100):
        """Evaluate chatbot on test set"""
        print(f"\nEvaluating on {num_samples} samples from test set...")
        
        # Sample test data
        if len(test_df) > num_samples:
            test_sample = test_df.sample(n=num_samples, random_state=self.config.RANDOM_SEED)
        else:
            test_sample = test_df
        
        results = []
        
        for idx, row in tqdm(test_sample.iterrows(), total=len(test_sample), desc="Evaluating"):
            context = str(row['prompt'])
            reference_response = str(row['utterance'])
            
            # Get chatbot response
            chat_result = self.chatbot.chat(context)
            generated_response = chat_result['response']
            
            # Compute metrics
            bleu_scores = self.compute_bleu(reference_response, generated_response)
            rouge_scores = self.compute_rouge(reference_response, generated_response)
            
            # Empathy score
            empathy_score = self.compute_empathy_score(generated_response)
            
            # Sentiment
            ref_sentiment = self.sia.polarity_scores(reference_response)['compound']
            gen_sentiment = self.sia.polarity_scores(generated_response)['compound']
            
            results.append({
                'context': context,
                'reference': reference_response,
                'generated': generated_response,
                'detected_emotion': chat_result['emotion'],
                'emotion_confidence': chat_result['confidence'],
                **bleu_scores,
                **rouge_scores,
                'empathy_score': empathy_score,
                'ref_sentiment': ref_sentiment,
                'gen_sentiment': gen_sentiment
            })
        
        results_df = pd.DataFrame(results)
        
        # Compute BERTScore for all samples
        print("Computing BERTScore...")
        bert_scores = self.compute_bert_score(
            results_df['reference'].tolist(),
            results_df['generated'].tolist()
        )
        results_df['bertscore_f1'] = bert_scores['f1']
        
        return results_df
    
    def print_evaluation_summary(self, results_df):
        """Print evaluation summary"""
        print("\n" + "="*50)
        print("Evaluation Summary")
        print("="*50)
        
        metrics = {
            'BLEU-1': results_df['bleu1'].mean(),
            'BLEU-2': results_df['bleu2'].mean(),
            'BLEU-3': results_df['bleu3'].mean(),
            'BLEU-4': results_df['bleu4'].mean(),
            'ROUGE-1': results_df['rouge1'].mean(),
            'ROUGE-2': results_df['rouge2'].mean(),
            'ROUGE-L': results_df['rougeL'].mean(),
            'BERTScore F1': results_df['bertscore_f1'].mean(),
            'Empathy Score': results_df['empathy_score'].mean()
        }
        
        for metric, value in metrics.items():
            print(f"{metric:20s}: {value:.4f}")
        
        print("="*50)
        
        return metrics


class BaselineChatbot:
    """Simple rule-based baseline for comparison"""
    
    def __init__(self):
        self.emotion_keywords = {
            'sad': [ 'depressed', 'down', 'miserable'],
            'happy': ['happy', 'great', 'wonderful'],
            'angry': ['angry', 'annoyed', 'irritated'],
            'anxious': ['anxious', 'worried','concerned'],
            'afraid': ['afraid', 'scared', 'terrified']
        }
        
        self.responses = {
            'sad': "Things will get better.",
            'happy': "That's great! I'm happy for you.",
            'angry': "I understand.Try to calm down.",
            'anxious': "Don't worry too much. It will be okay.",
            'afraid': "There's nothing to be afraid of.",
            'neutral': "Tell me more about how you feel."
        }
    
    def detect_emotion(self, text):
        """Simple keyword-based emotion detection"""
        text_lower = text.lower()
        
        for emotion, keywords in self.emotion_keywords.items():
            if any(kw in text_lower for kw in keywords):
                return emotion, 0.7
        
        return 'neutral', 0.5
    
    def chat(self, user_input):
        """Generate rule-based response"""
        emotion, confidence = self.detect_emotion(user_input)
        response = self.responses.get(emotion, self.responses['neutral'])
        
        return {
            'response': response,
            'emotion': emotion,
            'confidence': confidence
        }


def compare_with_baseline(chatbot, baseline, test_df, num_samples=100):
    """Compare chatbot with baseline"""
    print("\n" + "="*50)
    print("Comparing with Baseline")
    print("="*50)
    
    # Sample test data
    if len(test_df) > num_samples:
        test_sample = test_df.sample(n=num_samples, random_state=Config.RANDOM_SEED)
    else:
        test_sample = test_df
    
    evaluator = ChatbotEvaluator(chatbot)
    sia = SentimentIntensityAnalyzer()
    lexicon = Empath()
    
    our_results = []
    baseline_results = []
    
    for idx, row in tqdm(test_sample.iterrows(), total=len(test_sample), desc="Comparing"):
        context = str(row['prompt'])
        reference = str(row['utterance'])
        
        # Our chatbot
        our_result = chatbot.chat(context)
        our_response = our_result['response']
        
        # Baseline
        baseline_result = baseline.chat(context)
        baseline_response = baseline_result['response']
        
        # Compute metrics for our chatbot
        our_bleu = evaluator.compute_bleu(reference, our_response)
        our_rouge = evaluator.compute_rouge(reference, our_response)
        our_empathy = evaluator.compute_empathy_score(our_response)
        
        # Compute metrics for baseline
        baseline_bleu = evaluator.compute_bleu(reference, baseline_response)
        baseline_rouge = evaluator.compute_rouge(reference, baseline_response)
        baseline_empathy = evaluator.compute_empathy_score(baseline_response)
        
        our_results.append({
            'bleu1': our_bleu['bleu1'],
            'rouge1': our_rouge['rouge1'],
            'empathy': our_empathy
        })
        
        baseline_results.append({
            'bleu1': baseline_bleu['bleu1'],
            'rouge1': baseline_rouge['rouge1'],
            'empathy': baseline_empathy
        })
    
    # Compute averages
    our_avg = pd.DataFrame(our_results).mean()
    baseline_avg = pd.DataFrame(baseline_results).mean()
    
    # Print comparison
    print("\n" + "="*50)
    print("Performance Comparison")
    print("="*50)
    print(f"{'Metric':<20} {'Our Chatbot':<15} {'Baseline':<15} {'Improvement':<15}")
    print("-"*65)
    
    for metric in ['bleu1', 'rouge1', 'empathy']:
        improvement = ((our_avg[metric] - baseline_avg[metric]) / baseline_avg[metric] * 100)
        print(f"{metric.upper():<20} {our_avg[metric]:<15.4f} {baseline_avg[metric]:<15.4f} {improvement:<15.1f}%")
    
    print("="*50)
    
    return our_avg, baseline_avg


def main():
    """Main evaluation pipeline"""
    
    # Load test data
    test_df = pd.read_csv(os.path.join(Config.PROCESSED_DATA_DIR, 'test_preprocessed.csv'))
    
    print(f"Test set size: {len(test_df)}")
    
    # Initialize chatbot
    print("\nInitializing chatbot...")
    chatbot = EmotionalSupportChatbot(
        bert_model_path=Config.BERT_MODEL_PATH,
        gpt2_model_path=Config.GPT2_MODEL_PATH
    )
    
    # Evaluate
    evaluator = ChatbotEvaluator(chatbot)
    results_df = evaluator.evaluate_on_test_set(test_df, num_samples=100)
    
    # Print summary
    metrics = evaluator.print_evaluation_summary(results_df)
    
    # Save results
    results_df.to_csv(os.path.join(Config.RESULTS_DIR, 'evaluation_results.csv'), index=False)
    
    # Save metrics
    pd.DataFrame([metrics]).to_csv(
        os.path.join(Config.RESULTS_DIR, 'evaluation_metrics.csv'),
        index=False
    )
    
    # Compare with baseline
    baseline = BaselineChatbot()
    our_avg, baseline_avg = compare_with_baseline(chatbot, baseline, test_df, num_samples=50)
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()