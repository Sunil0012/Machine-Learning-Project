"""
Classical Machine Learning Models: Logistic Regression, Random Forest, SVM
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import joblib
import os
import time
from config import Config

class ClassicalMLModels:
    """Train and evaluate classical ML models"""
    
    def __init__(self):
        self.config = Config()
        self.vectorizer = None
        self.models = {}
        
    def create_tfidf_features(self, train_texts, val_texts, test_texts):
        """Create TF-IDF features"""
        print("Creating TF-IDF features...")
        
        self.vectorizer = TfidfVectorizer(
            max_features=self.config.TFIDF_MAX_FEATURES,
            ngram_range=self.config.TFIDF_NGRAM_RANGE,
            min_df=self.config.TFIDF_MIN_DF,
            max_df=self.config.TFIDF_MAX_DF,
            sublinear_tf=True
        )
        
        X_train = self.vectorizer.fit_transform(train_texts)
        X_val = self.vectorizer.transform(val_texts)
        X_test = self.vectorizer.transform(test_texts)
        
        print(f"Feature matrix shape: {X_train.shape}")
        
        return X_train, X_val, X_test
    
    def train_logistic_regression(self, X_train, y_train, X_val, y_val):
        """Train Logistic Regression"""
        print("\n" + "="*50)
        print("Training Logistic Regression")
        print("="*50)
        
        start_time = time.time()
        
        model = LogisticRegression(
            max_iter=300,
            solver='lbfgs',
            multi_class='auto',
            class_weight='balanced',
            n_jobs=-1,
            random_state=self.config.RANDOM_SEED
        )
        
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Predictions
        start_time = time.time()
        val_preds = model.predict(X_val)
        inference_time = time.time() - start_time
        
        # Metrics
        accuracy = accuracy_score(y_val, val_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val, val_preds, average='weighted', zero_division=0
        )
        
        results = {
            'model': 'Logistic Regression',
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'train_time': train_time,
            'inference_time': inference_time
        }
        
        print(f"Validation Accuracy: {accuracy:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Training Time: {train_time:.2f}s")
        print(f"Inference Time (per 1000 samples): {(inference_time/len(y_val)*1000):.2f}s")
        
        self.models['logistic_regression'] = model
        return results
    
    def train_random_forest(self, X_train, y_train, X_val, y_val):
        """Train Random Forest"""
        print("\n" + "="*50)
        print("Training Random Forest")
        print("="*50)
        
        start_time = time.time()
        
        model = RandomForestClassifier(
            n_estimators=80,
            max_depth=18,
            min_samples_split=5,
            class_weight='balanced',
            n_jobs=-1,
            random_state=self.config.RANDOM_SEED
        )
        
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Predictions (batch to handle memory)
        start_time = time.time()
        batch_size = 2000
        val_preds = []
        
        for i in range(0, X_val.shape[0], batch_size):
            batch = X_val[i:i+batch_size]
            val_preds.extend(model.predict(batch))
        
        val_preds = np.array(val_preds)
        inference_time = time.time() - start_time
        
        # Metrics
        accuracy = accuracy_score(y_val, val_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val, val_preds, average='weighted', zero_division=0
        )
        
        results = {
            'model': 'Random Forest',
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'train_time': train_time,
            'inference_time': inference_time
        }
        
        print(f"Validation Accuracy: {accuracy:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Training Time: {train_time:.2f}s")
        print(f"Inference Time (per 1000 samples): {(inference_time/len(y_val)*1000):.2f}s")
        
        self.models['random_forest'] = model
        return results
    
    def train_svm(self, X_train, y_train, X_val, y_val):
        """Train Support Vector Machine"""
        print("\n" + "="*50)
        print("Training SVM")
        print("="*50)
        
        start_time = time.time()
        
        model = LinearSVC(
            C=1.0,
            class_weight='balanced',
            max_iter=3000,
            random_state=self.config.RANDOM_SEED,
            verbose=1
        )
        
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Predictions
        start_time = time.time()
        val_preds = model.predict(X_val)
        inference_time = time.time() - start_time
        
        # Metrics
        accuracy = accuracy_score(y_val, val_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val, val_preds, average='weighted', zero_division=0
        )
        
        results = {
            'model': 'SVM',
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'train_time': train_time,
            'inference_time': inference_time
        }
        
        print(f"Validation Accuracy: {accuracy:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Training Time: {train_time:.2f}s")
        print(f"Inference Time (per 1000 samples): {(inference_time/len(y_val)*1000):.2f}s")
        
        self.models['svm'] = model
        return results
    
    def save_models(self, save_dir):
        """Save trained models"""
        print(f"\nSaving models to {save_dir}...")
        
        # Save vectorizer
        joblib.dump(self.vectorizer, os.path.join(save_dir, 'tfidf_vectorizer.pkl'))
        
        # Save models
        for name, model in self.models.items():
            joblib.dump(model, os.path.join(save_dir, f'{name}.pkl'))
        
        print("Models saved successfully!")
    
    def compare_models(self, results_list):
        """Compare all models"""
        print("\n" + "="*50)
        print("Model Comparison")
        print("="*50)
        
        df = pd.DataFrame(results_list)
        print(df.to_string(index=False))
        
        # Save comparison
        df.to_csv(os.path.join(self.config.RESULTS_DIR, 'classical_ml_comparison.csv'), index=False)


def main():
    """Main training pipeline for classical ML"""
    
    # Load preprocessed data
    train_df = pd.read_csv(os.path.join(Config.PROCESSED_DATA_DIR, 'train_preprocessed.csv'))
    val_df = pd.read_csv(os.path.join(Config.PROCESSED_DATA_DIR, 'val_preprocessed.csv'))
    test_df = pd.read_csv(os.path.join(Config.PROCESSED_DATA_DIR, 'test_preprocessed.csv'))
    
    print(f"Loaded data: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    # Initialize trainer
    trainer = ClassicalMLModels()
    
    # Create TF-IDF features
    X_train, X_val, X_test = trainer.create_tfidf_features(
        train_df['text_lemmatized'].fillna(''),
        val_df['text_lemmatized'].fillna(''),
        test_df['text_lemmatized'].fillna('')
    )
    
    y_train = train_df['emotion_label'].values
    y_val = val_df['emotion_label'].values
    y_test = test_df['emotion_label'].values
    
    # Train models
    results = []
    
    # Logistic Regression
    lr_results = trainer.train_logistic_regression(X_train, y_train, X_val, y_val)
    results.append(lr_results)
    
    # Random Forest
    rf_results = trainer.train_random_forest(X_train, y_train, X_val, y_val)
    results.append(rf_results)
    
    # SVM
    svm_results = trainer.train_svm(X_train, y_train, X_val, y_val)
    results.append(svm_results)
    
    # Compare models
    trainer.compare_models(results)
    
    # Save models
    trainer.save_models(Config.CLASSICAL_ML_DIR)
    
    print("\nClassical ML training complete!")


if __name__ == "__main__":
    main()