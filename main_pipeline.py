"""
Main Pipeline - Complete Training and Evaluation Workflow
Author: Khethavath Sunil Naik
Project: Enhancing Emotional Support Chatbots with ML

This script runs the complete pipeline:
1. Data preprocessing
2. Classical ML models
3. BERT training
4. Semi-supervised learning
5. GPT-2 training
6. Evaluation
"""

import os
import sys
import argparse
from config import Config

def run_preprocessing():
    """Run data preprocessing"""
    print("\n" + "="*60)
    print("STEP 1: DATA PREPROCESSING")
    print("="*60)
    
    from data_loader import main as preprocess_main
    preprocess_main()

def run_classical_ml():
    """Run classical ML models"""
    print("\n" + "="*60)
    print("STEP 2: CLASSICAL MACHINE LEARNING MODELS")
    print("="*60)
    
    from classical_ml import main as classical_main
    classical_main()

def run_bert_training():
    """Run BERT training"""
    print("\n" + "="*60)
    print("STEP 3: BERT EMOTION CLASSIFIER TRAINING")
    print("="*60)
    
    from bert_classifier import main as bert_main
    bert_main()

def run_semi_supervised():
    """Run semi-supervised learning"""
    print("\n" + "="*60)
    print("STEP 4: SEMI-SUPERVISED LEARNING")
    print("="*60)
    
    from semi_supervised import main as ssl_main
    ssl_main()

def run_gpt2_training():
    """Run GPT-2 training"""
    print("\n" + "="*60)
    print("STEP 5: GPT-2 RESPONSE GENERATOR TRAINING")
    print("="*60)
    
    from gpt2_generator import main as gpt2_main
    gpt2_main()

def run_evaluation():
    """Run evaluation"""
    print("\n" + "="*60)
    print("STEP 6: COMPREHENSIVE EVALUATION")
    print("="*60)
    
    from evaluation import main as eval_main
    eval_main()

def run_interactive_chat():
    """Run interactive chatbot"""
    print("\n" + "="*60)
    print("INTERACTIVE CHATBOT SESSION")
    print("="*60)
    
    from chatbot_system import interactive_chat
    interactive_chat()

def main():
    """Main pipeline"""
    parser = argparse.ArgumentParser(
        description='Emotional Support Chatbot Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python main_pipeline.py --all
  
  # Run specific steps
  python main_pipeline.py --preprocess --classical-ml
  
  # Skip certain steps
  python main_pipeline.py --all --skip-gpt2
  
  # Interactive chat
  python main_pipeline.py --chat
        """
    )
    
    parser.add_argument('--all', action='store_true', 
                       help='Run complete pipeline')
    parser.add_argument('--preprocess', action='store_true',
                       help='Run data preprocessing')
    parser.add_argument('--classical-ml', action='store_true',
                       help='Train classical ML models')
    parser.add_argument('--bert', action='store_true',
                       help='Train BERT classifier')
    parser.add_argument('--semi-supervised', action='store_true',
                       help='Run semi-supervised learning')
    parser.add_argument('--gpt2', action='store_true',
                       help='Train GPT-2 generator')
    parser.add_argument('--evaluate', action='store_true',
                       help='Run evaluation')
    parser.add_argument('--chat', action='store_true',
                       help='Start interactive chat')
    
    # Skip options
    parser.add_argument('--skip-preprocess', action='store_true',
                       help='Skip preprocessing')
    parser.add_argument('--skip-classical-ml', action='store_true',
                       help='Skip classical ML')
    parser.add_argument('--skip-bert', action='store_true',
                       help='Skip BERT training')
    parser.add_argument('--skip-semi-supervised', action='store_true',
                       help='Skip semi-supervised learning')
    parser.add_argument('--skip-gpt2', action='store_true',
                       help='Skip GPT-2 training')
    parser.add_argument('--skip-evaluation', action='store_true',
                       help='Skip evaluation')
    
    args = parser.parse_args()
    
    # Create directories
    print("Setting up directories...")
    Config.create_directories()
    
    # If no arguments, show help
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    # Interactive chat
    if args.chat:
        run_interactive_chat()
        return
    
    # Determine which steps to run
    run_all = args.all
    
    try:
        # Step 1: Preprocessing
        if (run_all and not args.skip_preprocess) or args.preprocess:
            run_preprocessing()
        
        # Step 2: Classical ML
        if (run_all and not args.skip_classical_ml) or args.classical_ml:
            run_classical_ml()
        
        # Step 3: BERT
        if (run_all and not args.skip_bert) or args.bert:
            run_bert_training()
        
        # Step 4: Semi-supervised
        if (run_all and not args.skip_semi_supervised) or args.semi_supervised:
            run_semi_supervised()
        
        # Step 5: GPT-2
        if (run_all and not args.skip_gpt2) or args.gpt2:
            run_gpt2_training()
        
        # Step 6: Evaluation
        if (run_all and not args.skip_evaluation) or args.evaluate:
            run_evaluation()
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETE!")
        print("="*60)
        print("\nNext steps:")
        print("  1. Review results in the 'results/' directory")
        print("  2. Check training logs and plots")
        print("  3. Run interactive chat: python main_pipeline.py --chat")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error in pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()