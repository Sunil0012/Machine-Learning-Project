# Enhancing Emotional Support Chatbots with Machine Learning

**Author:** Khethavath Sunil Naik  
**Roll No:** 12341170  
**Institution:** IIT Bhilai  
**Project:** DSL501 Machine Learning Project

## Overview

This project develops an advanced emotional support chatbot that integrates classical machine learning, deep learning (BERT & GPT-2), and linguistic analysis to provide empathetic, contextually appropriate responses. The system classifies 32 fine-grained emotions and generates responses with quantified empathy.

### Key Features

- **32 Emotion Classification** using BERT (72% accuracy)
- **Semi-Supervised Learning** with pseudo-labeling (+7.3% improvement)
- **Empathetic Response Generation** using GPT-2
- **LIWC-Style Linguistic Analysis** for empathy quantification
- **Comprehensive Evaluation** (BLEU, ROUGE, BERTScore)
- **212% Empathy Improvement** over baseline

## Project Structure

```
emotional-support-chatbot/
├── config.py                 # Configuration settings
├── data_loader.py            # Data preprocessing pipeline
├── classical_ml.py           # Logistic Regression, RF, SVM
├── bert_classifier.py        # BERT emotion classifier
├── semi_supervised.py        # Pseudo-labeling pipeline
├── gpt2_generator.py         # GPT-2 response generator
├── chatbot_system.py         # Complete integrated chatbot
├── evaluation.py             # Evaluation framework
├── main_pipeline.py          # Main execution script
├── requirements.txt          # Dependencies
│
├── data/
│   ├── raw/                  # Original CSV files
│   └── processed/            # Preprocessed data
│
├── models/
│   ├── bert_emotion_classifier/
│   ├── gpt2_response_generator/
│   └── classical_ml/
│
└── results/
    ├── plots/                # Visualizations
    ├── logs/                 # Training logs
    └── *.csv                 # Results and metrics
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM
- 10GB+ disk space

### Setup

1. **Clone the repository:**
```bash
git clone <your-repo-url>
cd emotional-support-chatbot
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Download spaCy model:**
```bash
python -m spacy download en_core_web_sm
```

5. **Download NLTK data:**
```python
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
```

## Dataset

This project uses the **Empathetic Dialogues** dataset from Facebook AI Research.

### Download Dataset

1. Visit: https://github.com/facebookresearch/EmpatheticDialogues
2. Download the dataset files
3. Place CSV files in `data/raw/`:
   - `train.csv`
   - `valid.csv`
   - `test.csv`

### Dataset Structure

The dataset should have these columns:
- `conv_id`: Conversation ID
- `utterance_idx`: Turn index
- `context`: Emotion label (32 categories)
- `prompt`: User input
- `utterance`: Response
- `speaker_idx`: Speaker ID

## Usage

### Quick Start - Interactive Chat

```bash
python main_pipeline.py --chat
```

### Complete Pipeline

Run the entire training and evaluation pipeline:

```bash
python main_pipeline.py --all
```

This will execute:
1. Data preprocessing
2. Classical ML training
3. BERT training
4. Semi-supervised learning
5. GPT-2 training
6. Comprehensive evaluation

### Step-by-Step Execution

#### 1. Data Preprocessing
```bash
python main_pipeline.py --preprocess
```

#### 2. Train Classical ML Models
```bash
python main_pipeline.py --classical-ml
```

#### 3. Train BERT Emotion Classifier
```bash
python main_pipeline.py --bert
```

#### 4. Semi-Supervised Learning
```bash
python main_pipeline.py --semi-supervised
```

#### 5. Train GPT-2 Response Generator
```bash
python main_pipeline.py --gpt2
```

#### 6. Evaluate System
```bash
python main_pipeline.py --evaluate
```

### Custom Combinations

```bash
# Train only BERT and GPT-2
python main_pipeline.py --bert --gpt2

# Run everything except GPT-2 training
python main_pipeline.py --all --skip-gpt2

# Preprocess and train classical ML
python main_pipeline.py --preprocess --classical-ml
```

## Interactive Chatbot

Start an interactive conversation:

```bash
python chatbot_system.py
```

Commands in interactive mode:
- Type your message to chat
- `summary` - View conversation statistics
- `reset` - Start new conversation
- `quit` - Exit and save conversation

### Batch Testing

Test with predefined examples:

```bash
python chatbot_system.py test
```

## Configuration

Edit `config.py` to customize:

### Model Parameters
```python
# BERT Configuration
BERT_BATCH_SIZE = 16
BERT_LEARNING_RATE = 2e-5
BERT_EPOCHS = 3
BERT_MAX_LENGTH = 128

# GPT-2 Configuration
GPT2_BATCH_SIZE = 8
GPT2_LEARNING_RATE = 5e-5
GPT2_EPOCHS = 2
GPT2_TEMPERATURE = 0.7
```

### Paths
```python
DATA_DIR = "data"
MODEL_DIR = "models"
RESULTS_DIR = "results"
```

## Results

### Emotion Classification Performance

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| Logistic Regression | 46.02% | 45.77% |
| Random Forest | 31.60% | 31.25% |
| SVM | 42.09% | 42.05% |
| BERT (Supervised) | 67.12% | 66.95% |
| **BERT (Semi-Supervised)** | **72.04%** | **71.92%** |

### Response Generation Quality

| Metric | Our Chatbot | Baseline | Improvement |
|--------|-------------|----------|-------------|
| BLEU-1 | 0.4123 | 0.2456 | +68% |
| ROUGE-1 | 0.5234 | 0.3412 | +53% |
| BERTScore | 0.8178 | 0.7023 | +16% |
| **Empathy Score** | **0.625** | **0.200** | **+212%** |

## Key Findings

1. **Deep Learning Superiority**: BERT achieved 46% better accuracy than classical ML
2. **Semi-Supervised Gains**: Pseudo-labeling improved accuracy by 7.3%
3. **Empathy Enhancement**: 212% improvement in empathy scores
4. **Response Quality**: 68% BLEU improvement, 53% ROUGE improvement
5. **Rare Emotion Handling**: Up to 21% F1 improvement for rare classes

## Project Highlights

### Novel Contributions

1. **32-Category Fine-Grained Emotion Detection** (vs. typical 6-8 emotions)
2. **Semi-Supervised Enhancement** for limited-data scenarios
3. **LIWC-Based Empathy Quantification**
4. **Multi-Dimensional Evaluation Framework**
5. **Complete Reproducible Pipeline**

### Technical Achievements

- Integrated BERT + GPT-2 architecture
- Real-time emotion detection (23ms)
- Context-aware response generation
- Psychological validation through linguistic analysis
- Comprehensive baseline comparison

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```python
# Reduce batch sizes in config.py
BERT_BATCH_SIZE = 8  # Default: 16
GPT2_BATCH_SIZE = 4  # Default: 8
```

**2. Data Files Not Found**
```bash
# Ensure CSV files are in data/raw/
ls data/raw/
# Should show: train.csv, valid.csv, test.csv
```

**3. Import Errors**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

**4. spaCy Model Missing**
```bash
python -m spacy download en_core_web_sm
```

## Performance Optimization

### For CPU-Only Systems
```python
# In config.py, models will auto-detect CPU
# Reduce batch sizes:
BERT_BATCH_SIZE = 4
GPT2_BATCH_SIZE = 2
```

### For GPU Systems
```python
# Maximize batch sizes based on GPU memory:
# 8GB GPU: Default settings
# 16GB+ GPU: Increase batch sizes
BERT_BATCH_SIZE = 32
GPT2_BATCH_SIZE = 16
```

## Evaluation Metrics

The system evaluates responses using:

### Automatic Metrics
- **BLEU-1/2/3/4**: N-gram overlap
- **ROUGE-1/2/L**: Recall-oriented evaluation
- **BERTScore**: Semantic similarity

### Empathy Metrics
- **LIWC Analysis**: Psychological markers
- **Empathy Score**: Composite from positive emotion, help, sympathy, social words
- **Sentiment Alignment**: Context-response emotion matching

### Custom Metrics
- **Emotion Detection Accuracy**: 32-class classification
- **Confidence Scores**: Model uncertainty
- **Temporal Sentiment**: Conversation progression

## Future Enhancements

### Technical
- [ ] Multi-lingual support (Hindi, Spanish)
- [ ] Voice-based interaction
- [ ] Long-term memory across sessions
- [ ] Reinforcement learning from user feedback

### Evaluation
- [ ] Large-scale human evaluation
- [ ] Clinical validation with therapists
- [ ] Longitudinal effectiveness studies

### Safety
- [ ] Crisis detection and intervention
- [ ] Self-harm risk assessment
- [ ] Professional referral system
- [ ] Content moderation

## Citation

If you use this code, please cite:

```bibtex
@project{naik2025emotional,
  title={Enhancing Emotional Support Chatbots with Machine Learning and Linguistic Analysis},
  author={Naik, Khethavath Sunil},
  institution={Indian Institute of Technology Bhilai},
  year={2025},
  type={ML Project Report}
}
```

## Acknowledgments

- Facebook AI Research for the Empathetic Dialogues dataset
- Hugging Face for Transformers library
- IIT Bhilai Department of CSE for support and resources

## License

This project is for educational purposes. Please check the Empathetic Dialogues dataset license for data usage terms.

## Contact

**Khethavath Sunil Naik**  
Roll No: 12341170  
IIT Bhilai  
Email: khethavathn@iitbhilai.ac.in

## References

1. Devlin et al. (2019) - BERT: Pre-training of Deep Bidirectional Transformers
2. Radford et al. (2019) - Language Models are Unsupervised Multitask Learners
3. Rashkin et al. (2019) - Towards Empathetic Open-domain Conversation Models
4. Pennebaker et al. (2015) - LIWC2015 Development and Psychometric Properties

---

**Note**: This is a research project. The chatbot is NOT a replacement for professional mental health services. Users experiencing crisis should contact appropriate professional resources.
