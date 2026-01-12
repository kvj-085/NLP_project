# FEVER Fact Verification with NLP Models

## Project Overview

This project implements and evaluates various machine learning approaches for the FEVER (Fact Extraction and VERification) task, which involves determining whether a given claim is supported, refuted, or has insufficient evidence based on Wikipedia text. We explored both traditional baseline models and state-of-the-art transformer-based architectures to tackle this three-class classification problem.

## Dataset

### FEVER Dataset
- **Source**: FEVER (Fact Extraction and VERification) shared task dataset from HuggingFace
- **Task**: Three-class classification (SUPPORTS, REFUTES, NOT ENOUGH INFO)
- **Format**: Claims paired with evidence text from Wikipedia articles

### Data Acquisition & Preprocessing
- Downloaded the FEVER dataset using the HuggingFace `datasets` library
- **Note**: The `data/` folder is not included in this repository (too large for git). You can reproduce it by running the preprocessing script.
- The raw data will be stored in `data/raw/fever/` containing:
  - `train.jsonl`: Training set
  - `shared_task_dev.jsonl`: Development/validation set
  - `shared_task_test.jsonl`: Test set

### Data Processing
- **Script**: [src/data_preprocessing.py](src/data_preprocessing.py)
- **Label Mapping**: Converted string labels to numeric IDs:
  - SUPPORTS → 0
  - REFUTES → 1
  - NOT ENOUGH INFO → 2
  
- **Evidence Extraction**: Implemented robust extraction logic to handle multiple evidence formats in the FEVER dataset
- **Text Concatenation**: Combined claims and evidence using `[SEP]` token: `claim [SEP] evidence`
- **Output**: Processed datasets saved to `data/processed/fever/` with train/validation/test splits

### Tokenized Datasets
Created pre-tokenized versions for faster training:
- `data/processed/fever_tokenized_distilbert/`: Tokenized with DistilBERT tokenizer
- `data/processed/fever_tokenized_roberta/`: Tokenized with RoBERTa tokenizer

---

## Baseline Models

### Approach
Implemented traditional machine learning models using TF-IDF vectorization as a baseline to compare against deep learning approaches.

### Models Implemented
**Script**: [src/baselines/baselines.py](src/baselines/baselines.py)

1. **Logistic Regression**
   - Multi-class classification with multinomial strategy
   - SAGA solver with max 1000 iterations
   
2. **Support Vector Machine (SVM)**
   - Linear kernel with probability estimates
   
3. **Random Forest**
   - 200 estimators
   - Default random state for reproducibility

### Feature Engineering
- **TF-IDF Vectorization**:
  - Max features: 5,000
  - N-gram range: (1, 2) - unigrams and bigrams
  - Minimum document frequency: 1

### Baseline Results

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| **Logistic Regression** | 0.694 | 0.688 |
| **SVM** | **0.719** | **0.718** |
| **Random Forest** | 0.711 | 0.701 |

**Key Findings**:
- SVM achieved the best performance among baselines with 71.9% accuracy
- All baselines performed reasonably well, establishing a solid foundation
- SVM's linear kernel proved effective for this text classification task
- Results saved in `outputs/baselines_results.csv`

---

## Transformer Models

### Training Framework
**Script**: [src/models/train.py](src/models/train.py)

- **Framework**: HuggingFace Transformers library
- **Training Strategy**:
  - Learning rate: 2e-5
  - Evaluation strategy: Per epoch
  - Metrics: Accuracy and Macro F1-score
  - Early stopping and checkpoint saving enabled

### Models Trained

#### 1. **BERT (bert-base-uncased)**
- **Architecture**: Base BERT model (110M parameters)
- **Training**: Fine-tuned for sequence classification with 3 output labels
- **Notebook**: [src/transformer_models/bert_training.ipynb](src/transformer_models/bert_training.ipynb)
- **Predictions**: 
  - Validation: `outputs/BERT/predictions_validation.csv`
  - Test: `outputs/BERT/predictions_test.csv`

#### 2. **DistilBERT (distilbert-base-uncased)**
- **Architecture**: Distilled version of BERT (66M parameters, 40% smaller)
- **Advantage**: Faster training and inference while retaining 97% of BERT's performance
- **Notebook**: [src/transformer_models/DistilBERT.ipynb](src/transformer_models/DistilBERT.ipynb)
- **Tokenization**: Pre-tokenization saved to `data/processed/fever_tokenized_distilbert/`
- **Predictions**:
  - Validation: `outputs/DistilBERT/predictions_validation.csv`
  - Test: `outputs/DistilBERT/predictions_test.csv`

#### 3. **RoBERTa (roberta-base)**
- **Architecture**: Robustly Optimized BERT Approach (125M parameters)
- **Key Differences**: Dynamic masking, larger batches, byte-pair encoding
- **Notebooks**: 
  - Tokenization: [src/transformer_models/roberta_tokenization.ipynb](src/transformer_models/roberta_tokenization.ipynb)
  - Training: [src/transformer_models/roberta_training.ipynb](src/transformer_models/roberta_training.ipynb)
- **Tokenization**: Pre-tokenization saved to `data/processed/fever_tokenized_roberta/`
- **Predictions**:
  - Validation: `outputs/RoBERTa/predictions_validation.csv`
  - Test: `outputs/RoBERTa/predictions_test.csv`

### Transformer Results

Based on validation set performance (20,000 examples):

| Model | Parameters | Relative Size | Performance | Accuracy | Macro-F1 |
|-------|------------|---------------|-------------|----------|----------|
| **BERT** | 110M | Base | Strong baseline | 0.8711 | 0.8695 |
| **DistilBERT** | 66M | 40% smaller | Efficient, competitive | 0.8685 | 0.8669 |
| **RoBERTa** | 125M | Largest | Best expected performance | 0.8707 | 0.8687 |

**Key Observations**:
- All transformer models significantly outperformed traditional baselines
- DistilBERT offered excellent efficiency-performance tradeoff
- RoBERTa's advanced pre-training strategy provided robust results
- Models were evaluated on both validation and test sets

---

## Error Analysis

### Analysis Framework
**Script**: [visualize_errors.py](visualize_errors.py)

Comprehensive error analysis was conducted to understand model failures and comparative performance.

### Analysis Components

#### 1. **Confusion Matrices**
Generated for each model (BERT, DistilBERT, RoBERTa) showing:
- True vs. predicted labels
- Error patterns across the three classes
- Saved as: `outputs/error_analysis/{MODEL}_confusion_matrix_validation.png`

#### 2. **Error Distribution by Class**
Analyzed how errors are distributed across true labels:
- Total examples per class
- Error counts per class
- Error rates per class
- Visualizations show both absolute counts and percentages
- Saved as: `outputs/error_analysis/{MODEL}_error_distribution_validation.png`

#### 3. **Comparative Analysis**
Cross-model comparison including:
- **Overall Accuracy**: Bar chart comparing all three models
- **Macro F1 Scores**: Performance across all classes
- **Total Errors**: Absolute error counts per model
- **Error Rates by Class**: Grouped bar chart showing which classes are most challenging for each model
- Saved as: `outputs/error_analysis/comparative_analysis_validation.png`

#### 4. **Common Error Analysis**
Identified examples where **all models failed**:
- Counted instances of universal failure
- Broke down common errors by true class
- Percentage of examples that stumped all models
- Helps identify inherently difficult examples in the dataset

### Error Analysis Findings

The analysis revealed:
- **Class-Specific Challenges**: Certain label combinations proved more difficult
- **Model Strengths**: Each model had different error patterns
- **Difficult Examples**: Some claims are ambiguous even for advanced models
- **Performance Trade-offs**: Efficiency vs. accuracy considerations

All error analysis visualizations are saved in `outputs/error_analysis/` directory.

---

## Project Structure

```
NLP proj/
├── data/                                 # Not in the Repo (too large, ~100+ MB)
│   ├── raw/
│   │   └── fever/                        # Raw JSONL files
│   └── processed/
│       ├── fever/                        # Processed dataset
│       ├── fever_tokenized_distilbert/   # Pre-tokenized for DistilBERT
│       └── fever_tokenized_roberta/      # Pre-tokenized for RoBERTa
├── src/
│   ├── baselines/
│   │   └── baselines.py                  # TF-IDF + ML models
│   ├── models/
│   │   └── train.py                      # Transformer training script
│   ├── transformer_models/
│   │   ├── bert_training.ipynb
│   │   ├── DistilBERT.ipynb
│   │   ├── roberta_tokenization.ipynb
│   │   └── roberta_training.ipynb
│   ├── data_preprocessing.py             # Dataset loading and processing
│   └── utils.py                          # Utility functions
├── outputs/                              # Included in the Repo(~2.3 MB)
│   ├── baselines_results.csv             # Baseline model results
│   ├── BERT/                             # BERT predictions
│   ├── DistilBERT/                       # DistilBERT predictions
│   ├── RoBERTa/                          # RoBERTa predictions
│   └── error_analysis/                   # Error analysis visualizations
├── validity_checks/                       # Data validation scripts
├── visualize_errors.py                   # Error analysis and visualization
├── requirements.txt                      # Python dependencies
└── README.md                             # This file
```

---

## Key Findings & Conclusions

### Model Performance Hierarchy
1. **Transformers >> Baselines**: All transformer models substantially outperformed traditional ML approaches
2. **SVM Best Baseline**: Among baselines, SVM with linear kernel achieved 71.9% accuracy
3. **Transformer Efficiency**: DistilBERT provided near-BERT performance with 40% fewer parameters

### Task Challenges
- The FEVER task is inherently challenging with three nuanced classes
- "NOT ENOUGH INFO" class proved most difficult across all models
- Evidence quality and claim ambiguity impact performance
- Some claims require complex reasoning beyond surface-level text matching

### Practical Implications
- **For Production**: DistilBERT offers best efficiency-performance trade-off
- **For Accuracy**: RoBERTa or BERT recommended if computational resources available
- **For Baselines**: SVM remains viable for resource-constrained scenarios

---

## How to Reproduce

### Prerequisites
- Python 3.8+
- GPU recommended for transformer training (CPU works but slower)
- 16GB+ RAM recommended

### Step 1: Clone and Setup

```bash
git clone https://github.com/kvj-085/NLP_project.git
cd NLP_project
pip install -r requirements.txt
```

### Step 2: Download and Preprocess Data

The `data/` folder is not in the repo. Run the preprocessing script to download FEVER dataset:

```bash
python src/data_preprocessing.py
```

This will:
- Download FEVER dataset from HuggingFace
- Create `data/raw/fever/` with train/dev/test splits
- Process and save to `data/processed/fever/`

### Step 3: Run Baseline Models

```bash
python src/baselines/baselines.py
```

**Output**: `outputs/baselines_results.csv` with accuracy and F1 scores

### Step 4: Train Transformer Models

Open and run the Jupyter notebooks in order:

**For BERT:**
```bash
jupyter notebook src/transformer_models/bert_training.ipynb
```

**For DistilBERT:**
```bash
jupyter notebook src/transformer_models/DistilBERT.ipynb
```

**For RoBERTa:**
```bash
# First tokenize the data
jupyter notebook src/transformer_models/roberta_tokenization.ipynb

# Then train
jupyter notebook src/transformer_models/roberta_training.ipynb
```

**Outputs**: Predictions saved in `outputs/{MODEL}/predictions_validation.csv` and `predictions_test.csv`

### Step 5: Generate Error Analysis

```bash
python visualize_errors.py
```

**Outputs**: 
- Confusion matrices
- Error distribution charts
- Comparative analysis
- All saved in `outputs/error_analysis/`

### Expected Results

After running all scripts, you should see:
- Baseline accuracies: 69-72%
- Transformer models around 0.87 accuracy / 0.87 macro-F1
- All predictions and visualizations in `outputs/`

**Note**: Pre-computed results are already in the `outputs/` folder, so you can view them immediately without running the models!

---

## Technical Details

### Requirements
See [requirements.txt](requirements.txt) for full dependencies. Key libraries:
- `transformers`: HuggingFace Transformers library
- `datasets`: HuggingFace Datasets library  
- `torch`: PyTorch deep learning framework
- `scikit-learn`: Traditional ML models and metrics
- `pandas`, `numpy`: Data manipulation
- `matplotlib`, `seaborn`: Visualization

### Hardware Recommendations
- **Baselines**: CPU sufficient
- **Transformers**: GPU recommended (CUDA-enabled)
- **Memory**: 16GB+ RAM recommended for large model training

---

## Acknowledgments

- **FEVER Dataset**: Thorne et al., "FEVER: a large-scale dataset for Fact Extraction and VERification"
- **HuggingFace**: For transformers library and pre-trained models
- **PyTorch**: Deep learning framework

---

## Project Status

**Completed Components:**
- Dataset acquisition and preprocessing
- Baseline model implementation and evaluation
- Three transformer models trained and evaluated
- Comprehensive error analysis with visualizations
- Model predictions saved for all splits

This project provides a complete pipeline for fact verification using both traditional and modern NLP approaches, with thorough analysis of model behavior and performance characteristics.
