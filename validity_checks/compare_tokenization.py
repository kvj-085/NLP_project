"""Compare DistilBERT and RoBERTa tokenization on FEVER dataset."""
from datasets import load_from_disk
from transformers import AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

# Load dataset
fever_dataset = load_from_disk('data/processed/fever')
example = fever_dataset['train'][0]

# Load tokenizers
distilbert_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
roberta_tokenizer = AutoTokenizer.from_pretrained('roberta-base')

text = example['text']

print("="*70)
print("TOKENIZATION COMPARISON: DistilBERT vs RoBERTa")
print("="*70)
print(f"\nOriginal Text:\n{text[:200]}...\n")
print(f"Label: {example['label']}\n")

# DistilBERT tokenization
distilbert_tokens = distilbert_tokenizer(text, truncation=True, max_length=128)
print("="*70)
print("DistilBERT (WordPiece Tokenization)")
print("="*70)
print(f"Tokenizer: bert-base-uncased vocabulary")
print(f"Algorithm: WordPiece")
print(f"Special tokens: [CLS], [SEP], [PAD]")
print(f"Total tokens: {len(distilbert_tokens['input_ids'])}")
print(f"\nFirst 30 tokens:")
db_tokens = distilbert_tokenizer.convert_ids_to_tokens(distilbert_tokens['input_ids'][:30])
for i, token in enumerate(db_tokens):
    print(f"  {i:2d}. {token}")

# RoBERTa tokenization
roberta_tokens = roberta_tokenizer(text, truncation=True, max_length=128)
print("\n" + "="*70)
print("RoBERTa (Byte-Pair Encoding - BPE)")
print("="*70)
print(f"Tokenizer: roberta-base vocabulary")
print(f"Algorithm: BPE (Byte-Pair Encoding)")
print(f"Special tokens: <s>, </s>, <pad>")
print(f"Total tokens: {len(roberta_tokens['input_ids'])}")
print(f"\nFirst 30 tokens:")
rb_tokens = roberta_tokenizer.convert_ids_to_tokens(roberta_tokens['input_ids'][:30])
for i, token in enumerate(rb_tokens):
    print(f"  {i:2d}. {token}")

print("\n" + "="*70)
print("KEY DIFFERENCES")
print("="*70)
print("""
1. ALGORITHM:
   - DistilBERT: WordPiece tokenization (splits words using ## prefix)
   - RoBERTa: BPE tokenization (splits words using Ġ for spaces)

2. SPECIAL TOKENS:
   - DistilBERT: [CLS] at start, [SEP] as separator
   - RoBERTa: <s> at start, </s> as separator

3. SUBWORD HANDLING:
   - DistilBERT: Uses ## prefix for subword pieces (e.g., "##j", "##er")
   - RoBERTa: Uses Ġ (space character) to mark word boundaries
   
4. CASE SENSITIVITY:
   - DistilBERT: Uncased (converts to lowercase)
   - RoBERTa: Cased (preserves original case)

5. VOCABULARY:
   - DistilBERT: ~30k tokens (BERT vocabulary)
   - RoBERTa: ~50k tokens (larger vocabulary)
""")

print("="*70)
print("EXAMPLE FROM OUTPUT ABOVE:")
print("="*70)
print("""
Word: "Nikolaj"

DistilBERT splits as:
  - 'nikola' (lowercase, main part)
  - '##j' (suffix piece marked with ##)

RoBERTa splits as:
  - 'Nik' (preserves case, first part)
  - 'ol' (middle part)
  - 'aj' (last part, Ġ indicates word boundary before next token)
  
This shows how WordPiece focuses on linguistic subword units while
BPE learns purely data-driven byte-pair merges.
""")
