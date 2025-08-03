# LLM-Based Sentence Splitter for Italian 🇮🇹

This module implements sentence splitting using Large Language Models (LLMs) fine-tuned with QLoRA.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements_llm.txt
```

### 2. Set Environment Variable
```bash
export HF_TOKEN="your_huggingface_token"
```

### 3. Test with Pre-trained Model
```bash
python example_usage.py
```

### 4. Fine-tune Your Own Model
```python
from src.sentence_splitter_llm.llm_sentence_splitter import LLMSentenceSplitter

splitter = LLMSentenceSplitter("microsoft/Phi-3.5-mini-instruct")
splitter.train(output_dir="./my-sentence-splitter", num_epochs=2)
```

## 🔧 How It Works

### Data Conversion
- Converts your token-level dataset (`fax4ever/manzoni`) to instruction format
- Creates training examples like:
  ```
  Instruction: "Dividi il seguente testo italiano in frasi..."
  Input: "Quel ramo del lago di Como che volge..."
  Output: "1. Quel ramo del lago di Como. 2. Che volge a mezzogiorno."
  ```

### QLoRA Fine-tuning
- Uses 4-bit quantization to reduce memory usage
- Applies LoRA (Low-Rank Adaptation) for efficient fine-tuning
- Trains only ~1% of model parameters
- Requires ~8GB GPU memory instead of 80GB+

### Supported Models
- `microsoft/Phi-3.5-mini-instruct` (recommended for Italian)
- `swap-uniba/LLaMAntino-2-7b-hf-ITA` (Italian Llama2)  
- `Qwen/Qwen2.5-7B-Instruct` (multilingual)
- Any instruction-tuned model on HuggingFace

## 📊 Advantages vs BERT Approach

### ✅ LLM Advantages
- **Better context**: Understands long-range dependencies
- **Flexibility**: Can handle various text types and lengths
- **Generalization**: Works on unseen text patterns
- **Human-readable output**: Generates actual sentences

### ⚠️ LLM Challenges  
- **Computational cost**: 10-100x slower than BERT
- **Memory requirements**: Needs GPU with 8GB+ VRAM
- **Consistency**: May give slightly different results each run
- **Complex evaluation**: Generated text is harder to score

## 🎯 Use Cases

**Use LLMs when:**
- You have diverse text types (literature, news, technical)
- You need to handle very long documents
- You want human-readable sentence outputs
- You have sufficient computational resources

**Use BERT when:**
- You need fast, consistent predictions
- You have limited computational resources  
- Your text is similar to training data
- You need token-level precision

## 🔧 Hardware Requirements

### Minimum
- GPU: 8GB VRAM (RTX 4070, RTX 3080, etc.)
- RAM: 16GB system memory
- Training time: 1-3 hours

### Recommended
- GPU: 16GB+ VRAM (RTX 4080, A100, etc.)
- RAM: 32GB+ system memory
- Training time: 30-60 minutes

## 📈 Performance Tips

1. **Batch size**: Start with 1, increase if you have more memory
2. **Gradient accumulation**: Use steps=8 for effective batch size of 8
3. **Learning rate**: 2e-4 works well for LoRA fine-tuning
4. **Max length**: 512 tokens is usually sufficient for sentence splitting
5. **Epochs**: 2-3 epochs are usually enough to avoid overfitting