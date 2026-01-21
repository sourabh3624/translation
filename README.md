Translation project done three language 


## Aksharantar Dataset (Transliteration)

Source:
https://huggingface.co/datasets/ai4bharat/Aksharantar

Task:
Roman (Latin) → Native Script Transliteration

Languages Used:
- Hindi (hi)
- Bengali (bn)
- Tamil (ta)

Preprocessing:
- Added language tags: <hi>, <bn>, <ta>
- Combined multilingual dataset
- Train/Test split: 90/10

Output Location:
data/processed/


## Evaluation

The model was evaluated on the Aksharantar test split using Character Error Rate (CER) and exact-match accuracy.

| Metric | Value |
|------|------|
| Character Error Rate (CER) | **6.87%** |
| Exact Match Accuracy | **69.16%** |
| Eval Loss | **0.35** |

CER was chosen as the primary metric, as transliteration operates at the character level and allows for partial correctness. Exact-match accuracy is reported for reference and is lower due to the strict nature of the metric and the presence of multiple valid transliterations.

## Optimization Results

The fine-tuned multilingual mT5 transliteration model was optimized using CTranslate2 (INT8).

| Model | Avg Latency | Speedup |
|------|------------|--------|
| PyTorch mT5 | 76.4 ms | 1× |
| CTranslate2 INT8 | 17.1 ms | 4.47× |

The optimized model achieves over 4× faster inference with no observable quality loss.


## Deployment

The optimized CTranslate2 model was deployed on Hugging Face Spaces using Gradio.
GPU-backed inference enables real-time multilingual transliteration.

🔗 Live Demo: https://huggingface.co/spaces/sourabh3624/translation
