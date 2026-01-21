import gradio as gr
import ctranslate2
from transformers import MT5Tokenizer

MODEL_PATH = "../models/ct2-mt5"
TOKENIZER_PATH = "../models/mt5-translit"

translator = ctranslate2.Translator(
    MODEL_PATH,
    device="cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu",
    compute_type="int8"
)

tokenizer = MT5Tokenizer.from_pretrained(TOKENIZER_PATH)

LANG_MAP = {
    "Hindi": "hi",
    "Bengali": "bn",
    "Tamil": "ta"
}

def transliterate(text, language):
    lang = LANG_MAP[language]
    input_text = f"<{lang}> {text}"

    tokens = tokenizer.convert_ids_to_tokens(
        tokenizer(input_text)["input_ids"]
    )

    results = translator.translate_batch(
        [tokens],
        max_decoding_length=32
    )

    output_tokens = results[0].hypotheses[0]
    return tokenizer.decode(
        tokenizer.convert_tokens_to_ids(output_tokens),
        skip_special_tokens=True
    )

demo = gr.Interface(
    fn=transliterate,
    inputs=[
        gr.Textbox(label="Input (English / Latin)"),
        gr.Dropdown(["Hindi", "Bengali", "Tamil"], label="Target Language")
    ],
    outputs=gr.Textbox(label="Transliterated Output"),
    title="Multilingual Indic Transliteration (mT5 + CTranslate2)",
    description="Fast multilingual transliteration using optimized CTranslate2 inference."
)

if __name__ == "__main__":
    demo.launch()
