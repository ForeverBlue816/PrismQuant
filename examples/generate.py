"""Run: python examples/generate.py"""
from prismquant import load_model

loaded = load_model('qwen3_0.6b_base')
try:
    print(loaded.generate('The key idea behind quantization is', max_new_tokens=64))
finally:
    loaded.close()
