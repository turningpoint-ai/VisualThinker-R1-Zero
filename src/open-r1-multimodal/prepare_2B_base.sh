# Prepare base model with chat template for SFT training
git lfs install
git clone https://huggingface.co/Qwen/Qwen2-VL-2B
mv Qwen2-VL-2B Qwen2-VL-2B-Base

huggingface-cli download Qwen/Qwen2-VL-2B-Instruct chat_template.json tokenizer_config.json --local-dir ./Qwen2-VL-2B-Base

