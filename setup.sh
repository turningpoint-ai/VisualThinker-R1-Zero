conda create -n VisualThinker python=3.11 
conda activate VisualThinker

# Install the packages in open-r1-multimodal .
cd src/open-r1-multimodal
pip install -e ".[dev]"

# Additional modules
pip install wandb==0.18.3
pip install tensorboardx
pip install qwen_vl_utils torchvision
pip install flash-attn --no-build-isolation

pip install transformers==4.49.0 # correct deepspeed support
pip install duckdb
pip install opencv-python
pip install pandas
pip install math_verify==0.5.2
pip install datasets
pip install accelerate
pip install deepspeed
