# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Supervised fine-tuning script for decoder language models.

Usage:

# One 1 node of 8 x H100s
accelerate launch --config_file=configs/zero3.yaml src/open_r1/sft.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --dataset_name HuggingFaceH4/Bespoke-Stratos-17k \
    --learning_rate 2.0e-5 \
    --num_train_epochs 1 \
    --packing \
    --max_seq_length 4096 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing \
    --bf16 \
    --logging_steps 5 \
    --eval_strategy steps \
    --eval_steps 100 \
    --output_dir data/Qwen2.5-1.5B-Open-R1-Distill
"""
import torch

from datasets import load_dataset
from transformers import AutoTokenizer, AutoProcessor, default_data_collator
from qwen_vl_utils import process_vision_info

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

from transformers import AutoModelForCausalLM, Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration, Qwen2VLConfig, Qwen2VLForConditionalGeneration
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
AutoModelForCausalLM.register(config_class=Qwen2_5_VLConfig, model_class=Qwen2_5_VLForConditionalGeneration)
AutoModelForCausalLM.register(config_class=Qwen2VLConfig, model_class=Qwen2VLForConditionalGeneration)

from torch.utils.data import Dataset

from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)
CHAT_TEMPLATE = {
    "chat_template": "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
}

# oracle answer 

def main(script_args, training_args, model_args):
    ################
    # Model init kwargs & Tokenizer
    ################
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=model_args.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    training_args.model_init_kwargs = model_kwargs
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################
    # dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    if script_args.dataset_name == "SAT":
        def make_conversation_sat(example):
            return [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image",
                                        "image": dataset_prefix + example["images"][0],
                                    },
                                    {
                                        "type": "text",
                                        "text": example["messages"][0]["content"],
                                    },
                                ],
                            },
                            {
                                "role": "assistant",
                                "content": [{"type": "text", "text": example["messages"][1]["content"]}],
                            },
                        ]

        dataset_prefix = "../data/SAT/"
        dataset_path = "SAT_train_15000.json"
        
        import json
        # load json file 
        with open(dataset_prefix + dataset_path, 'r') as f:
            sat_dataset = json.load(f)
        # import pdb; pdb.set_trace()
        dataset = [make_conversation_sat(sample) for sample in sat_dataset]

        print("Dataset is ready")

    dataset = CustomDataset(dataset)

    # import pdb; pdb.set_trace()

    ################
    # Define processor
    ################
    def collate_fn(examples):
        # Get the texts and images, and apply the chat template
        texts = [
        processor.apply_chat_template(example, CHAT_TEMPLATE['chat_template'], tokenize=False) for example in examples
            ]  # Prepare texts for processing
        image_inputs = [process_vision_info(example)[0] for example in examples]

        # Tokenize the texts and process the images
        batch = processor(
            text=texts, images=image_inputs, return_tensors="pt", padding=True
        )  # Encode texts and images into tensors

        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels = batch["input_ids"].clone()  # Clone input IDs for labels
        labels[labels == processor.tokenizer.pad_token_id] = -100  # Mask padding tokens in labels

        # Ignore the image token index in the loss computation (model specific)
        if isinstance(processor, Qwen2VLProcessor):  # Check if the processor is Qwen2VLProcessor
            image_tokens = [151652, 151653, 151655]  # Specific image token IDs for Qwen2VLProcessor
        else:
            image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]  # Convert image token to ID

        # Mask image token IDs in the labels
        for image_token_id in image_tokens:
            labels[labels == image_token_id] = -100  # Mask image token IDs in labels
            
        batch["labels"] = labels  # Add labels to the batch

        return batch

    ################
    # Training
    ################
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, 
                                                    torch_dtype=torch.bfloat16,
                                                    attn_implementation="flash_attention_2",
    )

    max_pixels = 512*28*28
    model.visual.requires_grad_ = True
    processor = Qwen2VLProcessor.from_pretrained(model_args.model_name_or_path, max_pixels=max_pixels, padding_side='right')
    processor.chat_template = CHAT_TEMPLATE["chat_template"]

    training_args.model_init_kwargs = None
    training_args.dataset_text_field = ""
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
        tokenizer=processor.tokenizer,
        data_collator=collate_fn,
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    # print(training_args)
    main(script_args, training_args, model_args)
