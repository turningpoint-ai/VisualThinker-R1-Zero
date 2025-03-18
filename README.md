# VisualThinker-R1-Zero: First ever R1-Zero's Aha Moment on just a 2B non-SFT Model
[![Notion](https://img.shields.io/badge/Notion-%23000000.svg?style=for-the-badge&logo=notion&logoColor=white)](https://turningpointai.notion.site/the-multimodal-aha-moment-on-2b-model)

![Reinforcement Learning](https://img.shields.io/badge/Algo-Reinforcement--Learning-red) 
![R1](https://img.shields.io/badge/Algo-R1-red) 
![Vision-Centric](https://img.shields.io/badge/Task-Vision--Centric-yellow) 
![Qwen2-VL-2B](https://img.shields.io/badge/Model-Qwen2--VL--2B-green)
![Aha-Moment](https://img.shields.io/badge/Analysis-Aha--moment-blue) 

VisualThinker-R1-Zero is a replication of [DeepSeek-R1-Zero](https://arxiv.org/abs/2501.12948) in visual reasoning. We are **the first** to successfully observe **the emergent “aha moment”** and **increased response length** in **visual reasoning** on just a **2B non-SFT models**.

For more details, please refer to the notion [report](https://turningpointai.notion.site/the-multimodal-aha-moment-on-2b-model).

<div align="center">
<img src="https://multimodal-r1.s3.us-west-1.amazonaws.com/Training_Steps.png" width="700" alt="visualthinking-intro-figure_00">
</div>

> Training dynamics of our VisualThinker-R1-Zero training starting from the Qwen-VL-2B, without SFT or reward models. An aha moment and increasing response length is ever observed at a multimodal model.

## 🔮 Highlights
1. We are the **first to successfully produce the emergent “aha moment” and increased response length** for multimodal reasoning on just a **non-SFT 2B model**.
2. We showed that **vision-centric** tasks could also benefit from improved reasoning capabilities.  

Similar to DeepSeek R1, self reflection behavior is also observed during our RL training on vision-centric reasoning tasks. The model exhibits an emergent ability to rethink and correct its mistakes:

```
. . .
Therefore, dark brown wooden bed with white blanket is not above the doorway.
But wait! I can think of something else.
Maybe it's just higher than above the doorway, but slightly lower than above the doorway.
. . .
```

## 📢 Updates
- 2025-03-16: 🤗We released the model [checkpoint](https://huggingface.co/turningpoint-ai/VisualThinker-R1-Zero) at huggingface!
- 2025-02-26: 🔥We share our main findings in this [notion blog](https://turningpointai.notion.site/the-multimodal-aha-moment-on-2b-model).
- 2025-02-26: 🔥We release the VisualThinker R1 Zero repo.

## 💻 Hardware Requirements

\* *estimated*

| Method                   | Bits |   2B   |
| ------------------------ | ---- | ------ |
| GRPO Full Fine-Tuning    |  AMP | 4*80GB |

## 🧱 Setup

```bash
bash setup.sh
```
## 🤗 Prepare Dataset

```bash
cd src/data/SAT
bash prepare_dataset.sh
```

## 🏋️ Training

### GRPO Training
To reproduce the multimodal aha moment, run the following code to train the non-SFT model with GRPO on SAT:
```bash
cd src/open-r1-multimodal
bash run_grpo_SAT.sh # Adjust open-r1-multimodal/configs/zero3.yaml or zero2.yaml accordingly
```

### SFT Training
To obtain SFT model for comparison, run the following code to train the non-SFT model on SAT:
```bash
cd src/open-r1-multimodal
bash run_sft.sh # Adjust open-r1-multimodal/configs/zero3.yaml or zero2.yaml accordingly
```

## 📈 Evaluation

### CVBench Evaluation
We provide following commands to reproduce our evaluation results on the CVBench. First change to evaluation directory:
```bash
cd src/eval 
```

To evaluate Base + GRPO (VisualThinker R1 Zero) model:
```bash
python evaluate_Qwen2_VL_CVBench-base.py --model_path <path_to_your_model> \
    --bs 8 \
    --use_reasoning_prompt
```
To evaluate Base model:
```bash
python evaluate_Qwen2_VL_CVBench-base.py --model_path <path_to_your_model> \
    --bs 8 \
    --no-use_reasoning_prompt
```
To evaluate Instruct + GRPO model:
```bash
python evaluate_Qwen2_VL_CVBench.py --model_path <path_to_your_model> \
    --bs 8 \
    --use_reasoning_prompt
```
To evaluate Instruct model:
```bash
python evaluate_Qwen2_VL_CVBench.py --model_path <path_to_your_model> \
    --bs 8 \
    --no-use_reasoning_prompt
```
## 🔍 Resources

**Full experiment log:** Upcoming

**Models CKPT:** [🤗VisualThinker-R1-Zero](https://huggingface.co/turningpoint-ai/VisualThinker-R1-Zero) at huggingface

## :coffee: Stay Connected!

We are always open to engaging discussions, collaborations, or even just sharing a virtual coffee. To get in touch or join our team, visit [TurningPoint AI](https://www.turningpoint-ai.com/)'s homepage for contact information.

## 📖 Acknowledgements

We sincerely thank [DeepSeek](https://github.com/deepseek-ai/DeepSeek-R1), [Open-R1](https://github.com/huggingface/open-r1), [QwenVL](https://github.com/QwenLM/Qwen2.5-VL), [Open-R1-Multimodal](https://github.com/EvolvingLMMs-Lab/open-r1-multimodal), [R1-V](https://github.com/Deep-Agent/R1-V), [SAT](https://arxiv.org/abs/2412.07755), and [CV-Bench](https://cambrian-mllm.github.io/) for providing open source resources that laid the foundation of our project. 

## 🤝 Contributors

Here are the key contributors from [TurningPoint AI](https://www.turningpoint-ai.com/) to this project:

[Hengguang Zhou](https://hengguangzhou.github.io/)<sup>1</sup><sup>* </sup>, [Xirui Li](https://xirui-li.github.io/)<sup>1</sup><sup>* </sup>, [Ruochen Wang](https://ruocwang.github.io/)<sup>1</sup><sup>† </sup>, [Minhao Cheng](https://cmhcbb.github.io/)<sup>2</sup>, [Tianyi Zhou](https://tianyizhou.github.io/)<sup>3</sup> and [Cho-Jui Hsieh](https://web.cs.ucla.edu/~chohsieh/)<sup>1</sup><sup>4</sup>

<sup>*</sup> Project Leads, <sup>†</sup> Main Advisor
<sup>1</sup>University of California, Los Angeles, <sup>2</sup>Penn State University, <sup>3</sup>University of Maryland and <sup>4</sup>Google Research


## :white_check_mark: Cite

If you find our work useful for your projects, please kindly cite the following BibTeX:

```latex
@misc{zhou2025r1zerosahamomentvisual,
      title={R1-Zero's "Aha Moment" in Visual Reasoning on a 2B Non-SFT Model}, 
      author={Hengguang Zhou and Xirui Li and Ruochen Wang and Minhao Cheng and Tianyi Zhou and Cho-Jui Hsieh},
      year={2025},
      eprint={2503.05132},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2503.05132}, 
}
```

