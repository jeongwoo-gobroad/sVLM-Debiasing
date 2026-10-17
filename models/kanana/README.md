---
license: other
license_name: kanana
license_link: LICENSE
language:
- ko
- en
base_model:
- kakaocorp/kanana-1.5-v-3b-instruct
pipeline_tag: image-text-to-text
library_name: transformers
---

<p align="center">
<br>
    <picture>
        <img src="./assets/logo/kanana-logo.png" width="60%" style="margin: 40px auto;">
    </picture>
</br>

<p align="center">
🤗 <a href="https://kko.kakao.com/kananallm">1.5 HF Models</a> &nbsp |
&nbsp 📕 <a href="https://tech.kakao.com/posts/714">Blog</a> &nbsp

<br>

## Table of Contents

- [Kanana-1.5-v-3b-instruct](#kanana-15-v-3b-instruct)
- [Intended Use](#intended-use)
- [Model Details](#model-details)
- [Evaluation](#evaluation)
  - [Model Configuration Summary](#model-configuration-summary)
  - [Overview](#overview)
  - [Image Benchmarks (EN)](#image-benchmarks-en)
  - [Image Benchmarks (KO)](#image-benchmarks-ko)
  - [Multimodal Instruction Following Benchmarks (EN, KO)](#multimodal-instruction-following-benchmarks-en-ko)
  - [Note on Benchmarking Methodology](#note-on-benchmarking-methodology)
- [Usage](#usage)
  - [Requirements](#requirements)
  - [Quickstart](#quickstart)
- [Limitations](#limitations)
- [Contributors](#contributors)
- [Contact](#contact)

<br>


# kanana-1.5-v-3b-instruct

The Unified Foundation Model (UFO) task force of Kanana at Kakao developed and released the Kanana-V family of multimodal large language models (MLLMs), a collection of pretrained text/image-to-text (TI2T) models.



## Intended Use

kanana-1.5-v-3b-instruct is intended for research and application development in multimodal understanding and text generation tasks. Typical use cases include image captioning, document understanding, OCR-based reasoning, and multimodal instruction following in both English and Korean. The model is optimized for both general-purpose and Korea-specific benchmarks, making it suitable for bilingual environments.




## Model Details

- **Developed by:** Unified Foundation Model (UFO) TF at Kakao
- **Language(s) :** ['en', 'ko']
- **Model Architecture:** kanana-1.5-v-3b-instruct has 3.6B parameters and contains image encoder, C-abstractor, and kanana-1.5-3b-instruct language model.
- **Input:** The models accept text and image inputs.
- **Output:** The models generate text only.
- **Context Length:** 32k
- **Knowledge Cutoff Date:** June 30, 2024
- **Model Release Date:** Jul 24, 2025.
- **License:** kanana-license
 



## Evaluation

### Model Configuration Summary

| Model                      | LLM                              | Total Parameter |
|----------------------------|----------------------------------|-----------|
| **kanana-1.5-v-3b-instruct**        | kanana-1.5-3b-instruct  | 3.67B     |
| HCX-SEED-Vision-3B         | HyperCLOVAX-SEED-Text-Base-3B    | 3.72B     |
| Phi-3-Vision               | Phi-3-Mini                       | 4.15B     |
| Qwen2.5-VL-3B-Instruct     | Qwen2.5-3B                       | 3.75B     |
| InternVL2.5-4B             | Qwen2.5-3B-Instruct              | 3.94B     |

### Overview

| Model                      | All    | Image (EN) | Image (KO) | IF (EN, KO) |
|----------------------------|--------|------------|------------|-------------|
| **kanana-1.5-v-3b-instruct**        | 73.22  | 74.00      | 68.27      | 77.39       |
| HCX-SEED-Vision-3B         | 59.00  | 64.81      | 51.96      | 60.23       |
| Phi-3-Vision               | 48.84  | 65.41      | 36.40      | 44.71       |
| Qwen2.5-VL-3B-Instruct     | 63.54  | 73.97      | 60.60      | 56.04       |
| InternVL2.5-4B             | 61.35  | 74.73      | 54.68      | 54.63       |

### Image Benchmarks (EN)

| Model                      | average | MMMU (Val) | MathVista | DocVQA | ChartQA | OCRBench | InfoVQA | TextVQA | RealWorldQA | MMStar | MMB   | SEED-image | MMVet | LLaVA-Wild | scienceqa | AI2D  |
|----------------------------|--------------|------------|-----------|--------|---------|----------|---------|---------|-------------|--------|-------|------------|-------|------------|-----------|-------|
| **kanana-1.5-v-3b-instruct**            | 74.00        | 43.89      | 56.00     | 93.06  | 81.20   | 82.50    | 73.62   | 78.62   | 65.36       | 56.32  | 78.44 | 75.17      | 65.87 | 89.60      | 95.61     | 74.81 |
| HCX-SEED-Vision-3B         | 64.81        | 38.89      | 47.40     | 79.87  | 71.88   | 62.90    | 55.59   | 73.51   | 62.48       | 46.66  | 72.42 | 74.84      | 47.27 | 79.30      | 86.84     | 72.31 |
| Phi-3-Vision               | 65.41        | 45.33      | 43.60     | 87.04  | 81.40   | 63.60    | 54.80   | 69.61   | 59.08       | 47.47  | 73.37 | 71.69      | 45.96 | 70.40      | 90.84     | 76.98 |
| Qwen2.5-VL-3B-Instruct     | 73.97        | 50.67      | 62.00     | 94.19  | 83.60   | 79.10    | 77.22   | 77.77   | 59.74       | 56.26  | 77.75 | 74.83      | 61.06 | 96.90      | 79.69     | 78.79 |
| InternVL2.5-4B             | 74.73        | 52.33      | 61.80     | 92.13  | 82.76   | 79.20    | 69.73   | 78.24   | 62.88       | 59.72  | 81.96 | 75.59      | 61.38 | 86.30      | 97.14     | 79.83 |


### Image Benchmarks (KO)

| Model                      | average | KoOCRBench | KoMMDBench | KoChartTask | KoMathSolution | KoCosMed | KoFoodMenu | KoEntity | KoExam | KoCelebV2 |
|----------------------------|--------------|----------------------|------------|-------------|----------------|----------|------------|----------|--------|-----------|
| **kanana-1.5-v-3b-instruct**            | 68.27        | 85.93                | 74.00      | 84.96       | 36.88          | 87.58    | 70.84      | 72.04    | 58.99  | 43.24     |
| HCX-SEED-Vision-3B         | 51.96        | 32.91                | 64.57      | 73.55       | 27.88          | 78.16    | 57.08      | 64.12    | 31.82  | 37.58     |
| Phi-3-Vision               | 36.40        | 25.13                | 37.93      | 52.36       | 38.75          | 56.75    | 34.70      | 31.71    | 24.05  | 26.25     |
| Qwen2.5-VL-3B-Instruct     | 60.60        | 50.67                | 61.75      | 84.96       | 47.13          | 82.01    | 66.32      | 58.15    | 60.68  | 33.72     |
| InternVL2.5-4B             | 54.68        | 20.52                | 62.65      | 82.61       | 46.50          | 82.66    | 65.09      | 50.42    | 47.43  | 34.23     |

### Multimodal Instruction Following Benchmarks (EN, KO)

| Model                      | average      | MIABench | MIABench-Ko | MM-IFEval | MM-OmniAlign |
|----------------------------|--------------|----------|-------------|-----------|--------------|
| **kanana-1.5-v-3b-instruct**            | 77.39        | 90.28    | 91.17       | 56.67     | 71.43        |
| HCX-SEED-Vision-3B         | 60.23        | 85.81    | 81.80       | 47.91     | 25.40        |
| Phi-3-Vision               | 44.71        | 85.78    | 38.35       | 44.37     | 10.32        |
| Qwen2.5-VL-3B-Instruct     | 56.04        | 82.55    | 59.61       | 39.14     | 42.86        |
| InternVL2.5-4B             | 54.63        | 85.68    | 68.35       | 43.06     | 21.43        |



### Note on Benchmarking Methodology

All benchmarks were re-measured under identical software conditions to ensure fair comparison.

- **[VLMEvalKit](https://github.com/open-compass/VLMEvalKit)** was used for MMMU, MathVista, ScienceQA, MIA-Bench, MM-IFEval and MM-OmniAlign.

- **[lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)** was employed for DocVQA, ChartQA, OCRBench, InfoVQA, TextVQA, RealWorldQA, MMStar, MMB, and SEED-image.

- HCX-SEED-Vision-3B was evaluated without the use of any auxiliary tools (e.g., external OCR engines or Lens features), as such tools are not publicly available and therefore not included in our evaluation setup.

- **Important note for ChartQA**: It was identified that the original rule-based parser used by lmms-eval marked answers ending with a period (".") as incorrect due to parsing issues. To address this, the parser logic was modified to remove any trailing period before parsing the response. All ChartQA evaluations presented here reflect results obtained after applying this parser adjustment.


The following in-house benchmarks evaluate Korean-language tasks and Korea-specific knowledge:

| Benchmark | Purpose |
|-----------|---------|
| **KoOCRBench** | Korean character recognition (OCR) |
| **KoMMDBench**, **KoEntity**, **KoCelebV2** | Korean knowledge & cultural visual QA |
| **KoFoodMenu**, **KoCosMed** | Korean text-based visual QA |
| **KoChartTask** | Chart understanding in Korean |
| **KoExam**, **KoMathSolution** | Multimodal Problem-solving in Korean (general exams & mathematics) |
| **MIABench-Ko** | Korean multimodal instruction-following benchmark (derived from MIABench) |



## Usage

### Requirements

```
pip install transformers accelerate timm omegaconf
```
`transformers>=4.45.0` or the latest version is recommended.

### Quickstart

The following is a code snippet that briefly demonstrates how to load a model and process input data using the `AutoClass` from `transformers`.
```python
from PIL import Image
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

MODEL = "kakaocorp/kanana-1.5-v-3b-instruct"

# Load the model on the available device(s)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
model.eval()

# Load processor
processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)

# Prepare input batch
batch = []
for _ in range(1):  # dummy loop to demonstrate batch processing
    image_files = [
        "./examples/waybill.png"
    ]

    sample = {
        "image": [Image.open(image_file_path).convert("RGB") for image_file_path in image_files],
        "conv": [
            {"role": "user", "content": " ".join(["<image>"] * len(image_files))},
            {"role": "user", "content": "사진에서 보내는 사람과 받는 사람 정보를 json 형태로 정리해줘."},
        ]
    }

    batch.append(sample)
    
inputs = processor.batch_encode_collate(
    batch, padding_side="left", add_generation_prompt=True, max_length=8192
)
inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

# Set the generation config
gen_kwargs = {
    "max_new_tokens": 2048,
    "temperature": 0,
    "top_p": 1.0,
    "num_beams": 1,
    "do_sample": False,
}

# Generate text
gens = model.generate(
    **inputs,
    **gen_kwargs,
)
text_outputs = processor.tokenizer.batch_decode(gens, skip_special_tokens=True)
print(text_outputs)  # ['```json\n{\n  "보내는분": {\n    "성명": "카카오",\n    "주소": "경기도 성남시 판교역로 166"\n  },\n  "받는분": {\n    "성명": "카나나",\n    "주소": "제주도 제주시 첨단로 242"\n  }\n}\n```']
```



## Limitations

- The model may generate inaccurate or misleading content, especially in scenarios requiring precise factual understanding (e.g., scientific diagrams or mathematical reasoning).
- Performance on languages other than Korean and English has not been evaluated and may be poor.
- The model is not designed for medical, legal, or other high-stakes domains.
- The model may reflect social biases present in the pretraining data.



## Contributors
- Beomhee Park, Byeonguk Bae, Byungseok Roh, Daejin Jo, Donghee Son, Dongjin Lee, Hyunwoong Ko, Jaemyung Lee, Jeehye Lee, Sunghun Kang, Wooyoung Kang
- Listed in alphabetical order (first name)



## Contact
- Kanana MLLM Core Team Technical Support: kanana-mllm@kakaocorp.com
- Business & Partnership Contact: alpha.k@kakaocorp.com