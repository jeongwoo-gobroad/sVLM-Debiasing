# (c) 2026 Jeongwoo Kim from KNU CSE, All rights reserved
# sVLM Inference-time Debiasing Research
from pathlib import Path
import os
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("SVLM_DATA_ROOT", REPO_ROOT / "datasets")).expanduser().resolve()

import os
import transformers.dynamic_module_utils
transformers.dynamic_module_utils.resolve_trust_remote_code = lambda *args, **kwargs: True

import json
import re
import torch
import torch.nn.functional as F
import numpy as np

import argparse

from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration, AutoTokenizer, LogitsProcessor, LogitsProcessorList

# VCD 하이퍼파라미터 (논문 기본값)
ALPHA = 1.0   # contrastive 강도 (Eq.3)
BETA  = 0.1   # adaptive plausibility constraint (Eq.4)
T     = 500   # diffusion 스텝 수

parser = argparse.ArgumentParser()
parser.add_argument("--constrained", action="store_true", help="Apply the copied trie mask before VCD and compute beta cutoff within allowed tokens")
parser.add_argument("--alpha", type=float, default=ALPHA)
parser.add_argument("--beta",  type=float, default=BETA)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--audit_visual_inputs", action="store_true", help="Compare actual visual and token-only forward paths on the first image")
parser.add_argument("--T",     type=int,   default=T)
parser.add_argument('--prompt_variant', choices=['default', 'a', 'b', 'c'], default='default',
                    help='default or paper Prompt 1/2/3 (a/b/c)')
parser.add_argument('--limit', type=int, default=0, help='Smoke test image count; 0 means the full split')
parser.add_argument('--output_dir', type=Path, default=REPO_ROOT / 'results' / 'test')
parser.add_argument('--revision', help='Override the recorded local-cache model revision')
args = parser.parse_args()
if not 0 <= args.T < 1000 or not 0 < args.beta <= 1:
    parser.error("Require 0 <= T < 1000 and 0 < beta <= 1")
if args.limit < 0:
    parser.error('--limit must be nonnegative')
if getattr(args, 'deb', False) and not args.con:
    parser.error('--debiasing requires --constrained')
args.output_dir.mkdir(parents=True, exist_ok=True)


ALPHA    = args.alpha
BETA     = args.beta
T        = args.T

stat_save_file_name = str(args.output_dir / "qwen3.5_4b/qwen4b_vcd")
stat_save_file_name += "" if args.prompt_variant == "default" else f"_prompt[{args.prompt_variant}]"

pics_folder = str(DATA_ROOT / "imgs/test")
tags_file = str(DATA_ROOT / "tag_candidates/refined_tags.txt")
ground_truth_file = str(DATA_ROOT / "tag_gts/human_gt_test.jsonl")
MODEL_NAME           = "Qwen/Qwen3.5-4B"
MODEL_REVISION = args.revision or json.loads((REPO_ROOT / "models/revisions.json").read_text(encoding="utf-8"))[MODEL_NAME]["revision"]
MAX_IMG_SIDE         = 720

final_stat_save_file = f"{stat_save_file_name}{'_trie' if args.constrained else ''}.txt"

# ==========================================
# 1. 데이터 로드
# ==========================================
Path(final_stat_save_file).parent.mkdir(parents=True, exist_ok=True)

with open(tags_file, "r", encoding="utf-8") as f:
    tags = [line.strip() for line in f if line.strip()]

with open(tags_file, "r", encoding="utf-8") as f:
    tags_raw_1 = [" " + line.strip() for line in f if line.strip()]

tags_raw = []
tags_raw.extend(tags)
tags_raw.extend(tags_raw_1)

ground_truth = {}
with open(ground_truth_file, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            ground_truth.update(json.loads(line.strip()))

image_filenames = []
for fname in os.listdir(pics_folder):
    if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')) and fname in ground_truth:
        image_filenames.append(fname)

if set(image_filenames) != set(ground_truth):
    raise ValueError('Image filenames and ground-truth keys do not match')
if args.limit:
    image_filenames = image_filenames[:args.limit]
if not image_filenames:
    raise ValueError('No images found')

model = Qwen3_5ForConditionalGeneration.from_pretrained(
    MODEL_NAME, revision=MODEL_REVISION,
    dtype="auto",
    device_map="auto"
)
model.eval()
processor = AutoProcessor.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)

eos_id = tokenizer.eos_token_id
if eos_id is None:
    eos_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]

if args.constrained:
    from vcd_trie import TokenTrie, TrieLogitsProcessor
    stt_trie, space_trie = TokenTrie(), TokenTrie()
    for tag in tags:
        stt_trie.insert(tokenizer.encode(tag, add_special_tokens=False), tag.replace(",", "").strip())
        space_trie.insert(tokenizer.encode(" " + tag, add_special_tokens=False), tag.replace(",", "").strip())

tags_str = "\n".join(tags)
system_prompt = {
    'default': "정해진 단어로만 답하십시오.\n\n",
    'a': "",
    'b': "'이 이미지에서 볼 수 있는 것들은'과 같은 불필요한 응답을 생성하지 말고, 유저의 지시에 따르십시오.",
    'c': "불필요한 말과 문장을 생성하지 마십시오. 정해진 단어로만 답하십시오.\n\n",
}[args.prompt_variant]
user_prompt_text = f"""<tags>\n{tags_str}\n</tags>\n\n위 <tags> 내 단어 중, 이 사진에서 볼 수 있는 핵심 객체들을 , 로 구분하여 나열하십시오."""

def add_diffusion_noise(image_tensor, noise_step, seed):
    # The official VCD sigmoid schedule is applied to processor-normalized pixels.
    betas = torch.sigmoid(torch.linspace(-6, 6, 1000)) * (0.5e-2 - 1e-5) + 1e-5
    alpha_bar = torch.cumprod(1 - betas, dim=0)[noise_step]
    generator = torch.Generator(device='cpu').manual_seed(seed)
    noise = torch.randn(image_tensor.shape, generator=generator, dtype=torch.float32)
    return (alpha_bar.sqrt() * image_tensor.float().cpu() + (1 - alpha_bar).sqrt() * noise).to(image_tensor.device, image_tensor.dtype)

class VCDLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        dist_inputs: dict,   # v' 에 대한 processor 출력 (input_ids, attention_mask 등)
        dist_prompt_length: int,
        alpha: float,
        beta: float,
    ):
        self.dist_inputs        = dist_inputs
        self.dist_prompt_length = dist_prompt_length
        self.alpha              = alpha
        self.beta               = beta

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 현재 생성된 토큰 시퀀스 (prompt 이후 부분)
        generated_ids = input_ids[0, self.dist_prompt_length:].tolist()

        # print(len(generated_ids)) # 디버그용
        dist_input_ids = self.dist_inputs["input_ids"]
        if generated_ids:
            gen_tensor     = torch.tensor([generated_ids], dtype=torch.long, device=dist_input_ids.device)
            dist_input_ids = torch.cat([dist_input_ids, gen_tensor], dim=1)

        dist_kwargs = dict(self.dist_inputs)
        dist_kwargs['input_ids'] = dist_input_ids
        dist_kwargs['attention_mask'] = torch.ones_like(dist_input_ids)
        if 'mm_token_type_ids' in dist_kwargs and generated_ids:
            extra = torch.zeros((1, len(generated_ids)), dtype=dist_kwargs['mm_token_type_ids'].dtype, device=dist_input_ids.device)
            dist_kwargs['mm_token_type_ids'] = torch.cat([dist_kwargs['mm_token_type_ids'], extra], dim=1)
        # Each branch receives its visual tensors; no mutable KV cache is shared.
        with torch.no_grad():
            dist_out = model(**dist_kwargs, use_cache=False)
            null_logits = dist_out.logits[0, -1, :].float()

        # Eq.3: logit_vcd = (1 + α) * logit_v - α * logit_v'
        vcd_logits = (1.0 + self.alpha) * scores.float() - self.alpha * null_logits

        # Eq.4: Adaptive Plausibility Constraint
        probs_orig = F.softmax(scores.float(), dim=-1)
        max_prob   = probs_orig.max(dim=-1, keepdim=True).values
        mask       = probs_orig < self.beta * max_prob
        vcd_logits[mask] = -float("inf")

        return vcd_logits.to(scores.dtype)

# ==========================================
# 6. 추론 루프
# ==========================================
total_jaccard_distance = 0.0
all_predictions        = {}
valid_filenames        = []

for fname in tqdm(image_filenames, desc="eval"):
    path = os.path.join(pics_folder, fname)

    try:
        orig_img = Image.open(path).convert("RGB")

        w, h = orig_img.size
        if max(w, h) > MAX_IMG_SIDE:
            scale    = MAX_IMG_SIDE / max(w, h)
            orig_img = orig_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": system_prompt},
                {"type": "image_url", "image_url": {"url": path.replace("\\", "/")}},
                {"type": "text",      "text": user_prompt_text},
            ]}
        ]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to("cuda")

        prompt_length         = inputs["input_ids"].shape[1]
        logits_processor_list = LogitsProcessorList()

        import hashlib
        image_seed = (args.seed + int(hashlib.sha256(fname.encode()).hexdigest()[:8], 16)) % (2**63)
        dist_inputs = dict(inputs)
        dist_inputs['pixel_values'] = add_diffusion_noise(inputs['pixel_values'], T, image_seed)
        if args.audit_visual_inputs and not valid_filenames:
            calls = []
            hook = model.model.visual.register_forward_pre_hook(lambda *unused: calls.append(1))
            with torch.no_grad():
                visual_logits = model(**dist_inputs, use_cache=False).logits[0, -1].float()
                visual_calls = len(calls)
                calls.clear()
                token_logits = model(input_ids=dist_inputs['input_ids'], attention_mask=dist_inputs['attention_mask'], use_cache=False).logits[0, -1].float()
                token_calls = len(calls)
            hook.remove()
            audit = {'processor_keys': list(inputs), 'vision_calls_with_pixels': visual_calls, 'vision_calls_tokens_only': token_calls,
                     'max_abs_logit_difference': (visual_logits-token_logits).abs().max().item(),
                     'same_argmax': bool(visual_logits.argmax()==token_logits.argmax())}
            (args.output_dir / 'vcd_visual_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
            assert visual_calls > 0 and token_calls == 0

        vcd_processor = VCDLogitsProcessor(
            dist_inputs=dist_inputs,
            dist_prompt_length=prompt_length,
            alpha=ALPHA,
            beta=BETA,
        )
        if args.constrained:
            # 후보 안에서 beta 기준을 잡아 허용 토큰이 모두 사라지지 않게 한다.
            logits_processor_list.append(TrieLogitsProcessor(
                prompt_length, eos_id, set(), stt_trie, space_trie, tokenizer))
        logits_processor_list.append(vcd_processor)

        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                logits_processor=logits_processor_list,
                eos_token_id=tokenizer.encode("<|im_end|>", add_special_tokens=False)[0],
            )

        new_ids     = gen_ids[0, inputs["input_ids"].shape[1]:].tolist()
        text_output = tokenizer.decode(new_ids, skip_special_tokens=True)
        print(text_output)
        text_output = re.sub(r'[-*]', ',', text_output)

        parsed_tags = [t.strip() for t in text_output.replace(",", " ").replace("\n", " ").split(' ')]
        pred_tags   = set([t.strip() for t in parsed_tags if t.strip()])

        all_predictions[fname] = list(pred_tags)
        valid_filenames.append(fname)

        gt_tags = set(ground_truth.get(fname, []))

        if not pred_tags and not gt_tags:
            j_dist = 0.0
        elif not pred_tags or not gt_tags:
            j_dist = 1.0
        else:
            intersection = len(pred_tags.intersection(gt_tags))
            union        = len(pred_tags.union(gt_tags))
            j_dist       = 1.0 - (intersection / union)

        total_jaccard_distance += j_dist

    except Exception as e:
        raise RuntimeError(f"Inference failed for {fname}") from e

if not valid_filenames:
    raise RuntimeError("No predictions were produced")

average_jaccard_distance = total_jaccard_distance / len(valid_filenames)

os.makedirs(os.path.dirname(os.path.abspath(final_stat_save_file)), exist_ok=True)

with open(final_stat_save_file, "w", encoding="utf-8") as f:
    f.write(f"Average_Jaccard_Distance: {average_jaccard_distance:.4f}\n")
    f.write("-" * 50 + "\n")
    for fname, predicted_tags in all_predictions.items():
        f.write(f"{fname}: {predicted_tags}\n")
Path(final_stat_save_file + ".metadata.json").write_text(json.dumps({"arguments":vars(args), "model_revision": MODEL_REVISION, "samples": len(valid_filenames), "constraint_order": "trie_then_vcd_beta_within_trie" if args.constrained else "vcd_only", "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "vcd_upstream_commit": "d6568ff81b8fd306a49e630df44f2db5c2300191"},default=str,indent=2),encoding="utf-8")
