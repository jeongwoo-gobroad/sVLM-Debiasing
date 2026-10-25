# Third-party licenses and attribution

The root [LICENSE](LICENSE) covers only material whose rights Jeongwoo Kim controls. It does not replace the licenses below. In particular, `models/kanana/`, `third_party/`, and upstream portions of mixed-source files retain their upstream terms. A project header does not claim ownership of upstream implementations. Model weights and datasets are not distributed in this repository.

| Component | Included or referenced material | Applicable terms |
|---|---|---|
| Qwen3.5 2B/4B; Qwen3-VL 2B/4B | Model identifiers/revisions; implementations loaded through Transformers | Apache-2.0 at the pinned revisions; upstream notices and conditions remain applicable when redistributing model materials. |
| Kakao Kanana 1.5 Vision 3B | Code, tokenizer, configuration, and documentation in `models/kanana/`; weights downloaded separately | [Kanana License Agreement](models/kanana/LICENSE), including prohibited-use, redistribution, and additional commercial-license terms. |
| NAVER KELIP | `third_party/KELIP/` code; weights downloaded separately | [Apache-2.0](third_party/KELIP/LICENSE) for the upstream code, with OpenAI CLIP portions below. Copyright (c) 2022-present NAVER Corp. |
| OpenAI CLIP | CLIP-derived implementation in `third_party/KELIP/kelip/model.py` | [MIT license](third_party/KELIP/LICENSE-CLIP). Copyright (c) 2021 OpenAI. |
| VCD, DAMO-NLP-SG | Diffusion-noise and contrastive-decoding adaptations in `seomun_test/qwen_vcd.py` | Upstream portions: [Apache-2.0](third_party/VCD/LICENSE). Local Qwen/Trie integration is identified in the source. |
| Hugging Face Transformers | External dependency; attention projection/calculation adaptations in `studies/attention_probe.py` | Upstream portions: [Apache-2.0](third_party/transformers/LICENSE). Qwen attention: Copyright 2025 The Qwen Team and The HuggingFace Inc. team. Llama implementation: Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved. |

## Kanana redistribution

**Powered by Kanana**

Kanana is licensed in accordance with the Kanana License Agreement. Copyright © KAKAO Corp. All Rights Reserved.

The required notice is also provided in [models/kanana/Notice](models/kanana/Notice). The five vendored Python implementations match the pinned upstream implementation apart from project/provenance comments. These local header modifications are identified in each file.

Users and distributors must comply with Section 2.2 of the [full agreement](models/kanana/LICENSE), including applicable laws and [KAKAO's Guidelines For Responsible AI](https://www.kakaocorp.com/page/responsible/detail/guidelinesForResponsibleAI). Subsequent recipients must be notified of these restrictions, receive the agreement and required notices, and retain modification notices and the required Kanana attribution. Sections 4.1 and 4.2 require separate KAKAO commercial permission for specified activities or scale; Jeongwoo Kim's permission is not a substitute. Section 4.3 permits other own-service development/operation subject to the agreement. Consult the full terms for the exact scope.

## Upstream sources

Model revisions are fixed in [models/revisions.json](models/revisions.json).

- Qwen3.5 [2B license](https://huggingface.co/Qwen/Qwen3.5-2B/blob/15852e8c16360a2fea060d615a32b45270f8a8fc/LICENSE) and [4B license](https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/LICENSE).
- Qwen3-VL [2B model-card license metadata](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/blob/89644892e4d85e24eaac8bacfd4f463576704203/README.md) and [4B model-card license metadata](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/blob/ebb281ec70b05090aa6165b016eac8ec08e71b17/README.md). A separate `LICENSE` file was not available at these pins; the cards specify Apache-2.0.
- [Kanana pinned license](https://huggingface.co/kakaocorp/kanana-1.5-v-3b-instruct/blob/2e00ef13ccec2e99459a8eada18a1bfd05bff44b/LICENSE).
- [KELIP code and license](https://github.com/navervision/KELIP), [CLIP MIT license](https://github.com/openai/CLIP/blob/main/LICENSE). The KELIP Hugging Face weight revision did not provide a separate license or model card in this audit; the verified code license alone is not treated here as independent confirmation of the weight license. Confirm the provider's applicable terms before redistributing those weights.
- [VCD revision d6568ff](https://github.com/DAMO-NLP-SG/VCD/tree/d6568ff81b8fd306a49e630df44f2db5c2300191).
- Transformers [Qwen3.5 implementation, v5.3.0](https://github.com/huggingface/transformers/blob/v5.3.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py) and [Llama implementation, v4.45.0](https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/models/llama/modeling_llama.py). The latter implementation derives from EleutherAI GPT-NeoX and OPT implementations with architectural adaptations for Llama; this is a code notice, not a grant for Meta model weights.

Dependency packages installed through requirements retain their own licenses. No dependency is relicensed by the root license. Apache-2.0 permits additional terms for original modifications or a derivative work as a whole subject to its notice and other obligations; merely calling a model or binding to its interface does not automatically impose a single license on all project code. Kanana's distinct contractual conditions still apply to its materials and derivatives.

## Prior project releases

The root project license was GPL-3.0 through revision `2717bdbd7d8f0bae1360d62bea4c18d00a2832e5`. Its [license text](licenses/GPL-3.0-previous.txt) and Git history are preserved. Valid prior GPL grants and upstream grants remain effective; the new permission policy does not revoke them or add an attribution condition to their exercise.
