# sVLM-Debiasing

논문 실험의 실행·평가 코드입니다. 데이터, 모델 가중치, 실험 결과는 포함하지 않습니다. 재현과 실행에는 별도 허가가 필요하지 않습니다. 다른 창작물·프로젝트에 코드를 활용할 때는 [이용 조건](#이용-조건)을 따릅니다.

## 환경

Python 3.12, Windows, NVIDIA CUDA GPU 기준입니다. requirements에 CUDA 13.0용 PyTorch와 패키지 버전을 고정했습니다. 모델별 환경을 분리합니다. Linux 실행은 검증하지 않았습니다.

| 환경 | 모델·용도 | requirements |
|---|---|---|
| `.venv-qwen35` | Qwen 3.5 2B·4B, VCD | [qwen35.txt](requirements/qwen35.txt) |
| `.venv-qwen3vl` | Qwen3-VL 2B·4B | [qwen3vl.txt](requirements/qwen3vl.txt) |
| `.venv-kanana` | Kanana 1.5 Vision 3B | [kanana.txt](requirements/kanana.txt) |
| `.venv-kelip` | KELIP ViT-B/32 | [kelip.txt](requirements/kelip.txt) |
| `.venv-analysis` | 평가·표·그림 생성 | [attention_analysis.txt](requirements/attention_analysis.txt) |

아래 명령은 저장소 루트의 PowerShell에서 실행합니다. 다른 모델 환경도 표의 이름과 requirements 파일로 생성합니다.

```powershell
py -3.12 -m venv .venv-qwen35
./.venv-qwen35/Scripts/python.exe -m pip install -r requirements/qwen35.txt
py -3.12 -m venv .venv-analysis
./.venv-analysis/Scripts/python.exe -m pip install -r requirements/attention_analysis.txt
$env:HF_HOME = "$PWD/.model"
$env:HUGGINGFACE_HUB_CACHE = "$env:HF_HOME/hub"
```

모델은 [고정 리비전](models/revisions.json)에서 내려받습니다. 캐시가 준비된 환경에서는 `$env:HF_HUB_OFFLINE = "1"`로 실행합니다. 예측 파일의 점수만 계산할 경우 [evaluation.txt](requirements/evaluation.txt)를 설치하면 됩니다.

### Kanana

Transformers 4.45.0과 `models/kanana/`의 코드·설정·토크나이저를 함께 사용합니다.

```powershell
py -3.12 -m venv .venv-kanana
./.venv-kanana/Scripts/python.exe -m pip install -r requirements/kanana.txt
./.venv-kanana/Scripts/python.exe tools/prepare_kanana.py
```

준비 도구는 가중치를 내려받고 SHA-256을 검사합니다. 기존 가중치를 복사하려면 `--local_source "모델 폴더"`를 지정합니다. 별도의 완성된 모델 폴더는 `KANANA_MODEL_PATH`에 절대 경로로 지정합니다. 로컬 코드는 `trust_remote_code=True`로 로드합니다.

**Powered by Kanana**. [Kanana License Agreement](models/kanana/LICENSE)와 [Notice](models/kanana/Notice)가 적용됩니다. 제2.2조의 [책임 있는 AI 지침](https://www.kakaocorp.com/page/responsible/detail/guidelinesForResponsibleAI), 재배포 고지, 제4조의 별도 상업적 허가 조건은 [외부 라이선스 고지](THIRD_PARTY_NOTICES.md)를 참조합니다.

### KELIP

Transformers 4.36.2와 `third_party/KELIP` 구현을 사용합니다.

```powershell
py -3.12 -m venv .venv-kelip
./.venv-kelip/Scripts/python.exe -m pip install -r requirements/kelip.txt
./.venv-kelip/Scripts/python.exe kelip_train/kelip.py
./.venv-kelip/Scripts/python.exe kelip_test/kelip_infer.py --threshold 0.33
```

`0.33`은 논문 실험의 임계값입니다. 다시 탐색한 경우 `results/val/kelip/kelip_threshold_best_result.json`의 `best_threshold`를 사용합니다.

## 데이터

데이터는 별도로 준비합니다. `SVLM_DATA_ROOT`를 지정하지 않으면 저장소의 `datasets/`를 사용합니다.

```powershell
$env:SVLM_DATA_ROOT = (Resolve-Path "D:/path/to/datasets").Path
```

```text
datasets/
  imgs/val/val_1.jpg ... val_440.jpg
  imgs/test/test_1.jpg ... test_294.jpg
  tag_candidates/refined_tags.txt
  tag_gts/human_gt_val.jsonl
  tag_gts/human_gt_test.jsonl
  noise/reference_noise.jpg
  manifest.json
```

후보 파일은 한 줄에 태그 하나씩, 총 124개입니다. GT는 `{"test_1.jpg": ["태그1", "태그2"]}` 형식의 JSONL입니다. 검증 440장과 테스트 294장을 모두 준비합니다. 조건 간 후보 순서·GT·기준 노이즈를 동일하게 유지합니다.

```powershell
./.venv-analysis/Scripts/python.exe tools/generate_reference_noise.py --output "$env:SVLM_DATA_ROOT/noise/reference_noise.jpg"
./.venv-analysis/Scripts/python.exe tools/verify_dataset.py
```

노이즈 생성 도구는 논문 실험 JPEG의 해시와 대조합니다. `manifest.json`은 선택 사항이며, 있으면 `[{"destination":"datasets/상대경로", "sha256":"해시"}, ...]` 형식의 파일 해시도 검사합니다.

## 생성

`plain`은 자유 생성, `c`는 Trie, `dc`는 Trie + 보정입니다. 생성 상한은 새 토큰 32개이며 greedy decoding을 사용합니다.

| 모델 | 논문 실험 α |
|---|---:|
| Qwen 3.5 2B / 4B | 0.55 / 0.09 |
| Qwen3-VL 2B / 4B | 0.35 / 0.31 |
| Kanana 1.5 Vision 3B | 0.71 |

```powershell
./.venv-qwen35/Scripts/python.exe seomun_test/qwen_n.py --model_size 4b
./.venv-qwen35/Scripts/python.exe seomun_test/qwen_n.py --model_size 4b --constrained
./.venv-qwen35/Scripts/python.exe seomun_test/qwen_n.py --model_size 4b --constrained --debiasing --ratio_thres 0.09
./.venv-kanana/Scripts/python.exe seomun_test/kanana_n.py --constrained --debiasing --ratio_thres 0.71
```

Qwen3-VL은 `.venv-qwen3vl`에서 `seomun_test/qwen_n_v3.py`를 실행합니다. 기본 출력은 `results/test/<모델>/`입니다. `--output_dir`로 경로, `--limit 1`로 한 장 실행, `--prompt_variant default|a|b|c`로 프롬프트를 지정합니다.

### 계수 탐색 및 대조군

```powershell
./.venv-qwen35/Scripts/python.exe seomun_train/search_thres_qw.py --model_size 4b
./.venv-qwen3vl/Scripts/python.exe seomun_train/search_thres_q3.py --model_size 4b
./.venv-kanana/Scripts/python.exe seomun_train/search_thres_ka.py
./.venv-qwen35/Scripts/python.exe seomun_test/qwen_vcd.py --alpha 1 --beta 0.1 --T 500 --seed 42 --audit_visual_inputs
./.venv-qwen35/Scripts/python.exe seomun_test/qwen_vcd.py --alpha 1 --beta 0.1 --T 500 --seed 42 --audit_visual_inputs --constrained
```

탐색한 α는 `results/val/<모델>/abs2_thres_best_result_None.json`의 `best_n`에 저장됩니다. VCD 시각 입력 검사 결과는 `vcd_visual_audit.json`에 저장됩니다.

### 일괄 실행

```powershell
./.venv-analysis/Scripts/python.exe tools/reproduce.py --phase test --python_qwen .venv-qwen35/Scripts/python.exe --python_qwen3vl .venv-qwen3vl/Scripts/python.exe --python_kanana .venv-kanana/Scripts/python.exe --output results/reproduction
```

`test`는 위 α로 모델·프롬프트·VCD 조건을 순차 실행합니다. 새 α를 사용할 경우 같은 출력 경로에서 `--phase search`, `--phase final` 순서로 실행합니다. `final` 결과는 `results/reproduction/final/test/`에 저장됩니다. `--resume`은 코드·인자가 동일하고 성공 기록이 있는 작업을 건너뜁니다. KELIP은 별도 실행합니다.

## 평가 및 Table 5

```powershell
./.venv-analysis/Scripts/python.exe evaluation/evaluate.py --baseline results/test/qwen3.5_4b/qwen3.5_4b_test_plain.txt --corrected results/test/qwen3.5_4b/qwen3.5_4b_test_dc.txt --output results/evaluation.json
./.venv-analysis/Scripts/python.exe studies/table5_significance.py --reference_root results --output results/table5
```

출력은 평균 자카드 거리, 이미지별 평균 F1, 양측 Wilcoxon 부호순위 검정 p값입니다. Table 5의 p값은 다중 비교 보정 전 값입니다.

일괄 실행 결과는 `evaluation/evaluate.py --reference --reference_root results/reproduction`으로 평가합니다. KELIP을 포함하려면 해당 결과도 `--out_file results/reproduction/test/kelip/kelip_test.txt`로 저장합니다.

## Table 6

Qwen 3.5 4B와 Kanana의 테스트 294장, `plain/c/dc` 전체 응답을 사용합니다. 원문·토큰 ID·종료 사유가 필요하므로 점수용 `.txt` 파일만으로는 분석할 수 없습니다.

```powershell
./.venv-analysis/Scripts/python.exe studies/run_tag_capture.py --python_qwen .venv-qwen35/Scripts/python.exe --python_kanana .venv-kanana/Scripts/python.exe --split test --modes plain c dc --output results/tag_outputs
./.venv-analysis/Scripts/python.exe studies/analyze_plain_errors.py --input results/tag_outputs --output results/plain_errors --data "$env:SVLM_DATA_ROOT"
./.venv-analysis/Scripts/python.exe studies/review_plain_response.py --analysis results/plain_errors
```

기존 응답이 같은 형식이면 재사용합니다. 생성은 `--models kanana`, `--modes plain`으로 범위를 제한하고 `--resume`으로 이어갈 수 있습니다.

`review_queue.jsonl`에 미판정 항목이 있으면 최종 집계가 중단됩니다. 검토 도구에서 이미지와 응답을 확인하고 `--key`, `--tags_json`, `--prose yes|no`, `--reason`으로 판정을 저장합니다. 이후 오류 분석기에 `--overrides results/plain_errors/overrides.jsonl`을 추가해 다시 실행합니다.

```powershell
./.venv-analysis/Scripts/python.exe studies/build_response_ledger.py --input results/tag_outputs --plain_reviews results/plain_errors/pair_reviews.jsonl --data "$env:SVLM_DATA_ROOT" --output results/response_ledger
pwsh -NoProfile -File studies/run_noise_perplexity.ps1 -PythonQwen .venv-qwen35/Scripts/python.exe -PythonKanana .venv-kanana/Scripts/python.exe -Output results/noise_ppl
./.venv-analysis/Scripts/python.exe studies/compare_ppl_concordance.py --captures results/noise_ppl --input results/response_ledger --data "$env:SVLM_DATA_ROOT" --output results/ppl_concordance
```

오류 집계는 `results/plain_errors/counts.csv`, 순위 비교는 `results/ppl_concordance/condition_summary.csv`, `paired_concordance_changes.csv`, `report.json`에 저장됩니다.

## 어텐션 지도

```powershell
./.venv-analysis/Scripts/python.exe studies/prepare_leading_outputs.py --input results/tag_outputs --output results/attention_inputs --candidates "$env:SVLM_DATA_ROOT/tag_candidates/refined_tags.txt"
pwsh -NoProfile -File studies/run_attention_study.ps1 -PythonQwen .venv-qwen35/Scripts/python.exe -PythonKanana .venv-kanana/Scripts/python.exe -Output results/attention -FirstTags results/attention_inputs
./.venv-analysis/Scripts/python.exe studies/analyze_attention.py --input results/attention --first_tags results/attention_inputs --data "$env:SVLM_DATA_ROOT" --figure_height .85
```

Qwen 3.5는 full-attention 층, Kanana는 언어 어텐션 층을 저장합니다. 기존 NPZ로 그림만 다시 그릴 때는 GPU가 필요하지 않습니다. 이 분석은 Table 6과 별도입니다.

## 테스트

```powershell
./.venv-qwen35/Scripts/python.exe -m pip install -r requirements/attention_analysis.txt
./.venv-qwen35/Scripts/python.exe -m unittest discover -s tests -v
```

## 이용 조건

- 열람·다운로드·원본 포크·독립 실행·실험 재현 및 검증: 별도 허가 없이 가능합니다. 실행 환경에 맞춘 경로·설정·코드 조정도 포함합니다.
- 개인 창작물, 별도 연구·소프트웨어 프로젝트, 제품·서비스에 코드 활용: 사전 허가가 필요합니다. 원본·수정본, 영리·비영리, 공개·비공개 여부에 관계없이 적용합니다.
- 재현·검증 결과의 발표: 별도 허가 없이 가능합니다. 저자명과 저장소를 출처로 표시합니다.

허가 요청: **jeongwoo.gobroad@gmail.com**. 버전, 용도, 수정·배포 계획을 기재합니다. 이메일·메시지 또는 대면·비대면의 명시적 구두 허가를 인정합니다. 허가받아 코드를 활용하는 프로젝트는 `Contributions(기여)`에 Jeongwoo Kim을 재사용 코드의 원저자로 표시하고 저장소 링크를 기재합니다.

적용 범위는 [LICENSE](LICENSE)를 따릅니다. 기존 [GPL-3.0](licenses/GPL-3.0-previous.txt) 배포의 권리와 [외부 코드·모델의 라이선스](THIRD_PARTY_NOTICES.md)는 유지됩니다.

## Contributions

Jeongwoo Kim — sVLM-Debiasing 프로젝트 코드 원저자입니다. 외부 구현의 저작권과 출처는 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)에 명시합니다.
