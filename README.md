# sVLM-Debiasing

전통시장 이미지의 다중 레이블 분류를 위한 소형 VLM 추론 시점 편향 보정 코드입니다. 검증셋에서 계수 α를 선택하고 Trie 제약 디코딩과 함께 적용하며, 모델 가중치는 학습하지 않습니다.

최종 제출 논문의 생성·평가와 Table 6 분석을 재현할 수 있습니다. 저장소에는 코드, requirements, 모델 실행 설정만 포함합니다. 이미지·정답·실험 출력·논문 파일은 별도 보관하며, 실행에는 아래 형식의 데이터가 필요합니다.

## 설치

실험 환경은 **Python 3.12 / Windows / NVIDIA CUDA GPU**입니다. requirements는 실험에 사용한 CUDA 13.0 PyTorch와 패키지 버전을 고정합니다. Kanana와 KELIP은 Transformers 버전이 달라 환경을 분리합니다. Linux에서는 `Scripts/python.exe` 대신 `bin/python`을 사용하며, 별도 실행 검증은 하지 않았습니다.

| 환경 | 대상 | requirements |
|---|---|---|
| `.venv-qwen35` | Qwen 3.5 2B·4B, VCD, VCD + Trie | [qwen35.txt](requirements/qwen35.txt) |
| `.venv-qwen3vl` | Qwen3-VL 2B·4B | [qwen3vl.txt](requirements/qwen3vl.txt) |
| `.venv-kanana` | Kanana 1.5 Vision 3B | [kanana.txt](requirements/kanana.txt) |
| `.venv-kelip` | KELIP ViT-B/32 | [kelip.txt](requirements/kelip.txt) |
| `.venv-analysis` | 예측 평가·Table 6·어텐션 그림 | [attention_analysis.txt](requirements/attention_analysis.txt) |

저장소 루트의 PowerShell에서 실행합니다. 다른 환경은 이름과 requirements 파일을 바꿔 설치합니다.

```powershell
py -3.12 -m venv .venv-qwen35
./.venv-qwen35/Scripts/python.exe -m pip install -r requirements/qwen35.txt
py -3.12 -m venv .venv-analysis
./.venv-analysis/Scripts/python.exe -m pip install -r requirements/attention_analysis.txt
$env:HF_HOME = "$PWD/.model"
$env:HUGGINGFACE_HUB_CACHE = "$env:HF_HOME/hub"
```

첫 실행에 [고정 모델 리비전](models/revisions.json)의 가중치를 받습니다. 캐시를 준비한 뒤에는 `$env:HF_HUB_OFFLINE = "1"`로 오프라인 실행할 수 있습니다. GPU 없이 예측 점수만 계산하려면 [evaluation.txt](requirements/evaluation.txt)로도 충분합니다.

### Kanana 실행

Transformers **4.45.0**과 포함된 `models/kanana/` 코드·설정·토크나이저를 함께 사용합니다.

```powershell
py -3.12 -m venv .venv-kanana
./.venv-kanana/Scripts/python.exe -m pip install -r requirements/kanana.txt
./.venv-kanana/Scripts/python.exe tools/prepare_kanana.py
```

준비 도구는 고정 리비전의 두 가중치 파일을 내려받고 SHA-256을 확인합니다. 기존 가중치는 `--local_source "모델 폴더"`로 복사할 수 있습니다. 다른 완성된 폴더를 쓰려면 `KANANA_MODEL_PATH`에 절대 경로를 지정합니다. 로컬 구현은 `trust_remote_code=True`로 로드하며 [모델 라이선스](models/kanana/LICENSE)를 따릅니다.

Kanana의 시각 입력 생성은 새 토큰만 반환하고 Qwen은 입력 토큰도 포함합니다. 전체 응답 저장 도구가 이 차이를 처리하므로 출력 접두부를 임의로 다시 잘라내지 않습니다.

### KELIP 실행

포함된 `third_party/KELIP` 구현과 Transformers **4.36.2** 환경을 사용합니다. 첫 실행에 가중치와 토크나이저를 받습니다. 검증 이미지·텍스트 유사도로 임계값을 선택한 뒤 테스트에 전달합니다.

```powershell
py -3.12 -m venv .venv-kelip
./.venv-kelip/Scripts/python.exe -m pip install -r requirements/kelip.txt
./.venv-kelip/Scripts/python.exe kelip_train/kelip.py
./.venv-kelip/Scripts/python.exe kelip_test/kelip_infer.py --threshold 0.33
```

`0.33`은 제출 실험의 선택값입니다. 새 탐색에서는 출력된 `best_threshold`를 사용합니다.

## 입력 데이터

`SVLM_DATA_ROOT`를 데이터 폴더의 절대 경로로 지정합니다. 미지정 시 저장소 아래 `datasets/`를 사용합니다. 이 저장소에서 데이터는 자동으로 내려받지 않습니다.

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
  manifest.json                 # 파일 해시 검증용, 선택 사항
```

후보는 한 줄에 한 태그인 **124개** 목록입니다. GT는 각 줄이 `{"test_1.jpg": ["태그1", "태그2"]}`인 JSONL입니다. 모든 조건에서 동일한 후보 순서·GT·노이즈를 사용합니다. 도구는 검증 440장·테스트 294장 구성을 검사하며, 전체 응답 저장 시에도 두 split을 준비합니다.

기준 노이즈가 없으면 다음 명령으로 생성합니다. 도구는 제출 실험 JPEG의 SHA-256(`f5e0722fda2a4a6f307b5df242658d818845958140e002d307f907975624ef62`)과 일치하는지 확인합니다.

```powershell
./.venv-analysis/Scripts/python.exe tools/generate_reference_noise.py --output "$env:SVLM_DATA_ROOT/noise/reference_noise.jpg"
./.venv-analysis/Scripts/python.exe tools/verify_dataset.py
```

`manifest.json`이 있으면 `[{"destination":"datasets/상대경로", "sha256":"해시"}, ...]`의 파일 해시도 검사합니다. 없으면 키·후보·이미지 중복 및 파일 판독 검사를 수행합니다.

## 태그 생성과 계수 탐색

`plain`은 자유 생성, `c`는 Trie, `dc`는 Trie + 편향 보정입니다. 최대 **32개 새 토큰**, greedy decoding을 사용하며 첫 위치 EOS도 허용합니다.

| 모델 | 선택 α |
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

Qwen3-VL은 `seomun_test/qwen_n_v3.py`를 사용합니다. 출력은 `results/test/<모델>/`이며 `--output_dir`로 변경합니다. `--limit 1`로 한 장을 확인할 수 있고 `--prompt_variant default|a|b|c`로 프롬프트를 선택합니다.

제출 실험의 디코더를 보존합니다. 보정은 최초·쉼표 뒤 시작 위치와 완성된 Trie 노드의 쉼표/EOS/후속 자식 선택에 적용됩니다. 최초와 완성 노드는 normal 기준, 쉼표 뒤는 space 기준을 사용합니다. EOS는 정규화에 포함하지만 차감하지 않습니다. Kanana의 space 기준은 후보 목록 각 줄에 선행 공백을 추가한 프롬프트에서 사전 계산한 초기 로짓입니다.

```powershell
# 검증셋에서 새 계수 선택
./.venv-qwen35/Scripts/python.exe seomun_train/search_thres_qw.py --model_size 4b
./.venv-qwen3vl/Scripts/python.exe seomun_train/search_thres_q3.py --model_size 4b
./.venv-kanana/Scripts/python.exe seomun_train/search_thres_ka.py
# Qwen 3.5 4B의 VCD 대조군
./.venv-qwen35/Scripts/python.exe seomun_test/qwen_vcd.py --alpha 1 --beta 0.1 --T 500 --seed 42 --audit_visual_inputs
./.venv-qwen35/Scripts/python.exe seomun_test/qwen_vcd.py --alpha 1 --beta 0.1 --T 500 --seed 42 --audit_visual_inputs --constrained
```

계수 탐색은 [0,1]에서 0.1 간격, 최적점 주변에서 0.01 간격으로 진행합니다. 검증 자카드 거리의 소수 넷째 자리 값을 최소화하며 동률은 먼저 평가한 값을 유지합니다. 선택값은 `results/val/<모델>/abs2_thres_best_result_None.json`의 `best_n`입니다.

VCD는 왜곡 시각 텐서와 1,000단계 sigmoid 노이즈의 인덱스 500을 사용합니다. VCD + Trie의 순서는 정상 로짓의 Trie 마스크 → VCD 대조 → 허용 후보 안의 β 제한입니다. 시각 입력 검사는 `vcd_visual_audit.json`에 저장합니다.

전체 생성형 모델의 순차 실행은 다음과 같습니다. KELIP은 위 명령으로 별도 실행합니다.

```powershell
./.venv-analysis/Scripts/python.exe tools/reproduce.py --phase test --python_qwen .venv-qwen35/Scripts/python.exe --python_qwen3vl .venv-qwen3vl/Scripts/python.exe --python_kanana .venv-kanana/Scripts/python.exe --output results/reproduction
```

`test`는 제시된 α로 plain/c/dc·프롬프트·VCD를 실행합니다. `search` 다음 `final`을 실행하면 새로 찾은 α의 dc 결과를 얻습니다. `--resume`은 성공 기록·실행 스크립트 해시·인자가 같은 작업을 건너뜁니다. 로그와 단계별 상태 파일은 출력 폴더에 남습니다.

## 점수와 Table 5

```powershell
./.venv-analysis/Scripts/python.exe evaluation/evaluate.py --baseline results/test/qwen3.5_4b/qwen3.5_4b_test_plain.txt --corrected results/test/qwen3.5_4b/qwen3.5_4b_test_dc.txt --output results/evaluation.json
./.venv-analysis/Scripts/python.exe studies/table5_significance.py --reference_root results --output results/table5
```

예측 파일과 GT의 테스트 294개 키 일치를 검사하고 평균 자카드 거리·이미지별 평균 F1·양측 Wilcoxon 부호순위 검정 p값을 계산합니다. Table 5는 c와 dc 각각을 plain과 비교하며 자기 비교 p값은 비워 둡니다. dc 대 c는 별도 보존합니다. 이 p값은 다중 비교 보정 전입니다.

`evaluation/evaluate.py --reference --reference_root results/reproduction`은 해당 루트의 `test/` 아래 전체 모델·프롬프트·VCD·KELIP 결과를 함께 평가합니다. 이를 사용할 때 KELIP 출력도 `--out_file results/reproduction/test/kelip/kelip_test.txt`로 저장합니다.

## Table 6: 기본 응답 오류와 노이즈 PPL 선호 순위

Qwen 3.5 4B와 Kanana의 공통 테스트 **294장 × 3조건**을 사용합니다. 전체 응답 원문·토큰 ID·EOS/길이 상한 종료 여부가 필요하므로 다음 도구로 저장합니다. 이미 같은 형식으로 저장했다면 재생성하지 않습니다.

```powershell
./.venv-analysis/Scripts/python.exe studies/run_tag_capture.py --python_qwen .venv-qwen35/Scripts/python.exe --python_kanana .venv-kanana/Scripts/python.exe --split test --modes plain c dc --output results/tag_outputs
./.venv-analysis/Scripts/python.exe studies/analyze_plain_errors.py --input results/tag_outputs --output results/plain_errors --data "$env:SVLM_DATA_ROOT"
```

`--models kanana` 또는 `--modes plain`으로 일부만 실행하고, 같은 코드·설정으로 중단된 실행은 `--resume`으로 이어갑니다. 기본 점수용 `.txt`에는 완전한 원문과 종료 토큰 정보가 없으므로 그것만으로 잘린 말미를 추정하지 않습니다.

Table 6(a)는 **자유 생성만** 지시 불이행(태그 목록 외 설명)과 후보 밖 생성으로 분류합니다. 두 오류는 이미지별로 독립·중복 집계합니다. 길이 상한 종료에서는 마지막 미완성 조각만 제외하고 앞서 완성된 항목과 정확한 후보 말미는 유지합니다. EOS 종료의 비후보 항목은 제외하지 않습니다.

모호한 문장/명사구는 `review_queue.jsonl`에 남으며 최종 집계가 중단됩니다. 아래 도구는 이미지 경로와 응답을 한 건씩 보여 줍니다. 이미지를 함께 확인한 뒤 `--key`, `--tags_json`, `--prose yes|no`, `--reason`으로 판정을 저장하고, 분석기에 `--overrides results/plain_errors/overrides.jsonl`을 추가하여 다시 실행합니다. 원문·평가 부분의 해시가 달라지면 기존 판정을 재사용하지 않습니다.

```powershell
./.venv-analysis/Scripts/python.exe studies/review_plain_response.py --analysis results/plain_errors
# 검토가 끝난 자료로 세 조건의 완성 태그 원장을 생성
./.venv-analysis/Scripts/python.exe studies/build_response_ledger.py --input results/tag_outputs --plain_reviews results/plain_errors/pair_reviews.jsonl --data "$env:SVLM_DATA_ROOT" --output results/response_ledger
# 고정 노이즈 이미지에 후보 124개를 각각 teacher forcing
pwsh -NoProfile -File studies/run_noise_perplexity.ps1 -PythonQwen .venv-qwen35/Scripts/python.exe -PythonKanana .venv-kanana/Scripts/python.exe -Output results/noise_ppl
./.venv-analysis/Scripts/python.exe studies/compare_ppl_concordance.py --captures results/noise_ppl --input results/response_ledger --data "$env:SVLM_DATA_ROOT" --output results/ppl_concordance
```

후보 태그 본문의 토큰별 음의 로그 확률 평균을 지수화한 PPL을 계산합니다. 프롬프트·쉼표·EOS는 점수에서 제외하고, 전체 어휘 softmax와 모델의 모든 층을 사용합니다. PPL이 낮은 태그가 높은 선호 순위입니다. 태그 접미부 없이 같은 예측을 계산하는 causal-prefix 검사도 저장합니다.

오답 빈도는 각 조건에서 **GT에 없는 완성된 후보를 생성한 이미지 수**입니다. 이미지 안의 반복은 한 번 세고 후보 밖 오답은 별도 집계합니다. 0회 후보도 포함한 124개 전체의 태그 쌍을 비교하여 `C=(선호 순서와 일치하는 쌍 + 0.5×빈도 동률 쌍) / 비교 가능한 쌍`을 계산합니다. PPL 자체가 정확히 같은 쌍만 제외합니다.

Table 6(b)의 검정은 이미지별 두 조건의 오답 벡터 전체를 교환하는 양측 순열검정 **99,999회, seed 42**, plus-one p값입니다. 두 모델 × 세 비교(plain→c, plain→dc, c→dc)의 **여섯 p값에 Holm 보정**을 적용합니다. 논문에는 plain→c와 plain→dc 네 행을 표시하되 여섯 비교의 보정값을 유지합니다. `ΔC<0`은 노이즈 PPL 선호와의 일치가 감소했다는 뜻이며, 인과성이나 모델 간 효과 차이를 검정하는 값은 아닙니다.

출력은 `condition_summary.csv`, `paired_concordance_changes.csv`, 후보·쌍별 CSV, 이미지별 오류 행렬, 순열 원자료, 해시가 포함된 `report.json`입니다. 기본 응답 오류 집계는 `plain_errors/counts.csv`에 저장됩니다. 이전 탐색 분석을 실행할 필요는 없습니다.

## 어텐션 재현 (선택)

최종 논문에 포함하지 않은 어텐션 지도도 저장된 전체 응답으로 재현할 수 있습니다.

```powershell
./.venv-analysis/Scripts/python.exe studies/prepare_leading_outputs.py --input results/tag_outputs --output results/attention_inputs --candidates "$env:SVLM_DATA_ROOT/tag_candidates/refined_tags.txt"
pwsh -NoProfile -File studies/run_attention_study.ps1 -PythonQwen .venv-qwen35/Scripts/python.exe -PythonKanana .venv-kanana/Scripts/python.exe -Output results/attention -FirstTags results/attention_inputs
./.venv-analysis/Scripts/python.exe studies/analyze_attention.py --input results/attention --first_tags results/attention_inputs --data "$env:SVLM_DATA_ROOT" --figure_height .85
```

첫 새 토큰을 예측하는 마지막 프롬프트 query에서 시각 임베딩·후보 태그·기타 프롬프트를 구분합니다. 실제 토큰 ID, 관찰 전후 로짓, 공식 eager 계산식과 저장된 첫 토큰 일치를 검증합니다. Qwen 3.5는 softmax full-attention 층만, Kanana는 언어 어텐션 층을 관찰합니다. 선형 Gated DeltaNet 층을 일반 어텐션 확률 지도로 해석하지 않습니다.

그림 색은 softmax 후 확률의 head·이미지 평균을 로그 척도로 표시합니다. 64개 상대 위치 구간과 모든 관찰 층을 보존합니다. 후보 앞 10개 경계 및 첫 태그 보조 통계는 이 도구의 탐색 출력이며 **최종 Table 6의 지표·검정과 다릅니다**. 기존 NPZ를 사용한 그림 재생성에는 GPU가 필요 없습니다.

## 코드 검증과 라이선스

```powershell
./.venv-qwen35/Scripts/python.exe -m pip install -r requirements/attention_analysis.txt
./.venv-qwen35/Scripts/python.exe -m unittest discover -s tests -v
```

테스트는 디코딩·VCD 제약, 응답 말미 처리, PPL 토큰 정렬, 대응 순열검정과 상태 저장을 검사합니다. 전체 모델 추론 재실행을 대신하지는 않습니다.

프로젝트는 기존 [GPL-3.0](LICENSE)을 유지합니다. 포함된 [Kanana](models/kanana/LICENSE), [KELIP](third_party/KELIP/LICENSE), [VCD](third_party/VCD/LICENSE)의 코드·모델에는 각 원저작권 및 별도 라이선스가 적용됩니다. 모델 가중치와 실행 결과는 커밋하지 않습니다.
