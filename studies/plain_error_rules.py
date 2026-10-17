# (c) 2026 Jeongwoo Kim from KNU CSE, All rights reserved
# sVLM Inference-time Debiasing Research
"""Two independent errors; right-censor an unfinished final item at the token cap."""
import re

PROSE = re.compile(
    r'(?:입니다|습니다|합니다|됩니다|있어요|보여요|볼\s*수\s*있|확인할\s*수|'
    r'나열하|태그는|태그들|사진에서|이미지에서|사진에는|이미지에는|사진\s*속|이미지\s*속|'
    r'다음과\s*같|핵심\s*객체|정답은|알\s*수\s*없|'
    r'\b(?:this\s+(?:image|photo)|the\s+image|there\s+(?:is|are)|shows?|following\s+tags)\b)', re.I)
SENTENCE_END = re.compile(r'(?:이다|있다|한다|된다|보인다|같다|있으며|있고)(?=[.!?\s]|$)')
LEADING_LIST = re.compile(r'^\s*(?:[-*•]+\s*|\d+[.)]\s*)')


def censor_tail(row, candidates):
    text = row['raw_text']
    if row['ended_with_eos']:
        if row['hit_token_limit']: raise ValueError('Conflicting stop flags')
        return text, '', 'eos'
    if row['hit_token_limit'] is not True or row['new_tokens'] != 32:
        raise ValueError('Unknown stopping reason; do not infer truncation from the text')
    tail = re.split(r'[,\n;]', text)[-1]
    clean = LEADING_LIST.sub('', tail.strip()).strip('`*\"\'[] ')
    if not clean or clean in candidates or re.search(r'[.!?]\s*$', clean):
        return text, '', 'cap_with_complete_last_item'
    # EOS나 구분자가 없는 비후보 말미의 완성 여부는 확정할 수 없다.
    reason = 'cap_candidate_prefix' if any(c.startswith(clean) for c in candidates) else (
        'cap_incomplete_character' if '\ufffd' in clean else 'cap_undelimited_tail_uncertain')
    return text[:-len(tail)], tail, reason


def classify(text, candidates):
    prose, tags, ambiguous = [], [], []
    for original in re.split(r'[,\n;]', text):
        segment = LEADING_LIST.sub('', original.strip()).strip('`*\"\'[] ')
        if not segment: continue
        if PROSE.search(segment) or SENTENCE_END.search(segment):
            prose.append(segment)
            if ':' in segment or '：' in segment:
                tail = re.split(r'[:：]', segment, maxsplit=1)[1].strip()
                if tail and not (PROSE.search(tail) or SENTENCE_END.search(tail)):
                    tags.extend(tail.strip(' .').split())
            else:
                ambiguous.append(segment)
            continue
        clean = segment.rstrip('.!?').strip()
        words = clean.split()
        if len(words) > 1 and not all(word in candidates for word in words):
            ambiguous.append(segment)
            tags.append(clean)
        else:
            tags.extend(words)
    outside = sorted(set(tags) - candidates)
    return {'prompt_noncompliance': bool(prose), 'out_of_candidate': bool(outside),
            'generated_tags': sorted(set(tags)), 'outside_tags': outside,
            'prose_evidence': prose, 'ambiguous_segments': ambiguous,
            'needs_review': bool(ambiguous), 'empty_evaluated_response': not text.strip()}
