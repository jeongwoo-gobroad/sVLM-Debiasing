# (c) 2026 Jeongwoo Kim from KNU CSE, All rights reserved
# sVLM Inference-time Debiasing Research
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
from test_vcd import definitions

ROOT=Path(__file__).resolve().parents[1]
def classes(path, tokenizer):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    nodes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in ['TrieNode','TokenTrie','TrieLogitsProcessor']]
    ns={'torch':torch,'LogitsProcessor':object,'tokenizer':tokenizer}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
    return ns

class TrieVCDTests(unittest.TestCase):
    def test_matches_original_trie_and_preserves_nonempty_vcd_candidates(self):
        tokenizer=SimpleNamespace(encode=lambda *a,**kw:[3])
        old=classes(ROOT/'seomun_test/qwen_n.py',tokenizer)
        new=classes(ROOT/'seomun_test/vcd_trie.py',tokenizer)
        processors=[]
        for ns,is_old in [(old,True),(new,False)]:
            normal,space=ns['TokenTrie'](),ns['TokenTrie']()
            normal.insert([1,2],'tag'); space.insert([4,2],'tag')
            args=[1,0,set(),normal,space]
            args+= [None,None,False] if is_old else [tokenizer]
            processors.append(ns['TrieLogitsProcessor'](*args))
        model=lambda **kw:SimpleNamespace(logits=torch.zeros(1,1,6))
        vcd=definitions(model)['VCDLogitsProcessor']({'input_ids':torch.tensor([[5]])},1,1.,.1)
        prefix=[5]
        # Token 5 dominates the unconstrained logits but is forbidden by the trie.
        scores=torch.tensor([[-4.,0.,1.,2.,3.,100.]])
        for token in [1,2,3,4,2]:
            ids=torch.tensor([prefix])
            expected=processors[0](ids,scores.clone())
            actual=processors[1](ids,scores.clone())
            torch.testing.assert_close(actual,expected)
            combined=vcd(ids,actual)
            self.assertTrue(torch.isfinite(combined).any())
            self.assertFalse(torch.isnan(combined).any())
            self.assertTrue(torch.isneginf(combined[torch.isneginf(actual)]).all())
            prefix.append(token)

if __name__=='__main__': unittest.main()
