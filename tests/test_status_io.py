# (c) 2026 Jeongwoo Kim from KNU CSE, All rights reserved
# sVLM Inference-time Debiasing Research
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'studies'))
from status_io import write_status


class StatusFileTests(unittest.TestCase):
    def test_transient_reader_lock_retries_without_partial_json(self):
        replace=Path.replace
        calls=[]
        def transient(source,target):
            calls.append(source)
            if len(calls)<3:raise PermissionError(5,'file busy')
            return replace(source,target)
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/'status.json'
            target.write_text('{"completed":1}',encoding='utf-8')
            with patch.object(Path,'replace',transient),patch('status_io.time.sleep'):
                self.assertTrue(write_status(target,{'completed':2}))
            self.assertEqual(json.loads(target.read_text()),{'completed':2})
            self.assertEqual(len(calls),3)

    def test_busy_progress_preserves_previous_status_but_final_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/'status.json'
            target.write_text('{"completed":1}',encoding='utf-8')
            with patch.object(Path,'replace',side_effect=PermissionError(5,'file busy')),patch('status_io.time.sleep'):
                self.assertFalse(write_status(target,{'completed':2},required=False,attempts=2))
                self.assertEqual(json.loads(target.read_text()),{'completed':1})
                with self.assertRaises(PermissionError):write_status(target,{'stage':'complete'},attempts=2)


if __name__=='__main__':unittest.main()
