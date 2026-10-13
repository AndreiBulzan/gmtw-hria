"""Checks for cohort integrity and accidental data loss in the new file runner."""
import json
from pathlib import Path
import tempfile
import unittest

from rescore_saved_outputs import load_cohort, load_outputs, verify_sources


class SavedOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.standard, cls.retained = load_cohort('standard')

    def load(self, rows, **kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'outputs.jsonl'
            path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
            return load_outputs(path, self.standard, self.retained, **kwargs)

    def rows(self):
        return [{'instance_id': ident, 'output': ''} for ident in sorted(self.retained)]

    def test_sources_are_exact_copies(self):
        verify_sources()

    def test_both_frozen_cohorts(self):
        self.assertEqual((len(self.standard), len(self.retained)), (500, 438))
        hard, retained = load_cohort('adversarial')
        self.assertEqual((len(hard), len(retained)), (300, 240))

    def test_blank_responses_are_kept(self):
        outputs, _ = self.load(self.rows())
        self.assertEqual(set(outputs), self.retained)
        self.assertTrue(all(r['output'] == '' for r in outputs.values()))

    def test_all_original_ids_are_allowed(self):
        outputs, _ = self.load([{'instance_id': ident, 'output': ''} for ident in self.standard])
        self.assertEqual(len(outputs), 500)

    def test_missing_retained_output_is_not_silently_dropped(self):
        with self.assertRaisesRegex(ValueError, 'Missing 1 retained'):
            self.load(self.rows()[1:])

    def test_duplicate_seed_rows_are_rejected(self):
        rows = self.rows()
        with self.assertRaisesRegex(ValueError, 'Duplicate task ID'):
            self.load(rows + rows[:1])

    def test_wrong_split_is_rejected(self):
        rows = self.rows()
        rows[0]['instance_id'] = 'not_a_standard_task'
        with self.assertRaisesRegex(ValueError, 'Unknown task ID'):
            self.load(rows)

    def test_null_is_not_silently_converted_to_blank(self):
        rows = self.rows()
        rows[0]['output'] = None
        with self.assertRaisesRegex(ValueError, 'raw response string'):
            self.load(rows)

    def test_mixed_models_are_rejected_even_with_unique_ids(self):
        rows = self.rows()
        rows[0]['model'] = 'baseline'
        rows[1]['model'] = 'repaired'
        with self.assertRaisesRegex(ValueError, 'Mixed run metadata'):
            self.load(rows)

    def test_custom_keys_preserve_text_and_metadata(self):
        rows = [{'task': ident, 'raw_response': 'Explicație înainte de {}', 'seed': 2}
                for ident in self.retained]
        outputs, metadata = self.load(rows, id_field='task', output_field='raw_response')
        self.assertEqual(metadata, {'seed': 2})
        self.assertTrue(all(r['output'] == 'Explicație înainte de {}' for r in outputs.values()))


if __name__ == '__main__':
    unittest.main()
