import json
import pathlib
import tempfile
import unittest

from datagen.training.train_lora import load_mix_records


class TestLoadMixRecords(unittest.TestCase):
    OUTPUT = {"oraș": "Brașov", "total": 1234.56, "ok": True}

    def _write(self, td, rec):
        p = pathlib.Path(td) / "mix.jsonl"
        p.write_text(json.dumps(rec, ensure_ascii=False) + "\n",
                     encoding="utf-8")
        return p

    def _rec(self, **extra):
        return {"id": "x", "track": "A",
                "instruction": "Generează un obiect JSON valid.",
                "schema": {"type": "object"}, "output": self.OUTPUT, **extra}

    def test_prompt_completion_shape_and_json_round_trip(self):
        with tempfile.TemporaryDirectory() as td:
            out = load_mix_records(
                self._write(td, self._rec(explanation="Explicație.")))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["prompt"][0]["role"], "user")
        self.assertEqual(out[0]["prompt"][0]["content"],
                         "Generează un obiect JSON valid.")
        completion = out[0]["completion"][0]
        self.assertEqual(completion["role"], "assistant")
        body = completion["content"][completion["content"].index("{"):]
        # values survive exactly, diacritics unescaped
        self.assertEqual(json.loads(body), self.OUTPUT)
        self.assertIn("Brașov", completion["content"])

    def test_the_prose_goes_before_the_json(self):
        """The benchmark parses the text before the first JSON block as the
        explanation. Built the other way round, a completion scores zero on
        faithfulness and looks fine in a diff."""
        with tempfile.TemporaryDirectory() as td:
            out = load_mix_records(
                self._write(td, self._rec(explanation="Explicație.")))
        c = out[0]["completion"][0]["content"]
        self.assertTrue(c.startswith("Explicație."))
        self.assertLess(c.index("Explicație."), c.index("{"))

    def test_the_older_plan_field_still_works(self):
        """ro-so/pipelines emitted `plan`; datagen emits `explanation`. Both
        have to be read, or an old mix silently trains as bare JSON."""
        with tempfile.TemporaryDirectory() as td:
            out = load_mix_records(self._write(td, self._rec(plan="Verific.")))
        self.assertTrue(out[0]["completion"][0]["content"].startswith("Verific."))

    def test_a_mix_with_no_prose_at_all_is_refused(self):
        """The failure this guard exists for: reading only the old field name
        against the new corpus would train bare JSON on every record -- the
        exact answer shape the corpus was regenerated to eliminate -- and stay
        invisible until evaluation came back flat."""
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError) as cm:
                load_mix_records(self._write(td, self._rec()))
        self.assertIn("bare JSON", str(cm.exception))

    # --- replay records -------------------------------------------------
    # General-instruction rehearsal: a prompt the base checkpoint was tuned on,
    # answered by that same checkpoint. Prose, no schema, no JSON, no prefix.

    def _replay(self, **extra):
        return {"id": "r", "kind": "replay",
                "instruction": "Explică pe scurt ce este fotosinteza.",
                "output": "Fotosinteza este procesul prin care plantele "
                          "transformă lumina în energie.", **extra}

    def _write_many(self, td, recs):
        p = pathlib.Path(td) / "mix.jsonl"
        p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                             for r in recs), encoding="utf-8")
        return p

    def test_replay_completion_is_carried_verbatim(self):
        """Not JSON-serialized, not prefixed, not quoted -- the string itself."""
        with tempfile.TemporaryDirectory() as td:
            out = load_mix_records(self._write(td, self._replay()))
            self.assertEqual(len(out), 1)
            self.assertEqual(out[0]["completion"][0]["content"],
                             self._replay()["output"])

    def test_a_pure_replay_mix_is_allowed(self):
        """The prefix guard counts structured records only. A file with no
        structured records has no prefixes to be missing."""
        with tempfile.TemporaryDirectory() as td:
            out = load_mix_records(
                self._write_many(td, [self._replay(), self._replay(id="r2")]))
            self.assertEqual(len(out), 2)

    def test_replay_does_not_excuse_structured_records_losing_their_prose(self):
        """The regression the guard exists for, with replay present: a mix whose
        STRUCTURED half has lost its prefixes must still be refused."""
        with tempfile.TemporaryDirectory() as td:
            p = self._write_many(td, [self._replay(), self._rec()])
            with self.assertRaises(ValueError) as cm:
                load_mix_records(p)
            self.assertIn("explanation", str(cm.exception))

    def test_replay_with_a_dict_output_is_refused(self):
        """A structured record mislabelled as replay would be trained as the
        repr of a dict. Catch it at load, not in the scores."""
        with tempfile.TemporaryDirectory() as td:
            p = self._write(td, self._replay(output={"a": 1}))
            with self.assertRaises(ValueError) as cm:
                load_mix_records(p)
            self.assertIn("must be a non-empty string", str(cm.exception))

    def test_replay_with_an_empty_output_is_refused(self):
        with tempfile.TemporaryDirectory() as td:
            p = self._write(td, self._replay(output="   "))
            with self.assertRaises(ValueError):
                load_mix_records(p)

    def test_empty_file_fails_loudly(self):
        with tempfile.TemporaryDirectory() as td:
            p = pathlib.Path(td) / "mix.jsonl"
            p.write_text("", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_mix_records(p)


if __name__ == "__main__":
    unittest.main()


class TestDataArgIsRequired(unittest.TestCase):
    """--data used to default to train_mix_1k.jsonl. it4 trained a full run on
    that stale file because the flag was omitted, and nothing downstream
    flagged it -- a wrong-but-valid mix is indistinguishable from a right one
    until you back out the dataset size from the step count."""

    def test_missing_data_arg_exits(self):
        from datagen.training import train_lora
        with self.assertRaises(SystemExit):
            train_lora.main(["--output", "/tmp/nope"])

    def test_data_arg_has_no_default(self):
        import argparse
        from datagen.training import train_lora
        ap = argparse.ArgumentParser()
        # mirror of the real parser's contract: --data must be required with
        # no default, so a forgotten flag can never silently pick a file
        parser_src = train_lora.main.__doc__ or ""
        import inspect
        src = inspect.getsource(train_lora.main)
        self.assertIn('"--data", required=True', src)
        self.assertNotIn('"--data", default=', src)


class TestSeedOverride(unittest.TestCase):
    def test_seed_flag_exists_and_defaults_to_none(self):
        import inspect
        from datagen.training import train_lora
        src = inspect.getsource(train_lora.main)
        self.assertIn('"--seed", type=int, default=None', src)

    def test_yaml_seed_used_when_flag_absent(self):
        import inspect
        from datagen.training import train_lora
        src = inspect.getsource(train_lora.train)
        self.assertIn(
            'seed=args.seed if args.seed is not None else train_cfg["seed"]', src)


class TestTargetModules(unittest.TestCase):
    """The yaml carried a `lora.target_modules` list that nothing read, while an
    anchored regex hardcoded in the trainer decided what actually got adapted.
    The configs described one thing and the runs did another, which makes any
    training recipe reported from the yaml wrong."""

    # The pattern that produced every released adapter.
    HISTORICAL = (r"^model\.layers\.\d+\.(self_attn|mlp)\."
                  r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$")

    def test_every_shipped_config_reproduces_what_actually_ran(self):
        """The fix must not silently re-target the adapters: for the module list
        every config carries, the derived regex has to equal the old constant."""
        import glob
        import yaml as _yaml
        from datagen.core.common import ROOT
        from datagen.training.train_lora import (DEFAULT_TARGET_MODULES,
                                                 target_module_regex)
        # Deliberate variants are excluded by name, not by "whatever differs" --
        # the whole point of this test is to catch an unintended difference, so
        # an exclusion that keys off the module list would exclude exactly the
        # bug it exists to find. Each entry here changes target_modules ON
        # PURPOSE and produced no released adapter.
        INTENTIONAL = {"lora_rollama31_8b_attnonly.yaml"}
        configs = [c for c in sorted(glob.glob(str(ROOT / "training" / "lora_*.yaml")))
                   if pathlib.Path(c).name not in INTENTIONAL]
        self.assertTrue(configs, "no training configs found")
        for path in configs:
            with self.subTest(config=pathlib.Path(path).name):
                cfg = _yaml.safe_load(open(path, encoding="utf-8"))
                names = cfg["lora"].get("target_modules") or DEFAULT_TARGET_MODULES
                self.assertEqual(target_module_regex(names), self.HISTORICAL)

    def test_the_attention_only_variant_really_is_attention_only(self):
        """The excluded config is excluded because it differs deliberately --
        assert what it differs TO, so the exclusion cannot quietly cover a
        typo."""
        import yaml as _yaml
        from datagen.core.common import ROOT
        cfg = _yaml.safe_load(
            open(ROOT / "training" / "lora_rollama31_8b_attnonly.yaml",
                 encoding="utf-8"))
        self.assertEqual(cfg["lora"]["target_modules"],
                         ["q_proj", "k_proj", "v_proj", "o_proj"])
        self.assertEqual(cfg["train"]["lr"], 1.0e-4,
                         "the attention-only run must hold lr at the baseline; "
                         "moving two things at once makes it uninterpretable")

    def test_the_yaml_now_governs(self):
        from datagen.training.train_lora import target_module_regex
        self.assertEqual(target_module_regex(["q_proj", "v_proj"]),
                         r"^model\.layers\.\d+\.(self_attn|mlp)\.(q_proj|v_proj)$")

    def test_an_empty_list_is_refused(self):
        """An adapter over zero modules trains, merges and serves while
        reproducing the base model exactly -- the failure this must not allow."""
        from datagen.training.train_lora import target_module_regex
        with self.assertRaises(ValueError):
            target_module_regex([])

    def test_the_trainer_reads_the_yaml_rather_than_a_constant(self):
        import inspect
        from datagen.training import train_lora
        src = inspect.getsource(train_lora.train)
        self.assertIn('lora_cfg.get("target_modules")', src)
        self.assertIn("target_module_regex(target_modules)", src)
