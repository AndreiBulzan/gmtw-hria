"""Tests that need no teacher server.

The one thing worth guarding above all: the completion must be explanation
first, JSON last. That ordering is the reason this corpus is being regenerated,
and the benchmark's parser treats the text before the first JSON block as the
explanation — build a completion the other way round and it scores zero on
faithfulness while looking fine in a diff.

Run from the repo root:  python -m unittest datagen.tests.test_datagen
"""
import json
import pathlib
import random
import tempfile
import unittest

from datagen.core.common import (ROOT, completion_text, load_templates,
                                 make_record, read_jsonl, resolve_template)
from datagen.core.constraints import check_constraints
from datagen.core.explain import (explain_extraction, explain_schema_object,
                                  handoff, join_list)
from datagen import track_b, track_c


class TestCompletionShape(unittest.TestCase):
    def _rec(self, **over):
        base = dict(rid="x", track="A", instruction="i", schema={},
                    output={"a": 1}, explanation="Explicație.", meta={})
        return make_record(**{**base, **over})

    def test_explanation_comes_first_and_json_last(self):
        text = completion_text(self._rec())
        self.assertTrue(text.startswith("Explicație."))
        self.assertTrue(text.rstrip().endswith("}"))
        self.assertLess(text.index("Explicație."), text.index("{"))

    def test_an_empty_explanation_is_refused(self):
        """A record with no prose is the old answer shape sneaking back in."""
        for bad in ("", "   ", "\n"):
            with self.subTest(value=repr(bad)):
                with self.assertRaises(ValueError):
                    self._rec(explanation=bad)

    def test_the_json_half_round_trips(self):
        text = completion_text(self._rec(output={"a": [1, 2], "b": "ș"}))
        block = text[text.index("{"):]
        self.assertEqual(json.loads(block), {"a": [1, 2], "b": "ș"})

    def test_diacritics_survive_the_record_boundary(self):
        rec = self._rec(explanation="Am ales cărți și țevi.")
        self.assertIn("ș", completion_text(rec))
        self.assertIn("ț", completion_text(rec))


class TestRomanianLists(unittest.TestCase):
    def test_reads_as_prose_not_a_column_dump(self):
        self.assertEqual(join_list(["a"]), "a")
        self.assertEqual(join_list(["a", "b"]), "a și b")
        self.assertEqual(join_list(["a", "b", "c"]), "a, b și c")
        self.assertEqual(join_list(["a", "b"], "en"), "a and b")

    def test_empty_list_does_not_crash(self):
        self.assertEqual(join_list([]), "")


class TestTrackAExplanation(unittest.TestCase):
    def setUp(self):
        self.t = load_templates(ROOT / "templates/explain_a_ro.txt")
        self.rng = random.Random(0)

    def test_it_names_the_fields_that_are_there(self):
        obj = {"invoice_number": "A-1", "total_amount": 10}
        text = explain_schema_object(obj, {}, self.t, self.rng)
        for k in obj:
            self.assertIn(k, text)

    def test_it_uses_the_schema_title_when_there_is_one(self):
        text = explain_schema_object({"a": 1}, {"title": "Purchase Order"},
                                     self.t, self.rng)
        self.assertIn("Purchase Order", text)

    def test_it_invents_no_subject_when_the_schema_is_silent(self):
        """A guessed subject is a false statement in the training target."""
        text = explain_schema_object({"a": 1}, {}, self.t, self.rng)
        self.assertIn("conform schemei", text)

    def test_a_non_object_answer_does_not_claim_to_have_fields(self):
        for answer in ([1, 2, 3], "text", 42, {}):
            with self.subTest(answer=answer):
                text = explain_schema_object(answer, {}, self.t, self.rng)
                self.assertTrue(text.strip())
                self.assertNotIn("câmpuri de nivel superior", text)

    def test_every_phrasing_renders(self):
        """A template with a placeholder we never fill would raise only on the
        run that happens to draw it."""
        for i, tpl in enumerate(self.t):
            with self.subTest(phrasing=i):
                explain_schema_object({"a": 1, "b": 2}, {"title": "T"}, [tpl],
                                      random.Random(0))


class TestTrackBExplanation(unittest.TestCase):
    def setUp(self):
        self.t = load_templates(ROOT / "templates/explain_b_ro.txt")

    def test_it_names_labels_and_not_values(self):
        """Listing the values back would make the explanation a second copy of
        the JSON — padding, not explaining, and it inflates faithfulness."""
        gold = {"invoice_number": "SECRET-9999", "currency": "RON"}
        labels = {"invoice_number": "Număr factură", "currency": "Moneda"}
        text = explain_extraction(gold, labels, "o factură fiscală", self.t,
                                  random.Random(0))
        self.assertIn("Număr factură", text)
        self.assertNotIn("SECRET-9999", text)

    def test_every_phrasing_renders_and_names_the_document(self):
        gold, labels = {"a": 1}, {"a": "Câmp"}
        for i, tpl in enumerate(self.t):
            with self.subTest(phrasing=i):
                text = explain_extraction(gold, labels, "o factură fiscală",
                                          [tpl], random.Random(0))
                self.assertIn("factură fiscală", text)

    def test_no_phrasing_puts_a_bare_preposition_before_the_document(self):
        """Romanian contracts the article: 'dintr-o', never 'din o'."""
        for i, tpl in enumerate(self.t):
            text = explain_extraction({"a": 1}, {"a": "C"}, "o factură", [tpl],
                                      random.Random(0))
            with self.subTest(phrasing=i):
                for bad in ("din o ", "cu o factură de mai sus", "în o "):
                    self.assertNotIn(bad, text)

    def test_every_domain_declares_a_document_label(self):
        for name, dom in track_b.DOMAINS.items():
            with self.subTest(domain=name):
                self.assertTrue(dom.get("doc_label", "").strip())


class TestTheExplanationHandsOffToTheJson(unittest.TestCase):
    """The measured failure this guards: RoMistral-SFT seed 1 answered 17% of the
    standard split in prose and never emitted the JSON — `No JSON block found`,
    no `{` anywhere, far short of any length limit. The explanations ended on a
    bullet and nothing marked where the prose stopped and the answer began, so
    "end of explanation" was a plausible "end of turn". Seed 1 was the worst run
    in six of seven families on the identical mix."""

    def _ends_with_handoff(self, text, lang="ro"):
        from datagen.core.explain import _HANDOFF
        return text.rstrip().endswith(_HANDOFF[lang])

    def test_track_a_ends_on_a_handoff(self):
        t = load_templates(ROOT / "templates/explain_a_ro.txt")
        text = explain_schema_object({"a": 1, "b": 2}, {"title": "T"}, t,
                                     random.Random(0))
        self.assertTrue(self._ends_with_handoff(text), repr(text[-80:]))

    def test_track_a_hands_off_even_when_the_answer_is_not_an_object(self):
        """The short fallback branch is the easiest one to forget."""
        t = load_templates(ROOT / "templates/explain_a_ro.txt")
        for answer in ([1, 2, 3], "text", 42, {}):
            with self.subTest(answer=answer):
                text = explain_schema_object(answer, {}, t, random.Random(0))
                self.assertTrue(self._ends_with_handoff(text), repr(text[-80:]))

    def test_track_b_ends_on_a_handoff(self):
        t = load_templates(ROOT / "templates/explain_b_ro.txt")
        text = explain_extraction({"a": 1}, {"a": "Câmp"}, "o factură fiscală",
                                  t, random.Random(0))
        self.assertTrue(self._ends_with_handoff(text), repr(text[-80:]))

    def test_english_track_a_hands_off_in_english(self):
        t = load_templates(ROOT / "templates/explain_a_en.txt")
        text = explain_schema_object({"a": 1}, {"title": "T"}, t,
                                     random.Random(0), lang="en")
        self.assertTrue(self._ends_with_handoff(text, "en"), repr(text[-80:]))

    def test_the_handoff_is_not_always_the_same_sentence(self):
        """One phrasing everywhere makes the trained prose stereotyped."""
        seen = {handoff(random.Random(i)) for i in range(40)}
        self.assertGreater(len(seen), 1)


class TestTrackAActuallyDescribesTheFields(unittest.TestCase):
    """Track A was the shortest track at a median of 26 words and 55% of the
    mix, and trained models converged on ~46-word answers where the benchmark
    gives full length credit at 100."""

    def setUp(self):
        self.t = load_templates(ROOT / "templates/explain_a_ro.txt")
        self.schema = {
            "title": "Invoice", "required": ["invoice_number"],
            "properties": {
                "invoice_number": {"type": "string"},
                "status": {"type": "string", "enum": ["paid", "unpaid"]},
                "issued": {"type": "string", "format": "date"},
                "tags": {"type": "array", "items": {"type": "string"}},
            }}
        self.obj = {"invoice_number": "A-1", "status": "paid",
                    "issued": "2026-01-01", "tags": ["x"]}

    def test_it_states_what_the_schema_says_about_each_field(self):
        text = explain_schema_object(self.obj, self.schema, self.t,
                                     random.Random(0))
        self.assertIn("șir de caractere", text)
        self.assertIn("format date", text)
        self.assertIn('"paid"', text)          # enum values verbatim, untranslated
        self.assertIn("obligatoriu", text)     # required field marked

    def test_a_list_agrees_in_number(self):
        """'listă de șir de caractere' is wrong Romanian."""
        text = explain_schema_object(self.obj, self.schema, self.t,
                                     random.Random(0))
        self.assertIn("listă de șiruri de caractere", text)
        self.assertNotIn("listă de șir de caractere", text)

    def test_it_is_substantially_longer_than_the_bare_list(self):
        text = explain_schema_object(self.obj, self.schema, self.t,
                                     random.Random(0))
        self.assertGreater(len(text.split()), 45)

    def test_it_invents_nothing_when_the_schema_describes_nothing(self):
        """A guessed type is a false statement in the training target, so a
        schema with no `properties` gets no per-field block at all."""
        text = explain_schema_object({"a": 1}, {}, self.t, random.Random(0))
        self.assertNotIn("- `a`", text)

    def test_a_wide_object_does_not_produce_a_wall_of_text(self):
        wide = {f"f{i}": i for i in range(30)}
        schema = {"properties": {k: {"type": "integer"} for k in wide}}
        text = explain_schema_object(wide, schema, self.t, random.Random(0))
        self.assertLessEqual(text.count("\n- "), 9)   # 8 described + 1 summary
        self.assertIn("restul de 22", text)


class TestTrackCClosesItsChecklist(unittest.TestCase):
    def _recs(self, n=8, seed=0):
        with tempfile.TemporaryDirectory() as td:
            p = pathlib.Path(td) / "c.jsonl"
            track_c.main(["--n", str(n), "--seed", str(seed), "--out", str(p)])
            return list(read_jsonl(p))

    def test_no_explanation_ends_on_a_bullet(self):
        """The exact shape that let the model treat the checklist as the whole
        answer: list stops, blank line, JSON, nothing marking the boundary."""
        for rec in self._recs():
            last = [l for l in rec["explanation"].splitlines() if l.strip()][-1]
            self.assertFalse(last.lstrip().startswith("-"), rec["id"])

    def test_every_explanation_ends_by_naming_the_json(self):
        for rec in self._recs():
            self.assertIn("JSON", rec["explanation"].splitlines()[-1], rec["id"])

    def test_the_closing_line_still_never_claims_a_failure(self):
        for rec in self._recs():
            self.assertNotIn("Nu toate condițiile", rec["explanation"], rec["id"])


class TestTemplatePaths(unittest.TestCase):
    """Every usage line in the docs writes `templates/x.txt`, but callers run
    from the repo root. The bare form has to resolve, or the documented command
    dies with a missing-file error that looks like the file is absent."""

    def test_a_package_relative_path_resolves_from_anywhere(self):
        self.assertTrue(resolve_template("templates/instructions_en.txt").exists())

    def test_an_absolute_path_is_used_as_given(self):
        direct = ROOT / "templates/instructions_ro.txt"
        self.assertEqual(resolve_template(direct), direct)

    def test_a_genuinely_missing_file_still_raises(self):
        with self.assertRaises(FileNotFoundError):
            resolve_template("templates/no_such_file.txt")

    def test_every_template_the_scripts_name_actually_loads(self):
        for name in ("instructions_ro.txt", "instructions_en.txt",
                     "extraction_ro.txt", "select_ro.txt",
                     "explain_a_ro.txt", "explain_a_en.txt",
                     "explain_b_ro.txt"):
            with self.subTest(template=name):
                self.assertTrue(load_templates(f"templates/{name}"))


class TestTrackCRecords(unittest.TestCase):
    def _generate(self, n=8, seed=0, extra=None):
        with tempfile.TemporaryDirectory() as td:
            p = pathlib.Path(td) / "c.jsonl"
            track_c.main(["--n", str(n), "--seed", str(seed), "--out", str(p)]
                         + (extra or []))
            return list(read_jsonl(p)), p.read_bytes()

    def test_every_record_carries_an_explanation(self):
        for rec in self._generate()[0]:
            self.assertTrue(rec["explanation"].strip(), rec["id"])

    def test_the_answer_satisfies_its_own_rules(self):
        for rec in self._generate()[0]:
            self.assertEqual(
                check_constraints(rec["meta"]["constraints"], rec["output"],
                                  rec["meta"]["catalogue"]), [], rec["id"])

    def test_the_working_out_never_claims_a_failure(self):
        """Gold always passes, so no verdict may read as a violation. The old
        renderer had the opposite bug — it asserted success unconditionally —
        and this guards the direction that is actually reachable here."""
        for rec in self._generate()[0]:
            for phrase in ("nu e respectată", "peste limita", "prea puține"):
                self.assertNotIn(phrase, rec["explanation"], rec["id"])

    def test_the_limit_forces_leaving_items_behind(self):
        """If taking everything ever fit, the task would teach nothing."""
        for rec in self._generate()[0]:
            cat = rec["meta"]["catalogue"]
            spec = next(s for s in rec["meta"]["constraints"]
                        if s["kind"] == "sum_at_most")
            everything = sum(e[spec["attr"]] for e in cat.values())
            self.assertGreater(everything, spec["max"], rec["id"])

    def test_same_seed_same_bytes(self):
        self.assertEqual(self._generate(6, seed=3)[1],
                         self._generate(6, seed=3)[1])

    def test_no_show_work_flag_survived_the_move(self):
        """That flag belongs with the constraint experiments, not here."""
        with self.assertRaises(SystemExit):
            track_c.main(["--n", "1", "--out", "/tmp/nope.jsonl",
                          "--show-work", "budget"])


class TestTemplatesAskForTheRightShape(unittest.TestCase):
    """The prompts must ask for prose then JSON. The previous corpus asked for
    the opposite and that is the main reason it is being regenerated."""

    FORBIDDEN = ["doar cu JSON-ul", "fără explicații",
                 "Fără text în afara JSON-ului", "exclusiv cu obiectul JSON"]

    def test_no_instruction_template_forbids_an_explanation(self):
        for name in ("instructions_ro.txt", "extraction_ro.txt"):
            text = (ROOT / "templates" / name).read_text(encoding="utf-8")
            body = "\n".join(l for l in text.splitlines()
                             if not l.startswith("#"))
            for bad in self.FORBIDDEN:
                with self.subTest(template=name, phrase=bad):
                    self.assertNotIn(bad, body)

    def test_the_language_slices_have_the_same_number_of_phrasings(self):
        """The English slice is meant to vary the language and nothing else."""
        self.assertEqual(
            len(load_templates(ROOT / "templates/instructions_ro.txt")),
            len(load_templates(ROOT / "templates/instructions_en.txt")))
        self.assertEqual(
            len(load_templates(ROOT / "templates/explain_a_ro.txt")),
            len(load_templates(ROOT / "templates/explain_a_en.txt")))


if __name__ == "__main__":
    unittest.main()
