"""Regression tests of demonstrated defects and the new operational contract."""
import copy
import json
import unittest
from pathlib import Path
from rescore import (ROOT, Instance, parse, score, answer_ok, ANSWER_ALIASES,
                     canonical_plan, priority_slot, dataset_audit, expected_keys,resolve_unique)

STANDARD=[json.loads(s) for s in (ROOT/'inputs/v0_DE_test.jsonl').read_text().splitlines()]
HARD=[json.loads(s) for s in (ROOT/'inputs/hard.jsonl').read_text().splitlines()]
INSTANCES=[Instance.from_dict(r) for r in STANDARD+HARD]


class ParserTests(unittest.TestCase):
    def test_strings_braces_and_escapes(self):
        plan={'answer':'A } B { C; he said "yes".'}
        p=parse('An explanation.\n'+json.dumps(plan))
        self.assertEqual(p['plan'],plan); self.assertTrue(p['raw_ok'])

    def test_typographic_quotes_within_valid_string_are_not_modified(self):
        plan={'answer':'He said “yes” and } then left.'}
        p=parse(json.dumps(plan,ensure_ascii=False)+'\nTrailing prose.')
        self.assertEqual(p['plan'],plan); self.assertTrue(p['raw_ok'])

    def test_normalization_is_logged(self):
        p=parse('Explicație.\n{„answer”: „București”}')
        self.assertEqual(p['stage'],'normalized'); self.assertFalse(p['raw_ok'])
        self.assertTrue(p['normalized_ok'])

    def test_actual_repair_is_logged(self):
        p=parse('{"answer":"București",}')
        self.assertEqual(p['stage'],'repaired'); self.assertFalse(p['normalized_ok'])

    def test_last_object_policy_fenced_and_naked(self):
        for output in ['```json\n{"answer":"first"}\n```\n{"answer":"last"}',
                       '{"answer":"first"}\n```json\n{"answer":"last"}\n```']:
            self.assertEqual(parse(output)['plan'],{'answer':'last'})

    def test_no_fallback_to_earlier_correct_answer(self):
        self.assertNotEqual(parse('{"answer":"correct"}\n{"answer":"wrong"')['plan'],{'answer':'correct'})

    def test_duplicate_keys_not_raw_validity(self):
        self.assertFalse(parse('{"answer":"a","answer":"b"}')['raw_ok'])

    def test_nonfinite_numbers_not_raw_validity(self):
        self.assertFalse(parse('{"answer":NaN}')['raw_ok'])

    def test_missing_output_is_not_a_found_object(self):
        self.assertFalse(parse('There is no object.')['recovered'])
        r=score(INSTANCES[0],'There is no object.')
        self.assertEqual(r['T'],0); self.assertFalse(r['schema']); self.assertFalse(r['joint'])


class CheckerTests(unittest.TestCase):
    def test_expected_fact_values_pass_other_context_values_fail(self):
        for i in INSTANCES:
            if i.world.world_type!='fact': continue
            expected=i.world.payload['question']['expected_answer']
            self.assertTrue(answer_ok(i.world,{'answer':expected}),i.instance_id)
            for other in i.world.payload['facts'].values():
                if other!=expected:self.assertFalse(answer_ok(i.world,{'answer':other}),(i.instance_id,other))
            self.assertFalse(answer_ok(i.world,{'answer':''}))
            self.assertFalse(answer_ok(i.world,{'answer':expected+' sau ceva complet diferit'}))

    def test_arbitrary_substrings_rejected_but_valid_numeric_aliases_allowed(self):
        year=next(i for i in INSTANCES if i.world.world_type=='fact' and i.world.payload['question']['expected_answer']=='2007')
        self.assertFalse(answer_ok(year.world,{'answer':'2'}))
        sea=next(i for i in INSTANCES if i.world.world_type=='fact' and i.world.payload['question']['expected_answer']=='Marea Neagră')
        self.assertFalse(answer_ok(sea.world,{'answer':'M'}))
        self.assertFalse(answer_ok(sea.world,{'answer':'București'}))
        self.assertTrue(answer_ok(sea.world,{'answer':'Black Sea'}))

    def test_scheduling_entity_alias_invariance(self):
        for i in INSTANCES:
            if i.world.world_type!='schedule': continue
            keys=sorted(expected_keys(i.world))
            entities=list(i.world.canonical_entities.values())[:len(keys)]
            plans=[]
            for form in ['id','name','alias']:
                p={k:None for k in keys}
                for k,e in zip(keys,entities):
                    unambiguous=[a for a in e.aliases if resolve_unique(i.world,a)==e.id]
                    p[k]=e.id if form=='id' else e.name if form=='name' else unambiguous[-1] if unambiguous else e.name
                plans.append(p)
            scores=[score(i,json.dumps(p,ensure_ascii=False)) for p in plans]
            self.assertEqual([x['T'] for x in scores],[scores[0]['T']]*3,i.instance_id)

    def test_ambiguous_entity_alias_is_not_assigned_to_first_entity(self):
        i=next(i for i in INSTANCES if i.instance_id=='schedule_hard_010118')
        self.assertNotIn(resolve_unique(i.world,'client meeting'),i.world.canonical_entities)
        self.assertEqual(resolve_unique(i.world,'client meeting (2)'),'M2')

    def test_missing_high_priority_appointment_rejected(self):
        i=next(i for i in INSTANCES if any(c.check_fn=='check_keep_high_priority' for c in i.world.constraints))
        s=score(i,json.dumps({k:None for k in expected_keys(i.world)}))
        c=next(c for c in s['constraints'] if c['check_fn']=='check_keep_high_priority')
        self.assertFalse(c['satisfied'])

    def test_priority_slot_checker(self):
        i=next(i for i in INSTANCES if any(c.check_fn=='check_priority_slot_restriction' for c in i.world.constraints))
        c=next(c for c in i.world.constraints if c.check_fn=='check_priority_slot_restriction')
        e=next(e for e in i.world.canonical_entities.values() if e.attributes['priority']=='high')
        for key in ['Luni_dimineață','Monday_morning','Montag_Vormittag']:
            p,_=canonical_plan(i.world,{key:e.id});self.assertTrue(priority_slot(i.world,p,c.params))
        p,_=canonical_plan(i.world,{'Monday_afternoon':e.id});self.assertFalse(priority_slot(i.world,p,c.params))

    def test_key_alias_collision_not_silently_accepted(self):
        i=next(i for i in INSTANCES if i.world.world_type=='schedule')
        self.assertFalse(score(i,'{"Monday_morning":null,"Luni_dimineață":null}')['schema'])

    def test_unknown_checker_aborts(self):
        i=copy.deepcopy(INSTANCES[0]);i.world.constraints[0].check_fn='unregistered_checker'
        with self.assertRaises(ValueError):score(i,'{}')

    def test_known_infeasibility_and_duplicate_filter(self):
        a=dataset_audit(STANDARD)
        self.assertEqual(sum(x['duplicate_of'] is not None for x in a),23)
        bad=[x for x in a if 'proven_infeasible_type_constraints' in x['exclusion_reasons']]
        self.assertEqual(len(bad),5);self.assertTrue(all(not x['eligible'] for x in bad))

    def test_arrays_and_wrong_types_do_not_earn_task_credit(self):
        i=INSTANCES[0]
        for output in ['[]','{"day1":1}','{"day1":[1]}']:
            r=score(i,output);self.assertEqual(r['T'],0);self.assertFalse(r['schema'])


if __name__=='__main__':unittest.main(verbosity=2)
