#!/usr/bin/env python3
"""Re-score fixed public generations. No model inference, training, or network.

The proposal audits specific defects; it does not certify every inherited
constraint or prove feasibility of every retained world. See ISSUE_STATUS.txt.
"""
from pathlib import Path
import collections
import copy
import hashlib
import importlib.metadata
import json
import random
import re
import sys
import unicodedata

import numpy as np
from json_repair import repair_json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'code/vendor'))
from rombench.gmtw_ro.worlds.base import Instance
from rombench.gmtw_ro.eval import constraints as inherited

SMART = str.maketrans({c: '"' for c in '\u201e\u201d\u201c\u00ab\u00bb\u201a\u2018\u2019'})
HIDDEN_CHECKS = {'check_max_duration_per_day', 'check_max_prep_time_per_day', 'check_quick_breakfast'}
MODELS = {
 'llama-3.1-8b-instruct':'Llama-3.1-8B-Inst', 'gemma-2-9b-it':'Gemma-2-9B-it',
 'gemma-7b-it':'Gemma-7B-it', 'rogemma-7b-instruct-dpo':'RoGemma-7B-DPO',
 'rogemma2-9b-instruct-dpo':'RoGemma2-9B-DPO', 'rollama2-7b-instruct':'RoLlama2-7B-Inst',
 'rollama3-8b-instruct-dpo':'RoLlama3-8B-DPO', 'rollama3.1-8b-instruct-dpo':'RoLlama3.1-8B-DPO',
 'romistral-7b-instruct-dpo':'RoMistral-7B-DPO'}
PAIRS = [('llama-3.1-8b-instruct','rollama3.1-8b-instruct-dpo'),
         ('gemma-2-9b-it','rogemma2-9b-instruct-dpo')]


def normalize(s):
    s = unicodedata.normalize('NFKD', str(s).casefold())
    s = ''.join(c for c in s if not unicodedata.combining(c))
    return ' '.join(re.findall(r'\w+', s))


# Explicit acceptable equivalents, defined from task specifications. Whole
# normalized strings must match. Arbitrary substrings are never accepted.
ANSWER_ALIASES = {
 '1 Decembrie':['1 December','December 1','1. Dezember','1 decembrie'],
 '24 Ianuarie':['24 January','January 24','24. Januar'],
 'București':['Bucharest','Bukarest'], 'Dunărea':['Danube','Donau'],
 'Mureșul':['Mures','Mureș','Mures River','Marosch'],
 'Marea Neagră':['Black Sea','the Black Sea','Schwarzes Meer','das Schwarze Meer'],
 'Marea Caspică':['Caspian Sea','the Caspian Sea','Kaspisches Meer','das Kaspische Meer'],
 'Leul (RON)':['leu','leul','RON','Romanian leu','rumänischer Leu'],
 '4 țări':['4','patru','patru țări','4 countries','four countries','vier Länder','vier'],
 '5 (Ucraina, Moldova, Bulgaria, Serbia, Ungaria)':['5','cinci','cinci țări','5 țări','five countries','5 countries','fünf Länder','fünf'],
 'aproximativ 2 milioane':['2 milioane','approximately 2 million','about 2 million','2 million','etwa 2 Millionen','ungefähr 2 Millionen','2 Millionen'],
 'aproximativ 19 milioane':['19 milioane','approximately 19 million','about 19 million','19 million','etwa 19 Millionen','ungefähr 19 Millionen','19 Millionen'],
 'aproximativ 300.000 km/s':['300.000 km/s','300000 km/s','approximately 300,000 km/s','about 300,000 km/s','etwa 300.000 km/s'],
 'Moldoveanu (2544m)':['Moldoveanu','Moldoveanu (2544 m)','Moldoveanu Peak','Vârful Moldoveanu'],
 'Negoiu (2535m)':['Negoiu','Negoiu (2535 m)','Negoiu Peak','Vârful Negoiu'],
 '100°C':['100 °C','100 degrees Celsius','100 grade Celsius','100 Grad Celsius'],
 '85°C':['85 °C','85 degrees Celsius','85 grade Celsius','85 Grad Celsius'],
}


def answer_ok(world, plan):
    expected = world.payload['question']['expected_answer']
    answer = plan.get('answer') if isinstance(plan, dict) else None
    if not isinstance(answer, str) or not answer.strip():
        return False
    allowed = [expected] + ANSWER_ALIASES.get(expected, [])
    return normalize(answer) in {normalize(a) for a in allowed}


def strict_load(s):
    def pairs(kvs):
        out = {}
        for k, v in kvs:
            if k in out:
                raise ValueError('Duplicate JSON key')
            out[k] = v
        return out
    def bad_constant(x):
        raise ValueError('Non-finite JSON number: '+x)
    return json.loads(s, object_pairs_hook=pairs, parse_constant=bad_constant)


def spans(text):
    """Top-level object spans, respecting quoted strings and escapes."""
    result = []; start = None; depth = 0; quote = None; escaped = False
    for i, c in enumerate(text):
        if start is None:
            if c == '{': start = i; depth = 1
            continue
        if quote:
            if escaped: escaped = False
            elif c == '\\': escaped = True
            elif c == quote: quote = None
            continue
        if c in ['"', "'"]: quote = c
        elif c == '{': depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                result.append((start, i+1)); start = None
    if start is not None:
        result.append((start, len(text)))
    return result


def parse(output):
    """Select the last outer object under a raw-or-quote-normalized scan.

    The same span is used for strict, normalized and repaired comparisons.
    An invalid final object is never replaced by an earlier successful one.
    """
    raw_spans = spans(output)
    valid_raw = []
    for a,b in raw_spans:
        try:
            if isinstance(strict_load(output[a:b]),dict): valid_raw.append((a,b))
        except (ValueError,TypeError,RecursionError): pass
    # Quote normalization must not corrupt the boundary of an already-valid
    # raw object containing typographic quotes *inside* an ordinary string.
    normalized_spans = [(a,b) for a,b in spans(output.translate(SMART))
                        if not any(c <= a < d for c,d in valid_raw)]
    candidates = sorted(set(raw_spans + normalized_spans))
    outer = [(a,b) for a,b in candidates if not any(c <= a and b <= d and (c,d)!=(a,b) for c,d in candidates)]
    result = {'plan':None, 'stage':'missing', 'raw_ok':False, 'normalized_ok':False,
              'recovered':False, 'block':None, 'candidate_count':len(outer)}
    if not outer: return result
    a,b = max(outer, key=lambda x:x[0]); raw = output[a:b]
    result.update(block=raw, start=a, end=b, stage='invalid')
    for stage, text in [('raw',raw), ('normalized',raw.translate(SMART))]:
        try:
            plan = strict_load(text)
            if not isinstance(plan,dict): continue
            result.update(plan=plan, stage=stage, raw_ok=stage=='raw', normalized_ok=True, recovered=True)
            return result
        except (ValueError,TypeError,RecursionError): pass
    try:
        plan = strict_load(repair_json(raw.translate(SMART)))
        if isinstance(plan,dict): result.update(plan=plan,stage='repaired',recovered=True)
    except (ValueError,TypeError,RecursionError): pass
    return result


DAY_GROUPS = [('luni','monday','montag'),('marți','tuesday','dienstag'),
 ('miercuri','wednesday','mittwoch'),('joi','thursday','donnerstag'),
 ('vineri','friday','freitag'),('sâmbătă','saturday','samstag'),('duminică','sunday','sonntag')]
SLOT_GROUPS = [('dimineață','morning','morgen','vormittag'),('după-amiază','afternoon','nachmittag'),('seară','evening','abend')]
MEAL_GROUPS = [('mic_dejun','breakfast','frühstück','fruehstueck'),('pranz','lunch','mittagessen'),('cina','dinner','abendessen')]


def alias(s, groups):
    n = normalize(s)
    for group in groups:
        if n in {normalize(v) for v in group}: return group[0]
    return s


def entity_name(world, value):
    if not isinstance(value,str): return value
    eid = inherited._resolve_entity_id(world,value)
    return world.canonical_entities[eid].name if eid in world.canonical_entities else value


def resolve_unique(world, ref):
    if not isinstance(ref,str): return '__INVALID_ENTITY_TYPE__'
    if ref in world.canonical_entities:return ref
    if ref.startswith('__UNRESOLVED__'):return ref
    exact={eid for eid,e in world.canonical_entities.items() if normalize(e.name)==normalize(ref)}
    if len(exact)==1:return next(iter(exact))
    matches={eid for eid,e in world.canonical_entities.items()
             if normalize(ref) in {normalize(n) for n in [e.name]+e.aliases}}
    if len(matches)==1:return next(iter(matches))
    if not matches:
        matches={eid for eid,e in world.canonical_entities.items()
                 if inherited._normalize_entity_ref(ref) in {inherited._normalize_entity_ref(n) for n in [e.name]+e.aliases}}
        if len(matches)==1:return next(iter(matches))
    return '__UNRESOLVED__'+ref


# All inherited semantic checks now use the same language-independent resolver.
inherited._resolve_entity_id=resolve_unique


def canonical_plan(world, plan):
    if not isinstance(plan,dict): return None,False
    out = {}; collision = False
    for key,value in plan.items():
        k = key
        if world.world_type=='schedule' and '_' in key:
            day,slot=key.split('_',1); k=alias(day,DAY_GROUPS)+'_'+alias(slot,SLOT_GROUPS)
        elif world.world_type=='recipe' and '_' in key:
            day,meal=key.split('_',1); k=day+'_'+alias(meal,MEAL_GROUPS)
        if k in out: collision=True
        if world.world_type!='fact':
            value = [entity_name(world,v) for v in value] if isinstance(value,list) else entity_name(world,value)
        out[k]=value
    return out,collision


def expected_keys(world):
    p=world.payload
    if world.world_type=='travel': return {f'day{i}' for i in range(1,p['num_days']+1)}
    if world.world_type=='recipe': return {f'day{i}_{alias(m,MEAL_GROUPS)}' for i in range(1,p['num_days']+1) for m in p.get('meals_per_day',['mic_dejun','pranz','cina'])}
    if world.world_type=='fact': return {'answer'}
    days=next(p[k] for k in ['days_ro','days_en','days_de','days'] if p.get(k))
    slots=next(p[k] for k in ['slots_ro','slots_en','slots_de','slots'] if p.get(k))
    return {alias(d,DAY_GROUPS)+'_'+alias(s,SLOT_GROUPS) for d in days for s in slots}


def correct_value_types(world,plan):
    if not isinstance(plan,dict): return False
    if world.world_type=='travel': return all(isinstance(v,list) and all(isinstance(x,str) for x in v) for v in plan.values())
    if world.world_type=='fact': return isinstance(plan.get('answer'),str)
    return all(v is None or isinstance(v,str) for v in plan.values())


def priority_slot(world,plan,params):
    allowed={alias(s,SLOT_GROUPS) for s in params['required_slots']}
    for key,v in plan.items():
        if not v or v=='null': continue
        eid=inherited._resolve_entity_id(world,v)
        e=world.canonical_entities.get(eid)
        if e and e.attributes.get('priority')==params['priority']:
            if '_' not in key or alias(key.split('_',1)[1],SLOT_GROUPS) not in allowed: return False
    return True


FUNCTIONS = dict(inherited.CONSTRAINT_FUNCTIONS)
FUNCTIONS['check_priority_slot_restriction']=priority_slot


def checks(world):
    return [c for c in world.constraints if c.type.value=='instruction'] + [g for g in world.goals if g.type.value=='structural']


def score(instance, output):
    w=instance.world; p=parse(output); plan,collision=canonical_plan(w,p['plan'])
    types=correct_value_types(w,plan)
    schema=bool(p['recovered'] and not collision and types and set(plan)==expected_keys(w))
    results=[]
    for c in checks(w):
        if c.check_fn not in FUNCTIONS: raise ValueError('Unknown checker '+c.check_fn)
        if not isinstance(plan,dict) or not plan or collision or not types:
            ok=False
        elif w.world_type=='fact':
            ok=answer_ok(w,plan)
        else:
            try: ok=bool(FUNCTIONS[c.check_fn](w,plan,c.params))
            except (TypeError,ValueError,AttributeError,KeyError,IndexError): ok=False
        results.append({'id':c.id,'check_fn':c.check_fn,'satisfied':ok})
    t=sum(c['satisfied'] for c in results)/len(results)
    return {'instance_id':instance.instance_id,'family':w.world_type,'T':t,
            'raw_ok':p['raw_ok'],'normalized_ok':p['normalized_ok'],'recovered':p['recovered'],
            'schema':schema,'joint':schema and t==1,'QA':answer_ok(w,plan) and schema if w.world_type=='fact' else None,
            'parse_stage':p['stage'],'candidate_count':p['candidate_count'],
            'parse_start':p.get('start'),'parse_end':p.get('end'),
            'constraints':results}


def dataset_audit(raws):
    seen={}; entries=[]
    for r in raws:
        w=r['world']; reasons=[]
        signature=hashlib.sha256(json.dumps({k:w[k] for k in ['payload','constraints','goals','canonical_entities']},sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        duplicate=seen.get(signature); seen.setdefault(signature,r['instance_id'])
        if duplicate: reasons.append('duplicate_content')
        if any(c['check_fn'] in HIDDEN_CHECKS for c in w['constraints']): reasons.append('required_timing_information_not_shown')
        if w['world_type']=='travel':
            available={e['attributes'].get('type') for e in w['canonical_entities'].values()}
            forbidden={c['params']['type_forbidden'] for c in w['constraints'] if c['check_fn']=='check_must_exclude_type'}
            required={c['params']['type_required'] for c in w['constraints'] if c['check_fn']=='check_must_include_type'}
            min_types=max([c['params'].get('min_types',0) for c in w['constraints'] if c['check_fn']=='check_type_diversity'] or [0])
            if min_types>len(available-forbidden) or not required <= available-forbidden: reasons.append('proven_infeasible_type_constraints')
        unknown=[c['check_fn'] for c in w['constraints']+w['goals'] if c['check_fn'] not in FUNCTIONS]
        if unknown: raise ValueError('Unregistered checks '+repr(unknown))
        entries.append({'instance_id':r['instance_id'],'family':w['world_type'],'world_hash':signature,
            'duplicate_of':duplicate,'exclusion_reasons':reasons,'eligible':not reasons})
    return entries


def ci(x, seed=20240617):
    x=np.asarray(x,dtype=float); rng=np.random.default_rng(seed)
    samples=x[rng.integers(0,len(x),(5000,len(x)))].mean(axis=1)
    return {'mean':float(x.mean()),'low':float(np.quantile(samples,.025)),
            'high':float(np.quantile(samples,.975)),
            'p':max(1/5000,min(1.,2*min(float(np.mean(samples<=0)),float(np.mean(samples>=0)))))}


def holm(items):
    order=sorted(range(len(items)),key=lambda i:items[i]['p']); last=0
    for j,i in enumerate(order):
        last=max(last,min(1.,(len(items)-j)*items[i]['p']))
        items[i]['p_holm']=last


def main():
    data={}; audits={}; scored={}; summaries=[]; records=[]
    for split,file in [('standard','v0_DE_test.jsonl'),('adversarial','hard.jsonl')]:
        raw=[json.loads(x) for x in (ROOT/'inputs'/file).read_text().splitlines()]
        data[split]={r['instance_id']:Instance.from_dict(r) for r in raw}
        audits[split]=dataset_audit(raw)
        eligible={r['instance_id'] for r in audits[split] if r['eligible']}
        for lang in (['ro','en','de'] if split=='standard' else ['ro']):
            folder='outputs_de_test_'+lang if split=='standard' else 'outputs_hard_ro'
            for model in MODELS:
                outputs=[json.loads(x) for x in (ROOT/'inputs'/folder/f'{lang}_{model}.jsonl').read_text().splitlines()]
                if len(outputs)!=len(data[split]) or {x['instance_id'] for x in outputs} != set(data[split]): raise ValueError('Mismatched or duplicate IDs')
                all_scores=[score(data[split][o['instance_id']],o['output']) for o in outputs]
                chosen=sorted((r for r in all_scores if r['instance_id'] in eligible),key=lambda r:r['instance_id'])
                scored[(split,lang,model)]=chosen
                summary={'split':split,'lang':lang,'model':model,'n':len(chosen)}
                for key in ['T','raw_ok','normalized_ok','recovered','schema','joint']:
                    summary[key]=ci([r[key] for r in chosen])
                summary['QA']=ci([r['QA'] for r in chosen if r['QA'] is not None])
                summary['QA_n']=sum(r['QA'] is not None for r in chosen)
                summary['families']={f:{'n':sum(r['family']==f for r in chosen),
                    'T':float(np.mean([r['T'] for r in chosen if r['family']==f])),
                    'joint':float(np.mean([r['joint'] for r in chosen if r['family']==f]))}
                    for f in ['travel','schedule','fact','recipe']}
                summaries.append(summary)
                records.extend(dict(r,split=split,lang=lang,model=model,eligible=r['instance_id'] in eligible) for r in all_scores)
    contrasts=[]
    for base,adapt in PAIRS:
        for lang in ['ro','en','de']:
            for metric in ['T','raw_ok','joint']:
                b=scored[('standard',lang,base)]; a=scored[('standard',lang,adapt)]
                delta=ci([float(x[metric])-float(y[metric]) for x,y in zip(a,b)],42)
                contrasts.append(dict(base=base,adapted=adapt,lang=lang,metric=metric,**delta))
    holm(contrasts)
    did=[]; lang_deltas=[]
    for model in MODELS:
        en=scored[('standard','en',model)]; ro=scored[('standard','ro',model)]
        lang_deltas.append(dict(model=model,**ci([x['T']-y['T'] for x,y in zip(en,ro)],42)))
    holm(lang_deltas)
    for b,a in PAIRS:
        ar=scored[('standard','ro',a)]; ae=scored[('standard','en',a)]
        br=scored[('standard','ro',b)]; be=scored[('standard','en',b)]
        did.append(dict(base=b,adapted=a,**ci([x['T']-y['T']-z['T']+q['T'] for x,y,z,q in zip(ae,ar,be,br)],42)))
    holm(did)
    report={'method':'proposal audit v1; fixed generations; post-hoc rules; no training',
        'bootstrap':{'n':5000,'engine':'numpy.default_rng','single_seed':20240617,'paired_seed':42},
        'versions':{p:importlib.metadata.version(p) for p in ['numpy','json-repair']},
        'dataset_audits':audits,'summaries':summaries,'paired_contrasts':contrasts,'did':did,'language_deltas':lang_deltas}
    (ROOT/'evidence/corrected_results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    with (ROOT/'evidence/rescored_records.jsonl').open('w') as f:
        for r in records:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    (ROOT/'evidence/answer_aliases.json').write_text(json.dumps(ANSWER_ALIASES,ensure_ascii=False,indent=2))
    print(json.dumps({'eligible':{s:sum(x['eligible'] for x in v) for s,v in audits.items()},'did':did,
        'RO':[{'model':x['model'],'T':x['T']['mean'],'raw':x['raw_ok']['mean'],'schema':x['schema']['mean'],'joint':x['joint']['mean'],'QA':x['QA']['mean']} for x in summaries if x['split']=='standard' and x['lang']=='ro']},indent=2))


if __name__=='__main__':main()
