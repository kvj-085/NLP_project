import json

path = 'data/raw/fever/train.jsonl'
print('Inspecting', path)
with open(path, 'r', encoding='utf-8') as fh:
    for i in range(10):
        line = fh.readline()
        if not line:
            break
        try:
            obj = json.loads(line)
        except Exception as e:
            print('Line', i, 'failed to parse:', e)
            continue
        eid = obj.get('id')
        claim = obj.get('claim')
        label = obj.get('label')
        evidence = obj.get('evidence')
        ev_preview = None
        try:
            if isinstance(evidence, list) and len(evidence) > 0:
                ev_preview = evidence[0]
            else:
                ev_preview = evidence
        except Exception:
            ev_preview = str(evidence)
        print(f'RECORD {i}: id={eid} claim_len={len(claim) if claim else 0} label={label} evidence_preview_type={type(ev_preview).__name__}')
        if claim:
            print('  claim sample:', claim[:200].replace('\n',' '))
        if ev_preview:
            print('  evidence preview:', str(ev_preview)[:200].replace('\n',' '))
        print('')
print('Done')
