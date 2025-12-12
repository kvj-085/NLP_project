import json

paths = ['data/raw/fever/shared_task_dev.jsonl', 'data/raw/fever/shared_task_test.jsonl']
for path in paths:
    print('---', path)
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            for i in range(5):
                line = fh.readline()
                if not line:
                    break
                try:
                    obj = json.loads(line)
                except Exception as e:
                    print('  line', i, 'parse error:', e)
                    continue
                print('  id=', obj.get('id'), 'label=', obj.get('label'))
                claim = obj.get('claim')
                if claim:
                    print('   claim:', claim[:120].replace('\n',' '))
    except FileNotFoundError:
        print('  file not found')
    print()
