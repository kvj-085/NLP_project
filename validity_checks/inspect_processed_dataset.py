from datasets import load_from_disk

def main():
    d = load_from_disk('data/processed/fever')
    print('Splits and sizes:')
    for k in d.keys():
        print(f'  {k}:', len(d[k]))

    print('\nSample train examples (first 10):')
    for i in range(min(10, len(d['train']))):
        ex = d['train'][i]
        text = ex.get('text', '')
        label = ex.get('label', repr(ex.get('label')))
        print(f'{i}: label={label} text={text[:200].replace(chr(10), " ").replace(chr(13), " ")}')

    # show label types and unique labels
    labels = [ex.get('label') for ex in d['train']]
    print('\nLabel sample types (first 20):', [type(x).__name__ for x in labels[:20]])
    try:
        uniq = sorted(set(labels))
        print('Unique labels in train:', uniq)
    except Exception as e:
        print('Could not list unique labels:', e)

if __name__ == '__main__':
    main()
