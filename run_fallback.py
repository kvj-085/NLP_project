import traceback

from src.data_preprocessing import prepare_fever_via_hf_hub

if __name__ == '__main__':
    try:
        print('Running HF-hub fallback to prepare FEVER...')
        out = prepare_fever_via_hf_hub()
        print('prepare_fever_via_hf_hub finished, processed data at:', out)
    except Exception:
        print('prepare_fever_via_hf_hub raised an exception:')
        traceback.print_exc()
