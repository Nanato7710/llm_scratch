from datasets import load_dataset, interleave_datasets

def get_mixed_dataset(jp_ratio=0.6):
    ds_jp = load_dataset('HuggingFaceFW/fineweb-2', 'jpn_Jpan', split='train', streaming=True)
    ds_en = load_dataset('HuggingFaceFW/fineweb', 'default', split='train', streaming=True)
    ds_jp = ds_jp.remove_columns(['id', 'dump', 'url', 'date', 'file_path'])
    ds_en = ds_en.remove_columns(['id', 'dump', 'url'])
    mixed_ds = interleave_datasets([ds_jp, ds_en], probabilities=[jp_ratio, 1 - jp_ratio], seed=42, stopping_strategy='first_exhausted')
    return mixed_ds
