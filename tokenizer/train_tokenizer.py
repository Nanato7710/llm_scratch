from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.processors import TemplateProcessing

from datasets import load_dataset, interleave_datasets

# ds = load_dataset('OmniAICreator/Japanese-Wikipedia-202506', split='train', streaming=True).shuffle(seed=42)
ds_jp = load_dataset('HuggingFaceFW/fineweb-2', 'jpn_Jpan', split='train', streaming=True)
ds_en = load_dataset('HuggingFaceFW/fineweb', 'default', split='train', streaming=True)
# ds = ds.remove_columns(['id', 'title', 'raw_text'])
ds_jp = ds_jp.remove_columns(['id', 'dump', 'url', 'date', 'file_path'])
ds_en = ds_en.remove_columns(['id', 'dump', 'url'])
mixed_ds = interleave_datasets([ds_jp, ds_en], probabilities=[0.6, 0.4], seed=42, stopping_strategy='first_exhausted')

# def train_dataset():
#     i = 0
#     for s in mixed_ds['text']:
#         if i % 1024 == 0:
#             yield s
#         i += 1

def train_dataset(n: int):
    ds = iter(mixed_ds)
    for _ in range(n):
        yield next(ds)['text']


# === 入力 ===
# corpus_dir = Path("corpus")  # 中に *.txt を置く（巨大でもOK、逐次読み込み）
save_dir = Path("artifacts"); save_dir.mkdir(parents=True, exist_ok=True)

# === モデル本体 ===
tokenizer: Tokenizer = Tokenizer(BPE(unk_token=None))
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
tokenizer.decoder = ByteLevelDecoder()

special_tokens = [
    "<pad>", "<bos>", "<eos>",
    "<think>", "</think>", "<answer>", "</answer>",
    "<user>", "</user>", "<assistant>", "</assistant>",
    "<system>", "</system>"
]

trainer = BpeTrainer(
    vocab_size=80_000,
    min_frequency=2,
    special_tokens=special_tokens,
    initial_alphabet=ByteLevel.alphabet(),
    # continuing_subword_prefix=None,
)

# files = [str(p) for p in corpus_dir.glob("**/*.txt")]
# assert files, "corpus/*.txt が見つかりません。"

# tokenizer.train(files=files, trainer=trainer)
tokenizer.train_from_iterator(iterator=train_dataset(400_000), trainer=trainer)

# --- GPT系ポストプロセッサ（BOS/EOS自動付与、不要ならコメントアウト） ---
# tokenizer.post_processor = TemplateProcessing(
#     single="<bos> $A <eos>",
#     pair="<bos> $A <eos> <bos>:1 $B:1 <eos>:1",
#     special_tokens=[
#         ("<bos>", tokenizer.token_to_id("<bos>")),
#         ("<eos>", tokenizer.token_to_id("<eos>")),
#     ],
# )

# === 保存（2系統） ===
# 1) tokenizer.json（transformers からそのまま読める）
tokenizer.save(str(save_dir / "tokenizer.json"))

# 2) vocab.json / merges.txt（解析・互換用途）
tokenizer.model.save(str(save_dir))

print("Saved to:", save_dir.resolve())
