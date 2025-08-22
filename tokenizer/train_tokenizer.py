from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.processors import TemplateProcessing

from datasets import load_dataset

ds = load_dataset('OmniAICreator/Japanese-Wikipedia-202506')
def train_dataset():
    for s in ds['train']['text']:
        yield s

# === 入力 ===
# corpus_dir = Path("corpus")  # 中に *.txt を置く（巨大でもOK、逐次読み込み）
# save_dir = Path("artifacts"); save_dir.mkdir(parents=True, exist_ok=True)

# === モデル本体 ===
tokenizer: Tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
tokenizer.decoder = ByteLevelDecoder()

special_tokens = [
    "<pad>", "<bos>", "<eos>", "<unk>",
    "<think>", "</think>", "<answer>", "</answer>"
]

trainer = BpeTrainer(
    vocab_size=80000,
    min_frequency=2,
    special_tokens=special_tokens,
    initial_alphabet=ByteLevel.alphabet(),
    # continuing_subword_prefix=None,
)

# files = [str(p) for p in corpus_dir.glob("**/*.txt")]
# assert files, "corpus/*.txt が見つかりません。"

# tokenizer.train(files=files, trainer=trainer)
tokenizer.train_from_iterator(iterator=train_dataset(), trainer=trainer)

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
