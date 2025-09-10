from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPretokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

from pathlib import Path
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

from data.get_pre_ds import get_mixed_dataset


def train_dataset(n: int):
    ds = iter(get_mixed_dataset())
    for _ in range(n):
        yield next(ds)['text']


special_tokens = [
    "<|startoftext|>", "<|endoftext|>",
    "<|start|>", "<|end|>", "<|message|>",
    "<|channel|>", "<|constrain|>", "<|return|>", "<|call|>"
]

tok = Tokenizer(BPE(unk_token=None, byte_fallback=False))
tok.pre_tokenizer = ByteLevelPretokenizer(add_prefix_space=False)
tok.decoder = ByteLevelDecoder()

trainer = BpeTrainer(
    vocab_size=80_000,
    min_frequency=2,
    initial_alphabet=ByteLevelPretokenizer.alphabet(),
    special_tokens=special_tokens,
    show_progress=True
)

# tok.add_special_tokens({"bos_token": "<|startoftext|>", "eos_token": "<|return|>", "pad_token": "<|endoftext|>"})

tok.train_from_iterator(iterator=train_dataset(400_000), trainer=trainer)


save_dir = Path("artifacts"); save_dir.mkdir(parents=True, exist_ok=True)
tok.save(os.path.join(save_dir, "tokenizer.json"))

