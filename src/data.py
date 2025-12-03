from datasets import load_dataset
from torch.utils.data import Dataset
from typing import Any, Dict, List, Optional

class TranslationDataset(Dataset):
    """
    Dataset wrapper for HF translation splits.

    Each example contains:
        {"translation": {src_lang: <src>, tgt_lang: <tgt>}}
    """

    def __init__(self, hf_split, src_lang: str = "de", tgt_lang: str = "en"):
        """
        Args:
            hf_split: HuggingFace dataset split object.
            src_lang: Key for the source language string.
            tgt_lang: Key for the target language string.
        """
        self.data = hf_split
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]


class TranslationCollator:
    """
    Collator that tokenizes batches of translation pairs using HF tokenizers.
    Handles padding, truncation, and label preparation for seq2seq training.
    """

    def __init__(
        self,
        src_tokenizer,      # tokenizer for the source language
        tgt_tokenizer,      # tokenizer for the target language
        max_src_len: int = 128,
        max_tgt_len: int = 128,
        pad_to_max: bool = True
    ):
        """
        Args:
            src_tokenizer: Tokenizer for source text.
            tgt_tokenizer: Tokenizer for target text.
            max_src_len: Maximum source sequence length.
            max_tgt_len: Maximum target sequence length.
            pad_to_max: Pad to max length instead of dynamic padding.
        """
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.pad_to_max = pad_to_max

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Tokenizes a batch of translation pairs.

        Args:
            batch: List of samples from the dataset split.

        Returns:
            Dict of batched tensors for model input/output.
        """
        src_texts = [item["translation"]["de"] for item in batch]
        tgt_texts = [item["translation"]["en"] for item in batch]

        src_enc = self.src_tokenizer(
            src_texts,
            padding="max_length" if self.pad_to_max else True,
            truncation=True,
            max_length=self.max_src_len,
            return_tensors="pt"
        )

        tgt_enc = self.tgt_tokenizer(
            tgt_texts,
            padding="max_length" if self.pad_to_max else True,
            truncation=True,
            max_length=self.max_tgt_len,
            return_tensors="pt"
        )

        # Prepare labels: mask out padding for CE loss
        labels = tgt_enc["input_ids"].clone()
        pad_id = self.tgt_tokenizer.pad_token_id or 0
        labels[labels == pad_id] = -100

        return {
            "src_input_ids": src_enc["input_ids"],
            "src_pad_mask": src_enc["attention_mask"].bool(),
            "tgt_input_ids": tgt_enc["input_ids"],
            "tgt_pad_mask": tgt_enc["attention_mask"].bool(),
            "tgt_outputs": labels,
        }


def load_wmt17_subset(
    split: str = "train",
    size: Optional[int] = 50000,
    src_lang: str = "de",
    tgt_lang: str = "en"
):
    """
    Loads the WMT17 de–en dataset split and optionally returns a subset.

    Args:
        split: One of 'train', 'validation', 'test'.
        size: Number of samples to keep. None loads the full split.
        src_lang: Source language code.
        tgt_lang: Target language code.

    Returns:
        HuggingFace Dataset object (full or sliced).
    """
    ds = load_dataset("wmt17", "de-en", split=split)
    if size is not None and size > 0 and len(ds) > size:
        ds = ds.select(range(size))
    return ds
