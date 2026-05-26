"""
Preprocessing functions for mBERT and Gemma 2.
These are factory functions that return a preprocessing function that should be used
with SQuAD-style datasets via map method.

Usage:
preprocess_fn = factory_preprocess(tokenizer, max_seq_length)
dataset.map(preprocess_fn, batched=True, remove_columns=dataset.column_names)
"""

# Standard imports
import logging
from typing import Callable

# Third party imports
from datasets import Dataset
from transformers import AutoTokenizer

# Set up logger
logger = logging.getLogger(__name__)


def factory_preprocess_mbert(
    tokenizer: AutoTokenizer, max_seq_length: int, doc_stride: int = 128
) -> Callable[[Dataset], dict]:
    """
    Factory function to create a preprocessing function for mBERT.
    To be used with SQuAD-style datasets via map method.

    Args:
        tokenizer: The tokenizer to use for encoding the inputs.
        max_seq_length: The maximum total input sequence length after tokenization.
        doc_stride: The stride to take when splitting up a long sequence into chunks.

    Returns:
        A preprocessing function that can be used with Dataset.map().
    """
    # Validate and set max length with tokenizer limit as the upper bound.
    if max_seq_length <= 0:
        raise ValueError(f"max_seq_length must be positive, got {max_seq_length}")
    # Max seq length for mBERT is 512 tokens.
    tkn_seq_len_limit = getattr(tokenizer, "model_max_length", max_seq_length)
    max_length = min(tkn_seq_len_limit, max_seq_length)

    # Validate doc_stride
    if doc_stride <= 0 or doc_stride >= max_length:
        raise ValueError(
            f"doc_stride must be >0 and < max_length ({max_length}),got {doc_stride}"
        )

    # Log the preprocessing configuration
    logger.info(
        f"mBERT preprocessing function set with max_length={max_length} "
        f"and doc_stride={doc_stride}."
    )

    def preprocess_fn(samples: dict[str, list]) -> dict[str, list]:
        """
        Usage:
        Dataset.map(fn, batched=True, remove_columns=dataset.column_names)
        """
        # Strip leading/trailing whitespace from questions
        # Context is not stripped to preserve answer character positions
        questions = [q.strip() for q in samples["question"]]

        inputs = tokenizer(  # type: ignore
            questions,
            samples["context"],
            max_length=max_length,
            truncation="only_second",
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
            stride=doc_stride,  # Shift when over max len(overlap = max length - stride)
        )

        sample_map = inputs.pop("overflow_to_sample_mapping")
        answers = samples["answers"]
        start_positions = []
        end_positions = []
        sample_ids = []

        for i, offset in enumerate(inputs["offset_mapping"]):
            sample_idx = sample_map[i]
            answer = answers[sample_idx]
            # Only first answer is used for training
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while idx < len(sequence_ids) and sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while idx < len(sequence_ids) and sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # Store example id, and nullify offset_mapping for non-context tokens
            # (required by compute_metrics)
            sample_ids.append(samples["id"][sample_idx])
            inputs["offset_mapping"][i] = [
                o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
            ]

            # If the answer is not fully inside the context, label it (0, 0)
            if (
                offset[context_start][0] > end_char
                or offset[context_end][1] < start_char
            ):
                start_positions.append(0)
                end_positions.append(0)
            else:
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["example_id"] = sample_ids
        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions

        return inputs

    return preprocess_fn


def factory_preprocess_gemma2_train(
    tokenizer: AutoTokenizer, max_seq_length: int
) -> Callable[[Dataset], dict]:
    """
    Factory function to create a preprocessing function for Gemma 2.
    To be used with SQuAD-style datasets via map method.

    Args:
        tokenizer: The tokenizer to use for encoding the inputs.
        max_seq_length: The maximum total input sequence length after tokenization.

    Returns:
        A preprocessing function that can be used with Dataset.map().
    """
    # Validate and set max length with set upper bound based on tokenizer specs.
    if max_seq_length <= 0:
        raise ValueError(f"max_seq_length must be positive, got {max_seq_length}")
    # Cap max length at tokenizer limit.
    # Gemma 2 can support up to 8192 tokens with a 4096 token sliding window.
    # The window size can be confirmed by checking model.config.sliding_window).
    # We will use 70% of the sliding window as a hard cap to be safe (~2867 tokens).
    max_length = min(int(4096 * 0.7), max_seq_length)

    # Log the preprocessing configuration
    logger.info(f"Gemma 2 preprocessing function set with max_length={max_length}")

    def preprocess_fn(examples: dict[str, list]) -> dict[str, list]:
        """
        Usage:
        Dataset.map(fn, batched=True, remove_columns=dataset.column_names)
        """
        questions = [q.strip() for q in examples["question"]]
        contexts = [c.strip() for c in examples["context"]]
        answers = [a["text"][0].strip() for a in examples["answers"]]

        formatted_prompt = [
            f"Pregunta: {q}\nContexto: {c}\nRespuesta:"
            for q, c in zip(questions, contexts)
        ]  # Trailing space removed to avoid count issues with tokenization

        formatted_input = [
            f"Pregunta: {q}\nContexto: {c}\nRespuesta: {a}{tokenizer.eos_token}"
            for q, c, a in zip(questions, contexts, answers)
        ]

        # Tokenize without answer to measure prompt length for attention mask
        token_prompt = tokenizer(  # type: ignore
            formatted_prompt,
            max_length=max_length,
            truncation=True,
            padding=False,  # No padding for prompt
        )

        # Tokenize full input
        token_input = tokenizer(  # type: ignore
            formatted_input,
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )

        # Build labels
        labels = []

        for i in range(len(token_input["input_ids"])):
            prompt_length = len(token_prompt["input_ids"][i])
            input_ids = token_input["input_ids"][i]
            attention_mask = token_input["attention_mask"][i]

            padding_length = attention_mask.count(
                0
            )  # how many padding tokens are at the left of the sequence

            label = [
                (
                    -100
                    if attention_mask[j] == 0  # padding positions
                    or j < padding_length + prompt_length  # prompt positions
                    else input_ids[j]
                )  # answer positions
                for j in range(len(input_ids))
            ]
            labels.append(label)

        token_input["labels"] = labels

        return token_input

    return preprocess_fn
