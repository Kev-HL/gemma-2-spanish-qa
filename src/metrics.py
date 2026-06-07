"""
Functions to compute evaluation metrics and related auxiliary functions.

Metrics are computed using the Hugging Face 'evaluate' library, which implements
the standard SQuAD (1.1) evaluation script.
Metrics computed are F1 score and Exact Match (EM).
"""

# Standard imports
import logging
from typing import Callable

# Third party imports
import evaluate
import numpy as np
import torch
from datasets import Dataset
from rapidfuzz.fuzz import partial_ratio_alignment
from transformers import AutoTokenizer, EvalPrediction

# Set up logger
logger = logging.getLogger(__name__)


def factory_compute_metrics_mbert(
    raw_dataset: Dataset,
    preprocessed_dataset: Dataset,
    n_best: int = 20,
    max_answer_length: int = 50,
) -> Callable[[EvalPrediction], dict[str, float]]:
    """
    Factory function to create a compute_metrics function for mBERT-based QA models.

    Computes F1 and Exact Match metrics for SQuAD style QA.

    Args:
        raw_dataset: The original dataset containing the context and answers.
        preprocessed_dataset: The dataset after preprocessing including tokenization.
        n_best: The number of top predictions to consider for each example.
        max_answer_length: The maximum length of an answer span.

    Returns:
        A function that computes F1 and Exact Match metrics given model predictions,
        compatible with Hugging Face Trainer's compute_metrics argument.
    """
    # Validate n_best and max_answer_length
    if n_best <= 0:
        raise ValueError(f"n_best must be positive, got {n_best}")
    if max_answer_length <= 0:
        raise ValueError(f"max_answer_length must be positive, got {max_answer_length}")

    # Load SQuAD metric for evaluation
    squad_metric = evaluate.load("squad")
    logger.info("SQuAD metric loaded for mBERT")

    # Log compute_metrics configuration
    logger.info(
        f"mBERT compute_metrics function set up with n_best={n_best} and "
        f"max_answer_length={max_answer_length}."
    )

    def compute_metrics(eval_preds: EvalPrediction) -> dict[str, float]:
        """
        Compute F1 and Exact Match metrics for mBERT-based QA models.

        Args:
            eval_preds: EvalPrediction object containing model predictions and labels.

        Returns:
            A dictionary with F1 and Exact Match scores.
        """
        # Information passed from Trainer
        start_logits, end_logits = (
            eval_preds.predictions
        )  # each (num_examples, max_seq_len)

        # Map example_id to feature indices
        example_to_features = {}
        for idx, feature in enumerate(preprocessed_dataset):
            eid = feature["example_id"]
            if eid not in example_to_features:
                example_to_features[eid] = []
            example_to_features[eid].append(
                idx
            )  # for each id, store all associated feature indices

        # predictions [{'id': ..., 'prediction_text': ...}, ...]
        predictions = []
        for example in raw_dataset:
            example_id = example["id"]
            context = example["context"]
            answers = []

            # Safety check in case id from raw dataset not in preprocessed dataset
            if example_id not in example_to_features:
                logger.warning(f"Example {example_id} has no preprocessed features")
                predictions.append({"id": example_id, "prediction_text": ""})
                continue

            # Loop through all features associated with that example
            for feature_index in example_to_features[example_id]:
                start_logit = start_logits[feature_index]
                end_logit = end_logits[feature_index]
                offsets = preprocessed_dataset[feature_index]["offset_mapping"]

                start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
                end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # Skip answers that are not fully in the context
                        if offsets[start_index] is None or offsets[end_index] is None:
                            continue
                        # Skip answers with a length that  < 0 or > max_answer_length
                        if (
                            end_index < start_index
                            or end_index - start_index + 1 > max_answer_length
                        ):
                            continue

                        answer = {
                            "text": context[
                                offsets[start_index][0] : offsets[end_index][1]
                            ],
                            "logit_score": start_logit[start_index]
                            + end_logit[end_index],
                        }
                        answers.append(answer)

            # Select the answer with the best score
            if len(answers) > 0:
                best_answer = max(answers, key=lambda x: x["logit_score"])
                predictions.append(
                    {"id": example_id, "prediction_text": best_answer["text"]}
                )
            else:
                predictions.append({"id": example_id, "prediction_text": ""})

        # references format
        # [{'id': ..., 'answers': {'text': [...], 'answer_start': [...]}} , ...]
        references = [{"id": ex["id"], "answers": ex["answers"]} for ex in raw_dataset]

        # Compute metrics
        try:
            metrics = squad_metric.compute(
                predictions=predictions, references=references
            )
        except Exception as e:
            logger.error(f"Failed to compute metrics: {e}")
            raise

        return {"f1": metrics["f1"], "exact_match": metrics["exact_match"]}

    return compute_metrics


def factory_compute_metrics_gemma2(
    raw_dataset: Dataset, tokenizer: AutoTokenizer, fuzzy_threshold: int = 100
) -> Callable[[EvalPrediction], dict[str, float]]:
    """
    Factory function to create a compute_metrics function for causal LMs.
    Intended to be used with Gemma 2, but could be adapted for other similar models.

    Fuzzy matching is disabled by default, but can be enabled by setting a
    threshold < 100.

    Computes F1 and Exact Match metrics for SQuAD style QA.

    Args:
        raw_dataset: The original dataset containing the context and answers.
        tokenizer: The tokenizer used for encoding inputs, needed to decode predictions.

    Returns:
        A function that computes F1 and Exact Match scores given model predictions,
        compatible with Hugging Face Trainer's compute_metrics argument.
    """

    # Load SQuAD metric for evaluation
    squad_metric = evaluate.load("squad")
    logger.info("SQuAD metric loaded for Gemma 2")

    def compute_metrics(eval_preds: EvalPrediction) -> dict[str, float]:
        """
        Compute F1 and Exact Match metrics for causal LM-based QA models like Gemma 2.

        Args:
            eval_preds: EvalPrediction object containing model predictions and labels.

        Returns:
            A dictionary with F1 and Exact Match scores.
        """
        # Get predictions from Trainer
        encoded_preds, labels = eval_preds

        # Mask out prompt and padding tokens in the predictions with
        # tokenizer.pad_token_id, so that they are ignored when decoding
        encoded_preds[labels == -100] = tokenizer.pad_token_id

        # Decode answers
        decoded_preds = tokenizer.batch_decode(encoded_preds, skip_special_tokens=True)

        # Ground predictions to context using fuzzy matching
        if fuzzy_threshold < 100:
            for i in range(len(decoded_preds)):
                decoded_preds[i] = ground_prediction_to_context(
                    decoded_preds[i],
                    raw_dataset[i]["context"],
                    threshold=fuzzy_threshold,
                )

        # Build predictions in SQuAD format using dataset order
        # [{'id': ..., 'prediction_text': ...}, ...]
        predictions = [
            {"id": raw_dataset[i]["id"], "prediction_text": decoded_preds[i].strip()}
            for i in range(len(decoded_preds))
        ]

        # Get GT answers from raw dataset
        # [{'id': ..., 'answers': {'text': [...], 'answer_start': [...]}} , ...]
        references = [{"id": ex["id"], "answers": ex["answers"]} for ex in raw_dataset]

        # Compute metrics
        try:
            metrics = squad_metric.compute(
                predictions=predictions, references=references
            )
        except Exception as e:
            logger.error(f"Failed to compute metrics: {e}")
            raise

        return {"f1": metrics["f1"], "exact_match": metrics["exact_match"]}

    return compute_metrics


def preprocess_logits_for_metrics(
    logits: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """
    Preprocess logits to get predicted token IDs for metric computation.
    This function is used to convert raw model outputs (logits) into predicted token IDs
    taking the max logit as the predicted token for each position.
    Args:
        logits: Raw model outputs of shape (batch_size, seq_len, vocab_size)
        labels: Ground truth labels of shape (batch_size, seq_len)
    Returns:
        A tensor of shape (batch_size, seq_len) containing the predicted token IDs.
    """
    return logits.argmax(dim=-1)


def ground_prediction_to_context(
    prediction: str, context: str, threshold: int = 80
) -> str:
    """
    Ground the model's prediction to the context using fuzzy string matching.
    This is useful for causal LMs where the generated answer may not exactly match
    the context span but is close enough.
    """
    result = partial_ratio_alignment(prediction, context)
    if result.score >= threshold:
        # extract the matched span from the context
        grounded = context[result.dest_start : result.dest_end]
        return grounded
    return prediction  # fallback to raw generation
