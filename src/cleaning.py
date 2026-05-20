"""
Cleaning and exploration functions to be used during the data preprocessing
and EDA stages of the project.
"""

# Standard imports
import re
import warnings
from typing import Callable

# Third-party imports
import numpy as np
import pandas as pd
from datasets import Dataset
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer


# Function to harmonize (format) HF datasets
def harmonize_squad_format(dataset: Dataset) -> Dataset:
    """
    Harmonize a HF SQuAD-style dataset enforcing consistent types.

    This function is designed to handle specifically the format encountered when
    loading the translated SQuAD and the MLQA datasets used in this project.
    A thorough data validation is performed to ensure the input dataset has the
    expected structure.

    Expected input format is a HF Dataset with:
    - Features: ['version', 'data']
    - Num_rows: 1
    - 'data' field containing a list of titles with nested paragraphs and QAs

    Final output is a Dataset object with one row per QA sample and 4 features:
        - id (str): Unique question ID
        - context (str): Article passage containing the answer
        - question (str): Question text
        - answers (dict): Keys 'text' (list of str) and 'answer_start' (list of int32)

    Example usage:
        raw_squad_es = load_dataset(
            "json",
            data_files={
                "train": str(squad_es_raw_train_path),
                "validation": str(squad_es_raw_dev_path)
            }
        )
        squad_es_train = harmonize_squad_format(raw_squad_es["train"])
    """

    # Input validation
    if not isinstance(dataset, Dataset):
        raise TypeError("Input must be a Hugging Face Dataset object.")

    if len(dataset) != 1:
        raise ValueError("Input dataset must have exactly one row (num_rows=1).")

    if "data" not in dataset.features:
        raise ValueError("Input dataset must have a 'data' feature.")

    # Extract the 'data' field for validation and processing
    data = dataset["data"][0]

    # Data validation
    if not data:
        raise ValueError("The 'data' feature is empty.")

    if not isinstance(data, list):
        raise ValueError("The 'data' must be a list of titles.")

    for title_idx, title in enumerate(data):
        if not isinstance(title, dict) or "paragraphs" not in title:
            raise ValueError(f"Title {title_idx}: must be dict with 'paragraphs' key")

        for para_idx, paragraph in enumerate(title["paragraphs"]):
            if not isinstance(paragraph, dict):
                raise ValueError(
                    f"Title {title_idx}, Paragraph {para_idx}: must be dict"
                )

            if "context" not in paragraph:
                raise ValueError(
                    f"Title {title_idx}, Paragraph {para_idx}: missing 'context'"
                )

            if "qas" not in paragraph or not isinstance(paragraph["qas"], list):
                raise ValueError(
                    f"Title {title_idx}, Paragraph {para_idx}: invalid 'qas'"
                )

            for qa_idx, qa in enumerate(paragraph["qas"]):
                required_keys = {"question", "answers", "id"}
                if not required_keys.issubset(qa.keys()):
                    raise ValueError(
                        f"Title {title_idx}, QA {qa_idx}: missing required keys"
                    )
                if not isinstance(qa["answers"], list) or not qa["answers"]:
                    raise ValueError(
                        f"Title {title_idx}, QA {qa_idx}: answers must be nonempty list"
                    )

    # Data formatting
    records = []
    for title in data:
        for paragraph in title["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                text = [str(an["text"]) for an in qa["answers"]]
                answer_start = [np.int32(an["answer_start"]) for an in qa["answers"]]
                records.append(
                    {
                        "id": str(qa["id"]),
                        "context": str(context),
                        "question": str(qa["question"]),
                        "answers": {"text": text, "answer_start": answer_start},
                    }
                )

    return Dataset.from_list(records)


# Function to check for duplicates in a column of a dataframe
def check_column_unique(df: pd.DataFrame, col: str) -> None:
    """Check if there are duplicate entries in a dataframe column."""
    n_dup = df.shape[0] - df[col].nunique()
    if n_dup:
        print(f"{n_dup} duplicate entries in '{col}'!")
    else:
        print(f"No duplicates in '{col}'.")


# Function to check entries with duplicate combinations (several columns) in a dataframe
def check_multi_column_unique(df: pd.DataFrame, cols: list[str]) -> None:
    """Check for duplicate combinations of entries from multiple dataframe columns."""
    temp_df = df.copy()
    # answers contains dicts, non-hashable, so convert to str for duplication check
    if "answers" in cols:
        temp_df["answers"] = temp_df["answers"].apply(
            str
        )  # Order guaranteed due to df source being HF Dataset
    dup_count = temp_df.duplicated(subset=cols).sum()
    if dup_count:
        print(f"{dup_count} duplicate row(s) for columns {cols}!")
    else:
        print(f"No duplicate combinations for columns {cols}.")


# Function to check for null or empty values in a dataframe column of type string/object
def check_null_or_empty(df: pd.DataFrame, col: str) -> None:
    """Check for null or empty entries in a dataframe column of type string/object."""
    if not pd.api.types.is_string_dtype(df[col]):
        raise TypeError(
            f"Column '{col}' must be of string/object type for check_null_or_empty."
        )
    mask = df[col].isnull() | (df[col] == "")
    count = mask.sum()
    if count:
        print(f"{count} null or empty entries in '{col}'!")
    else:
        print(f"No null or empty values in '{col}'.")


# Function to check presence of null or negative values in a numeric dataframe column
def check_null_or_negative(df: pd.DataFrame, col: str) -> None:
    """Check for null or negative entries in a dataframe column of numeric type."""
    if not pd.api.types.is_numeric_dtype(df[col]):
        raise TypeError(
            f"Column '{col}' must be of numeric type for check_null_or_negative."
        )
    mask = df[col].isnull() | (df[col] < 0)
    count = mask.sum()
    if count:
        print(f"{count} null or negative entries in '{col}'!")
    else:
        print(f"No null or negative values in '{col}'.")


# Function to detect common "bad characters" in a string
def has_bad_char(s: str) -> bool:
    """
    Define a filter for "universal bad chars"
    It looks for some non-printable ASCII control characters
    and the replacement character '�':
    Between ASCII code \x00 and \x08 (null byte through backspace)
    \x0b (vertical tab)
    \x0c (form feed)
    Between \x0e and \x1f (various ASCII control chars)
    \x7f (ASCII DEL control character)
    \ufffd (Unicode replacement character '�')
    Control chars (Unicode categories: Cc, Cf, Cs, Co, Cn)
    """
    return bool(re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ufffd]", s))


# Function to check for common "bad characters" in specified columns of a dataframe
def check_bad_chars(df: pd.DataFrame, cols: list[str]) -> None:
    """
    Check for bad/universal control characters in specified string/object columns."""
    for col in cols:
        if not pd.api.types.is_string_dtype(df[col]):
            raise TypeError(f"Column '{col}' must be string type.")
        count = df[col].apply(has_bad_char).sum()
        print(f"{count} rows in '{col}' with bad characters.")


# Function to explode the 'answers' column into separate columns
def explode_answers_dict(df: pd.DataFrame) -> pd.DataFrame:
    """
    Explode the 'answers' column (dict) of a dataframe
    into separate 'answer_start' and 'text' columns.
    """
    df = df.copy()
    df["answer_start"] = df["answers"].apply(lambda x: x["answer_start"])
    df["text"] = df["answers"].apply(lambda x: x["text"])
    return df.drop(columns=["answers"])


# Function to explode the 'answer_start' and 'text' lists into separate rows
def explode_answers_lists(df: pd.DataFrame) -> pd.DataFrame:
    """
    Explode answer_start and text lists into separate rows.

    Each row with multiple answers becomes multiple rows, one per answer.
    Validates that both lists have matching lengths or raises an error.
    """
    df = df.copy()

    # Validate lengths before transforming
    mismatch = (df["answer_start"].apply(len) != df["text"].apply(len)).sum()
    if mismatch > 0:
        raise ValueError(f"{mismatch} rows have mismatched answer_start/text lengths")

    # Zip lists to maintain pairs and explode
    df["answer_pair"] = df.apply(
        lambda x: (list(zip(x["answer_start"], x["text"]))), axis=1
    )
    df = df.explode(
        "answer_pair", ignore_index=True
    )  # ignore_index to reset index after explosion

    # Drop rows where answer_pair is NaN (from empty lists after explode)
    # Warn if any rows are dropped due to empty answer lists after explode
    initial_rows = len(df)
    df = df.dropna(subset=["answer_pair"])
    dropped = initial_rows - len(df)
    if dropped > 0:
        warnings.warn(
            f"Dropped {dropped} rows with empty answer lists after explode.",
            UserWarning,
        )

    df["answer_start"] = df["answer_pair"].apply(lambda x: x[0])
    df["text"] = df["answer_pair"].apply(lambda x: x[1])
    return df.drop(columns=["answer_pair"])


# Function to fully explode the nested 'answers' structure into individual answer rows
def explode_answers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fully explode the nested 'answers' structure into individual answer rows.
    Raises an error if the lengths of 'answer_start' and 'text' don't match.
    """
    df = df.copy()
    df = explode_answers_dict(df)
    df = explode_answers_lists(df)  # Includes length validation
    return df


# Function to unexplode 'answer_start' and 'text' columns back into an 'answers' dict
def unexplode_answers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Unexplode the 'answer_start' and 'text' columns back into an 'answers' dict.
    Groups by id+context+question to ensure we reconstruct original entries.
    Note that any additional columns will be lost to guarantee correct grouping.
    """
    df = df.copy()

    grouped = df.groupby(["id", "context", "question"])
    records = []
    for (id, context, question), group in grouped:
        answer_start = group["answer_start"].tolist()
        text = group["text"].tolist()
        records.append(
            {
                "id": id,
                "context": context,
                "question": question,
                "answers": {"answer_start": answer_start, "text": text},
            }
        )
    return pd.DataFrame.from_records(records)


# Function to check that 'text' is found in context at answer_start for a given row
def is_answer_in_context(row: pd.Series) -> bool:
    """
    Check that 'text' is actually found in context
    at the specified answer_start position (use after exploding).
    """
    context = row["context"]
    text = row["text"]
    answer_start = row["answer_start"]

    # Check if answer is within the bounds of the context
    if answer_start < 0 or answer_start + len(text) > len(context):
        return False

    # Extract the substring from context at the specified position
    extracted_text = context[answer_start : answer_start + len(text)]

    # Compare the extracted text with the answer text
    return extracted_text == text


# Function to check that 'text' is found in context at answer_start for each row
def check_answer_in_context(df: pd.DataFrame) -> None:
    """
    Check that 'text' is found in context at answer_start for each row of the dataframe.
    Use after exploding.
    """
    count = (~df.apply(is_answer_in_context, axis=1)).sum()
    if count:
        print(f"{count} rows where 'text' isn't found at answer_start in context.")
    else:
        print("All answers properly found in their context.")


# Factory function to create a function that checks if tokenized string has
# no tokens present for a given tokenizer
def factory_no_tokens(tokenizer: AutoTokenizer) -> Callable[[str], bool]:
    """
    Create a function to check if a string has no tokens when tokenized.
    """

    def no_tokens(s: str) -> bool:
        # Type check
        if not isinstance(s, str):
            raise TypeError(f"Expected string, got {type(s).__name__}")

        # Strip whitespace to avoid issues with tokenizers behaving differntly with
        # strings that are only whitespace.
        # This may also help catch empty entries that are not empty strings.
        tokens = tokenizer.tokenize(s.strip())
        return len(tokens) == 0

    return no_tokens


# Function to check for entries with no tokens in specified columns for given tokenizers
def check_no_tokens(
    df: pd.DataFrame, cols: list[str], tokenizers: dict[str, AutoTokenizer]
) -> None:
    """
    Check for rows in specified dataframe columns that have no tokens
    when tokenized by given tokenizers.
    """
    for col in cols:
        for name, tokenizer in tokenizers.items():
            fn = factory_no_tokens(tokenizer)
            count = df[col].apply(fn).sum()
            print(f"{count} '{col}' rows flagged (no tokens) for tokenizer {name}")


# Factory function to create a function that checks UNK tokens in a tokenized string
def factory_unk_tokens(tokenizer: AutoTokenizer) -> Callable[[str], bool]:
    """
    Create a function to check if a string has UNK tokens
    when tokenized by the given tokenizer.
    """

    def unk_tokens(s: str) -> bool:
        # Type check
        if not isinstance(s, str):
            raise TypeError(f"Expected string, got {type(s).__name__}")

        # Strip whitespace to avoid issues with tokenizers behaving differntly with
        # strings that are only whitespace.
        # This may also help catch empty entries that are not empty strings.
        tokens = tokenizer.tokenize(s.strip())
        unk_count = tokens.count(tokenizer.unk_token)
        return unk_count > 0

    return unk_tokens


# Function to check for entries with UNK tokens in specified columns
def check_unk_tokens(
    df: pd.DataFrame, cols: list[str], tokenizers: dict[str, AutoTokenizer]
) -> None:
    """
    Check for rows in specified dataframe columns that have UNK tokens
    when tokenized by given tokenizers.
    """
    for col in cols:
        for name, tokenizer in tokenizers.items():
            fn = factory_unk_tokens(tokenizer)
            count = df[col].apply(fn).sum()
            print(f"{count} '{col}' rows flagged (UNK tokens) for tokenizer {name}")


# Function to check for entries where context+question would be truncated by mBERT
def check_mbert_input_truncation(
    df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    max_seq_length: int,
) -> None:
    """
    Check for entries that would be truncated by the tokenizer of mBERT.
    This function is designed for extractive QA.
    """
    # Max seq length for mBERT is 512 tokens.
    tkn_seq_len_limit = getattr(tokenizer, "model_max_length", max_seq_length)
    max_length = min(tkn_seq_len_limit, max_seq_length)

    # Remove leading/trailing whitespace from questions
    questions = [q.strip() for q in df["question"]]
    # context will be left as is, to not cause mismatch with answer_start positions

    # Tokenize all inputs
    tokenized_input = tokenizer(
        questions,
        df["context"].tolist(),
        truncation=False,  # need false to check truncation
    )

    # Get length of each tokenized input
    tokenized_len = [len(ids) for ids in tokenized_input["input_ids"]]

    # Count how many entries exceed the max length, and the length of the longest entry
    max_found = max(tokenized_len) if tokenized_len else 0
    count = sum(1 for length in tokenized_len if length > max_length)

    # Print results
    print(f"{count} rows would be truncated.")
    print(f"Sequence limit set at {max_length} tokens.")
    print(f"Longest tokenized entry found is {max_found} tokens long.")


# Function to check for entries where context+question would be truncated by Gemma 2
def check_gemma2_input_truncation(
    df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    max_seq_length: int,
) -> None:
    """
    Check for entries that would be truncated by the tokenizer of Gemma 2.
    This function is designed for generative QA, with appropriate prompt format.
    Uses prompt format:
        Pregunta: {question}
        Contexto: {context}
        Respuesta: {answer}
    """
    # Cap max length at tokenizer limit.
    # Gemma 2 can support up to 8192 tokens with a 4096 token sliding window.
    # The window size can be confirmed by checking model.config.sliding_window).
    # We will use 70% of the sliding window as a hard cap to be safe (~2867 tokens).
    max_length = min(int(4096 * 0.7), max_seq_length)

    # Remove leading/trailing whitespace from questions, contexts, and answers
    # Only first answer is used during training which will be used for truncation check
    questions = [q.strip() for q in df["question"]]
    contexts = [c.strip() for c in df["context"]]
    answers = [a["text"][0].strip() for a in df["answers"]]

    # Format input as it would be during training to get accurate tokenization length
    formatted_input = [
        f"Pregunta: {q}\nContexto: {c}\nRespuesta: {a}{tokenizer.eos_token}"
        for q, c, a in zip(questions, contexts, answers)
    ]

    # Tokenize all inputs
    tokenized_input = tokenizer(
        formatted_input,
        truncation=False,  # need false to check truncation
    )

    # Get length of each tokenized input
    tokenized_len = [len(ids) for ids in tokenized_input["input_ids"]]

    # Count how many entries exceed the max length, and the length of the longest entry
    max_found = max(tokenized_len) if tokenized_len else 0
    count = sum(1 for length in tokenized_len if length > max_length)

    # Print results
    print(f"{count} rows would be truncated.")
    print(f"Sequence limit set at {max_length} tokens.")
    print(f"Longest tokenized entry found is {max_found} tokens long.")


# Function to filter out rows where the tokenized length (generative) exceeds the limit
def filter_by_gemma2_tokenized_length(
    df: pd.DataFrame, tokenizer: AutoTokenizer, max_length: int = 512
) -> pd.DataFrame:
    """
    Filter out rows where the tokenized length of an entry exceeds the limit when
    encoding context and question together in a formatted prompt for use with a
    generative model.

    Uses prompt format:
        Pregunta: {question}
        Contexto: {context}
        Respuesta: {answer}
    """
    if max_length <= 0 or not isinstance(max_length, int):
        raise ValueError("max_length must be a positive integer.")

    # Gemma 2 can support up to 8192 tokens with a 4096 token sliding window.
    # The window size can be confirmed by checking model.config.sliding_window).
    # We will use 70% of the sliding window as a hard cap to be safe (~2867 tokens).
    hard_cap = int(4096 * 0.7)
    if max_length > hard_cap:
        raise ValueError(
            f"max_length should be <= 70% of the sliding window size ({hard_cap})."
        )

    df = df.copy()
    print(f"Initial number of entries: {len(df)}")

    # Format inputs
    formatted_inputs = [
        f"Pregunta: {str(row['question']).strip()}\n"
        f"Contexto: {str(row['context']).strip()}\n"
        f"Respuesta: {str(row['answers']['text'][0]).strip()}{tokenizer.eos_token}"
        for _, row in df.iterrows()
    ]

    # Tokenize inputs without truncation to get true tokenized lengths
    tokenized_input = tokenizer(formatted_inputs, truncation=False)

    # Get lengths
    token_lengths = [len(ids) for ids in tokenized_input["input_ids"]]

    # Filter DataFrame
    mask = [length <= max_length for length in token_lengths]
    df = df[mask]

    # Print results
    print(
        f"Number of entries after pruning by tokenized length "
        f"<= {max_length}: {len(df)}"
    )
    return df


def embed_sim_matrix(
    model: SentenceTransformer, list1: list[str], list2: list[str]
) -> np.ndarray:
    """
    Compute cosine similarity matrix between two lists of texts.

    Args:
        model: SentenceTransformer model for encoding
        list1: First set of texts
        list2: Second set of texts

    Returns:
        Similarity matrix of shape (len(list1), len(list2))
    """
    # Encode both lists into embeddings
    embeddings1 = model.encode(list1, convert_to_tensor=True)
    embeddings2 = model.encode(list2, convert_to_tensor=True)

    # Compute cosine similarity matrix
    sim_matrix_tensor = util.cos_sim(embeddings1, embeddings2)

    # Convert to numpy array and return
    sim_matrix = sim_matrix_tensor.cpu().numpy()
    return sim_matrix
