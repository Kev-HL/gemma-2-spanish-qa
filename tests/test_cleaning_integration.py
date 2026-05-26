"""
Unit tests for some of the cleaning.py functions, located in src/cleaning.py
These are integration tests that use real tokenizers (external resource).
This is intended to run locally and not in CI.
"""

# Third-party imports
import pandas as pd
import pytest
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

# Local imports
from cleaning import (
    factory_no_tokens,
    factory_unk_tokens,
    check_mbert_input_truncation,
    check_gemma2_input_truncation,
    filter_by_gemma2_tokenized_length,
    embed_sim_matrix,
)
from src.credentials import load_hf_credentials

# Load Hugging Face access token from .env
# (Assumes .env file with HF_TOKEN=hf_... exists on cwd)
hf_token = load_hf_credentials()


# Fixtures for tokenizers
@pytest.fixture(scope="module")
def tkn_mbert():
    """Load mBERT tokenizer once per module"""
    return AutoTokenizer.from_pretrained(
        "bert-base-multilingual-cased",
        token=hf_token,  # Public access, but passing token to increase rate limits
    )


@pytest.fixture(scope="module")
def tkn_gemma2():
    """Load Gemma2 tokenizer once per module"""
    return AutoTokenizer.from_pretrained(
        "google/gemma-2-2b", token=hf_token  # Gated access, token required
    )


@pytest.fixture(params=["tkn_mbert", "tkn_gemma2"])
def tokenizer(request):
    """Provides both tokenizers parametrized."""
    return request.getfixturevalue(request.param)


@pytest.mark.integration
class TestFactoryNoTokens:
    """Tests for factory_no_tokens function using real tokenizers"""

    def test_valid_string(self, tokenizer):
        """Test that the check returns False with some random valid string."""
        no_tokens_fn = factory_no_tokens(tokenizer)
        result = no_tokens_fn("some input text")
        assert not result  # False

    def test_empty_string(self, tokenizer):
        """Test that the check returns True with an empty string."""
        no_tokens_fn = factory_no_tokens(tokenizer)
        result = no_tokens_fn("")
        assert result  # True

    def test_whitespace(self, tokenizer):
        """Test that the check returns True with a string of only whitespace."""
        # Function strips the string, so tokenizer receives empty string
        no_tokens_fn = factory_no_tokens(tokenizer)
        result = no_tokens_fn("    ")
        assert result  # True


@pytest.mark.integration
class TestFactoryUnkTokens:
    """Tests for factory_unk_tokens function using real tokenizers"""

    def test_valid_string(self, tokenizer):
        """Test that the check returns False with some random valid string."""
        unk_tokens_fn = factory_unk_tokens(tokenizer)
        result = unk_tokens_fn("some input text")
        assert not result  # False

    def test_string_with_unk_token(self, tokenizer):
        """
        Test that the check returns True with a string with a UNK token.

        Using tokenizer.unk_token directly is a valid approach because both
        mBERT and Gemma2 tokenize it as actual UNK tokens in the output.
        It is NOT the same as writing just <unk> or [UNK] on the string.
        """

        unk_tokens_fn = factory_unk_tokens(tokenizer)
        result = unk_tokens_fn(f"This is an {tokenizer.unk_token} token")
        assert result  # True

    def test_empty_string(self, tokenizer):
        """Test that the check returns False with an empty string."""
        unk_tokens_fn = factory_unk_tokens(tokenizer)
        result = unk_tokens_fn("")
        assert not result  # False

    def test_whitespace(self, tokenizer):
        """Test that the check returns False with a string of only whitespace."""
        # Function strips the string, so tokenizer receives empty string
        unk_tokens_fn = factory_unk_tokens(tokenizer)
        result = unk_tokens_fn("    ")
        assert not result  # False


@pytest.mark.integration
class TestCheckMbertInputTruncation:
    """Tests for check_mbert_input_truncation function using a real tokenizer"""

    def test_basic_case(self, capsys, tkn_mbert):
        """
        Test that the function correctly identifies rows that would be truncated.

        Different tokenizers use different special tokens and rules, in our case:
        - mBERT: [CLS] question [SEP] context [SEP]
        meaning sequence length will be 3 special tokens + tokens of context & question
        """
        df = pd.DataFrame(
            {
                "context": [
                    "one two three four",
                    "one two three four five six",
                    "one two three four five",  # edge case
                    "one two three four five six seven eight",
                    "one two three four five six",
                ],
                "question": [
                    "question",
                    "question",
                    "question",
                    "question",
                    "question",
                ],
            }
        )
        limit_sample = tkn_mbert(
            df["question"][2],
            df["context"][2],
            truncation=False,
        )
        max_seq_length = len(limit_sample["input_ids"])

        check_mbert_input_truncation(df, tkn_mbert, max_seq_length)
        captured = capsys.readouterr()
        assert "3 rows would be truncated" in captured.out
        assert f"Sequence limit set at {max_seq_length} tokens" in captured.out
        assert "Longest tokenized entry found is 12 tokens long" in captured.out


@pytest.mark.integration
class TestCheckGemma2InputTruncation:
    """Tests for check_gemma2_input_truncation function using a real tokenizer"""

    def test_basic_case(self, capsys, tkn_gemma2):
        """
        Test that the function correctly identifies rows that would be truncated.

        Different tokenizers use different special tokens and rules, in our case:
        - Gemma2: <bos> input
        with input being the tokenized:
        "Pregunta: {questions}\nContexto: {contexts}\nRespuesta: {answers}" + <eos>

        So sequence length will be 2 special tokens (<bos> and <eos>) + tokens the
        tokenized formatted prompt.

        This should approximate to the sum of the tokenized question + context + answer,
        plus 10 tokens (<bos>, "Pregunta", ":", "\n", "Contexto", ":", "\n",
        "Respuesta", ":", <eos>)
        """
        df = pd.DataFrame(
            {
                "context": [
                    "one two three four",
                    "one two three four five six",
                    "one two three four five",  # edge case
                    "one two three four five six seven eight",
                    "one two three four five six",
                ],
                "question": [
                    "question",
                    "question",
                    "question",
                    "question",
                    "question",
                ],
                "answers": [
                    {"text": ["answer"]},
                    {"text": ["answer"]},
                    {"text": ["answer"]},
                    {"text": ["answer"]},
                    {"text": ["answer"]},
                ],
            }
        )

        formatted_input = (
            f"Pregunta: {str(df['question'][2]).strip()}\n"
            f"Contexto: {str(df['context'][2]).strip()}\n"
            f"Respuesta: {str(df['answers'][2]['text'][0]).strip()}"
            f"{tkn_gemma2.eos_token}"
        )
        limit_sample = tkn_gemma2(formatted_input, truncation=False)
        max_seq_length = len(limit_sample["input_ids"])  # Should be 17

        check_gemma2_input_truncation(df, tkn_gemma2, max_seq_length)
        captured = capsys.readouterr()
        assert "3 rows would be truncated" in captured.out
        assert f"Sequence limit set at {max_seq_length} tokens" in captured.out
        assert "Longest tokenized entry found is 20 tokens long" in captured.out


@pytest.mark.integration
class TestFilterByGemma2TokenizedLength:
    """Tests for filter_by_gemma2_tokenized_length function using a real tokenizer"""

    def test_basic_case(self, tkn_gemma2):
        """
        Test filtering with real Gemma2 tokenizer.
        Verifies that rows over the tokenized length limit are removed, and rows at or
        under the limit are kept.
        Uses edge case (row 2) as the limit threshold.
        """
        df = pd.DataFrame(
            {
                "context": [
                    "one two three four",
                    "one two three four five six",
                    "one two three four five",  # edge case
                    "one two three four five six seven eight",
                    "one two three four five six",
                ],
                "question": ["first", "second", "third", "fourth", "fifth"],
                "answers": [
                    {"text": ["answer"]},
                    {"text": ["answer"]},
                    {"text": ["answer"]},
                    {"text": ["answer"]},
                    {"text": ["answer"]},
                ],
            }
        )

        formatted_input = (
            f"Pregunta: {str(df['question'][2]).strip()}\n"
            f"Contexto: {str(df['context'][2]).strip()}\n"
            f"Respuesta: {str(df['answers'][2]['text'][0]).strip()}"
            f"{tkn_gemma2.eos_token}"
        )
        limit_sample = tkn_gemma2(formatted_input, truncation=False)
        max_length = len(limit_sample["input_ids"])  # Should be 17

        filtered_df = filter_by_gemma2_tokenized_length(df, tkn_gemma2, max_length)

        assert len(filtered_df) == 2
        assert filtered_df.iloc[0]["question"] == "first"
        assert filtered_df.iloc[1]["question"] == "third"


@pytest.mark.integration
class TestEmbedSimMatrix:
    """Tests for embed_sim_matrix function using a real SentenceTransformer"""

    @pytest.fixture(scope="class")
    def embed_model(self):
        """Load SentenceTransformer model once per class"""
        return SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

    def test_basic_case(self, embed_model):
        """Test with real model: identical sentences should have high similarity."""
        sentences_a = ["This is a test.", "This is one test."]
        sentences_b = ["This is a test.", "hello world"]

        sim_matrix = embed_sim_matrix(embed_model, sentences_a, sentences_b)

        assert sim_matrix.shape == (2, 2)
        assert ((sim_matrix >= 0.0) & (sim_matrix <= 1.0)).all()
        assert sim_matrix[0, 0] == pytest.approx(1.0)  # Identical
        assert sim_matrix[1, 0] > 0.8  # Very similar
        assert sim_matrix[0, 1] < 0.5  # Different
