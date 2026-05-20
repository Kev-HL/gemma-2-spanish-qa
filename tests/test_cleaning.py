"""Unit tests for some of the cleaning.py functions, located in src/cleaning.py"""

# Standard imports
from unittest.mock import Mock

# Third party imports
import numpy as np
import pandas as pd
import pytest

# Local imports
from src.cleaning import (
    explode_answers_dict,
    explode_answers_lists,
    explode_answers,
    unexplode_answers,
    is_answer_in_context,
    factory_no_tokens,
    factory_unk_tokens,
    check_mbert_input_truncation,
    check_gemma2_input_truncation,
    filter_by_gemma2_tokenized_length,
    embed_sim_matrix,
)


class TestExplodeAnswersDict:
    """Tests for explode_answers_dict function"""

    def test_basic_case(self):
        """Test normal operation with expected input format"""
        # Create sample dataframe with expected format
        df = pd.DataFrame(
            {
                "id": ["Q1", "Q2"],
                "context": ["ctx1", "ctx2"],
                "question": ["q1?", "q2?"],
                "answers": [
                    {"answer_start": [0, 10], "text": ["foo", "bar"]},
                    {"answer_start": [5], "text": ["baz"]},
                ],
            }
        )

        # Act
        result = explode_answers_dict(df)

        # Assert
        assert "answer_start" in result.columns
        assert "text" in result.columns
        assert "answers" not in result.columns
        assert len(result) == len(df)  # Same number of rows
        assert result.iloc[0]["answer_start"] == [0, 10]
        assert result.iloc[0]["text"] == ["foo", "bar"]
        assert result.iloc[1]["answer_start"] == [5]
        assert result.iloc[1]["text"] == ["baz"]


class TestExplodeAnswersLists:
    """Tests for explode_answers_lists function"""

    def test_basic_case(self):
        """Test normal operation with expected input format"""
        # Create sample dataframe with expected format
        df = pd.DataFrame(
            {
                "id": ["Q1", "Q2"],
                "context": ["ctx1", "ctx2"],
                "question": ["q1?", "q2?"],
                "answer_start": [[0, 10], [5]],  # Already lists, not dicts
                "text": [["foo", "bar"], ["baz"]],
            }
        )

        result = explode_answers_lists(df)

        assert len(result) == 3  # 2 from Q1 and 1 from Q2
        assert result.iloc[0]["id"] == "Q1"
        assert result.iloc[0]["answer_start"] == 0
        assert result.iloc[0]["text"] == "foo"
        assert result.iloc[1]["id"] == "Q1"
        assert result.iloc[1]["answer_start"] == 10
        assert result.iloc[1]["text"] == "bar"
        assert result.iloc[2]["id"] == "Q2"
        assert result.iloc[2]["answer_start"] == 5
        assert result.iloc[2]["text"] == "baz"

    def test_empty_answers_dropped_with_warning(self):
        """Test that rows with empty answer lists are dropped with warning"""
        df = pd.DataFrame(
            {
                "id": ["Q1", "Q2", "Q3"],
                "context": ["ctx1", "ctx2", "ctx3"],
                "question": ["q1?", "q2?", "q3?"],
                "answer_start": [[], [0, 10], [5]],  # Q1 has empty
                "text": [[], ["foo", "bar"], ["baz"]],  # Q1 has empty
            }
        )

        with pytest.warns(
            UserWarning, match="Dropped 1 rows with empty answer lists after explode."
        ):
            result = explode_answers_lists(df)

        # Q1 should be dropped, Q2 and Q3 should remain (exploded)
        assert len(result) == 3  # Q2 becomes 2 rows, Q3 becomes 1 row
        assert result.iloc[0]["answer_start"] == 0
        assert result.iloc[1]["answer_start"] == 10
        assert result.iloc[2]["answer_start"] == 5

    def test_mismatch_lengths_error(self):
        """Test that mismatched list lengths raise ValueError"""
        df = pd.DataFrame(
            {"answer_start": [[10, 20]], "text": [["foo"]]}  # Length mismatch
        )

        with pytest.raises(ValueError):
            explode_answers_lists(df)


class TestUnexplodeAnswers:
    """Tests for unexplode_answers function"""

    def test_basic_case(self):
        """Test normal operation with expected input format"""
        df = pd.DataFrame(
            {
                "id": ["Q1", "Q1", "Q2"],
                "context": ["ctx1", "ctx1", "ctx2"],
                "question": ["q1?", "q1?", "q2?"],
                "answer_start": [0, 10, 5],
                "text": ["foo", "bar", "baz"],
            }
        )

        result = unexplode_answers(df)

        assert len(result) == 2
        assert result.iloc[0]["id"] == "Q1"
        assert result.iloc[0]["answers"]["answer_start"] == [0, 10]
        assert result.iloc[0]["answers"]["text"] == ["foo", "bar"]
        assert result.iloc[1]["id"] == "Q2"
        assert result.iloc[1]["answers"]["answer_start"] == [5]
        assert result.iloc[1]["answers"]["text"] == ["baz"]
        assert list(result.index) == [0, 1]  # groupby method should reset index

    def test_extra_columns_dropped(self):
        """Test that extra columns are dropped and not affect grouping"""
        df = pd.DataFrame(
            {
                "id": ["Q1", "Q1"],
                "context": ["ctx1", "ctx1"],
                "question": ["q1?", "q1?"],
                "answer_start": [0, 10],
                "text": ["foo", "bar"],
                "extra_col": ["x", "y"],  # Extra column that should be ignored
            }
        )

        result = unexplode_answers(df)

        assert len(result) == 1
        assert "extra_col" not in result.columns

    def test_explode_unexplode(self):
        """Test that explode then unexplode returns to original structure and content"""
        original = pd.DataFrame(
            {
                "id": ["Q1", "Q2"],
                "context": ["ctx1", "ctx2"],
                "question": ["q1?", "q2?"],
                "answers": [
                    {"answer_start": [0, 10], "text": ["foo", "bar"]},
                    {"answer_start": [5], "text": ["baz"]},
                ],
            }
        )

        # Explode then Unexplode
        exploded = explode_answers(original)
        result = unexplode_answers(exploded)

        # Should match original structure
        assert len(result) == len(original)
        assert result.iloc[0]["id"] == original.iloc[0]["id"]
        assert result.iloc[0]["answers"]["answer_start"] == [0, 10]
        assert result.iloc[0]["answers"]["text"] == ["foo", "bar"]
        assert result.iloc[1]["answers"]["answer_start"] == [5]
        assert result.iloc[1]["answers"]["text"] == ["baz"]


class TestIsAnswerInContext:
    """Tests for is_answer_in_context function"""

    def test_answer_at_start(self):
        """Test answer at start of context"""
        row = pd.Series({"context": "hello world", "text": "hello", "answer_start": 0})
        assert is_answer_in_context(row) is True

    def test_answer_in_middle(self):
        """Test answer in middle of context"""
        # Note that end of context would be the same case (literal substring match).
        row = pd.Series(
            {"context": "hello world again", "text": "world", "answer_start": 6}
        )
        assert is_answer_in_context(row) is True

    def test_answer_is_full_context(self):
        """Test answer is the entire context"""
        row = pd.Series(
            {"context": "hello world", "text": "hello world", "answer_start": 0}
        )
        assert is_answer_in_context(row) is True

    def test_case_sensitivity(self):
        """Test that matching is case-sensitive"""
        row = pd.Series({"context": "Hello world", "text": "hello", "answer_start": 0})
        assert is_answer_in_context(row) is False

    def test_answer_out_of_bounds(self):
        """Test answer extending beyond context"""
        row = pd.Series({"context": "hello", "text": "hello world", "answer_start": 0})
        assert is_answer_in_context(row) is False

    def test_negative_start(self):
        """Test negative answer_start"""
        row = pd.Series({"context": "hello", "text": "hello", "answer_start": -1})
        assert is_answer_in_context(row) is False


class TestFactoryNoTokens:
    """Tests for factory_no_tokens function using a mock tokenizer"""

    def test_tokenizer_returns_empty_list(self):
        """Test that the check returns True if tokenizer returns an empty list."""
        # Create a mock tokenizer that always returns an empty list
        mock_tokenizer = Mock()
        mock_tokenizer.tokenize.return_value = []

        no_tokens_fn = factory_no_tokens(mock_tokenizer)
        result = no_tokens_fn("some input text")

        assert result  # True

    def test_tokenizer_returns_non_empty_list(self):
        """Test that the check returns False if tokenizer returns a non-empty list."""
        # Create a mock tokenizer that always returns a non-empty list
        mock_tokenizer = Mock()
        mock_tokenizer.tokenize.return_value = ["token"]

        no_tokens_fn = factory_no_tokens(mock_tokenizer)
        result = no_tokens_fn("some input text")

        assert not result  # False

    def test_whitespace_stripped_before_tokenizing(self):
        """Test that whitespace is stripped before tokenizing"""
        mock_tokenizer = Mock()
        mock_tokenizer.tokenize.return_value = []

        no_tokens_fn = factory_no_tokens(mock_tokenizer)
        result = no_tokens_fn("   ")

        # Verify tokenizer was called with stripped string (empty)
        mock_tokenizer.tokenize.assert_called_with("")
        assert result  # True


class TestFactoryUnkTokens:
    """Tests for factory_unk_tokens function using a mock tokenizer"""

    def test_tokenizer_returns_empty_list(self):
        """Test that the check returns True if tokenizer returns an empty list."""
        # Create a mock tokenizer that always returns an empty list
        mock_tokenizer = Mock()
        mock_tokenizer.tokenize.return_value = []

        unk_tokens_fn = factory_unk_tokens(mock_tokenizer)
        result = unk_tokens_fn("some input text")

        assert not result  # False

    def test_tokenizer_returns_only_valid_tokens(self):
        """Test that the check returns False if tokenizer returns only valid tokens."""
        # Create a mock tokenizer that always returns a non-empty list
        mock_tokenizer = Mock()
        mock_tokenizer.tokenize.return_value = ["token"]

        unk_tokens_fn = factory_unk_tokens(mock_tokenizer)
        result = unk_tokens_fn("some input text")

        assert not result  # False

    def test_tokenizer_returns_unk_tokens(self):
        """Test that the check returns True if tokenizer returns only unknown tokens."""
        # Create a mock tokenizer that always returns a list with a UNK token.
        mock_tokenizer = Mock()
        mock_tokenizer.tokenize.return_value = ["<unk>"]
        mock_tokenizer.unk_token = "<unk>"

        unk_tokens_fn = factory_unk_tokens(mock_tokenizer)
        result = unk_tokens_fn("some input text")

        assert result  # True

    def test_whitespace_stripped_before_tokenizing(self):
        """Test that whitespace is stripped before tokenizing"""
        mock_tokenizer = Mock()
        mock_tokenizer.tokenize.return_value = []

        unk_tokens_fn = factory_unk_tokens(mock_tokenizer)
        result = unk_tokens_fn("   ")

        # Verify tokenizer was called with stripped string (empty)
        mock_tokenizer.tokenize.assert_called_with("")
        assert not result  # False


class TestCheckMbertInputTruncation:
    """Tests for check_mbert_input_truncation function using a mock tokenizer"""

    def test_basic_case(self, capsys):
        """Test that the function correctly identifies rows that would be truncated."""
        mock_tokenizer = Mock()
        mock_tokenizer.model_max_length = 5
        mock_tokenizer.return_value = {
            "input_ids": [
                [1, 2, 3, 4],
                [1, 2, 3, 4, 5, 6],
                [1, 2, 3, 4, 5],  # edge case
                [1, 2, 3, 4, 5, 6, 7, 8],
                [1, 2, 3, 4, 5, 6],
            ],
        }
        max_seq_length = 5
        df = pd.DataFrame(
            {
                "context": ["ctx1", "ctx2", "ctx3", "ctx4", "ctx5"],
                "question": ["q1?", "q2?", "q3?", "q4?", "q5?"],
            }
        )

        check_mbert_input_truncation(df, mock_tokenizer, max_seq_length)

        captured = capsys.readouterr()
        assert "3 rows would be truncated" in captured.out
        assert "Sequence limit set at 5 tokens" in captured.out
        assert "Longest tokenized entry found is 8 tokens long" in captured.out

    def test_correct_tokenizer_input(self):
        """Test that the tokenizer receives the expected input format."""
        mock_tokenizer = Mock()
        mock_tokenizer.model_max_length = 5
        mock_tokenizer.return_value = {"input_ids": [[1, 2, 3, 4]]}
        max_seq_length = 5
        df = pd.DataFrame(
            {
                "context": [" ctx "],
                "question": [" q? "],
            }
        )

        check_mbert_input_truncation(df, mock_tokenizer, max_seq_length)
        mock_tokenizer.assert_called_with(["q?"], [" ctx "], truncation=False)


class TestCheckGemma2InputTruncation:
    """Tests for check_gemma2_input_truncation function using a mock tokenizer"""

    def test_basic_case(self, capsys):
        """Test that the function correctly identifies rows that would be truncated."""
        mock_tokenizer = Mock()
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.return_value = {
            "input_ids": [
                [1, 2, 3, 4],
                [1, 2, 3, 4, 5, 6],
                [1, 2, 3, 4, 5],  # edge case
                [1, 2, 3, 4, 5, 6, 7, 8],
                [1, 2, 3, 4, 5, 6],
            ],
        }
        max_seq_length = 5
        df = pd.DataFrame(
            {
                "context": ["ctx1", "ctx2", "ctx3", "ctx4", "ctx5"],
                "question": ["q1?", "q2?", "q3?", "q4?", "q5?"],
                "answers": [
                    {"text": ["a1"]},
                    {"text": ["a2"]},
                    {"text": ["a3"]},
                    {"text": ["a4"]},
                    {"text": ["a5"]},
                ],
            }
        )

        check_gemma2_input_truncation(df, mock_tokenizer, max_seq_length)

        captured = capsys.readouterr()
        assert "3 rows would be truncated" in captured.out
        assert "Sequence limit set at 5 tokens" in captured.out
        assert "Longest tokenized entry found is 8 tokens long" in captured.out

    def test_empty_entry(self, capsys):
        """Test that the function correctly handles an empty entry."""
        mock_tokenizer = Mock()
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.return_value = {"input_ids": [[]]}
        max_seq_length = 5
        df = pd.DataFrame(
            {
                "context": [],
                "question": [],
                "answers": [],
            }
        )

        check_gemma2_input_truncation(df, mock_tokenizer, max_seq_length)

        captured = capsys.readouterr()
        assert "0 rows would be truncated" in captured.out
        assert "Sequence limit set at 5 tokens" in captured.out
        assert "Longest tokenized entry found is 0 tokens long" in captured.out

    def test_correct_tokenizer_input(self):
        """Test that the tokenizer receives the expected input format."""
        mock_tokenizer = Mock()
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.return_value = {"input_ids": [[1, 2, 3, 4]]}
        max_seq_length = 5
        df = pd.DataFrame(
            {
                "context": [" ctx "],
                "question": [" q? "],
                "answers": [{"text": [" a "]}],
            }
        )

        check_gemma2_input_truncation(df, mock_tokenizer, max_seq_length)
        mock_tokenizer.assert_called_with(
            ["Pregunta: q?\nContexto: ctx\nRespuesta: a<eos>"], truncation=False
        )


class TestFilterByGemma2TokenizedLength:
    """Tests for filter_by_gemma2_tokenized_length function using a mock tokenizer"""

    def test_basic_case(self):
        """Test that the function correctly filters out rows that would be truncated."""
        mock_tokenizer = Mock()
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.return_value = {
            "input_ids": [
                [1, 2, 3, 4],
                [1, 2, 3, 4, 5, 6],
                [1, 2, 3, 4, 5],  # edge case
                [1, 2, 3, 4, 5, 6, 7, 8],
                [1, 2, 3, 4, 5, 6],
            ],
        }
        max_length = 5
        df = pd.DataFrame(
            {
                "context": ["ctx1", "ctx2", "ctx3", "ctx4", "ctx5"],
                "question": ["q1?", "q2?", "q3?", "q4?", "q5?"],
                "answers": [
                    {"text": ["a1"]},
                    {"text": ["a2"]},
                    {"text": ["a3"]},
                    {"text": ["a4"]},
                    {"text": ["a5"]},
                ],
            }
        )

        filter_df = filter_by_gemma2_tokenized_length(df, mock_tokenizer, max_length)

        assert len(filter_df) == 2
        assert filter_df.iloc[0]["question"] == "q1?"
        assert filter_df.iloc[1]["question"] == "q3?"

    def test_correct_tokenizer_input(self):
        """Test that the tokenizer receives the expected input format."""
        mock_tokenizer = Mock()
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.return_value = {"input_ids": [[1, 2, 3, 4]]}
        max_length = 5
        df = pd.DataFrame(
            {
                "context": [" ctx "],
                "question": [" q? "],
                "answers": [{"text": [" a "]}],
            }
        )

        filter_by_gemma2_tokenized_length(df, mock_tokenizer, max_length)
        mock_tokenizer.assert_called_with(
            ["Pregunta: q?\nContexto: ctx\nRespuesta: a<eos>"], truncation=False
        )


class TestEmbedSimMatrix:
    """Tests for embed_sim_matrix function using a mock embedding model"""

    def test_basic_case(self):
        """Test that the function returns a similarity matrix of the correct shape."""
        mock_model = Mock()
        # Mock encode to return fixed-size embeddings
        # Returns a 5-dimensional embedding of all 1.0 for each text input in the list
        mock_model.encode.side_effect = lambda texts, convert_to_tensor: [
            [1.0] * 5 for _ in texts
        ]

        list1 = ["text1", "text2"]
        list2 = ["text3", "text4", "text5"]

        sim_matrix = embed_sim_matrix(mock_model, list1, list2)

        assert sim_matrix.shape == (len(list1), len(list2))
        assert np.allclose(sim_matrix, 1.0)  # All 1.0 due to Mock setup
