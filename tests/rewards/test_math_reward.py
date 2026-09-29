import pytest

from rllm.rewards import RewardConfig, RewardOutput, RewardType
from rllm.rewards.math_reward import RewardMathFn


class TestMathReward:
    """Test class for math reward functionality."""

    def test_correct_answer_simple(self):
        """Test math reward with simple correct answer."""
        model_response = "<think>2 + 2 = 4</think>\nThe answer is \\boxed{4}."

        reward = RewardMathFn(RewardConfig())
        task_info = {"problem": "What is 2 + 2?", "problem_type": RewardType.MATH, "data_source": "test", "ground_truth": "4"}
        output = reward(task_info, model_response)

        assert output.is_correct is True
        assert output.reward == 1.0

    def test_correct_answer_multiple_ground_truths(self):
        """Test math reward with multiple possible correct answers."""
        model_response = "<think>The answer could be written as a fraction or decimal</think>\nThe answer is \\boxed{0.5}."

        reward = RewardMathFn(RewardConfig())
        task_info = {"problem": "What is 1/2?", "problem_type": RewardType.MATH, "data_source": "test", "ground_truth": ["0.5", "1/2", "one half"]}
        output = reward(task_info, model_response)

        assert output.is_correct is True
        assert output.reward == 1.0

    def test_incorrect_answer(self):
        """Test math reward with incorrect answer."""
        model_response = "<think>2 + 2 = 5</think>\nThe answer is \\boxed{5}."

        reward = RewardMathFn(RewardConfig())
        task_info = {"problem": "What is 2 + 2?", "problem_type": RewardType.MATH, "data_source": "test", "ground_truth": "4"}
        output = reward(task_info, model_response)

        assert not output.is_correct
        assert output.reward == 0.0

    def test_format_error_no_delimiter(self):
        """Test math reward with format error - no thought delimiter."""
        model_response = "The answer is 4."

        reward = RewardMathFn(RewardConfig())
        task_info = {"problem": "What is 2 + 2?", "problem_type": RewardType.MATH, "data_source": "test", "ground_truth": "4"}
        output = reward(task_info, model_response)

        assert not output.is_correct
        assert output.reward == 0.0  # format_error_reward

    def test_format_error_no_boxed_answer(self):
        """Test math reward with format error - no boxed answer."""
        model_response = "<think>2 + 2 = 4</think>\nThe answer is 4."

        reward = RewardMathFn(RewardConfig())
        task_info = {"problem": "What is 2 + 2?", "problem_type": RewardType.MATH, "data_source": "test", "ground_truth": "4"}
        output = reward(task_info, model_response)

        assert not output.is_correct
        assert output.reward == 0.0  # format_error_reward

    def test_unknown_error_no_ground_truth(self):
        """Test math reward with unknown error - no ground truth."""
        model_response = "<think>2 + 2 = 4</think>\nThe answer is \\boxed{4}."

        reward = RewardMathFn(RewardConfig())
        task_info = {"problem": "What is 2 + 2?", "problem_type": RewardType.MATH, "data_source": "test", "ground_truth": None}
        output = reward(task_info, model_response)

        assert not output.is_correct
        assert output.reward == 0.0  # unk_error_reward

    def test_wrong_problem_type(self):
        """Test math reward with wrong problem type."""
        model_response = "<think>This is not a math problem</think>\nSome code here."

        reward = RewardMathFn(RewardConfig())
        task_info = {
            "problem": "Write code to add numbers",
            "problem_type": RewardType.CODE,  # Wrong type
            "data_source": "test",
            "ground_truth": "def add(a, b): return a + b",
        }
        output = reward(task_info, model_response)

        assert not output.is_correct
        assert output.reward == 0.0

    def test_toolcall_bonus(self):
        """Test math reward with tool call bonus."""
        config = RewardConfig()
        config.toolcall_bonus = 0.5
        config.correct_reward = 1.0

        model_response = "<think>I'll use calculation tools</think>\nThe answer is \\boxed{4}."

        reward = RewardMathFn(config)
        task_info = {"problem": "What is 2 + 2?", "problem_type": RewardType.MATH, "data_source": "test", "ground_truth": "4", "has_toolcall": True}
        output = reward(task_info, model_response)

        assert output.is_correct is True
        assert output.reward == 1.5  # correct_reward + toolcall_bonus

    def test_complex_mathematical_expression(self):
        """Test math reward with complex mathematical expression."""
        model_response = "<think>Let me solve this quadratic equation</think>\nThe answer is \\boxed{x = 2, x = -3}."

        reward = RewardMathFn(RewardConfig())
        task_info = {"problem": "Solve x^2 + x - 6 = 0", "problem_type": RewardType.MATH, "data_source": "test", "ground_truth": "x = 2, x = -3"}
        output = reward(task_info, model_response)

        assert output.is_correct is True
        assert output.reward == 1.0

    def test_answer_with_latex_in_ground_truth(self):
        """Test math reward when ground truth contains LaTeX boxed format."""
        model_response = "<think>Let me calculate this carefully</think>\nThe answer is \\boxed{24}."

        reward = RewardMathFn(RewardConfig())
        task_info = {
            "problem": "What is 4!?",
            "problem_type": RewardType.MATH,
            "data_source": "test",
            "ground_truth": "\\boxed{24}",  # Ground truth also in boxed format
        }
        output = reward(task_info, model_response)

        assert output.is_correct is True
        assert output.reward == 1.0

    def test_numerical_tolerance(self):
        """Test math reward with numerical values that should be considered equal."""
        model_response = "<think>The answer is approximately pi</think>\nThe answer is \\boxed{3.14159}."

        reward = RewardMathFn(RewardConfig())
        task_info = {"problem": "What is π to 5 decimal places?", "problem_type": RewardType.MATH, "data_source": "test", "ground_truth": "3.14159"}
        output = reward(task_info, model_response)

        assert output.is_correct is True
        assert output.reward == 1.0

    def test_fraction_equivalence(self):
        """Test math reward with equivalent fractions."""
        model_response = "<think>Let me simplify this fraction</think>\nThe answer is \\boxed{\\frac{1}{2}}."

        reward = RewardMathFn(RewardConfig())
        task_info = {"problem": "Simplify 2/4", "problem_type": RewardType.MATH, "data_source": "test", "ground_truth": "1/2"}
        output = reward(task_info, model_response)

        assert output.is_correct is True
        assert output.reward == 1.0

    def test_empty_ground_truth_list(self):
        """Test math reward with empty ground truth list after processing."""
        model_response = "<think>Calculating the answer</think>\nThe answer is \\boxed{42}."

        reward = RewardMathFn(RewardConfig())
        task_info = {
            "problem": "What is the answer?",
            "problem_type": RewardType.MATH,
            "data_source": "test",
            "ground_truth": ["\\boxed{}", ""],  # Invalid boxed formats that will be filtered out
        }
        output = reward(task_info, model_response)

        assert not output.is_correct
        assert output.reward == 0.0

    def test_empty_boxed_answer_not_rewarded(self):
        """Test that an empty \\boxed{} model answer does not receive a reward,
        even when the ground truth is also empty. See GitHub issue #929."""
        reward = RewardMathFn(RewardConfig())

        # Case 1: empty \boxed{} with empty ground truth — should NOT get full reward
        task_info = {"problem": "Some problem", "problem_type": RewardType.MATH, "data_source": "test", "ground_truth": ""}
        model_response = "<think>I don't know</think>\nThe answer is \\boxed{}."
        output = reward(task_info, model_response)
        assert not output.is_correct
        assert output.reward == 0.0  # format_error_reward, not correct_reward

        # Case 2: empty \boxed{} with a valid ground truth — should NOT get full reward
        task_info = {"problem": "What is 2 + 2?", "problem_type": RewardType.MATH, "data_source": "test", "ground_truth": "4"}
        model_response = "<think>I give up</think>\nThe answer is \\boxed{}."
        output = reward(task_info, model_response)
        assert not output.is_correct
        assert output.reward == 0.0

    def test_custom_config_values(self):
        """Test math reward with custom configuration values."""
        config = RewardConfig()
        config.correct_reward = 2.0
        config.incorrect_reward = -1.0
        config.format_error_reward = -0.5

        # Test correct answer
        reward = RewardMathFn(config)
        task_info = {"problem": "What is 5 + 5?", "problem_type": RewardType.MATH, "data_source": "test", "ground_truth": "10"}
        model_response = "<think>5 + 5 = 10</think>\nThe answer is \\boxed{10}."
        output = reward(task_info, model_response)

        assert output.is_correct is True
        assert output.reward == 2.0

        # Test incorrect answer
        model_response = "<think>5 + 5 = 11</think>\nThe answer is \\boxed{11}."
        output = reward(task_info, model_response)

        assert not output.is_correct
        assert output.reward == -1.0

        # Test format error
        model_response = "The answer is 10."  # No thought delimiter
        output = reward(task_info, model_response)

        assert not output.is_correct
        assert output.reward == -0.5

    @pytest.mark.skip(reason="ORM functionality requires external API access")
    def test_orm_evaluation(self):
        """Test math reward with ORM (Oracle Reward Model) evaluation."""
        config = RewardConfig()
        config.use_math_orm = True

        model_response = "<think>This is a complex answer</think>\nThe answer is \\boxed{equivalent_but_different_format}."

        reward = RewardMathFn(config)
        task_info = {"problem": "Complex math problem", "problem_type": RewardType.MATH, "data_source": "test", "ground_truth": "different_but_equivalent_format"}
        output = reward(task_info, model_response)

        # This test would require mocking the ORM calls
        # For now, we just ensure the code path doesn't crash
        assert isinstance(output, RewardOutput)
        assert isinstance(output.is_correct, bool)
        assert isinstance(output.reward, int | float)
