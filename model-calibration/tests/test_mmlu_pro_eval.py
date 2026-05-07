from examples.evaluate_mmlu_pro_lut import (
    extract_choice,
    format_mmlu_pro_prompt,
    option_letters,
)


def test_option_letters_cover_mmlu_pro_ten_choice_format():
    assert option_letters(10) == list("ABCDEFGHIJ")


def test_format_mmlu_pro_prompt_includes_question_options_and_answer_instruction():
    sample = {
        "question": "What is 2 + 2?",
        "options": ["1", "2", "3", "4"],
    }

    prompt = format_mmlu_pro_prompt(sample)

    assert "Question: What is 2 + 2?" in prompt
    assert "A. 1" in prompt
    assert "D. 4" in prompt
    assert prompt.endswith("The answer is (")


def test_extract_choice_prefers_parenthesized_answer_letter():
    assert extract_choice("I think carefully. The answer is (C).", 4) == "C"
    assert extract_choice("Answer: D", 4) == "D"
    assert extract_choice("No valid option here", 4) is None
