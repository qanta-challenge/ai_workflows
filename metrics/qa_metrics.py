# %%
from unidecode import unidecode


def answer_match(prediction: str, answer: str) -> bool:
    """Check if the prediction matches the answer.

    Args:
        prediction: The predicted answer
        answer: The correct answer

    Returns:
        True if prediction matches answer, where prediction can be a sequence of words
        in answer but not a substring of a word. For example:
        - "hot press" in "Polka hot press" -> True
        - "press" in "compress" -> False
    """
    import re

    pred = unidecode(prediction.lower().strip())
    ans = unidecode(answer.strip().lower())

    # Handle empty strings
    if not pred or not ans:
        return False

    # Exact match
    if pred == ans:
        return True

    # Create regex pattern that matches word boundaries
    # \b ensures we match whole words only
    pattern = r"\b" + re.escape(pred) + r"\b"

    return bool(re.search(pattern, ans))


def evaluate_prediction(prediction: str, clean_answers: list[str] | str) -> int:
    """Evaluate the buzz of a prediction against the clean answers."""
    if isinstance(clean_answers, str):
        clean_answers = [clean_answers]
    pred = prediction.lower().strip()
    if not pred:
        return 0
    for answer in clean_answers:
        answer = answer.strip()
        if not answer:
            continue
        if answer_match(pred, answer):
            return 1
    return 0
