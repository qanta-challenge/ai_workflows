def evaluate_prediction(prediction: str, clean_answers: list[str] | str) -> int:
    """Evaluate the buzz of a prediction against the clean answers."""
    if isinstance(clean_answers, str):
        clean_answers = [clean_answers]
    pred = prediction.lower().strip()
    if not pred:
        return 0
    for answer in clean_answers:
        answer = answer.strip().lower()
        if answer and answer in pred:
            return 1
    return 0
