# %%
import re

import inflect
from unidecode import unidecode

p = inflect.engine()


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

    pred_original = unidecode(prediction.lower().strip())
    ans = unidecode(answer.strip().lower())

    # Handle empty strings
    if not pred_original or not ans:
        return False

    # Exact match
    if pred_original == ans:
        return True

    # --- Helper function to check a given prediction form ---
    def check_form(pred_to_check: str) -> bool:
        if not pred_to_check:  # Skip if empty (e.g. p.singular_noun("") is False)
            return False
        # Create regex pattern that matches word boundaries
        # \b ensures we match whole words only
        pattern = r"\b" + re.escape(pred_to_check) + r"\b"
        return bool(re.search(pattern, ans))

    # --- End helper ---

    # Check original form
    if check_form(pred_original):
        return True

    # Check singular form
    # p.singular_noun(word) returns False if already singular or no reliable singular form
    pred_singular = p.singular_noun(pred_original)
    if pred_singular and pred_singular != pred_original:  # Check if a singular form was found and it's different
        if check_form(pred_singular):
            return True

    # Check plural form
    pred_plural = p.plural(pred_original)
    if pred_plural != pred_original:  # Check if plural is different (it usually will be)
        if check_form(pred_plural):
            return True

    # Check for cases where the prediction might be a plural and the answer is singular
    # (inflect.singular_noun might not catch all if the input is already "singular" by its rules)
    # This is a bit more heuristic
    if pred_original.endswith("s"):
        pred_ending_s_removed = pred_original[:-1]
        if check_form(pred_ending_s_removed):
            return True
    if pred_original.endswith("es"):
        pred_ending_es_removed = pred_original[:-2]
        if check_form(pred_ending_es_removed):
            return True

    return False


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


# %%
if __name__ == "__main__":
    # Test cases: (prediction, answer, expected_result, description)
    test_cases = [
        # Exact Matches & Basic Cases
        ("hello world", "hello world", True, "Exact match"),
        ("Hello World", "hello world", True, "Case insensitive match"),
        ("hello", "hello world", True, "Prediction is a prefix word"),
        ("world", "hello world", True, "Prediction is a suffix word"),
        ("invalid", "hello world", False, "No match"),
        ("", "hello world", False, "Empty prediction"),
        ("hello world", "", False, "Empty answer"),
        ("", "", False, "Both empty"),
        # Unicode & Accent Handling (via unidecode)
        ("naïve", "naive", True, "Prediction with accent, answer plain"),
        ("naive", "naïve", True, "Prediction plain, answer with accent"),
        ("crème brûlée", "creme brulee", True, "Unicode prediction, ascii answer"),
        ("creme brulee", "crème brûlée", True, "Ascii prediction, unicode answer"),
        # Word Boundary Checks
        ("press", "compress", False, "Substring, but not whole word ('press' in 'compress')"),
        ("hot press", "Polka hot press", True, "Multi-word prediction, whole words match"),
        ("art", "state-of-the-art", True, "Match part of hyphenated word"),
        ("state", "state-of-the-art", True, "Match start of hyphenated word"),
        ("cat", "caterpillar", False, "Prediction substring of a word in answer"),
        (
            "apple tree",
            "apple",
            False,
            "Prediction longer than answer, answer is prefix",
        ),  # Needs careful thought if this is desired
        # Simple Pluralization (s/es)
        ("cat", "cats", True, "Simple plural: cat -> cats"),
        ("cats", "cat", True, "Simple singular: cats -> cat"),
        ("bus", "buses", True, "Simple plural (es): bus -> buses"),
        ("buses", "bus", True, "Simple singular (es): buses -> bus"),
        ("wish", "wishes", True, "Simple plural (sh+es): wish -> wishes"),
        ("wishes", "wish", True, "Simple singular (sh+es): wishes -> wish"),
        ("fox", "foxes", True, "Simple plural (x+es): fox -> foxes"),
        ("foxes", "fox", True, "Simple singular (x+es): foxes -> fox"),
        # Inflect Library Plural/Singular (Irregular, -y -> -ies, -f -> -ves etc.)
        ("mouse", "mice", True, "Irregular plural: mouse -> mice"),
        ("mice", "mouse", True, "Irregular singular: mice -> mouse"),
        ("goose", "geese", True, "Irregular plural: goose -> geese"),
        ("geese", "goose", True, "Irregular singular: geese -> goose"),
        ("woman", "women", True, "Irregular plural: woman -> women"),
        ("women", "woman", True, "Irregular singular: women -> woman"),
        ("leaf", "leaves", True, "Plural (-f to -ves): leaf -> leaves"),
        ("leaves", "leaf", True, "Singular (-ves to -f): leaves -> leaf"),
        ("baby", "babies", True, "Plural (-y to -ies): baby -> babies"),
        ("babies", "baby", True, "Singular (-ies to -y): babies -> baby"),
        ("criterion", "criteria", True, "Greek plural: criterion -> criteria"),
        ("criteria", "criterion", True, "Greek singular: criteria -> criterion"),
        ("analysis", "analyses", True, "Plural (-is to -es): analysis -> analyses"),
        ("analyses", "analysis", True, "Singular (-es to -is): analyses -> analysis"),
        ("appendix", "appendices", True, "Plural: appendix -> appendices (inflect default)"),
        ("appendices", "appendix", True, "Singular: appendices -> appendix"),
        ("appendix", "appendixes", True, "Plural: appendix -> appendixes (alternative)"),
        ("appendixes", "appendix", True, "Singular: appendixes -> appendix"),
        # Words that are same singular/plural (inflect should handle gracefully)
        ("fish", "fish", True, "Same singular/plural: fish"),
        ("sheep", "sheep", True, "Same singular/plural: sheep"),
        ("fish", "a school of fish", True, "Same S/P in context"),
        # Regex Special Characters in Prediction (should be escaped)
        ("c++", "c++ language", True, "Prediction with '+' special char"),
        ("value[key]", "value[key] lookup", True, "Prediction with '[' ']' special chars"),
        ("dot.net", "Microsoft dot.net framework", True, "Prediction with '.' special char"),
        # Harder cases / Ambiguities / Specific inflect behaviors
        ("news", "news", True, "'news' is mass noun, singular_noun is False"),  # `p.singular_noun('news')` is False
        ("The United States", "United States of America", True, "Proper noun phrase match"),
        ("United States", "The United States", True, "Partial proper noun phrase"),
        ("datum", "data", True, "Datum -> Data"),  # Inflect handles data as plural of datum
        ("data", "datum", True, "Data -> Datum"),  # Inflect should handle singular_noun('data')
        ("focus", "foci", True, "Focus -> Foci"),
        ("foci", "focus", True, "Foci -> Focus"),
        ("focus", "focuses", True, "Focus -> Focuses (alternative plural)"),
        ("focuses", "focus", True, "Focuses -> Focus"),
        # Cases where heuristic s/es removal might be the only match
        ("cookies", "cookie", True, "Heuristic singular s: cookies -> cookie"),
        ("branches", "branch", True, "Heuristic singular es: branches -> branch"),
        # A case where the prediction is a substring of the answer, but also a word
        ("a", "a cat sat", True, "Single letter prediction match word"),
        ("a", "alphabet", False, "Single letter prediction, not word match"),
        # Prediction longer than answer
        ("long prediction", "short", False, "Prediction longer than answer"),
        # Test from original docstring
        ("hot press", "Polka hot press", True, "Docstring Example 1"),
        ("press", "compress", False, "Docstring Example 2"),
    ]

    print("--- Running Test Cases for answer_match ---")
    passed_count = 0
    failed_count = 0

    for i, (pred, ans, expected, desc) in enumerate(test_cases):
        result = answer_match(pred, ans)
        status = "\033[92mPASSED\033[0m" if result == expected else "\033[91mFAILED\033[0m"
        if result == expected:
            passed_count += 1
        else:
            failed_count += 1
        print(
            f"Test {i + 1:02d}: {status} | Pred: '{pred}', Ans: '{ans}' | Expected: {expected}, Got: {result} | Desc: {desc}"
        )

    print("\n--- Test Summary ---")
    print(f"Total tests: {len(test_cases)}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {failed_count}")
    print("---------------------")
