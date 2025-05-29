# %%
import re

import inflect
import nltk

nltk.download("stopwords")
from nltk.corpus import stopwords
from unidecode import unidecode

from .utils import normalize_answer, normalize_apostrophe, remove_articles, white_space_fix

p = inflect.engine()
stopwords = set(stopwords.words("english"))


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

    def normalize(text):
        text = unidecode(normalize_apostrophe(text.lower().strip()))
        text = text.replace("·", "")
        return white_space_fix(remove_articles(text, start_only=True))

    pred = normalize(prediction)
    ans = normalize(answer)

    # Handle empty strings
    if not pred or not ans:
        return False

    # Exact match
    if pred == ans:
        return True

    def check_presence(pred_to_check: str, ans: str) -> bool:
        """Check if the answer is a subsequence of the prediction."""
        pattern = r"\b" + re.escape(ans) + r"\b"
        if re.search(pattern, pred_to_check):
            return True

        # Check for the length of pred_to_check. Don't search if pred_to_check is a stopword.
        if pred_to_check in stopwords or len(pred_to_check) <= 2:
            return pred_to_check == ans
        return False

    # --- Helper function to check a given prediction form ---
    def check_form(pred_to_check: str) -> bool:
        if not pred_to_check:  # Skip if empty (e.g. p.singular_noun("") is False)
            return False
        return check_presence(pred_to_check, ans)

    candidates = {
        pred,
        p.singular_noun(pred),
        p.plural(pred),
        pred.removesuffix("s"),
        pred.removesuffix("es"),
    } - {False}

    # If pred is a single word, allow -ing, -ism, etc forms
    if len(pred.split()) == 1:
        suffixes = ["ing", "ism", "ist", "ian", "ment", "ness", "ity", "est", "ed", "er", "al", "ous"]
        for suffix in suffixes:
            candidates.add(pred + suffix)

    for candidate in candidates:
        if check_form(candidate):
            return True

    return False


def evaluate_answer_match(prediction: str, clean_answers: list[str] | str) -> int:
    """Evaluate the buzz of a prediction against the clean answers."""
    if isinstance(clean_answers, str):
        clean_answers = [clean_answers]
    pred = prediction.lower().strip()
    if not pred:
        return 0
    for answer in clean_answers:
        if not (a := answer.strip()):
            continue
        if answer_match(pred, a):
            return 1
    return 0


def evaluate_em_match(prediction: str, clean_answers: list[str] | str) -> int:
    """Check if the prediction matches the answer."""
    if isinstance(clean_answers, str):
        clean_answers = [clean_answers]
    pred = normalize_answer(prediction)
    if not pred:
        return 0
    for answer in clean_answers:
        if not (a := normalize_answer(answer)):
            continue
        if pred == a:
            return 1
        if answer_match(pred, a):
            return 1
    return 0


def evaluate_prediction(prediction: str, clean_answers: list[str] | str) -> int:
    """Evaluate the buzz of a prediction against the clean answers."""
    return evaluate_em_match(prediction, clean_answers)


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
