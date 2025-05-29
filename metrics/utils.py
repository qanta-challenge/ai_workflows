# %%
import re
import string
import unicodedata

import inflect
import regex
from nltk.stem import WordNetLemmatizer
from unidecode import unidecode

p = inflect.engine()

lemmatizer = WordNetLemmatizer()

punkts = set(string.punctuation)


def is_punkt(char):
    return unicodedata.category(char).startswith("P") or char in punkts


def normalize_apostrophe(text):
    return text.replace("’", "'").replace("`", "'").replace("‘", "'").replace("”", '"').replace("“", '"')


def remove_articles(text, start_only=False):
    if start_only:
        return regex.sub(r"^(?:\s*)(a|an|the)\b", " ", text, flags=regex.IGNORECASE)
    else:
        return regex.sub(r"\b(a|an|the)\b", " ", text)


def white_space_fix(text):
    return " ".join(text.split())


def fix_answer(s, lower=True):
    def remove_punc(text):
        return "".join(ch for ch in text if not is_punkt(ch))

    if lower:
        s = s.lower()

    return white_space_fix(remove_articles(remove_punc(normalize_apostrophe(str(s)))))


def normalize_answer(text, lower=True):
    if isinstance(text, list):
        result = []
        for elem in text:
            result.append(fix_answer(elem, lower=lower))
        return result
    else:
        return fix_answer(str(text), lower=lower)
