# %%
import os

import json_repair
import litellm
from loguru import logger

from shared.workflows.metrics.qa_metrics import evaluate_prediction

try:
    from src.envs import LITELLM_CACHE_DIR
except ImportError:
    logger.exception("LITELLM_CACHE_DIR not found in src.envs, trying to look in environment")
    LITELLM_CACHE_DIR = os.getenv("LITELLM_CACHE_DIR", None)

if LITELLM_CACHE_DIR is None or LITELLM_CACHE_DIR == "":
    logger.warning("LITELLM_CACHE_DIR not found or empty, disabling cache")
else:
    litellm.enable_cache(type="disk", disk_cache_dir=LITELLM_CACHE_DIR)


def get_original_prediction(question: str, model: str = "gpt-4o-mini"):
    sys_prompt = """
    You are a quizbowl contestant.
    Your task is to provide a final concise answer to the input question that is describing a person, location, concept, etc.
    """.strip()

    qa_response = litellm.completion(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"Question: {question}"},
        ],
    )
    return qa_response["choices"][0]["message"]["content"]


def helpfulness_score(
    question: str,
    answer_refs: list[str] | str,
    helper_guess: str,
    helper_explanation: str,
    orig_prediction: str | None = None,
    model: str = "gpt-4o-mini",
) -> float:
    """
    Compute a helpfulness score for a response to a question.
    """

    system_prompt = """
    You are a careful and logical Quiz-bowl contestant.
    Your task is to assess the helpfulness of another QA system's suggestion on a provided question. You will be given a question in quizbowl format, your original prediction, and the helper guess and explanation.
    You can respond with one of the following three options:
    * "keep": if you think your own prediction makes more sense, you should keep your original prediction.
    * "flip": if you believe the helper's guess is correct based on the explanation, you can flip to the helper's guess.
    * "revise": if the helper's guess is incorrect, but the explanation provides enough information to revise your prediction to a new prediction that is more correct, you should revise your prediction.

    ## Input Format:
    Question: <question>
    Your Prediction: <your_prediction>
    Helper Guess: <helper_guess>
    Helper Explanation: <helper_explanation>

    ## Output Format: a json with the following fields:
    * decision: "keep", "flip", or "revise"
    * revision: "<revised_prediction>" if decision is "revise", otherwise empty string
    * reason: a short explanation (single sentence) for your decision
    """
    system_prompt = """You are a conscientious, logic-driven Quizbowl contestant.

Your task is to evaluate whether a helper QA system’s suggestion improves your answer to a Quizbowl question. You will receive:

• **Question** – A leadin (premise) and a bonus question
• **Your Prediction** – the answer you originally gave
• **Helper Guess** – the helper system’s proposed answer
• **Helper Explanation** – the helper’s rationale

Choose exactly one of these decisions:

* **"keep"** – your original prediction still appears most plausible.
* **"flip"** – the helper’s guess is better supported and should replace yours.
* **"revise"** – the helper's guess is wrong, but its explanation enables you to craft a new, more accurate answer. Remember to use this option sparingly.

---

### Input (verbatim)

Question: <question>
Your Prediction: <your_prediction>
Helper Guess: <helper_guess>
Helper Explanation: <helper_explanation>

### Output (JSON)

{
  "decision": "keep" | "flip" | "revise",
  "revision": "<new_prediction or empty string>",
  "reason": "<one-sentence justification>"
}

Keep the justification concise (~1 sentence). Leave *revision* blank unless **decision** is "revise".
"""

    prompt_template = """
    Question: {question}
    Your Prediction: {orig_prediction}
    Helper Guess: {helper_guess}
    Helper Explanation: {helper_explanation}
    """

    if orig_prediction is None:
        orig_prediction = get_original_prediction(question, model)

    final_response = litellm.completion(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": prompt_template.format(
                    question=question,
                    orig_prediction=orig_prediction,
                    helper_guess=helper_guess,
                    helper_explanation=helper_explanation,
                ),
            },
        ],
    )
    response = json_repair.loads(final_response["choices"][0]["message"]["content"])
    response["answer_refs"] = answer_refs
    response["orig_prediction"] = orig_prediction
    response["helper_guess"] = helper_guess
    response["helper_explanation"] = helper_explanation
    if response["decision"] == "revise":
        response["final_guess"] = response["revision"]
    elif response["decision"] == "flip":
        response["final_guess"] = helper_guess
    else:
        response["final_guess"] = orig_prediction
    response["orig_correct"] = evaluate_prediction(orig_prediction, answer_refs)
    response["helper_correct"] = evaluate_prediction(helper_guess, answer_refs)
    response["final_correct"] = evaluate_prediction(response["final_guess"], answer_refs)
    return response
