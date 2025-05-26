from concurrent import futures

from datasets import Dataset
from loguru import logger
from tqdm import tqdm

from .metrics import evaluate_prediction, helpfulness_score
from .qb_agents import QuizBowlBonusAgent, QuizBowlTossupAgent


def get_question_runs(example: dict) -> list[str]:
    question_runs = []
    tokens = example["question"].split()
    for run_idx in example["run_indices"]:
        question_runs.append(" ".join(tokens[: run_idx + 1]))
    return question_runs


def run_and_evaluate_tossup(
    agent: QuizBowlTossupAgent,
    example: dict,
    return_extras: bool = False,
    early_stop: bool = True,
):
    results = []
    question_runs = get_question_runs(example)
    try:
        run_outputs = agent.run(question_runs, early_stop=early_stop)
    except Exception as e:
        logger.error(f"Error running {example['qid']}: {e}")
        run_outputs = []
    for run_output in run_outputs:
        if return_extras:
            run_out = run_output
        else:
            run_out = {k: run_output[k] for k in ["guess", "confidence", "buzz", "run_idx"]}
        run_out["correct"] = evaluate_prediction(run_out["guess"], example["clean_answers"])
        # This is 1-indexed token-position
        run_out["token_position"] = example["run_indices"][run_output["run_idx"] - 1] + 1
        results.append(run_out)
    return {
        "qid": example["qid"],
        "run_outputs": results,
    }


def run_and_eval_tossup_dataset(
    agent: QuizBowlTossupAgent,
    dataset: Dataset,
    early_stop: bool = True,
    num_workers: int = 4,
    return_extras: bool = False,
    tqdm_provider: tqdm = tqdm,
):
    def process_example(example: dict):
        return run_and_evaluate_tossup(agent, example, return_extras, early_stop)

    outputs_map = {}
    with futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_example = {executor.submit(process_example, example): example for example in dataset}

        # Use tqdm to show progress as futures complete
        for future in tqdm_provider(
            futures.as_completed(future_to_example), total=len(dataset), desc="Running Tossups"
        ):
            result = future.result()
            outputs_map[result["qid"]] = result

    tossup_outputs = [outputs_map[qid] for qid in dataset["qid"]]
    return tossup_outputs


def run_and_evaluate_bonus(agent: QuizBowlBonusAgent, example: dict, return_extras: bool = False) -> dict:
    results = []
    for i, part in enumerate(example["parts"], start=1):
        try:
            result = agent.run(example["leadin"], part["question"])
            if return_extras:
                result = result
            else:
                result = {k: result[k] for k in ["guess", "confidence", "explanation"]}
        except Exception as e:
            logger.error(f"Error running {example['qid']} part {i}: {e}")
            result = {"guess": "ERROR", "confidence": 0.0, "explanation": "Error producing answer."}
        result["number"] = i
        result["correct"] = evaluate_prediction(result["guess"], part["clean_answers"])
        results.append(result)
    return {
        "qid": example["qid"],
        "part_outputs": results,
    }


def run_and_eval_bonus_dataset(
    agent: QuizBowlBonusAgent,
    dataset: Dataset,
    num_workers: int = 4,
    return_extras: bool = False,
    tqdm_provider: tqdm = tqdm,
) -> list[dict]:
    def process_example(example: dict) -> dict:
        return run_and_evaluate_bonus(agent, example, return_extras)

    outputs_map = {}
    with futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_example = {executor.submit(process_example, example): example for example in dataset}

        # Use tqdm to show progress as futures complete
        for future in tqdm_provider(
            futures.as_completed(future_to_example), total=len(dataset), desc="Running Bonuses"
        ):
            result = future.result()
            outputs_map[result["qid"]] = result

    bonus_outputs = [outputs_map[qid] for qid in dataset["qid"]]
    return bonus_outputs


def inject_bonus_metrics_example(model_output: dict, example: dict, max_explanation_tokens: int = 30) -> dict:
    results = []
    for part, output in zip(example["parts"], model_output["part_outputs"]):
        question_text = f"Leadin: {example['leadin']}\n\nQuestion: {part['question']}"
        answer_refs = part["clean_answers"]
        explanation = output["explanation"]
        guess = output["guess"]
        confidence = output["confidence"]
        expl_tokens = explanation.split(" ")
        if len(expl_tokens) > max_explanation_tokens:
            explanation = " ".join(expl_tokens[:max_explanation_tokens]) + "...[TRUNCATED]"
        h_scores = helpfulness_score(question_text, answer_refs, guess, explanation)
        h_scores["helper_confidence"] = confidence
        results.append(h_scores)
    return {"scores": results}


def inject_bonus_metrics(system_outputs: list[dict], dataset: Dataset, num_workers: int = 4) -> list[dict]:
    outputs_map = {}
    for bonus_output in system_outputs:
        qid = bonus_output["qid"]
        outputs_map[qid] = bonus_output

    def process_example(example: dict) -> dict:
        qid = example["qid"]
        system_output = outputs_map[qid]
        return inject_bonus_metrics_example(system_output, example)

    scored_examples = dataset.map(process_example, num_proc=num_workers)
    for e, o in zip(scored_examples, system_outputs):
        o["scores"] = e["scores"]
    return system_outputs
