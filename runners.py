from concurrent import futures

from datasets import Dataset
from loguru import logger
from tqdm import tqdm

from .metrics import evaluate_prediction
from .qb_agents import QuizBowlBonusAgent, QuizBowlTossupAgent


def get_question_runs(example: dict) -> list[str]:
    question_runs = []
    tokens = example["question"].split()
    for run_idx in example["run_indices"]:
        question_runs.append(" ".join(tokens[: run_idx + 1]))
    return question_runs


def run_and_evaluate_tossup(agent: QuizBowlTossupAgent, example: dict):
    results = []
    question_runs = get_question_runs(example)
    try:
        run_outputs = agent.run(question_runs)
    except Exception as e:
        logger.error(f"Error running {example['qid']}: {e}")
        run_outputs = []
    for run_output in run_outputs:
        run_out = {
            "answer": run_output["answer"],
            "confidence": run_output["confidence"],
            "buzz": run_output["buzz"],
            "run_position": run_output["position"],
        }
        run_out["qid"] = example["qid"]
        run_out["score"] = evaluate_prediction(run_out["answer"], example["clean_answers"])

        # This is 1-indexed token-position
        run_out["token_position"] = example["run_indices"][run_output["position"] - 1] + 1
        results.append(run_out)
    return results


def run_and_eval_tossup_dataset(agent: QuizBowlTossupAgent, dataset: Dataset, num_workers: int = 4):
    def process_example(example: dict):
        return run_and_evaluate_tossup(agent, example)

    run_outputs_by_qid = {}
    with futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_example = {executor.submit(process_example, example): example for example in dataset}

        # Use tqdm to show progress as futures complete
        for future in tqdm(futures.as_completed(future_to_example), total=len(dataset), desc="Running Tossups"):
            example_results = future.result()
            qid = example_results[0]["qid"]  # Taking qid from the first run_output
            run_outputs_by_qid[qid] = example_results

    tossup_outputs = [run_outputs_by_qid[qid] for qid in dataset["qid"]]
    return tossup_outputs


def run_and_evaluate_bonus(agent: QuizBowlBonusAgent, example: dict) -> list[dict]:
    results = []
    for i, part in enumerate(example["parts"], start=1):
        try:
            result = agent.run(example["leadin"], part["part"])
            result = {
                "answer": result["answer"],
                "confidence": result["confidence"],
                "explanation": result["explanation"],
            }
        except Exception as e:
            logger.error(f"Error running {example['qid']} part {i}: {e}")
            result = {"answer": "ERROR", "confidence": 0.0, "explanation": "Error producing answer."}
        result["qid"] = example["qid"]
        result["part_number"] = i
        result["score"] = evaluate_prediction(result["answer"], part["clean_answers"])
        results.append(result)
    return results


def run_and_eval_bonus_dataset(agent: QuizBowlBonusAgent, dataset: Dataset, num_workers: int = 4) -> list[dict]:
    def process_example(example: dict) -> list[dict]:
        return run_and_evaluate_bonus(agent, example)

    part_outputs_by_qid = {}
    with futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_example = {executor.submit(process_example, example): example for example in dataset}

        # Use tqdm to show progress as futures complete
        for future in tqdm(futures.as_completed(future_to_example), total=len(dataset), desc="Running Bonuses"):
            example_results = future.result()
            qid = example_results[0]["qid"]
            part_outputs_by_qid[qid] = example_results

    bonus_outputs = [part_outputs_by_qid[qid] for qid in dataset["qid"]]
    return bonus_outputs
