import random
from typing import Literal

from loguru import logger


def get_buzz_info(model_output):
    # Find the first buzz, its position, and correctness
    for o in model_output.get("run_outputs", []):
        if o.get("buzz", False):
            return o["token_position"], o["correct"]
    return None, False


def get_points(correct: bool):
    if int(correct) not in [0, 1]:
        raise ValueError(f"Correct must be a boolean, got {correct}")
    return 1.5 * correct - 0.5


def compare_tossup_outputs(
    model1_output: dict,
    model2_output: dict,
    tie_strategy: Literal["first", "last", "random"] = "random",
) -> tuple[float, float]:
    """
    Compare the tossup outputs of two models and compute the points assigned to each.

    Args:
        model1_output (dict): Tossup output for model 1. Should contain "run_outputs" (list of dicts with "buzz", "correct", "token_position").
        model2_output (dict): Tossup output for model 2. Same format as model1_output.

    Returns:
        tuple[float, float]: (points for model 1, points for model 2)
    """

    # Verify that run_outputs are sorted by token_position
    for model_output in [model1_output, model2_output]:
        run_outputs = model_output["run_outputs"]
        if not all(
            run_outputs[i]["token_position"] < run_outputs[i + 1]["token_position"]
            for i in range(len(run_outputs) - 1)
        ):
            logger.info("run_outputs are not sorted by token_position, sorting them")
            model_output["run_outputs"].sort(key=lambda x: x["token_position"])

    # Get buzz positions and correctness
    pos1, correct1 = get_buzz_info(model1_output)
    pos2, correct2 = get_buzz_info(model2_output)

    # If neither buzzes, both get 0
    if pos1 is None and pos2 is None:
        return 0.0, 0.0

    # If only model1 buzzes
    elif pos2 is None:
        return get_points(correct1), 0.0

    # If only model2 buzzes
    elif pos1 is None:
        return 0.0, get_points(correct2)

    # Both buzzed

    # If same time, use tie strategy
    if pos1 == pos2:
        if tie_strategy == "first":
            pos2 += 1
        elif tie_strategy == "last":
            pos1 += 1
        elif tie_strategy == "random":
            if random.random() < 0.5:
                pos2 += 1
            else:
                pos1 += 1

    # Whoever buzzes first gets the first chance, if they are incorrect, they get -0.5 points, the second model gets the second chance, but if they are incorrect they get 0 points

    if pos1 < pos2:
        return get_points(correct1), get_points(correct2) * (1 - correct1)
    else:
        return get_points(correct1) * (1 - correct2), get_points(correct2)


def compare_nway_tossup_outputs(
    model_outputs: list[dict],
    tie_strategy: Literal["first", "last", "random"] = "random",
    verify_sorted: bool = True,
) -> list[float]:
    """
    Compare the tossup outputs of multiple models and compute the points assigned to each.

    In quiz bowl format:
    - Models buzz at different token positions
    - First correct buzzer gets +1.5 points and ends the question
    - Incorrect buzzers get -0.5 points and the question continues
    - Models that don't buzz get 0 points
    - If no one buzzes, everyone gets 0 points

    Args:
        model_outputs (list[dict]): List of tossup outputs for each model.
                                   Each should contain "run_outputs" (list of dicts with "buzz", "correct", "token_position").
        tie_strategy (Literal["first", "last", "random"]): How to break ties when models buzz at same position.
        verify_sorted (bool): Whether to verify that the run_outputs are sorted by token_position. Turn off if you know they are already sorted and you want to save time.

    Returns:
        list[float]: Points for each model in the same order as input
    """
    n_models = len(model_outputs)

    # Verify that run_outputs are sorted by token_position for each model
    if verify_sorted:
        for i, model_output in enumerate(model_outputs):
            run_outputs = model_output["run_outputs"]
            if not all(
                run_outputs[j]["token_position"] < run_outputs[j + 1]["token_position"]
                for j in range(len(run_outputs) - 1)
            ):
                logger.info(f"run_outputs for model {i} are not sorted by token_position, sorting them")
                model_output["run_outputs"].sort(key=lambda x: x["token_position"])

    # Get buzz info for all models
    buzz_info = []
    for i, model_output in enumerate(model_outputs):
        pos, correct = get_buzz_info(model_output)
        sec_sort_key = i if tie_strategy == "first" else -i if tie_strategy == "last" else random.random()
        if pos is not None:
            buzz_info.append((pos, sec_sort_key, i, correct))  # (position, sort_key, model_index, correctness)

    # Initialize points for all models
    points = [0.0] * n_models

    # If no one buzzes, everyone gets 0
    if not buzz_info:
        return points

    # Sort by buzz position (earliest first)
    buzz_info.sort()

    # Process buzzes in order
    for pos, sec_sort_key, model_idx, correct in buzz_info:
        points[model_idx] = get_points(correct)
        if correct:
            break

    return points
