import numpy as np
import pandas as pd


def create_tossup_df_entry(
    run_outputs: list[dict],
    run_indices: list[int],
    human_buzz_positions: list[tuple[int, int]] | None = None,
) -> dict:
    """Create a dataframe entry from a list of model outputs."""
    chosen_idx = None
    earliest_ok_idx = None
    is_correct = None
    for i, o in enumerate(run_outputs):
        if chosen_idx is None and o["buzz"]:
            chosen_idx = o["token_position"]
            is_correct = o["score"]
        if earliest_ok_idx is None and o["score"]:
            earliest_ok_idx = o["token_position"]
    if is_correct is None:
        is_correct = False

    # if buzz is not the last index, correct scores 10, incorrect scores -5
    # if buzz is the final index, correct scores 5, incorrect scores 0

    if chosen_idx is None:
        tossup_score = 0
    elif chosen_idx == run_indices[-1] + 1:
        tossup_score = 5 if is_correct else 0
    else:
        tossup_score = 10 if is_correct else -5

    gap = None if (chosen_idx is None or earliest_ok_idx is None) else chosen_idx - earliest_ok_idx
    if earliest_ok_idx is None:
        cls = "hopeless"
    elif chosen_idx is None:
        cls = "never-buzzed"  # Opportunity missed to score
    elif chosen_idx == earliest_ok_idx:
        cls = "best-buzz"  # Perfect timing
    elif chosen_idx > earliest_ok_idx:
        cls = "late-buzz"  # Opportunity missed to buzz earlier
    elif chosen_idx < earliest_ok_idx:
        cls = "premature"  # Opportunity missed to score

    human_win_rate = None
    human_win_rate_strict = None
    if human_buzz_positions is not None:
        human_win_rate = 0
        win_counts = 0
        soft_win_counts = 0
        for pos, score in human_buzz_positions:
            if is_correct and chosen_idx <= pos:
                win_counts += 1
                soft_win_counts += 1
            elif (chosen_idx is None or pos < chosen_idx) and score < 0:
                soft_win_counts += 1
        human_win_rate = soft_win_counts / len(human_buzz_positions)
        human_win_rate_strict = win_counts / len(human_buzz_positions)

    metric_dict = {
        "chosen_idx": chosen_idx,
        "earliest_ok_idx": earliest_ok_idx,
        "gap": gap,
        "cls": cls,
        "tossup_score": tossup_score,
        "is_correct": int(is_correct),
    }
    if human_buzz_positions is not None:
        metric_dict["human_win_rate"] = human_win_rate
        metric_dict["human_win_rate_strict"] = human_win_rate_strict
    return metric_dict


def prepare_tossup_results_df(
    system_outputs: list[list[dict]],
    run_indices: list[list[int]],
    human_buzz_positions: list[list[tuple[int, int]]] | None = None,
) -> pd.DataFrame:
    """Create a dataframe from a list of model outputs."""
    records = []
    for i, (indices, outputs) in enumerate(zip(run_indices, system_outputs)):
        human_buzz_positions_i = None
        if human_buzz_positions is not None:
            human_buzz_positions_i = human_buzz_positions[i]
        entry = create_tossup_df_entry(outputs, indices, human_buzz_positions_i)
        records.append(entry)
    return pd.DataFrame.from_records(records)


def summarize_tossup_metrics(df: pd.DataFrame) -> dict:
    # Prepare a dataframe of aggregated metrics:
    # - Mean Tossup Score
    # - Buzz Accuracy
    # - Mean +ve Gap
    # - Mean -ve Gap
    # - Mean Buzz Position

    positions = df["chosen_idx"].dropna()
    gaps = df["gap"].dropna()
    pos_gaps = gaps.loc[gaps >= 0]
    neg_gaps = gaps.loc[gaps < 0]

    mean_tossup_score = df["tossup_score"].sum() / len(df)
    metrics = {
        "tossup_score": mean_tossup_score,
        "buzz_accuracy": df["is_correct"].mean(),
        "buzz_position": np.mean(positions),
        "gap_pos": pos_gaps.mean(),
        "gap_neg": neg_gaps.mean(),
    }
    if "human_win_rate" in df.columns:
        metrics["human_win_rate"] = df["human_win_rate"].mean()
        metrics["human_win_rate_strict"] = df["human_win_rate_strict"].mean()

    return metrics


def compute_tossup_metrics(
    model_outputs: list[list[dict]],
    run_indices: list[list[int]],
    human_buzz_positions: list[list[tuple[int, int]]] | None = None,
) -> dict:
    """Create a table from a dataframe."""
    df = prepare_tossup_results_df(model_outputs, run_indices, human_buzz_positions)
    return summarize_tossup_metrics(df)


def compute_bonus_metrics(system_outputs: list[list[dict]]) -> dict:
    """Create a table from a dataframe."""
    # Compute Metrics
    total_parts = 0
    total_questions = 0
    total_correct_parts = 0
    total_correct_questions = 0

    for part_outputs in system_outputs:
        total_questions += 1
        correct_parts = 0
        for output in part_outputs:
            total_parts += 1
            if output["score"] == 1:
                correct_parts += 1
        total_correct_parts += correct_parts
        total_correct_questions += int(correct_parts == len(part_outputs))

    part_accuracy = total_correct_parts / total_parts
    question_accuracy = total_correct_questions / total_questions
    metrics = {
        "part_accuracy": part_accuracy,
        "question_accuracy": question_accuracy,
    }
    return metrics
