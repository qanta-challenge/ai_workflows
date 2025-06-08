from ..configs import AVAILABLE_MODELS
from ..structs import ModelStep, Workflow


def get_step_tokens(step: ModelStep):
    prompt_tokens = len(step.system_prompt.split())
    output_format_tokens = 0
    for output_field in step.output_fields:
        output_format_tokens += len(output_field.name.split())
        output_format_tokens += len(output_field.description.split())
    for input_field in step.input_fields:
        if "." not in input_field.variable:
            prompt_tokens += 50
    return prompt_tokens + output_format_tokens


def get_step_cost(step: ModelStep):
    model_config = AVAILABLE_MODELS[step.full_name]
    input_cost = model_config["cost_per_million"] * get_step_tokens(step) / 1_000
    output_cost = model_config["cost_per_million"] * 1.5 * 50 / 1_000
    return input_cost + output_cost


def compute_workflow_cost(workflow: Workflow):
    # Compute the number of steps
    n_steps = len(workflow.steps)

    total_cost = 0
    total_tokens = 0
    for step_id, step in workflow.steps.items():
        total_cost += get_step_cost(step)
        total_tokens += get_step_tokens(step)

    return {
        "cost": total_cost,
        "n_tokens": total_tokens,
        "n_steps": n_steps,
    }
