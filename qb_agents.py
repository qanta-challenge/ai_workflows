import time
from typing import Any, Iterable, TypedDict

from loguru import logger

from .executors import WorkflowOutput, execute_workflow
from .structs import TossupWorkflow, Workflow


def _get_workflow_response(
    workflow: Workflow, available_vars: dict[str, Any], logprob_step: bool | str = False
) -> tuple[WorkflowOutput, float]:
    """Get response from executing a complete workflow."""
    start_time = time.time()
    workflow_output = execute_workflow(workflow, available_vars, return_full_content=True, logprob_step=logprob_step)
    response_time = time.time() - start_time
    return workflow_output, response_time


class TossupResult(TypedDict):
    answer: str  # the model's best guess
    confidence: float  # confidence score (0-1)
    logprob: float | None  # log probability of the answer
    buzz: bool  # whether the agent buzzed
    question_fragment: str  # prefix of the question text so far
    position: int  # 1-indexed question run index
    step_contents: list[str]  # string content outputs of each step
    response_time: float
    step_outputs: dict[str, Any]


class BonusResult(TypedDict):
    answer: str
    confidence: float
    explanation: str
    response_time: float
    step_contents: list[str]
    step_outputs: dict[str, Any]


class QuizBowlTossupAgent:
    """Agent for handling tossup questions with multiple steps in the workflow."""

    external_input_variable = "question_text"
    output_variables = ["answer", "confidence"]

    def __init__(self, workflow: TossupWorkflow):
        """Initialize the multi-step tossup agent.

        Args:
            workflow: The workflow containing multiple steps
            buzz_threshold: Confidence threshold for buzzing
        """
        self.workflow = workflow
        self.output_variables = list(workflow.outputs.keys())

        # Validate input variables
        if self.external_input_variable not in workflow.inputs:
            raise ValueError(f"External input variable {self.external_input_variable} not found in workflow inputs")

        # Validate output variables
        for out_var in self.output_variables:
            if out_var not in workflow.outputs:
                raise ValueError(f"Output variable {out_var} not found in workflow outputs")

    def _single_run(self, question_run: str, position: int) -> TossupResult:
        """Process a single question run.
        Args:
            question_run: The question run to process
            position: The position of the question run

        Returns:
            A TossupResult containing the answer, confidence, logprob, buzz, question fragment, position, step contents, response time, and step outputs
        """
        answer_var_step = self.workflow.outputs["answer"].split(".")[0]
        workflow_output, response_time = _get_workflow_response(
            self.workflow, {self.external_input_variable: question_run}, logprob_step=answer_var_step
        )
        final_outputs = workflow_output["final_outputs"]
        buzz = self.workflow.buzzer.run(final_outputs["confidence"], logprob=workflow_output["logprob"])
        result: TossupResult = {
            "position": position,
            "answer": final_outputs["answer"],
            "confidence": final_outputs["confidence"],
            "logprob": workflow_output["logprob"],
            "buzz": buzz,
            "question_fragment": question_run,
            "step_contents": workflow_output["step_contents"],
            "step_outputs": workflow_output["intermediate_outputs"],  # Include intermediate step outputs
            "response_time": response_time,
        }
        return result

    def run(self, question_runs: list[str], early_stop: bool = True) -> Iterable[TossupResult]:
        """Process a tossup question and decide when to buzz based on confidence.

        Args:
            question_runs: Progressive reveals of the question text
            early_stop: Whether to stop after the first buzz

        Yields:
            Dict containing:
                - answer: The model's answer
                - confidence: Confidence score
                - buzz: Whether to buzz
                - question_fragment: Current question text
                - position: Current position in question
                - step_contents: String content outputs of each step
                - response_time: Time taken for response
                - step_outputs: Outputs from each step
        """
        for i, question_text in enumerate(question_runs):
            # Execute the complete workflow
            result = self._single_run(question_text, i + 1)

            yield result

            # If we've reached the confidence threshold, buzz and stop
            if early_stop and result["buzz"]:
                if i + 1 < len(question_runs):
                    yield self._single_run(question_runs[-1], len(question_runs))
                return


class QuizBowlBonusAgent:
    """Agent for handling bonus questions with multiple steps in the workflow."""

    external_input_variables = ["leadin", "part"]
    output_variables = ["answer", "confidence", "explanation"]

    def __init__(self, workflow: Workflow):
        """Initialize the multi-step bonus agent.

        Args:
            workflow: The workflow containing multiple steps
        """
        self.workflow = workflow
        self.output_variables = list(workflow.outputs.keys())

        # Validate input variables
        for input_var in self.external_input_variables:
            if input_var not in workflow.inputs:
                raise ValueError(f"External input variable {input_var} not found in workflow inputs")

        # Validate output variables
        for out_var in self.output_variables:
            if out_var not in workflow.outputs:
                raise ValueError(f"Output variable {out_var} not found in workflow outputs")

    def run(self, leadin: str, part: str) -> BonusResult:
        """Process a bonus part with the given leadin.

        Args:
            leadin: The leadin text for the bonus question
            part: The specific part text to answer

        Returns:
            Dict containing:
                - answer: The model's answer
                - confidence: Confidence score
                - explanation: Explanation for the answer
                - step_contents: String content outputs of each step
                - response_time: Time taken for response
                - step_outputs: Outputs from each step
        """
        workflow_output, response_time = _get_workflow_response(
            self.workflow,
            {
                "leadin": leadin,
                "part": part,
            },
        )
        final_outputs = workflow_output["final_outputs"]
        return {
            "answer": final_outputs["answer"],
            "confidence": final_outputs["confidence"],
            "explanation": final_outputs["explanation"],
            "step_contents": workflow_output["step_contents"],
            "response_time": response_time,
            "step_outputs": workflow_output["intermediate_outputs"],  # Include intermediate step outputs
        }


# Example usage
if __name__ == "__main__":
    # Load the Quizbowl dataset
    from datasets import load_dataset

    from shared.workflows.factory import create_quizbowl_bonus_workflow, create_quizbowl_tossup_workflow

    ds_name = "qanta-challenge/leaderboard_co_set"
    ds = load_dataset(ds_name, split="train")

    # Create the agents with multi-step workflows
    tossup_workflow = create_quizbowl_tossup_workflow()
    tossup_agent = QuizBowlTossupAgent(workflow=tossup_workflow, buzz_threshold=0.9)

    bonus_workflow = create_quizbowl_bonus_workflow()
    bonus_agent = QuizBowlBonusAgent(workflow=bonus_workflow)

    # Example for tossup mode
    print("\n=== TOSSUP MODE EXAMPLE ===")
    sample_question = ds[30]
    print(sample_question["question_runs"][-1])
    print(sample_question["gold_label"])
    print()
    question_runs = sample_question["question_runs"]

    results = tossup_agent.run(question_runs, early_stop=True)
    for result in results:
        print(result["step_contents"])
        print(f"Guess at position {result['position']}: {result['answer']}")
        print(f"Confidence: {result['confidence']}")
        print("Step outputs:", result["step_outputs"])
        if result["buzz"]:
            print("Buzzed!\n")

    # Example for bonus mode
    print("\n=== BONUS MODE EXAMPLE ===")
    sample_bonus = ds[31]  # Assuming this is a bonus question
    leadin = sample_bonus["leadin"]
    parts = sample_bonus["parts"]

    print(f"Leadin: {leadin}")
    for i, part in enumerate(parts):
        print(f"\nPart {i + 1}: {part['part']}")
        result = bonus_agent.run(leadin, part["part"])
        print(f"Answer: {result['answer']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Explanation: {result['explanation']}")
        print(f"Response time: {result['response_time']:.2f}s")
        print("Step outputs:", result["step_outputs"])
