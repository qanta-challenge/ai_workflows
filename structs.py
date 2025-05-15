# %%
from copy import deepcopy
from enum import Enum
from typing import Any, Literal, Optional

import numpy as np
from pydantic import BaseModel, Field, model_validator

from .configs import AVAILABLE_MODELS

"""
Core data structures for defining workflows and their components.

This module defines the primary classes used to model workflows, steps, and their
input/output fields. These data structures serve as the foundation for workflow
definition, validation, and execution throughout the workflows package.

The primary components are:
- InputField: Represents an input to a model step with name and source variable
- OutputField: Represents an output from a model step with name and type
- ModelStep: Represents a single step in a workflow with inputs and outputs
- Workflow: A collection of interconnected steps with defined inputs and outputs

All classes use Pydantic's BaseModel for validation and serialization support.
"""
FieldType = Literal["input", "output"]


SUPPORTED_TYPES = Literal["str", "int", "float", "bool", "list[str]", "list[int]", "list[float]", "list[bool]"]
"""Supported field types for input and output fields"""


class InputField(BaseModel):
    """
    Defines an input field for a model step.

    An input field specifies what data a step requires, where it comes from,
    and optional pre-processing to apply before use.

    Attributes:
        name: The name of the input field within the step's context
        description: Human-readable description of the input's purpose
        variable: Reference to the source variable (format: "{step_id}.{field_name}" or external input name)
        func: Optional function name to transform the input value before use
    """

    name: str
    description: str
    variable: str

    # function to call on the input before passing it to the model
    func: str | None = None

    class Config:
        frozen = True


class OutputField(BaseModel):
    """
    Defines an output field produced by a model step.

    An output field specifies a value that the step will produce, including
    its data type and optional post-processing.

    Attributes:
        name: The name of the output field within the step's context
        description: Human-readable description of the output's purpose
        type: The data type of the output (one of SUPPORTED_TYPES)
        func: Optional function name to transform the raw output value
    """

    name: str
    type: SUPPORTED_TYPES = Field(default="str")
    description: str

    # function to call on the output string from the model
    func: str | None = None

    class Config:
        frozen = True


class CallType(str, Enum):
    LLM = "llm"
    SEARCH = "search"
    PYTHON_FUNC = "python_func"


class ModelStep(BaseModel):
    """
    Represents a single step in a workflow.

    A model step encapsulates the details of a specific operation within a workflow,
    including what model to use, what inputs it requires, and what outputs it produces.

    Attributes:
        id: Unique identifier for this step within a workflow
        model: The model to use for this step (e.g., "gpt-4")
        provider: The provider of the model (e.g., "openai")
        call_type: The type of operation (e.g., "llm", "search")
        system_prompt: Instructions for the model
        input_fields: List of input fields required by this step
        output_fields: List of output fields produced by this step
    """

    id: str
    name: str
    model: str
    provider: str
    call_type: CallType = CallType.LLM

    # TODO: Validate that this is not None for call_type = llm
    temperature: Optional[float] = None

    system_prompt: str
    input_fields: list[InputField]
    output_fields: list[OutputField]

    class Config:
        use_enum_values = True

    def fields(self, field_type: FieldType) -> list[InputField | OutputField]:
        return self.input_fields if field_type == "input" else self.output_fields

    def get_full_model_name(self) -> str:
        return f"{self.provider}/{self.model}"

    def get_produced_variables(self) -> list[str]:
        return [f"{self.id}.{field.name}" for field in self.output_fields if field.name]

    def update(self, update: dict[str, Any]) -> "ModelStep":
        """Returns a new copy with the updated properties."""
        return self.model_copy(update=update)

    def update_property(self, field: str, value: Any) -> "ModelStep":
        "Update the `field` key of the model step with `value`."
        return self.update({field: value})

    def update_field(self, field_type: FieldType, index: int, key: str, value: str) -> "ModelStep":
        """Update a specific field of an input or output field at the given index."""
        if field_type == "input":
            fields = self.input_fields
        elif field_type == "output":
            fields = self.output_fields
        else:
            raise ValueError(f"Invalid field type: {field_type}")

        if index < len(fields):
            fields[index] = fields[index].model_copy(update={key: value})
        return self.model_copy()

    @staticmethod
    def create_new_field(field_type: FieldType, input_var: str | None = None) -> InputField | OutputField:
        if field_type == "input":
            return InputField(name="", description="", variable=input_var)
        elif field_type == "output":
            return OutputField(name="", description="")
        else:
            raise ValueError(f"Invalid field type: {field_type}")

    def add_field(self, field_type: FieldType, index: int = -1, input_var: str | None = None) -> "ModelStep":
        """Add a new field to the state and update visibility.

        Args:
            field_type: Type of field to add ('input' or 'output').
            index: Position to insert the new field (-1 to append).
        Returns:
            A new ModelStep with the updated fields.
        """
        if field_type == "input":
            fields = deepcopy(self.input_fields)
            new_field = ModelStep.create_new_field(field_type, input_var)
            fields.insert(index + 1, new_field) if index != -1 else fields.append(new_field)
            return self.model_copy(update={"input_fields": fields})
        else:
            fields = deepcopy(self.output_fields)
            new_field = ModelStep.create_new_field(field_type)
            fields.insert(index + 1, new_field) if index != -1 else fields.append(new_field)
            return self.model_copy(update={"output_fields": fields})

    def delete_field(self, field_type: FieldType, index: int) -> "ModelStep":
        """
        Delete an input or output field from the state and update visibility.

        Args:
            field_type: Type of field to delete ('input' or 'output').
            index: Index of the field to delete. [-1 to delete the last field]

        Returns:
            A new ModelStep with the updated fields.
        """
        fields = self.input_fields if field_type == "input" else self.output_fields
        fields = deepcopy(fields)
        fields.pop(index)
        return self.model_copy(update={"input_fields": fields} if field_type == "input" else {"output_fields": fields})


class Workflow(BaseModel):
    """
    Represents a complete workflow composed of interconnected steps.

    A workflow defines a directed acyclic graph of model steps, where outputs
    from earlier steps can be used as inputs to later steps.

    Attributes:
        inputs: List of input variables required by the workflow
        outputs: List of output variables produced by the workflow
        steps: Dictionary mapping step IDs to ModelStep instances

    The inputs and outputs lists use the format "{step_id}.{field_name}"
    to uniquely identify variables within the workflow.
    """

    # variables of form {node}.{field}
    inputs: list[str] = Field(default_factory=list)

    # variables of form {node}.{field}
    outputs: dict[str, str | None] = Field(default_factory=dict)
    steps: dict[str, ModelStep] = Field(default_factory=dict)

    def model_dump(self, *args, **kwargs):
        data = super().model_dump(*args, **kwargs)
        if "steps" in data:
            data["steps"] = list(data["steps"].values())
        return data

    @model_validator(mode="before")
    def dictify_steps(cls, data):
        if "steps" in data and isinstance(data["steps"], list):
            steps_dict = {}
            for step in data["steps"]:
                if isinstance(step, ModelStep):
                    step_id = step.id
                else:
                    step_id = step["id"]
                if step_id in steps_dict:
                    raise ValueError(f"Duplicate step ID: {step_id}")
                steps_dict[step_id] = step
            data["steps"] = steps_dict
        return data

    def get_step_variables(self, step_id: str) -> list[str]:
        """Get all variables from a specific step."""
        step = self.steps[step_id]
        variables = []
        for output in step.output_fields:
            if output.name == "":
                continue
            output_var = f"{step.id}.{output.name}"
            variables.append(output_var)
        return variables

    def get_available_variables(self) -> list[str]:
        """Get all output variables from all steps."""
        variables = set(self.inputs)
        for step in self.steps.values():
            variables.update(self.get_step_variables(step.id))
        return list(variables)

    def get_step_model_selections(self) -> dict[str, str]:
        """Get all model selections for all steps."""
        return {step_id: step.get_full_model_name() for step_id, step in self.steps.items()}

    def get_output_model_selections(self) -> dict[str, str]:
        """Get all output model selections for all steps."""
        return {
            output_var: target_var.split(".")[0] if target_var else None
            for output_var, target_var in self.outputs.items()
        }

    # Step update method

    def add_step(self, step: ModelStep) -> "Workflow":
        """Add a step to the workflow."""
        steps = self.steps | {step.id: step}
        return self.model_copy(update={"steps": steps})

    def remove_step(self, step_id: str) -> "Workflow":
        """Remove a step from the workflow."""
        self.steps.pop(step_id)
        workflow = self.model_copy(update={"steps": self.steps})
        workflow.refresh_output_variables()
        return workflow

    def update_step(self, step: ModelStep) -> "Workflow":
        """Update a step in the workflow."""
        self.steps[step.id] = step
        steps = self.steps | {step.id: step}
        workflow = self.model_copy(update={"steps": steps})
        workflow.refresh_output_variables()
        return workflow

    # Output variables
    def refresh_output_variables(self) -> "Workflow":
        """Refresh the output variables for the workflow."""
        produced_variables = self.get_available_variables()
        self.outputs = {k: (v if v in produced_variables else None) for k, v in self.outputs.items()}
        return self


class BuzzerMethod(str, Enum):
    AND = "AND"
    OR = "OR"


class Buzzer(BaseModel):
    """Configuration for when to buzz in a tossup question."""

    method: BuzzerMethod = BuzzerMethod.AND  # Logic to combine thresholds
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)  # Minimum confidence to trigger a buzz
    prob_threshold: float | None = None  # Optional log probability threshold

    class Config:
        use_enum_values = True
        frozen = True

    def update(self, **kwargs) -> "Buzzer":
        """Update the buzzer with the given kwargs."""
        return self.model_copy(update=kwargs)

    def run(self, confidence: float, prob: float | None = None, logprob: float | None = None) -> bool:
        """Run the buzzer logic."""
        if logprob is not None and prob is not None:
            raise ValueError("Cannot provide both logprob and prob")
        if self.prob_threshold is None:
            return confidence >= self.confidence_threshold
        if logprob is None and prob is None:
            raise ValueError("Must provide either logprob or prob if prob_threshold is not None")
        prob = prob or float(np.exp(logprob))
        if self.method == BuzzerMethod.AND:
            return confidence >= self.confidence_threshold and prob >= self.prob_threshold
        elif self.method == BuzzerMethod.OR:
            return confidence >= self.confidence_threshold or prob >= self.prob_threshold
        else:
            raise ValueError(f"Invalid buzzer method: {self.method}")

    @model_validator(mode="after")
    def validate_method_with_log_prob(cls, data):
        """Validate that if prob_threshold is None, method must be 'and'."""
        if data.prob_threshold is None and data.method != BuzzerMethod.AND:
            raise ValueError("If prob_threshold is None, method must be 'and'")
        return data


class TossupWorkflow(Workflow):
    """Workflow specialized for tossup questions with buzzing capability."""

    buzzer: Buzzer = Field(default_factory=Buzzer)

    def get_answer_model(self, answer_var: str | None = None) -> str | None:
        answer_var = answer_var or self.outputs["answer"]
        if answer_var is None:
            return None
        step_id = answer_var.split(".")[0]
        return self.steps[step_id].get_full_model_name()

    def is_token_probs_supported(self, answer_var: str | None = None) -> bool:
        model_name = self.get_answer_model(answer_var)
        if model_name is None:
            return True
        return AVAILABLE_MODELS[model_name].get("logprobs", False)

    def update_buzzer(self, buzzer: Buzzer) -> "TossupWorkflow":
        """Update the buzzer."""
        return self.model_copy(update={"buzzer": buzzer})

    def refresh_buzzer(self) -> "TossupWorkflow":
        if not self.is_token_probs_supported():
            return self.update_buzzer(self.buzzer.update(prob_threshold=None, method="AND"))
        return self
