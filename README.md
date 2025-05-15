# Workflows Subpackage

This subpackage provides a framework for defining, validating, and executing workflows composed of interconnected model steps with dependency management.

## Overview

The workflows subpackage enables the creation and execution of workflows where multiple model steps can be combined, with outputs from earlier steps feeding into inputs of later steps. The package handles dependency resolution, execution order, and error handling.

## Components

### `structs.py`

Contains the core data structures used throughout the workflow system:

- `InputField`: Represents an input field with name, description, and variable reference
- `OutputField`: Represents an output field with name, type, and description
- `ModelStep`: Represents a single step in a workflow with input fields, output fields, and model details
- `Workflow`: A collection of ModelSteps with their identifiers
- `TossupWorkflow`: Specialized workflow for quizbowl tossup questions with buzzing capability

### `configs.py`

Provides configuration settings and constants:

- `AVAILABLE_MODELS`: Supported model configurations from various providers
- `TYPE_MAP`: Mapping of supported field types to Python types
- `FUNCTION_MAP`: Built-in transformation functions for input/output processing

### `utils.py`

Provides utility functions for workflow operations:

- `create_dependency_graph`: Builds a dependency graph representing the execution order constraints
- `topological_sort`: Sorts steps in execution order based on their dependencies
- `detect_cycles`: Identifies cyclic dependencies in workflow definitions

### `executors.py`

Handles the execution of workflows:

- `execute_model_step`: Executes a single model step with input processing and output collection
- `execute_simple_workflow`: Handles single-step workflows
- `execute_multi_step_workflow`: Manages multi-step workflows with dependency resolution
- `execute_workflow`: Main entry point that routes to appropriate executor based on workflow complexity

### `validators.py`

Provides workflow validation functionality:

- `ValidationErrorType`: Enumeration of possible validation error types
- `WorkflowValidationError`: Base class for validation errors
- Validation functions for steps, DAGs, variables, and types

### `errors.py`

Defines custom exceptions for workflow-related errors:

- `WorkflowError`: Base class for workflow errors
- `CyclicDependencyError`: Raised when detecting cycles in the workflow graph
- `UnknownVariableError`: Raised when a step requires a variable that's not provided or produced

## Usage Example

```python
from shared.workflows.structs import InputField, ModelStep, OutputField, Workflow

# Define a workflow with two steps
step1 = ModelStep(
    id="step1",
    model="gpt-4o-mini",
    provider="OpenAI",
    call_type="llm",
    system_prompt="Step1 processing",
    input_fields=[InputField(name="value", description="Input value", variable="input.value")],
    output_fields=[OutputField(name="result", description="Processed result", type="str", func="upper")],
)

step2 = ModelStep(
    id="step2",
    model="gpt-4o-mini",
    provider="OpenAI",
    call_type="llm",
    system_prompt="Step2 processing",
    input_fields=[InputField(name="result", description="Result from step1", variable="step1.result")],
    output_fields=[OutputField(name="final", description="Final output", type="str", func="lower")],
)

workflow = Workflow(
    steps={"step1": step1, "step2": step2},
    inputs=["input.value"],
    outputs={"final": "step2.final"}
)

# Execute the workflow
from shared.workflows.executors import execute_workflow

result = execute_workflow(
    workflow=workflow,
    input_values={"input.value": "Hello, World!"},
    return_full_content=True,
    logprob_step="step2"
)

# Access results
final_output = result["final_outputs"]["final"]
intermediate_results = result["intermediate_outputs"]
step_contents = result["step_contents"]
logprob = result["logprob"]
```

## Error Handling

The workflows system provides robust error handling:

- Detects cyclic dependencies in workflow definitions
- Validates input/output variable references
- Ensures all required inputs are provided
- Supports custom validation rules through the validation system
- Provides detailed error messages for debugging

## Extending the Workflows System

To extend the workflows system:

1. Add new model step types by extending the `ModelStep` class
2. Create custom field types by extending validation in the execution logic
3. Implement additional error types in `errors.py` for specialized error handling
4. Add new transformation functions to `FUNCTION_MAP` in `configs.py`
5. Create specialized workflow types by extending the `Workflow` class 