"""
Custom exceptions for workflow validation and execution errors.

This module defines the exception hierarchy for the workflows package, enabling
specific error types to be raised and caught during workflow validation and execution.
Each exception provides detailed error messages to help diagnose and fix issues in
workflow definitions or execution.

Exception hierarchy:
- WorkflowError (base class)
  - UnknownVariableError (missing variable reference)
  - CyclicDependencyError (circular dependencies)
  - FunctionNotFoundError (missing function reference)
"""


# Define custom exceptions for workflow errors
class WorkflowError(Exception):
    """
    Base exception class for all workflow-related errors.

    This is the parent class for all workflow-specific exceptions and can be used
    to catch any error from the workflows package.
    """

    pass


class UnknownVariableError(WorkflowError):
    """
    Raised when a workflow step references a variable that doesn't exist.

    This typically occurs when a step's input field references a variable that is neither
    provided as an external input nor produced as an output by any previous step.
    """

    def __init__(self, var: str):
        super().__init__(f"Unknown variable referenced: {var}")


class CyclicDependencyError(WorkflowError):
    """
    Raised when a cyclic dependency is detected in a workflow.

    A cyclic dependency occurs when there is a circular reference in the workflow graph,
    such as step A depending on step B, which depends on step A. Such workflows cannot
    be executed because there's no valid order to process the steps.
    """

    def __init__(self):
        super().__init__("Cyclic dependency detected in workflow")


class FunctionNotFoundError(WorkflowError):
    """
    Raised when a referenced function cannot be found during workflow execution.

    This typically occurs when a step references a function that doesn't exist in
    the available function registry or namespace.
    """

    def __init__(self, func_name: str):
        super().__init__(f"Function not found: {func_name}")
