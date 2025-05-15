from collections import deque
from typing import Any, Iterable

from .errors import CyclicDependencyError, UnknownVariableError, WorkflowError
from .structs import Workflow

"""
Utilities for workflow dependency management and execution order determination.

This module provides functions for analyzing workflows, determining dependencies between steps,
and calculating the correct execution order to ensure all dependencies are satisfied. 
Key functionality includes:

- Variable to step mapping: Identifying which step produces each variable
- Dependency graph creation: Building a graph representing dependencies between steps
- Topological sorting: Determining a valid execution order based on dependencies
- Cycle detection: Identifying cyclic dependencies that would prevent execution

These utilities form the foundation for workflow validation and execution in the 
workflow_executor module.
"""


def _create_variable_step_mapping(workflow: Workflow) -> dict[str, str]:
    """
    Creates a mapping from produced variable names to the model step that produces them.

    Args:
        workflow (Workflow): The workflow containing steps and their input/output fields.

    Returns:
        dict[str, str]: A dictionary where keys are variable names (formatted as "{step_id}.{output name}")
                        and values are the step IDs that produce them.

    Raises:
        WorkflowError: If there are duplicate step IDs or if a variable is produced by multiple steps.

    Example:
        For a workflow with steps "extract" and "summarize" each producing outputs:
        >>> _create_variable_step_mapping(workflow)
        {'extract.keywords': 'extract', 'summarize.summary': 'summarize'}
    """
    variable_step_map: dict[str, str] = {}  # variable name -> step id
    for step_id, step in workflow.steps.items():
        for output in step.output_fields:
            var_name = f"{step_id}.{output.name}"
            if var_name in variable_step_map:
                raise WorkflowError(f"Variable '{output.name}' has duplicate entry in step {step_id}")
            variable_step_map[var_name] = step_id
    return variable_step_map


def create_dependency_graph(workflow: Workflow, input_values: dict[str, Any]) -> dict[str, set[str]]:
    """
    Creates a dependency graph from a workflow.

    This function analyzes the workflow and determines which steps depend on others
    based on their input/output relationships. A step depends on another if it requires
    a variable that is produced by the other step. External inputs provided through
    input_values don't create dependencies.

    Args:
        workflow (Workflow): The workflow containing steps and their input/output fields.
        input_values (dict[str, Any]): A dictionary of external input values provided to the workflow.

    Returns:
        dict[str, set[str]]: A dictionary where keys are step IDs and values are sets of step IDs
                             that the key step depends on.

    Raises:
        UnknownVariableError: If an input field references a variable that is not provided
                              externally nor produced by any step.

    Example:
        For a workflow where step "classify" depends on output from "extract":
        >>> create_dependency_graph(workflow, {})
        {'extract': set(), 'classify': {'extract'}}

        With external input provided for "text" variable:
        >>> create_dependency_graph(workflow, {'text': 'Sample text'})
        {'extract': set(), 'classify': {'extract'}}
    """
    produced_by = _create_variable_step_mapping(workflow)
    dependencies: dict[str, set[str]] = {step_id: set() for step_id in workflow.steps}
    for step_id, step in workflow.steps.items():
        for input_field in step.input_fields:
            var = input_field.variable
            # If the variable was provided externally, then no dependency is needed.
            if var in input_values:
                continue
            # Otherwise, check if the variable is produced by a step.
            if var in produced_by:
                producer_step_id = produced_by[var]
                if producer_step_id != step_id:  # Avoid self-dependency
                    dependencies[step_id].add(producer_step_id)
            else:
                raise UnknownVariableError(f"Variable '{var}' is not provided externally nor produced by any step")
    return dependencies


def detect_cycles(dep_graph: dict[str, Iterable[str]]) -> str | None:
    """Detects cycles in the dependency graph.
    Args:
        dep_graph: A dictionary where the keys are node IDs and the values are the dependent node IDs
    Returns:
        The first step id of a model_step that is part of a cycle, None if no cycles are found
    """
    # Check for cycles in step dependencies
    visited = set()
    path = set()

    def has_cycle(node: str) -> bool:
        if node in path:
            return True
        if node in visited:
            return False

        visited.add(node)
        path.add(node)

        for neighbor in dep_graph.get(node, set()):
            if has_cycle(neighbor):
                return True

        path.remove(node)
        return False

    # Check each step for cycles
    for node_id in dep_graph:
        if has_cycle(node_id):
            return node_id
    return None


def topological_sort(dependencies: dict[str, set[str]]) -> list[str]:
    """
    Performs a topological sort on a dependency graph and detects cycles using Kahn's algorithm.

    A topological sort orders the steps such that for every dependency from step A to step B,
    step A comes before step B in the ordering. This ensures that all dependencies are satisfied
    when executing steps in the returned order.

    Args:
        dependencies (dict[str, set[str]]): A dictionary where each key is a node identifier and
                                            each value is a set of nodes that the key node depends on.

    Returns:
        list[str]: A list representing the nodes in topological order if no cycle is detected.

    Raises:
        CyclicDependencyError: If a cycle is detected in the graph.

    Example:
        >>> topological_sort({'A': set(), 'B': {'A'}, 'C': {'B'}})
        ['A', 'B', 'C']

        >>> topological_sort({'A': {'B'}, 'B': {'A'}})  # Cyclic dependency
        CyclicDependencyError

    Algorithm:
        This implementation uses Kahn's algorithm:
        1. Calculate in-degree for all nodes (number of dependencies)
        2. Start with nodes having 0 in-degree (no dependencies)
        3. Process each node by removing its outgoing edges
        4. Add newly dependency-free nodes to the processing queue
        5. If not all nodes are processed, a cycle exists
    """

    nodes = list(dependencies.keys())
    dependents: dict[str, list[str]] = {node: [] for node in nodes}
    in_degree: dict[str, int] = dict.fromkeys(nodes, 0)

    # Calculate in-degrees and build dependents list
    for node, deps in dependencies.items():
        in_degree[node] = len(deps)
        for dep in deps:
            dependents[dep].append(node)

    # Initialize queue with nodes having zero in-degree
    queue = deque([node for node, deg in in_degree.items() if deg == 0])
    execution_order: list[str] = []

    # Process nodes in topological order
    while queue:
        current = queue.popleft()
        execution_order.append(current)
        for dep in dependents[current]:
            in_degree[dep] -= 1
            if in_degree[dep] == 0:
                queue.append(dep)

    # If execution order includes all nodes, no cycle exists
    if len(execution_order) != len(nodes):
        raise CyclicDependencyError()
    return execution_order
