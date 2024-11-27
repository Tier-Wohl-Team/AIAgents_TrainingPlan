import pytest
from agents.DistractionSpecialist import DistractionSpecialist
from states.state_types import BehaviorState


@pytest.mark.integration
def test_distraction_specialist_integration_contains_progression_plan():
    """Integration test: Ensure the LLM response contains a progression plan."""

    # Define a sample state with a behavior, current status, and goal
    state = BehaviorState(
        behavior="sit",
        status="dog stays sitting when trainer moves arms",
        goal="dog stays sitting when a ball is thrown"
    )

    # Call the action method to invoke the LLM
    result = DistractionSpecialist.action(state)

    # Assertions
    assert result["draft_plan"], "The LLM response is empty. Expected a progression plan."
    assert "progression" in result["draft_plan"].lower(), (
        "The LLM response does not contain a progression plan."
    )
    assert "status" in result["draft_plan"].lower(), (
        "The LLM response does not include the current status of the behavior."
    )
    assert "goal" in result["draft_plan"].lower(), (
        "The LLM response does not include the goal for the behavior."
    )
    assert "ball is thrown" in result["draft_plan"].lower(), (
        "The LLM response does not include the goal behavior."
    )

    # Print the result for debugging or manual inspection
    print("LLM Response:", result["draft_plan"])
