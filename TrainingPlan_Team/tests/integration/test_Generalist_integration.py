import pytest
from agents.Generalist import Generalist
from states.state_types import BehaviorState


@pytest.mark.integration
def test_generalist_integration_contains_progression_plan():
    """Integration test: Ensure the LLM response contains a progression plan."""

    # Define a sample state with a behavior, current status, and goal
    state = BehaviorState(
        behavior="fetch",
        status="dog retrieves the ball but drops it halfway back",
        goal="dog retrieves the ball and delivers it to the trainer's hand"
    )

    # Call the action method to invoke the LLM
    result = Generalist.action(state)

    # Assertions
    assert result["draft_plan"], "The LLM response is empty. Expected a progression plan."
    assert "progression" in result["draft_plan"].lower(), (
        "The LLM response does not contain a progression plan."
    )
    assert "dog retrieves the ball" in result["draft_plan"], (
        "The LLM response does not include the expected behavior."
    )
    assert "delivers it to the trainer's hand" in result["draft_plan"], (
        "The LLM response does not include the goal for the behavior."
    )

    # Print the result for debugging or manual inspection
    print("LLM Response:", result["draft_plan"])
