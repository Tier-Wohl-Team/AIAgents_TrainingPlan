import pytest
from states.state_types import BehaviorResearchState
from agents.OutlinePlanEvaluator import OutlinePlanEvaluator

@pytest.mark.integration
def test_outline_plan_evaluator_integration():
    # Prepare a test state with realistic data for the evaluation
    test_state = BehaviorResearchState(
        outline_plan="Step 1: Teach the dog to sit.\nStep 2: Teach the dog to stay.",
        new_dog_details="The dog is reactive to certain noises.",
        dog_details="The dog is a Border Collie with high energy.",
        iteration_count=0
    )

    # Call the action method of the OutlinePlanEvaluator
    result = OutlinePlanEvaluator.action(test_state)

    # Check that the result contains the expected keys
    assert "is_finished" in result, "The result does not include the 'is_finished' key."
    assert "iteration_count" in result, "The result does not include the 'iteration_count' key."
    assert "dog_details" in result, "The result does not include the 'dog_details' key."
    assert "new_dog_details" in result, "The result does not include the 'new_dog_details' key."

    # Validate the updated state
    assert result["iteration_count"] == 1, "The iteration count was not incremented correctly."
    assert "The dog is reactive to certain noises." in result["dog_details"], (
        "The 'new_dog_details' were not appended to 'dog_details'."
    )
    assert result["new_dog_details"] == [], "The 'new_dog_details' were not cleared after appending."

    # Validate the LLM response (it should be either "rewrite" or "no_rewrite")
    is_finished = result["is_finished"]
    assert isinstance(is_finished, bool), "'is_finished' should be a boolean value."
