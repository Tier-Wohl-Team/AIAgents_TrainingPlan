import pytest
from unittest.mock import patch
from agents.DogFeatureInteractionAgent import DogFeatureInteractionAgent
from states.state_types import BehaviorResearchState


@pytest.mark.integration
def test_dog_feature_interaction_agent_integration():
    # Prepare a test state for a common, simple behavior
    test_state = BehaviorResearchState(
        question="sit",
        outline_plan="Step 1: Lure the dog into a sitting position\nStep 2: Reward the dog with a treat",
        dog_details=""
    )

    # Mock input() so the test does not block for interactive input
    with patch("builtins.input", return_value="Handler test answer"):
        # Run the agent action method which internally calls the LLM
        result = DogFeatureInteractionAgent.action(test_state)

    new_dog_details = result.get("new_dog_details", [])

    # If no questions were asked, it might mean the LLM did not return the expected JSON structure.
    assert len(new_dog_details) > 0, "No questions were returned by the LLM."

    # Check that each detail entry is a tuple with a question and an answer
    for detail in new_dog_details:
        assert isinstance(detail, tuple), f"Expected tuple, got {type(detail)}"
        assert len(detail) == 2, f"Expected tuple with 2 elements, got {len(detail)}"
        question, answer = detail
        assert isinstance(question, str), f"Expected question to be a string, got {type(question)}"
        assert isinstance(answer, str), f"Expected answer to be a string, got {type(answer)}"
