import pytest
from unittest.mock import patch
from agents.DogFeatureInteractionAgent import DogFeatureInteractionAgent
from states.state_types import TeamState

@pytest.mark.integration
def test_dog_feature_interaction_agent_integration():
    # Prepare a test state for a common, simple behavior
    test_state = TeamState(
        question="sit",
        outline_plan="Step 1: Lure the dog into a sitting position\nStep 2: Reward the dog with a treat"
    )

    # Mock input() so the test does not block for interactive input
    with patch("builtins.input", return_value="Handler test answer"):
        # Run the agent action method which internally calls the LLM
        result = DogFeatureInteractionAgent.action(test_state)

    # The result from action() is expected to be a dictionary with a "dog_details" key.
    # dog_details is a list of Q&A pairs that the agent collected from the user.
    # For there to be Q&A pairs, the LLM must have returned questions in JSON format.
    dog_details = result.get("dog_details", [])

    # If no questions were asked, it might mean the LLM did not return the expected JSON structure.
    assert len(dog_details) > 0, "No questions were returned by the LLM."

    # Each entry in dog_details should have a "query" and "answer"
    # The presence of these implies that the LLM returned a 'questions' key in JSON.
    for entry in dog_details:
        assert "query" in entry, "One of the returned details entries is missing 'query'."
        assert "answer" in entry, "One of the returned details entries is missing 'answer'."

    # If we reached this point, it means:
    # 1. The LLM returned content that could be parsed as JSON (otherwise action() would fail).
    # 2. The JSON contained 'questions' key with at least one question, which was asked to the handler.
    # Therefore, the integration test criteria are met.
