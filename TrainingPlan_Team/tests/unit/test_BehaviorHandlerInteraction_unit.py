import json

import pytest
from unittest.mock import patch, MagicMock
from agents.DogFeatureInteractionAgent import DogFeatureInteractionAgent
from states.state_types import TeamState

@pytest.mark.unit
@patch('agents.DogFeatureInteractionAgent.DogFeatureInteractionAgent.LLM', autospec=True)  # Mock the LLM class
def test_action_mock(mock_llm):
    # Configure the mocked LLM to have an `invoke` method
    fake_response_content = {
        "questions": [
            "Does your dog have any hip issues?",
            "What type of treats does your dog prefer?"
        ]
    }
    mock_llm.invoke.return_value.content = json.dumps(fake_response_content)

    # Mock state
    state = TeamState(
        question="sit",
        outline_plan="Step 1: Lure the dog into a sitting position\nStep 2: Reward the dog with a treat"
    )

    # Mock input() to avoid blocking user interaction
    with patch("builtins.input", return_value="My dog loves chicken treats!"):
        # Call the action method
        response = DogFeatureInteractionAgent.action(state)

    # Extract the result
    dog_details = response.get("dog_details", [])
    assert len(dog_details) > 0, "No questions were processed."

    # Check that each detail entry has a query and answer
    for detail in dog_details:
        assert "query" in detail, f"Expected 'query' key in detail entry, got {detail}"
        assert "answer" in detail, f"Expected 'answer' key in detail entry, got {detail}"

    # Verify the mock LLM was called
    mock_llm.invoke.assert_called_once()
