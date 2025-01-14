import json

import pytest
from unittest.mock import patch, MagicMock
from agents.DogFeatureInteractionAgent import DogFeatureInteractionAgent
from states.state_types import TeamState, BehaviorResearchState


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
    state = BehaviorResearchState(
        question="sit",
        outline_plan="Step 1: Lure the dog into a sitting position\nStep 2: Reward the dog with a treat",
        dog_details=""
    )

    # Mock input() to avoid blocking user interaction
    with patch("builtins.input", return_value="My dog loves chicken treats!"):
        # Call the action method
        response = DogFeatureInteractionAgent.action(state)

    # Extract the result
    new_dog_details = response.get("new_dog_details", [])
    assert len(new_dog_details) > 0, "No questions were processed."

    # Check that each detail entry is a tuple with a question and an answer
    for detail in new_dog_details:
        assert isinstance(detail, tuple), f"Expected tuple, got {type(detail)}"
        assert len(detail) == 2, f"Expected tuple with 2 elements, got {len(detail)}"
        question, answer = detail
        assert isinstance(question, str), f"Expected question to be a string, got {type(question)}"
        assert isinstance(answer, str), f"Expected answer to be a string, got {type(answer)}"

    # Verify the mock LLM was called
    mock_llm.invoke.assert_called_once()
