import pytest
from unittest.mock import patch, MagicMock
from agents.DogFeatureInteractionAgent import DogFeatureInteractionAgent
from states.state_types import TeamState

class MockResponse:
    def __init__(self, tool_calls):
        self.tool_calls = tool_calls

@pytest.fixture
def sample_state():
    return TeamState({
        "question": "sit",
        "outline_plan": "1. Lure the dog into a sitting position\n2. Reward the dog when it sits"
    })

@pytest.mark.unit
def test_action(sample_state):
    # Create a real object to simulate the LLM response
    mock_response = MockResponse(tool_calls=[
        {"name": "handler_input", "args": {"query": "What is your dog's breed?"}}
    ])

    # Create a mock LLM
    mock_llm = MagicMock()
    # The call chain is: DogFeatureInteractionAgent.LLM.bind_tools(...) -> llm.invoke(...)
    # So we need to set mock_llm.bind_tools().invoke() to return our mock_response.
    mock_llm.bind_tools.return_value.invoke.return_value = mock_response

    with patch.object(DogFeatureInteractionAgent, 'LLM', mock_llm):
        # Mock input response
        with patch('builtins.input', return_value='Labrador'):
            result = DogFeatureInteractionAgent.action(sample_state)

    assert "dog_details" in result
    assert len(result["dog_details"]) == 1
    assert result["dog_details"][0]["query"] == "What is your dog's breed?"
    assert result["dog_details"][0]["answer"] == "Labrador"
