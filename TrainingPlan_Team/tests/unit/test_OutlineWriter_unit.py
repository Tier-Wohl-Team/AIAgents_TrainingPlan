import pytest
from unittest.mock import patch, MagicMock
from agents.OutlineWriter import OutlineWriter
from states.state_types import TeamState


@pytest.mark.unit
@patch('agents.OutlineWriter.OutlineWriter.LLM', autospec=True)  # Mock the entire LLM object
def test_action_mock(mock_llm):
    # Configure the mocked LLM to have an `invoke` method
    mock_answer = "You first have to get the behavior of sitting"
    mock_llm.invoke.return_value.content = mock_answer

    # Mock state
    state = TeamState(
        question="My dog sits for 10 seconds. I want to extend this duration to 25 seconds.",
    )

    # Call the action method
    response = OutlineWriter.action(state)

    # Assert the result
    assert response["outline_plan"] == mock_answer

    # Verify the mock was called
    mock_llm.invoke.assert_called_once()
