import pytest
from unittest.mock import patch

from agents.FinalPlanWriter import FinalPlanWriter


@pytest.mark.unit
@patch('agents.FinalPlanWriter.FinalPlanWriter.LLM', autospec=True)  # Mock the entire LLM object
def test_action_mock(mock_llm):
    # Configure the mocked LLM to have an `invoke` method
    mock_answer = "Final plan content"
    mock_llm.invoke.return_value.content = mock_answer

    # Mock state
    state = {
        "question": "How can I train my dog to stay?",
        "outline_plan": "Step 1: Basic stay command",
        "internet_research_results": ["Use treats", "Be consistent"],
        "handler_input": [("Why is this necessary?", "Helps reinforce the behavior")],
        "plans": [("Detailed Plan 1", "Detailed content")],
    }

    # Call the action method
    response = FinalPlanWriter.action(state)

    # Assert the result
    assert response["final_plan"] == mock_answer

    # Verify the mock was called
    mock_llm.invoke.assert_called_once()
