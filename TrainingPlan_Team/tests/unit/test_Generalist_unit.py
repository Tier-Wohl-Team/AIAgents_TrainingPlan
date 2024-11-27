import pytest
from unittest.mock import patch, MagicMock
from agents.Generalist import Generalist
from states.state_types import BehaviorState


@pytest.mark.unit
@patch("agents.Generalist.Generalist.LLM", autospec=True)
def test_generalist_action_valid_response(mock_llm):
    """Test the Generalist action method with a valid response."""
    # Mock the LLM response
    mock_llm_response = MagicMock()
    mock_llm_response.content = "Mocked draft progression plan for the behavior."
    mock_llm.invoke.return_value = mock_llm_response

    # Define a sample state
    state = BehaviorState(
        behavior="fetch",
        status="dog retrieves the ball but drops it halfway back",
        goal="dog retrieves the ball and delivers it to the trainer's hand"
    )

    # Call the action method
    result = Generalist.action(state)

    # Assertions
    assert result["draft_plan"] == "Mocked draft progression plan for the behavior.", (
        "The draft plan content is incorrect."
    )
    mock_llm.invoke.assert_called_once()

    # Ensure the correct prompt was sent
    messages = mock_llm.invoke.call_args[0][0]
    assert len(messages) == 2, "Expected two messages in the prompt."
    assert "You are an experienced dog trainer" in messages[0].content, "The background story is missing or incorrect."
    assert "Please write a training plan for the behaviour fetch." in messages[1].content
    assert "dog retrieves the ball but drops it halfway back" in messages[1].content, (
        "The current status is missing in the prompt."
    )
    assert "dog retrieves the ball and delivers it to the trainer's hand" in messages[1].content, (
        "The goal is missing in the prompt."
    )
