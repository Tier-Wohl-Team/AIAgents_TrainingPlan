import pytest
from unittest.mock import patch, MagicMock
from agents.DistractionSpecialist import DistractionSpecialist
from states.state_types import BehaviorState


@pytest.mark.unit
@patch("agents.DistractionSpecialist.DistractionSpecialist.LLM", autospec=True)
def test_distraction_specialist_action(mock_llm):
    """Test the DistractionSpecialist action method with a valid response."""
    # Mock the LLM response
    mock_llm_response = MagicMock()
    mock_llm_response.content = "Mocked draft progression plan."
    mock_llm.invoke.return_value = mock_llm_response

    # Define a sample state
    state = BehaviorState(
        behavior="sit",
        status="dog stays sitting when trainer moves arms",
        goal="dog stays sitting when a ball is thrown"
    )

    # Call the action method
    result = DistractionSpecialist.action(state)

    # Assertions
    assert result["draft_plan"] == "Mocked draft progression plan.", "The draft plan content is incorrect."
    mock_llm.invoke.assert_called_once()

    # Ensure the correct prompt was sent
    messages = mock_llm.invoke.call_args[0][0]
    assert len(messages) == 2, "Expected two messages in the prompt."
    assert "You are an experienced dog trainer" in messages[0].content, "The background story is missing or incorrect."
    assert "Please write a training plan to assure that the behaviour sit can be performed under distractions." in messages[1].content
    assert "dog stays sitting when trainer moves arms" in messages[1].content, "The current status is missing in the prompt."
    assert "dog stays sitting when a ball is thrown" in messages[1].content, "The goal is missing in the prompt."

