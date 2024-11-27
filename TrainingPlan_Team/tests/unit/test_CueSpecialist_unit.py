import pytest
from unittest.mock import patch, MagicMock
from agents.CueSpecialist import CueSpecialist
from states.state_types import BehaviorState


@pytest.mark.unit
@patch("agents.CueSpecialist.CueSpecialist.LLM", autospec=True)
def test_cue_specialist_action(mock_llm):
    """Test the CueSpecialist action method."""
    # Mock the LLM response
    mock_llm_response = MagicMock()
    mock_llm_response.content = "Mocked draft training plan."
    mock_llm.invoke.return_value = mock_llm_response

    # Define a sample state
    state = BehaviorState(
        behavior="sit",
        status="dog sits reliably when lured",
        goal="dog sits reliably on verbal cue"
    )

    # Call the action method
    result = CueSpecialist.action(state)

    # Assertions
    assert result["draft_plan"] == "Mocked draft training plan.", "The draft plan content is incorrect."
    mock_llm.invoke.assert_called_once()

    # Ensure the correct prompt was sent
    messages = mock_llm.invoke.call_args[0][0]
    assert len(messages) == 2, "Expected two messages in the prompt."
    assert "You are an experienced dog trainer" in messages[0].content, "The background story is missing or incorrect."
    assert "Please write a training plan to put the behaviour sit under cue control." in messages[1].content
    assert "dog sits reliably when lured" in messages[1].content, "The current status is missing in the prompt."
    assert "dog sits reliably on verbal cue" in messages[1].content, "The goal is missing in the prompt."


