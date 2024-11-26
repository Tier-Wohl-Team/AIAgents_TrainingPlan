import pytest
from unittest.mock import patch, MagicMock
from agents.WelfareSpecialist import WelfareSpecialist
from states.state_types import BehaviorState


@pytest.mark.unit
@patch("agents.WelfareSpecialist.WelfareSpecialist.LLM", autospec=True)
def test_welfare_specialist_action_plan_good(mock_llm):
    """Test the action method when the LLM responds with 'The plan is good.'."""
    # Mock the LLM response
    mock_llm_response = MagicMock()
    mock_llm_response.content = "The plan is good."
    mock_llm.invoke.return_value = mock_llm_response

    # Define a sample state
    state = BehaviorState(
        behavior="sit",
        draft_plan="Step 1: Use a treat to lure the dog into a sit position."
    )

    # Call the action method
    result = WelfareSpecialist.action(state)

    # Assertions
    assert result["welfare_review"] == "The plan is good.", "Expected feedback not returned."
    mock_llm.invoke.assert_called_once()

    # Ensure the correct prompt was sent
    messages = mock_llm.invoke.call_args[0][0]
    assert len(messages) == 2, "Expected two messages in the prompt."
    assert "expert in animal welfare regulations" in messages[0].content
    assert "Here is a training plan to teach a dog the behaviour" in messages[1].content
    assert "Step 1: Use a treat to lure the dog into a sit position." in messages[1].content


@pytest.mark.unit
@patch("agents.WelfareSpecialist.WelfareSpecialist.LLM", autospec=True)
def test_welfare_specialist_action_plan_needs_feedback(mock_llm):
    """Test the action method when the LLM provides feedback on the plan."""
    # Mock the LLM response
    mock_llm_response = MagicMock()
    mock_llm_response.content = (
        "The plan needs improvement. Avoid using any aversive methods, "
        "and ensure the dog is always rewarded for good behavior."
    )
    mock_llm.invoke.return_value = mock_llm_response

    # Define a sample state
    state = BehaviorState(
        behavior="sit",
        draft_plan="Step 1: Use a leash correction to enforce the sit position."
    )

    # Call the action method
    result = WelfareSpecialist.action(state)

    # Assertions
    assert (
            "The plan needs improvement." in result["welfare_review"]
    ), "Expected feedback not returned."
    mock_llm.invoke.assert_called_once()

    # Ensure the correct prompt was sent
    messages = mock_llm.invoke.call_args[0][0]
    assert len(messages) == 2, "Expected two messages in the prompt."
    assert "expert in animal welfare regulations" in messages[0].content
    assert "Here is a training plan to teach a dog the behaviour" in messages[1].content
    assert "Step 1: Use a leash correction to enforce the sit position." in messages[1].content
