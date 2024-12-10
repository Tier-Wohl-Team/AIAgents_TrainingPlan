import pytest
from unittest.mock import patch, MagicMock
from states.state_types import BehaviorResearchState
from agents.OutlinePlanEvaluator import OutlinePlanEvaluator


@pytest.mark.unit
@patch('agents.OutlinePlanEvaluator.OutlinePlanEvaluator.LLM', autospec=True)  # Mock the LLM class
def test_action_mock(mock_llm):
    # Configure the mocked LLM to have an `invoke` method
    fake_response = MagicMock()
    fake_response.content = "no_rewrite"
    mock_llm.invoke.return_value = fake_response

    # Mock state
    state = BehaviorResearchState(
        outline_plan="Step 1: Teach the dog to sit.\nStep 2: Teach the dog to stay.",
        new_dog_details="The dog has a preference for verbal commands over hand signals.",
        dog_details="The dog is a Border Collie with a high energy level.",
        iteration_count=2
    )

    # Call the action method
    response = OutlinePlanEvaluator.action(state)

    # Assertions on the response
    assert response["is_finished"] is True, "Expected is_finished to be True for 'no_rewrite' response."
    assert response["iteration_count"] == 3, "Expected iteration_count to increment by 1."
    assert "The dog has a preference for verbal commands over hand signals." in response["dog_details"], (
        "Expected new_dog_details to be appended to dog_details."
    )
    assert response["new_dog_details"] == [], "Expected new_dog_details to be cleared."

    # Verify the mock LLM was called with the correct messages
    mock_llm.invoke.assert_called_once()
    messages = mock_llm.invoke.call_args[0][0]  # Extract messages passed to invoke
    assert any("You are an experienced dog trainer" in message.content for message in messages), (
        "Expected background story in the SystemMessage."
    )
    assert any("Here's the current outline of the training plan" in message.content for message in messages), (
        "Expected task_prompt in the HumanMessage."
    )
