import pytest
from unittest.mock import patch
from states.state_types import BehaviorResearchState
from agents.BehaviorHandlerInteraction import BehaviorHandlerInteraction

@pytest.fixture
def mock_handler_input():
    with patch.object(BehaviorHandlerInteraction, 'handler_input_method') as mock_input:
        yield mock_input

def test_action(mock_handler_input):
    # Arrange
    mock_handler_input.side_effect = [
        "The dog is calm during training.",
    ]

    start_state = BehaviorResearchState(
        question="My dog sits for 10 seconds. I want to extend this duration to 25 seconds.",
        internet_research_results=["The dog is a canine."]
    )

    # Act
    result = BehaviorHandlerInteraction.action(start_state)

    # Assert
    assert len(result["handler_input"]) == 1
    assert result["asked_human"] is True
    assert result["handler_input"][0]["answer"] == "The dog is calm during training."
