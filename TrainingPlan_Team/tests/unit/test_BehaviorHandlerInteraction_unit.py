import unittest
from unittest.mock import patch
from states.state_types import BehaviorResearchState
from agents.BehaviorHandlerInteraction import BehaviorHandlerInteraction


@patch.object(BehaviorHandlerInteraction, 'handler_input_method')
def test_action(self, mock_input):
    # Arrange
    mock_input.side_effect = [
        "The dog is calm during training.",
        "Try extending the sit duration incrementally."
    ]

    start_state = BehaviorResearchState(
        question="My dog sits for 10 seconds. I want to extend this duration to 25 seconds.",
        internet_research_results=["The dog is a canine."]
    )

    # Act
    result = BehaviorHandlerInteraction.action(start_state)

    # Assert
    self.assertEqual(len(result["handler_input"]), 2)
    self.assertTrue(result["asked_human"])
    self.assertEqual(result["handler_input"][0]["answer"], "The dog is calm during training.")
    self.assertEqual(result["handler_input"][1]["answer"], "Try extending the sit duration incrementally.")
