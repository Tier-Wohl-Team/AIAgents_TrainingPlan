import pytest
from agents.GoalDefinitionAgent import GoalDefinitionAgent
from states.state_types import TeamState

@pytest.mark.integration
def test_action_integration():
    question = "My dog can sit for 30 seconds. I want to extend this duration to 1 minute."
    state = TeamState(
        question=question,
    )
    response = GoalDefinitionAgent.action(state)
    print(response["goal"])
    assert response["goal"]["current_status"] != "Not defined"
    assert response["goal"]["target_goal"] == "Not defined"

@pytest.mark.integration
def test_action_current_status_not_defined():
    question = "My dog can walk at heel. I want to extend the distance to 20 meters."
    state = TeamState(
        question=question,
    )
    response = GoalDefinitionAgent.action(state)
    print(response["goal"])
    assert response["goal"]["current_status"] == "Not defined"
    assert response["goal"]["target_goal"] != "Not defined"

@pytest.mark.integration
def test_action_target_goal_not_defined():
    question = "My dog can sit for 30 seconds. I want to extend this duration."
    state = TeamState(
        question=question,
    )
    response = GoalDefinitionAgent.action(state)
    print(response["goal"])
    assert response["goal"]["current_status"] != "Not defined"
    assert (response["goal"]["target_goal"] == "Not defined"

@pytest.mark.integration)
def test_action_target_goal_both_not_defined():
    question = "My dog should sit for longer."
    state = TeamState(
        question=question,
    )
    response = GoalDefinitionAgent.action(state)
    print(response["goal"])
    assert response["goal"]["current_status"] == "Not defined"
    assert response["goal"]["target_goal"] == "Not defined"