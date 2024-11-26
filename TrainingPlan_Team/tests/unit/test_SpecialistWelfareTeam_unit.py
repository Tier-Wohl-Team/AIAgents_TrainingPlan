import pytest
from unittest.mock import MagicMock
from agents.DistanceDurationSpecialist import DistanceDurationSpecialist
from agents.WelfareSpecialist import WelfareSpecialist
from states.state_types import BehaviorState
from langgraph.constants import START, END
from teams.SpecialistWelfareTeam import SpecialistWelfareTeam

@pytest.mark.unit
def test_team_initialization():
    """Test the initialization of the SpecialistWelfareTeam."""
    max_iterations = 3
    team = SpecialistWelfareTeam(
        name="Distance Duration Team",
        specialist_agent=DistanceDurationSpecialist,
        max_welfare_iterations=max_iterations
    )

    # Assertions
    assert team.name == "Distance Duration Team", "Team name is incorrect."
    assert team.graph is not None, "Team graph should not be None."


@pytest.mark.unit
def test_graph_nodes_and_edges():
    """Test the graph nodes and edges."""
    max_iterations = 3
    team = SpecialistWelfareTeam(
        name="Distance Duration Team",
        specialist_agent=DistanceDurationSpecialist,
        max_welfare_iterations=max_iterations
    )

    # Get the graph from the team
    graph = team.graph

    # Assert nodes
    expected_nodes = {START, DistanceDurationSpecialist.NAME, WelfareSpecialist.NAME}
    assert set(graph.nodes.keys()) == expected_nodes, "Graph nodes are incorrect."


@pytest.mark.unit
def test_should_continue_good_plan():
    """Test should_continue when the plan is good."""
    state = BehaviorState(
        task="My dog sits for 10 seconds. I want to extend this duration to 25 seconds.",
        behavior="sit",
        mode="duration",
        status=10,
        goal=25,
        draft_plan="Some draft plan",
        welfare_review="The plan is good.",
        iteration_count=0,
        is_finished=False,
        plans=[]
    )

    def mock_should_continue(state):
        if state.get("welfare_review", "") == "The plan is good.":
            return END
        return DistanceDurationSpecialist.NAME

    result = mock_should_continue(state)
    assert result == END, "should_continue did not return END for a good plan."


@pytest.mark.unit
def test_should_continue_exceeded_iterations():
    """Test should_continue when max welfare iterations are exceeded."""
    state = BehaviorState(
        task="My dog sits for 10 seconds. I want to extend this duration to 25 seconds.",
        behavior="sit",
        mode="duration",
        status=10,
        goal=25,
        draft_plan="Some draft plan",
        welfare_review="",
        iteration_count=3,
        is_finished=False,
        plans=[]
    )

    def mock_should_continue(state):
        if state.get("iteration_count", 0) >= 3:
            return END
        return DistanceDurationSpecialist.NAME

    result = mock_should_continue(state)
    assert result == END, "should_continue did not return END for exceeded iterations."

