import pytest
from unittest.mock import patch, MagicMock, call
from agents.SpecialistsTeamLeader import SpecialistsTeamLeader
from states.state_types import TeamState

@pytest.mark.unit
@patch('agents.SpecialistsTeamLeader.Send', autospec=True)  # Mock the Send function
@patch('agents.SpecialistsTeamLeader.SpecialistsTeamLeader.LLM', autospec=True)  # Mock the LLM
def test_action_send_mock(mock_llm, mock_send):
    # Configure the NODE_MAPPING
    SpecialistsTeamLeader.task_team_mapping({
        "duration": "distance_duration_welfare_graph",
        "distance": "distance_duration_welfare_graph",
        "cue introduction": "cue_welfare_graph",
        "distractions": "distraction_welfare_graph",
        "other": "generalist_team_graph"
    })
    # Mock LLM response
    mock_llm_response = MagicMock()
    mock_llm_response.content = '''
    {
        "training_steps": [
            {
                "task": "Sit on the cue",
                "behavior": "sit",
                "mode": "cue introduction",
                "status": "sitting",
                "goal": "sit on the cue"
            },
            {
                "task": "Extend the duration of the sit from 10 to 25 seconds",
                "behavior": "sit",
                "mode": "duration",
                "status": 10,
                "goal": 25
            }
        ]
    }
    '''
    mock_llm.invoke.return_value = mock_llm_response

    # Mock state
    test_training_plan = """
    1. Introduce a sit cue
    2. Extend the duration of the sit from 10 to 25 seconds
    """
    state = TeamState(outline_plan=test_training_plan)

    # Call the action method
    response = SpecialistsTeamLeader.action(state)

    # Assert the Send function was called with correct arguments
    expected_calls = [
        call("cue_welfare_graph", {
            "task": "Sit on the cue",
            "behavior": "sit",
            "mode": "cue introduction",
            "status": "sitting",
            "goal": "sit on the cue"
        }),
        call("distance_duration_welfare_graph", {
            "task": "Extend the duration of the sit from 10 to 25 seconds",
            "behavior": "sit",
            "mode": "duration",
            "status": 10,
            "goal": 25
        })
    ]

    mock_send.assert_has_calls(expected_calls, any_order=False)

    # Verify the LLM was called
    mock_llm.invoke.assert_called_once()
