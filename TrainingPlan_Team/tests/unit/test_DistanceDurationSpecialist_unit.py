import pytest
import yaml
from unittest.mock import patch, MagicMock
from agents.DistanceDurationSpecialist import DistanceDurationSpecialist
from states.state_types import BehaviorState


@pytest.mark.unit
def test_load_trials():
    """Test if YAML trials are loaded correctly."""
    with patch("builtins.open", create=True) as mock_open:
        mock_open.return_value.__enter__.return_value.read.return_value = """
        distance_and_duration:
          durations:
            0.5: [0.5, 0.5, 0.5, 0.5]
            1.0: [1, 1, 1, 1]
          distances:
            0.5: [0.5, 0.5, 0.5, 0.5]
            1.0: [1, 1, 1, 1]
        """
        with patch("yaml.safe_load", autospec=True) as mock_yaml_load:
            mock_yaml_load.return_value = {
                "distance_and_duration": {
                    "durations": {"0.5": [0.5, 0.5, 0.5, 0.5], "1.0": [1, 1, 1, 1]},
                    "distances": {"0.5": [0.5, 0.5, 0.5, 0.5], "1.0": [1, 1, 1, 1]},
                }
            }
            # Reload the trials data
            DistanceDurationSpecialist.TRIALS = yaml.safe_load(mock_open.return_value)

            # Assertions
            assert "distance_and_duration" in DistanceDurationSpecialist.TRIALS
            assert "durations" in DistanceDurationSpecialist.TRIALS["distance_and_duration"]
            assert "distances" in DistanceDurationSpecialist.TRIALS["distance_and_duration"]


@pytest.mark.unit
@patch("agents.DistanceDurationSpecialist.DistanceDurationSpecialist.LLM", autospec=True)
def test_action(mock_llm):
    """Test the action method of the agent."""
    # Mock the LLM response
    expected_answer = """
    You should start with a distance of 1.0 meters. after two successful repetitions,
    extend the distance to 2.0 meters and perform two further repetitions.
    """
    mock_llm_response = MagicMock()
    mock_llm_response.content = expected_answer
    mock_llm.invoke.return_value = mock_llm_response

    # Mock YAML trials
    DistanceDurationSpecialist.TRIALS = {
        "distance_and_duration": {
            "durations": {"0.5": [0.5, 0.5], "1.0": [1.0, 1.0]},
            "distances": {"1.0": [1.0, 1.0], "2.0": [2.0, 2.0]},
        }
    }

    # Define the input state
    state = BehaviorState(mode="distance", goal=2.0, status=1.0, behavior="sit")

    # Call the action method
    result = DistanceDurationSpecialist.action(state)
    assert result["draft_plan"] == expected_answer
    # Ensure the LLM was invoked with the correct prompt
    mock_llm.invoke.assert_called_once()
    messages = mock_llm.invoke.call_args[0][0]
    distances = DistanceDurationSpecialist.TRIALS["distance_and_duration"]["distances"]
    expected_training_steps = "\n\n".join([f"{key}: {value}" for key, value in distances.items() if 1.0 <= float(key) <= 2.0])

    assert len(messages) == 2
    assert "experienced dog trainer" in messages[0].content
    assert expected_training_steps in messages[1].content
