import pytest
from agents.DistanceDurationSpecialist import DistanceDurationSpecialist
from states.state_types import BehaviorState
from tests.utils.llm_evaluator import evaluate_with_llm


@pytest.mark.llm
def test_distance_duration_specialist_integration():
    """Integration test to verify LLM output using mock TRIALS."""

    # Mock TRIALS directly in the class to avoid using the user-modifiable config file
    DistanceDurationSpecialist.TRIALS = {
        "distance_and_duration": {
            "durations": {"0.5": [0.5, 0.7], "1.0": [1.0, 1.2]},
            "distances": {"1.0": [1.0, 1.3], "2.0": [2.0, 2.5]},
        }
    }

    # Define a sample state
    state = BehaviorState(mode="distance", goal=2.0, status=1.0, behavior="sit")

    # Call the action method
    response = DistanceDurationSpecialist.action(state)

    # Print the result for manual inspection (optional during debugging)
    print(response["draft_plan"])

    # Validate that the output is non-empty
    assert response["draft_plan"], "The LLM output is empty."

    # Validate that the draft plan includes the expected progressions
    distances = DistanceDurationSpecialist.TRIALS["distance_and_duration"]["distances"]
    expected_training_steps = "\n\n".join([f"{key}: {value}" for key, value in distances.items() if 1.0 <= float(key) <= 2.0])
    expected_output = f"""
    Currently, the dog can sit at a distance of 1.0 meters.
    The goal is that the dog should sit at a distance of 2.0 meters.
    The training steps are:
    {expected_training_steps}
    """
    features = """
    Take care that indeed every distance mentioned in the training steps of the expected output
    is used in the generated output. Do not consider the structure of the generated output.
    Additional information is also allowed. The main point to check it the presence of all training steps.
    """
    evaluation = evaluate_with_llm(expected_output=expected_output,
                                   generated_output=response["draft_plan"],
                                   features=features)
    assert evaluation["correct"], f"LLM Evaluation failed: {evaluation['explanation']}"
