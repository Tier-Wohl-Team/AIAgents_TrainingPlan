import pytest
from agents.OutlineWriter import OutlineWriter
from states.state_types import TeamState
from tests.utils.llm_evaluator import evaluate_with_llm

@pytest.mark.llm
def test_action_integration():
    question = "My dog can walk at heel for 10 meters. Now, I want to extend the distance to 20 meters."
    expected_output = "Gradually increase the distance from 10 meters to 20 meters in small increments."
    features = ("The answer should not contain additional points like 'getting the behavior',"
                "'Adding distractions', or 'cue introduction'.")
    state = TeamState(
        question=question,
    )
    response = OutlineWriter.action(state)
    evaluation = evaluate_with_llm(expected_output=expected_output,
                                   generated_output=response["outline_plan"],
                                   features=features)
    assert evaluation["correct"], f"LLM Evaluation failed: {evaluation['explanation']}"
