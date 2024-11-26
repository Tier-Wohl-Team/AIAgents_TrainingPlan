import pytest
import json
from agents.SpecialistsTeamLeader import SpecialistsTeamLeader
from langchain_core.messages import SystemMessage, HumanMessage


@pytest.mark.integration
def test_specialists_team_leader_llm_response():
    # Example input for the LLM
    example_training_plan = """
    1. Sit on the cue
    2. Extend the duration of the sit from 10 to 25 seconds
    3. Stay in the Sit even when I throw a ball
    """

    # Construct the prompt
    prompt = """
        You are an experiences dog trainer. Given a training plan, you identify and extract all specific training 
        steps. For each step, you identify
        the behavior, the current status and the goal the trainer wants to reach.
        Return JSON with the single key "training_steps" and the value a list of dictionaries. 
        Each dictionary has five keys: "task", "behavior", "mode", "status" and "goal". The key "task" is the original 
        text in the training plan used to extract this specific step. The key "behavior" defines the behavior 
        the trainer wants to trains. The key "mode" can have one of the values "distance", "duration", 
        "cue introduction", "distractions". These have the following meanings:
        - distance: the trainer wants to increase the distance of the dog to the trainer or an object. The status and the goal of a distance are always floats.
        - duration: the trainer wants to increase the duration of the behavior. The status and the goal of a duration are always floats.
        - cue introduction: the dog should show a behaviour when the trainer says a specific word. The goal of a cue introduction is always a string.
        - distractions: the dog should show a behaviour even when there are distractions. The goal of a distractions is always a string.
        The keys "status" and "goal" describe the current status and the goal the trainer wants to reach.

        Example:
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
                    "task": "Stay in the Sit even when I trow a ball",
                    "behavior": "sit",
                    "mode": "distractions",
                    "status": "stays in the sit when trainer moves his arms",
                    "goal": "stay in the sit when trainer throws a ball"
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
    """

    # Call the LLM
    llm = SpecialistsTeamLeader.LLM
    response = llm.invoke([SystemMessage(content=prompt)] + [HumanMessage(content=example_training_plan)])

    # Parse the response as JSON
    try:
        training = json.loads(response.content)
    except json.JSONDecodeError as e:
        pytest.fail(f"Response is not valid JSON: {e}")

    # Validate the JSON structure
    assert "training_steps" in training, "The key 'training_steps' is missing from the JSON response."
    assert isinstance(training["training_steps"], list), "'training_steps' should be a list."

    for step in training["training_steps"]:
        assert isinstance(step, dict), "Each step in 'training_steps' should be a dictionary."
        for key in ["task", "behavior", "mode", "status", "goal"]:
            assert key in step, f"The key '{key}' is missing in a training step."
            assert step[key] is not None, f"The key '{key}' in a training step should not be None."

    print("Integration test passed: The LLM response is valid and follows the specified format.")
