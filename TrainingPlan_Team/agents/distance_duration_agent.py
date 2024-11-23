import textwrap
import dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from states.BehaviorState import BehaviorState
dotenv.load_dotenv("TrainingPlan_Team/.env")
llm = ChatOpenAI(temperature=0.0, model_name="gpt-4o-mini")


spector_durations = {
    0.1: [0.1, 0.1, 0.1, 0.1],
    0.3: [0.3, 0.3, 0.3, 0.3],
    0.5: [0.5, 0.5, 0.5, 0.5],
    0.7: [0.7, 0.7, 0.7, 0.7],
    1.0: [1, 1, 1, 1],
    1.5: [1.5, 1.5, 1.5, 1.5],
    2.0: [2, 0.5, 1.5, 2.5, 0.5, 1.5, 2, 0.5, 2, 1],
    3.0: [3, 1, 3, 1, 4, 1, 3, 1, 4, 2, 1, 3, 1],
    4.0: [4, 2, 5, 2, 6, 2, 4, 1, 6, 3, 2, 4, 1],
    5.0: [5, 2, 6, 3, 7, 3, 5, 1, 7, 4, 3, 5, 1],
    7.0: [7, 4, 6, 11, 1, 4, 7, 3, 11, 5, 4, 7, 2, 11, 1],
    10.0: [10, 6, 8, 15, 1, 5, 10, 3, 15, 6, 10, 2, 15, 1],
    15.0: [15, 7, 10, 22, 1, 7, 15, 5, 22, 10, 7, 15, 3, 22, 1],
    22.0: [22, 11, 15, 3, 7, 33, 11, 22, 1, 15, 11, 33, 22, 5, 33, 3],
    33.0: [33, 16, 5, 24, 45, 1, 16, 8, 33, 45, 3, 16, 10, 33, 5, 24, 45, 16, 1],
    45.0: [45, 22, 7, 33, 61, 1, 22, 11, 45, 61, 4, 22, 14, 45, 7, 3],
}

spector_distances = {
    0.5: [0.5, 0.5, 0.5, 0.5],
    1.0: [1, 1, 1, 1],
    2.0: [2, 2, 2, 2],
    3.0: [3, 3, 3, 3],
    4.0: [4, 4, 4, 4],
    5.0: [5, 2, 6, 3, 7, 3, 5, 1, 7, 4, 3, 5, 1],
    7.0: [7, 3, 10, 5, 8, 4, 7, 2, 5, 10, 3, 7, 2],
    10.0: [10, 5, 8, 4, 12, 6, 10, 3, 7, 10, 5, 12, 1, 10, 3],
    12.0: [12, 6, 10, 4, 15, 7, 12, 6, 15, 8, 12, 2, 10, 4],
    15.0: [15, 7, 12, 8, 13, 6, 20, 2, 10, 7, 15, 9, 20, 12, 8, 15, 7, 4],
    20.0: [20, 10, 12, 8, 15, 9, 17, 10, 25, 15, 3, 20, 10, 25, 13, 5],
    25.0: [15, 25, 12, 20, 5, 15, 25, 2, 18, 30, 12, 25, 15, 30, 25, 12, 3],
    30.0: [25, 30, 15, 20, 10, 17, 35, 5, 25, 30, 15, 20, 35, 17, 30, 15, 5],
    38.0: [31, 38, 19, 25, 12, 21, 44, 6, 31, 38, 19, 25, 44, 21, 38, 19, 6],
    48.0: [39, 48, 24, 31, 15, 26, 55, 8, 39, 48, 24, 31, 55, 26, 48, 24, 8],
    60.0: [49, 60, 30, 39, 19, 32, 69, 10, 49, 60, 30, 39, 69, 32, 60, 30, 10],
    75.0: [61, 75, 38, 49, 24, 40, 86, 12, 61, 75, 38, 49, 86, 40, 75, 38, 12],
    94.0: [76, 94, 48, 61, 30, 50, 108, 15, 76, 94, 48, 61, 108, 50, 94, 48, 15],
}


def agent(state: BehaviorState):
    """Distance and Duration Agent
    This agent generates training steps and variations to build distance and duration.
    """
    print("distance_duration")
    key_source = spector_distances if state["mode"] == "distance" else spector_durations
    state["goal"] = float(state["goal"])
    state["status"] = float(state["status"])
    steps = [key for key in key_source.keys() if state["status"] <= key <= state["goal"]]
    training_steps = [{step: key_source[step]} for step in steps]
    spector_backstory = textwrap.dedent("""
        You are an experienced dog trainer with a special focus on writing training plans for novice trainers.
        """)
    task_prompt = textwrap.dedent("""
        The next steps in the training plan is to extend the {mode} of the behavior '{behavior}' from {status} 
        to {goal}.
        Following you find progressions with variations. Each line starts with the target {mode} of the progression 
        followed by a list of variations to reach this target:

        {training_steps}

        Please write a human readable text from for these training steps. Start by stating the current status and the 
        final goal. For the variations, format them as list. Only write about these specific steps, do not add any 
        additional training information.
        """)
    messages = [
        SystemMessage(content=spector_backstory),
        HumanMessage(content=task_prompt.format(
            behavior=state["behavior"],
            mode=state["mode"],
            status=state["status"],
            goal=state["goal"],
            training_steps="\n\n".join([f"{step}: {variation}" for training_step in
                                        training_steps for step, variation in training_step.items()])
        ))
    ]
    response = llm.invoke(messages)
    return {"draft_plan": response.content}

