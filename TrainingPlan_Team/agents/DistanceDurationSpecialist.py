import os
import textwrap
import dotenv
import yaml
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from agents.BaseAgent import BaseAgent
from states.state_types import BehaviorState
dotenv.load_dotenv("../.env")


class DistanceDurationSpecialist(BaseAgent):
    """Distance and Duration Agent
    This agent generates training steps and variations to build distance and duration.
    """
    NAME = "DistanceDurationSpecialist"
    LLM_MODEL = "gpt-4o-mini"
    LLM = ChatOpenAI(temperature=0.0, model_name="gpt-4o-mini")
    # Get the absolute path to the YAML file
    CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..\\configs\\distance_duration_trials.yaml")
    print(CONFIG_PATH)
    with open(CONFIG_PATH, "r") as file:
        TRIALS = yaml.safe_load(file)

    @staticmethod
    def action(state: BehaviorState):
        llm = DistanceDurationSpecialist.LLM
        DistanceDurationSpecialist.greetings()

        trials = DistanceDurationSpecialist.TRIALS
        durations = {float(k): v for k, v in trials["distance_and_duration"]["durations"].items()}
        distances = {float(k): v for k, v in trials["distance_and_duration"]["distances"].items()}
        key_source = distances if state["mode"] == "distance" else durations
        state["goal"] = float(state["goal"])
        state["status"] = float(state["status"])
        steps = [key for key in key_source.keys() if state["status"] <= key <= state["goal"]]
        training_steps = [{step: key_source[step]} for step in steps]

        background_story = textwrap.dedent("""
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
            SystemMessage(content=background_story),
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

