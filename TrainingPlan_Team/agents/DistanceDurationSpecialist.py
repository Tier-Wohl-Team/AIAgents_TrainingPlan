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
            
            INFORMATION ABOUT THE DOG:
            
            {dog_details}
    
            Please write a human readable text from for these training steps. Start by stating the current status and the 
            final goal. Then list ALL training progressions including all repetitions exactly as the are given in the 
            list of variations. .
            The goal is to give the trainer a step by step training plan. Also give information on how the trainer 
            should react if the dog fails to perform a training step. Finally, add any special considerations about
            the dog to personalize the training plan. Here is an example for the training steps:
            
            5.0: [2, 6, 3, 7]
            10.0: [5, 15, 10, 20]
            
            **Current Status:** The dog has successfully learned to perform the 'sit' command at a distance of 1.0 meters. 

            **Final Goal:** The objective is to extend the distance of the 'sit' command to 2.0 meters.

            **Training Progressions:**

            1. ** 5 meters:**
                - Start with 2 meters.
                - Repeat with 6 meters.
                - 3 meters
                - 7 meters
            2. ** 10 meters:**
                - Start with 5 meters.
                - Repeat with 15 meters.
                - 10 meters
                - 20 meters
                If the dog fails at any distance, repeat this step. If he fails again, go to the next step. If he
                fails at more than 3 distances, go back to the previous progression.

            **Considerations**
            - As your dog likes to work for treats, use these as reinforcements. This allows you to keep a rate
              of repetitions high.
            """)
        training_steps = "\n\n".join([f"{step}: {variation}" for training_step in
                                     training_steps for step, variation in training_step.items()])
        print(training_steps)
        messages = [
            SystemMessage(content=background_story),
            HumanMessage(content=task_prompt.format(
                behavior=state["behavior"],
                dog_details=state["dog_details"],
                mode=state["mode"],
                status=state["status"],
                goal=state["goal"],
                training_steps=training_steps
            ))
        ]
        response = llm.invoke(messages)
        return {"draft_plan": response.content}

