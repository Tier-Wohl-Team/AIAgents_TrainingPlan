import json
import textwrap
import dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.constants import Send

from agents.BaseAgent import BaseAgent
from states.state_types import TeamState

dotenv.load_dotenv("../.env")


class SpecialistsTeamLeader(BaseAgent):
    """Distribution Agent
    This agent is responsible for extracting the training sections from the plan and handing them to the
    specialized agents in JsoN format.
    """
    NAME = "specialists_team_leader"
    LLM_MODEL = "gpt-4o-mini"
    LLM = ChatOpenAI(temperature=0.0, model_name="gpt-4o-mini",
                     model_kwargs={"response_format": {"type": "json_object"}})
    NODE_MAPPING = None  # This has to be set by the user. At the minimum, it has to have a 'other' mapping

    @classmethod
    def task_team_mapping(cls, mapping):
        """
        Dynamically sets the NODE_MAPPING for the class.

        Args:
            mapping (dict): A dictionary mapping modes to team names.
        """
        cls.NODE_MAPPING = mapping

    @staticmethod
    def action(state: TeamState):
        llm = SpecialistsTeamLeader.LLM
        SpecialistsTeamLeader.greetings()

        if SpecialistsTeamLeader.NODE_MAPPING is None:
            raise ValueError(
                "NODE_MAPPING is not set. Please configure NODE_MAPPING using "
                "`SpecialistsTeamLeader.set_node_mapping(mapping)` before calling `action`."
            )

        node_mapping = SpecialistsTeamLeader.NODE_MAPPING

        prompt = textwrap.dedent("""
            You are an experiences dog trainer. Given a training plan, you identify and extract all specific training 
            steps. For each step, you identify
            the behavior, the current status and the goal the trainer wants to reach.
            Return JSON with the single key "training_steps" and the value a list of dictionaries. 
            Each dictionary has five keys: "task", "behavior", "mode", "status" and "goal". The key "task" is the 
            original text in the training plan used to extract this specific step. The key "behavior" defines the 
            behavior the trainer wants to trains. The key "mode" can have one of the values "distance", "duration", 
            "cue introduction", "distractions", or "other". These have the following meanings:
            
            - distance: the trainer wants to increase the distance of the dog to the trainer or an object. The status and the goal of a distance are always floats.
            - duration: the trainer wants to increase the duration of the behavior. The status and the goal of a duration are always floats.
            - cue introduction: the dog should show a behaviour when the trainer says a specific word. The goal of a cue introduction is always a string.
            - distractions: the dog should show a behaviour even when there are distractions. The goal of a distractions is always a string.
            - other: use this mode if and only if you can not assign the step to any of the other modes. 
            
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
        """)
        print(state["outline_plan"])
        distance_duration = llm.invoke(
            [SystemMessage(content=prompt)] + [HumanMessage(content=state["outline_plan"])])
        training = json.loads(distance_duration.content)
        # for step in training["training_steps"]:
        #     print("Sending to: ", node_mapping.get(step["mode"], "other"))
        #     print(step)
        return [Send(node_mapping.get(step["mode"], "other"), step)
                for step in training["training_steps"]
                ]
