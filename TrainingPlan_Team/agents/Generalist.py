import textwrap

import dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from agents.BaseAgent import BaseAgent
from states.state_types import BehaviorState

dotenv.load_dotenv("TrainingPlan_Team/.env")

class Generalist(BaseAgent):
    """Generalist
    This agent is responsible for handling parts of the training plan which are not handled
    by any of the specialist agents.
    """
    NAME = "generalist"
    LLM_MODEL = "gpt-4o-mini"
    LLM = ChatOpenAI(temperature=0.0, model_name=LLM_MODEL)

    @staticmethod
    def action(state: BehaviorState):
        llm = Generalist.LLM
        Generalist.greetings()

        background_story = textwrap.dedent("""
            You are an experienced dog trainer with a special focus on teaching humans to train animals. Novice
            trainers love to work with you as you help them to break their trainings plans down into actionable 
            progression plans. 
        """)
        task_prompt = textwrap.dedent("""
            Please write a training plan for the behaviour {behavior}. 
            
            the current status is
            
            {status}
            
            The goal is to reach the following status:
            
            {goal}
            
            In your training plan, start by stating the current status and then the goal. Next, develop a 
            detailed progression plan for the novice trainer. Clearly state what the novice trainer should do
            and how he should react when the dog is not performing as expected. Focus only on how to get from the
            current status to the goal. Do not add any additional training information.
        """)

        messages = [
            SystemMessage(content=background_story),
            HumanMessage(content=task_prompt.format(
                behavior=state["behavior"],
                status=state["status"],
                goal=state["goal"],
            ))
        ]
        response = llm.invoke(messages)
        return {"draft_plan": response.content}
