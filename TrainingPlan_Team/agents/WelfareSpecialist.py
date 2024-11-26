import textwrap
import dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from agents.BaseAgent import BaseAgent
from states.state_types import BehaviorState
dotenv.load_dotenv("../.env")


class WelfareSpecialist(BaseAgent):
    """Welfare Specialist
    This agent checks training plans and gives feedback on how to improve them.
    If the plan follows the animal welfare aspect, it will answer 'The plan is good.'
    """
    NAME = "WelfareSpecialist"
    LLM_MODEL = "gpt-4o-mini"
    LLM = ChatOpenAI(temperature=0.0, model_name=LLM_MODEL)

    @staticmethod
    def action(state: BehaviorState):
        llm = WelfareSpecialist.LLM
        WelfareSpecialist.greetings()

        background_story = textwrap.dedent("""
            You are an expert in animal welfare regulations with a deep understanding of humane
            treatment. Your primary responsibility is to give feedback on training plans to ensure they 
            comply with animal welfare standards. Especially, you ensurer that the used methods are
            based on positive reinforcement and do not use any aversives.
            """)
        task_prompt = textwrap.dedent("""
            Here is a training plan to teach a dog the behaviour "{behavior}". 
            
            TRAINING_PLAN:
            {training_plan}
            
            Don't change the plan itself, only add feedback.
            If you think the plan is good, just write "The plan is good."
        """)
        messages = [
            SystemMessage(content=background_story),
            HumanMessage(content=task_prompt.format(
                behavior=state["behavior"],
                training_plan=state["draft_plan"]
            ))
        ]
        response = llm.invoke(messages)
        return {"welfare_review": response.content}
