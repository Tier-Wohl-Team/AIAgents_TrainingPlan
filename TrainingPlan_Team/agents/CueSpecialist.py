import textwrap

import dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from agents.BaseAgent import BaseAgent
from states.state_types import BehaviorState

dotenv.load_dotenv("TrainingPlan_Team/.env")


class CueSpecialist(BaseAgent):
    """Cue Specialist
    """
    NAME = "cue_specialist"
    LLM_MODEL = "gpt-4o-mini"
    LLM = ChatOpenAI(temperature=0.0, model_name=LLM_MODEL)

    @staticmethod
    def action(state: BehaviorState):
        llm = CueSpecialist.LLM
        CueSpecialist.greetings()

        background_story = textwrap.dedent("""
                You are an experienced dog trainer with a special focus on teaching humans to train animals. Novice trainers 
                love to work with you as you help them to bring an existing behaviour under cue control. Cue control includes 
                the following four criteria:
                - the behavior is shown when the cue is given
                - the behavior is not shown when no cue is given
                - when the cue is given, only the target behavior and not other behavior is shown
                - the behavior is not shown when a different cue is given
                
                Your first step to reach this goal is to connect the cue with the behavior. To avoid any misunderstandings,
                you start to introduce the cue only when the behavior is reliably shown without luring it. You take care that
                the cue is given BEFORE the behavior is initiated.
            """)
        task_prompt = textwrap.dedent("""
            Please write a training plan to put the behaviour {behavior} under cue control. 
            
            CURRENT STATUS
            
            {status} 
            
            GOAL
        
            {goal}
            
            INFORMATION ABOUT THE DOG:
            
            {dog_details}
                
            In your training plan, start by stating the current status and then the goal. Next, develop a 
            detailed progression plan for the novice trainer. Clearly state what the novice trainer should do
            and how he should react when the dog is not performing as expected. Finally, add any special considerations about
            the dog to personalize the training plan.
                
            If there is already a previous version of the plan, you can use it as a reference, but take into account
            the review from the welfare specialist if present.
            
            PREVIOUS VERSION
            
            {draft_plan}
            
            WELFARE REVIEW
            
            {welfare_review}
            
            """)
        messages = [
            SystemMessage(content=background_story),
            HumanMessage(content=task_prompt.format(
                behavior=state["behavior"],
                status=state["status"],
                goal=state["goal"],
                dog_details=state.get("dog_details", ""),
                draft_plan=state.get("draft_plan", ""),
                welfare_review=state.get("welfare_review", "")
            ))
        ]
        response = llm.invoke(messages)
        return {"draft_plan": response.content}
