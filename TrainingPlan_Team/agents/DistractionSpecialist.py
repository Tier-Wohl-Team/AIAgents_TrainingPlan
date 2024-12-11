import textwrap

import dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from agents.BaseAgent import BaseAgent
from states.state_types import BehaviorState

dotenv.load_dotenv("TrainingPlan_Team/.env")


class DistractionSpecialist(BaseAgent):
    """Distraction Specialist
    This agent is responsible for training distractions.
    """
    NAME = "distraction_specialist"
    LLM_MODEL = "gpt-4o-mini"
    LLM = ChatOpenAI(temperature=0.0, model_name=LLM_MODEL)

    @staticmethod
    def action(state: BehaviorState):
        llm = DistractionSpecialist.LLM
        DistractionSpecialist.greetings()

        background_story = textwrap.dedent("""
            You are an experienced dog trainer having trained dogs to perform behaviours even under incredible 
            distractions. In addition, you have a knack for helping novice trainers to teach their own dogs to cope 
            with extreme distractions. The novice trainers love to work with you, as you give them extremely detailed
            instructions on how to start with mild distractions and build up from there in minuscule steps which enable 
            the dog to master each training step successfully. You give the novice trainers a clear progression plan 
            detailing out which distractions they should train, what the difficulty in this distraction is, when
            exactly they should reinforce the dog. In addition, each of your steps holds the information how the
            novice  trainer should react in case the distraction is too strong and the dog breaks the behaviour.
        """)
        task_prompt = textwrap.dedent("""
            Please write a training plan to assure that the behaviour {behavior} can be performed under distractions.
            The dog already knows the behaviour and can perform it. Only look at the distractions and leave other 
            elements to the other members of your team. 
            
            CURRENT STATUS
            
            {status} 
            
            GOAL
        
            {goal}
            
            INFORMATION ABOUT THE DOG:
            
            {dog_details}            
            
            In your training plan, start by stating the current status and then the goal. Next, develop a 
            detailed progression plan for the trainer. In your 
            progression plan, start from easy to hard. Clearly state the distraction and explain what the challenge is 
            in this setup. When choosing distractions, take into account the information about the dog (if they are given)
            Also add the information how the trainer should react in case the distraction is too strong 
            and the dog breaks the behaviour.
            
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
