import textwrap

import dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from states.BehaviorState import BehaviorState
dotenv.load_dotenv("TrainingPlan_Team/.env")
llm = ChatOpenAI(temperature=0.0, model_name="gpt-4o-mini")

def cue_specialist(state: BehaviorState):
    """Cue Specialist
    """
    print("cue specialist")
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
        Please write a training plan to put the behaviour {behavior} under cue control. The goal is to give the novice 
        trainer a detailed progression plan which the novice trainer can follow step by step. The current status is
        {status} and the goal is to reach {goal}.
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
