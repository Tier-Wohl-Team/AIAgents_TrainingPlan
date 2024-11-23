# %% imports
import textwrap
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
import dotenv
from states.TeamState import TeamState

dotenv.load_dotenv("TrainingPlan_Team/.env")
# %%
llm = ChatOpenAI(temperature=0.0, model_name="gpt-4o-mini")

def agent(state: TeamState):
    """Plan Outline Generator
    This agent is responsible for generating a plan outline for the given question.
    """
    prompt = textwrap.dedent("""
        You are an experiences dog trainer. Given a question, you generate a plan outline.
        The plan outline breaks the question down into smaller steps. Typical steps are 'getting the behaviour',
        'increasing the duration', 'adding distractions', or 'increasing the distance'. Your job is to identify those
        steps which are necessary to reach the goal addressed in the question. Only include the steps that are
        necessary to reach the goal. Do not include steps that are not necessary.
        Once you have defined the steps, order them. Typically, a trainer will start by 'getting the behaviour' by e.g.
        luring, targetting, or capturing. Then, they will introduce an intermediate cue. This intermediate cue differs 
        from the final cue and can be e.g. a hand signal. If they got the behaviour by
        targetting or luring, it might be a good idea to convert this signal into the cue. Before we move on, it is
        important that there cue works without any luring component, as only then we are sure that the dog understood 
        the behaviour and paired it with the intermediate cue. Once we have the intermediate cue, we can start
        introducing mild distractions. When the dog is able to hold the behaviour despite these distractions,
        we start introducing some distance. Only then we work on duration. Once we have done this, we add further distractions. 
        Only then we can introduce the final cue.
        Remember, this is just an overview. It is your job to identify the steps that are necessary to reach the goal.
        Do not include steps that are not necessary.
        
        As an example, if the question is 'My dog sits for 10 seconds. I want to extend this duration to 25 seconds.',
        the plan outline would be:
        1. Introduce duration: Increase duration from 10 to 25 seconds
        """)
    task_prompt = textwrap.dedent("""
        Here is the goal of our client:
        
        {question}
        
        Please write the plan outline.
        """)
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=task_prompt.format(
            question=state["question"],
        ))
    ]

    outline = llm.invoke(messages)
    return {"outline_plan": outline.content}


