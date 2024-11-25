# %% imports
import textwrap
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from agents.BaseAgent import BaseAgent
from states.state_types import TeamState


class OutlineWriter(BaseAgent):
    NAME = "OutlineWriter"
    LLM_MODEL = "gpt-4o-mini"
    LLM = ChatOpenAI(temperature=0.0, model_name=LLM_MODEL)

    @staticmethod
    def action(state: TeamState):
        llm = OutlineWriter.LLM
        OutlineWriter.greetings()

        background_story = textwrap.dedent("""
        You are an experienced dog trainer specializing in creating precise, minimalistic training plans. Your task is 
        to generate a plan outline that includes only the necessary steps to achieve the stated goal. 
        
        1. **Evaluate the current status**: Begin by identifying what the dog can already do (e.g., the existing 
            behavior or skill level mentioned in the question). Exclude any steps that are already accomplished or 
            unnecessary for the stated goal.
        2. **Define necessary steps only**: Based on the current status, outline only the steps that are required to 
            progress toward the goal. Avoid including irrelevant or redundant steps. 
           - For example, if the behavior is already learned, skip "getting the behavior."
           - If no distractions or distance are part of the goal, do not include those steps.
        3. **Be concise and goal-focused**: Provide only the steps that directly address the client’s question. 
            Avoid adding general training advice, such as "adding distractions" or "introducing a final cue," unless 
            explicitly required to achieve the goal.
        
        ### Example:
        **Question**: "My dog sits for 10 seconds. I want to extend this duration to 25 seconds."
        **Correct Plan Outline**:
        - Gradually increase the duration from 10 seconds to 25 seconds in small increments.
        
        **Incorrect Plan Outline**:
        1. Getting the behavior: Ensure the dog can reliably perform the sit command.
        2. Increasing the duration: Gradually increase the time...
        3. Adding distractions: Introduce mild distractions...
        4. Final cue: Once the dog can sit...
        
        Remember, your job is to write the most efficient and minimalistic plan possible to achieve the goal. Avoid 
        adding unnecessary steps or advice that is not directly related to the stated goal.
                """)
        task_prompt = textwrap.dedent("""
        Here is the goal of our client:
        
        {question}
        
        Based on the current status provided in the question, write a concise and minimalistic plan outline that 
        includes only the necessary steps to achieve the goal. Avoid unnecessary or irrelevant steps.
                """)
        messages = [
            SystemMessage(content=background_story),
            HumanMessage(content=task_prompt.format(
                question=state["question"],
            ))
        ]

        outline = llm.invoke(messages)
        return {"outline_plan": outline.content}
