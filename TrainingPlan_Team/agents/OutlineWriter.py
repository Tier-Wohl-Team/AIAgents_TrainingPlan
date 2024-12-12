# %% imports
import textwrap

import dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from agents.BaseAgent import BaseAgent
from states.state_types import BehaviorResearchState

dotenv.load_dotenv("../.env")

class OutlineWriter(BaseAgent):
    NAME = "OutlineWriter"
    LLM_MODEL = "gpt-4o-mini"
    LLM = ChatOpenAI(temperature=0.0, model_name=LLM_MODEL)

    @staticmethod
    def action(state: BehaviorResearchState):
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
        DATA SECTION:
        
        GOAL:
        
        {question}
        
        DOG INFORMATION:
        
        {dog_details}
        
        INTERNET INFORMATION:
        
        {internet_research_results}
        
        ADDITIONAL INFORMATION FROM THE DOG HANDLER:
        
        {handler_input}
        
        END OF DATA SECTION
        
        Based on the current status provided in the question, write a concise and minimalistic plan outline that 
        includes only the necessary steps to achieve the GOAL. Avoid unnecessary or irrelevant steps.
            
        If INTERNET INFORMATION is not given and you need more information about the behaviour and how to train 
        it, answer with 'I need more information from the internet.'.
        
        Only If the ADDITIONAL INFORMATION FROM THE DOG HANDLER section is empty, you can ask the dog handler. Use this
        in case the status and the goal is not defined with measurable parameters.In this case, answer with:
        'I need more information from the dog handler.' 
                        """)

        internet_research_results = state.get("internet_research_results", "")
        handler_input = state.get("handler_input", "")
        formatted_dog_details = "\n".join([f"- {d[0]}: {d[1]}" for d in state.get("dog_details", [])])
        messages = [
            SystemMessage(content=background_story),
            HumanMessage(content=task_prompt.format(
                question=state["question"],
                dog_details=formatted_dog_details,
                internet_research_results=internet_research_results,
                handler_input=handler_input,
            ))
        ]

        outline = llm.invoke(messages)
        print(outline.content)
        return {"outline_plan": outline.content}

