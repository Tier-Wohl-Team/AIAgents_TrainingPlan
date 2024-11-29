# %% imports
import textwrap

import dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from agents.BaseAgent import BaseAgent
from states.state_types import TeamState

dotenv.load_dotenv("../.env")

class FinalPlanWriter(BaseAgent):
    NAME = "FinalPlanWriter"
    LLM_MODEL = "gpt-4o-mini"
    LLM = ChatOpenAI(model_name=LLM_MODEL, temperature=0.0)

    @staticmethod
    def action(state: TeamState):
        llm = FinalPlanWriter.LLM
        FinalPlanWriter.greetings()

        background_story = textwrap.dedent("""
            You are an expert in communicating with humans. Your speciality is to convert a training plan written
            by dog training experts into a plan which is easily understood by a non-professional dog handler.
            As a member of a Training Plan Team, you use this experience to add helpful information to
            a short plan outline.
        """)
        task_prompt = textwrap.dedent("""
            The question of the client was: {question}
            
            Your colleagues have written the following outline of a training plan:
            
            {outline_plan}
            
            They have used the following additional information retrieved from the internet:
            
            {internet_research_results}
            
            Additionally they got the following feedback from the dog handler:
            
            {handler_input}
            
            The specialists in the team have written the following detailed plans:
            
            {plans}
            
            Based on all these information, please write a final plan. Here are some guidelines:
            - add all points of the outline plan unchanged into your plan. They will be used to link to the detail plans.
            - Do not add any new points
            - Do not include any of the details from the detailed plans in your final report. The client
                will get the detailed plans.
            - add a section of all the training aids like special material, stuff used for positive reinforcement
              to your final plan such that the client directly sees, what he has to have at hand.

        """)
        internet_research_results = ""
        if "internet_research_results" in state:
            internet_research_results = "\n\n".join(state["internet_research_results"])
        handler_input = ""
        if "handler_input" in state:
            "\n".join([f"Question: {q}\nAnswer: {a}" for q, a in handler_input])
        plans = "\n".join([plan for _,plan in state["plans"]])
        messages = [
            SystemMessage(content=background_story),
            HumanMessage(content=task_prompt.format(
                question=state["question"],
                outline_plan=state["outline_plan"],
                internet_research_results=internet_research_results,
                handler_input=handler_input,
                plans=plans
            ))
        ]

        final_plan = llm.invoke(messages)
        print(final_plan.content)
        return {"final_plan": final_plan.content}