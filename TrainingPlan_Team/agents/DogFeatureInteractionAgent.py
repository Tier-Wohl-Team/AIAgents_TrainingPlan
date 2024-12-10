# %% imports
import textwrap

import dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from agents.BaseAgent import BaseAgent
from states.state_types import TeamState

dotenv.load_dotenv("../.env")


class DogFeatureInteractionAgent(BaseAgent):
    NAME = "DogFeatureInteractionAgent"
    LLM_MODEL = "gpt-4o-mini"
    LLM = (ChatOpenAI(model_name=LLM_MODEL, temperature=0.0))

    @staticmethod
    def handler_input(query: str):
        """
        Ask the handler of the animal for additional information about the his dog.
        The input should be a question for the human.
        """
        print(f"{query}")
        answer = input("Your answer: ")
        return answer

    @staticmethod
    def action(state: TeamState):
        llm = DogFeatureInteractionAgent.LLM.bind_tools([DogFeatureInteractionAgent.handler_input])
        DogFeatureInteractionAgent.greetings()

        background_story = textwrap.dedent("""
            You are an experienced animal trainer with a deep understanding of animal behavior and training techniques.
            What sets you apart is your ability to communicate with the handler of the animal to gather additional information
            about the behavior. Your clients love your patience and you politeness in your communication.
        """)

        task_prompt = textwrap.dedent("""
            Your team is working on a training plan to teach the behavior {behavior} to dog. Your team
            members have already drafted an initial outline of a training plan 
        
            PLAN_OUTLINE: 
            {plan_outline}
            
            Please go over the plan and check whether there are specific challenges for the dog in the plan. Take 
            into account any aspects which might be affected by the breed, age, health, size, temperament, 
            training history. Then ask the handler about these details of his dog. The handler is a person who
            intimately knows the dog and is able to provide additional information the dog.
        """)
        messages = [
            SystemMessage(content=background_story),
            HumanMessage(content=task_prompt.format(
                behavior=state["question"],
                plan_outline=state["outline_plan"],
            ))
        ]
        response = llm.invoke(messages)
        print("Please answer the following question to help us tune the training plan for you and your dog!")
        handler_information = []
        for tool_call in response.tool_calls:
            if tool_call["name"] == "handler_input":
                query = tool_call["args"]["query"]
                answer = DogFeatureInteractionAgent.handler_input(query)
                handler_information.append({"query": query, "answer": answer})
        return {"dog_details": handler_information}
