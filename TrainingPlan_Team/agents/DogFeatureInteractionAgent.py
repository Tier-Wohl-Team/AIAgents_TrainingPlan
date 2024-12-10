# %% imports
import json
import textwrap

import dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from agents.BaseAgent import BaseAgent
from states.state_types import BehaviorResearchState

dotenv.load_dotenv("../.env")


class DogFeatureInteractionAgent(BaseAgent):
    NAME = "DogFeatureInteractionAgent"
    LLM_MODEL = "gpt-4o-mini"
    LLM = ChatOpenAI(model_name=LLM_MODEL, temperature=0.0,
                     model_kwargs={"response_format": {"type": "json_object"}})

    @staticmethod
    def handler_input(query: str):
        print(f"{query}")
        answer = input("Your answer: ")
        return answer

    @staticmethod
    def action(state: BehaviorResearchState):
        # llm = DogFeatureInteractionAgent.LLM.bind_tools([DogFeatureInteractionAgent.handler_input])
        llm = DogFeatureInteractionAgent.LLM
        DogFeatureInteractionAgent.greetings()

        background_story = textwrap.dedent("""
    You are a highly knowledgeable canine behaviorist with expertise in training techniques, breed characteristics, and canine health.
    Your primary objective is to ensure that the training plan is safe and effective for the specific dog in question. 
    Your secondary objective is to retrieve additional information to tune the training plan for the dog's specific preferences
    You are particularly skilled at identifying potential breed-related constraints and health considerations.
        """)

        task_prompt = textwrap.dedent("""
            Your team is working on a training plan to teach the behavior {behavior} to dog. Your team
            members have already drafted an initial outline of a training plan. Assess the feasibility of the provided 
            dog training plan, considering the health and breed-specific needs of the dog.
        
            PLAN_OUTLINE: 
            {plan_outline}
            
            We already have additional information about the dog from the dog handler. Take into account the following 
            information about the dog. Do not ask the same questions again.
            
            {dog_details}
            
            Analyze the provided training plan for teaching the dog the specified behavior. Consider whether any part 
            of the plan might be influenced by the dog's health, age breed or training preferences. If health or breed 
            constraints are relevant, interact with the dog handler to gather additional information. If the plan
            includes rewarding the dog, ask the handler for reward preferences of his dog.
            Only ask questions directly
            related to the behavior and the training plan's suitability.
            
            Here are some examples:
            - If the behaviour is sit, you realize that this involves the hips. Many dogs, especially of larger breeds,
              might have problems with sitting due to their hip joints. You ask the handler about the dog's hip joints
              and if they are prone to hip dysplasia.
            - If the behaviour is fetch, you realize that this involves the neck. Many dogs, especially of smaller breeds,
              might have problems with fetching due to their neck joints. You ask the handler about the dog's neck joints
              and if they are prone to neck dysplasia.
            - if the behaviour includes physical exercise, you realize that this might involve panting. Many dogs,
              especially of brachycephalic breeds, might have problems with panting due to their respiratory system.
              You ask the handler about the dog's respiratory system and if they are prone to respiratory issues.
            - You should also consider the feasibility of the plan. For example, if the plan involves a lot of jumping,
              you might want to ask the handler about the dog's jumping ability and if it is suitable for the behavior.
            - If the training plan includes rewarding the dog, you realize that rewarding is very dog specific. You ask
              the handler about reward preferences of his dog.
              
            Remember to ask the handler about the specifics of the plan and not just the general behavior. The handler
            is a person who intimately knows the dog and is able to provide additional information the dog.
            
            Return a JSON with the single key "questions" and the value is the list of all questions you have to
            the dog handler.
        """)
        formatted_dog_details = "\n".join([f"- {d[0]}: {d[1]}" for d in state["dog_details"]])
        messages = [
            SystemMessage(content=background_story),
            HumanMessage(content=task_prompt.format(
                behavior=state["question"],
                plan_outline=state["outline_plan"],
                dog_details=formatted_dog_details,
            ))
        ]
        response = llm.invoke(messages)
        print("Please answer the following question to help us tune the training plan for you and your dog!")
        handler_information = []
        questions_json = json.loads(response.content)
        for question in questions_json["questions"]:
            answer = DogFeatureInteractionAgent.handler_input(question)
            handler_information.append((question, answer))

        return {"new_dog_details": handler_information}
