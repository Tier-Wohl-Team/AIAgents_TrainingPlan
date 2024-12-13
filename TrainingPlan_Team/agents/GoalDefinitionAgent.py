import json
import textwrap

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from agents.BaseAgent import BaseAgent
from states.state_types import TeamState


class GoalDefinitionAgent(BaseAgent):
    NAME = "GoalDefinitionAgent"
    LLM_MODEL = "gpt-4o-mini"
    LLM = ChatOpenAI(temperature=0.0, model_name=LLM_MODEL,
                     model_kwargs={"response_format": {"type": "json_object"}})

    @staticmethod
    def action(state: TeamState):
        llm = GoalDefinitionAgent.LLM
        GoalDefinitionAgent.greetings()

        background_story = textwrap.dedent("""
            You are the client interaction agent of the Training Plan Team. Together with the client, you sharpen the
            training goal of the client to enable the best results of the training plan.
        """)
        task_prompt = textwrap.dedent("""
            The client has the following request:
                                      
            {question}
            
            From this question, extract the following information:
            
            - Current Status: If the current status is not a testable statement, say "Not defined".
            - Target Goal: If the current status is not a testable statement, say "Not defined".
            
            Return the information in JSON format. It should have two keys: "current_status" and "target_goal". 
            Both keys should have a string value.
            
            Example 1:
            My dog can sit for 30 seconds. I want to extend this duration to 1 minute.
                "current_status": "The dog can sit for 30 seconds.",
                "target_goal": "The dog should sit for 45 seconds."
            
            Example 2:
            My dog can walk at heel. I want to extend the distance to 20 meters.
                "current_status": "Not defined",
                "target_goal": "The dog should walk at heel for 20 meters."
        """)

        messages = [
            SystemMessage(content=background_story),
            HumanMessage(content=task_prompt.format(
                question=state["question"]
            ))
        ]
        response = llm.invoke(messages)

        goal = json.loads(response.content)

        return {"goal": goal}