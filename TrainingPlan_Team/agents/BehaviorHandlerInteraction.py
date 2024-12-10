import textwrap

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from agents.BaseAgent import BaseAgent
from states.state_types import BehaviorResearchState


class BehaviorHandlerInteraction(BaseAgent):
    NAME = "HandlerInteraction"
    LLM_MODEL = "gpt-4o-mini"
    LLM = ChatOpenAI(temperature=0.0, model_name=LLM_MODEL)

    handler_input_method = input  # Can I use this to for web based input?

    @staticmethod
    @tool
    def handler_input(query: str):
        """
        Ask the handler of the animal for additional information about the animal or the behavior.
        The input should be a question for the human.
        """
        print(f"{query}")
        answer = BehaviorHandlerInteraction.handler_input_method("Your answer: ")
        return answer

    @staticmethod
    def action(state: BehaviorResearchState):
        llm = BehaviorHandlerInteraction.LLM.bind_tools([BehaviorHandlerInteraction.handler_input])
        BehaviorHandlerInteraction.greetings()

        background_story = textwrap.dedent("""
            You are an experienced animal trainer with a deep understanding of animal behavior and training techniques.
            What sets you apart is your ability to communicate with the handler of the animal to gather additional information
            about the behavior. Your clients love your patience and you politeness in your communication.
        """)

        task_prompt = textwrap.dedent("""
            Your team is working on a training plan to teach the behavior {behavior} to an animal. The team has already
            performed an internet research on the behavior and has gathered the following information:
            
            {internet_research_results}
            
            Even with these results, the team is still unsure how to teach the behavior to the animal. Your task is to
            get additional information about the behavior from the handler of the animal. The handler is a person who 
            is familiar with the behavior and is able to provide additional information about the behavior. Make sure
            that we have a defined and measurable status of the behavior and a defined and measurable goal. Without
            this information, we cannot train the behavior.
        """)

        messages = [
            SystemMessage(content=background_story),
            HumanMessage(content=task_prompt.format(
                behavior=state["question"],
                internet_research_results="\n\n".join(state.get("internet_research_results", []))
            ))
        ]
        response = llm.invoke(messages)

        handler_information = []
        for tool_call in response.tool_calls:
            if tool_call["name"] == "handler_input":
                query = tool_call["args"]["query"]
                answer = BehaviorHandlerInteraction.handler_input(query)
                handler_information.append({"query": query, "answer": answer})
        return {"handler_input": handler_information, "asked_human": True}


#%% test
# start_state = BehaviorResearchState(
#     question="My dog sits for 10 seconds. I want to extend this duration to 25 seconds.",
#     internet_research_results=["The dog is a canine."],
# )
# result = BehaviorHandlerInteraction.action(start_state)
# print(result)
