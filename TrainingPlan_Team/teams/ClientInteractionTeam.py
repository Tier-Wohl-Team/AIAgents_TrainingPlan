# %% imports
import textwrap

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.prebuilt import create_react_agent
from states.state_types import ClientInteractionState


class ClientInteractionTeam:
    """A team of a client interaction agent."""

    NAME = "GoalDefinitionAgent"
    LLM_MODEL = "gpt-4o-mini"
    LLM = ChatOpenAI(temperature=0.0, model_name=LLM_MODEL)

    def __init__(self, name, ):
        self.name = name
        self.graph = self._create_team_graph()

    @staticmethod
    def saver(state: ClientInteractionState):
        behavior_to_train = state["messages"][-1].content
        return {"question": behavior_to_train}

    @staticmethod
    @tool
    def ask_client(question: str):
        """Use this to ask questions to the client."""
        # print(question)
        answer = input(question)
        return {"messages": [answer]}

    @staticmethod
    def _create_team_graph():
        llm = ClientInteractionTeam.LLM
        prompt = textwrap.dedent("""
            You are a member of a team which writes training plans for dogs. You are the contact point for the client. 
            Your job is to make to extract the following information from the interaction with the client.
            
            - What is the behaviour he wants to train?
            - What is the current status? We need a defined and measurable value here.
            - What should the result be? Again, we need a measurable behavior.
            - Any additional specifics of the behavior the plan should consider.
            
            If you can extract it from the current interaction with the client, format the result like this:
            
            **Behaviour to train**: Sit
            **Current Status**: The dog sits for 30 seconds
            **Goal**: The dog sits for 1 minute
            **Additional Specifics**: The dog should stay in the sit when I throw a ball.
            
            If you need additional information, acquire these details from the client. You can use a tool for this.
            If there is no interaction with the client yet, use the tool to introduce yourself and ask the client what behavior
            he wants to train.
        """)

        memory = MemorySaver()
        tools = [ClientInteractionTeam.ask_client]
        react_agent = create_react_agent(llm, tools=tools, state_modifier=prompt,
                                    checkpointer=memory)

        graph_memory = MemorySaver()
        graph_builder = StateGraph(ClientInteractionState)
        graph_builder.add_node("chatbot", react_agent)
        graph_builder.add_node("saver", ClientInteractionTeam.saver)
        graph_builder.add_edge(START, "chatbot")
        graph_builder.add_edge("chatbot", "saver")
        graph_builder.add_edge("saver", END)
        client_interaction_team = graph_builder.compile(checkpointer=graph_memory)
        return client_interaction_team
