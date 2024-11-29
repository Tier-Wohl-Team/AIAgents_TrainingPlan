from langgraph.constants import START, END
from langgraph.graph import StateGraph

from agents.BehaviorHandlerInteraction import BehaviorHandlerInteraction
from agents.InternetResearcher import InternetResearcher
from agents.OutlineWriter import OutlineWriter
from states.state_types import BehaviorResearchState

def should_continue(state):
    if state.get("outline_plan", "") == "I need more information.":
        if not "internet_research_results" in state:
            return InternetResearcher.NAME
        elif not "asked_human" in state:
            return BehaviorHandlerInteraction.NAME
        else:
            return "collector"
    else:
        return "collector"

def collector(state):
    print("in collector")
    print(state)
    return {"ask_human": True}

behavior_research_team_builder = StateGraph(BehaviorResearchState)
behavior_research_team_builder.add_node(OutlineWriter.NAME, OutlineWriter.action)
behavior_research_team_builder.add_node(InternetResearcher.NAME, InternetResearcher.action)
behavior_research_team_builder.add_node(BehaviorHandlerInteraction.NAME, BehaviorHandlerInteraction.action)
behavior_research_team_builder.add_node("collector", collector)
behavior_research_team_builder.add_edge(START, OutlineWriter.NAME)
behavior_research_team_builder.add_conditional_edges(
    OutlineWriter.NAME,
    should_continue,
    [InternetResearcher.NAME, BehaviorHandlerInteraction.NAME, "collector"]
)
behavior_research_team_builder.add_edge(InternetResearcher.NAME, OutlineWriter.NAME)
behavior_research_team_builder.add_edge(BehaviorHandlerInteraction.NAME, OutlineWriter.NAME)
behavior_research_team_builder.add_edge("collector", END)

behavior_research_team = behavior_research_team_builder.compile()
