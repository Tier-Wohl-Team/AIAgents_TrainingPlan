# %% imports and settings
from langgraph.constants import START, END
from langgraph.graph import StateGraph

from agents.BehaviorHandlerInteraction import BehaviorHandlerInteraction
from agents.DogFeatureInteractionAgent import DogFeatureInteractionAgent
from agents.InternetResearcher import InternetResearcher
from agents.OutlinePlanEvaluator import OutlinePlanEvaluator
from agents.OutlineWriter import OutlineWriter
from states.state_types import BehaviorResearchState

class BehaviorResearchTeam:
    """A team of a behavior research specialist and a handler interaction specialist."""
    MAX_ITERATIONS = 2

    def __init__(self, name):
        self.name = name
        self.graph = self._create_team_graph()

    @staticmethod
    def _create_team_graph():

        def should_get_more_infos(state):
            if state.get("outline_plan", "") == "I need more information from the internet.":
                return InternetResearcher.NAME
            if state.get("outline_plan", "") == "I need more information from the dog handler.":
                return BehaviorHandlerInteraction.NAME
            return DogFeatureInteractionAgent.NAME

        def should_rewrite(state):
            if (state.get("is_finished", True) or
                    state.get("iteration_count", 0) >= BehaviorResearchTeam.MAX_ITERATIONS):
                return END
            else:
                return OutlineWriter.NAME

        # %% build the graph
        behavior_research_team_builder = StateGraph(BehaviorResearchState)
        behavior_research_team_builder.add_node(OutlineWriter.NAME, OutlineWriter.action)
        behavior_research_team_builder.add_node(InternetResearcher.NAME, InternetResearcher.action)
        behavior_research_team_builder.add_node(BehaviorHandlerInteraction.NAME, BehaviorHandlerInteraction.action)
        behavior_research_team_builder.add_node(DogFeatureInteractionAgent.NAME, DogFeatureInteractionAgent.action)
        behavior_research_team_builder.add_node(OutlinePlanEvaluator.NAME, OutlinePlanEvaluator.action)
        behavior_research_team_builder.add_edge(START, OutlineWriter.NAME)
        behavior_research_team_builder.add_conditional_edges(
            OutlineWriter.NAME,
            should_get_more_infos,
            [InternetResearcher.NAME, BehaviorHandlerInteraction.NAME, DogFeatureInteractionAgent.NAME]
        )
        behavior_research_team_builder.add_edge(InternetResearcher.NAME, OutlineWriter.NAME)
        behavior_research_team_builder.add_edge(BehaviorHandlerInteraction.NAME, OutlineWriter.NAME)
        behavior_research_team_builder.add_edge(DogFeatureInteractionAgent.NAME, OutlinePlanEvaluator.NAME)
        behavior_research_team_builder.add_conditional_edges(
            OutlinePlanEvaluator.NAME,
            should_rewrite,
            [OutlineWriter.NAME, END]
        )

        behavior_research_team = behavior_research_team_builder.compile()
        return behavior_research_team

# %% test graph
# question = "How do I teach my dog to extend sitting duration?"
# start_state = BehaviorResearchState(
#     question=question,
#   )
# for s in behavior_research_team.stream(start_state):
#     print(s)
# #
# # %% test graph
# question = "How do I teach my dog laser directionals?"
# start_state = BehaviorResearchState(
#     question=question,
# )
# for s in behavior_research_team.stream(start_state):
#     print(s)