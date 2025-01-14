from langgraph.constants import START, END
from langgraph.graph import StateGraph

from agents.WelfareSpecialist import WelfareSpecialist
from states.state_types import BehaviorState

class SpecialistWelfareTeam:
    """A team of a specialist agent and a welfare specialist."""

    def __init__(self, name, specialist_agent, max_welfare_iterations=3):
        self.name = name
        self.graph = self._create_team_graph(specialist_agent, max_welfare_iterations)
        self.specialist_agent = specialist_agent

    @staticmethod
    def _create_team_graph(specialist_agent, max_welfare_iterations):

        """Create a team graph dynamically with the specified specialist agent."""
        def should_continue(state):
            # print("In should_continue")
            # print(state)
            if state.get("welfare_review", "") == "The plan is good.":
                state["is_finished"] = True
                state["plans"] = [(state["task"], state["draft_plan"])]
                return "collect_plan"

            if state.get("iteration_count", 0) >= max_welfare_iterations:
                state["plans"] = [(state["task"], "No Plan")]
                return "collect_plan"

            state["iteration_count"] = state.get("iteration_count", 0) + 1
            return specialist_agent.NAME

        def collect_plan(state):
            return {"plans": [(state["task"], state["draft_plan"])]}

        # Build the graph
        graph_builder = StateGraph(BehaviorState)
        graph_builder.add_node(specialist_agent.NAME, specialist_agent.action)
        graph_builder.add_node(WelfareSpecialist.NAME, WelfareSpecialist.action)
        graph_builder.add_node("collect_plan", collect_plan)
        graph_builder.add_edge(START, specialist_agent.NAME)
        graph_builder.add_edge(specialist_agent.NAME, WelfareSpecialist.NAME)
        graph_builder.add_conditional_edges(
            WelfareSpecialist.NAME,
            should_continue,
            [specialist_agent.NAME, "collect_plan"],
        )
        graph_builder.add_edge("collect_plan", END)
        return graph_builder.compile()

# # %%
# from agents.DistanceDurationSpecialist import DistanceDurationSpecialist
# start_state = BehaviorState(
#     task="My dog sits for 10 seconds. I want to extend this duration to 25 seconds.",
#     behavior="sit",
#     mode="duration",
#     status=10,
#     goal=25,
#     draft_plan="",
#     welfare_review="",
#     iteration_count=0,
#     is_finished=False,
#     plans=[]
# )
# distance_team = SpecialistWelfareTeam(name="Distance Duration Team", specialist_agent=DistanceDurationSpecialist)
# for s in distance_team.graph.stream(start_state):
#     print(s)