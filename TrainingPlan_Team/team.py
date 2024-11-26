# %% settings
from langgraph.constants import START, END
from langgraph.graph import StateGraph
import sys
import os


# Add the directory containing your package to sys.path
# print(os.path.abspath("./TrainingPlan_Team"))
sys.path.append(os.path.abspath("./TrainingPlan_Team"))  # Adjust the path as needed
# %% import the agents
from states.state_types import TeamState, BehaviorState
from agents.DistanceDurationSpecialist import agent as distance_duration_agent
from agents.WelfareSpecialist import welfare_specialist
from agents.OutlineWriter import OutlineWriter
from agents.SpecialistsTeamLeader import SpecialistsTeamLeader

# %% distance duration team
distance_duration_welfare_builder = StateGraph(BehaviorState)
distance_duration_welfare_builder.add_node("distance_duration_agent", distance_duration_agent)
distance_duration_welfare_builder.add_node("welfare_specialist", welfare_specialist)
distance_duration_welfare_builder.add_edge(START, "distance_duration_agent")
distance_duration_welfare_builder.add_edge("distance_duration_agent", "welfare_specialist")
distance_duration_welfare_builder.add_edge("welfare_specialist", END)
distance_duration_welfare_graph = distance_duration_welfare_builder.compile()

# %% build the team graph
team_graph_builder = StateGraph(TeamState)
team_graph_builder.add_node(OutlineWriter.NAME, OutlineWriter.action)
team_graph_builder.add_node("distance_duration_welfare_graph", distance_duration_welfare_graph)
team_graph_builder.add_edge(START, "plan_outline_writer")
team_graph_builder.add_conditional_edges("plan_outline_writer", SpecialistsTeamLeader.action,
                                    ["distance_duration_welfare_graph"])
team_graph_builder.add_edge("distance_duration_welfare_graph", END)

# %% compile the team graph
team = team_graph_builder.compile()

# %% test graph
# question = "My dog sits for 10 seconds. I want to extend this duration to 25 seconds."
# for s in team.stream({"question": question}):
#     print(s)

