# %% settings
from langgraph.constants import START, END
from langgraph.graph import StateGraph
import sys
import os

# Add the directory containing your package to sys.path
# print(os.path.abspath("./TrainingPlan_Team"))
sys.path.append(os.path.abspath("./TrainingPlan_Team"))  # Adjust the path as needed
# %% import the agents
from states.TeamState import TeamState
from agents.outline_writer import agent as outline_writer


# %% build the team graph
team_graph_builder = StateGraph(TeamState)
team_graph_builder.add_node("plan_outline_writer", outline_writer)
team_graph_builder.add_edge(START, "plan_outline_writer")
team_graph_builder.add_edge("plan_outline_writer", END)

# %% compile the team graph
team = team_graph_builder.compile()

# %% test graph
# question = "My dog sits for 10 seconds. I want to extend this duration to 25 seconds."
# for s in team.stream({"question": question}):
#     print(s)

