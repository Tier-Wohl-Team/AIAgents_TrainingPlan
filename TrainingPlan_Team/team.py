# %% settings
import sys
import os
sys.path.append(os.path.abspath("./TrainingPlan_Team"))  # Adjust the path as needed

from langgraph.constants import START, END
from langgraph.graph import StateGraph


from teams.SpecialistWelfareTeam import SpecialistWelfareTeam
from states.state_types import TeamState, BehaviorState
from agents.OutlineWriter import OutlineWriter
from agents.SpecialistsTeamLeader import SpecialistsTeamLeader
from agents.DistanceDurationSpecialist import DistanceDurationSpecialist
from agents.CueSpecialist import CueSpecialist
from agents.DistractionSpecialist import DistractionSpecialist
from agents.Generalist import Generalist

# %% Build mini teams
distance_duration_team = SpecialistWelfareTeam(name="Distance Duration Team", specialist_agent=DistanceDurationSpecialist)
cue_team = SpecialistWelfareTeam(name="Cue Specialist Team", specialist_agent=CueSpecialist)
distraction_team = SpecialistWelfareTeam(name="Distraction Specialist Team", specialist_agent=DistractionSpecialist)
generalist_team = SpecialistWelfareTeam(name="Generalist Team", specialist_agent=Generalist)

SpecialistsTeamLeader.task_team_mapping({
    "duration": distance_duration_team.name,
    "distance": distance_duration_team.name,
    "cue introduction": cue_team.name,
    "distraction": distraction_team.name,
    "other": generalist_team.name,
})
# %% collector - dummy agent for testing
def collector(state: TeamState):
    return {"final_plan": "We collected all the plans."}

# %% build the team graph
team_graph_builder = StateGraph(TeamState)
team_graph_builder.add_node(OutlineWriter.NAME, OutlineWriter.action)
team_graph_builder.add_node(distance_duration_team.name, distance_duration_team.graph)
team_graph_builder.add_node(cue_team.name, cue_team.graph)
team_graph_builder.add_node(distraction_team.name, distraction_team.graph)
team_graph_builder.add_node(generalist_team.name, generalist_team.graph)
team_graph_builder.add_node("Collector", collector)
team_graph_builder.add_edge(START, OutlineWriter.NAME)
team_graph_builder.add_conditional_edges(OutlineWriter.NAME, SpecialistsTeamLeader.action,
                                    [distance_duration_team.name, cue_team.name, distraction_team.name, generalist_team.name])
team_graph_builder.add_edge(distance_duration_team.name, "Collector")
team_graph_builder.add_edge(cue_team.name, "Collector")
team_graph_builder.add_edge(distraction_team.name, "Collector")
team_graph_builder.add_edge(generalist_team.name, "Collector")
team_graph_builder.add_edge("Collector", END)
# %% compile the team graph
team = team_graph_builder.compile()

# %% test graph
# question = "When retrieving,, my dog drops the ball 2 meters in front of me. I want him to deliver to hand."
# for s in team.stream({"question": question}):
#   print(s)

