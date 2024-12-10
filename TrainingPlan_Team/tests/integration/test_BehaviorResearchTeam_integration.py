# %% imports
from states.state_types import BehaviorResearchState
from teams.BehaviorResearchTeam import BehaviorResearchTeam

behavior_research_team = BehaviorResearchTeam(name="Behavior Research Team").graph

# #%% test graph
# question = "How do I teach my dog to extend sitting duration?"
# start_state = BehaviorResearchState(
#     question=question,
#    )
# for s in behavior_research_team.graph.stream(start_state):
#     print(s)