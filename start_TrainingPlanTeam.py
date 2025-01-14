from TrainingPlan_Team.team import team

config = {"configurable": {"thread_id": "1"}}
question = ""
for s in team.stream({"question": ""}, config=config):
    print(s)