# %% define the state
import operator
from typing import TypedDict, List, Annotated


class TeamState(TypedDict):
    question: str
    internet_research_results: List[str]
    handler_input: Annotated[List[tuple[str, str]], operator.add]
    outline_plan: str
    plans: Annotated[List[tuple[str, str]], operator.add]
    final_plan: str


class BehaviorState(TypedDict):
    task: str
    behavior: str
    mode: str
    status: float
    goal: float
    draft_plan: str
    welfare_review: str
    iteration_count: int
    is_finished: bool
    plans: Annotated[List[tuple[str, str]], operator.add] # This should overwrite the TeamState plans

class BehaviorResearchState(TypedDict):
    question: str
    internet_research_results: List[str]
    handler_input: Annotated[List[tuple[str, str]], operator.add]
    asked_human: bool
    outline_plan: str
    wrote_plan: bool
