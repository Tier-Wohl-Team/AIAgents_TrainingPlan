# %% define the state
import operator
from typing import TypedDict, List, Annotated


class TeamState(TypedDict):
    question: str
    outline_plan: str
    plans: Annotated[List[tuple[str, str]], operator.add]


class BehaviorState(TypedDict):
    task: str
    behavior: str
    mode: str
    status: float
    goal: float
    draft_plan: str
    plans: Annotated[list[tuple[str, str]], operator.add]
