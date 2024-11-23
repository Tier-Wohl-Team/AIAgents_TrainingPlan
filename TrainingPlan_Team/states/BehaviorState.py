import operator
from typing import TypedDict, Annotated


class BehaviorState(TypedDict):
    task: str
    behavior: str
    mode: str
    status: float
    goal: float
    draft_plan: str
    plans: Annotated[list[tuple[str, str]], operator.add]
