# %% define the state
import operator
from typing import TypedDict, List, Annotated


class TeamState(TypedDict):
    question: str
    outline_plan: str
    plans: Annotated[list[tuple[str, str]], operator.add]
