# %% define the state
import operator
from typing import TypedDict, List, Annotated
from langgraph.graph import add_messages


class TeamState(TypedDict):
    question: str
    internet_research_results: List[str]
    handler_input: Annotated[List[tuple[str, str]], operator.add]
    outline_plan: str
    dog_details: Annotated[List[tuple[str, str]], operator.add]
    plans: Annotated[List[tuple[str, str]], operator.add]
    final_plan: str


class ClientInteractionState(TypedDict):
    messages: Annotated[list, add_messages]
    question: str


class BehaviorState(TypedDict):
    task: str
    behavior: str
    dog_details: Annotated[List[tuple[str, str]], operator.add]
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
    dog_details: Annotated[List[tuple[str, str]], operator.add]
    new_dog_details: Annotated[List[tuple[str, str]], operator.add]
    iteration_count: int
    is_finished: bool
