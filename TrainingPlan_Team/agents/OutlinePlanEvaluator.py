import textwrap

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from agents.BaseAgent import BaseAgent
from states.state_types import BehaviorResearchState

class OutlinePlanEvaluator(BaseAgent):
    NAME = "OutlinePlanEvaluator"
    LLM_MODEL = "gpt-4o-mini"
    LLM = ChatOpenAI(model_name=LLM_MODEL, temperature=0.0)

    @staticmethod
    def action(state: BehaviorResearchState):
        llm = OutlinePlanEvaluator.LLM
        OutlinePlanEvaluator.greetings()

        background_story = textwrap.dedent("""
            You are an experienced dog trainer with a focus on health and breed-specific training. Your team
            members have written an outline for a training plan. Now, we've got additional information about
            the dog from the dog handler. You need to evaluate the plan and decide whether it needs to be
            rewritten or not.
        """)
        task_prompt = textwrap.dedent("""
            Here's the current outline of the training plan:
            
            {outline_plan}
            
            And here are the additional details from the dog handler:
            
            {new_dog_details}
            
            Based on the information provided, evaluate whether the plan needs to be rewritten or not. If the plan 
            needs to be rewritten answer 'rewrite', otherwise answer 'no_rewrite'.
        """)

        messages = [
            SystemMessage(content=background_story),
            HumanMessage(content=task_prompt.format(
                outline_plan=state["outline_plan"],
                new_dog_details=state["new_dog_details"]
            ))
        ]
        response = llm.invoke(messages)
        print("OutlinePlanEvaluator response: ", response.content)
        updated_details = state["dog_details"] + state["new_dog_details"]
        return {"is_finished": response.content == "no_rewrite", "iteration_count": state.get("iteration_count",0) + 1,
                "dog_details": updated_details, "new_dog_details": []}