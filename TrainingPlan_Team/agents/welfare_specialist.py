import textwrap
import dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from states.BehaviorState import BehaviorState
dotenv.load_dotenv("TrainingPlan_Team/.env")

llm = ChatOpenAI(temperature=0.0, model_name="gpt-4o-mini")

def welfare_specialist(state: BehaviorState):
    """Welfare Specialist
    This agent is responsible for rewriting the training plan to make it more animal friendly.
    """
    print("Improving the animal welfare aspect of the plan")
    background_story = textwrap.dedent("""
                You are an expert in animal welfare regulations with a deep understanding of humane 
                treatment. Your primary responsibility is to improve training plans and ensure they 
                comply with animal welfare standards. Especially, you ensurer that the used methods are
                based on positive reinforcement and do not use any aversives.
            """)
    task_prompt = textwrap.dedent("""
            Here is a training plan to teach a dog the behaviour "{behavior}". 
            
            TRAINING_PLAN:
            {training_plan}
            
            Do not delete any of the training steps, variations or repetitions. You might add some 
            information for the steps and repetitions on how the trainer can make this training even 
            more animal friendly.
        """)
    messages = [
        SystemMessage(content=background_story),
        HumanMessage(content=task_prompt.format(
            behavior=state["behavior"],
            training_plan=state["draft_plan"]
        ))
    ]
    response = llm.invoke(messages)
    return {"plans": [(state["task"], response.content)]}

