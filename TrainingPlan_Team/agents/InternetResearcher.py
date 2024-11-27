import os
import textwrap
from typing import List

import dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from agents.BaseAgent import BaseAgent
from states.state_types import TeamState
from tavily import TavilyClient
dotenv.load_dotenv("TrainingPlan_Team/.env")

class InternetResearcher(BaseAgent):
    """
    This agent is responsible for gathering relevant information about the target behavior from the internet.
    """
    NAME = "InternetResearcher"
    LLM_MODEL = "gpt-4o-mini"
    LLM = ChatOpenAI(temperature=0.0, model_name=LLM_MODEL)
    TAVILY = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

    class Queries(BaseModel):
        """Queries
        Helper class for formatted output
        """
        queries: List[str]

    @staticmethod
    def action(state: TeamState):
        llm = InternetResearcher.LLM
        InternetResearcher.greetings()
        tavily = InternetResearcher.TAVILY

        background_story = textwrap.dedent("""
            You are an animal behavior enthusiast with extensive knowledge in the science of animal training.
            Your passion for understanding animal behavior is matched only by your love for using computers to 
            conduct thorough research. You are the go-to person for any in-depth research needed on animal 
            behavior, and you provide valuable insights to enhance training plans.
        """)

        task_prompt = textwrap.dedent("""
            Here's a query from our client:
            
            QUERY
            -----
            {question}
            
            Based on this query generate a list of search queries that will gather any relevant information about 
            how this animal training query can be addressed. Make sure to search only for approaches that use positive 
            reinforcement and avoid aversives.
            Only generate 3 queries max.
        """)

        queries = llm.with_structured_output(InternetResearcher.Queries).invoke([
            SystemMessage(content=background_story),
            HumanMessage(content=task_prompt.format(question=state["question"]))
        ])
        internet_search_result = []
        for q in queries.queries:
            response = tavily.search(query=q, max_results=2)
            for r in response['results']:
                internet_search_result.append(r['content'])
        return {"internet_research_results": internet_search_result}