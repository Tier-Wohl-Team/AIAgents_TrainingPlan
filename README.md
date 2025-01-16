
<!-- ![An example Training Plan Team Implementation](documentation/images/TrainingPlanTeam.excalidraw.png) -->
<div style="display: flex; justify-content: center; align-items: center;">
    <img src="documentation/images/TrainingPlanTeam.excalidraw.png" width="640" 
        alt="An example Training Plan Team Implementation">
</div>

# AI Agents for Animal Training Plan Generation

## Abstract
Effective animal training depends on well-structured training plans that ensure consistent progress and
measurable outcomes. However, the creation of such plans is often time-intensive, repetitive, and detracts from
hands-on training. Recent advancements in generative AI powered by large language models (LLMs) provide
potential solutions but frequently fail to produce actionable, individualized plans tailored to specific
contexts. This limitation is particularly significant given the diverse tasks performed by dogs—ranging from
working roles in military and police operations to competitive sports—and the varying training philosophies
among practitioners. To address these challenges, a modular agentic workflow framework is proposed, leveraging
LLMs while mitigating their shortcomings. By decomposing the training plan generation process into specialized
building blocks—autonomous agents that handle subtasks such as structuring progressions, ensuring welfare
compliance, and adhering to team-specific standard operating procedures (SOPs)—this approach facilitates the
creation of specific, actionable plans. The modular design further allows workflows to be tailored to the unique
requirements of individual tasks and philosophies. As a proof of concept, a complete training plan generation
workflow is presented, integrating these agents into a cohesive system. This framework prioritizes
flexibility and adaptability, empowering trainers to create customized solutions while leveraging generative
AI’s capabilities. In summary, agentic workflows bridge the gap between cutting-edge technology and the
practical, diverse needs of the animal training community. As such, they could form a crucial foundation for
advancing computer-assisted animal training methodologies.

## Installation

### Requirements
- Python 3.11 (LangGraph CLI only supports Python >= 3.11.)
- OpenAI API Key (if you want to use the default LLM gtp-4o-mini)
- Tavily API Key (if you want to use the *Internet Research Agent*)

### Installation
1. Clone the repository

    `git clone https://github.com/Tier-Wohl-Team/AIAgents_TrainingPlan.git`
2. change into directory

    `cd AIAgents_TrainingPlan`
3. Create virtual environment
   
    (i) If your python version is < 3.11, you might use conda to build a 3.11 environment

       ```
       conda create -n AIAgents_TrainingPlan python=3.11
       conda activate
       ```
    (ii) **Or** use venv

       ```
       python3 -m venv AIAgents_TrainingPlan`
       source AIAgents_TrainingPlan/bin/activate
       ```
4. Install the required packages using pip (in local environment)

   `pip install -r requirements.txt`
5. Add your API keys

   copy `TrainingPlan_Team/.env_sample` to `TrainingPlan_Team/.env` and add your API keys

## Usage

### Command line
`python .\start_TrainingPlanTeam.py`

### With LangGraph Studio (local)
see also https://langchain-ai.github.io/langgraph/how-tos/local-studio/

`langgraph dev`

Use `CTRL-C`to stop

### Running Tests
1. Unit Testing

    `pytest -m unit`
2. Integration Testing (calling the LLM)

    `pytest -m integration`
3. Probabilistic Output Validation

    `pytest -m llm`
