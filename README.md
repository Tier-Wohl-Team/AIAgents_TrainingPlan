
<!-- ![An example Training Plan Team Implementation](documentation/images/TrainingPlanTeam.excalidraw.png) -->
<div style="display: flex; justify-content: center; align-items: center;">
    <img src="documentation/images/TrainingPlanTeam.excalidraw.png" width="640" 
        alt="An example Training Plan Team Implementation">
</div>

# AI Agents for Animal Training Plan Generation


## Installation

### Requirements
- Python 3.10 or higher (tested with 3.10 and 3.11)
- OpenAI API Key (if you want to use the default LLM gtp-4o-mini)
- Tavily API Key (if you want to use the *Internet Research Agent*)

### Installation
1. Clone the repository

    `git clone https://github.com/JoergSchultz-TWT/AIAgents_TrainingPlan_Dev.git`
2. change into directory

    `cd AIAgents_TrainingPlan_Dev`
3. Create virtual environment

    `python3.x -m venv AIAgents_TrainingPlan_Dev`
4. Activate environment

    `source AIAgents_TrainingPlan_Dev/bin/activate`
5. Install the required packages using pip (in local environment)

   `pip install -r requirements.txt`
6. Create `.env`

   copy `.env_sample` to `.env` and add your API keys

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