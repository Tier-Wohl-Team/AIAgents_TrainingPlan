# An agentic workflow to generate personalized, actionable training plans

## Tasks

### Long Term Memory

Store the user input acquired by the BehaviorHandlerInteraction agent in the long term memory.

## Refs

- Paper showing that combinations of small models can outperform large models
- LLM can come up with better Reinforcement strategies than humans

## Agent Frameworks

| **Framework**    | **URL**                                                | **Summary**                                                                                                                                       | **Special Features**                                                                                       |
|------------------|--------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------|
| **OpenAI Swarm** | [OpenAI Swarm GitHub](https://github.com/openai)       | A framework enabling the deployment of multiple cooperative AI agents using OpenAI’s API. Agents interact to solve complex tasks collaboratively. | Emphasizes scalability and efficient task distribution in multi-agent setups.                              |
| **LangGraph**    | [LangGraph](https://github.com/langgraph)              | A graph-based agent orchestration framework designed for controlled, iterative workflows and safe AI interactions.                                | Modular design allows flexible and programmable workflows, ideal for research and safety-critical domains. |
| **AutoGen**      | [AutoGen GitHub](https://github.com/microsoft/autogen) | A Microsoft framework for building collaborative multi-agent systems with shared memory and task execution.                                       | Offers strong integration with memory systems and facilitates natural agent collaboration.                 |
| **CrewAI**       | [CrewAI GitHub](https://github.com/crewai)             | A framework specializing in team-based agent workflows, where agents take on defined roles and responsibilities.                                  | Focuses on role-based collaboration, making it suitable for task-specific teamwork and simulations.        |

When designing agentic workflows to generate training plans for living organisms, such as dogs, it is crucial to
prioritize control and reproducibility to ensure safe and consistent outcomes. LangGraph was chosen for this work
because its graph-based architecture enables precise orchestration of agent interactions and workflows. This structure
allows each step of the plan-generation process to be explicitly defined, monitored, and adjusted as needed, ensuring a
high degree of transparency and traceability. Unlike more autonomous frameworks that emphasize rapid iteration or
emergent behaviors, LangGraph provides fine-grained control over agent operations, minimizing the risk of unintended
actions. Its modular design not only allows for the incorporation of domain-specific constraints but also supports the
implementation of automated tests at each stage of the workflow. These tests further enhance safety and reproducibility
by validating the correctness of intermediate outputs and ensuring compliance with predefined standards. These features
make LangGraph particularly well-suited for applications involving living beings, where safety, predictability, and
ethical considerations are paramount.