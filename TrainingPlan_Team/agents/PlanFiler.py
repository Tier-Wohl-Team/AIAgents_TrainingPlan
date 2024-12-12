from pathlib import Path

from agents.BaseAgent import BaseAgent
from states.state_types import TeamState


class PlanFiler(BaseAgent):
    NAME = "PlanFiler"

    @staticmethod
    def action(state: TeamState):
        PlanFiler.greetings()

        # Create a unique folder in 'plans/'
        plans_dir = Path('plans')
        plans_dir.mkdir(exist_ok=True)

        unique_folder_name = f"plan_{len(list(plans_dir.iterdir())) + 1}"
        unique_folder_path = plans_dir / unique_folder_name
        unique_folder_path.mkdir()

        # Write the final plan to 'final_plan.md'
        final_plan_path = unique_folder_path / "final_plan.md"
        with final_plan_path.open('w', encoding='utf-8') as final_plan_file:
            final_plan_file.write(state['final_plan'])

        # Write each plan in the 'plans' list to a separate file
        for idx, (_, plan_content) in enumerate(state['plans'], start=1):
            plan_file_path = unique_folder_path / f"plan_{idx}.md"
            with plan_file_path.open('w', encoding='utf-8') as plan_file:
                plan_file.write(plan_content)

        print(f"Plans saved successfully in {unique_folder_path}.")
