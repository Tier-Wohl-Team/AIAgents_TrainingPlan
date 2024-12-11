import pytest
from agents.WelfareSpecialist import WelfareSpecialist
from states.state_types import BehaviorState


@pytest.mark.integration
def test_welfare_specialist_integration_leash_correction():
    """Integration test: Plan with leash correction should not return 'The plan is good.'"""
    # Define the input state with a training plan using leash correction
    state = BehaviorState(
        behavior="sit",
        draft_plan="Step 1: Use a leash correction to enforce the sit position."
    )

    # Call the action method
    result = WelfareSpecialist.action(state)

    # Validate that the response is NOT 'The plan is good.'
    assert result["welfare_review"] != "The plan is good.", (
        "The LLM incorrectly approved a plan using leash correction."
    )
    print("Response for leash correction plan:", result["welfare_review"])


@pytest.mark.integration
def test_welfare_specialist_integration_treat_reward():
    """Integration test: Plan with treat reward should return 'The plan is good.'"""
    # Define the input state with a training plan using positive reinforcement
    state = BehaviorState(
        behavior="sit",
        draft_plan="Step 1: Use a treat to lure the dog into a sit position."
    )

    # Call the action method
    result = WelfareSpecialist.action(state)

    # Validate that the response is 'The plan is good.'
    assert result["welfare_review"] == "The plan is good.", (
        "The LLM did not approve a plan using treat rewards."
    )
    print("Response for treat reward plan:", result["welfare_review"])


@pytest.mark.integration
def test_welfare_specialist_integration_treat_health():
    """Integration test: Plan which with lots of running for a dog with health problems
    should not return 'The plan is good.'"""
    # Define the input state with a training plan using positive reinforcement
    state = BehaviorState(
        behavior="sit",
        draft_plan="Step 1: Get the dog motivated by throwing a ball and let him run a lot.",
        dog_details=[("Does the dog have any health problems?",
                      "As typical for French Bulldogs, he can have problems breathing.")]
    )

    # Call the action method
    result = WelfareSpecialist.action(state)

    # Validate that the response is not 'The plan is good.'
    assert result["welfare_review"] != "The plan is good.", (
        "The LLM did not approve a plan using treat rewards."
    )
    print("Response for health issued plan:", result["welfare_review"])
