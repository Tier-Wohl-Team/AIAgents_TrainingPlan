import pytest
from agents.InternetResearcher import InternetResearcher
from states.state_types import TeamState

@pytest.mark.integration
def test_internet_researcher_integration_valid():
    """Integration test with real LLM and TavilyClient."""
    # Define a sample state
    state = TeamState(
        question="What are the best methods for training a dog to stop barking?"
    )

    # Call the action method
    result = InternetResearcher.action(state)

    # Assertions
    assert "internet_research_results" in result, "The result dictionary is missing the expected key."
    assert isinstance(result["internet_research_results"], list), "Results should be a list."
    assert len(result["internet_research_results"]) > 0, (
        "Expected non-empty research results from the Tavily API."
    )
    for content in result["internet_research_results"]:
        assert isinstance(content, str), "Each result content should be a string."
        print(content)

