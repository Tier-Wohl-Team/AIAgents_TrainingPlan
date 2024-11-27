import pytest
from unittest.mock import patch, MagicMock
from agents.InternetResearcher import InternetResearcher
from states.state_types import TeamState


@pytest.mark.unit
@patch("agents.InternetResearcher.InternetResearcher.LLM", autospec=True)
@patch("agents.InternetResearcher.InternetResearcher.TAVILY", autospec=True)
def test_internet_researcher_action_valid_response(mock_tavily, mock_llm):
    """Test the InternetResearcher action method with valid responses."""
    # Mock LLM response
    mock_llm_response = MagicMock()
    mock_llm_response.queries = ["query 1", "query 2", "query 3"]
    mock_llm.with_structured_output.return_value.invoke.return_value = mock_llm_response

    # Mock TavilyClient response
    mock_tavily.search.return_value = {
        "results": [
            {"content": "Mocked result 1"},
            {"content": "Mocked result 2"}
        ]
    }

    # Define a sample state
    state = TeamState(
        question="How can I train a dog to sit on command?"
    )

    # Call the action method
    result = InternetResearcher.action(state)

    # Assertions for the final result
    assert "internet_research_results" in result, "The result dictionary is missing the expected key."
    assert len(result["internet_research_results"]) == 6, (
        "The number of research results does not match the mocked data."
    )
    assert result["internet_research_results"] == [
        "Mocked result 1",
        "Mocked result 2",
        "Mocked result 1",
        "Mocked result 2",
        "Mocked result 1",
        "Mocked result 2"
    ], "The internet research results are incorrect."

    # Assertions for LLM
    mock_llm.with_structured_output.assert_called_once_with(InternetResearcher.Queries)
    mock_llm.with_structured_output.return_value.invoke.assert_called_once()

    # Assertions for Tavily
    assert mock_tavily.search.call_count == 3, "The number of Tavily searches does not match the number of queries."
    mock_tavily.search.assert_any_call(query="query 1", max_results=2)
    mock_tavily.search.assert_any_call(query="query 2", max_results=2)
    mock_tavily.search.assert_any_call(query="query 3", max_results=2)

