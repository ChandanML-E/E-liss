import warnings
from langchain_groq import ChatGroq
from langchain.agents import create_react_agent, AgentExecutor
from tools.react_prompt_template import get_prompt_template
from tools.pdf_query_tools import pdf_query_tool


def agent(query: str):
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Initialize the Language Model
    LLM = ChatGroq(model="llama3-8b-8192")

    # Define tools for the agent
    tools = [pdf_query_tool]

    # Load prompt template
    prompt_template = get_prompt_template()

    # Create the reactive agent
    trading_agent = create_react_agent(
        LLM,
        tools,
        prompt_template
    )

    # Create the agent executor
    agent_executor = AgentExecutor(
        agent=trading_agent, tools=tools, verbose=False, handle_parsing_errors=True
    )

    # Execute the query and fetch results
    result = agent_executor.invoke({"input": query})

    return result["output"]
