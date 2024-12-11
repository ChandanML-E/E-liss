from langchain_core.messages import HumanMessage, AIMessage
import os
from dotenv import dotenv_values
import streamlit as st
from agent import agent
import warnings
from typing import List
import requests
import json

# Load environment variables
try:
    ENVs = dotenv_values(".env")  # for dev env
    API_KEY = ENVs["API_KEY"]
except:
    ENVs = st.secrets  # for streamlit deployment
    API_KEY = ENVs["API_KEY"]

os.environ["API_KEY"] = ENVs['API_KEY']

# Configure the Streamlit app
st.set_page_config(
    page_title="E-liss Trading Assistant",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("E-liss Trading AI 📈")
st.markdown(
    """
    #### Welcome to E-liss Trading AI, your crypto trading assistant!
    #### I can help analyze the market, suggest strategies, and answer your questions about Solana chain trading.
    > Note: This AI is currently tailored for trading on the Solana blockchain.
    """
)

# Initialize session state
if "store" not in st.session_state:
    st.session_state.store = []

store = st.session_state.store

# Display chat history
for message in store:
    if message.type == "ai":
        avatar = "📈"
    else:
        avatar = "💬"
    with st.chat_message(message.type, avatar=avatar):
        st.markdown(message.content)

# Function to fetch market data
def fetch_market_data(token: str):
    url = f"https://api.fakecryptoapi.com/market_data?token={token}&apikey={os.getenv('API_KEY')}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": "Failed to fetch market data."}

# Function to fetch trading advice
def fetch_trading_advice(token: str):
    url = f"https://api.fakecryptoapi.com/advice?token={token}&apikey={os.getenv('API_KEY')}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": "Failed to fetch trading advice."}

# Function to fetch historical data
def fetch_historical_data(token: str):
    url = f"https://api.fakecryptoapi.com/historical?token={token}&apikey={os.getenv('API_KEY')}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": "Failed to fetch historical data."}

# Function to fetch risk analysis
def fetch_risk_analysis(token: str):
    url = f"https://api.fakecryptoapi.com/risk?token={token}&apikey={os.getenv('API_KEY')}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": "Failed to fetch risk analysis."}

# Define commands
COMMANDS = {
    "help": "Displays the list of commands.",
    "market [token]": "Fetches market data for the specified token.",
    "advice [token]": "Provides trading advice for the specified token.",
    "history [token]": "Fetches historical data for the specified token.",
    "risk [token]": "Provides risk analysis for the specified token.",
    "greet": "Sends a friendly greeting.",
    "info": "Provides information about this assistant.",
}

def process_command(command: str) -> str:
    command = command.strip().lower()
    parts = command.split()
    if parts[0] == "help":
        return "Available commands:\n" + "\n".join([f"/{cmd}: {desc}" for cmd, desc in COMMANDS.items()])
    elif parts[0] == "greet":
        return "Hello! I'm here to assist you with your trading needs. 😊"
    elif parts[0] == "info":
        return "I'm E-liss AI, a trading assistant specializing in Solana blockchain trading."
    elif parts[0] == "market" and len(parts) > 1:
        token = parts[1]
        data = fetch_market_data(token)
        if "error" in data:
            return data["error"]
        return f"Market Data for {token}:\nPrice: {data['price']}\nVolume: {data['volume']}\nMarket Cap: {data['market_cap']}"
    elif parts[0] == "advice" and len(parts) > 1:
        token = parts[1]
        advice = fetch_trading_advice(token)
        if "error" in advice:
            return advice["error"]
        return f"Trading Advice for {token}:\n{advice['message']}"
    elif parts[0] == "history" and len(parts) > 1:
        token = parts[1]
        history = fetch_historical_data(token)
        if "error" in history:
            return history["error"]
        return f"Historical Data for {token}:\n{json.dumps(history, indent=2)}"
    elif parts[0] == "risk" and len(parts) > 1:
        token = parts[1]
        risk = fetch_risk_analysis(token)
        if "error" in risk:
            return risk["error"]
        return f"Risk Analysis for {token}:\n{risk['details']}"
    else:
        return f"Unknown command: {command}. Type '/help' for a list of commands."

# Handle user input
if prompt := st.chat_input("What would you like to ask?"):
    # Display user message in chat container
    st.chat_message("user", avatar="💬").markdown(prompt)

    if prompt.startswith("/"):
        response_content = process_command(prompt[1:])
    else:
        st.chat_message("📈").markdown("Analyzing...")
        store.append(HumanMessage(content=prompt))
        try:
            response_content = agent(prompt)
        except Exception as e:
            response_content = f"Sorry, I couldn't process your query due to an error: {str(e)}"

    response = AIMessage(content=response_content)
    store.append(response)

    # Display assistant response in chat container
    st.chat_message("assistant", avatar="📈").markdown(response.content)

# Agent definition
from langchain_groq import ChatGroq
from langchain.agents import create_react_agent, AgentExecutor
from tools.react_prompt_template import get_prompt_template
from tools.pdf_query_tools import pdf_query_tool

# Define the agent
warnings.filterwarnings("ignore", category=FutureWarning)

LLM = ChatGroq(model="llama3-8b-8192")

def agent(query: str):
    tools = [pdf_query_tool]
    prompt_template = get_prompt_template()
    trading_agent = create_react_agent(
        LLM,
        tools,
        prompt_template
    )

    agent_executor = AgentExecutor(
        agent=trading_agent, tools=tools, verbose=False, handle_parsing_errors=True
    )

    result = agent_executor.invoke({"input": query})
    return result["output"]

# Add a footer
st.markdown(
    """
    ---
    Powered by E-liss AI | Built with Streamlit and LangChain.
    """
)
