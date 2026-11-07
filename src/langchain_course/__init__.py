
from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch


import os


@tool
def search(query: str) -> str:
    """New on Stocks"""
    print(f"Searching for {query}")
    return str(tavily.invoke({"query":query}))
    


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

tools = [TavilySearch()]

agent = create_agent(model=llm, tools=tools)


def main():

    result = agent.invoke({
        "messages": [
            {
            "role":"user",
            "content" : "Latest news on PC Jewellery"
            }
        ]
    })

    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
