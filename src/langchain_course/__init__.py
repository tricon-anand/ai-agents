
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch


class Source(BaseModel):
    """Schema used for sources returned by the agent."""
    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema for the agent's answer and sources."""
    answer: str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(
        default_factory=list,
        description="Sources used to generate the answer"
    )


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

tools = [TavilySearch(max_results=5)]

agent = create_agent(
    model=llm,
    tools=tools
)

structured_llm = llm.with_structured_output(AgentResponse)

def main():

    result = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": "Search job for AI engineer in Bengaluru"
            }
        ]
    })

    # Step 2: Extract the agent's final answer
    agent_answer = result["messages"][-1].content

    # Step 3: Convert the answer into Pydantic structure
    structured_result = structured_llm.invoke(
        f"""
        Extract the answer and source URLs from the following
        research result.

        Return the answer and sources using the AgentResponse schema.

        Research result:
        {agent_answer}
        """ 
    )

     # Step 4: Display structured output
    print("\nAnswer:")
    print(structured_result.answer)

    print("\nSources:")
    for source in structured_result.sources:
        print(source.url)

if __name__ == "__main__":
    main()
