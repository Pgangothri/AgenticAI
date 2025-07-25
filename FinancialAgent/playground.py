import os
from dotenv import load_dotenv

import phi
import phi.api
from phi.agent import Agent
from phi.playground import Playground, serve_playground_app
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo

# ✅ Load environment variables
load_dotenv()

# ✅ Set your PHI API key correctly
phi.api.api_key = os.getenv("PHI_API_KEY")

# ✅ Web Search Agent
web_search_agent = Agent(
    name="Web Search Agent",
    role="Search the web for information",
    model=Groq(id="llama3-8b-8192"),  # ✅ Correct model ID format
    tools=[DuckDuckGo()],
    instructions=["Always include sources."],
    show_tool_calls=True,
    markdown=True,
)

finance_agent = Agent(
    name="Finance AI Agent",
    model=Groq(id="llama3-8b-8192"),
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            stock_fundamentals=True,
            company_news=True,
        )
    ],
    instructions=[
        "Always provide a valid stock symbol (like AAPL, TSLA, GOOG) when calling financial tools.",
        "If the user gives a company name (like Apple or Tesla), convert it to the correct stock symbol before calling the tool.",
        "If you cannot determine the symbol, ask the user to specify it.",
        "Use tables to display the data."
    ],
    show_tool_calls=True,
    markdown=True,
)


# ✅ Create Playground App
app = Playground(agents=[finance_agent, web_search_agent]).get_app()

if __name__ == "__main__":
    serve_playground_app("playground:app", reload=True)
