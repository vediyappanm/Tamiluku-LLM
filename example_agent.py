from strands import Agent
from strands_tools import calculator, python_repl, http_request

# Create an agent with community tools (uses Bedrock Claude 4 Sonnet by default)
agent = Agent(
    tools=[calculator, python_repl, http_request],
    system_prompt="You are a helpful assistant that can perform calculations, execute Python code, and make HTTP requests."
)

# Test the agent with a simple question
response = agent("What is 25 * 47 + 123?")
print("Response:", response)

# The agent maintains conversation context
agent("My favorite color is blue")
response2 = agent("What's my favorite color?")
print("\nContext test:", response2)
