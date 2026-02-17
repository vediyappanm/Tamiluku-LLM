from strands import Agent, tool

@tool
def get_weather(location: str) -> str:
    """Get weather information for a location.
    
    Args:
        location: City name to get weather for
    """
    # In a real app, you'd call a weather API here
    return f"Weather in {location}: Sunny, 72°F with light breeze"

@tool
def convert_temperature(temp: float, from_unit: str, to_unit: str) -> str:
    """Convert temperature between Celsius and Fahrenheit.
    
    Args:
        temp: Temperature value to convert
        from_unit: Source unit ('C' or 'F')
        to_unit: Target unit ('C' or 'F')
    """
    if from_unit == 'F' and to_unit == 'C':
        result = (temp - 32) * 5/9
        return f"{temp}°F = {result:.1f}°C"
    elif from_unit == 'C' and to_unit == 'F':
        result = (temp * 9/5) + 32
        return f"{temp}°C = {result:.1f}°F"
    else:
        return f"Temperature is {temp}°{from_unit}"

# Create agent with custom tools
agent = Agent(
    tools=[get_weather, convert_temperature],
    system_prompt="You are a weather assistant that can check weather and convert temperatures."
)

# Test the agent
response = agent("What's the weather in San Francisco? Also convert 72°F to Celsius.")
print(response)
