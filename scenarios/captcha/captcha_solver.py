import argparse
import uvicorn
import logging
from dotenv import load_dotenv

load_dotenv()

from google.adk.agents import Agent
from google.adk.a2a.utils.agent_to_a2a import to_a2a

from a2a.types import (
    AgentCapabilities,
    AgentCard,
)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("captcha_solver")

# Enable verbose logging for Google AI SDK to see requests/responses
logging.getLogger("google_genai").setLevel(logging.DEBUG)
logging.getLogger("google_adk").setLevel(logging.DEBUG)


SYSTEM_INSTRUCTION = """You are a CAPTCHA solver. Your task is to extract text from CAPTCHA images.

When given a CAPTCHA image:
1. Carefully analyze the image to identify all characters
2. Return ONLY the characters you see - typically 5 alphanumeric characters
3. Do not include any explanation, just the characters
4. Use lowercase letters
5. If you cannot read a character, make your best guess

Example response format: 2a3b4
"""


def main():
    parser = argparse.ArgumentParser(description="Run the CAPTCHA solver agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9019, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="External URL to provide in the agent card")
    args = parser.parse_args()

    # Create ADK agent with Gemini 2.0 Flash model
    root_agent = Agent(
        name="captcha_solver",
        model="gemini-2.0-flash",
        description="Solves CAPTCHA puzzles by extracting text from images.",
        instruction=SYSTEM_INSTRUCTION,
    )

    # Create agent card
    agent_card = AgentCard(
        name="CaptchaSolver",
        description='An agent that attempts to solve CAPTCHA puzzles by extracting text from images.',
        url=args.card_url or f'http://{args.host}:{args.port}/',
        version='1.0.0',
        default_input_modes=['text', 'image'],
        default_output_modes=['text'],
        capabilities=AgentCapabilities(streaming=True),
        skills=[],
    )

    logger.info(f"Starting CAPTCHA solver on {args.host}:{args.port}")
    a2a_app = to_a2a(root_agent, agent_card=agent_card)
    uvicorn.run(a2a_app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
