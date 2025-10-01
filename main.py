from mechanigo_agent import MechaniGoAgent
import asyncio

from dotenv import load_dotenv
import os

load_dotenv()

async def main():
    mgo = MechaniGoAgent(api_key=os.getenv("OPENAI_API_KEY"))
    res = await mgo.inquire("How much is PMS for Toyota Vios?")
    print(res)

if __name__ == "__main__":
    asyncio.run(main())