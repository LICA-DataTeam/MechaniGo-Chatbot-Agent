from mechanigo_agent import MechaniGoAgent
import asyncio

from dotenv import load_dotenv
import os

load_dotenv()

async def main():
    mgo = MechaniGoAgent(api_key=os.getenv("OPENAI_API_KEY"))
    res = await mgo.inquire("My name is Walter Hartwell White, I live at 308 Negra Arroyo Lane, Albuquerque, New Mexico, 87104. I drive a Toyota Vios 2019 and I am in need of a PMS.")
    print(res)

if __name__ == "__main__":
    asyncio.run(main())