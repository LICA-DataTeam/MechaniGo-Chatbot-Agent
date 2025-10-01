from mechanigo_agent import MechaniGoAgent
import asyncio

async def main():
    mgo = MechaniGoAgent(
        api_key=None,
        name="Rock Music Helper",
        handoff_description="A specialized agent knowledgeable in rock music history",
        instructions="You provide assistance with information and fun facts about the history of rock music."
    )

    res = await mgo.inquire("When was the Beatles formed?")
    print(res)

if __name__ == "__main__":
    asyncio.run(main())