import websockets
import asyncio
import sys

async def hello(str1, ID):
    uri = "ws://localhost:8000/generate"
    async with websockets.connect(uri) as websocket:
        await websocket.send(ID)
        
        name = str1
        await websocket.send(name)
        print(f"> {name}")

        while 1:
            greeting = await websocket.recv()
            #print(greeting,end='\n')
            sys.stdout.write(greeting)
            sys.stdout.flush()
            
            if(greeting=='[END OF GENERATION]'):
                break

user_id = input("\nID: ")

while 1:
    str1 = input("\n질문: ")
    asyncio.get_event_loop().run_until_complete(hello(str1, user_id))
