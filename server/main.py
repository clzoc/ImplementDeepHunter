import asyncio
from typing import Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

def calculate():
    item = np.random.randint(25, 50, (25, 2, 7))
    return item

data = calculate()

app = FastAPI()

html = """
<!DOCTYPE html>
<html>
    <head>
        <title>Chat</title>
    </head>
    <body>
        <h1>WebSocket Chat</h1>
        <form action="" onsubmit="sendMessage(event)">
            <input type="text" id="messageText" autocomplete="off"/>
            <button>Send</button>
        </form>
        <ul id='messages'>
        </ul>
        <script>
            var ws = new WebSocket("ws://localhost:8000/ws");
            ws.onmessage = function(event) {
                var messages = document.getElementById('messages')
                var message = document.createElement('li')
                var content = document.createTextNode(event.data)
                message.appendChild(content)
                messages.appendChild(message)
            };
            function sendMessage(event) {
                var input = document.getElementById("messageText")
                ws.send(input.value)
                input.value = ''
                event.preventDefault()
            }
        </script>
    </body>
</html>
"""

class Item(BaseModel):
    name: str
    price: float
    is_offer: Optional[bool] = None

@app.get("/")
async def get():
    return HTMLResponse(html)

@app.get("/favicon.ico")
async def get_icon():
    return {200}

@app.websocket("/try")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        # data = await websocket.receive_text()
        # await websocket.send_text(f"Message text was: {data}")
        index = await websocket.receive_text()
        index = int(index)
        
        for iter in range(5):
            for_send = np.ndarray.tolist(data[index*5 + iter])
            await websocket.send_json(for_send)
            await asyncio.sleep(1)
        
        

@app.get("/items/{item_id}")
async def read_item(item_id: int, q: Optional[str]=None):
    return {"item_id": item_id, "q": q}

@app.put("/items/{item_id}")
async def update_item(item_id: int, item: Item):
    return {"item_name": item.name, "item_id": item_id}

if __name__ == "__main__":
    uvicorn.run(app="main:app", host="59.78.194.133", port=8000, reload=True)
