from fastapi import FastAPI, Response, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
from pathlib import Path
import uvicorn

app = FastAPI()

# Ensure static folder exists
static_folder = Path("static")
static_folder.mkdir(exist_ok=True)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

class AudioRequest(BaseModel):
    content: str

@app.post('/get-audio')
async def get_audio(request: AudioRequest):
    content = request.content
    
    print(f"Received content: {content}")
    
    # Get the appropriate audio file name or use a default
    audio_file = "three-random-tunes-girl-200030.mp3"
    
    file_path = os.path.join(static_folder, audio_file)
    
    # Check if the file exists
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Audio file {audio_file} not found")
    
    # Read the file as bytes
    with open(file_path, "rb") as file:
        audio_data = file.read()
    
    # Return the audio bytes directly
    return Response(content=audio_data, media_type="audio/mpeg")

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)