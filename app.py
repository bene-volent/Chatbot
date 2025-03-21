# from fastapi import FastAPI, Request, Form, HTTPException, WebSocket, WebSocketDisconnect, Body
# from fastapi.templating import Jinja2Templates
# from fastapi.staticfiles import StaticFiles
# from fastapi.responses import HTMLResponse, JSONResponse
# from pydantic import BaseModel, Field
# from typing import Optional, Dict, List, Any, Union
# import torch
# import asyncio
# import threading
# import re
# import json
# import time
# import uvicorn
# from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList

# # Initialize FastAPI
# app = FastAPI()

# # Mount static files directory for CSS and JS
# app.mount("/static", StaticFiles(directory="static"), name="static")

# # Setup templates
# templates = Jinja2Templates(directory="templates")

# # Set device to GPU if available
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using device: {device}")

# # Load DeepSeek model
# MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME, 
#     torch_dtype=torch.float16 if device == "cuda" else torch.float32
# ).to(device)

# # Warm-up pass to load model into GPU memory
# print("Performing warm-up pass...")
# dummy_input = tokenizer("Warm-up pass", return_tensors="pt").to(device)
# with torch.no_grad():
#     model.generate(**dummy_input, max_length=10)
# print("Model ready!")

# # Custom stopping criteria to prevent model from continuing as user
# class UserMessageStoppingCriteria(StoppingCriteria):
#     def __init__(self, tokenizer, stop_strings=["User:", "\nUser", "System:", "\nSystem", "<STOP>", 
#                                               "[END]"]):
#         self.tokenizer = tokenizer
#         self.stop_string_ids = [tokenizer.encode(s, add_special_tokens=False) for s in stop_strings]
#         # Process specific end markers separately for efficiency
#         self.end_marker_ids = [
#             tokenizer.encode("[END]", add_special_tokens=False),
#             tokenizer.encode("<STOP>", add_special_tokens=False)
#         ]
#         self.max_length = 1000  # Maximum reasonable response length 
#         self.token_count = 0
        
#     def __call__(self, input_ids, scores, **kwargs):
#         self.token_count += 1
        
#         # Check if the response has become too long
#         if self.token_count > self.max_length:
#             print("Response too long, stopping.")
#             return True
        
#         # Check for end markers by looking for their token IDs in the recent part of the sequence
#         # This is more efficient than decoding the full text
#         recent_tokens = input_ids[0, -20:].tolist()  # Only look at recent tokens
        
#         # Check for end markers in a sliding window manner
#         for marker_ids in self.end_marker_ids:
#             for i in range(len(recent_tokens) - len(marker_ids) + 1):
#                 if recent_tokens[i:i+len(marker_ids)] == marker_ids:
#                     print(f"Found end marker: {self.tokenizer.decode(marker_ids)}")
#                     return True
        
#         # Check for other stop strings
#         for stop_ids in self.stop_string_ids:
#             if len(stop_ids) <= input_ids.shape[1]:
#                 if input_ids[0, -len(stop_ids):].tolist() == stop_ids:
#                     print(f"Found stop sequence: {self.tokenizer.decode(stop_ids)}")
#                     return True
                    
#         return False

# # Request schemas
# class ChatMessage(BaseModel):
#     message: str
#     conversation_history: Optional[str] = None
#     continue_last: Optional[bool] = False
#     session_id: Optional[str] = "default"

# class StructuredOutputRequest(BaseModel):
#     prompt: str
#     output_schema: Dict[str, Any]
#     schema_description: Optional[str] = None
#     max_tokens: Optional[int] = 1024
#     temperature: Optional[float] = 0.7

# # Store conversation history
# conversation_histories = {}
# # Store both complete responses and partial states
# conversation_states = {}
# # Store active WebSocket connections
# active_connections: Dict[str, WebSocket] = {}

# # Helper function to create prompts
# def format_conversation(history, new_message):
#     system_message = ("You are a helpful, concise assistant. Provide clear and accurate answers. "
#                      "Don't include any internal thoughts or repeat the user's message. "
#                      "Always complete your sentences and thoughts fully. "
#                      "After you've completed your full response, add '[END]' on a new line.")
    
#     if not history:
#         # formatted = f"System: {system_message}\nUser: {new_message}\nAssistant:"
#         formatted = f"System: {system_message}\nUser: {new_message}\nAssistant:"
#     else:
#         formatted = f"{history}\nUser: {new_message}\nAssistant:"
#     return formatted

# # Helper function to clean responses
# def clean_response(text):
#     """Clean up response text from unwanted artifacts"""
#     # Remove any <think> or </think> tags
#     text = re.sub(r'</?think>', '', text)
    
#     # Remove any System: prefixes that got generated
#     text = re.sub(r'^System:', '', text)
    
#     # Remove any lines that start with System:, User: or Assistant: that aren't part of the conversation
#     text = re.sub(r'\n(System|User|Assistant):[^\n]*', '', text)
    
#     # Fix missing spaces between words (common with some models)
#     # text = re.sub(r'([a-zA-Z])([A-Z])', r'\1 \2', text)
#     last_sentence = text.split('.')[-1]
#     if len(last_sentence) > 30 and not any(text.endswith(c) for c in ['.', '!', '?', ':', ';']):
#         # Likely an incomplete sentence - add ellipsis
#         text += "..."
#     # Remove end markers
#     text = re.sub(r'\[END\]', '', text)
#     text = re.sub(r'<STOP>', '', text)
    
#     # Clean up extra whitespace
#     text = re.sub(r'\s+', ' ', text).strip()
    
#     return text

# def truncate_at_sentence_end(text, max_chars=500):
#     """Truncate text at a sentence boundary if it's too long"""
#     if len(text) <= max_chars:
#         return text
        
#     # Try to find a sentence end within the last portion of the text
#     truncated = text[:max_chars]
#     sentence_ends = [pos for pos, char in enumerate(truncated[-100:]) if char in ['.', '!', '?']]
    
#     if sentence_ends:
#         # Find the last complete sentence
#         return truncated[:max_chars-100+sentence_ends[-1]+1]
#     else:
#         # If no sentence end found, just truncate and add ellipsis
#         return truncated + "..."

# @app.get("/", response_class=HTMLResponse)
# async def home(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})

# @app.websocket("/ws/{client_id}")
# async def websocket_endpoint(websocket: WebSocket, client_id: str):
#     await websocket.accept()
#     active_connections[client_id] = websocket
    
#     try:
#         while True:
#             data = await websocket.receive_json()
#             message = data.get("message", "")
#             session_id = data.get("session_id", "default")
#             continue_last = data.get("continue_last", False)
            
#             # Process in a separate task to not block the WebSocket
#             asyncio.create_task(process_message(client_id, session_id, message, continue_last))
            
#     except WebSocketDisconnect:
#         if client_id in active_connections:
#             del active_connections[client_id]

# async def process_message(client_id: str, session_id: str, message: str, continue_last: bool):
#     if client_id not in active_connections:
#         return
        
#     websocket = active_connections[client_id]
    
#     try:
#         # Get or initialize conversation history and state
#         history = conversation_histories.get(session_id, "")
#         state = conversation_states.get(session_id, {})
#         last_response = state.get("last_response", "")
        
#         # Handle continuation requests
#         if continue_last and ("complete it" in message.lower() or "continue" in message.lower()):
#             # Use the last response as context
#             prompt = format_conversation(history, "Please continue your previous response.")
#             starting_point = last_response
#         else:
#             # Normal new message
#             prompt = format_conversation(history, message)
#             starting_point = ""
        
#         # Tokenize input
#         inputs = tokenizer(prompt, return_tensors="pt").to(device)
#         input_length = len(inputs["input_ids"][0])
        
#         # Dynamic max length
#         dynamic_max_length = min(input_length + 150, input_length * 2, 512)
        
#         # Set up the text streamer
#         streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
#         # Send initial message to indicate processing started
#         await websocket.send_json({
#             "type": "start",
#             "is_continuation": continue_last
#         })
        
#         # Configure stopping criteria
#         stopping_criteria = StoppingCriteriaList([
#             UserMessageStoppingCriteria(tokenizer)
#         ])
        
#         # Start generation in a separate thread
#         generation_kwargs = {
#             "input_ids": inputs["input_ids"],
#             "max_length": input_length + dynamic_max_length,
#             "do_sample": True,
#             "temperature": 0.7,
#             "top_k": 50,
#             "top_p": 0.95,
#             "repetition_penalty": 1.2,  # Increased from 1.1
#             "no_repeat_ngram_size": 3,  # Add this parameter to prevent repeating phrases
#             "stopping_criteria": stopping_criteria,
#             "streamer": streamer,
#             "attention_mask": inputs.get("attention_mask", None)
#         }
        
#         thread = threading.Thread(
#             target=lambda: model.generate(**generation_kwargs)
#         )
#         thread.start()
        
#         # Stream the response
#         # Stream the response
#         collected_response = ""
#         stream_start_time = time.time()
#         max_stream_time = 60  # Maximum 30 seconds for streaming
#         last_token_time = time.time()
#         token_timeout = 3.0  # If no tokens for 3 seconds, consider generation complete

#         for new_text in streamer:
#             # Reset the token timeout
#             last_token_time = time.time()
#             # Check for timeout
#             if time.time() - stream_start_time > max_stream_time:
#                 print("Stream timeout reached")
#                 break
                
#             # Check for token delay timeout
#             if time.time() - last_token_time > token_timeout:
#                 print("Token delay timeout reached")
#                 break
                
#             # Clean each token before sending
#             new_text = clean_response(new_text)
            
#             # Check for explicit end markers
#             # if '[END]' in new_text or '<STOP>' in new_text:
#             #     # Remove the markers and break
#             #     new_text = new_text.replace('[END]', '').replace('<STOP>', '').strip()
#             #     if new_text:  # Only add if there's something left
#             #         collected_response += new_text
#             #         await websocket.send_json({
#             #             "type": "token", 
#             #             "token": new_text
#             #         })
#             #     break

#             if new_text:  # Only send non-empty tokens
#                 # Ensure proper spacing when adding to collected response
#                 if collected_response and not collected_response.endswith(' ') and not new_text.startswith(' ') and not new_text in [',', '.', '!', '?', ':', ';', ')', ']', '}']:
#                     collected_response += ' ' + new_text
#                 else:
#                     collected_response += new_text
                    
#                 # Send the properly formatted token
#                 await websocket.send_json({
#                     "type": "token", 
#                     "token": new_text
#                 })
#                 # Small delay to not overwhelm the client
#                 await asyncio.sleep(0.01)

#         # Check if the response was cut off mid-sentence
#         if not any(collected_response.rstrip().endswith(c) for c in ['.', '!', '?']):
#             # Add ellipsis to indicate truncation
#             ellipsis_token = "..."
#             collected_response += ellipsis_token
#             await websocket.send_json({
#                 "type": "token",
#                 "token": ellipsis_token
#             })
        
#         # Final cleaning of the complete response
#         collected_response = clean_response(collected_response)
#         collected_response = truncate_at_sentence_end(collected_response)

#         # Store this response for potential future continuations
#         state["last_response"] = collected_response
#         conversation_states[session_id] = state
        
#         # Update conversation history
#         if continue_last and (("complete it" in message.lower()) or ("continue" in message.lower())):
#             # For continuations, replace the last assistant message
#             history_parts = history.rsplit("Assistant:", 1)
#             if len(history_parts) > 1:
#                 updated_history = f"{history_parts[0]}Assistant: {collected_response}"
#             else:
#                 updated_history = f"{history}\nAssistant: {collected_response}"
#         else:
#             # For new messages, append normally
#             updated_history = f"{prompt} {collected_response}"
        
#         # Limit history size
#         words = updated_history.split()
#         if len(words) > 1024:
#             updated_history = " ".join(words[-1024:])
        
#         # Store updated history
#         conversation_histories[session_id] = updated_history
        
#         # Send completion message
#         await websocket.send_json({
#             "type": "end",
#             "is_continuation": continue_last
#         })
    
#     except Exception as e:
#         print(f"Error: {str(e)}")
#         await websocket.send_json({
#             "type": "error",
#             "error": str(e)
#         })

# # JSON API endpoint for structured output
# @app.post("/api/structured-output")
# async def structured_output(request: StructuredOutputRequest):
#     try:
#         # Create a prompt that instructs the model to generate structured output
#         schema_json = json.dumps(request.output_schema, indent=2)
#         schema_description = request.schema_description or "Follow this JSON schema exactly."
        
#         prompt = f"""Generate a JSON object that follows this exact schema:
# ```json
# {schema_json}
# ```
# {schema_description}

# Based on this request: {request.prompt}

# Return ONLY valid JSON that matches the schema exactly, with no additional text, explanations, or formatting."""        # Tokenize input
#         inputs = tokenizer(prompt, return_tensors="pt").to(device)
#         input_length = len(inputs["input_ids"][0])
        
#         # Set up generation parameters
#         max_new_tokens = min(request.max_tokens, 4096 - input_length)  # Ensure we don't exceed model context
        
#         start_time = time.time()
#         max_generation_time = 60  # 60 seconds max

#         # Create a function to generate with timeout
#         def generate_with_timeout():
#             try:
#                 return model.generate(
#                     inputs["input_ids"],
#                     max_new_tokens=max_new_tokens,
#                     temperature=request.temperature,
#                     top_p=0.95,
#                     do_sample=(request.temperature > 0),
#                     repetition_penalty=1.1,
#                     no_repeat_ngram_size=3,
#                     attention_mask=inputs.get("attention_mask", None)
#                 )
#             except Exception as e:
#                 print(f"Generation error: {str(e)}")
#                 return None

#         # Use a thread with timeout
#         generation_thread = threading.Thread(target=generate_with_timeout)
#         generation_thread.start()
#         generation_thread.join(timeout=max_generation_time)

#         # Check if generation completed
#         if generation_thread.is_alive():
#             return JSONResponse(
#                 status_code=408,  # Request Timeout
#                 content={"error": f"Generation timed out after {max_generation_time} seconds"}
#             )
        
#         # Get the output from the thread
#         output = generation_thread._target()
#         if output is None:
#             return JSONResponse(
#                 status_code=500,
#                 content={"error": "Generation failed"}
#             )
            
#         # Decode the generated output, skipping the input prompt
#         generated_text = tokenizer.decode(output[0][input_length:], skip_special_tokens=True)
        
#         # Extract JSON from the generated text
#         json_match = re.search(r'```json\n(.*?)\n```', generated_text, re.DOTALL)
#         if json_match:
#             json_str = json_match.group(1)
#         else:
#             # If no code block, try to find JSON directly
#             json_str = generated_text.strip()
        
#         # Parse the JSON to validate it
#         try:
#             json_obj = json.loads(json_str)
#             generation_time = time.time() - start_time
            
#             # Return the parsed JSON and metadata
#             return {
#                 "data": json_obj,
#                 "metadata": {
#                     "generation_time_seconds": generation_time,
#                     "model": MODEL_NAME
#                 }
#             }
#         except json.JSONDecodeError as e:
#             # If JSON parsing fails, return error with the generated text for debugging
#             return JSONResponse(
#                 status_code=422,
#                 content={
#                     "error": "Failed to parse generated JSON",
#                     "error_details": str(e),
#                     "generated_text": generated_text
#                 }
#             )

#     except Exception as e:
#         return JSONResponse(
#             status_code=500,
#             content={"error": f"Generation failed: {str(e)}"}
#         )

# # Reset conversation endpoint
# @app.post("/reset")
# async def reset_conversation(session_id: str = Form("default")):
#     # Remove conversation history and state for this session
#     if session_id in conversation_histories:
#         del conversation_histories[session_id]
#     if session_id in conversation_states:
#         del conversation_states[session_id]
#     return {"status": "conversation reset"}

# if __name__ == "__main__":
#     uvicorn.run("app:app", host="0.0.0.0", port=8000, timeout_keep_alive=120, reload=True)


from fastapi import FastAPI, Request, Form, HTTPException, WebSocket, WebSocketDisconnect, Body
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any, Union
import torch
import asyncio
import threading
import re
import json
import time
import uvicorn
import os
from transformers import TextIteratorStreamer

# Import our model handlers
from models import ModelHandlerFactory

# Initialize FastAPI
app = FastAPI()

# Mount static files directory for CSS and JS
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Choose model based on environment variable or default to Phi-2
MODEL_NAME = os.environ.get("LLM_MODEL", "microsoft/phi-2")

# Initialize the model handler using the factory
model_handler = ModelHandlerFactory.get_handler(MODEL_NAME)
print(f"Using model: {model_handler.model_name}")

# Request schemas
class ChatMessage(BaseModel):
    message: str
    conversation_history: Optional[str] = None
    continue_last: Optional[bool] = False
    session_id: Optional[str] = "default"

class StructuredOutputRequest(BaseModel):
    prompt: str
    output_schema: Dict[str, Any]
    schema_description: Optional[str] = None
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.7

# Store conversation history
conversation_histories = {}
# Store both complete responses and partial states
conversation_states = {}
# Store active WebSocket connections
active_connections: Dict[str, WebSocket] = {}

def truncate_at_sentence_end(text, max_chars=500):
    """Truncate text at a sentence boundary if it's too long"""
    if len(text) <= max_chars:
        return text
        
    # Try to find a sentence end within the last portion of the text
    truncated = text[:max_chars]
    sentence_ends = [pos for pos, char in enumerate(truncated[-100:]) if char in ['.', '!', '?']]
    
    if sentence_ends:
        # Find the last complete sentence
        return truncated[:max_chars-100+sentence_ends[-1]+1]
    else:
        # If no sentence end found, just truncate and add ellipsis
        return truncated + "..."

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    active_connections[client_id] = websocket
    
    try:
        while True:
            data = await websocket.receive_json()
            message = data.get("message", "")
            session_id = data.get("session_id", "default")
            continue_last = data.get("continue_last", False)
            
            # Process in a separate task to not block the WebSocket
            asyncio.create_task(process_message(client_id, session_id, message, continue_last))
            
    except WebSocketDisconnect:
        if client_id in active_connections:
            del active_connections[client_id]

async def process_message(client_id: str, session_id: str, message: str, continue_last: bool):
    if client_id not in active_connections:
        return
        
    websocket = active_connections[client_id]
    
    try:
        # Get or initialize conversation history and state
        history = conversation_histories.get(session_id, "")
        state = conversation_states.get(session_id, {})
        last_response = state.get("last_response", "")
        
        # Handle continuation requests
        if continue_last and ("complete it" in message.lower() or "continue" in message.lower()):
            # Use the last response as context
            prompt = model_handler.format_conversation(history, "Please continue your previous response.")
            starting_point = last_response
        else:
            # Normal new message
            prompt = model_handler.format_conversation(history, message)
            starting_point = ""
        
        # Tokenize input
        inputs = model_handler.tokenizer(prompt, return_tensors="pt").to(model_handler.device)
        
        # Set up the text streamer
        streamer = TextIteratorStreamer(model_handler.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        # Send initial message to indicate processing started
        await websocket.send_json({
            "type": "start",
            "is_continuation": continue_last
        })
        
        # Get generation kwargs from the model handler
        generation_kwargs = model_handler.get_generation_kwargs(inputs, streamer)
        
        # Start generation in a separate thread
        thread = threading.Thread(
            target=lambda: model_handler.model.generate(**generation_kwargs)
        )
        thread.start()
        
        # Stream the response
        collected_response = ""
        stream_start_time = time.time()
        max_stream_time = 60  # Maximum 60 seconds for streaming
        last_token_time = time.time()
        token_timeout = 3.0  # If no tokens for 3 seconds, consider generation complete

        for new_text in streamer:
            # Reset the token timeout
            last_token_time = time.time()
            # Check for timeout
            if time.time() - stream_start_time > max_stream_time:
                print("Stream timeout reached")
                break
                
            # Check for token delay timeout
            if time.time() - last_token_time > token_timeout:
                print("Token delay timeout reached")
                break

            # Check for explicit end markers
            if '[END]' in new_text or '<STOP>' in new_text:
                # Remove the markers and break
                new_text = new_text.replace('[END]', '').replace('<STOP>', '').strip()
                if new_text:  # Only add if there's something left
                    collected_response += new_text
                    await websocket.send_json({
                        "type": "token", 
                        "token": new_text
                    })
                break
                
            # Clean each token before sending
            new_text = model_handler.clean_response(new_text)
            
            if new_text:  # Only send non-empty tokens
                # Ensure proper spacing when adding to collected response
                if collected_response and not collected_response.endswith(' ') and not new_text.startswith(' ') and not new_text in [',', '.', '!', '?', ':', ';', ')', ']', '}']:
                    collected_response += ' ' + new_text
                else:
                    collected_response += new_text
                    
                # Send the properly formatted token
                await websocket.send_json({
                    "type": "token", 
                    "token": new_text
                })
                # Small delay to not overwhelm the client
                await asyncio.sleep(0.01)

        # Check if the response was cut off mid-sentence
        if not any(collected_response.rstrip().endswith(c) for c in ['.', '!', '?']):
            # Add ellipsis to indicate truncation
            ellipsis_token = "..."
            collected_response += ellipsis_token
            await websocket.send_json({
                "type": "token",
                "token": ellipsis_token
            })
        
        # Final cleaning of the complete response
        collected_response = model_handler.clean_response(collected_response)
        collected_response = truncate_at_sentence_end(collected_response)

        # Store this response for potential future continuations
        state["last_response"] = collected_response
        conversation_states[session_id] = state
        
        # Update conversation history
        if continue_last and (("complete it" in message.lower()) or ("continue" in message.lower())):
            # For continuations, replace the last assistant message
            history_parts = history.rsplit("Assistant:", 1)
            if len(history_parts) > 1:
                updated_history = f"{history_parts[0]}Assistant: {collected_response}"
            else:
                updated_history = f"{history}\nAssistant: {collected_response}"
        else:
            # For new messages, append normally
            updated_history = f"{prompt} {collected_response}"
        
        # Limit history size
        words = updated_history.split()
        if len(words) > 1024:
            updated_history = " ".join(words[-1024:])
        
        # Store updated history
        conversation_histories[session_id] = updated_history
        
        # Send completion message
        await websocket.send_json({
            "type": "end",
            "is_continuation": continue_last
        })
    
    except Exception as e:
        print(f"Error: {str(e)}")
        await websocket.send_json({
            "type": "error",
            "error": str(e)
        })

# JSON API endpoint for structured output
@app.post("/api/structured-output")
async def structured_output(request: StructuredOutputRequest):
    try:
        # Create a prompt that instructs the model to generate structured output
        schema_json = json.dumps(request.output_schema, indent=2)
        schema_description = request.schema_description or "Follow this JSON schema exactly."
        
        prompt = f"""Generate a JSON object that follows this exact schema:
```json
{schema_json}
```
{schema_description}

Based on this request: {request.prompt}

Return ONLY valid JSON that matches the schema exactly, with no additional text, explanations, or formatting."""

        # Tokenize input
        inputs = model_handler.tokenizer(prompt, return_tensors="pt").to(model_handler.device)
        input_length = len(inputs["input_ids"][0])
        
        # Set up generation parameters
        max_new_tokens = min(request.max_tokens, 4096 - input_length)  # Ensure we don't exceed model context
        
        start_time = time.time()
        max_generation_time = 60  # 60 seconds max

        # Create a function to generate with timeout
        def generate_with_timeout():
            try:
                return model_handler.model.generate(
                    inputs["input_ids"],
                    max_new_tokens=max_new_tokens,
                    temperature=request.temperature,
                    top_p=0.95,
                    do_sample=(request.temperature > 0),
                    repetition_penalty=1.1,
                    no_repeat_ngram_size=3,
                    attention_mask=inputs.get("attention_mask", None)
                )
            except Exception as e:
                print(f"Generation error: {str(e)}")
                return None

        # Use a thread with shared result variable
        result = {"output": None}
        
        def generate_with_timeout_wrapper():
            result["output"] = generate_with_timeout()
            
        generation_thread = threading.Thread(target=generate_with_timeout_wrapper)
        generation_thread.start()
        generation_thread.join(timeout=max_generation_time)

        # Check if generation completed
        if generation_thread.is_alive():
            return JSONResponse(
                status_code=408,  # Request Timeout
                content={"error": f"Generation timed out after {max_generation_time} seconds"}
            )
        
        # Get the output from the shared result
        output = result["output"]
        if output is None:
            return JSONResponse(
                status_code=500,
                content={"error": "Generation failed"}
            )
            
        # Decode the generated output, skipping the input prompt
        generated_text = model_handler.tokenizer.decode(output[0][input_length:], skip_special_tokens=True)
        
        # Extract JSON from the generated text
        json_match = re.search(r'```json\n(.*?)\n```', generated_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # If no code block, try to find JSON directly
            json_str = generated_text.strip()
        
        # Parse the JSON to validate it
        try:
            json_obj = json.loads(json_str)
            generation_time = time.time() - start_time
            
            # Return the parsed JSON and metadata
            return {
                "data": json_obj,
                "metadata": {
                    "generation_time_seconds": generation_time,
                    "model": model_handler.model_name
                }
            }
        except json.JSONDecodeError as e:
            # If JSON parsing fails, return error with the generated text for debugging
            return JSONResponse(
                status_code=422,
                content={
                    "error": "Failed to parse generated JSON",
                    "error_details": str(e),
                    "generated_text": generated_text
                }
            )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Generation failed: {str(e)}"}
        )

# Endpoint to get available models
@app.get("/api/models")
async def get_models():
    """Return a list of available models that can be used"""
    return {
        "current_model": model_handler.model_name,
        "available_models": [
            {"name": "microsoft/phi-2", "description": "Microsoft Phi-2 (2.7B) - Great for educational content"},
            {"name": "google/gemma-2b", "description": "Google Gemma 2B - Good general purpose model"},
            {"name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "description": "DeepSeek R1 Distill (1.5B) - Current model"},
            {"name": "TheBloke/Mistral-7B-Instruct-v0.1-GPTQ", "description": "Mistral 7B (Quantized) - Requires 'optimum' package"}
        ]
    }

# Endpoint to change models
@app.post("/api/change-model")
async def change_model(model_name: str = Form(...)):
    """Change the active model"""
    global model_handler
    
    try:
        # Unload the current model to free memory
        del model_handler.model
        del model_handler.tokenizer
        torch.cuda.empty_cache()
        
        # Load the new model
        model_handler = ModelHandlerFactory.get_handler(model_name)
        
        return {"status": "success", "message": f"Model changed to {model_handler.model_name}"}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Failed to change model: {str(e)}"}
        )

# Reset conversation endpoint
@app.post("/reset")
async def reset_conversation(session_id: str = Form("default")):
    # Remove conversation history and state for this session
    if session_id in conversation_histories:
        del conversation_histories[session_id]
    if session_id in conversation_states:
        del conversation_states[session_id]
    return {"status": "conversation reset"}

# Add this route for checking system status
@app.get("/api/system-status")
async def system_status():
    """Get system status including VRAM usage"""
    vram_info = {}
    
    if torch.cuda.is_available():
        # Get total VRAM
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
        
        # Get allocated VRAM
        allocated_vram = torch.cuda.memory_allocated(0) / (1024**3)  # Convert to GB
        
        # Get cached VRAM
        reserved_vram = torch.cuda.memory_reserved(0) / (1024**3)  # Convert to GB
        
        vram_info = {
            "total_vram_gb": round(total_vram, 2),
            "allocated_vram_gb": round(allocated_vram, 2),
            "reserved_vram_gb": round(reserved_vram, 2),
            "free_vram_gb": round(total_vram - reserved_vram, 2)
        }
    
    return {
        "status": "running",
        "model": model_handler.model_name,
        "device": model_handler.device,
        "vram_info": vram_info,
        "active_connections": len(active_connections)
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, timeout_keep_alive=120, reload=True)