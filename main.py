from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from agent import run_evaluation

app = FastAPI(title="Intern Evaluation AI Agent Server")

class EvalRequest(BaseModel):
    week_number: int

@app.get("/")
def home():
    return {"status": "Agent server is running!"}

@app.post("/trigger-evaluation")
def trigger_eval(payload: EvalRequest, background_tasks: BackgroundTasks):
    """
    Triggers the AI evaluation agent asynchronously in the background so 
    the caller doesn't experience a timeout while the LLM works.
    """
    if payload.week_number < 1:
        raise HTTPException(status_code=400, detail="Invalid week number.")
    
    # Run the agent execution in a background thread
    background_tasks.add_task(run_evaluation, payload.week_number)
    
    return {
        "status": "Accepted",
        "message": f"AI Agent started evaluating Week {payload.week_number}. HR will be notified via Google Chat when done."
    }