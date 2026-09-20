import os

from fastapi import FastAPI, Body

from src.utils.DTOs import StepResponse
from src.utils.controllers.params import Params
from src.utils.controllers.hive_policy import HiveMind

HOST = "0.0.0.0"
PORT = 9052

# Trained parameters if they exist, hand-set defaults otherwise. A missing checkpoint
# used to fall back silently, so an evaluation could run hand-set defaults without anyone
# noticing - it says so now.
PARAMS_PATH = os.environ.get("HIVE_PARAMS", "checkpoints/train_paired/best.json")
if os.path.exists(PARAMS_PATH):
    print(f"Policy parameters: {PARAMS_PATH}")
    params = Params.load(PARAMS_PATH)
else:
    print(f"WARNING: {PARAMS_PATH} not found - serving hand-set defaults, not trained "
          f"parameters. Set HIVE_PARAMS to pick a checkpoint.")
    params = Params()

app = FastAPI(title="Survival Simulator Agent Endpoint")

# One controller for the whole species, stateful across ticks. The evaluation runs three
# simulations back to back against the same process, so HiveMind.decide() watches for
# sim_time going backwards and resets its memory on a new episode.
hive = HiveMind(params)


@app.post("/predict")
def predict(step: StepResponse = Body(...)):
    """
    Receives the current simulation state and returns actions for all agents.
    """
    statuses = [
        {
            "agent_id": a.agent_id,
            "observations": a.observations,
            "energy": a.energy,
            "biome": a.biome,
            "age": a.age,
            "speed": a.speed,
            "sprint_speed": a.sprint_speed,
            "hearing_radius": a.hearing_radius,
            "vision_angle": a.vision_angle,
            "vision_range": a.vision_range,
            "max_energy": a.max_energy,
        }
        for a in step.agent_status
    ]
    # Plain dicts, not ActionRequest models - the response has to average under ~20 ms
    # to stay inside the server's 600 s accumulated-wait budget over 30,000 ticks.
    return {"actions": hive.decide(statuses, step.sim_time)}


@app.get("/")
def index():
    return {"message": "Agent endpoint running!", "params": PARAMS_PATH
            if os.path.exists(PARAMS_PATH) else "defaults"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
