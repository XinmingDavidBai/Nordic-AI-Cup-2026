import os
import pygame
import random

from src.core import SimulationCore
from src.utils.DTOs import ActionRequest
from src.utils.controllers.params import Params
from src.utils.controllers.hive_policy import HiveMind

# Same resolution as agent_server.py, so what you watch here is what the evaluation runs.
PARAMS_PATH = os.environ.get("HIVE_PARAMS", "checkpoints/train_paired/best.json")


def load_params():
    """Trained parameters if they exist, hand-set defaults otherwise (loudly)."""
    if os.path.exists(PARAMS_PATH):
        print(f"Policy parameters: {PARAMS_PATH}")
        return Params.load(PARAMS_PATH)
    print(f"WARNING: {PARAMS_PATH} not found - running hand-set defaults, "
          f"not trained parameters. Set HIVE_PARAMS to pick a checkpoint.")
    return Params()


def local_simulation(verbose=True, seed=None):
    if seed is None:  # If no seed is provided, generate a random one
        seed = random.randint(0, 2**32 - 1)

    sim = SimulationCore(seed=seed)
    # One controller for the whole species, stateful across ticks. Seeded from the
    # episode seed so a given (params, seed) pair replays identically.
    hive = HiveMind(load_params(), rng_seed=seed)

    pygame.init()
    screen, clock = None, None

    # Optional render
    if verbose:
        info = pygame.display.Info()
        env_ratio = sim.env_width / sim.env_height
        screen_height = int(info.current_h * 0.9) # 90% of screen height
        screen_width = int(screen_height * env_ratio) # Keep aspect ratio
        screen = pygame.display.set_mode((screen_width, int(screen_height)), pygame.SCALED)
        clock = pygame.time.Clock()

    running = True
    actions = []

    while running:
        if verbose:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: # Check if user closes window
                    running = False

        state = sim.step(actions)

        # The policy decides for every agent at once - it needs the whole population to
        # size the hive against its target.
        decided = hive.decide(state["observations"], sim.env.time)
        actions = [(d["agent_id"], ActionRequest(**d)) for d in decided]

        if verbose:
            sim.env.draw(screen)
            font = pygame.font.SysFont(None, 24)
            img = font.render(f'Score: {state["score"]:.2f}', True, (255,255,255))
            screen.blit(img, (20, 20))
            pygame.display.flip()
            clock.tick(60) # Control max FPS

        print(f'Score: {state["score"]:.2f} | Agents alive: {state["num_agents"]:.0f} | Time: {sim.env.time:.2f}')

        if state["num_agents"] == 0 or sim.env.time > 3000:
            print(f"Game over! Final Score: {state['score']}")
            print(f"Seed: {seed}")
            running = False

    pygame.quit()

if __name__ == "__main__":
    local_simulation(verbose=True)
