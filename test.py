import wandb

import random

import wandb

# Start a new wandb run to track this script.
run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    project="thesis-fm-latent",
    id="j6wn6nwn", 
    resume="must"
)


run.log({"test_n/iou": 0.81145}, step=236)
run.log({"test_n/dice": 0.89591}, step=236)
run.log({"test/dice": 0.87634}, step=236)

run.finish() 