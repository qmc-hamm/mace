import os
import click
import numpy as np

import mlflow
from mlflow.entities import Param, RunTag
from mlflow.tracking import MlflowClient

tracking_client = mlflow.tracking.MlflowClient()


def run_train(experiment_id, r_max, forces_weight, energy_weight, train_file, valid_file, backend_config="slurm_config.json", parent_run_id=None):
    p = mlflow.projects.run(
        uri=os.path.dirname(os.path.realpath(__file__)),
        entry_point="main",
        parameters={
            "r_max": float(r_max),
            "forces_weight": float(forces_weight),
            "energy_weight": float(energy_weight),
            "train_file": str(train_file),
            "valid_file": str(valid_file)
        },
        experiment_id=experiment_id,
        synchronous=False,
        backend="slurm",
        backend_config=backend_config
    )
    tracking_client.log_batch(run_id=p.run_id, metrics=[],
                             params=[Param("r_max", str(r_max)), Param("forces_weight", str(forces_weight)), Param("energy_weight", str(energy_weight))],
                             tags=[RunTag(mlflow.utils.mlflow_tags.MLFLOW_PARENT_RUN_ID, str(parent_run_id))])

    return p


@click.command(help="Perform grid search over train (main entry point).")
@click.option("--num_runs", type=click.INT, default=2, help="Maximum number of runs to evaluate.")
@click.option("--train_backend_config", type=click.STRING, default="slurm_config.json", help="Json file for training jobs")
@click.option("--train_file", type=click.STRING, default="qmc/training.xyz", help="Training File Path")
@click.option("--valid_file", type=click.STRING, default="qmc/testing.xyz", help="Testing File Path")
def run(num_runs, train_backend_config, train_file, valid_file):
    provided_run_id = os.environ.get("MLFLOW_RUN_ID", None)
    with mlflow.start_run(run_id=provided_run_id) as run:
        if not provided_run_id:
            provided_run_id = run.info.run_id
        print("Search is run_id ", run.info.run_id)
        experiment_id = run.info.experiment_id
        runs = [(round(np.random.uniform(2.5, 3.0),2), round(np.random.uniform(10, 100),2)) for _ in range(num_runs)]
        jobs = []
        for r_max, forces_weight in runs:
            jobs.append(run_train(
                experiment_id,
                r_max=r_max, forces_weight=forces_weight, energy_weight=1.0 ,train_file=train_file,
                valid_file=valid_file, backend_config=train_backend_config,
                parent_run_id=provided_run_id)
            )
        
        print("Launched Jobs: ", jobs)
        print("Started %d slurm jobs" % len(jobs))


if __name__ == "__main__":
    run()