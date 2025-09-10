from pathlib import Path

import fev
import modal

app = modal.App(name="timecopilot-fev")
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu24.04",
        add_python="3.11",
    )
    # uploaded to s3 by makefile
    .apt_install("git")
    .pip_install("uv")
    .add_local_file("pyproject.toml", "/root/pyproject.toml", copy=True)
    .add_local_file(".python-version", "/root/.python-version", copy=True)
    .add_local_file("uv.lock", "/root/uv.lock", copy=True)
    .workdir("/root")
    .run_commands("uv pip install . --system --compile-bytecode")
)
aws_secret = modal.Secret.from_name(
    "aws-secret",
    required_keys=["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"],
)
volume = {
    "/s3-bucket": modal.CloudBucketMount(
        bucket_name="timecopilot-fev",
        secret=aws_secret,
    )
}


@app.function(
    image=image,
    volumes=volume,
    # 3 hours timeout
    timeout=60 * 60 * 3,
    gpu="A10G",
    # as my local
    cpu=8,
    secrets=[modal.Secret.from_name("hf-secret")],
)
def evaluate_task_modal(task: fev.Task):
    from .evaluate_model import evaluate_task

    evaluation_summary = evaluate_task(task=task)
    save_path = Path(f"/s3-bucket/results/{task.dataset_config}.csv")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_summary.to_csv(save_path, index=False)


@app.local_entrypoint()
def main():
    from .evaluate_model import tasks

    list(evaluate_task_modal.map(tasks()[:2]))
