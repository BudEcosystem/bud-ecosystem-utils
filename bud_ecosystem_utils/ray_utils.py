import os
import re
from ray.job_submission import JobSubmissionClient, JobStatus


def submit_job_to_ray(
    data,
    session_id,
    node_id,
    callback_id,
    requirements_file="requirements.txt",
    entrypoint=None,
    runtime_env=None,
):
    if isinstance(data, dict):
        args = ()
        for key, value in data.items():
            if value is not None:
                args += (f"--{key}", str(value))
    elif isinstance(data, tuple):
        args = data
    else:
        raise ValueError("data should be of type dict or tuple")

    _runtime_env = runtime_env or {}
    blob_provider = os.environ.get("BLOB_PROVIDER", "s3")

    with open(requirements_file, "r") as fin:
        lines = fin.read().splitlines()
        
        requirements = []
        for line in lines:
            if re.split("\<=|\>=|==|\<|\>", line)[0] in ["ray"]:
                continue
            requirements.append(line)

    runtime_env = {
        "working_dir": "./",
        "excludes": [
            ".env",
            ".env.example",
            "poetry.lock",
            "run.sh",
            "node.py",
            "models.py",
        ],
        # "py_modules": ["modules", "config", "utils"],
        "pip": {
            "packages": requirements,
            "pip_version": "==23.3.1;python_version=='3.9'",
        },
        "env_vars": {
            "BLOB_PROVIDER": blob_provider,
            "LOG_PUBLISH_INTERVAL": str(os.environ.get("LOG_PUBLISH_INTERVAL", 30)),
        },
    }
    if blob_provider == "s3":
        keys = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_BUCKET_NAME"]
    elif blob_provider == "gcp":
        keys = ["GOOGLE_BUCKET_NAME"]
        if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
            keys.append("GOOGLE_APPLICATION_CREDENTIALS")
        if os.environ.get("GOOGLE_API_TOKEN"):
            keys.append("GOOGLE_API_TOKEN")
    elif blob_provider == "azure":
        keys = ["AZURE_STORAGE_CONNECTION_STRING", "AZURE_BUCKET_NAME"]

    for key in keys:
        runtime_env["env_vars"][key] = os.environ[key]
    runtime_env.update(_runtime_env)

    runtime_env["env_vars"]["SESSION_ID"] = session_id
    runtime_env["env_vars"]["NODE_ID"] = node_id
    runtime_env["env_vars"]["CALLBACK_ID"] = callback_id
    runtime_env["env_vars"]["INTERNAL_ENDPOINT"] = os.environ["INTERNAL_ENDPOINT"]

    client = JobSubmissionClient(os.environ["RAY_HEAD_URL"])
    job_id = client.submit_job(
        # Entrypoint shell command to execute
        entrypoint=f"python train.py {' '.join(args)}"
        if not entrypoint
        else entrypoint,
        # Path to the local directory that contains the script.py file
        runtime_env=runtime_env,
    )
    return job_id


def stop_ray_job(job_id):
    # TODO: Handle invalid job_id, request failures
    client = JobSubmissionClient(os.environ["RAY_HEAD_URL"])
    status = client.get_job_status(job_id)
    if status in [JobStatus.RUNNING, JobStatus.PENDING]:
        return client.stop_job(job_id)
    return True