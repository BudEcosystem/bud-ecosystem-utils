import shutil
import zipfile
import requests
import json
import io
import os

from pathlib import Path
from uuid import UUID
from huggingface_hub import HfApi
from huggingface_hub import DatasetFilter
from urllib.parse import urljoin, quote_plus
from tqdm import tqdm


class BudMLOpsClient:
    def __init__(self, api_url: str = None, api_token: str = None) -> None:
        self.api_url = api_url or os.environ["BUD_MLOPS_API_URL"]
        self.api_token = api_token or os.environ["BUD_MLOPS_API_TOKEN"]
        self.session = None
        self.connect()

    def connect(self):
        sess = requests.Session()
        sess.headers.update({"x-token": self.api_token})
        resp = sess.get(self.multi_urljoin(self.api_url, "/ping"))
        if resp.status_code != 200:
            raise ConnectionError(
                f"Server returned an invalid response [{resp.status_code}]: {resp.content.decode()}"
            )
        self.session = sess

    def api_request(self, method, path, raise_for_status=True, **kwargs):
        method = getattr(self.session, method)
        url = self.multi_urljoin(self.api_url, path)
        resp = method(url, **kwargs)

        if raise_for_status:
            resp.raise_for_status()

        return resp

    @staticmethod
    def multi_urljoin(*parts):
        return urljoin(
            parts[0], "/".join(quote_plus(part.strip("/"), safe="/") for part in parts[1:])
        )

    def fetch_dataset(self, dataset_id=None, dataset_name=None):
        params = {}
        endpoint = "/dataset/"
        if dataset_id is not None:
            endpoint += dataset_id
        if dataset_name is not None:
            params["dataset_name"] = dataset_name
        resp = self.api_request(
            "get", endpoint, params=params
        )
        if not resp.json()["status"]:
            raise ValueError("Dataset fetching failed!!!")

        dataset = resp.json()["data"]
        if not len(dataset):
            raise ValueError(f"Dataset doesn't exist")

        return dataset[0] if dataset_id is None else dataset

    def fetch_model(self, model_id=None, model_name=None):
        params = {}
        endpoint = "/models/"
        if model_id is not None:
            endpoint += model_id
        if model_name is not None:
            params["model_name"] = model_name
        resp = self.api_request(
            "get", endpoint, params=params
        )
        if not resp.json()["status"]:
            raise ValueError("Model fetching failed!!!")

        model = resp.json()["data"]
        if not len(model):
            raise ValueError(f"Model doesn't exist")

        return model[0] if model_id is None else model

    def download_dataset(self, dataset_name: str, save_dir: str = None):
        dataset = self.fetch_dataset(dataset_name=dataset_name)
        endpoint = f"/dataset/download/{dataset['dataset_id']}"

        chunk_size = 100000000  # size of 1 chunk to download 100000000 = 100 MB
        save_dir = save_dir or "."

        buffer = io.BytesIO()

        with self.api_request(
            "get", endpoint, raise_for_status=False, stream=True, allow_redirects=True
        ) as resp:
            if resp.status_code == 204:
                # raise NotImplementedError("Download from external source")
                print(
                    "[WARNING] The specified dataset is from an external source, skipping download..."
                )
                return dataset
            resp.raise_for_status()

            total_size = resp.headers.get("Content-Length")  # total size in bytes
            if total_size is not None:
                total_size = int(total_size)

            for chunk in tqdm(
                resp.iter_content(chunk_size=chunk_size),
                total=total_size // chunk_size,
            ):  # Download a 100 mb chunk
                buffer.write(chunk)

        archive_file = zipfile.ZipFile(buffer)
        for name in archive_file.namelist():
            if name.endswith("/"):
                Path(os.path.join(save_dir, name)).mkdir(exist_ok=True, parents=True)
            else:
                with archive_file.open(name) as file:
                    content = file.read()
                with open(os.path.join(save_dir, name), "wb") as file:
                    file.write(content)

        print("[INFO] Download complete")
        dataset["source"] = os.path.join(save_dir, dataset["dataset_id"])
        return dataset

    def upload_dataset(
        self, dataset_name: str, metadata_filepath: str, image_dirpath: str = None
    ):
        source_type = 1
        _type = 0
        archive_file = None

        image_dirpath = image_dirpath or None

        if not os.path.isfile(metadata_filepath):
            raise FileNotFoundError(
                f"Metedata file '{metadata_filepath}' doesn't exist."
            )
        if image_dirpath:
            if os.path.isfile(image_dirpath) and image_dirpath.endswith(".zip"):
                archive_file = image_dirpath
            elif not os.path.isdir(image_dirpath):
                raise NotADirectoryError(
                    f"Image directory '{image_dirpath}' doesn't exist"
                )

        if image_dirpath is not None and archive_file is None:
            archive_file = create_zipfile_buffer_from_dir(
                image_dirpath
            ).getbuffer()
        elif archive_file is not None:
            archive_file = open(archive_file, "rb")

        if archive_file is not None:
            _type = 1

        resp = self.api_request(
            "post",
            "/dataset/",
            files={
                "metadata_file": (
                    "metadata" + Path(metadata_filepath).suffix,
                    open(metadata_filepath, "rb"),
                ),
                "archive_file": ("images.zip", archive_file),
                "name": (None, dataset_name),
                "source": (None, None),
                "source_type": (None, source_type),
                "type": (None, _type),
            },
        )
        print("[INFO] Dataset succefully created")
        return resp.json()

    def register_model(self, model_name: str, source: str, model_type: str, family: str, base_model_id: str = None):
        families = {"causal": 0, "sd1_5": 1, "sdxl": 2}
        model_types = {"adapter": 0, "delta": 1, "full": 2}
        if model_type not in model_types:
            raise ValueError(f"Only supports model_type of the following {tuple(model_types.keys())}")
        if family not in families:
            raise ValueError(f"Only supports family of the following {tuple(families.keys())}")

        if source.split("://")[0] in ["s3", "gs", "azure", "http", "https", "ftp", "sftp"]:
            source_type = 2
        else:
            source_type = 0

        resp = self.api_request(
            "post",
            "/models/",
            files={
                "name": (None, model_name),
                "source": (None, source),
                "type": (None, model_types[model_type]),
                "source_type": (None, source_type),
                "family": (None, families[family]),
                "base_model_id": (None, base_model_id)
            },
        )
        print("[INFO] Model succefully created")
        return resp.json()


def create_zipfile_buffer_from_dir(basedir, exclusions=None):
    zip_bytes_io = io.BytesIO()
    base_dir = basedir.rstrip("/")
    base_name = os.path.dirname(base_dir)
    exclusions = exclusions or []
    with zipfile.ZipFile(zip_bytes_io, "w", zipfile.ZIP_DEFLATED) as zipped:
        for dirpath, subdirs, files in os.walk(basedir):
            keyname = dirpath.replace(base_dir, "").strip("/")
            skip = False
            for excl in exclusions:
                if dirpath.startswith(os.path.join(base_dir, excl)):
                    skip = True
                    break
            if skip:
                continue
            zipped.write(dirpath, keyname)
            for filename in files:
                zipped.write(
                    os.path.join(dirpath, filename), os.path.join(keyname, filename)
                )
    return zip_bytes_io


def does_dataset_exist_in_hf_hub(dataset_id: str):
    hf_api = HfApi()
    dataset = list(hf_api.list_datasets(filter=DatasetFilter(dataset_name=dataset_id)))
    return bool(len(dataset))


def does_model_exist_in_hf_hub(model_path: str) -> bool:
    if not model_path:
        return False
    url = f"https://huggingface.co/{model_path}/resolve/main/.gitattributes"
    response = requests.head(url)
    if response.status_code == 200:
        return True
    else:
        return False


def load_metadata(metadata_path):
    if not os.path.isfile(metadata_path):
        metadata_path = os.path.join(metadata_path, "metadata.jsonl")
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"'metadata.jsonl' file missing in '{metadata_path}'")

    with open(metadata_path, "r") as file:
        ext = os.path.splitext(metadata_path)[-1]
        if ext == ".json":
            metadata = json.load(file)
        elif ext in [".jsonl", ".txt"]:
            lines = file.read().splitlines()
            metadata = [json.loads(line) for line in lines]
    return metadata


def save_as_metadata(metadata, save_path):
    Path(os.path.dirname(save_path)).mkdir(exist_ok=True, parents=True)
    with open(save_path, "w") as f:
        if save_path.endswith(".jsonl"):
            for entry in metadata:
                json.dump(entry, f)
                f.write("\n")
        elif save_path.endswith(".json"):
            json.dump(metadata, f)
        elif save_path.endswith(".txt"):
            f.write("\n".join(metadata) if isinstance(metadata, list) else metadata)
        else:
            raise NotImplementedError(f"{Path(save_path).suffix} is not supported")


def extract_and_process_image_archives(dataset_dir: str, image_column: str, skip_metadata: bool = False):
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Couldn't locate dataset dir '{dataset_dir}'")

    if not os.path.isdir(os.path.join(dataset_dir, "images")):
        if not os.path.isfile(os.path.join(dataset_dir, "images.zip")):
            raise FileNotFoundError(
                f"Couldn't find any image archives at '{dataset_dir}'"
            )

        try:
            with zipfile.ZipFile(os.path.join(dataset_dir, "images.zip"), "r") as zip_ref:
                zip_ref.extractall(os.path.join(dataset_dir, "images"))
        except Exception as e:
            print(f"Couldn't extract images.zip file '{os.path.join(dataset_dir, 'images.zip')}', e => {e}")
            raise Exception("Couldn't extract images.zip file")
        
        if skip_metadata:
            return os.path.join(dataset_dir, "images")

        metadata_path = None
        if os.path.isfile(os.path.join(dataset_dir, "metadata.jsonl")):
            metadata_path = os.path.join(dataset_dir, "metadata.jsonl")
        elif os.path.isfile(os.path.join(dataset_dir, "metadata.json")):
            metadata_path = os.path.join(dataset_dir, "metadata.json")
        else:
            raise FileNotFoundError(
                f"Couldn't find any metadata file at '{metadata_path}'"
            )

        shutil.copyfile(
            metadata_path,
            os.path.join(dataset_dir, f"metadata_copy{Path(metadata_path).suffix}"),
        )
        metadata = load_metadata(metadata_path)
        for data in metadata:
            if image_column not in data:
                raise ValueError(f"Image column '{image_column}' missing in {data}")
            data[image_column] = os.path.join(dataset_dir, "images", data[image_column])

        save_as_metadata(metadata, metadata_path)
        return dataset_dir


def resolve_dataset(dataset_name_or_id, **kwargs):
    try:
        obj = UUID(dataset_name_or_id, version=4)
        is_uuid = str(obj) == dataset_name_or_id
    except ValueError:
        is_uuid = False
    
    if is_uuid:
        mlops_client = BudMLOpsClient()
        dataset = mlops_client.fetch_dataset(dataset_id=dataset_name_or_id)
        dataset_name_or_id = dataset["source"]
        dataset_type = dataset["type"]
    else:
        dataset_type = 1

    if does_dataset_exist_in_hf_hub(dataset_name_or_id):
        return dataset_name_or_id, "hf", is_uuid
    elif dataset_name_or_id.startswith("s3://"):
        from bud_ecosystem_utils.blob import BlobService
        blob_service = BlobService()
        savepath = blob_service.download_file(dataset_name_or_id, os.path.join(Path.home(), ".cache", "bud_ecosystem"))
    else:
        raise NotImplementedError("Only supports Hugging Face and AWS S3 datasets")

    if "image_column" in kwargs:
        extract_and_process_image_archives(savepath)
    elif dataset_type == 2:
        savepath = extract_and_process_image_archives(savepath, skip_metadata=True)

    return savepath, "local", is_uuid


def resolve_model(model_name_or_id, **kwargs):
    try:
        obj = UUID(model_name_or_id, version=4)
        is_uuid = str(obj) == model_name_or_id
    except ValueError:
        is_uuid = False
    
    if is_uuid:
        mlops_client = BudMLOpsClient()
        model = mlops_client.fetch_model(model_id=model_name_or_id)
        model_name_or_id = model["source"]

    if does_model_exist_in_hf_hub(model_name_or_id):
        return model_name_or_id, "hf", is_uuid
    else:
        raise NotImplementedError("Only supports Hugging Face models")
    