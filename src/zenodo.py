"""
Zenodo helpers for uploading and downloading resources associated with the
personalizedVEP manuscript.

Uses the zenodo_client package: https://pypi.org/project/zenodo_client/

Install: pip install zenodo_client

Access token: set ZENODO_API_TOKEN (and optionally ZENODO_SANDBOX_API_TOKEN for
sandbox) in the environment or in a .env file in the repo root (requires
python-dotenv). This module passes the token explicitly so one token
(ZENODO_API_TOKEN) works for both sandbox and production if you use the same
value. Create tokens at https://zenodo.org/.../tokens/new/ and
https://sandbox.zenodo.org/.../tokens/new/
"""

import os
import pathlib
import shutil
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

# Load .env so ZENODO_API_TOKEN is available (zenodo_client reads it from env)
def _load_dotenv():
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    cwd = pathlib.Path.cwd()
    for d in [cwd, cwd.parent]:
        env_file = d / ".env"
        if env_file.is_file():
            load_dotenv(env_file)
            break

_load_dotenv()

import requests
from tqdm.auto import tqdm
from zenodo_client import Creator, Metadata, Zenodo, ensure_zenodo, update_zenodo

# Default Zenodo key and title for this manuscript (used for ensure_zenodo key and metadata)
DEFAULT_REPO_KEY = "personalizedVEP"
DEFAULT_REPO_TITLE = "personalizedVEP"

# Default path to upload (relative to cwd)
DEFAULT_UPLOAD_DIR = os.path.join("results", "data")

# Pystow module used by zenodo_client to store deposition id (for ensure compatibility)
ZENODO_PYSTOW_MODULE = "zenodo_client"


def _base_url(sandbox: bool) -> str:
    return "https://sandbox.zenodo.org/api" if sandbox else "https://zenodo.org/api"


def _metadata_to_api(data: Metadata) -> dict:
    """Convert zenodo_client Metadata to Zenodo API metadata dict."""
    if hasattr(data, "model_dump"):
        out = data.model_dump()
    elif hasattr(data, "dict"):
        out = data.dict()
    else:
        out = {k: v for k, v in vars(data).items() if v is not None}
    # Zenodo API expects creators; ensure list of dicts (Creator may serialize to dict)
    if not out.get("creators"):
        out["creators"] = [{"name": "Unknown"}]
    return out


def _get_stored_deposition_id(key: str) -> Optional[str]:
    try:
        import pystow
        return pystow.get_config(ZENODO_PYSTOW_MODULE, key, raise_on_missing=False)
    except Exception:
        return None


def _set_stored_deposition_id(key: str, dep_id: str) -> None:
    try:
        import pystow
        pystow.write_config(ZENODO_PYSTOW_MODULE, key, dep_id)
    except Exception:
        pass


def get_token(sandbox: bool = False) -> Optional[str]:
    """
    Return Zenodo API token from environment (or .env).
    When sandbox=True, prefers ZENODO_SANDBOX_API_TOKEN, then ZENODO_API_TOKEN.
    When sandbox=False, prefers ZENODO_API_TOKEN, then ZENODO_SANDBOX_API_TOKEN.
    """
    if sandbox:
        return os.environ.get("ZENODO_SANDBOX_API_TOKEN") or os.environ.get("ZENODO_API_TOKEN")
    return os.environ.get("ZENODO_API_TOKEN") or os.environ.get("ZENODO_SANDBOX_API_TOKEN")


def _normalize_upload_dir(dir_path: Optional[str | pathlib.Path] = None, create_if_missing: bool = False) -> pathlib.Path:
    path = pathlib.Path(dir_path or DEFAULT_UPLOAD_DIR)
    if not path.is_absolute():
        path = pathlib.Path.cwd() / path
    if not path.exists():
        if create_if_missing:
            path.mkdir(parents=True, exist_ok=True)
        else:
            raise FileNotFoundError(
                f"Upload directory does not exist: {path}. "
                "Create it or set create_if_missing=True."
            )
    if not path.is_dir():
        raise NotADirectoryError(f"Not a directory: {path}")
    return path


def _collect_paths(dir_path: pathlib.Path) -> list[str]:
    """Return list of absolute path strings for all files under dir_path."""
    return sorted(str(p) for p in dir_path.rglob("*") if p.is_file())


def _create_deposition_api(metadata: dict, token: str, base_url: str) -> dict:
    """Create an unpublished deposition via Zenodo API. Returns deposition dict with id, links."""
    r = requests.post(
        f"{base_url.rstrip('/')}/deposit/depositions",
        params={"access_token": token},
        json={"metadata": metadata},
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def _get_deposition_api(dep_id: str, token: str, base_url: str) -> dict:
    """Fetch deposition by id."""
    r = requests.get(
        f"{base_url.rstrip('/')}/deposit/depositions/{dep_id}",
        params={"access_token": token},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def _upload_file_to_bucket(
    bucket_url: str,
    file_path: str | pathlib.Path,
    key: str,
    token: str,
    timeout: Optional[int] = None,
    retries: int = 3,
    session: Optional[requests.Session] = None,
) -> None:
    """Upload a single file to a Zenodo bucket. Retries on timeout/connection errors."""
    path = pathlib.Path(file_path)
    with open(path, "rb") as f:
        data = f.read()
    # Timeout: use provided value, or scale with size (min 5 min, ~60 s per MB, max 1 h)
    if timeout is None:
        size_mb = len(data) / (1024 * 1024)
        timeout = max(300, min(3600, int(60 * size_mb + 60)))
    url = f"{bucket_url.rstrip('/')}/{key}"
    put_kw = dict(
        params={"access_token": token},
        data=data,
        timeout=timeout,
    )
    last_err = None
    for attempt in range(retries):
        try:
            if session is not None:
                r = session.put(url, **put_kw)
            else:
                r = requests.put(url, **put_kw)
            r.raise_for_status()
            return
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise
    if last_err is not None:
        raise last_err


def _publish_deposition_api(dep_id: str, token: str, base_url: str) -> dict:
    """Publish a deposition. Returns published record."""
    r = requests.post(
        f"{base_url.rstrip('/')}/deposit/depositions/{dep_id}/actions/publish",
        params={"access_token": token},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def default_metadata(
    title: str = DEFAULT_REPO_TITLE,
    upload_type: str = "dataset",
    description: Optional[str] = None,
    creators: Optional[list] = None,
    **kwargs,
) -> Metadata:
    """
    Build Metadata for the personalizedVEP Zenodo record.

    Args:
        title: Record title.
        upload_type: Zenodo upload type (e.g. "dataset").
        description: Optional description.
        creators: Optional list of Creator(...) instances.
        **kwargs: Extra metadata fields passed to Metadata().

    Returns:
        Metadata instance for use with ensure_zenodo().
    """
    return Metadata(
        title=title,
        upload_type=upload_type,
        description=description or "Data and results for the personalizedVEP manuscript.",
        creators=creators or [],
        **kwargs,
    )


def upload_results_data(
    dir_path: Optional[str | pathlib.Path] = None,
    create_if_missing: bool = False,
    key: str = DEFAULT_REPO_KEY,
    title: str = DEFAULT_REPO_TITLE,
    sandbox: bool = False,
    upload_timeout: Optional[int] = None,
    upload_retries: int = 3,
    max_workers: int = 4,
    skip_existing: bool = True,
    **metadata_kwargs,
):
    """
    Ensure the personalizedVEP Zenodo record exists and upload all files from
    results/data/ (or the given directory).

    Uses zenodo_client.ensure_zenodo: on first run creates a new deposition,
    publishes it, and stores the identifier under `key` (in ~/.config/zenodo.ini
    via pystow). On subsequent runs, the deposition is looked up by key and
    data can be uploaded again (versioning is handled by zenodo_client).

    Args:
        dir_path: Directory to upload. Default: results/data (relative to cwd).
        create_if_missing: If True, create dir_path if it does not exist (empty upload).
        key: Key used to store the deposition id locally (default: "personalizedVEP").
        title: Record title in metadata.
        sandbox: If True, use Zenodo sandbox (https://sandbox.zenodo.org).
        upload_timeout: Timeout in seconds per file (default: auto from file size, 5 min–1 h).
        upload_retries: Number of retries on timeout/connection error (default: 3).
        max_workers: Upload up to this many files in parallel (default: 4). Use 1 for sequential.
        skip_existing: If True (default), skip files already in the record with same key and size.
        **metadata_kwargs: Passed to default_metadata() (e.g. description, creators).

    Returns:
        Response from ensure_zenodo (use .json() for the record payload).

    Requires:
        ZENODO_API_TOKEN in the environment.
    """
    path = _normalize_upload_dir(dir_path, create_if_missing=create_if_missing)
    paths = _collect_paths(path)
    if not paths:
        raise FileNotFoundError(f"No files found under {path}. Add files and try again.")
    data = default_metadata(title=title, **metadata_kwargs)
    token = get_token(sandbox=sandbox)
    if not token:
        raise ValueError(
            "Zenodo token not set. Put ZENODO_API_TOKEN (or ZENODO_SANDBOX_API_TOKEN for sandbox) "
            "in .env or the environment."
        )
    base_url = _base_url(sandbox)
    path_root = pathlib.Path(path).resolve()

    def _do_upload() -> dict:
        dep_id = _get_stored_deposition_id(key)
        if dep_id:
            dep = _get_deposition_api(dep_id, token, base_url)
            bucket_url = dep["links"]["bucket"]
        else:
            meta = _metadata_to_api(data)
            dep = _create_deposition_api(meta, token, base_url)
            bucket_url = dep["links"]["bucket"]
        # Existing files in deposition: key -> size (Zenodo uses "filename" and "filesize")
        existing = {}
        if skip_existing:
            for f in dep.get("files", []):
                k = f.get("filename") or f.get("key")
                s = f.get("filesize") or f.get("size")
                if k is not None:
                    existing[k] = s
        paths_to_upload = []
        for fp in paths:
            rel = pathlib.Path(fp).resolve().relative_to(path_root)
            key_str = str(rel).replace(os.sep, "/")
            if skip_existing and key_str in existing:
                local_size = pathlib.Path(fp).stat().st_size
                if existing[key_str] == local_size:
                    continue
            paths_to_upload.append(fp)
        skipped = len(paths) - len(paths_to_upload)
        if skipped and skip_existing:
            tqdm.write(f"Skipping {skipped} file(s) already present in record (same key and size).")
        if not paths_to_upload:
            return _get_deposition_api(str(dep["id"]), token, base_url)
        session = requests.Session() if max_workers <= 1 else None

        def _upload_one(fp: str) -> None:
            rel = pathlib.Path(fp).resolve().relative_to(path_root)
            key_str = str(rel).replace(os.sep, "/")
            _upload_file_to_bucket(
                bucket_url, fp, key_str, token,
                timeout=upload_timeout, retries=upload_retries,
                session=session,
            )

        try:
            if max_workers <= 1:
                for fp in tqdm(paths_to_upload, desc="Uploading to Zenodo", unit="file"):
                    _upload_one(fp)
            else:
                workers = min(max_workers, len(paths_to_upload))
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    futures = {ex.submit(_upload_one, fp): fp for fp in paths_to_upload}
                    for fut in tqdm(as_completed(futures), total=len(paths_to_upload), desc="Uploading to Zenodo", unit="file"):
                        fut.result()
        finally:
            if session is not None:
                session.close()
        if not dep_id:
            _publish_deposition_api(str(dep["id"]), token, base_url)
            _set_stored_deposition_id(key, str(dep["id"]))
        return _get_deposition_api(str(dep["id"]), token, base_url)

    try:
        dep = _do_upload()
        # Return a response-like object so .json() works if caller expects it
        class _DepResponse:
            def __init__(self, d): self._d = d
            def json(self): return self._d
        return _DepResponse(dep)
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 403:
            hint = (
                "403 FORBIDDEN: Sandbox and production use different tokens. "
                "With sandbox=True use a token from https://sandbox.zenodo.org/account/settings/applications/tokens/new/ "
                "(set ZENODO_SANDBOX_API_TOKEN in .env). With sandbox=False use a token from https://zenodo.org/.../tokens/new/ "
                "(set ZENODO_API_TOKEN in .env)."
            )
            raise type(e)(f"{e}\n\n{hint}", response=e.response) from e
        raise


def create_draft(
    key: str = DEFAULT_REPO_KEY,
    title: str = DEFAULT_REPO_TITLE,
    sandbox: bool = False,
    **metadata_kwargs,
) -> dict:
    """
    Create an unpublished Zenodo draft (deposition) and store its id under `key`.
    If a draft for this key already exists, return it without creating a new one.
    Use this first; then call upload_to_draft() to add files. That way failed
    uploads can be retried against the same draft.

    Args:
        key: Key to store the deposition id (default: "personalizedVEP").
        title: Record title in metadata.
        sandbox: If True, use Zenodo sandbox.
        **metadata_kwargs: Passed to default_metadata() (e.g. description, creators).

    Returns:
        Deposition dict (id, links, metadata, ...).
    """
    token = get_token(sandbox=sandbox)
    if not token:
        raise ValueError(
            "Zenodo token not set. Put ZENODO_API_TOKEN (or ZENODO_SANDBOX_API_TOKEN for sandbox) "
            "in .env or the environment."
        )
    base_url = _base_url(sandbox)
    dep_id = _get_stored_deposition_id(key)
    if dep_id:
        dep = _get_deposition_api(dep_id, token, base_url)
        if dep.get("state") == "unsubmitted":
            return dep
        raise ValueError(
            f"Stored deposition {dep_id} for key '{key}' is already published (state={dep.get('state')}). "
            "Use a different key for a new draft, or create a new version from the published record."
        )
    data = default_metadata(title=title, **metadata_kwargs)
    meta = _metadata_to_api(data)
    dep = _create_deposition_api(meta, token, base_url)
    _set_stored_deposition_id(key, str(dep["id"]))
    return dep


def upload_to_draft(
    key: Optional[str] = None,
    deposition_id: Optional[str | int] = None,
    dir_path: Optional[str | pathlib.Path] = None,
    sandbox: bool = False,
    skip_existing: bool = True,
    upload_timeout: Optional[int] = None,
    upload_retries: int = 3,
    max_workers: int = 4,
    publish: bool = False,
) -> dict:
    """
    Upload files from a directory to an existing draft (unpublished deposition).
    Use create_draft() first so the same draft is reused even if uploads fail.

    Args:
        key: Key used when creating the draft (default: "personalizedVEP"). Ignored if deposition_id is set.
        deposition_id: Deposition id if not using stored key.
        dir_path: Directory to upload (default: results/data).
        sandbox: If True, use Zenodo sandbox.
        skip_existing: If True, skip files already in the draft with same key and size.
        upload_timeout: Timeout in seconds per file.
        upload_retries: Retries on timeout/connection error.
        max_workers: Parallel uploads (1 = sequential).
        publish: If True, publish the draft after uploading.

    Returns:
        Deposition dict (or published record if publish=True).
    """
    if key is None and deposition_id is None:
        raise ValueError("Provide key or deposition_id.")
    token = get_token(sandbox=sandbox)
    if not token:
        raise ValueError(
            "Zenodo token not set. Put ZENODO_API_TOKEN (or ZENODO_SANDBOX_API_TOKEN for sandbox) "
            "in .env or the environment."
        )
    base_url = _base_url(sandbox)
    if deposition_id is not None:
        dep_id = str(deposition_id)
    else:
        dep_id = _get_stored_deposition_id(key)
        if not dep_id:
            raise ValueError(
                f"No draft found for key '{key}'. Run create_draft(key='{key}', ...) first."
            )
    path = _normalize_upload_dir(dir_path, create_if_missing=False)
    paths = _collect_paths(path)
    if not paths:
        raise FileNotFoundError(f"No files found under {path}.")
    dep = _get_deposition_api(dep_id, token, base_url)
    if dep.get("state") != "unsubmitted":
        raise ValueError(
            f"Deposition {dep_id} is not a draft (state={dep.get('state')}). "
            "Upload only to unpublished drafts."
        )
    bucket_url = dep["links"]["bucket"]
    path_root = pathlib.Path(path).resolve()
    existing = {}
    if skip_existing:
        for f in dep.get("files", []):
            k = f.get("filename") or f.get("key")
            s = f.get("filesize") or f.get("size")
            if k is not None:
                existing[k] = s
    paths_to_upload = []
    for fp in paths:
        rel = pathlib.Path(fp).resolve().relative_to(path_root)
        key_str = str(rel).replace(os.sep, "/")
        if skip_existing and key_str in existing:
            local_size = pathlib.Path(fp).stat().st_size
            if existing[key_str] == local_size:
                continue
        paths_to_upload.append(fp)
    skipped = len(paths) - len(paths_to_upload)
    if skipped and skip_existing:
        tqdm.write(f"Skipping {skipped} file(s) already present in draft (same key and size).")
    if not paths_to_upload:
        if publish:
            dep = _publish_deposition_api(dep_id, token, base_url)
        else:
            dep = _get_deposition_api(dep_id, token, base_url)
        return dep
    session = requests.Session() if max_workers <= 1 else None

    def _upload_one(fp: str) -> None:
        rel = pathlib.Path(fp).resolve().relative_to(path_root)
        key_str = str(rel).replace(os.sep, "/")
        _upload_file_to_bucket(
            bucket_url, fp, key_str, token,
            timeout=upload_timeout, retries=upload_retries,
            session=session,
        )

    try:
        if max_workers <= 1:
            for fp in tqdm(paths_to_upload, desc="Uploading to Zenodo", unit="file"):
                _upload_one(fp)
        else:
            workers = min(max_workers, len(paths_to_upload))
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = {ex.submit(_upload_one, fp): fp for fp in paths_to_upload}
                for fut in tqdm(as_completed(futures), total=len(paths_to_upload), desc="Uploading to Zenodo", unit="file"):
                    fut.result()
    finally:
        if session is not None:
            session.close()
    if publish:
        dep = _publish_deposition_api(dep_id, token, base_url)
    else:
        dep = _get_deposition_api(dep_id, token, base_url)
    return dep


def build_zip_from_items(
    items: dict[str, str | pathlib.Path],
    zip_path: str | pathlib.Path,
    verbose: bool = True,
) -> pathlib.Path:
    """
    Build a zip file from a dict mapping paths inside the zip to local files or dirs.

    Args:
        items: Dict[zip_path, local_path]. Keys are paths inside the zip (e.g. "data/foo.parquet"
            or "plots/"). Values are local file or directory paths. If local_path is a directory,
            all its contents are added under zip_path/ in the zip.
        zip_path: Path to write the zip file (created or overwritten).
        verbose: If True, show progress with tqdm.

    Returns:
        Resolved path to the written zip file.

    Example:
        build_zip_from_items({
            "data/results.parquet": "results/data/results.parquet",
            "plots/": "results/plots/",
        }, "archive.zip")
    """
    zip_path = pathlib.Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    for arc_prefix, local in items.items():
        local = pathlib.Path(local).resolve()
        arc_prefix = arc_prefix.replace(os.sep, "/").rstrip("/")
        if local.is_file():
            entries.append((arc_prefix, local, None))
        elif local.is_dir():
            for f in local.rglob("*"):
                if f.is_file():
                    rel = f.relative_to(local)
                    arc = f"{arc_prefix}/{rel}".replace(os.sep, "/")
                    entries.append((arc, f, None))
        else:
            raise FileNotFoundError(f"Not a file or directory: {local}")
    it = tqdm(entries, desc="Building zip", unit="file", disable=not verbose)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for arcname, src, _ in it:
            zf.write(src, arcname)
    return zip_path.resolve()


def upload_zip_to_draft(
    items: dict[str, str | pathlib.Path],
    key: Optional[str] = None,
    deposition_id: Optional[str | int] = None,
    zip_filename: str = "data.zip",
    sandbox: bool = False,
    skip_existing: bool = True,
    upload_timeout: Optional[int] = None,
    upload_retries: int = 3,
    publish: bool = False,
) -> dict:
    """
    Build a zip from a dict of (zip path -> local file or dir), then upload that
    single zip file to an existing draft. Use create_draft() first.

    Args:
        items: Dict mapping path inside the zip -> local file or directory.
            E.g. {"data/": "results/data/", "plots/fig.png": "results/plots/fig.png"}.
            Directory values add all contents under the given zip path.
        key: Key used when creating the draft. Ignored if deposition_id is set.
        deposition_id: Deposition id if not using stored key.
        zip_filename: Name of the zip file in the draft (default: "data.zip").
        sandbox: If True, use Zenodo sandbox.
        skip_existing: If True, skip upload if a file with the same name and size exists.
        upload_timeout: Timeout in seconds for the upload.
        upload_retries: Retries on timeout/connection error.
        publish: If True, publish the draft after uploading.

    Returns:
        Deposition dict (or published record if publish=True).
    """
    if key is None and deposition_id is None:
        raise ValueError("Provide key or deposition_id.")
    if not items:
        raise ValueError("items must be a non-empty dict of zip_path -> local_path.")
    token = get_token(sandbox=sandbox)
    if not token:
        raise ValueError(
            "Zenodo token not set. Put ZENODO_API_TOKEN (or ZENODO_SANDBOX_API_TOKEN for sandbox) "
            "in .env or the environment."
        )
    base_url = _base_url(sandbox)
    if deposition_id is not None:
        dep_id = str(deposition_id)
    else:
        dep_id = _get_stored_deposition_id(key)
        if not dep_id:
            raise ValueError(
                f"No draft found for key '{key}'. Run create_draft(key='{key}', ...) first."
            )
    dep = _get_deposition_api(dep_id, token, base_url)
    if dep.get("state") != "unsubmitted":
        raise ValueError(
            f"Deposition {dep_id} is not a draft (state={dep.get('state')}). "
            "Upload only to unpublished drafts."
        )
    bucket_url = dep["links"]["bucket"]
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = pathlib.Path(tmpdir) / zip_filename
        build_zip_from_items(items, zip_path, verbose=True)
        zip_size = zip_path.stat().st_size
        if skip_existing:
            for f in dep.get("files", []):
                k = f.get("filename") or f.get("key")
                s = f.get("filesize") or f.get("size")
                if k == zip_filename and s == zip_size:
                    tqdm.write(f"Zip '{zip_filename}' already in draft (same name and size). Set skip_existing=False to replace.")
                    if publish:
                        dep = _publish_deposition_api(dep_id, token, base_url)
                    else:
                        dep = _get_deposition_api(dep_id, token, base_url)
                    return dep
        _upload_file_to_bucket(
            bucket_url,
            zip_path,
            zip_filename,
            token,
            timeout=upload_timeout,
            retries=upload_retries,
        )
    if publish:
        dep = _publish_deposition_api(dep_id, token, base_url)
    else:
        dep = _get_deposition_api(dep_id, token, base_url)
    return dep


def publish_draft(
    key: Optional[str] = None,
    deposition_id: Optional[str | int] = None,
    sandbox: bool = False,
) -> dict:
    """
    Publish a draft (unpublished deposition). Use after create_draft() and upload_to_draft().

    Args:
        key: Key used when creating the draft. Ignored if deposition_id is set.
        deposition_id: Deposition id if not using stored key.
        sandbox: If True, use Zenodo sandbox.

    Returns:
        Published record dict.
    """
    if key is None and deposition_id is None:
        raise ValueError("Provide key or deposition_id.")
    token = get_token(sandbox=sandbox)
    if not token:
        raise ValueError(
            "Zenodo token not set. Put ZENODO_API_TOKEN (or ZENODO_SANDBOX_API_TOKEN for sandbox) "
            "in .env or the environment."
        )
    base_url = _base_url(sandbox)
    if deposition_id is not None:
        dep_id = str(deposition_id)
    else:
        dep_id = _get_stored_deposition_id(key)
        if not dep_id:
            raise ValueError(f"No draft found for key '{key}'.")
    return _publish_deposition_api(dep_id, token, base_url)


def update_deposition(deposition_id: str | int, paths: list[str | pathlib.Path], sandbox: bool = False):
    """
    Upload files to an existing Zenodo deposition (by id).

    Uses zenodo_client.update_zenodo. Use this when you already have a
    deposition id and want to add or replace files without going through
    ensure_zenodo.

    Args:
        deposition_id: Zenodo deposition id (string or int).
        paths: List of local file paths to upload.
        sandbox: If True, use Zenodo sandbox.

    Requires:
        ZENODO_API_TOKEN (or ZENODO_SANDBOX_API_TOKEN for sandbox) in .env or environment.
    """
    path_strs = [str(p) for p in paths]
    token = get_token(sandbox=sandbox)
    if not token:
        raise ValueError(
            "Zenodo token not set. Put ZENODO_API_TOKEN (or ZENODO_SANDBOX_API_TOKEN for sandbox) "
            "in .env or the environment."
        )
    return update_zenodo(str(deposition_id), path_strs, sandbox=sandbox, access_token=token)


def get_zenodo(sandbox: bool = False, access_token: Optional[str] = None) -> Zenodo:
    """Return a Zenodo client instance (for downloads and lookups)."""
    token = access_token or get_token(sandbox=sandbox)
    return Zenodo(sandbox=sandbox, access_token=token)


def get_latest_record(record_id: str | int, sandbox: bool = False) -> dict:
    """
    Fetch the latest version of a published record (by conceptrecid or record id).

    Uses zenodo_client Zenodo.get_latest_record().
    """
    return get_zenodo(sandbox=sandbox).get_latest_record(str(record_id))


def list_record_files(record_id: str | int, sandbox: bool = False) -> list[dict]:
    """
    List file entries for the latest version of a record.
    Each entry has "key", "filename", "links", etc.
    """
    record = get_latest_record(record_id, sandbox=sandbox)
    return record.get("files", [])


def download_file(
    record_id: str | int,
    filename: str,
    dest_dir: Optional[str | pathlib.Path] = None,
    sandbox: bool = False,
) -> pathlib.Path:
    """
    Download a single file from the latest version of a Zenodo record.

    Uses zenodo_client Zenodo.download_latest(), which stores the file in
    the pystow cache (~/.data/zenodo/...). If dest_dir is given, the file
    is also copied there.

    Args:
        record_id: Published record id or conceptrecid.
        filename: File key/filename as in the record (e.g. "data.parquet").
        dest_dir: Optional directory to copy the file into (preserves filename).
        sandbox: If True, use Zenodo sandbox.

    Returns:
        Path to the file (in dest_dir if provided, else in pystow cache).
    """
    zenodo = get_zenodo(sandbox=sandbox)
    cached = zenodo.download_latest(str(record_id), filename)
    if dest_dir is None:
        return pathlib.Path(cached)
    dest = pathlib.Path(dest_dir) / pathlib.Path(filename).name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cached, dest)
    return dest.resolve()


def download_record_files(
    record_id: str | int,
    dest_dir: str | pathlib.Path,
    sandbox: bool = False,
    verbose: bool = True,
) -> list[pathlib.Path]:
    """
    Download all files from the latest version of a Zenodo record into a local
    directory. Creates subdirs when file keys contain slashes.

    Args:
        record_id: Published record id or conceptrecid.
        dest_dir: Local directory to write files.
        sandbox: If True, use Zenodo sandbox.
        verbose: If True, show progress bar.

    Returns:
        List of paths to downloaded files.
    """
    files = list_record_files(record_id, sandbox=sandbox)
    dest_root = pathlib.Path(dest_dir)
    dest_root.mkdir(parents=True, exist_ok=True)
    paths = []
    it = tqdm(files, desc="Downloading from Zenodo", disable=not verbose)
    for f in it:
        key = f.get("key") or f.get("filename", "unknown")
        it.set_postfix_str(key[:50])
        local = dest_root / key
        local.parent.mkdir(parents=True, exist_ok=True)
        p = download_file(record_id, key, dest_dir=None, sandbox=sandbox)
        shutil.copy2(p, local)
        paths.append(local.resolve())
    return paths
