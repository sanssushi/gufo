#!/usr/bin/env python3
"""Install the fork's catalog entries using small local files, not model weights."""
import copy
import hashlib
import http.server
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading

root = Path(__file__).resolve().parents[3]
extension = root / "src/models/qwen38_flash_next"
manifest = json.loads((extension / "extension.json").read_text())
lock = json.loads((extension / "models.lock.json").read_text())
ids = {"qfn-iq3-xxs": "IQ3_XXS", "qfn-iq3-s": "IQ3_S",
       "qfn-iq4-xs": "UD-IQ4_XS"}
old_ids = {"flash-next-xxs", "flash-next-s", "flash-next-iq4"}
models = {model["id"]: model for model in manifest["models"]}
assert set(ids) <= models.keys(), f"missing fork models: {set(ids) - models.keys()}"
assert set(ids) <= set(manifest["providers"][0]["model_ids"])
assert old_ids.isdisjoint(models)
assert old_ids.isdisjoint(manifest["providers"][0]["model_ids"])
assert old_ids.isdisjoint(lock["models"])

# Keep the declared paths and component selection, but replace the large
# remote files' digests with small fixture bytes for the local HTTP server.
files = {}
fixture_models = []
fixture_lock = {}
for model_id, quant in ids.items():
    model = models[model_id]
    iq4 = model_id == "qfn-iq4-xs"
    assert model["provider"] == "flash-next-q4"
    components = {component["role"]: component for component in model["components"]}
    assert set(components) == {"weights", "mtp", "mmproj"}
    weights = components["weights"]
    weights_repo = ("unsloth/Qwen3.8-Flash-Next-GGUF" if iq4
                    else "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF")
    assert weights["source"]["hf"] == weights_repo
    assert weights["include"] == [f"{quant}/*"]
    assert weights["entry"].startswith(f"{quant}/")
    assert weights["entry"].endswith("-00001-of-00004.gguf" if iq4
                                     else "-00001-of-00002.gguf")
    assert weights["install"] == "required"
    assert components["mtp"]["option"] == "--mtp-model"
    assert components["mtp"]["source"]["hf"] == "unsloth/Qwen3.8-Flash-Next-GGUF"
    assert components["mtp"]["include"] == ["MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf"]
    assert components["mmproj"]["option"] == "--mmproj"
    assert components["mmproj"]["source"]["hf"] == weights["source"]["hf"]
    assert components["mmproj"]["include"] == (
        ["mmproj-BF16.gguf"] if iq4 else ["mmproj-Qwen3.8-Flash-Next-BF16.gguf"])
    assert components["mtp"]["install"] == components["mmproj"]["install"] == "default"
    locked = copy.deepcopy(lock["models"][model_id])
    assert set(locked) == set(components)
    assert len(locked["weights"]["files"]) == (4 if iq4 else 2)
    assert weights["entry"] in {item["path"] for item in locked["weights"]["files"]}
    assert all(item["path"].startswith(f"{quant}/") for item in locked["weights"]["files"])
    for role, component in components.items():
        entry = locked[role]
        assert entry["source"] == component["source"]
        assert entry["include"] == component["include"]
        for item in entry["files"]:
            payload = ("fixture:" + item["path"]).encode()
            files[item["path"]] = payload
            item["size"] = len(payload)
            item["sha256"] = hashlib.sha256(payload).hexdigest()
    fixture_models.append(model)
    fixture_lock[model_id] = locked

with tempfile.TemporaryDirectory() as temporary:
    directory = Path(temporary)
    env = {**os.environ, "GUFO_EXTENSIONS_DIR": str(directory / "extensions"),
           "GUFO_MODELS_DIR": str(directory / "models")}
    listing = subprocess.run([sys.argv[1], "model", "list", "--json"], env=env,
                             text=True, capture_output=True, check=True)
    catalog = json.loads(listing.stdout)
    rows = {row["id"]: row for row in catalog["models"]}
    assert set(ids) <= rows.keys(), f"missing built-in fork models: {set(ids) - rows.keys()}"
    assert old_ids.isdisjoint(rows)
    for model_id in ids:
        assert rows[model_id]["downloadable"] and rows[model_id]["extensionStatus"] == "built-in"
        assert rows[model_id]["provider"] == "flash-next-q4"
        listed = {item["role"]: item for item in rows[model_id]["components"]}
        for role, pinned in lock["models"][model_id].items():
            source = pinned["source"]
            assert listed[role]["source"] == f'{source["hf"]}@{source["revision"]}'
            assert listed[role]["bytes"] == sum(item["size"] for item in pinned["files"])


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        name = self.path.split("/resolve/", 1)[1].split("/", 1)[1]
        data = files[name]
        self.send_response(206)
        self.send_header("Content-Range", f"bytes 0-{len(data) - 1}/{len(data)}")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
try:
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        extension_store = directory / "extensions"
        source = extension_store / "qwen38_flash_next/source"
        source.mkdir(parents=True)
        fixture_manifest = {"schema": 1, "name": "qwen38_flash_next",
                            "version": "1.0.0", "models": fixture_models}
        (source / "extension.json").write_text(json.dumps(fixture_manifest))
        (source / "models.lock.json").write_text(json.dumps({"schema": 1, "models": fixture_lock}))
        (extension_store / "installed.json").write_text(json.dumps({
            "schema": 1, "extensions": [{"name": "qwen38_flash_next", "version": "1.0.0",
                "digest": "", "source": {"kind": "path", "location": str(source), "revision": ""}}]}))
        store = directory / "models"
        env = {**os.environ, "GUFO_EXTENSIONS_DIR": str(extension_store),
               "GUFO_MODELS_DIR": str(store), "HF_HOME": str(directory / "hf"),
               "HF_TOKEN": "", "HF_ENDPOINT": f"http://127.0.0.1:{server.server_address[1]}"}
        env.pop("HF_HUB_CACHE", None)

        def gufo(*args):
            result = subprocess.run([sys.argv[1], *args], env=env, text=True, capture_output=True)
            assert result.returncode == 0, (args, result.stdout, result.stderr)
            return result.stdout

        rows = {row["id"]: row for row in json.loads(gufo("model", "list", "--json"))["models"]}
        for model_id in ids:
            assert rows[model_id]["status"] == "available"
            assert rows[model_id]["downloadable"]
            missing = subprocess.run([sys.argv[1], "prompt", "--model", model_id],
                                     env=env, text=True, capture_output=True)
            assert f"gufo model install {model_id}" in missing.stderr
            output = gufo("model", "install", model_id, "--connections", "1")
            installed_components = {item["role"]: item for item in models[model_id]["components"]}
            for role in ("weights", "mtp", "mmproj"):
                for item in fixture_lock[model_id][role]["files"]:
                    assert (store / model_id / item["path"]).read_bytes() == files[item["path"]]
                if role != "weights":
                    path = store / model_id / fixture_lock[model_id][role]["files"][0]["path"]
                    assert f'{installed_components[role]["option"]} {path}' in output
            rows = {row["id"]: row for row in json.loads(gufo("model", "list", "--json"))["models"]}
            assert rows[model_id]["status"] == "installed"
            assert rows[model_id]["path"] == str(store / model_id / models[model_id]["components"][0]["entry"])
            assert "Usage:" in gufo("prompt", "--model", model_id, "--help")
            assert f"{model_id}: ok" in gufo("model", "verify", model_id)
finally:
    server.shutdown()
    server.server_close()
    thread.join()
print("Fork model catalog and fixture installs passed")
