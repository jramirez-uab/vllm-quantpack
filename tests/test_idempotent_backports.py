"""The Inkling2 Ampere backports replace vLLM modules only on the exact vLLM build they target, never twice, and
never a module already imported; the SenseNova MTP mapping rewrites only `sensenova_loop`."""
import os
import shutil
import sys
import types

import pytest

from vllm_quantpack.families.inkling2 import ampere


def _fake_vllm(tmp_path, source):
    """A vLLM tree whose replaced files are the targeted originals (`source="orig"`) or our backports ("ours")."""
    root = tmp_path / "vllm"
    for mod, sha in ampere.MODULES.items():
        if source == "orig" and sha is None:
            continue
        path = root.joinpath(*mod.split(".")[1:]).with_suffix(".py")
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ampere._ours(mod), path)
    if source == "orig":   # make each replaced file hash like the original
        for mod, sha in ampere.MODULES.items():
            if sha is not None:
                path = root.joinpath(*mod.split(".")[1:]).with_suffix(".py")
                path.write_bytes(b"orig " + mod.encode())
    return str(root)


def test_plan_installs_on_target_build(tmp_path, monkeypatch):
    root = _fake_vllm(tmp_path, "orig")
    shas = {m: (None if s is None else ampere._sha(os.path.join(root, *m.split(".")[1:]) + ".py"))
            for m, s in ampere.MODULES.items()}
    monkeypatch.setattr(ampere, "MODULES", shas)
    state, mapping = ampere.plan(root)
    assert state == "install"
    assert set(mapping) == {ampere._import_name(m) for m in shas}
    assert mapping["vllm.models.inkling2.nvidia.ops"].endswith("vllm.models.inkling2.nvidia.ops.__init__.py")


def test_plan_is_idempotent_when_vllm_has_the_backports(tmp_path):
    assert ampere.plan(_fake_vllm(tmp_path, "ours"))[0] == "applied"


def test_plan_leaves_another_vllm_build_alone(tmp_path):
    root = _fake_vllm(tmp_path, "orig")    # hashes differ from the recorded ones
    state, detail = ampere.plan(root)
    assert state == "mismatch" and "differs" in detail


def test_plan_refuses_modules_already_imported(tmp_path, monkeypatch):
    root = _fake_vllm(tmp_path, "orig")
    shas = {m: (None if s is None else ampere._sha(os.path.join(root, *m.split(".")[1:]) + ".py"))
            for m, s in ampere.MODULES.items()}
    monkeypatch.setattr(ampere, "MODULES", shas)
    monkeypatch.setitem(sys.modules, "vllm.models.inkling2.quant_config", types.ModuleType("x"))
    state, detail = ampere.plan(root)
    assert state == "loaded" and "quant_config" in detail


def test_sensenova_mtp_mapping_only_rewrites_sensenova(monkeypatch):
    spec = pytest.importorskip("vllm.config.speculative")
    from vllm_quantpack.families import sensenova

    seen = []

    class Fake:
        @staticmethod
        def hf_config_override(hf_config, *a, **k):
            seen.append(hf_config.model_type)
            return hf_config

    monkeypatch.setattr(spec, "SpeculativeConfig", Fake)
    assert sensenova._register_mtp(None) is True
    assert sensenova._register_mtp(None) is False          # second call: already wrapped
    for mt in ("sensenova_loop", "qwen3_5"):
        Fake.hf_config_override(types.SimpleNamespace(model_type=mt))
    assert seen == ["qwen3_5", "qwen3_5"]


def test_sensenova_mtp_mapping_is_picklable(monkeypatch):
    """vLLM keeps hf_config_override in the draft ModelConfig, which is pickled for the engine process."""
    import pickle

    spec = pytest.importorskip("vllm.config.speculative")
    from vllm_quantpack.families import sensenova

    class Fake:
        @staticmethod
        def hf_config_override(hf_config, *a, **k):
            return hf_config

    monkeypatch.setattr(spec, "SpeculativeConfig", Fake)
    monkeypatch.setattr(sensenova, "_ORIG_HF_CONFIG_OVERRIDE", None)
    sensenova._register_mtp(None)
    assert pickle.loads(pickle.dumps(Fake.hf_config_override)) is Fake.hf_config_override
