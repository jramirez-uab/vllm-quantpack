"""Ampere (SM80/SM86) backports of the vLLM nightly's Inkling2 CUDA modules.

The nightly's `vllm/models/inkling2` kernels assume SM90+: PDL (griddepcontrol) calls and hardware e4m3 conversion in
Triton. The modules in this directory are the same files with PDL compiled out below SM90 (`ops/pdl.py`, `use_pdl`) and
a software e4m3 path below SM89 (`ops/e4m3.py`); on SM90+ they behave like the originals. Each file is named after the
module it replaces.

`install()` puts them in front of vLLM's own modules with an import hook, and only when it is safe and needed:
  * every replaced vLLM file must be byte-identical to the one these backports were written against (SHA-256 below),
    and every added module must be absent from vLLM; otherwise vLLM's files are left alone (a newer nightly, or a vLLM
    that already carries Ampere support, keeps its own code);
  * if vLLM already contains these exact backports, nothing is replaced (idempotent);
  * none of the modules may be imported yet (a module already in sys.modules cannot be redirected).
All or nothing: a partial set would mix kernel generations.
"""
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

# module -> SHA-256 of the vLLM file it replaces (vLLM 0.34.0rc1.dev1136+gcb19fedd8), None for a module vLLM lacks.
MODULES = {
    "vllm.models.inkling.nvidia.sconv_swa_attn": "6b3c610e043ec00ce2a8ea65607eca9f15b1194d0951bda75c0494eac38db48a",
    "vllm.models.inkling2.nvidia.attention": "c85802fed4e579f9741d708e1b42d84e67e3f42f50e549d0a167461dc91bbc40",
    "vllm.models.inkling2.nvidia.model": "7665dddf51148cfd11910669d5ffd7a4879145ffaaa7325320eb83c79f6cf5b7",
    "vllm.models.inkling2.nvidia.ops.activation": "cfe5c9953da78906a4c4f09848d5fee15ef0260cf110438e37a3c0b341aa4a2f",
    "vllm.models.inkling2.nvidia.ops.attention": "2c18a3c6697a10dd0702c2563ef0eb12d7e3f6232d5369848ae9c13c004896c4",
    "vllm.models.inkling2.nvidia.ops.attention_sm90": "19c042fd75cfa352b14cfc8743b65e3a4a669bada415f19ebec3cd392338de9f",
    "vllm.models.inkling2.nvidia.ops.attn_res": "39f8392a06c76980ebf7b1981894451ecf74501fa5b43791c1e46cf185589981",
    "vllm.models.inkling2.nvidia.ops.conv_mix": "766940eb9c6fb4e5e1ba795f0bcbbe9666dc400a4287ff3e0995e4b5ef1a5cdb",
    "vllm.models.inkling2.nvidia.ops.e4m3": None,
    "vllm.models.inkling2.nvidia.ops.__init__": "1a297f5d55911a0e6b64337d1aaeb03c63a586410beb2117f80ebc03234edb8b",
    "vllm.models.inkling2.nvidia.ops.linear": "dd44b991c6766649507817e01393be65958c7bbf12c23841779cf6bcc5a4f2fe",
    "vllm.models.inkling2.nvidia.ops.moe": "10704d847d97ed429ac7a44ee36cc7594456e3bd077e7cbddf9c725af09f64f8",
    "vllm.models.inkling2.nvidia.ops.mxfp8": "58431169170558babc59b193eecfa94ddc57d4bfd89d02c5fab20698decc7799",
    "vllm.models.inkling2.nvidia.ops.nvfp4": "7830385a15c4b57f12aaf4b84103724de0b4a0ba3cf3bafef16bb1e42c9151bd",
    "vllm.models.inkling2.nvidia.ops.pdl": None,
    "vllm.models.inkling2.nvidia.ops.sconv": "982ece524f9efde091793f3128376f4690a35347cf6d12a7daf89c3d1b70e3a4",
    "vllm.models.inkling2.nvidia.vision": "fe7e95f9d7cb262b43b759c1a497c75cfd7fb81db69938ec637a1376b7b97d14",
    "vllm.models.inkling2.quant_config": "be02955071d5a2cf07bd61294b582f2d9584b2cc97601c73a56684a23a9c91e9",
}


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _ours(mod):
    return os.path.join(_HERE, mod + ".py")


def _vllm_path(vllm_root, mod):
    return os.path.join(vllm_root, *mod.split(".")[1:]) + ".py"


def _import_name(mod):
    return mod[: -len(".__init__")] if mod.endswith(".__init__") else mod


def plan(vllm_root):
    """(state, detail): 'install' with the module map, or 'applied' / 'mismatch' / 'loaded' with a reason."""
    mapping, applied = {}, True
    for mod, want in MODULES.items():
        theirs = _vllm_path(vllm_root, mod)
        exists = os.path.exists(theirs)
        if exists and _sha(theirs) == _sha(_ours(mod)):
            mapping[_import_name(mod)] = None          # vLLM already has our file
            continue
        applied = False
        if want is None and exists:
            return "mismatch", f"{mod} exists in vLLM"
        if want is not None and (not exists or _sha(theirs) != want):
            return "mismatch", f"{mod} differs from the vLLM build the backports target"
        mapping[_import_name(mod)] = _ours(mod)
    if applied:
        return "applied", "vLLM already carries the backports"
    loaded = sorted(n for n in mapping if n in sys.modules)
    if loaded:
        return "loaded", "already imported: " + ", ".join(loaded)
    return "install", {n: p for n, p in mapping.items() if p is not None}


class _Finder(importlib.abc.MetaPathFinder):
    def __init__(self, mapping, vllm_root):
        self.mapping, self.vllm_root = mapping, vllm_root

    def find_spec(self, name, path=None, target=None):
        file = self.mapping.get(name)
        if file is None:
            return None
        if os.path.basename(file).endswith(".__init__.py"):   # a package: its other submodules stay vLLM's
            pkg_dir = os.path.join(self.vllm_root, *name.split(".")[1:])
            return importlib.util.spec_from_file_location(name, file, submodule_search_locations=[pkg_dir])
        return importlib.util.spec_from_file_location(name, file)


def install(logger):
    """Install the import hook when the plan allows it; return True when the backports are active."""
    if any(isinstance(f, _Finder) for f in sys.meta_path):
        return True
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        return False
    vllm_root = list(spec.submodule_search_locations)[0]
    state, detail = plan(vllm_root)
    if state == "applied":
        return True
    if state != "install":
        logger.info("vllm-quantpack: Inkling2 Ampere backports not used (%s)", detail)
        return False
    sys.meta_path.insert(0, _Finder(detail, vllm_root))
    logger.info("vllm-quantpack: Inkling2 Ampere backports active (%d modules)", len(detail))
    return True
