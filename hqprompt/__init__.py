"""hqprompt: HQ-code prompts for one-step OSDFace (Proposal 1)."""
import os as _os

# pyiqa (~/.cache/pyiqa), openai-clip (~/.cache/clip) and face_alignment (~/.cache/face_alignment)
# hard-code expanduser("~") for downloads. On SCC, jobs/env.sh sets HQP_HOME_CACHE so that these
# land in the write area instead of $HOME. Only this Python process sees the changed HOME.
# No-op when HQP_HOME_CACHE is unset (local tests).
if _os.environ.get("HQP_HOME_CACHE"):
    _os.makedirs(_os.environ["HQP_HOME_CACHE"], exist_ok=True)
    _os.environ["HOME"] = _os.environ["HQP_HOME_CACHE"]
