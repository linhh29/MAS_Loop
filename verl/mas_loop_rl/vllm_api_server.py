"""MAS-Loop-local vLLM OpenAI server entrypoint with a tunable TCP backlog.

vLLM 0.10 creates the listening socket itself and then gives it to Uvicorn.
Uvicorn's default accept backlog is 2048, which is lower than the Editor and
sub-agent admission used by the large DeepScaleR trials.  A burst above that
limit can fail during TCP connect before the request reaches vLLM.  This module
keeps the upstream parser/server unchanged and only supplies ``backlog`` to
``run_server``.  It lives in MAS_Loop's own PYTHONPATH and does not modify the
container's vLLM installation or any sibling project.
"""

from __future__ import annotations

import os

import uvloop
from vllm.entrypoints.openai.api_server import (
    FlexibleArgumentParser,
    make_arg_parser,
    run_server,
    validate_parsed_serve_args,
)


def main() -> None:
    parser = FlexibleArgumentParser(
        description="MAS-Loop vLLM OpenAI-compatible server with configurable HTTP backlog."
    )
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    # vLLM releases differ on where this validator is exported. Calling the
    # API-server export preserves the exact behavior of its official __main__.
    validate_parsed_serve_args(args)
    backlog = int(os.getenv("MAS_LOOP_VLLM_HTTP_BACKLOG", "16384"))
    if backlog < 128:
        raise ValueError(f"MAS_LOOP_VLLM_HTTP_BACKLOG must be >=128, got {backlog}")
    print(f"[MAS-Loop] vLLM HTTP accept backlog={backlog}", flush=True)
    uvloop.run(run_server(args, backlog=backlog))


if __name__ == "__main__":
    main()
