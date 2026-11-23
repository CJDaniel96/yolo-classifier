from __future__ import annotations

import argparse
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .config import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AOI YOLO OBB folder microservice")
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--check-config", action="store_true", help="Validate paths and model file, then exit")
    parser.add_argument("--once", action="store_true", help="Run one scan cycle, then exit (use stable_seconds=0 for fixtures)")
    parser.add_argument("--retry-failed", action="store_true", help="Requeue failed jobs at startup")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        config.prepare()
        level = logging.WARNING if config.runtime.quiet else (
            logging.DEBUG if config.runtime.verbose else logging.INFO)
        handler = RotatingFileHandler(config.paths.log_dir / "service.log",
                                      maxBytes=10 * 1024 * 1024, backupCount=10, encoding="utf-8")
        logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s %(message)s",
                            handlers=[logging.StreamHandler(), handler], force=True)
        if args.check_config:
            print("Configuration valid. Model file exists; model loading is checked at startup.")
            return 0
        from .inference import Predictor
        from .service import Service
        if args.once:
            service = Service(config, lambda: Predictor(config))
            try:
                if args.retry_failed:
                    service.state.retry_failed(service.clock())
                service.tick()
                counts = service.state.counts()
                print(counts)
                return 1 if any(counts.get(s, 0) for s in ("failed", "waiting", "publishing", "archiving")) else 0
            finally:
                service.close()
        from .api import create_app
        import uvicorn

        app = create_app(config, retry_failed=args.retry_failed)
        uvicorn.run(app, host=config.http.host, port=config.http.port,
                    workers=1, log_config=None)
        return 0
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        logging.getLogger(__name__).error("Service could not start: %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
