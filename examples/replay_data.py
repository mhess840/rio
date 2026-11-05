import multiprocessing as mp
from dataclasses import asdict, dataclass, field

import tyro
from loguru import logger
from rio_hw import time
from rio_hw.middleware import ServerManager

from rio.envs.factory import make_node


def replay_loop(args, loader, visualizer):
    try:
        logger.info("Starting replay loop")

        # Get trajectory info
        initial_state = loader.get_state()
        num_timesteps = initial_state["num_timesteps"]
        features = initial_state["features"]

        logger.info(f"Loaded trajectory: {num_timesteps} timesteps")
        logger.info(f"Features: {features}")

        # Reset to beginning
        loader.reset()
        time.sleep(0.1)  # Give time for reset to complete

        # Set up robot model
        # if args.urdf_path is not None:
        #     visualizer.set_robot_model("robot", robot_description=args.urdf_path, variant=None)

        freq = args.freq
        dt = 1.0 / freq
        t_start = time.now()

        logger.info(f"Replaying at {freq} Hz. Press Ctrl+C to stop")

        for timestep in range(num_timesteps):
            t_cycle_end = t_start + (timestep + 1) * dt
            step = loader.get_step()
            visualizer.log_env_state("demo", step)
            loader.step()

            if timestep % freq == 0:
                logger.info(f"Timestep: {timestep}/{num_timesteps} ({100 * timestep / num_timesteps:.1f}%)")

            time.precise_wait(t_cycle_end)
        logger.info("Replay complete!")

    except KeyboardInterrupt:
        logger.info("Replay interrupted by user")


def main(args):
    logger.info(f"Loading trajectory from: {args.loader_cfg.path}")

    loader_server, loader_client = make_node(args.mw, "data", "Loader", asdict(args.loader_cfg), package="rio")
    visualizer_server, visualizer_client = make_node(
        args.mw, "visualization", "Rerun", asdict(args.visualizer_cfg), package="rio"
    )

    servers = {
        "loader": loader_server,
        "visualizer": visualizer_server,
    }

    with ServerManager(args.mw, list(servers.values())):
        with loader_client() as loader, visualizer_client() as visualizer:
            try:
                replay_loop(args, loader, visualizer)
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":
    from examples import get_station_cfg

    StationCfg = get_station_cfg()

    @dataclass
    class Args(StationCfg):
        @dataclass
        class LoaderConfig:
            path: str = "/data/rollouts/xarm/place_coke_can/traj_0018.vla"
            auto_play: bool = False  # Manual control via step() calls
            loop: bool = False
            verbose: bool = True
            embodiment_type: str = "SINGLE_ARM"

        @dataclass
        class VisualizerConfig:
            app_id: str = "replay"
            spawn: bool = True

        loader_cfg: LoaderConfig = field(default_factory=lambda: Args.LoaderConfig())
        visualizer_cfg: VisualizerConfig = field(default_factory=lambda: Args.VisualizerConfig())

        mw: str = "Thread"
        mp_method: str | None = "spawn"
        freq: int = 50
        # Playback does not record and does not drive to a home.
        recorder: str | None = None

        def __post_init__(self) -> None:
            self.bind_home = False
            StationCfg.__post_init__(self)

    args = tyro.cli(Args)
    print(args)
    mp.set_start_method(args.mp_method, force=True)
    main(args)
