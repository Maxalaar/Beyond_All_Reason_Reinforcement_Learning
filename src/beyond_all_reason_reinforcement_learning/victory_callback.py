from collections import defaultdict, deque

from ray.rllib.algorithms.callbacks import DefaultCallbacks

# Episodes the rates are averaged over, on each environment runner.
WINDOW = 100


class VictoryCallback(DefaultCallbacks):
    """Victory and defeat rates of the agents' alliance over the last WINDOW episodes of each environment runner
    (env_runners/victory_rate, env_runners/defeat_rate; RLlib then averages the runners).

    The episode return does not give them once agents can die: a dead agent misses the victory reward. The latest
    information of the episode (beyond_all_reason_environment: allies_alive, enemies_alive) tells who is left; with
    both alive the episode reached its time limit, a draw.

    Also the victory rate against each number of opponents at the start of the episode
    (env_runners/victory_rate_against_<number>), over the last WINDOW episodes with that number: it shows where the
    agents stand when that number is drawn for each episode (TeamConfiguration.units given as (minimum, maximum)).

    The windows are kept here: RLlib's own window (log_value(window=...)) restarts at every training iteration, so a
    stopping criterion on it would judge ~40 episodes only.

    Stopping criteria go on env_runners/victory_rate_full_window: the victory rate once WINDOW episodes are seen, 0
    before. On victory_rate, a lucky start stopped stages after ~15 episodes (5 per environment runner) in
    2026-10-23's curriculum; Tune's stopping dictionary stops when any of its keys is reached, so it cannot also ask
    for a number of episodes.
    """

    def __init__(self):
        super().__init__()
        # (victory, defeat) of the last episodes of this environment runner: the callback lives as long as it.
        self._outcomes: deque[tuple[float, float]] = deque(maxlen=WINDOW)
        # Victories of the last episodes, by number of opponents at the start.
        self._victories_against: defaultdict[int, deque[float]] = defaultdict(lambda: deque(maxlen=WINDOW))

    def on_episode_end(self, *, episode, metrics_logger=None, **kwargs) -> None:
        informations = [
            agent_episode.get_infos(-1)
            for agent_episode in episode.agent_episodes.values()
            if len(agent_episode.infos) > 0
        ]
        if not informations or metrics_logger is None:
            return
        last = max(informations, key=lambda information: information.get("episode_frame", 0))
        allies_alive, enemies_alive = last.get("allies_alive", 0), last.get("enemies_alive", 0)
        victory = float(allies_alive > 0 and enemies_alive == 0)
        self._outcomes.append((victory, float(allies_alive == 0 and enemies_alive > 0)))
        # Every agent is there at the reset: the first information of any of them counts the opponents.
        opponents = next(iter(episode.agent_episodes.values())).get_infos(0).get("enemies_alive")
        if opponents is not None:
            victories = self._victories_against[opponents]
            victories.append(victory)
            metrics_logger.log_value(
                f"victory_rate_against_{opponents}", sum(victories) / len(victories), reduce="mean", window=1
            )

        # window=1: the iteration reports the rates after its last episode.
        metrics_logger.log_value(
            "victory_rate", sum(v for v, _ in self._outcomes) / len(self._outcomes), reduce="mean", window=1
        )
        metrics_logger.log_value(
            "defeat_rate", sum(d for _, d in self._outcomes) / len(self._outcomes), reduce="mean", window=1
        )
        metrics_logger.log_value("victory_rate_episodes", len(self._outcomes), reduce="mean", window=1)
        full_window = len(self._outcomes) == WINDOW
        metrics_logger.log_value(
            "victory_rate_full_window",
            sum(v for v, _ in self._outcomes) / WINDOW if full_window else 0.0,
            reduce="mean",
            window=1,
        )
