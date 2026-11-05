"""An evaluation of many episodes, as short rounds.

EnvRunner.sample(num_episodes=...) keeps every episode it finished, with all its observations, until it returns: an
environment runner evaluating 225 games of 38 agents grew to 17 GB (~65 kB per agent step, for the terrain alone), until
Ray's memory monitor killed workers at 95 % of the RAM (2026-11-05). A round frees them: evaluation_duration is its
number of episodes, and algorithm.evaluate() is called until enough have been played.
"""

# Episodes of a round, over all the evaluation env runners: ~25 MB each at most in the memory of an env runner.
ROUND_EPISODES = 20


def evaluate_in_rounds(algorithm, episodes: int, counted_metrics: tuple[str, ...]) -> dict[str, float]:
    """algorithm.evaluate() until ``episodes`` episodes are counted; the sums of ``counted_metrics``, which contain
    "counted_episodes", and of their "episode_len_mean" weighted by the episodes of each round."""
    totals = dict.fromkeys(counted_metrics, 0.0)
    episode_length = 0.0
    while totals["counted_episodes"] < episodes:
        results = algorithm.evaluate()["env_runners"]
        round_episodes = results["counted_episodes"]
        for metric in counted_metrics:
            totals[metric] += results[metric]
        episode_length += results["episode_len_mean"] * round_episodes
        print(f"round of {round_episodes:.0f} episodes: {totals['counted_episodes']:.0f}/{episodes}", flush=True)
    return {**totals, "episode_len_mean": episode_length / totals["counted_episodes"]}
