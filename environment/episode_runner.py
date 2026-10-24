class EpisodeRunner:
    """Play complete Wiki Game episodes using an environment and a policy."""

    def __init__(self, env, policy):
        """Store the environment and policy to reuse across episodes.

        Args:
            env: An existing WikiGameEnv instance 
            policy: An existing WikiGamePolicy instance
        """
        self.env = env
        self.policy = policy

    def run_episode(self, start, target, deterministic=False):
        """Play one complete game and collect the route and policy decisions.

        Args:
            start: Starting article ID
            target: Target article ID
            deterministic: If False, sample actions for training rollouts.
                If True, choose actions greedily for evaluation.
        """
        self.env.reset(options={"start": start, "target": target})
        steps = []

        while True:
            observation = self.env.get_llm_observation()
            action, log_prob = self.policy.choose_action(
                observation, deterministic=deterministic
            )
            _, reward, terminated, truncated, info = self.env.step(action)

            steps.append({
                "observation": observation,
                "action": action,
                "log_prob": log_prob,
                "reward": reward,
            })

            if terminated or truncated:
                break

        return {
            "start": start,
            "target": target,
            "steps": steps,
            "terminated": terminated,
            "truncated": truncated,
            **info,
        }
