import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class WikiGameEnv(gym.Env):
    """Navigate Wikispeedia using local link IDs and a fixed click budget.

    Gym observations contain fixed-size numeric article IDs and padding.
    Use get_llm_observation() for readable titles and full current article text.
    Only the current article's text is exposed. Forward actions index its
    outgoing links; back_action returns to the previous navigation-stack entry.
    Both kinds of move consume one click. The full visited route is retained.
    The caller supplies start and target IDs through reset options.

    Intermediate steps and failed episodes receive zero reward. Success earns
    1 plus 0.1 times the fraction of the click budget remaining.
    """

    def __init__(self, articles_path="data/processed/articles.json", max_clicks=20,
                 history_size=5):
        """Create the navigation environment from preprocessed article data."""
        super().__init__()
        if max_clicks < 1 or history_size < 1:
            raise ValueError("Click budget and history size must be positive.")
        with Path(articles_path).open(encoding="utf-8") as file:
            self.articles = json.load(file)
        if not self.articles:
            raise ValueError("The article dataset is empty.")

        self.article_ids = list(self.articles)
        self.article_index = {}
        for index, article_id in enumerate(self.article_ids):
            self.article_index[article_id] = index
        for article in self.articles.values():
            for target in article["outgoing"]:
                if target not in self.articles:
                    raise ValueError(f"Unknown link destination: {target}")

        self.max_clicks = max_clicks
        self.history_size = history_size
        self.max_links = max(len(a["outgoing"]) for a in self.articles.values())
        if self.max_links == 0:
            raise ValueError("The graph has no outgoing links.")
        self.back_action = self.max_links
        self.action_space = spaces.Discrete(self.max_links + 1)
        last_index = len(self.article_ids) - 1
        self.observation_space = spaces.Dict({
            "current": spaces.Discrete(len(self.article_ids)),
            "target": spaces.Discrete(len(self.article_ids)),
            "remaining_clicks": spaces.Discrete(max_clicks + 1),
            "links": spaces.Box(-1, last_index, (self.max_links,), dtype=np.int32),
            "history": spaces.Box(-1, last_index, (history_size,), dtype=np.int32),
            "navigation_stack": spaces.Box(-1, last_index, (max_clicks + 1,), dtype=np.int32),
        })
        self.current = None
        self.target = None
        self.done = True

    def validate_pair(self, start, target):
        """Check explicit article IDs; reachability must be checked when making task splits."""
        if start not in self.articles or target not in self.articles:
            raise ValueError("Start and target must be IDs from articles.json.")
        if start == target:
            raise ValueError("Start and target must be different articles.")
        if not self.articles[start]["outgoing"]:
            raise ValueError("The starting article has no outgoing links.")

    def reset(self, *, seed=None, options=None):
        """Start an episode using caller-supplied start and target article IDs."""
        super().reset(seed=seed)
        if seed is not None:
            self.action_space.seed(seed)
        options = options or {}
        if "start" not in options or "target" not in options:
            raise ValueError("Provide both start and target in reset options.")
        start, target = options["start"], options["target"]
        self.validate_pair(start, target)

        self.current = start
        self.target = target
        self.history = [start]
        self.navigation_stack = [start]
        self.clicks = 0
        self.episode_return = 0.0
        self.reason = ""
        self.done = False
        return self.get_observation(), self.get_info()

    def action_masks(self):
        """Allow current links and Back when there is a previous stack entry."""
        mask = np.zeros(self.action_space.n, dtype=bool)
        if not self.done:
            mask[:len(self.articles[self.current]["outgoing"])] = True
            mask[self.back_action] = len(self.navigation_stack) > 1
        return mask

    def get_observation(self):
        """Build a fixed-size numeric observation, padding links and history with -1."""
        if self.current is None:
            raise gym.error.ResetNeeded("Call reset() before requesting an observation.")
        links = np.full(self.max_links, -1, dtype=np.int32)
        for index, article_id in enumerate(self.articles[self.current]["outgoing"]):
            links[index] = self.article_index[article_id]
        history = np.full(self.history_size, -1, dtype=np.int32)
        for index, article_id in enumerate(self.history[-self.history_size:]):
            history[index] = self.article_index[article_id]
        navigation_stack = np.full(self.max_clicks + 1, -1, dtype=np.int32)
        for index, article_id in enumerate(self.navigation_stack):
            navigation_stack[index] = self.article_index[article_id]
        return {
            "current": self.article_index[self.current],
            "target": self.article_index[self.target],
            "remaining_clicks": self.max_clicks - self.clicks,
            "links": links,
            "history": history,
            "navigation_stack": navigation_stack,
        }

    def get_llm_observation(self):
        """Return visible content, full route, navigation stack, and Back availability."""
        if self.current is None:
            raise gym.error.ResetNeeded("Call reset() before requesting an observation.")
        actions = []
        back_available = not self.done and len(self.navigation_stack) > 1
        if not self.done:
            for index, article_id in enumerate(self.articles[self.current]["outgoing"]):
                actions.append({"id": index, "title": self.articles[article_id]["title"]})
        return {
            "current_title": self.articles[self.current]["title"],
            "target_title": self.articles[self.target]["title"],
            "text": self.articles[self.current]["text"],
            "actions": actions,
            "history": [self.articles[a]["title"] for a in self.history],
            "navigation_stack": [self.articles[a]["title"] for a in self.navigation_stack],
            "back_available": back_available,
            "back_destination": (
                self.articles[self.navigation_stack[-2]]["title"] if back_available else None
            ),
            "back_action_id": self.back_action if back_available else None,
            "remaining_clicks": self.max_clicks - self.clicks,
        }

    def get_info(self):
        """Return episode metrics and the visited route for debugging and evaluation."""
        if self.current is None:
            raise gym.error.ResetNeeded("Call reset() before requesting episode info.")
        return {
            "success": self.current == self.target,
            "clicks": self.clicks,
            "episode_return": self.episode_return,
            "reason": self.reason,
            "path": list(self.history),
            "navigation_stack": list(self.navigation_stack),
        }

    def step(self, action):
        """Follow a link or go back, charging one click and retaining the full route.

        Success earns 1 + 0.1 * remaining_clicks / max_clicks. All other
        rewards are zero. Return observation, reward, terminated, truncated,
        and info.
        """
        if self.done:
            raise gym.error.ResetNeeded("Call reset() before starting or continuing an episode.")
        if not self.action_space.contains(action):
            raise ValueError(f"Action is outside the action space: {action}")
        action = int(action)
        links = self.articles[self.current]["outgoing"]
        self.clicks += 1
        reward = 0.0

        if not self.action_masks()[action]:
            self.reason = "invalid_action"
        else:
            if action == self.back_action:
                self.navigation_stack.pop()
                self.current = self.navigation_stack[-1]
            else:
                self.current = links[action]
                self.navigation_stack.append(self.current)
            self.history.append(self.current)
            if self.current == self.target:
                self.reason = "success"
            elif self.clicks >= self.max_clicks:
                self.reason = "click_budget"

        self.done = bool(self.reason)
        if self.reason == "success":
            remaining_fraction = (self.max_clicks - self.clicks) / self.max_clicks
            reward = 1.0 + 0.1 * remaining_fraction
        self.episode_return += reward
        return self.get_observation(), reward, self.done, False, self.get_info()
