import json
from pathlib import Path

import random

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

    reward="success" (the default): success earns 1 plus 0.1 times the fraction
    of the click budget remaining; every other reward is zero.

    reward="progress" adds closest-approach credit. Let d be the shortest-path
    distance to the target and d_closest the smallest d over the pages visited
    so far. Whenever a move sets a new d_closest, it earns
    PROGRESS_WEIGHT * (improvement) / d_start, so a route's progress credit
    totals PROGRESS_WEIGHT * (d_start - d_closest) / d_start. A success reaches
    d = 0 and therefore earns the full PROGRESS_WEIGHT plus its success reward,
    so every success (at least 1.25) outranks every failure (below 0.25).
    Revisits, self-links and backtracks cannot earn credit, and there is no
    incentive to linger near the target. The agent never observes distances;
    only the reward uses them.
    """

    PROGRESS_WEIGHT = 0.25
    CONSISTENCY_SAMPLES = 2000

    def __init__(self, articles_path="data/processed/articles.json", max_clicks=20,
                 history_size=5, reward="success",
                 distance_matrix_path="data/raw/graph/shortest-path-distance-matrix.txt",
                 article_order_path="data/processed/article_order.json"):
        """Create the navigation environment from preprocessed article data."""
        super().__init__()
        if max_clicks < 1 or history_size < 1:
            raise ValueError("Click budget and history size must be positive.")
        if reward not in ("success", "progress"):
            raise ValueError('reward must be "success" or "progress".')
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
        self.reward_mode = reward
        self.distances = None
        if reward == "progress":
            self.distances = self._load_distances(Path(distance_matrix_path), Path(article_order_path))

    def _load_distances(self, matrix_path, order_path):
        """Load the shortest-path matrix, checking it is aligned with articles.json."""
        with order_path.open(encoding="utf-8") as file:
            order = json.load(file)
        if order != self.article_ids:
            raise ValueError("article_order.json does not match the article order of articles.json.")
        with matrix_path.open(encoding="utf-8-sig") as file:
            rows = [line.strip() for line in file if line.strip() and not line.lstrip().startswith("#")]
        size = len(self.article_ids)
        if len(rows) != size:
            raise ValueError("The distance matrix has the wrong number of rows.")
        for index, row in enumerate(rows):
            if len(row) != size or set(row) - set("0123456789_") or row[index] != "0":
                raise ValueError(f"Invalid distance matrix row: {index}")
        self._check_distances(rows)
        return rows

    def _check_distances(self, rows):
        """Check that sampled distances satisfy d(s, t) = 1 + min over links of d(n, t).

        This confirms the matrix describes this environment's link graph, not
        merely that the task distances match. All pairs are checked for small graphs.
        """
        size = len(self.article_ids)
        pairs = [(s, t) for s in range(size) for t in range(size) if s != t]
        if len(pairs) > self.CONSISTENCY_SAMPLES:
            rng = random.Random(0)
            pairs = [(rng.randrange(size), rng.randrange(size)) for _ in range(self.CONSISTENCY_SAMPLES)]
            pairs = [(s, t) for s, t in pairs if s != t]
        for s, t in pairs:
            links = [self.article_index[n] for n in self.articles[self.article_ids[s]]["outgoing"]]
            reachable = [int(rows[n][t]) for n in links if rows[n][t] != "_"]
            expected = 1 + min(reachable) if reachable else None
            actual = None if rows[s][t] == "_" else int(rows[s][t])
            if actual != expected:
                raise ValueError(
                    f"The distance matrix does not match the link graph: "
                    f"{self.article_ids[s]} -> {self.article_ids[t]} is {actual}, links imply {expected}.")

    def shortest_distance(self, source, target):
        """Shortest-path clicks from source to target, or None if unreachable."""
        if self.distances is None:
            raise ValueError('Distances are only loaded with reward="progress".')
        value = self.distances[self.article_index[source]][self.article_index[target]]
        return None if value == "_" else int(value)

    def route_return(self, path, target, clicks, success):
        """A whole route's return; route audits recompute every episode with this."""
        total = 1.0 + 0.1 * (self.max_clicks - clicks) / self.max_clicks if success else 0.0
        if self.reward_mode == "progress":
            initial = self.shortest_distance(path[0], target)
            reached = [d for d in (self.shortest_distance(page, target) for page in path) if d is not None]
            total += self.PROGRESS_WEIGHT * (initial - min(reached)) / initial
        return total

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
        if self.reward_mode == "progress":
            self.start_distance = self.closest_distance = self.shortest_distance(start, target)
            if self.start_distance is None:
                raise ValueError("The target is unreachable from the start article.")
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
        page = self.articles[self.current]
        has_context = "link_order" in page  # Added by data/link_context.py.
        positions = {link: i for i, link in enumerate(page.get("link_order", []))}
        if not self.done:
            for index, article_id in enumerate(page["outgoing"]):
                action = {"id": index, "title": self.articles[article_id]["title"]}
                if has_context:
                    action["position"] = positions.get(article_id)
                    action["context"] = page["link_context"].get(article_id)
                actions.append(action)
        observation = {
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
        if has_context:
            observation["lead"] = page.get("lead")
        return observation

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

        Success earns 1 + 0.1 * remaining_clicks / max_clicks. With
        reward="progress", a move that sets a new closest distance to the target
        also earns its share of progress credit; see the class docstring.
        Return observation, reward, terminated, truncated, and info.
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
            if self.reward_mode == "progress":
                distance = self.shortest_distance(self.current, self.target)
                if distance is not None and distance < self.closest_distance:
                    reward += self.PROGRESS_WEIGHT * (self.closest_distance - distance) / self.start_distance
                    self.closest_distance = distance
            if self.current == self.target:
                self.reason = "success"
            elif self.clicks >= self.max_clicks:
                self.reason = "click_budget"

        self.done = bool(self.reason)
        if self.reason == "success":
            reward += 1.0 + 0.1 * (self.max_clicks - self.clicks) / self.max_clicks
        self.episode_return += reward
        return self.get_observation(), reward, self.done, False, self.get_info()
