import random

from environment.environment import WikiGameEnv
import gensim.downloader as api
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
import numpy as np
import re
import urllib.parse
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from collections import deque
import torch
import torch.nn as nn
import copy
import json
import gymnasium as gym


#
# Tokenizing words
#

def tokenize(s):
    return re.findall(r"[A-Za-z0-9]+", s)


def avg_vec(tokens, kv):
    vs = []
    for w in tokens:
        for cand in (w, w.lower(), w.capitalize()):
            if cand in kv:
                vs.append(kv[cand])
                break
    return np.mean(vs, axis=0) if vs else np.zeros(kv.vector_size, dtype=np.float32)


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v


def center_and_normalize(X):
    X = X - X.mean(0, keepdims=True)
    return (X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)).astype(np.float32)


def clean_title(title):
    return urllib.parse.unquote(title).replace("_", " ")


def build_word2vec_embeddings(env, kv, text_weight=0.5, text_words=200, center=True):
    # This gives 2 variables:
    # 1. embedding, which is just the title of the article
    # 2. embedding_context, which also adds the first few words in the article.
    N, d = len(env.article_ids), kv.vector_size
    embedding = np.zeros((N, d), dtype=np.float32)
    embedding_context = np.zeros((N, d), dtype=np.float32)
    for i, aid in enumerate(env.article_ids):
        art = env.articles[aid]
        tv = unit(avg_vec(tokenize(clean_title(art["title"])), kv))
        embedding[i] = tv
        xv = unit(avg_vec(tokenize(art["text"])[:text_words], kv)) if text_weight > 0 else 0.0
        embedding_context[i] = tv + text_weight * xv
    if center:
        embedding, embedding_context = center_and_normalize(embedding), center_and_normalize(embedding_context)
    else:
        embedding, embedding_context = (np.stack([unit(v) for v in m]).astype(np.float32) for m in (embedding, embedding_context))
    return embedding, embedding_context


# ----------------------------------------------------------------------------
# Observation flattening (shared by training wrapper, policy, and evaluation)
# layout: [current, target, remaining_clicks, links (K, pad -1), history (H, pad -1)]
# ----------------------------------------------------------------------------
def flatten_obs(obs):
    return np.concatenate([
        [obs["current"], obs["target"], obs["remaining_clicks"]], obs["links"], obs["history"],
    ]).astype(np.float32)


class FlatObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        base = env.unwrapped
        n = len(base.article_ids)
        size = 3 + base.max_links + base.history_size
        self.observation_space = gym.spaces.Box(-1.0, float(n), shape=(size,), dtype=np.float32)

    def observation(self, obs):
        return flatten_obs(obs)

    def action_masks(self):
        return self.unwrapped.action_masks()


# This class is able to hold the results of the tasks that the agents have done.
# It also has an attribute max_dist that causes the agents to only perform
# tasks of a limited size first before moving up.
class Task_Manager:
    def __init__(self, max_clicks, start=1, thresh=0.75, window=100):
        self.max_dist, self.max_clicks, self.thresh, self.window = start, max_clicks, thresh, window
        self.results = deque(maxlen=window)

    def report(self, success):
        self.results.append(float(success))
        if (len(self.results) == self.window and np.mean(self.results) >= self.thresh
                and self.max_dist < self.max_clicks):
            self.max_dist += 1
            self.results.clear()

    @property
    def success_rate(self):
        return float(np.mean(self.results)) if self.results else 0.0


class Task_Sampler(gym.Wrapper):
    # Picks a random task based on the difficulty.

    def __init__(self, env, tasks, task_manager):
        super().__init__(env)
        self.tasks = tasks
        self.task_manager = task_manager
        self.rng = np.random.default_rng()

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        # Randomly pick a task. It can only pick tasks with a maximum length, determined by task_manager.max_dist
        task_pool = []
        for distance in range(self.task_manager.max_dist):
            if str(distance + 1) in self.tasks:
                task_pool.extend(self.tasks[str(distance + 1)])
        task = random.choice(task_pool)
        return self.env.reset(seed=seed, options=task)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            info["is_success"] = bool(info["success"])
            self.task_manager.report(info["success"])
        return obs, reward, terminated, truncated, info


class SimilarityShaping(gym.Wrapper):
    # The main purpose of the embeddings. It gives a benefit to r based on word2vec embedding similarity.

    def __init__(self, env, embeddings, gamma, coef):
        super().__init__(env)
        self.embeddings, self.gamma, self.coef, self.phi = embeddings, gamma, coef, 0.0

    def set_phi(self, observation):
        return float(self.embeddings[observation["current"]] @ self.embeddings[observation["target"]])

    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        self.phi = self.set_phi(observation)
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        phi_next = 0.0 if (terminated or truncated) else self.set_phi(observation)
        reward += self.coef * (self.gamma * phi_next - self.phi)
        self.phi = phi_next
        return observation, reward, terminated, truncated, info


# ----------------------------------------------------------------------------
# Policy: per-link scorer inside MaskablePPO
# ----------------------------------------------------------------------------
class LinkScorer(nn.Module):
    """
    Plays the role of SB3's `mlp_extractor`:
        forward_actor(features)  -> [B, K] one logit per link slot
        forward_critic(features) -> [B, h] latent for the value head
    `features` is the flat observation from flatten_obs (FlattenExtractor passes it through).
    Link indices come straight from the observation, so no graph tables are needed here.
    """

    def __init__(self, tables, history_size, max_clicks, obs_dim, hidden_width=256):
        super().__init__()
        embeddings = torch.tensor(tables["embeddings"])
        embeddings_dim = embeddings.shape[1]
        self.history_size, self.max_clicks = history_size, float(max_clicks)
        self.num_links = obs_dim - 3 - history_size
        self.register_buffer("embeddings", embeddings)
        self.register_buffer("embeddings_context", torch.tensor(tables["embeddings_context"]))

        self.latent_dim_pi, self.latent_dim_vf = self.num_links, hidden_width

        # These layers all get combined into 1 big vector.
        # Word embeddings for current page
        self.fc_current_page = nn.Linear(embeddings_dim, hidden_width)
        # Word embeddings for target page
        self.fc_target_page = nn.Linear(embeddings_dim, hidden_width, bias=False)
        # Word embeddings for the links
        self.fc_link = nn.Linear(embeddings_dim, hidden_width, bias=False)
        # Word embeddings for links multiplied by target.
        self.fc_link_times_target = nn.Linear(embeddings_dim, hidden_width, bias=False)
        # 3 values:
        # first value represents cosine similarity between link and target
        # second value is 1 if the link was in the recent history of the agent, or 0 otherwise
        # third value is 1 if the link leads directly to the target.
        self.fc_link_scalars = nn.Linear(3, hidden_width, bias=False)
        # 2 values:
        # first value is the log of how many links are in the page, to see how "hub-like" the page is.
        # second value is the amount of click budget remaining.
        self.fc_context_scalars = nn.Linear(2, hidden_width, bias=False)

        # Combines these values into one
        self.head = nn.Sequential(nn.ReLU(), nn.Linear(hidden_width, hidden_width), nn.ReLU(), nn.Linear(hidden_width, 1))
        # The critic's MLP
        self.vf = nn.Sequential(nn.Linear(2 * embeddings_dim + 3, hidden_width), nn.ReLU(), nn.Linear(hidden_width, hidden_width), nn.ReLU())
        # near-uniform initial policy
        nn.init.orthogonal_(self.head[-1].weight, gain=0.01)
        nn.init.zeros_(self.head[-1].bias)

    # Takes an observation and turns it into variables
    def _parse(self, observation):
        current_page, target_page = observation[:, 0].long(), observation[:, 1].long()
        remaining_clicks = observation[:, 2] / self.max_clicks
        links = observation[:, 3:3 + self.num_links].long()
        history = observation[:, 3 + self.num_links:3 + self.num_links + self.history_size].long()
        mask = links >= 0
        links_clamped = links.clamp(min=0)
        return current_page, target_page, remaining_clicks, links_clamped, mask, history

    def forward_actor(self, f):
        current_page, target_page, remaining_clicks, links_clamped, mask, history = self._parse(f)
        # We get emb_ctx of the current page since we're allowed to read it since... we're on the page
        current_page_embeddings = self.embeddings_context[current_page]
        # But we can only see the target page's title
        target_page_embeddings = self.embeddings[target_page]
        # We can also only see title of links.
        link_embeddings = self.embeddings[links_clamped]
        # This is just the target page embeddings again, but with an extra dimension to be multiplied with link embeddings
        target_page_embeddings_multiplied = target_page_embeddings[:, None, :]

        link_scalars = torch.stack([
            # represents cosine similarity between link and target
            (link_embeddings * target_page_embeddings_multiplied).sum(-1),
            # 1 if the link was in the recent history of the agent, or 0 otherwise
            ((links_clamped[:, :, None] == history[:, None, :]).any(-1) & mask).float(),
            # 1 if the link leads directly to the target.
            (links_clamped == target_page[:, None]).float(),
        ], dim=-1)
        page_scalars = torch.stack([
            # number of links in the page
            torch.log1p(mask.sum(-1).float()),
            # click budget
            remaining_clicks], dim=-1)

        x = ((self.fc_current_page(current_page_embeddings) + self.fc_target_page(target_page_embeddings) + self.fc_context_scalars(page_scalars))[:, None, :]
             + self.fc_link(link_embeddings) + self.fc_link_times_target(link_embeddings * target_page_embeddings_multiplied) + self.fc_link_scalars(link_scalars))
        return self.head(x).squeeze(-1)

    def forward_critic(self, f):
        current_page, target_page, remaining_clicks, links_clamped, mask, history = self._parse(f)
        current_page_embeddings = self.embeddings_context[current_page]
        target_page_embeddings = self.embeddings[target_page]
        cos_similarity = (self.embeddings[current_page] * target_page_embeddings).sum(-1, keepdim=True)
        page_scalars = torch.stack([torch.log1p(mask.sum(-1).float()), remaining_clicks], dim=-1)
        return self.vf(torch.cat([current_page_embeddings, target_page_embeddings, cos_similarity, page_scalars], dim=-1))

    def forward(self, observation):
        return self.forward_actor(observation), self.forward_critic(observation)


class WikiPolicy(MaskableActorCriticPolicy):
    def __init__(self, *args, tables, history_size, max_clicks, hidden=256, **kwargs):
        # must be set before super().__init__, which calls _build()
        self.tables, self.history_size, self.max_clicks, self.hidden = tables, history_size, max_clicks, hidden
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self):
        self.mlp_extractor = LinkScorer(self.tables, self.history_size, self.max_clicks,
                                        self.observation_space.shape[0], self.hidden)

    def _build(self, lr_schedule):
        self._build_mlp_extractor()
        self.action_net = nn.Identity()  # latent_pi already is one logit per link slot
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        nn.init.orthogonal_(self.value_net.weight, gain=1.0)
        nn.init.zeros_(self.value_net.bias)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
class Task_Manager_Log(BaseCallback):
    def __init__(self, task_manager):
        super().__init__()
        self.task_manager = task_manager

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        self.logger.record("Task_Manager/max_dist", self.task_manager.max_dist)
        self.logger.record("Task_Manager/recent_success", self.task_manager.success_rate)


def train(env, train_targets, embeddings, embeddings_context, total_timesteps=1_000_000, n_envs=16,
          gamma=0.99, shape_coef=0.5, seed=0, device="auto", save_path="wiki_ppo_agent"):
    task_manager = Task_Manager(max_clicks=env.max_clicks)

    def make_env():
        # WikiGameEnv keeps all per-episode state in instance attributes that reset() reassigns, and
        # the (large) article data is read-only, so a shallow copy gives each worker its own state
        # without re-reading articles.json n_envs times.
        e = Task_Sampler(copy.copy(env), train_targets, task_manager)
        e = SimilarityShaping(e, embeddings, gamma, shape_coef)
        return Monitor(FlatObsWrapper(e))

    # DummyVecEnv keeps everything in one process, so the Task_Manager object is shared between them.
    venv = DummyVecEnv([make_env for _ in range(n_envs)])

    model = MaskablePPO(
        WikiPolicy, venv,
        policy_kwargs=dict(tables=dict(embeddings=embeddings, embeddings_context=embeddings_context), history_size=env.history_size,
                           max_clicks=env.max_clicks, hidden=256),
        n_steps=128,            # per env -> 128 * n_envs samples per update
        batch_size=256,         # must divide n_steps * n_envs
        n_epochs=4,
        learning_rate=3e-4,
        gamma=gamma,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=seed,
        device=device,
        verbose=1,
    )
    model.learn(total_timesteps=total_timesteps, callback=Task_Manager_Log(task_manager))
    model.save(save_path)
    return model

# ----------------------------------------------------------------------------
# Evaluation on held-out targets + baselines (all act on the raw dict observation)
# ----------------------------------------------------------------------------
def ppo_policy(model):
    def f(obs):
        mask = obs["links"] >= 0
        action, _ = model.predict(flatten_obs(obs)[None], action_masks=mask[None], deterministic=True)
        return int(action[0])
    return f


def greedy_cosine_policy(emb):
    def f(obs):
        links = obs["links"]
        scores = emb[np.clip(links, 0, None)] @ emb[obs["target"]]
        scores = np.where(np.isin(links, obs["history"]), scores - 10.0, scores)  # avoid recent pages
        scores[links < 0] = -np.inf
        return int(scores.argmax())
    return f


def random_policy(rng):
    def f(obs):
        links = obs["links"]
        ok = np.where((links >= 0) & ~np.isin(links, obs["history"]))[0]
        if len(ok) == 0:
            ok = np.where(links >= 0)[0]
        return int(rng.choice(ok))
    return f


def run_episode(env, policy, start_id, target_id):
    obs, info = env.reset(options={"start": start_id, "target": target_id})
    done = False
    while not done:
        obs, _, done, _, info = env.step(policy(obs))
    return info["success"], info["clicks"]


def evaluate(env, policies, tasks):
    for name, policy in policies.items():
        by_distance = {}
        for task in tasks:
            is_success, clicks = run_episode(env, policy, task["start"], task["target"])
            by_distance.setdefault(task["shortest_distance"], []).append((is_success, clicks))
        overall = np.mean([is_success for v in by_distance.values() for is_success, _ in v])
        print(f"\n{name}: overall success {overall:.3f}")
        for distance in sorted(by_distance):
            result = by_distance[distance]
            lens = [clicks for ok, clicks in result if ok]
            print(f"  optimal={distance}: n={len(result):3d} success={np.mean([ok for ok, _ in result]):.2f} "
                  f"avg_clicks(success)={np.mean(lens) if lens else float('nan'):.2f}")


if __name__ == "__main__":
    env = WikiGameEnv()
    model = PPO(ActorCriticPolicy, env, verbose=0)
    trains = json.load(open("data/processed/train.json"))
    tests = json.load(open("data/processed/test.json"))

    trains_by_distance = {}
    tests_by_distance = {}

    for task in trains:
        if str(task["shortest_distance"]) in trains_by_distance:
            trains_by_distance[str(task["shortest_distance"])].append(task)
        else:
            trains_by_distance[str(task["shortest_distance"])] = [task]
    for task in tests:
        if str(task["shortest_distance"]) in tests_by_distance:
            tests_by_distance[str(task["shortest_distance"])].append(task)
        else:
            tests_by_distance[str(task["shortest_distance"])] = [task]

    w2v_model = api.load('word2vec-google-news-300')
    embeddings, embedding_context = build_word2vec_embeddings(env, w2v_model, text_weight=0.5)

    model = train(env, trains_by_distance, embeddings, embedding_context, total_timesteps=100000)

    evaluate(env, {
        "MaskablePPO (greedy)": ppo_policy(model),
        "Greedy cosine": greedy_cosine_policy(embeddings),
        "Random (avoids recent pages)": random_policy(np.random.default_rng(2)),
    }, tests)

