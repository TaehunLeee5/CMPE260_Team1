import json
import unicodedata

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from policy.prompt import (LINK_CONTEXT_SYSTEM_PROMPT, LINK_CONTEXT_TEMPLATE, OBSERVATION_TEMPLATE,
                           SYSTEM_PROMPT, TOOLS)


PROMPT_STYLES = ("article_text", "link_context")


class WikiGamePolicy:
    """Generate one legal Qwen tool call and retain its token probabilities."""

    PROMPT_STYLES = PROMPT_STYLES

    def __init__(self, model_name="Qwen/Qwen3-1.7B", max_input_tokens=4096,
                 *, model=None, tokenizer=None, adapter_path=None, prompt_style="article_text"):
        """Load the actor or injected model, optionally adding a frozen local adapter.

        prompt_style="article_text" shows the current article's text (truncated to
        fit); "link_context" shows its opening paragraph and each link with the
        sentence it appears in, which needs articles processed by data/link_context.py.
        """
        if max_input_tokens is not None and max_input_tokens < 1:
            raise ValueError("The input token budget must be positive.")
        if prompt_style not in self.PROMPT_STYLES:
            raise ValueError(f"prompt_style must be one of {self.PROMPT_STYLES}.")
        self.prompt_style = prompt_style
        if prompt_style == "link_context":
            self.system_prompt, self.observation_template = LINK_CONTEXT_SYSTEM_PROMPT, LINK_CONTEXT_TEMPLATE
        else:
            self.system_prompt, self.observation_template = SYSTEM_PROMPT, OBSERVATION_TEMPLATE
        self.max_input_tokens = max_input_tokens
        # When True, training scores a sampled call by replaying the exact cached
        # computation it was sampled with (needs activation checkpointing off).
        self.replay_scoring = False
        # Greedy decisions use the KV-cached path by default; False restores the
        # original uncached per-prefix decoder (used for the first baselines).
        self.cached_greedy = True
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = "auto"
        if device == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.eos_token_id is None:
            raise ValueError("The tokenizer needs an end-of-sequence token.")
        self.model = model if model is not None else AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype,
        )
        if model is None:
            self.model.to(device)
        if adapter_path is not None:
            self.model = PeftModel.from_pretrained(
                self.model, adapter_path, is_trainable=False, local_files_only=True,
            )
        self.model.eval()

    def enable_lora(self, rank=8, alpha=16):
        """Freeze the actor and attach trainable, initially neutral LoRA adapters.

        Keep dropout disabled for rollout/training probability consistency.
        Use PEFT's save_pretrained/from_pretrained methods for adapter checkpoints.
        """
        if isinstance(self.model, PeftModel):
            raise ValueError("The policy already has adapters attached.")
        self.model = get_peft_model(self.model, LoraConfig(
            task_type="CAUSAL_LM", r=rank, lora_alpha=alpha,
            target_modules="all-linear", lora_dropout=0.0, bias="none",
            init_lora_weights=True,
        ))
        self.model.eval()

    @staticmethod
    def _title_sort_key(title):
        """Alphabetical by display title, ignoring case and accents (Åland sorts as Aland)."""
        plain = "".join(
            char for char in unicodedata.normalize("NFKD", title) if not unicodedata.combining(char)
        )
        return plain.casefold(), title

    def build_prompt(self, observation):
        """Format the visible state; internal environment action IDs are never shown.

        Links are listed alphabetically by title. Links to the current page or to
        articles already on the visited route are tagged; the tags are display
        only, since constrained decoding lets the model generate just the title.
        """
        values = observation.copy()
        visited = set(observation["history"])
        lines = []
        if self.prompt_style == "link_context":
            if "lead" not in observation:
                raise ValueError("link_context prompts need articles processed by data/link_context.py.")
            actions = self._page_order(observation["actions"])
            shown = observation.get("context_limit", len(actions))
            with_context = 0
        else:
            actions = sorted(observation["actions"], key=lambda a: self._title_sort_key(a["title"]))
        for action in actions:
            title = action["title"]
            if title == observation["current_title"]:
                title += " (current page)"
            elif title in visited:
                title += " (visited)"
            if self.prompt_style == "link_context" and action.get("context"):
                if with_context < shown:
                    title += f': "{action["context"]}"'
                with_context += 1
            lines.append(f"- {title}")
        values["links"] = "\n".join(lines) or "None"
        if self.prompt_style == "link_context":
            values["lead"] = observation["lead"] or "(not available)"
        values["history"] = " -> ".join(observation["history"])
        values["navigation_stack"] = " -> ".join(observation["navigation_stack"])
        values["back_destination"] = (
            observation["back_destination"] if observation["back_available"] else "unavailable"
        )
        return self.observation_template.format(**values)

    def _page_order(self, actions):
        """Links in the order they appear on the page; links not shown on it go last, alphabetically."""
        return sorted(actions, key=lambda a: (a.get("position") is None, a.get("position") or 0,
                                              self._title_sort_key(a["title"])))

    def _tool_calls(self, observation):
        """Map each legal internal environment action to its native tool call.

        The model names a link by its title; the environment action ID stays
        internal and is recovered from the title, so titles must be unique.
        """
        calls, titles = {}, set()
        for action in observation["actions"]:
            ident, title = action["id"], action["title"]
            if type(ident) is not int or ident < 0 or ident in calls:
                raise ValueError("Link IDs must be unique nonnegative integers.")
            if not isinstance(title, str) or not title.strip() or title in titles:
                raise ValueError(f"Link titles must be unique nonempty strings: {title!r}")
            titles.add(title)
            calls[ident] = {"name": "follow_link", "arguments": {"title": title}}
        if observation["back_available"]:
            ident = observation["back_action_id"]
            if type(ident) is not int or ident < 0 or ident in calls:
                raise ValueError("Back needs a distinct nonnegative action ID.")
            calls[ident] = {"name": "go_back", "arguments": {}}
        if not calls:
            raise ValueError("The observation has no available actions.")
        return calls

    def _action_sequences(self, calls):
        """Use one canonical Hermes-style encoding per action, ending at EOS.

        Titles stay as UTF-8 text (e.g. "Côte d'Ivoire"), not \\u escapes.
        """
        return {
            action: self.tokenizer.encode(
                "<tool_call>\n" + json.dumps(call, ensure_ascii=False) + "\n</tool_call>",
                add_special_tokens=False,
            ) + [self.tokenizer.eos_token_id]
            for action, call in calls.items()
        }

    @staticmethod
    def _allowed_tokens(sequences, prefix):
        return sorted({
            sequence[len(prefix)] for sequence in sequences.values()
            if len(sequence) > len(prefix) and sequence[:len(prefix)] == prefix
        })

    def _encode_observation(self, observation):
        """Apply the native template before any text truncation or device transfer."""
        return self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.build_prompt(observation)},
            ],
            tools=TOOLS if observation["back_available"] else TOOLS[:1],
            tokenize=True, add_generation_prompt=True, enable_thinking=False,
            return_dict=True, return_tensors="pt",
        )

    def _prepare_inputs(self, observation, output_tokens):
        """Preserve tools, state and every link title; shorten only article content to fit.

        article_text: truncate the article text. link_context: drop link
        sentences from the bottom of the list first, then shorten the lead.
        """
        limit = self.model.config.max_position_embeddings - output_tokens
        if self.max_input_tokens is not None:
            limit = min(limit, self.max_input_tokens)
        if limit < 1:
            raise ValueError("The input token budget must be positive.")
        visible = observation.copy()
        if self.prompt_style == "link_context":
            return self._fit_link_context(visible, limit)
        text_tokens = self.tokenizer.encode(observation["text"], add_special_tokens=False)
        while True:
            inputs = self._encode_observation(visible)
            overflow = inputs["input_ids"].shape[1] - limit
            if overflow <= 0:
                return inputs.to(self.model.device)
            if not text_tokens:
                raise ValueError("The tools, state, and links exceed the input token budget.")
            text_tokens = text_tokens[:max(0, len(text_tokens) - overflow)]
            visible["text"] = self.tokenizer.decode(
                text_tokens, clean_up_tokenization_spaces=False,
            )

    def _fit_link_context(self, visible, limit):
        fits = lambda inputs: inputs["input_ids"].shape[1] <= limit
        inputs = self._encode_observation(visible)
        if fits(inputs):
            return inputs.to(self.model.device)
        # Keep the largest number of link sentences that fits (binary search).
        low, high = 0, sum(1 for a in visible["actions"] if a.get("context"))
        best = None
        while low <= high:
            middle = (low + high) // 2
            visible["context_limit"] = middle
            candidate = self._encode_observation(visible)
            if fits(candidate):
                best, low = candidate, middle + 1
            else:
                high = middle - 1
        if best is not None:
            return best.to(self.model.device)
        visible["context_limit"] = 0
        lead_tokens = self.tokenizer.encode(visible["lead"] or "", add_special_tokens=False)
        while True:
            inputs = self._encode_observation(visible)
            overflow = inputs["input_ids"].shape[1] - limit
            if overflow <= 0:
                return inputs.to(self.model.device)
            if not lead_tokens:
                raise ValueError("The tools, state, and link titles exceed the input token budget.")
            lead_tokens = lead_tokens[:max(0, len(lead_tokens) - overflow)]
            visible["lead"] = self.tokenizer.decode(lead_tokens, clean_up_tokenization_spaces=False)

    def _next_log_probs(self, prompt_ids, prefix, choices):
        """Shared full-prefix calculation for rollout and gradient-enabled scoring."""
        input_ids = torch.tensor(
            [prompt_ids + prefix], device=self.model.device, dtype=torch.long,
        )
        logits = self.model(
            input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
            use_cache=False, logits_to_keep=1,
        ).logits[0, -1].float()
        return torch.log_softmax(logits[choices], dim=-1)

    def _decoder(self):
        """The transformer body (LoRA layers are injected in place) and its output layer."""
        base = self.model.get_base_model() if isinstance(self.model, PeftModel) else self.model
        return base.model, base.get_output_embeddings()

    def _hidden(self, input_ids, past_key_values=None, use_cache=False):
        """Final normalized hidden states; the full-vocabulary logits are never formed."""
        body, _ = self._decoder()
        ids = torch.tensor([input_ids], device=self.model.device, dtype=torch.long)
        output = body(input_ids=ids, past_key_values=past_key_values, use_cache=use_cache)
        return output.last_hidden_state[0], output.past_key_values

    def _choice_log_probs(self, hidden, choices):
        """Log probabilities over the legal tokens, with the output layer in float32.

        bfloat16 logits of size 16-64 are rounded to steps of 0.125-0.25, so two
        computations that accumulate in a different order can disagree by whole
        rounding steps. Scoring only the legal rows in float32 removes that.
        """
        _, head = self._decoder()
        logits = hidden.float() @ head.weight[choices].float().T
        if head.bias is not None:
            logits = logits + head.bias[choices].float()
        return torch.log_softmax(logits, dim=-1)

    def _sample_cached(self, prompt_ids, sequences, greedy=False):
        """Choose one legal call, reusing a KV cache between branching points.

        The prompt is encoded once; each later forward pass only covers the tokens
        appended since the previous branch. Forced tokens need no forward pass of
        their own. greedy=True takes the most likely legal token at each decision
        instead of sampling. Returns the tokens, their legal-token masks, and the
        log probabilities the choices were made from.
        """
        tokens, token_choices, sampled = [], [], []
        self._last_entropies = []
        cache, pending = None, list(prompt_ids)
        for _ in range(max(map(len, sequences.values()))):
            choices = self._allowed_tokens(sequences, tokens)
            if not choices:
                raise RuntimeError("Generation reached an invalid tool-call prefix.")
            if len(choices) == 1:
                token, score = choices[0], 0.0
            else:
                hidden, cache = self._hidden(pending, past_key_values=cache, use_cache=True)
                pending = []
                log_probs = self._choice_log_probs(hidden[-1], choices)
                self._last_entropies.append(-(log_probs.exp() * log_probs).sum().item())
                index = (log_probs.argmax().item() if greedy
                         else torch.multinomial(log_probs.exp(), num_samples=1).item())
                token, score = choices[index], log_probs[index].item()
            tokens.append(token)
            token_choices.append(choices)
            sampled.append(score)
            pending.append(token)
            if token == self.tokenizer.eos_token_id:
                break
        return tokens, token_choices, sampled

    def _replay_log_probs(self, prompt_ids, token_ids, token_choices):
        """Score a sampled call with exactly the cached computation it was drawn from.

        _sample_cached feeds the prompt plus forced tokens up to the first decision,
        then, for each later decision, the previous chosen token plus forced tokens
        since, reusing the KV cache. Replaying the same chunks reproduces the
        sampling probabilities exactly, so no separate rescoring pass is needed.
        """
        self._check_recorded(prompt_ids, token_ids, token_choices)
        body, _ = self._decoder()
        if torch.is_grad_enabled() and self.model.training and getattr(body, "gradient_checkpointing", False):
            raise ValueError("Replay scoring needs activation checkpointing off (it disables the KV cache).")
        zero = torch.zeros((), device=self.model.device, requires_grad=torch.is_grad_enabled())
        scores = [zero] * len(token_ids)
        cache, start, pending = None, 0, list(prompt_ids)
        for i, choices in enumerate(token_choices):
            if len(choices) == 1:
                continue
            hidden, cache = self._hidden(pending + token_ids[start:i], past_key_values=cache, use_cache=True)
            scores[i] = self._choice_log_probs(hidden[-1], choices)[choices.index(token_ids[i])]
            pending, start = [], i
        return torch.stack(scores)

    def _check_recorded(self, prompt_ids, token_ids, token_choices):
        if not prompt_ids or not token_ids or len(token_ids) != len(token_choices):
            raise ValueError("Provide a prompt and matching token/mask sequences.")
        for token, choices in zip(token_ids, token_choices):
            if not choices or len(choices) != len(set(choices)) or token not in choices:
                raise ValueError("Each recorded token must have a valid legal-token mask.")

    @torch.no_grad()
    def choose_action(self, observation, deterministic=False):
        """Choose one legal call; forced syntax never needs a forward pass.

        Greedy decisions (evaluation) use the same KV-cached path with argmax and
        record the probabilities they were chosen from. With cached_greedy=False
        they instead query the model uncached at each branching prefix, the
        decoder the first baselines were produced with.

        Sampled decisions (training rollouts) draw with a KV cache, then record
        token log probabilities from one teacher-forced pass over the chosen call:
        the same calculation training uses, so the recorded behavior
        probabilities are reproduced exactly at update time. The draw itself
        used the cached probabilities, which agree up to floating-point rounding;
        sampling_max_abs_error reports the largest difference. Temperature is 1
        with no top-k/top-p filtering.
        """
        self.model.eval()
        calls = self._tool_calls(observation)
        sequences = self._action_sequences(calls)
        inputs = self._prepare_inputs(observation, max(map(len, sequences.values())))
        prompt_ids = inputs["input_ids"][0].tolist()
        if not deterministic or self.cached_greedy:
            tokens, token_choices, sampled = self._sample_cached(prompt_ids, sequences, greedy=deterministic)
            action = next((a for a, sequence in sequences.items() if sequence == tokens), None)
            if action is None:
                raise RuntimeError("Generation did not produce a complete legal tool call.")
            if deterministic or self.replay_scoring:
                scores = sampled  # Greedy decisions are not trained on; replay reproduces these exactly.
            elif any(len(choices) > 1 for choices in token_choices):
                scores = self._step_log_probs(prompt_ids, tokens, token_choices).tolist()
            else:
                scores = [0.0] * len(tokens)
            errors = [abs(a - b) for a, b, c in zip(sampled, scores, token_choices) if len(c) > 1]
            return {
                "action": action, "tool_call": calls[action], "prompt_ids": prompt_ids,
                "token_ids": tokens, "token_choices": token_choices,
                "token_log_probs": scores,
                "log_prob": torch.tensor(scores, device=self.model.device, dtype=torch.float32).sum().item(),
                "sampling_abs_errors": errors,
                "sampling_max_abs_error": max(errors, default=0.0),
                # Entropy of each decision's legal-token distribution, in nats.
                "decision_entropies": self._last_entropies,
            }
        tokens, token_choices, scores = [], [], []
        for _ in range(max(map(len, sequences.values()))):
            choices = self._allowed_tokens(sequences, tokens)
            if not choices:
                raise RuntimeError("Generation reached an invalid tool-call prefix.")
            if len(choices) == 1:
                token, score = choices[0], 0.0
            else:
                log_probs = self._next_log_probs(prompt_ids, tokens, choices)
                index = log_probs.argmax().item() if deterministic else torch.multinomial(
                    log_probs.exp(), num_samples=1,
                ).item()
                token, score = choices[index], log_probs[index].item()
            tokens.append(token)
            token_choices.append(choices)
            scores.append(score)
            if token == self.tokenizer.eos_token_id:
                break
        action = next((a for a, sequence in sequences.items() if sequence == tokens), None)
        if action is None:
            raise RuntimeError("Generation did not produce a complete legal tool call.")
        return {
            "action": action, "tool_call": calls[action], "prompt_ids": prompt_ids,
            "token_ids": tokens, "token_choices": token_choices,
            "token_log_probs": scores,
            "log_prob": torch.tensor(scores, device=self.model.device, dtype=torch.float32).sum().item(),
        }

    def _step_log_probs(self, prompt_ids, token_ids, token_choices):
        """Score a whole recorded call with one teacher-forced forward pass.

        The model reads the prompt plus every recorded token except the last; the
        hidden states at the final len(token_ids) positions predict each recorded
        token, scored over its legal tokens in float32. Forced tokens contribute zero.
        """
        self._check_recorded(prompt_ids, token_ids, token_choices)
        zero = torch.zeros((), device=self.model.device, requires_grad=torch.is_grad_enabled())
        branches = [i for i, choices in enumerate(token_choices) if len(choices) > 1]
        if not branches:
            return torch.stack([zero] * len(token_ids))
        hidden, _ = self._hidden(prompt_ids + token_ids[:-1])
        hidden = hidden[-len(token_ids):]  # Row i predicts token_ids[i].
        scores = [zero] * len(token_ids)
        for i in branches:
            choices = token_choices[i]
            scores[i] = self._choice_log_probs(hidden[i], choices)[choices.index(token_ids[i])]
        return torch.stack(scores)

    def get_token_log_probs(self, prompt_ids, token_ids, token_choices):
        """Score recorded tokens with gradients, using their exact input and masks.

        With replay_scoring, this replays the cached sampling computation exactly;
        otherwise it is one teacher-forced pass per call, which sampled decisions
        then record. Forced tokens contribute zero. Training controls model mode;
        keep dropout disabled when comparing rollout and training probabilities.
        """
        if self.replay_scoring:
            return self._replay_log_probs(prompt_ids, token_ids, token_choices)
        return self._step_log_probs(prompt_ids, token_ids, token_choices)

    def iter_token_log_probs(self, prompt_ids, token_ids, token_choices):
        """Reference scorer: one uncached forward pass per branching prefix.

        This is the calculation greedy decisions use; it is kept to check them.
        """
        if not prompt_ids or not token_ids or len(token_ids) != len(token_choices):
            raise ValueError("Provide a prompt and matching token/mask sequences.")
        zero = torch.zeros((), device=self.model.device, requires_grad=torch.is_grad_enabled())
        for position, (token, choices) in enumerate(zip(token_ids, token_choices)):
            if not choices or len(choices) != len(set(choices)) or token not in choices:
                raise ValueError("Each recorded token must have a valid legal-token mask.")
            if len(choices) == 1:
                yield zero
                continue
            log_probs = self._next_log_probs(prompt_ids, token_ids[:position], choices)
            yield log_probs[choices.index(token)]

    def get_action_log_prob(self, observation, action):
        """Sum the masked token log probabilities of one complete legal call."""
        calls = self._tool_calls(observation)
        if action not in calls:
            raise ValueError("The action is not available in this observation.")
        sequences = self._action_sequences(calls)
        inputs = self._prepare_inputs(observation, max(map(len, sequences.values())))
        tokens = sequences[action]
        choices = [self._allowed_tokens(sequences, tokens[:i]) for i in range(len(tokens))]
        return self.get_token_log_probs(inputs["input_ids"][0].tolist(), tokens, choices).sum()
