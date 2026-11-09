import json

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from policy.prompt import OBSERVATION_TEMPLATE, SYSTEM_PROMPT, TOOLS


class WikiGamePolicy:
    """Generate one legal Qwen tool call and retain its token probabilities."""

    def __init__(self, model_name="Qwen/Qwen3-1.7B", max_input_tokens=4096,
                 *, model=None, tokenizer=None, adapter_path=None):
        """Load the actor or injected model, optionally adding a frozen local adapter."""
        if max_input_tokens is not None and max_input_tokens < 1:
            raise ValueError("The input token budget must be positive.")
        self.system_prompt = SYSTEM_PROMPT
        self.observation_template = OBSERVATION_TEMPLATE
        self.max_input_tokens = max_input_tokens
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

    def build_prompt(self, observation):
        """Format the visible state without exposing internal Back action IDs."""
        values = observation.copy()
        values["links"] = "\n".join(
            f"{action['id']}: {action['title']}" for action in observation["actions"]
        ) or "None"
        values["history"] = " -> ".join(observation["history"])
        values["navigation_stack"] = " -> ".join(observation["navigation_stack"])
        values["back_destination"] = (
            observation["back_destination"] if observation["back_available"] else "unavailable"
        )
        return self.observation_template.format(**values)

    def _tool_calls(self, observation):
        """Map each legal internal environment action to its native tool call."""
        calls = {}
        for action in observation["actions"]:
            ident = action["id"]
            if type(ident) is not int or ident < 0 or ident in calls:
                raise ValueError("Link IDs must be unique nonnegative integers.")
            calls[ident] = {"name": "follow_link", "arguments": {"link_id": ident}}
        if observation["back_available"]:
            ident = observation["back_action_id"]
            if type(ident) is not int or ident < 0 or ident in calls:
                raise ValueError("Back needs a distinct nonnegative action ID.")
            calls[ident] = {"name": "go_back", "arguments": {}}
        if not calls:
            raise ValueError("The observation has no available actions.")
        return calls

    def _action_sequences(self, calls):
        """Use one canonical Hermes-style encoding per action, ending at EOS."""
        return {
            action: self.tokenizer.encode(
                "<tool_call>\n" + json.dumps(call) + "\n</tool_call>",
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
        """Preserve tools and state; shorten only current article text to fit."""
        limit = self.model.config.max_position_embeddings - output_tokens
        if self.max_input_tokens is not None:
            limit = min(limit, self.max_input_tokens)
        if limit < 1:
            raise ValueError("The input token budget must be positive.")
        visible = observation.copy()
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

    @torch.no_grad()
    def choose_action(self, observation, deterministic=False):
        """Query the model only at branching prefixes of the legal-call grammar.

        Forced syntax has probability one and is appended without a forward pass.
        Actual choices use the same uncached calculation as training, with
        temperature 1 and no top-k/top-p filtering.
        """
        self.model.eval()
        calls = self._tool_calls(observation)
        sequences = self._action_sequences(calls)
        inputs = self._prepare_inputs(observation, max(map(len, sequences.values())))
        prompt_ids = inputs["input_ids"][0].tolist()
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

    def get_token_log_probs(self, prompt_ids, token_ids, token_choices):
        """Score recorded tokens with gradients, using their exact input and masks.

        Forced tokens contribute zero. Training controls model mode; keep dropout
        disabled when comparing rollout and training probabilities.
        """
        return torch.stack(list(self.iter_token_log_probs(prompt_ids, token_ids, token_choices)))

    def iter_token_log_probs(self, prompt_ids, token_ids, token_choices):
        """Yield scores separately so training can backpropagate one branch at a time."""
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
