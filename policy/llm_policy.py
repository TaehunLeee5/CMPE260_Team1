import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from policy.prompt import OBSERVATION_TEMPLATE, SYSTEM_PROMPT


class WikiGamePolicy:
    """Use a language model to choose links in the Wiki Game."""

    def __init__(self, model_name="Qwen/Qwen3-1.7B", max_input_tokens=None):
        """Load the model, tokenizer, and fixed game instructions."""
        self.system_prompt = SYSTEM_PROMPT
        self.max_input_tokens = max_input_tokens
        device = "cuda" if torch.cuda.is_available() else "cpu"

        dtype = "auto"
        if device == "cuda":
            dtype = (
                torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype
        )
        self.model.to(device)
        self.model.eval()

    def build_prompt(self, observation):
        """Fill the observation template after formatting links and history."""
        values = observation.copy()
        values["links"] = "\n".join(
            f"{action['id']}: {action['title']}"
            for action in observation["actions"]
        )
        values["history"] = " -> ".join(observation["history"])
        return OBSERVATION_TEMPLATE.format(**values)

    def _action_sequences(self, actions):
        """Encode valid action IDs, including an end token for each ID."""
        if not actions:
            raise ValueError("The observation has no available actions.")

        if self.tokenizer.eos_token_id is None:
            raise ValueError("The tokenizer needs an end-of-sequence token.")

        return {
            action["id"]: self.tokenizer.encode(
                str(action["id"]), add_special_tokens=False
            ) + [self.tokenizer.eos_token_id]
            for action in actions
        }

    def _allowed_tokens(self, sequences, prefix):
        """Return tokens that can continue a valid action ID."""
        return sorted({
            sequence[len(prefix)]
            for sequence in sequences.values()
            if len(sequence) > len(prefix)
            and sequence[:len(prefix)] == prefix
        })

    def _prepare_inputs(self, observation, output_tokens):
        """Tokenize the prompt, shortening only article text if needed."""
        limit = self.model.config.max_position_embeddings - output_tokens
        if self.max_input_tokens is not None:
            limit = min(limit, self.max_input_tokens)

        if limit < 1:
            raise ValueError("The input token budget must be positive.")

        visible = observation.copy()
        text_tokens = self.tokenizer.encode(
            observation["text"], add_special_tokens=False
        )

        while True:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.build_prompt(visible)},
            ]
            inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_dict=True,
                return_tensors="pt",
            )

            overflow = inputs["input_ids"].shape[1] - limit
            if overflow <= 0:
                return inputs.to(self.model.device)

            if not text_tokens:
                raise ValueError(
                    "The instructions, titles, history, and links exceed "
                    "the input token budget."
                )

            text_tokens = text_tokens[:max(0, len(text_tokens) - overflow)]
            visible["text"] = self.tokenizer.decode(
                text_tokens, clean_up_tokenization_spaces=False
            )

    @torch.no_grad()
    def choose_action(self, observation, deterministic=False):
        """Return a valid action ID and its log probability.

        Sample actions when collecting training routes. Set deterministic
        to True for greedy evaluation. The log probability is a float;
        use get_action_log_prob when gradients are needed.

        Sampling uses temperature 1 with top-k and top-p filtering disabled
        so rollout probabilities match the training calculation.
        """
        self.model.eval()
        sequences = self._action_sequences(observation["actions"])
        output_tokens = max(map(len, sequences.values()))
        inputs = self._prepare_inputs(observation, output_tokens)
        prompt_length = inputs["input_ids"].shape[1]

        def restrict_tokens(batch_id, input_ids):
            """Restrict generation to valid action ID continuations."""
            prefix = input_ids[prompt_length:].tolist()
            return self._allowed_tokens(sequences, prefix)

        config = GenerationConfig(
            max_new_tokens=output_tokens,
            do_sample=not deterministic,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )
        if not deterministic:
            config.temperature = 1.0
            config.top_k = 0
            config.top_p = 1.0

        result = self.model.generate(
            **inputs,
            generation_config=config,
            prefix_allowed_tokens_fn=restrict_tokens,
        )
        generated = result.sequences[0, prompt_length:].tolist()

        action = next(
            action
            for action, sequence in sequences.items()
            if sequence == generated
        )
        scores = self.model.compute_transition_scores(
            result.sequences,
            result.scores,
            normalize_logits=True,
        )

        return action, scores[0].sum().item()

    def get_action_log_prob(self, observation, action):
        """Calculate an action's log probability with gradients.

        Use this during training updates. It applies the same prompt and
        valid-action restrictions as choose_action.

        Training code controls model mode and should disable dropout when
        comparing old and new policy probabilities.
        """
        sequences = self._action_sequences(observation["actions"])
        if action not in sequences:
            raise ValueError("The action is not available in this observation.")

        inputs = self._prepare_inputs(
            observation, max(map(len, sequences.values()))
        )
        sequence = sequences[action]
        continuation = torch.tensor(
            [sequence[:-1]],
            device=self.model.device,
            dtype=torch.long,
        )
        input_ids = torch.cat(
            [inputs["input_ids"], continuation], dim=1
        )

        logits = self.model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=False,
            logits_to_keep=len(sequence),
        ).logits[0].float()

        log_probs = []
        for position, token in enumerate(sequence):
            choices = self._allowed_tokens(
                sequences, sequence[:position]
            )
            probabilities = torch.log_softmax(
                logits[position, choices], dim=-1
            )
            log_probs.append(probabilities[choices.index(token)])

        return torch.stack(log_probs).sum()
