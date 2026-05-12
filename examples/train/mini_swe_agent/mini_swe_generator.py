import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import yaml
import traceback
import ray
from pathlib import Path

from minisweagent.models import get_model
from minisweagent.agents.default import DefaultAgent
from minisweagent.config import get_config_path
from .mini_swe_utils import evaluate_trajectory, get_sb_environment

from skyrl.train.config import GeneratorConfig, SkyRLGymConfig
from skyrl.train.generators.skyrl_gym_generator import SkyRLGymGenerator, GeneratorOutput, GeneratorInput
from skyrl.train.generators.base import TrajectoryID, TrainingPhase, BatchMetadata
from skyrl.backends.skyrl_train.inference_engines.base import ConversationType
from skyrl.backends.skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import RemoteInferenceClient
from skyrl.backends.skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl.train.generators.utils import (
    get_rollout_metrics,
    get_response_ids_and_loss_mask_from_messages,
)


@dataclass
class MiniSWEGeneratorConfig(GeneratorConfig):
    """Extended generator config with Mini-SWE-Agent-specific fields."""

    miniswe_config_path: str = ""
    miniswe_traj_dir: str = ""
    use_tito_proxy: bool = False


class DefaultAgentWithReminder(DefaultAgent):
    """Subclass that preserves raw assistant messages on FormatError.

    In upstream mini-swe-agent, if the model's response fails action parsing
    (FormatError), the assistant message is never added to self.messages because
    the exception is raised inside model.query() before the message is returned.
    This override patches the model to always stash the raw response so it can
    be recovered and saved in the trajectory for debugging.
    """

    def __init__(self, model, env, **kwargs):
        super().__init__(model, env, **kwargs)
        self._patch_model_query()

    def _patch_model_query(self):
        """Wrap model._query to stash the raw response before parsing."""
        original_inner_query = self.model._query

        def patched_query(messages, **kwargs):
            response = original_inner_query(messages, **kwargs)
            # Stash raw assistant message so we can recover it on FormatError
            try:
                self.model._last_raw_message = response.choices[0].message.model_dump()
            except Exception:
                pass
            return response

        self.model._query = patched_query

    def step(self) -> list[dict]:
        from minisweagent.exceptions import FormatError

        try:
            ret = super().step()
            return ret
        except FormatError as e:
            # Preserve the raw assistant response that failed parsing
            raw_msg = getattr(self.model, '_last_raw_message', None)
            if raw_msg and not any(m is raw_msg for m in self.messages):
                raw_msg.setdefault("extra", {})["format_error"] = True
                self.add_messages(raw_msg)
            raise


@ray.remote(num_cpus=0.01)
def init_and_run(
    instance: dict,
    litellm_model_name: str,
    sweagent_config: dict,
    generator_cfg: GeneratorConfig,
    data_source: str,
    sampling_params: dict,
    trajectory_id: TrajectoryID,
    global_step: int,
    training_phase: TrainingPhase,
    base_url: str = None,
):
    import os

    from loguru import logger

    if base_url is not None:
        os.environ["OPENAI_BASE_URL"] = base_url

    model_config = sweagent_config.get("model", {})
    # Use new sampling parameters
    # Can also have custom sampling parameters per trajectory (ex: custom max tokens)
    # Convert integer logprobs to OpenAI chat completions format (boolean + top_logprobs)
    sp = dict(sampling_params)
    if isinstance(sp.get("logprobs"), int):
        top_n = sp.pop("logprobs")
        sp["logprobs"] = True
        sp["top_logprobs"] = top_n
    model_config.setdefault("model_kwargs", {}).update(sp)
    model = get_model(litellm_model_name, model_config)

    agent = None
    env = None
    extra_info = None
    result = None
    reward = 0
    error = None
    try:
        env = get_sb_environment(sweagent_config, instance, data_source)
        agent = DefaultAgentWithReminder(model, env, **sweagent_config.get("agent", {}))
        # v2: agent.run() returns a dict with exit_status/submission keys
        run_result = agent.run(instance["problem_statement"])  # type: ignore[arg-type]
        print("Agent run result:", run_result)
        exit_status = run_result.get("exit_status", "unknown")
        result = run_result.get("submission", "")
    except Exception as e:
        logger.error(f"Error processing instance {instance['instance_id']}: {e}", exc_info=True)
        exit_status, result = type(e).__name__, str(e)
        error = str(e)
        extra_info = {"traceback": traceback.format_exc()}
    finally:
        # Create trajectory directory with proper structure: step_{global_step}/{train/eval}
        path = Path(generator_cfg.miniswe_traj_dir) / f"step_{global_step}" / training_phase
        path.mkdir(parents=True, exist_ok=True)
        # Use instance_id and repetition_id for meaningful filename: {instance_id}_{repetition_id}.json
        instance_id = instance["instance_id"]
        filename = f"{instance_id}_{trajectory_id.repetition_id}.json"
        path = path / filename
        if agent is not None:
            eval_error = None
            try:
                result = evaluate_trajectory(instance, result, sweagent_config, data_source)
                reward = int(result["resolved"])
                eval_error = result["eval_error"]
                if eval_error:
                    error = eval_error
                    logger.debug(f"Error during evaluation {eval_error}")
            except Exception as e:
                logger.debug(f"Error during evaluation {e}")
                logger.debug(f"traceback: {traceback.format_exc()}")
                eval_error = str(e)
                error = str(e)

            agent.save(path, {"exit_status": exit_status, "result": result, "extra_info": extra_info, "reward": reward, "eval_error": eval_error})

    return (agent.messages if agent is not None else [], reward, error)


class MiniSweAgentGenerator(SkyRLGymGenerator):
    def __init__(
        self,
        generator_cfg: GeneratorConfig,
        skyrl_gym_cfg: SkyRLGymConfig,
        inference_engine_client: Tuple[InferenceEngineClient, RemoteInferenceClient],
        tokenizer,
        model_name: str,
    ):

        # Call parent constructor first
        super().__init__(generator_cfg, skyrl_gym_cfg, inference_engine_client, tokenizer)

        self.http_server_inference_engine_client_host = generator_cfg.inference_engine.http_endpoint_host

        self.http_server_inference_engine_client_port = generator_cfg.inference_engine.http_endpoint_port

        # Use the inference server's dynamic URL when available (new inference server path),
        # otherwise fall back to the legacy static host:port from config.
        try:
            from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import RemoteInferenceClient

            if isinstance(inference_engine_client, RemoteInferenceClient):
                backend_url = inference_engine_client.proxy_url
            else:
                backend_url = f"http://{self.http_server_inference_engine_client_host}:{self.http_server_inference_engine_client_port}"
        except ImportError:
            backend_url = f"http://{self.http_server_inference_engine_client_host}:{self.http_server_inference_engine_client_port}"

        # Optionally start a TITO proxy between litellm and the vLLM router.
        # When enabled, init_and_run tasks hit the proxy, which can intercept
        # and convert chat completions to token-in-token-out calls.
        # Each rollout gets a unique URL: http://proxy:PORT/session/{id}/v1
        self._tito_proxy = None
        if getattr(generator_cfg, "use_tito_proxy", False):
            from .tito_proxy import TITOProxy

            self._tito_proxy = TITOProxy(
                backend_url=backend_url,
                log_path=getattr(generator_cfg, "miniswe_traj_dir", None),
            )
            self._tito_proxy.start()
            # base_url is set per-rollout in minisweagent_agent_loop via session_url()
            self.base_url = None
        else:
            self.base_url = backend_url + "/v1"

        self.generator_cfg = generator_cfg
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.litellm_model_name = "openai/" + self.model_name

        if self.generator_cfg.chat_template.name_or_path is not None:
            raise NotImplementedError("MiniSWEAgentGenerator doesn't support custom chat template")

        # Load custom chat template for training-side tokenization so it
        # matches the vLLM engine's template (e.g. qwen3_acc_thinking.jinja2).
        self._tokenizer_kwargs: Dict[str, Any] = {}
        chat_template_path = generator_cfg.inference_engine.engine_init_kwargs.get("chat_template")
        if chat_template_path:
            tpl_path = Path(chat_template_path)
            if tpl_path.exists():
                self._tokenizer_kwargs["chat_template"] = tpl_path.read_text()

    async def minisweagent_agent_loop(
        self,
        prompt: ConversationType,
        env_extras: Dict[str, Any],
        max_tokens: int,
        max_input_length: int,
        sampling_params: Dict[str, Any],
        trajectory_id: TrajectoryID,
        batch_metadata: BatchMetadata,
    ) -> Tuple[List[int], float, str, List[int], List[int], Optional[List[int]]]:

        sweagent_config = yaml.safe_load(get_config_path(self.generator_cfg.miniswe_config_path).read_text())

        # Construct per-rollout base_url: when TITO proxy is active, each
        # rollout gets a unique URL with the session ID in the path.
        session_id = None
        if self._tito_proxy is not None:
            session_id = f"{trajectory_id.instance_id}_{trajectory_id.repetition_id}_{batch_metadata.global_step}"
            rollout_base_url = self._tito_proxy.session_url(session_id)
        else:
            rollout_base_url = self.base_url

        # NOTE (sumanthrh): Input `prompt` is not used here because mini-swe-agent uses a similar entry from the `instance` obj
        messages, reward, error = await init_and_run.remote(
            env_extras["instance"],
            self.litellm_model_name,
            sweagent_config,
            self.generator_cfg,
            env_extras["data_source"],
            sampling_params,
            trajectory_id,
            batch_metadata.global_step,
            batch_metadata.training_phase,
            rollout_base_url,
        )
        if not len(messages):
            return None, None, None, None, None, None

        # --- TITO path: read tokens + loss_mask directly from proxy ---
        if self._tito_proxy is not None and session_id is not None:
            return await self._read_tito_session(
                session_id, messages, reward, error, max_tokens, max_input_length,
            )

        # --- Legacy path: re-tokenize from messages ---
        # TODO (sumanthrh): This is currently hardcoded for SWEBench with 2 initial messages (system and user).
        response_messages = messages[2:]

        for message in messages[:2]:
            assert message["role"] in (
                "system",
                "user",
            ), "Expected the first two messages to be system and user messages"

        initial_input_ids = self.tokenizer.apply_chat_template(
            messages[:2], add_generation_prompt=False, return_dict=False, tokenize=True,
            **self._tokenizer_kwargs,
        )
        initial_prompt_length = len(initial_input_ids)

        # We remove trailing `user` and `exit` messages - `exit` is added by mini-swe-agent v2, `user` captures the final git diff
        last_idx = len(response_messages) - 1
        while last_idx >= 0 and response_messages[last_idx]["role"] in ("user", "exit"):
            last_idx -= 1
        if last_idx < 0:
            # Agent exited with no assistant turns (e.g. LLM error on first call) — treat as failed rollout
            from loguru import logger as _logger
            _logger.warning(f"No assistant messages found in trajectory (error={error}). Treating as failed rollout.")
            return None, None, None, None, None, None
        response_messages = response_messages[: last_idx + 1]

        # Normalize roles for get_response_ids_and_loss_mask_from_messages,
        # which only handles 'user' and 'assistant'. Tool-call results ('tool')
        # and format-error messages are environment observations — treat as 'user'.
        response_messages = [
            {**msg, "role": "user"} if msg["role"] not in ("user", "assistant") else msg
            for msg in response_messages
        ]

        response_ids, loss_mask, _ = get_response_ids_and_loss_mask_from_messages(
            response_messages,
            self.tokenizer,
            assistant_logprobs=None,
            tokenizer_kwargs=self._tokenizer_kwargs,
        )

        # Extract prompt ids
        prompt_ids = initial_input_ids

        # Calculate maximum response tokens allowed
        max_response_tokens = max_tokens + max_input_length - initial_prompt_length

        # Determine stop reason
        stop_reason = "complete"  # Default for trial completion
        if len(response_ids) > max_response_tokens:
            stop_reason = "length"

        # Truncate to maximum allowed length
        response_ids = response_ids[:max_response_tokens]
        loss_mask = loss_mask[:max_response_tokens]

        return (response_ids, reward, stop_reason, loss_mask, prompt_ids, None)

    async def _read_tito_session(
        self,
        session_id: str,
        messages: List[Dict],
        reward: float,
        error: Optional[str],
        max_tokens: int,
        max_input_length: int,
    ) -> Tuple[List[int], float, str, List[int], List[int], Optional[List[int]]]:
        """Read token_ids and loss_mask from the TITO proxy session endpoint.

        Splits the proxy's accumulated tokens into prompt_ids (initial
        system+user, loss_mask=0) and response_ids (everything after).
        """
        import httpx as _httpx

        data_url = f"{self._tito_proxy.url}/session/{session_id}/data"
        async with _httpx.AsyncClient() as client:
            resp = await client.get(data_url, timeout=10)

        if resp.status_code != 200:
            from loguru import logger as _logger
            _logger.warning(f"TITO session {session_id} data fetch failed: {resp.status_code}")
            return None, None, None, None, None, None

        session_data = resp.json()
        all_tokens = session_data["tokens"]
        all_loss_mask = session_data["loss_mask"]

        if not all_tokens:
            return None, None, None, None, None, None

        # Split into prompt (initial non-generated tokens) and response
        # The prompt is the leading run of loss_mask=0 tokens
        first_gen = next(
            (i for i, m in enumerate(all_loss_mask) if m == 1), len(all_loss_mask)
        )
        prompt_ids = all_tokens[:first_gen]
        response_ids = all_tokens[first_gen:]
        response_loss_mask = all_loss_mask[first_gen:]

        # Calculate maximum response tokens allowed
        max_response_tokens = max_tokens + max_input_length - len(prompt_ids)

        stop_reason = "complete"
        if len(response_ids) > max_response_tokens:
            stop_reason = "length"

        response_ids = response_ids[:max_response_tokens]
        response_loss_mask = response_loss_mask[:max_response_tokens]

        # Clean up session from proxy
        async with _httpx.AsyncClient() as client:
            await client.delete(
                f"{self._tito_proxy.url}/session/{session_id}", timeout=5
            )

        return (response_ids, reward, stop_reason, response_loss_mask, prompt_ids, None)

    async def generate(self, input_batch: GeneratorInput) -> GeneratorOutput:
        """
        Generate trajectories for the input batch.

        Returns outputs in the same order as the input batch.
        Args:
            input_batch: GeneratorInput
        Returns:
            GeneratorOutput
        """
        prompts = input_batch["prompts"]
        env_extras = input_batch["env_extras"]
        trajectory_ids = input_batch["trajectory_ids"]
        batch_metadata = input_batch["batch_metadata"]
        max_tokens = self.generator_cfg.sampling_params.max_generate_length
        max_input_length = self.generator_cfg.max_input_length
        sampling_params = get_sampling_params_for_backend(
            self.generator_cfg.inference_engine.backend, self.generator_cfg.sampling_params
        )

        tasks = []

        for i in range(len(prompts)):
            tasks.append(
                self.minisweagent_agent_loop(
                    prompts[i],
                    env_extras[i],
                    max_tokens=max_tokens,
                    max_input_length=max_input_length,
                    sampling_params=sampling_params,
                    trajectory_id=trajectory_ids[i],
                    batch_metadata=batch_metadata,
                )
            )

        all_outputs = await asyncio.gather(*tasks)

        # Filter out the `None` entries, which means that trajectory generation failed
        responses = [output[0] for output in all_outputs if output[0] is not None]
        rewards = [output[1] for output in all_outputs if output[0] is not None]
        stop_reasons = [output[2] for output in all_outputs if output[0] is not None]
        loss_masks = [output[3] for output in all_outputs if output[0] is not None]
        prompt_token_ids = [output[4] for output in all_outputs if output[0] is not None]
        if not len(responses):
            raise ValueError(
                "Found no valid responses for this step. This means that generation failed for all trajectories, likely due to errors in environment setup."
            )
        rollout_metrics = get_rollout_metrics(responses, rewards)

        generator_output: GeneratorOutput = {
            "prompt_token_ids": prompt_token_ids,
            "response_ids": responses,
            "rewards": rewards,
            "loss_masks": loss_masks,
            "stop_reasons": stop_reasons,
            "rollout_metrics": rollout_metrics,
            "rollout_logprobs": None,
        }

        return generator_output
