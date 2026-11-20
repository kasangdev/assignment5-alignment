import gc

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from cs336_alignment.vllm_utils import VLLMCompletion


class TransformersServer:
    def __init__(self, model_id, gpu=0, seed=0):
        self.model_id = model_id
        self.device = f"cuda:{gpu}"
        self.seed = seed
        self.model = None
        self.tokenizer = None

    def start(self):
        set_seed(self.seed)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            local_files_only=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            attn_implementation="sdpa",
            local_files_only=True,
        )
        self.model.eval()

    def generate_completions(self, prompts, sampling_params, batch_size=1):
        completions = []

        max_new_tokens = sampling_params.get("max_tokens", 512)
        temperature = sampling_params.get("temperature", 1.0)
        top_p = sampling_params.get("top_p", 1.0)
        seed = sampling_params.get("seed", self.seed)
        stop_strings = sampling_params.get("stop", None)

        pad_token_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )

        for index, prompt in enumerate(prompts):
            torch.manual_seed(seed + index)

            model_inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=False,
            )
            model_inputs = {
                key: value.to(self.device)
                for key, value in model_inputs.items()
            }
            input_length = model_inputs["input_ids"].shape[1]

            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
                "pad_token_id": pad_token_id,
            }
            if stop_strings:
                generation_kwargs["stop_strings"] = stop_strings
                generation_kwargs["tokenizer"] = self.tokenizer

            with torch.inference_mode():
                outputs = self.model.generate(
                    **model_inputs,
                    **generation_kwargs,
                )

            new_token_ids = outputs[0, input_length:]
            new_token_list = new_token_ids.tolist()
            text = self.tokenizer.decode(
                new_token_ids,
                skip_special_tokens=True,
            )

            if len(new_token_list) < max_new_tokens:
                finish_reason = "stop"
            elif stop_strings and any(text.endswith(s) for s in stop_strings):
                finish_reason = "stop"
            elif (
                    self.tokenizer.eos_token_id is not None
                    and new_token_list
                    and new_token_list[-1] == self.tokenizer.eos_token_id
            ):
                finish_reason = "stop"
            else:
                finish_reason = "length"

            completions.append(
                VLLMCompletion(
                    text=text,
                    token_ids=new_token_list,
                    finish_reason=finish_reason,
                )
            )

        return completions

    def stop(self):
        self.model = None
        self.tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()
