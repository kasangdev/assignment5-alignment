import json
import argparse
from pathlib import Path
from cs336_alignment.local_inference import TransformersServer
from cs336_alignment.vllm_utils import VLLMServer

from cs336_alignment.drgrpo_grader import (
    r1_zero_reward_fn,
    question_only_reward_fn,
)


def load_gsm8k_examples(data_path, limit=None):
    examples = []
    with open(data_path, "r", encoding="utf-8") as data_file:
        for line in data_file:
            if limit is not None and len(examples) >= limit:
                break
            if not line.strip():
                continue
            record = json.loads(line)
            question = record["question"]
            answer_text = record["answer"]
            ground_truth = answer_text.split("####")[-1].strip()
            examples.append({"question": question, "ground_truth": ground_truth})
    return examples


def build_prompts(examples, prompt_name):
    prompt_files = {
        "question_only": "question_only.prompt",
        "r1_zero": "r1_zero.prompt",
        "r1_zero_three_shot": "r1_zero_three_shot_gsm8k.prompt",
    }

    prompt_dir = Path(__file__).resolve().parent / "prompts"
    prompt_path = prompt_dir / prompt_files[prompt_name]
    template = prompt_path.read_text(encoding="utf-8")

    prompts = []
    for example in examples:
        question = example["question"]
        prompts.append(template.format(question=question))

    return prompts


def evaluate_prompting(examples, prompt_name, server, sampling_params, batch_size=32):
    prompts = build_prompts(examples, prompt_name)

    generation_params = sampling_params.copy()
    if prompt_name in ("r1_zero", "r1_zero_three_shot"):
        generation_params["stop"] = ["</answer>"]
        generation_params["include_stop_str_in_output"] = True

    completions = server.generate_completions(
        prompts,
        generation_params,
        batch_size,
    )

    reward_fns = {
        "question_only": question_only_reward_fn,
        "r1_zero": r1_zero_reward_fn,
        "r1_zero_three_shot": r1_zero_reward_fn,
    }
    reward_fn = reward_fns[prompt_name]

    records = []
    total_reward = 0.0
    total_format = 0.0
    total_answer = 0.0
    total_length = 0

    for prompt, example, completion in zip(prompts, examples, completions):
        response = completion.text
        ground_truth = example["ground_truth"]
        scores = reward_fn(response, ground_truth)

        records.append({
            "prompt": prompt,
            "question": example["question"],
            "ground_truth": ground_truth,
            "completion": response,
            "reward": scores["reward"],
            "format_reward": scores.get("format_reward", 0.0),
            "answer_reward": scores.get("answer_reward", 0.0),
            "length": len(completion.token_ids),
            "finish_reason": completion.finish_reason,
        })

        total_reward += scores["reward"]
        total_format += scores.get("format_reward", 0.0)
        total_answer += scores.get("answer_reward", 0.0)
        total_length += len(completion.token_ids)

    num_correct = int(round(total_answer))
    num_format_only = int(round(total_format - total_answer))
    num_bad_format = int(round(len(records) - total_format))

    n = len(records) if records else 1
    metrics = {
        "prompt_name": prompt_name,
        "num_examples": len(records),
        "avg_reward": total_reward / n,
        "avg_format_reward": total_format / n,
        "avg_answer_reward": total_answer / n,
        "avg_length": total_length / n,
        "num_correct": num_correct,
        "num_format_only": num_format_only,
        "num_bad_format": num_bad_format,
    }

    return metrics, records


def main():
    project_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description="Evaluate GSM8K prompting for the A5 alignment project."
    )

    parser.add_argument(
        "--model-id",
        type=str,
        default="allenai/OLMo-2-0425-1B",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=project_root / "data" / "gsm8k" / "test.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["vllm", "transformers"],
        default="vllm",
        help="Inference backend: vllm or transformers.",
    )

    args = parser.parse_args()

    examples = load_gsm8k_examples(args.data_path, args.limit)
    print(f"Loaded {len(examples)} examples from {args.data_path}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    sampling_params = {
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 512,
        "n": 1,
        "seed": args.seed,
    }

    prompt_names = [
        "question_only",
        "r1_zero",
        "r1_zero_three_shot",
    ]

    if args.backend == "vllm":
        server = VLLMServer(
            model_id=args.model_id,
            gpu=args.gpu,
            seed=args.seed,
        )
    else:  # transformers
        server = TransformersServer(
            model_id=args.model_id,
            gpu=args.gpu,
            seed=args.seed,
        )

    try:
        server.start()

        for prompt_name in prompt_names:
            metrics, records = evaluate_prompting(
                examples=examples,
                prompt_name=prompt_name,
                server=server,
                sampling_params=sampling_params,
                batch_size=args.batch_size,
            )
            print(metrics)

            metrics_path = output_dir / f"{prompt_name}_metrics.json"
            records_path = output_dir / f"{prompt_name}_records.jsonl"

            with open(metrics_path, "w", encoding="utf-8") as metrics_file:
                json.dump(
                    metrics,
                    metrics_file,
                    ensure_ascii=False,
                    indent=2,
                )

            with open(records_path, "w", encoding="utf-8") as records_file:
                for record in records:
                    records_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    finally:
        server.stop()


if __name__ == "__main__":
    main()
