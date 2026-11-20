import torch
import pytest

from cs336_alignment.data import tokenize_prompt_and_output
from cs336_alignment.grpo import get_response_log_probs, grpo_train_step


def toy_reward_fn(response, ground_truth):
    is_correct = 1.0 if response.strip() == ground_truth.strip() else 0.0
    return {
        "reward": is_correct,
        "answer_reward": is_correct,
        "format_reward": 1.0,
    }


def test_tiny_grpo_learning(tiny_train_model, tokenizer):
    torch.manual_seed(0)
    torch.set_num_threads(1)

    model = tiny_train_model
    prompt = "Hello"
    target = "world"
    num_steps = 20
    group_size = 32

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0)

    # 1. 构造 probe_batch 并测量训练前概率
    device = next(model.parameters()).device
    probe_batch = tokenize_prompt_and_output([prompt], [target], tokenizer)
    for k in probe_batch:
        if isinstance(probe_batch[k], torch.Tensor):
            probe_batch[k] = probe_batch[k].to(device)

    def get_target_prob():
        model.eval()
        with torch.no_grad():
            out = get_response_log_probs(
                model, probe_batch["input_ids"], probe_batch["labels"]
            )
            probs = out["log_probs"].exp()
            # 取 mask 为 1 的位置（即目标 token "world" 的概率）
            prob_val = probs[probe_batch["response_mask"] == 1].mean().item()
        return prob_val

    before_prob = get_target_prob()

    # 用于 generate 的 prompt 输入
    prompt_inputs = tokenizer([prompt], return_tensors="pt")
    prompt_input_ids = prompt_inputs["input_ids"].to(device)

    # 2. 训练循环
    for step in range(1, num_steps + 1):
        # 采样阶段
        model.eval()
        with torch.inference_mode():
            outputs = model.generate(
                input_ids=prompt_input_ids,
                max_new_tokens=1,
                num_return_sequences=group_size,
                do_sample=True,
                temperature=1.0,
                top_p=1.0,
                top_k=0,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            )

        # 提取新生成的 token 并解码
        new_token_ids = outputs[:, -1].tolist()
        rollout_responses = [
            tokenizer.decode([tid], skip_special_tokens=False) for tid in new_token_ids
        ]

        repeated_prompts = [prompt] * group_size
        repeated_ground_truths = [target] * group_size

        # 训练阶段
        model.train()
        loss, metadata = grpo_train_step(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            gradient_accumulation_steps=1,
            max_grad_norm=1.0,
            reward_fn=toy_reward_fn,
            repeated_prompts=repeated_prompts,
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_ground_truths,
            group_size=group_size,
        )

        # 打印日志
        if step in (1, 5, 10, 15, 20):
            print(
                f"Step {step:2d} | "
                f"avg_reward: {metadata.get('avg_reward', 0.0):.4f} | "
                f"grad_norm: {metadata.get('grad_norm', 0.0):.4f} | "
                f"token_entropy: {metadata.get('token_entropy', 0.0):.4f}"
            )

    # 3. 测量训练后概率
    after_prob = get_target_prob()

    print(f"\nBefore prob: {before_prob:.6f}")
    print(f"After prob:  {after_prob:.6f}")

    # 4. 断言概率上升
    assert after_prob > before_prob, "Model failed to increase probability of target token."
