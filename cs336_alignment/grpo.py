import torch
from cs336_alignment.data import tokenize_prompt_and_output


def get_response_log_probs(
        model,
        input_ids,
        labels,
        return_token_entropy=False,
):
    outputs = model(input_ids=input_ids)
    logits = outputs.logits  # [N, L, V]

    all_log_probs = torch.log_softmax(logits, dim=-1)  # [N, L, V]

    log_probs = torch.gather(
        all_log_probs,
        dim=-1,
        index=labels.unsqueeze(-1).long(),
    ).squeeze(-1)  # [N, L]

    result = {"log_probs": log_probs}

    if return_token_entropy:
        probs = all_log_probs.exp()  # [N, L, V]
        token_entropy = -(probs * all_log_probs).sum(dim=-1)  # [N, L]
        result["token_entropy"] = token_entropy

    return result


def compute_rollout_rewards(
        reward_fn,
        rollout_responses,
        repeated_ground_truths,
):
    reward_values = []
    format_reward_values = []
    answer_reward_values = []

    for response, ground_truth in zip(rollout_responses, repeated_ground_truths):
        scores = reward_fn(response, ground_truth)
        reward_values.append(float(scores["reward"]))
        format_reward_values.append(float(scores.get("format_reward", 0.0)))
        answer_reward_values.append(float(scores.get("answer_reward", 0.0)))

    raw_rewards = torch.tensor(reward_values, dtype=torch.float32)

    n = len(reward_values) if reward_values else 1
    metadata = {
        "avg_reward": float(sum(reward_values) / n),
        "avg_format_reward": float(sum(format_reward_values) / n),
        "avg_answer_reward": float(sum(answer_reward_values) / n),
    }
    return raw_rewards, metadata


def compute_group_normalized_rewards(
        raw_rewards,
        group_size,
        baseline="mean",
        advantage_eps=1e-6,
        advantage_normalizer="std",
):
    grouped_rewards = raw_rewards.view(-1, group_size)  # [B, G]
    group_means = grouped_rewards.mean(dim=1, keepdim=True)  # [B, 1]

    # 提前计算 std 供 metadata 使用
    group_stds = grouped_rewards.std(dim=1, keepdim=True)  # [B, 1]

    if baseline == "mean":
        baseline_values = group_means
    elif baseline == "none":
        baseline_values = torch.zeros_like(group_means)
    else:
        raise ValueError(f"Unsupported baseline: {baseline}")

    if advantage_normalizer == "std":
        normalizer = group_stds + advantage_eps
    elif advantage_normalizer == "mean":
        normalizer = group_means + advantage_eps
    elif advantage_normalizer == "none":
        normalizer = torch.ones_like(group_means)
    else:
        raise ValueError(f"Unsupported advantage_normalizer: {advantage_normalizer}")

    grouped_advantages = (grouped_rewards - baseline_values) / normalizer  # [B, G]
    advantages = grouped_advantages.reshape(-1)  # [N]

    metadata = {
        "avg_reward": float(raw_rewards.mean().item()) if raw_rewards.numel() > 0 else 0.0,
        "avg_group_std": float(group_stds.mean().item()) if group_stds.numel() > 0 else 0.0,
    }
    return advantages, metadata


def compute_policy_gradient_loss(
        raw_rewards_or_advantages,
        policy_log_probs,
        importance_reweighting_method="none",
        old_log_probs=None,
        cliprange=None,
        response_mask=None,
):
    # 统一将 weights 整理为 [N, 1] 以适配广播
    if raw_rewards_or_advantages.dim() == 1:
        weights = raw_rewards_or_advantages.unsqueeze(-1)
    else:
        weights = raw_rewards_or_advantages

    metadata = {}

    if importance_reweighting_method == "none":
        per_token_policy_gradient_loss = -weights * policy_log_probs
        return per_token_policy_gradient_loss, metadata

    if old_log_probs is None:
        raise ValueError(f"old_log_probs is required for {importance_reweighting_method}")
    if cliprange is None and importance_reweighting_method in ("grpo", "gspo"):
        raise ValueError(f"cliprange is required for {importance_reweighting_method}")

    # 计算 token 级比率
    log_ratio = policy_log_probs - old_log_probs  # [N, L]
    ratio = torch.exp(log_ratio)  # [N, L]

    if importance_reweighting_method == "noclip":
        unclipped_objective = ratio * weights
        per_token_policy_gradient_loss = -unclipped_objective

    elif importance_reweighting_method == "grpo":
        # 将 clipped_ratio 的计算移入该分支，避免 noclip 时报错
        clipped_ratio = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)  # [N, L]

        unclipped_objective = ratio * weights
        clipped_objective = clipped_ratio * weights
        per_token_policy_gradient_loss = -torch.min(unclipped_objective, clipped_objective)

    elif importance_reweighting_method == "gspo":
        # 按照 mask 计算每条序列的平均 log_ratio
        if response_mask is not None:
            mask_sum = response_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
            sequence_log_ratio = (log_ratio * response_mask).sum(dim=-1, keepdim=True) / mask_sum
        else:
            sequence_log_ratio = log_ratio.mean(dim=-1, keepdim=True)

        sequence_ratio = torch.exp(sequence_log_ratio)  # [N, 1]
        clipped_sequence_ratio = torch.clamp(sequence_ratio, 1.0 - cliprange, 1.0 + cliprange)  # [N, 1]

        unclipped_objective = sequence_ratio * weights  # [N, 1]
        clipped_objective = clipped_sequence_ratio * weights  # [N, 1]

        # 将 [N, 1] 的序列目标广播回 [N, L] 以返回逐 token 损失
        seq_loss = -torch.min(unclipped_objective, clipped_objective)  # [N, 1]
        per_token_policy_gradient_loss = seq_loss.expand_as(policy_log_probs)  # [N, L]

    else:
        raise ValueError(f"Unsupported importance_reweighting_method: {importance_reweighting_method}")

    return per_token_policy_gradient_loss, metadata


def aggregate_loss_across_microbatch(
        per_token_policy_gradient_loss,
        mask,
        loss_normalization="sequence",
        normalization_constant=None,
):
    mask = mask.float()
    masked_loss = per_token_policy_gradient_loss * mask  # [N, L]

    if loss_normalization == "sequence":
        token_counts = mask.sum(dim=-1)  # [N]
        safe_counts = token_counts.clamp_min(1.0)
        sequence_loss = masked_loss.sum(dim=-1) / safe_counts  # [N]
        loss = sequence_loss.mean()
    elif loss_normalization == "batch":
        total_tokens = mask.sum().clamp_min(1.0)
        loss = masked_loss.sum() / total_tokens
    elif loss_normalization == "constant":
        if normalization_constant is None:
            raise ValueError(
                "normalization_constant is required when loss_normalization='constant'"
            )
        loss = masked_loss.sum() / float(normalization_constant)
    else:
        raise NotImplementedError(
            f"Unsupported loss_normalization: {loss_normalization}"
        )

    return loss


def grpo_train_step(
        model,
        tokenizer,
        optimizer,
        gradient_accumulation_steps,
        max_grad_norm,
        reward_fn,
        repeated_prompts,
        rollout_responses,
        repeated_ground_truths,
        group_size,
        baseline="mean",
        advantage_eps=1e-6,
        advantage_normalizer="std",
        importance_reweighting_method="none",
        old_log_probs=None,
        cliprange=None,
        loss_normalization="sequence",
        normalization_constant=None,
):
    device = next(model.parameters()).device
    total_examples = len(rollout_responses)

    # ---------- 第一阶段：整批数据 ----------
    raw_rewards, reward_metadata = compute_rollout_rewards(
        reward_fn,
        rollout_responses,
        repeated_ground_truths,
    )
    advantages, adv_metadata = compute_group_normalized_rewards(
        raw_rewards,
        group_size,
        baseline=baseline,
        advantage_eps=advantage_eps,
        advantage_normalizer=advantage_normalizer,
    )

    train_batch = tokenize_prompt_and_output(
        repeated_prompts,
        rollout_responses,
        tokenizer,
    )
    input_ids = train_batch["input_ids"].to(device)
    labels = train_batch["labels"].to(device)
    response_mask = train_batch["response_mask"].to(device)

    microbatch_size = max(
        1,
        -(-total_examples // gradient_accumulation_steps),
    )

    # ---------- 第二 / 三阶段：微批前向、反向、日志累加 ----------
    total_loss = 0.0
    entropy_sum = 0.0
    response_token_count = 0.0

    optimizer.zero_grad()

    for start in range(0, total_examples, microbatch_size):
        end = min(start + microbatch_size, total_examples)
        mb_size = end - start

        mb_input_ids = input_ids[start:end]
        mb_labels = labels[start:end]
        mb_mask = response_mask[start:end]
        mb_advantages = advantages[start:end].to(device)

        mb_old_log_probs = None
        if old_log_probs is not None:
            mb_old_log_probs = old_log_probs[start:end].to(device)

        log_prob_output = get_response_log_probs(
            model,
            mb_input_ids,
            mb_labels,
            return_token_entropy=True,
        )
        mb_log_probs = log_prob_output["log_probs"]
        mb_entropy = log_prob_output["token_entropy"]

        per_token_loss, _ = compute_policy_gradient_loss(
            mb_advantages,
            mb_log_probs,
            importance_reweighting_method=importance_reweighting_method,
            old_log_probs=mb_old_log_probs,
            cliprange=cliprange,
            response_mask=mb_mask,
        )

        mb_loss = aggregate_loss_across_microbatch(
            per_token_loss,
            mb_mask,
            loss_normalization=loss_normalization,
            normalization_constant=normalization_constant,
        )

        if loss_normalization == "constant":
            scaled_loss = mb_loss
        else:
            scaled_loss = mb_loss * (mb_size / total_examples)

        scaled_loss.backward()

        with torch.no_grad():
            total_loss += scaled_loss.detach().item()
            entropy_sum += (mb_entropy * mb_mask).sum().item()
            response_token_count += mb_mask.sum().item()

    # ---------- 第四阶段：更新模型 ----------
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_grad_norm if max_grad_norm is not None else float("inf"),
    )
    grad_norm_val = float(grad_norm.item())

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    # ---------- 第五阶段：返回 ----------
    token_entropy_avg = (
        entropy_sum / response_token_count
        if response_token_count > 0
        else 0.0
    )

    metadata = {
        **reward_metadata,
        **adv_metadata,
        "loss": total_loss,
        "grad_norm": grad_norm_val,
        "token_entropy": token_entropy_avg,
    }

    loss_tensor = torch.tensor(total_loss, dtype=torch.float32, device=device)
    return loss_tensor, metadata
