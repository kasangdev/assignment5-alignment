import torch


def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer):
    prompt_ids = []
    for prompt_str in prompt_strs:
        prompt_ids.append(tokenizer.encode(prompt_str, add_special_tokens=False))

    output_ids = []
    for output_str in output_strs:
        output_ids.append(tokenizer.encode(output_str, add_special_tokens=False))

    combined_ids = []
    prompt_lengths = []
    for prompt_id, output_id in zip(prompt_ids, output_ids):
        combined_ids.append(prompt_id + output_id)
        prompt_lengths.append(len(prompt_id))

    max_length = 0
    for ids in combined_ids:
        max_length = max(max_length, len(ids))
    pad_token_id = tokenizer.pad_token_id

    padded_ids = []
    for ids in combined_ids:
        num_padding = max_length - len(ids)
        padded_ids.append(ids + [pad_token_id] * num_padding)

    full_response_masks = []
    for ids, length in zip(combined_ids, prompt_lengths):
        full_response_masks.append([0] * length + [1] * (len(ids) - length) + [0] * (max_length - len(ids)))

    padded_ids_tensor = torch.tensor(padded_ids, dtype=torch.long)
    full_response_masks_tensor = torch.tensor(full_response_masks, dtype=torch.bool)

    input_ids = padded_ids_tensor[:, :-1]
    labels = padded_ids_tensor[:, 1:]
    response_mask = full_response_masks_tensor[:, 1:]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }
