import random
import string
import math  # 用于指数函数计算

def random_insert_updated(text, insert_pct):
    """Randomly insert new chars into text after selected characters."""
    num_inserts = int(len(text) * insert_pct)
    indices = random.sample(range(len(text)), num_inserts)
    for idx in sorted(indices, reverse=True):
        new_char = random.choice(string.printable)
        text = text[:idx + 1] + new_char + text[idx + 1:]
    return text


def random_swap_updated(text, swap_pct):
    """Randomly swap chars within the text with new characters."""
    num_swaps = int(len(text) * swap_pct)
    indices = random.sample(range(len(text)), num_swaps)
    for i in indices:
        new_char = random.choice(string.printable)
        text = text[:i] + new_char + text[i+1:]
    return text


def random_patch(text, patch_pct):
    """Replace a random contiguous patch."""
    patch_len = int(len(text) * patch_pct)
    start_idx = random.randint(0, len(text)-patch_len)
    patch_str = ''.join(random.choice(string.printable) for _ in range(patch_len))
    text = text[:start_idx] + patch_str + text[start_idx+patch_len:]
    return text


def adaptive_perturb_pct(text, base_pct, min_len=10, max_len=1000):
    """Adapt perturbation percentage based on text length."""
    text_len = len(text)
    if text_len <= min_len:
        return base_pct / 2
    elif text_len >= max_len:
        return base_pct * 2
    else:
        return base_pct


def role_playing_perturb(T, message, perturb_func):
    """
    Function that takes in T (reputation score), message, and prompt,
    applies the chosen perturbation function based on T, and returns perturbed versions of message and prompt.
    """
    # 计算扰动比例，假设根据 T 的指数衰减来计算
    k = 1 #衰减系数
    q_min = 0
    q_max = 0.4
    perturb_pct =(q_max - q_min) * math.exp(-T * k)

    # 对 message 进行扰动
    perturbed_message = perturb_func(message, perturb_pct)

    # 对 prompt 进行扰动 perturbed_prompt = perturb_func(prompt, perturb_pct)

    # 返回扰动后的 message 和 prompt
    return perturbed_message

'''
# 示例调用
T = 2  # 声誉值
message = "Hello, how are you?"
prompt = "System prompt: Respond politely."

# 你可以选择 perturb_func 是 random_insert_updated、random_swap_updated 或 random_patch
perturb_func = random_patch  # 示例选择了 random_patch 方法

result = role_playing_perturb(T, message, prompt, perturb_func)
print(result)
'''