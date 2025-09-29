import torch 
from typing import Callable
from transformers import AutoModel

"""
Code assisted by Cursor/Claude4
"""

FLOAT32_BYTES_N: int = 4
ADAMW_PARAMS_N: int = 2

def _check_layer_params(layer_data: torch.Tensor,
                        layer_name: str
                        ) -> int:
    """
    For the given layer, determine how many bytes would be needed
    to store this layer when trained with OSFT
    """
    # If the layer is trainable, i.e. it contains any of these terms,
    # then we will need to store a SVD decomposition of the layer in memory.
    TARGET_TERMS: list[str] = ['self_attn.q_proj',
                            'self_attn.k_proj',
                            'self_attn.v_proj',
                            'self_attn.o_proj',
                            'mlp.gate_proj',
                            'mlp.up_proj',
                            'mlp.down_proj']
    for term in TARGET_TERMS:
        if layer_name.find(term) > -1 and layer_name.find('weight') > -1:
            if layer_data.dim() < 2:
                layer_data = layer_data.unsqueeze(0)
            U_bytes_n: int = layer_data.shape[0] * layer_data.shape[0] 
            S_bytes_n: int  = layer_data.shape[0] * layer_data.shape[1]
            V_bytes_n: int = layer_data.shape[1] * layer_data.shape[1]
            return (U_bytes_n + S_bytes_n + V_bytes_n) * FLOAT32_BYTES_N

    # If not, we'll only be storing the layer itself in memory. 
    return layer_data.numel() * FLOAT32_BYTES_N


def _calc_osft_params(model: torch.nn.Module) -> int:
    """
    Iterate through the layers in this model and determine how
    many bytes would be needed to store this model when trained with OSFT
    """
    total_bytes: int = 0
    for layer_name in model.state_dict().keys():
        total_bytes += _check_layer_params(model.state_dict()[layer_name],
                                            layer_name)
    return total_bytes


def memory_estimator(
    num_gpus: int,
    gpu_memory: int,
    model_path: str,
    effective_batch_size: int | None = None,
    max_seq_len: int | None = None,
    max_tokens_per_gpu: int | None = 16384,
    osft: bool = False
    ) -> tuple[int, int, int]:

    """
    Calculate the memory needed to fine tune the given model for the 
    given hyperparameters. After that, determine how possible it is for
    the given hardware to run the model, and note how much more memory
    is needed to make the training more feasible. 

    Note that this estimate assumes training_hub will be used, 
    in which all data types are float32 and the optimizer is always AdamW.

    Args:
        num_gpus: Number of GPUs to use for training (default: 8)
        gpu_memory: The VRAM of each GPU in bytes (default: 24000000000)
        model_path: Path to the model to fine-tune, namely a HuggingFace
                    model path (default: "Qwen/Qwen2.5-7B-Instruct")
        effective_batch_size: Effective batch size for training, typically
                              not as important as max_tokens_per_gpu
        max_seq_len: Maximum sequence length of dataset samples 
        max_tokens_per_gpu: The maximum number of tokens that can be placed
                            on a single GPU during each mini-batch.
        using_osft: If set to True, calculate the memory usage assuming the
                    model is trained via OSFT. If False, assume SFT. 

    Return:
        lower_bound (int): The lower bound of the memory usage (in bytes)
        expected (int): The expected amount of memory usage (in bytes)
        upper_bound (int): The upper bound of the memory usage (in bytes)
    """

    # TODO: Validate these estimations empirically 

    # Load model directly
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, dtype="auto")

    # Determine parameters needed for calculations
    num_params: int = model.num_parameters(only_trainable=False)
    num_trainable_params: int = model.num_parameters(only_trainable=True)

    # TODO: Find a more universal way to obtain these values...
    num_layers: int = model.config.num_hidden_layers
    hidden_size: int = model.config.hidden_size    

    # The number of tokens on each GPU will be bounded by either
    # The largest number of tokens in a batch (divided by # GPUs)
    # *or* the value of max_tokens_per_gpu
    if effective_batch_size is None or max_seq_len is None:
        tokens_per_gpu: int = max_tokens_per_gpu
    elif max_tokens_per_gpu is None:
        tokens_per_gpu: int = effective_batch_size * max_seq_len / num_gpus
    else:
        tokens_per_gpu: int = min(max_tokens_per_gpu,
                             effective_batch_size * max_seq_len / num_gpus)

    # Calculate the amount of VRAM needed to fine-tune the provided model
    if osft:
        gpu_vram_par: int = _calc_osft_params(model) / num_gpus
    else:
        gpu_vram_par: int = num_params * FLOAT32_BYTES_N / num_gpus
    gpu_vram_opt: int = FLOAT32_BYTES_N * num_trainable_params * ADAMW_PARAMS_N / num_gpus
    gpu_vram_grad: int = FLOAT32_BYTES_N * num_trainable_params / num_gpus

    # NOTE: Typically, K is somewhere in [10, 30], but some research suggests that K=17.
    # TODO: Add papers
    gpu_vram_act_low: int = tokens_per_gpu * FLOAT32_BYTES_N * 10 * num_layers * hidden_size
    gpu_vram_act_high: int = tokens_per_gpu * FLOAT32_BYTES_N * 30 * num_layers * hidden_size
    gpu_vram_act_mid: int = tokens_per_gpu * FLOAT32_BYTES_N * 17 * num_layers * hidden_size

    # Lambda functions to calculate the total amount of VRAM and misc overhead
    calc_subtotal: Callable[[int], int] = \
        lambda act_value: gpu_vram_par + gpu_vram_opt + gpu_vram_grad + act_value
    calc_overhead: Callable[[int, int], int] = \
        lambda tot_val, act_val: tot_val - gpu_vram_par - gpu_vram_opt - gpu_vram_grad - act_val

    # Perform those calculations using the lambdas!
    gpu_vram_total_high: int = int(1.3 * calc_subtotal(gpu_vram_act_high))
    gpu_vram_total_low: int = int(calc_subtotal(gpu_vram_act_low) + 1073741824*2)
    gpu_vram_total_mid: int = int(1.2 *  calc_subtotal(gpu_vram_act_mid))
    extra_low: int = calc_overhead(gpu_vram_total_low, gpu_vram_act_low)
    extra_high: int = calc_overhead(gpu_vram_total_high, gpu_vram_act_high)

    # Helper lambda to do the rounding when printing 
    rounded_helper = lambda value : str(round(value / 1073741824, 1))

    def print_overall_stats():
        """
        Print out a breakdown of the estimated memory requirements
        """
        print("Estimations for " + model_path + ":\n")

        print("The expected amount of memory needed to run this model is about " + 
                rounded_helper(gpu_vram_total_mid * num_gpus) + " GB") 
        print("The lower and upper bounds are " + 
                rounded_helper(gpu_vram_total_low * num_gpus)  + " - " + 
                rounded_helper(gpu_vram_total_high * num_gpus) + " GB") 
        print("If you have " +
                str(num_gpus) +
                " GPUs, you will need about " +
                rounded_helper(gpu_vram_total_mid) +
                " GB, with bounds of " +
                rounded_helper(gpu_vram_total_low) +
                " - " +
                rounded_helper(gpu_vram_total_high) +
                " GB per GPU")
        print("Each GPU will need about " +
                rounded_helper(gpu_vram_par) + " GB to store the model parameters")
        print("Each GPU will need " +
                rounded_helper(gpu_vram_opt) + " GB to store the optimizer states")
        print("Each GPU will need " +
                rounded_helper(gpu_vram_grad) + " GB to store the gradients")
        print("Each GPU will need " +
                rounded_helper(gpu_vram_act_low) + " - " +
                rounded_helper(gpu_vram_act_high) + " GB to store the activations")
        print("The remaining " +
                rounded_helper(extra_low) + " - " + 
                rounded_helper(extra_high) + " GB is estimated overhead")

    # TODO: Can we fit in param recommendations?

    def print_wont_work():
        """
        Print this out if the GPU memory is below the lower bound
        """
        print("The proposed training setup is impossible for your hardware.")
        print("To reach the lower bound of memory requirements, you will need " + 
                rounded_helper(gpu_vram_total_low - gpu_memory) + " more GB of memory.")
        print("You should use GPUs with " +
                rounded_helper(gpu_vram_total_mid - gpu_memory) +
                " more GB of memory to reach the likely estimated memory requirements.")
        print("But ideally, you should use GPUs with " +
                rounded_helper(gpu_vram_total_high - gpu_memory) +
                " more GB of memory to reach the upper bound of memory requirements.")


    def print_likely_wont_work():
        """
        Print this out if the GPU memory is below the estimated requirement
        but above the lower bound
        """
        print("The proposed training setup is not recommended for your hardware.")
        print("You should use GPUs with at least " +
                rounded_helper(gpu_vram_total_mid - gpu_memory) +
                " more GB of memory to reach the likely estimated memory requirements.")
        print("But ideally, you should use GPUs with " +
                rounded_helper(gpu_vram_total_high - gpu_memory) +
                " more GB of memory to reach the upper bound of memory requirements.")


    def print_might_work():
        """
        Print this out if the GPU memory is above the estimated requirement
        but below the upper bound
        """
        print("The proposed training setup might work for your hardware.")
        print("Ideally, you should use GPUs with " +
                rounded_helper(gpu_vram_total_high - gpu_memory) +
                " more GB of memory to reach the upper bound of memory requirements.")

    def print_will_work():
        """
        Print this out if the GPU memory is above the upper bound
        """
        print("The proposed training setup will work for your hardware!")

    # Print out the recommendations based on the calculated memory requirements
    # and the provided GPU memory
    print_overall_stats()
    if gpu_vram_total_high <= gpu_memory: print_will_work()
    elif gpu_vram_total_mid <= gpu_memory: print_might_work()
    elif gpu_vram_total_low <= gpu_memory: print_likely_wont_work()
    else: print_wont_work()

    # Return the lower bound, estimated value, and upper bound 
    return gpu_vram_total_low, gpu_vram_total_mid, gpu_vram_total_high
