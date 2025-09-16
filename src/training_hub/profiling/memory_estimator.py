import torch 
from typing import Union, Optional, Callable
from transformers import AutoModel
from datasets import load_dataset

"""
Code assisted by Cursor/Claude4
"""

def handle_dtypes(given_dtype: torch.dtype,
                    default_bytes: int) -> int:
    """
    Helper function to calculate the number of bytes in a given dtype
    If given_dtype is None, default to using default_bytes instead. 
    """
    if given_dtype is None:
        return default_bytes
    else:
        given_dtype = str(given_dtype)
        return int(int(given_dtype.split('t')[-1]) / 8)


def check_layer_params(layer_data: torch.Tensor,
                        layer_name: str, 
                        modl_bytes_n: int,
                        osft_bytes_n: int) -> int:
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
            return (U_bytes_n + S_bytes_n + V_bytes_n) * osft_bytes_n

    # If not, we'll only be storing the layer itself in memory. 
    return layer_data.numel() * modl_bytes_n


def calc_osft_params(model: torch.nn.Module,
                    modl_bytes_n: int,
                    osft_bytes_n: int) -> int:
    """
    Iterate through the layers in this model and determine how
    many bytes would be needed to store this model when trained with OSFT
    """
    accumulated_bytes: int = 0
    for layer_name in model.state_dict().keys():
        accumulated_bytes += check_layer_params(model.state_dict()[layer_name],
                                                layer_name,
                                                modl_bytes_n,
                                                osft_bytes_n)
    return accumulated_bytes


def memory_estimator(
    num_gpus: int = 8,
    gpu_memory: int = 24000000000,
    model_path: str =  "RedHatAI/Phi-3-mini-128k-instruct-FP8", # "Qwen/Qwen2.5-7B-Instruct",
    effective_batch_size: Optional[int] = None,
    max_seq_len: Optional[int] = None,
    max_tokens_per_gpu: Optional[int] = 2048,
    optim_dtype: Optional[torch.dtype]=None,
    grad_dtype: Optional[torch.dtype]=None,
    activations_type: Optional[torch.dtype]=None,
    using_osft: Optional[bool] = False,
    osft_dtype: Optional[torch.dtype]=None,
    **kwargs,
    ) -> tuple[int, int, int]:

    """
    Calculate the memory needed to run the given model for the 
    given hyperparameters. After that, determine how possible it is for
    the given hardware to run the model, and note how much more memory
    is needed to make the training more feasible. 

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
        optim_dtype: Data type of the optimizer. If None, assume float32
        grad_dtype: Data type of the gradients. If None, assume float32
        activations_type: Data type of the activations. If None, assume float32
        using_osft: If set to True, calculate the memory usage assuming the
                    model is trained via OSFT. If False, assume SFT. 
        osft_dtype: Data type of the OSFT's SVD decomposition matrices.
                    If None, assume float32

    Return:
        lower_bound (int): The lower bound of the memory usage (in bytes)
        expected (int): The expected amount of memory usage (in bytes)
        upper_bound (int): The upper bound of the memory usage (in bytes)
    """

    # NOTE: So the backend is always using AdamW with mixed FP precision...
    # TODO: Validate these estimations empirically 

    # Load model directly
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, dtype="auto")

    # Determine parameters needed for calculations
    num_params: int = model.num_parameters(only_trainable=False)
    num_trainable_params: int = model.num_parameters(only_trainable=True)
    model_type: str = str(model.dtype)
    num_layers: int = model.config.num_hidden_layers
    hidden_size: int = model.config.hidden_size

    # Extract the model dtype from the config file
    # The other dtypes will either be an inputted type
    # *or* whatever type the model had
    modl_bytes_n: int = int(int(model_type.split('t')[-1]) / 8)
    opt_bytes_n: int = handle_dtypes(optim_dtype, modl_bytes_n)
    grad_bytes_n: int = handle_dtypes(grad_dtype, modl_bytes_n)
    act_bytes_n: int = handle_dtypes(activations_type, modl_bytes_n)
    if using_osft:
        osft_bytes_n: int = handle_dtypes(osft_dtype, modl_bytes_n)

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
    if using_osft:
        gpu_vram_par: int = calc_osft_params(model, modl_bytes_n, osft_bytes_n) / num_gpus
    else:
        gpu_vram_par: int = num_params * modl_bytes_n / num_gpus
    gpu_vram_opt: int = opt_bytes_n * num_trainable_params * 2 / num_gpus
    gpu_vram_grad: int = grad_bytes_n * num_trainable_params / num_gpus

    # NOTE: Typically, K is somewhere in [10, 30], but some research suggests that K=17.
    # TODO: Add papers
    gpu_vram_act_low: int = tokens_per_gpu * act_bytes_n * 10 * num_layers * hidden_size
    gpu_vram_act_high: int = tokens_per_gpu * act_bytes_n * 30 * num_layers * hidden_size
    gpu_vram_act_mid: int = tokens_per_gpu * act_bytes_n * 17 * num_layers * hidden_size

    # Lambda functions to calculate the total amount of VRAM and misc overhead
    calc_subtotal: Callable[[int], int] = \
        lambda act_value: gpu_vram_par + gpu_vram_opt + gpu_vram_grad + act_value
    calc_overhead: Callable[[int, int], int] = \
        lambda tot_val, act_val: tot_val - gpu_vram_par - gpu_vram_opt - gpu_vram_grad - act_val

    # Perform those calculations using the lambdas!
    gpu_vram_total_high: int = int(1.3 * calc_subtotal(gpu_vram_act_high))
    gpu_vram_total_low: int = int(calc_subtotal(gpu_vram_act_low) + 2000000000)
    gpu_vram_total_mid: int = int(1.2 *  calc_subtotal(gpu_vram_act_mid))
    extra_low: int = calc_overhead(gpu_vram_total_low, gpu_vram_act_low)
    extra_high: int = calc_overhead(gpu_vram_total_high, gpu_vram_act_high)
    extra_mid: int = calc_overhead(gpu_vram_total_mid, gpu_vram_act_mid)

    # Helper lambda to do the rounding when printing 
    rounded_helper = lambda value : str(round(value / 1000000000, 1))

    def print_overall_stats():
        """
        Print out a breakdown of the estimated memory requirements
        """
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
    print('\n==============================================\n')
    if gpu_vram_total_high <= gpu_memory:
        print_will_work()
    elif gpu_vram_total_mid <= gpu_memory:
        print_might_work()
    elif gpu_vram_total_low <= gpu_memory:
        print_likely_wont_work()
    else:
        print_wont_work()

    # Return the lower bound, estimated value, and upper bound 
    return gpu_vram_total_low, gpu_vram_total_mid, gpu_vram_total_high

memory_estimator()