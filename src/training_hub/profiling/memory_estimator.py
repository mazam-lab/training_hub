from click import FLOAT
import torch 
from typing import Callable, Optional, override
from transformers import AutoModel
from transformers.models.perceiver.modeling_perceiver import PerceiverMultimodalPreprocessor

"""
Code assisted by Cursor/Claude4
"""

# TODO: Make this a class structure to make this more extendable
# TODO: Make a notebook 

FLOAT32_BYTES_N: int = 4
FLOAT16_BYTES_N: int = 2
FLOAT8_BYTES_N: int = 1
ADAMW_PARAMS_N: int = 2

# Helper lambda to do the rounding when printing 
ROUNDER = lambda value : str(round(value / 1073741824, 1))


class BasicEstimator:
    """
    Args:
            self.num_gpus: Number of GPUs to use for training (default: 8)
            gpu_memory: The VRAM of each GPU in bytes (default: 85899345920 for 80 GB)
            model_path: Path to the model to fine-tune, namely a HuggingFace
                        model path (default: "ibm-granite/granite-3.3-8b-instruct")
            self.effective_batch_size: The number of samples in a minibatch that the model has to see before backpropping.
            max_seq_len: Maximum sequence length of dataset samples 
            max_tokens_per_gpu: The maximum number of tokens that can be placed
                                on a single GPU during each mini-batch.
            using_osft: If set to True, calculate the memory usage assuming the
                        model is trained via OSFT. If False, assume SFT. 
    """

    def __init__(
        self,
        num_gpus: int = 8,
        gpu_memory: int = 85899345920,
        model_path: str = "ibm-granite/granite-3.3-8b-instruct",
        effective_batch_size: int = None,
        max_seq_len: int = None,
        max_tokens_per_gpu: int = None,
        use_liger: bool = False,
    ):
        self.num_gpus = num_gpus
        self.gpu_memory = gpu_memory
        self.model_path = model_path
        self.use_liger = use_liger
        
        # Load model directly
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)

        # Determine parameters needed for calculations
        self.num_params: int = self.model.num_parameters(only_trainable=False)
        self.num_trainable_params: int = self.model.num_parameters(only_trainable=True)

        # TODO: Use model._modules if the config can't give us the information we need
        # TODO: Handle the fact that gemma has that language model distinction -_-
        self.num_layers: int = self.model.config.num_hidden_layers
        self.hidden_size: int = self.model.config.hidden_size  

        # The number of tokens on each GPU will be bounded by either
        # The largest number of tokens in a batch (divided by # GPUs)
        # *or* the value of max_tokens_per_gpu
        # TODO: Should we try to bound this based on the expected dataset's lengths?
        if effective_batch_size is None or max_seq_len is None:
            self.tokens_per_gpu: int = max_tokens_per_gpu
        elif max_tokens_per_gpu is None:
            self.tokens_per_gpu: int = effective_batch_size * max_seq_len / self.num_gpus
        else:
            self.tokens_per_gpu: int = min(max_tokens_per_gpu, effective_batch_size * max_seq_len / self.num_gpus)

        # This is a scalar that's applied during the output memory calculation
        self.output_constant = 8/3
        self.main_dtype_bytes = FLOAT32_BYTES_N
        self.opt_params = ADAMW_PARAMS_N

    def _calc_model_params(self):
        return self.num_params * self.main_dtype_bytes / self.num_gpus

    def _calc_gradients(self):
        return self.main_dtype_bytes * self.num_trainable_params / self.num_gpus

    def _calc_optimizer(self):
        return self.main_dtype_bytes * self.num_trainable_params * self.opt_params / self.num_gpus

    def _calc_intermediate_activations(self):
        return (self.tokens_per_gpu * self.main_dtype_bytes  * self.num_layers * self.hidden_size)

    def _calc_outputs(self):
        # Liger removes the need to store the activated outputs of the model
        # If we're not using Liger, factor this in.
        if not self.use_liger:
            # TODO: This constant is how many times the output tensor gets
            # repeated empirically. Why is this happening...?
            return (self.tokens_per_gpu * self.main_dtype_bytes * self.model.embed_tokens.num_embeddings) * self.output_constant
        else:
            return 0

    def _calc_additional(self, **kwargs):
        return 0

    def _apply_overhead(self, subtotal):
        # Perform those calculations using the lambdas!
        gpu_vram_total_low: int = int(subtotal)
        gpu_vram_total_mid: int = int(1.1 *  subtotal)
        gpu_vram_total_high: int = int(1.3 * subtotal)
        extra_low: int = max(gpu_vram_total_low - subtotal, 0)
        extra_mid: int = gpu_vram_total_mid - subtotal
        extra_high: int = gpu_vram_total_high - subtotal
        return(gpu_vram_total_low, gpu_vram_total_mid, gpu_vram_total_high), (extra_low, extra_mid, extra_high)


    def _print_results(self, results, overhead, gpu_vram_par, gpu_vram_opt, gpu_vram_grad, gpu_vram_act, gpu_vram_outputs, gpu_vram_additional):
            """
            Print out a breakdown of the estimated memory requirements
            """
            print("Estimations for " + self.model_path + ":\n\n")

            print("Summary:")
            print("The expected amount of memory needed to run this model is about " + 
                    ROUNDER(results[1] * self.num_gpus) + " GB") 
            print("The lower and upper bounds are " + 
                    ROUNDER(results[0] * self.num_gpus)  + " - " + 
                    ROUNDER(results[2] * self.num_gpus) + " GB") 
            print("If you have " + str(self.num_gpus) + " GPUs, you will need about " + \
                    ROUNDER(results[1]) + " GB, with bounds of " +
                    ROUNDER(results[0]) + " - " + ROUNDER(results[2]) + " GB per GPU")
            print("\n")

            print("Component Breakdown:")
            print("Each GPU will need " + ROUNDER(gpu_vram_par) + " GB to store the model parameters")
            print("Each GPU will need " + ROUNDER(gpu_vram_opt) + " GB to store the optimizer states")
            print("Each GPU will need " + ROUNDER(gpu_vram_grad) + " GB to store the gradients")
            print("Each GPU will need " + ROUNDER(gpu_vram_act) + " GB to store the intermediate activations")
            if self.use_liger:
                print("Since Liger Kernels are being used, no additional memory is needed to store the outputs")
            else:
                print("Each GPU will need " + ROUNDER(gpu_vram_outputs) + " GB to store the outputs")
            if gpu_vram_additional > 0:
                print("This method also requires each GPU to use an additional " + ROUNDER(gpu_vram_additional) + " GB")
            print("Up to " + ROUNDER(overhead[2]) + " GB can be expected as overhead")


    def _print_tips(self, results):

        # TODO: Can we fit in param recommendations?

        min_message = "Minimum extra memory required (to reach the\nlow bound of memory requirements): " + \
            ROUNDER(results[0] - self.gpu_memory) + " GB"
        mid_message = "Recommended amount of extra memory (to reach the\nlikely estimated memory requirements): " + \
            ROUNDER(results[1] - self.gpu_memory) + " GB"
        max_message = "Ideal amount of extra memory required (to reach the\nupper bound of memory requirements): " + \
            ROUNDER(results[2] - self.gpu_memory) + " GB"

        print("\nDecision:")
        
        if results[2] <= self.gpu_memory:
            print("The proposed training setup should work for your hardware.\n")
        elif results[1] <= self.gpu_memory: 
            print("The proposed training setup will likely work for your hardware.\n")
            print(max_message)
        elif results[0] <= self.gpu_memory:
            print("The proposed training setup may work but isn't recommended for your hardware.\n")
            print(mid_message)
            print(max_message)
        else: 
            print("The proposed training setup is impossible for your hardware.\n\n")
            print(min_message)
            print(mid_message)
            print(max_message)


    def estimate(
        self,
    ) -> tuple[int, int, int]:

        """
        Calculate the memory needed to fine tune the given model for the 
        given hyperparameters. After that, determine how possible it is for
        the given hardware to run the model, and note how much more memory
        is needed to make the training more feasible. 

        Note that this estimate assumes training_hub will be used, 
        in which all data types are float32 and the optimizer is always AdamW.

        Return:
            lower_bound (int): The lower bound of the memory usage (in bytes)
            expected (int): The expected amount of memory usage (in bytes)
            upper_bound (int): The upper bound of the memory usage (in bytes)
        """  

        # Calculate the amount of VRAM needed to fine-tune the provided model
        # TODO: Validate this            
        gpu_vram_par = self._calc_model_params()

        # VALIDATED FOR ACCURACY:
        gpu_vram_opt: int = self._calc_optimizer()
        gpu_vram_grad: int = self._calc_gradients()

        # The VRAM needed to store the intermediate activations of the model
        # TODO: Validate this
        gpu_vram_act: int = self._calc_intermediate_activations()
        gpu_vram_outputs: int = self._calc_outputs()
        gpu_vram_additional: int = self._calc_additional()

        subtotal: int = gpu_vram_par + gpu_vram_opt + gpu_vram_grad + gpu_vram_act + gpu_vram_outputs + gpu_vram_additional
        results, overhead = self._apply_overhead(subtotal)

        # Print out the recommendations based on the calculated memory requirements
        # and the provided GPU memory
        self._print_results(results, overhead, gpu_vram_par, gpu_vram_opt, gpu_vram_grad, gpu_vram_act, gpu_vram_outputs, gpu_vram_additional)
        self._print_tips(results)

        # Return the lower bound, estimated value, and upper bound 
        return results 



class OSFTEstimator(BasicEstimator):

    @override
    def __init__(
        self,
        num_gpus: int = 8,
        gpu_memory: int = 85899345920,
        model_path: str = "ibm-granite/granite-3.3-8b-instruct",
        effective_batch_size: int = None,
        max_seq_len: int = None,
        max_tokens_per_gpu: int = None,
        use_liger: bool = False,
    ):
        super().__init__(num_gpus, gpu_memory, model_path, effective_batch_size, max_seq_len, max_tokens_per_gpu, use_liger)
        self.output_constant = 7/3


    def _check_layer_params(
                        self,
                        layer_name: str
                    ) -> int:
        """
        For the given layer, determine how many bytes would be needed
        to store this layer when trained with OSFT
        """
        # If the layer is trainable, i.e. it contains any of these terms,
        # then we will need to store a SVD decomposition of the layer in memory.

        # TODO: Okay, so according to Nikhil, we don't need the S tensor. 
        # Also, check through your profiles to see if you can hammer some stronger estimates
        # Maybe we should account for the fact that these matrices sometimes get offloaded
        # during the activation stage...?

        TARGET_TERMS: list[str] = ['self_attn.q_proj',
                                'self_attn.k_proj',
                                'self_attn.v_proj',
                                'self_attn.o_proj',
                                'mlp.gate_proj',
                                'mlp.up_proj',
                                'mlp.down_proj']
        layer_data = self.model.state_dict()[layer_name]
        if layer_data.dim() < 2:
            return layer_data.numel() * FLOAT32_BYTES_N
        for term in TARGET_TERMS:
            if layer_name.find(term) > -1 and layer_name.find('weight') > -1:
                U_bytes_n: int = layer_data.shape[0] * layer_data.shape[0] 
                S_bytes_n: int  = layer_data.shape[0] if layer_data.shape[0] < layer_data.shape[1] else layer_data.shape[1]
                V_bytes_n: int = layer_data.shape[1] * layer_data.shape[1]
                return (U_bytes_n + S_bytes_n + V_bytes_n) * FLOAT32_BYTES_N

        # If not, we'll only be storing the layer itself in memory. 
        byte_val = layer_data.numel() * FLOAT32_BYTES_N
        return byte_val


    def _calc_osft_params(self) -> int:
        """
        Iterate through the layers in this model and determine how
        many bytes would be needed to store this model when trained with OSFT
        """
        total_bytes: int = 0
        for layer_name in self.model.state_dict().keys():
            total_bytes += self._check_layer_params(layer_name)
        return total_bytes

    @override
    def _calc_model_params(self):
         temp_val =  super()._calc_model_params()
         return temp_val + (self._calc_osft_params() / self.num_gpus)


def estimate(
        estimator_type: str = "sft",
        num_gpus: int = 8,
        gpu_memory: int = 85899345920,
        model_path: str = "ibm-granite/granite-3.3-8b-instruct",
        effective_batch_size: int = None,
        max_seq_len: int = None,
        max_tokens_per_gpu: int = None,
        use_liger: bool = False,
    ):
    """
    Convenience function for performing estimation
    """
    if estimator_type == "osft":
        estimator = OSFTEstimator(num_gpus, gpu_memory, model_path, effective_batch_size, max_seq_len, max_tokens_per_gpu, use_liger)
    else:
        estimator = BasicEstimator(num_gpus, gpu_memory, model_path, effective_batch_size, max_seq_len, max_tokens_per_gpu, use_liger)
    return estimator.estimate()
