# Runtime Example Measurements

Below are example wall clock measurements for how long you can expect a model to take to be fine-tuned. 

For disclosure: Please note all bar plots shown in this document are based on code primarily generated via Cursor/Claude 4.

## Experiment notes
- The experiments were conducted by using the default settings provided by `sft_granite_example.py` and `osft_granite_example.py`
- **Models**: Two models were tested, **Granite 3.3 8B**, and **Granite 4 Tiny Preview** (a Mixture-of-Experts model that also has 8B Parameters)
- **Hardware**: Two different hardware configurations were tested, a server with **8x A100s**, and an Openshift cluster with **8x H100s**. 
- **Datasets**: Two datasets were tested, a simple dataset in TableGPT and a much larger and longer dataset in Bespoke-Stratos-17k.
    - Please note that both datasets were obtained by downloading the dataset from HuggingFace and then extracting the .jsonl file. 
- All experiments were run for the first full epoch two times, with the displayed time being the average of the two times. 
    - The variation between the two runs was negligible, never more than 6 seconds. 
- The time measurement is calculated by using the timestamps logged during the training process in the above scripts
- By default, OSFT makes use of Liger Kernels to improve memory usage and runtime. However, as of Nov 7th 2025, Liger Kernels currently don't have built-in support for Granite 4
    - As a result, the script was modified for allow Liger Kernels to be disabled for certain experiments
    - The tables will be updated once support for Liger Kernels is added.  

## Full Results

| Hardware | Training Type    | Model       | Dataset   | Train Time per Epoch (HH:MM:SS) |
| ---------| ---------------- | ----------- | --------- | ------------------------------- |
| 8x A100s | SFT              | Granite 3.3 | Table-GPT | 00:10:01                        | 
| 8x A100s | SFT              | Granite 3.3 | Bespoke   | 01:17:02                        | 
| 8x A100s | SFT              | Granite 4   | Table-GPT | 00:07:35                        | 
| 8x A100s | SFT              | Granite 4   | Bespoke   | 00:42:48                        | 
| 8x A100s | OSFT             | Granite 3.3 | Table-GPT | 00:36:09                        | 
| 8x A100s | OSFT             | Granite 3.3 | Bespoke   | 00:58:39                        | 
| 8x A100s | OSFT (No Liger)  | Granite 3.3 | Table-GPT | 00:37:32                        | 
| 8x A100s | OSFT (No Liger)  | Granite 3.3 | Bespoke   | 01:00:48                        | 
| 8x A100s | OSFT (No Liger)  | Granite 4   | Table-GPT | 00:14:11                        | 
| 8x A100s | OSFT (No Liger)  | Granite 4   | Bespoke   | 00:23:01                        | 
| 8x H100s | SFT              | Granite 3.3 | Table-GPT | 00:04:35                        | 
| 8x H100s | SFT              | Granite 3.3 | Bespoke   | | 
| 8x H100s | SFT              | Granite 4   | Table-GPT | | 
| 8x H100s | SFT              | Granite 4   | Bespoke   | | 
| 8x H100s | OSFT             | Granite 3.3 | Table-GPT | | 
| 8x H100s | OSFT             | Granite 3.3 | Bespoke   | | 
| 8x H100s | OSFT (No Liger)  | Granite 3.3 | Table-GPT | | 
| 8x H100s | OSFT (No Liger)  | Granite 3.3 | Bespoke   | | 
| 8x H100s | OSFT (No Liger)  | Granite 4   | Table-GPT | | 
| 8x H100s | OSFT (No Liger)  | Granite 4   | Bespoke   | | 

## Graph: A100 vs H100

### Tabled differences

## Graph: TableGPT vs Bespoke

Reference-style: 
![A graph comparing TableGPT with Bespoke][logo]

[logo]: https://github.com/Red-Hat-AI-Innovation-Team/training_hub/tree/main/examples/docs/example.png ""


### Tabled differences

## Graph: Granite 3.3 vs Granite 4

### Tabled differences

