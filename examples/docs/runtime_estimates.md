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
    - On the A100 machine, the variation between the two runs was negligible, never more than 6 seconds. 
    - The variation is a bit larger on the H100 machine, especially during the first run of a pod (the first result was discarded and reran if it varied significantly)
- The time measurement is calculated by using the timestamps logged during the training process in the above scripts
- By default, OSFT makes use of Liger Kernels to improve memory usage and runtime. However, as of Nov 7th 2025, Liger Kernels currently don't have built-in support for Granite 4
    - As a result, the script was modified for allow Liger Kernels to be disabled for certain experiments
    - The tables will be updated once support for Liger Kernels is added. 
- Some of these tests had the checkpointing hardcoded to be disabled in the script
    - This does not appear to impact runtime of the actual training loop, this was mostly done to conserve disk space

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
| 8x H100s | SFT              | Granite 3.3 | Bespoke   | 00:35:47                        | 
| 8x H100s | SFT              | Granite 4   | Table-GPT | 00:05:40                        | 
| 8x H100s | SFT              | Granite 4   | Bespoke   | 00:26:19                        | 
| 8x H100s | OSFT             | Granite 3.3 | Table-GPT | 00:46:04                        | 
| 8x H100s | OSFT             | Granite 3.3 | Bespoke   |                                 | 
| 8x H100s | OSFT (No Liger)  | Granite 3.3 | Table-GPT |                                 | 
| 8x H100s | OSFT (No Liger)  | Granite 3.3 | Bespoke   |                                 | 
| 8x H100s | OSFT (No Liger)  | Granite 4   | Table-GPT | 00:10:39                        | 
| 8x H100s | OSFT (No Liger)  | Granite 4   | Bespoke   | 00:16:46                        | 


## Graph: A100 vs H100

### Tabled differences

## Graph: TableGPT vs Bespoke

![A graph comparing TableGPT with Bespoke][logo]

[logo]: https://github.com/mazam-lab/training_hub/blob/main/examples/docs/example.png?raw=true ""


### Tabled differences

| Hardware | Training Type    | Model       | Table-GPT Time | Bespoke Time | Difference in Time | Multiplier Gain |
| -------- | ---------------- | ----------- | -------------- | ------------ | ------------------ | --------------- |
| 8x A100s | SFT              | Granite 3.3 | 00:10:01       | 01:17:02     | 01:07:01           | 7.69x           |
| 8x A100s | SFT              | Granite 4   | 00:07:35       | 00:42:48     | 00:35:13           | 5.64x           |
| 8x A100s | OSFT             | Granite 3.3 | 00:36:09       | 00:58:39     | 00:22:30           | 1.62x           |
| 8x A100s | OSFT (No Liger)  | Granite 3.3 | 00:37:32       | 01:00:48     | 00:23:16           | 1.62x           |
| 8x A100s | OSFT (No Liger)  | Granite 4   | 00:14:11       | 00:23:01     | 00:08:50           | 1.62x           |
| 8x H100s | SFT              | Granite 3.3 | 00:04:35       | 00:35:47     | 00:31:12           | 7.81x           |
| 8x H100s | SFT              | Granite 4   | 00:05:40       | 00:26:19     | 00:20:39           | 4.64x           |
| 8x H100s | OSFT             | Granite 3.3 |                |              |                    |                 |
| 8x H100s | OSFT (No Liger)  | Granite 3.3 |                |              |                    |                 |
| 8x H100s | OSFT (No Liger)  | Granite 4   | 00:10:39       | 00:16:46     | 00:06:07           | 1.57x           |


## Graph: Granite 3.3 vs Granite 4

### Tabled differences

