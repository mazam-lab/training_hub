# Data Formats

Training Hub uses standardized data formats for all training algorithms. This ensures consistency and compatibility across different backends.

## Overview

All training functions ([`sft()`](/api/functions/sft), [`osft()`](/api/functions/osft)) expect training data in **JSONL (JSON Lines)** format with a specific structure.

**JSONL Format:**
- One JSON object per line
- Each line is a complete, valid JSON object
- Lines are separated by newlines
- File extension: `.jsonl`

## Standard Format: Messages

The primary data format uses a `messages` field containing a conversation with multiple turns.

### Structure

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

### Message Roles

| Role | Description | Required |
|------|-------------|----------|
| `system` | System prompt or instructions | Optional |
| `user` | User input/question | Required |
| `assistant` | Model response/answer | Required (this is what the model learns to generate) |

### Example: Basic Q&A

```jsonl
{"messages": [{"role": "system", "content": "You are a helpful medical assistant."}, {"role": "user", "content": "What are the symptoms of flu?"}, {"role": "assistant", "content": "Common flu symptoms include fever, body aches, fatigue, cough, and congestion."}]}
{"messages": [{"role": "system", "content": "You are a helpful medical assistant."}, {"role": "user", "content": "How long does flu typically last?"}, {"role": "assistant", "content": "Flu symptoms typically last 3-7 days, though fatigue can persist for up to 2 weeks."}]}
```

### Example: Multi-Turn Conversation

```jsonl
{"messages": [{"role": "system", "content": "You are a coding assistant."}, {"role": "user", "content": "How do I create a list in Python?"}, {"role": "assistant", "content": "You can create a list using square brackets: `my_list = [1, 2, 3]`"}, {"role": "user", "content": "How do I add an item?"}, {"role": "assistant", "content": "Use the `append()` method: `my_list.append(4)`"}]}
```

## Optional Sample-Level Fields

### Reasoning Content

The `reasoning_content` field allows you to include chain-of-thought reasoning or internal monologue for assistant messages. This content is included in the training but typically not shown to end users during inference.

**Structure:**

```jsonl
{"messages": [{"role": "user", "content": "What is 25 * 17?"}, {"role": "assistant", "content": "425", "reasoning_content": "Let me break this down:\n25 * 17 = 25 * (10 + 7)\n= (25 * 10) + (25 * 7)\n= 250 + 175\n= 425"}]}
```

The `reasoning_content` field is part of the message object and provides additional training signal for how the model should think through problems.

### Unmask Field

The `unmask` field is a sample-level boolean that controls loss masking behavior during training. When `unmask=true`, the entire sample (including user messages and system prompts) contributes to the training loss, not just the assistant responses.

**Structure:**

```jsonl
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}], "unmask": true}
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}], "unmask": false}
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

**Default behavior:** If `unmask` is not specified, only assistant messages are used for loss calculation (standard supervised fine-tuning behavior).

## Complete Example File

**File:** `training_data.jsonl`

```jsonl
{"messages": [{"role": "system", "content": "You are an expert legal advisor."}, {"role": "user", "content": "What is a contract?"}, {"role": "assistant", "content": "A contract is a legally binding agreement between two or more parties that creates mutual obligations enforceable by law."}]}
{"messages": [{"role": "system", "content": "You are an expert legal advisor."}, {"role": "user", "content": "What makes a contract valid?"}, {"role": "assistant", "content": "A valid contract requires: 1) offer and acceptance, 2) consideration (exchange of value), 3) legal capacity of parties, 4) legal purpose, and 5) mutual consent."}]}
{"messages": [{"role": "system", "content": "You are an expert legal advisor."}, {"role": "user", "content": "Can a contract be verbal?"}, {"role": "assistant", "content": "Yes, verbal contracts can be legally binding, but they're harder to prove. Written contracts are strongly recommended for important agreements."}]}
{"messages": [{"role": "system", "content": "You are an expert legal advisor."}, {"role": "user", "content": "What is breach of contract?"}, {"role": "assistant", "content": "A breach of contract occurs when one party fails to fulfill their obligations as specified in the agreement, either by non-performance or inadequate performance."}]}
```

## Creating Training Data

### From CSV

```python
import csv
import json

# Read CSV
with open('data.csv', 'r') as csv_file:
    reader = csv.DictReader(csv_file)

    # Write JSONL
    with open('training_data.jsonl', 'w') as jsonl_file:
        for row in reader:
            message_data = {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": row['question']},
                    {"role": "assistant", "content": row['answer']}
                ]
            }
            jsonl_file.write(json.dumps(message_data) + '\n')
```

### From Python Dictionary

```python
import json

training_examples = [
    {
        "system": "You are a math tutor.",
        "question": "What is 2+2?",
        "answer": "2+2 equals 4."
    },
    {
        "system": "You are a math tutor.",
        "question": "What is 5*6?",
        "answer": "5*6 equals 30."
    }
]

with open('training_data.jsonl', 'w') as f:
    for example in training_examples:
        message_data = {
            "messages": [
                {"role": "system", "content": example["system"]},
                {"role": "user", "content": example["question"]},
                {"role": "assistant", "content": example["answer"]}
            ]
        }
        f.write(json.dumps(message_data) + '\n')
```

### From Existing Datasets

```python
from datasets import load_dataset
import json

# Load a HuggingFace dataset
dataset = load_dataset("your_dataset_name")

# Convert to JSONL
with open('training_data.jsonl', 'w') as f:
    for example in dataset['train']:
        # Adapt this based on your dataset structure
        message_data = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": example['input']},
                {"role": "assistant", "content": example['output']}
            ]
        }
        f.write(json.dumps(message_data) + '\n')
```

## See Also

- [**sft() Function**](/api/functions/sft) - Uses this data format
- [**osft() Function**](/api/functions/osft) - Uses this data format
- [**Examples**](/examples/README) - Sample datasets and notebooks
