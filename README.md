# VietAI Research Team

## BART denoising language modeling in JAX/Flax

In the following, we demonstrate how to train a BART model 
using denoising language modeling objective as introduced in [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461).
More specifically, we demonstrate how JAX/Flax can be leveraged 
to pre-train [**`bart-base`**](https://huggingface.co/facebook/bart-base)
in Norwegian on a single TPUv3-8 pod.

The example script uses the 🤗 Datasets library. You can easily customize them to your needs if you need extra processing on your datasets.

To setup all relevant files for training, let's create a directory.

```bash
mkdir ./norwegian-roberta-base
```

### Train tokenizer
In the first step, we train a tokenizer to efficiently process the text input for the model. Similar to how it is shown in [How to train a new language model from scratch using Transformers and Tokenizers](https://huggingface.co/blog/how-to-train), we use a **`ByteLevelBPETokenizer`**.
The tokenizer is trained on the complete Norwegian dataset of OSCAR
and consequently saved in the cloned model directory.
This can take up to 10 minutes depending on your hardware ☕.

```python
from datasets import load_dataset
from tokenizers import trainers, Tokenizer, normalizers, ByteLevelBPETokenizer

# load dataset
dataset = load_dataset("oscar", "unshuffled_deduplicated_no", split="train")

# Instantiate tokenizer
tokenizer = ByteLevelBPETokenizer()

def batch_iterator(batch_size=1000):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i: i + batch_size]["text"]

# Customized training
tokenizer.train_from_iterator(batch_iterator(), vocab_size=50265, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

# Save files to disk
tokenizer.save("./norwegian-bart-base/tokenizer.json")
```

### Create configuration

Next, we create the model's configuration file. This is as simple 
as loading and storing [`**facebook/bart-base**`](https://huggingface.co/facebook/bart-base)
in the local model folder:

```python
from transformers import BartConfig
config = BartConfig.from_pretrained("facebook/bart-base", vocab_size=50265)
config.save_pretrained("./norwegian-bart-base")
```

Great, we have set up our model repository. During training, we will automatically
push the training logs and model weights to the repo.

### Train model

Next we can run the example script to pretrain the model:

```bash
python train.py \
    --output_dir="./norwegian-bart-base" \
    --config_name="./norwegian-bart-base" \
    --tokenizer_name="./norwegian-bart-base" \
    --dataset_name="oscar" \
    --dataset_config_name="unshuffled_deduplicated_no" \
    --max_seq_length="1024" \
    --per_device_train_batch_size="32" \
    --per_device_eval_batch_size="32" \
    --learning_rate="1e-4" \
    --warmup_steps="2000" \
    --overwrite_output_dir \
    --logging_steps="500" \
    --save_steps="2000" \
    --eval_steps="2000" \
    --push_to_hub
```

Training should converge at a loss and accuracy 
of 1.36 and 0.77 respectively after 3 epochs on a single TPUv3-8.
This should take less than 6 hours.
Training statistics can be accessed on [tfhub.dev](https://tensorboard.dev/experiment/Maw62QlaSXWS0MOf2V2lbg/).
