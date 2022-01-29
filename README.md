# Train text generation models from the command line
Using some of the code from [tensorflow's website](https://www.tensorflow.org/guide/keras/rnn).

## Usage
```
usage: train.py [-h] --dataset_path DATASET_PATH
                [--dataset_cutoff DATASET_CUTOFF] [--buffer_size BUFFER_SIZE]
                --epochs EPOCHS --batch_size BATCH_SIZE --seq_size SEQ_SIZE
                --embedding_dim EMBEDDING_DIM --rnn_units RNN_UNITS
                [--checkpoint_path CHECKPOINT_PATH]
                [--checkpoint_prefix CHECKPOINT_PREFIX]
                [--epochs_per_checkpoint EPOCHS_PER_CHECKPOINT]
                [--epochs_per_text_log EPOCHS_PER_TEXT_LOG]
                [--text_log_prompt TEXT_LOG_PROMPT]
                [--text_log_prompt_len TEXT_LOG_PROMPT_LEN]
                [--text_out_prompt TEXT_OUT_PROMPT]
                [--text_out_prompt_len TEXT_OUT_PROMPT_LEN]
                [--text_out_file TEXT_OUT_FILE] --export_model_filepath
                EXPORT_MODEL_FILEPATH [--export_loss_file EXPORT_LOSS_FILE]
                [--export_text_logs EXPORT_TEXT_LOGS] [--use_wandb]
                [--in_sweep] [--wandb_project_name WANDB_PROJECT_NAME]
                [--wandb_entity WANDB_ENTITY] [--use_gpu_mem_growth]
```
