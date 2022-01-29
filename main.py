#encoding=utf-8
import argparse
from distutils.dir_util import copy_tree
import os
def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return arg  

parser = argparse.ArgumentParser(description="Easy trainable text generation RNN model with TensorFlow",
                                epilog="Copyright NikiTricky, 2021")

dst = parser.add_argument_group('Dataset preferences')
dst.add_argument('--dataset_path', required=True, help="The dataset filepath.", type=lambda x: is_valid_file(parser, x))
dst.add_argument('--dataset_cutoff', help="If set, the dataset will be trimmed to the specified length.", type=int)
dst.add_argument('--buffer_size', default=10000, help="Buffer size to shuffle the dataset.", type=int)

mpf = parser.add_argument_group('Model/Traning preferences')
mpf.add_argument('--epochs', required=True, help="Number of epochs to run the model.", type=int)
mpf.add_argument('--batch_size', required=True, help="Batch size for the model.", type=int)
mpf.add_argument('--seq_size', required=True, help="Sequence size/length to train the model on.", type=int)
mpf.add_argument('--embedding_dim', required=True, help="Model's embedding dimention.", type=int)
mpf.add_argument('--rnn_units', required=True, help="Model's RNN units.", type=int)

ckpt = parser.add_argument_group('Checkpoints')
ckpt.add_argument('--checkpoint_path', help="Path to store checkpoints in, if left empty, no checkpoints will be stored.")
ckpt.add_argument('--checkpoint_prefix', default="ckpt", help="Prefix to store checkpoint files with. Default is 'ckpt'.")
ckpt.add_argument('--epochs_per_checkpoint', default=10, help="The amount of epochs after to store a checkpoint. Default is 10.", type=int)

lg = parser.add_argument_group('Text logging')
lg.add_argument('--epochs_per_text_log', default=0, help="Epochs per text log. If --use_wandb is enabled, it will send the logs to wandb. If the value is 0, no text will be loged. Requires --text_log_prompt. Default is 0.", type=int)
lg.add_argument('--text_log_prompt', help="Text prompt for logs.")
lg.add_argument('--text_log_prompt_len', default=100, help="Text log generated text length. Default is 100.", type=int)
lg.add_argument('--text_out_prompt', help="Text output prompt. Runs at the end of training. Requires --text_out_prompt_len and --text_out_file.")
lg.add_argument('--text_out_prompt_len', default=0, help="Text length for output text.", type=int)
lg.add_argument('--text_out_file', help="The file to store the output text.")

exp = parser.add_argument_group('Exporting')
exp.add_argument('--export_model_filepath', required=True, help="Folder to export the model.")
#exp.add_argument('--export_keras_filepath', help="Save the model to a .h5 format or folder.")
exp.add_argument('--export_loss_file', help="Save the losses to a newline-separated file.")
exp.add_argument('--export_text_logs', help="Save the text logs to a text file. Requires --text_log_prompt.")

wndb = parser.add_argument_group('Weights and Biases fuctions')
wndb.add_argument('--use_wandb', action='store_const', default=False, const=True, help="Uses wandb. Logs the model loss and starting conditions. Requires for you to be logged in wandb and --wandb_project_name and --wandb_entity. Don't include if --in-sweep is enabled.")
wndb.add_argument('--in_sweep', action='store_const', default=False, const=True, help="If in wandb sweep. Removes some functions to help sweeps.")
wndb.add_argument('--wandb_project_name', help="Wandb project name to log the model training.")
wndb.add_argument('--wandb_entity', help="Entity to use when logging model.")

parser.add_argument('--use_gpu_mem_growth', action='store_const', default=False, const=True, help="If you are getting errors with the code. Works only if you have a GPU.")

args = parser.parse_args()

if args.epochs_per_text_log != 0 and (args.text_log_prompt is None):
    parser.error("--epochs_per_text_log requires --text_log_prompt.")

if args.text_out_prompt_len != 0 and (args.text_out_prompt is None or args.text_out_file is None):
    parser.error("--text_out_prompt_len requires --text_out_prompt and --text_out_file.")

if args.use_wandb and (args.wandb_project_name is None or args.wandb_entity is None):
    parser.error("--use_wandb requires --wandb_project_name and --wandb_entity.")

if args.export_loss_file and (args.epochs_per_text_log == 0):
    parser.error("--export_loss_file requires --epochs_per_text_log.")

if args.export_text_logs and (args.text_log_prompt is None):
    parser.error("--export_text_logs requires --text_log_prompt.")
    
import tensorflow as tf
if args.use_gpu_mem_growth:
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

import numpy as np
import os
import time

if args.use_wandb:
    import wandb
    
# Read, then decode for py2 compat.
text = open(args.dataset_path, 'rb').read().decode(encoding='utf-8')
if args.dataset_cutoff:
    text = text[:args.dataset_cutoff]
# length of text is the number of characters in it
print(f'Length of text: {len(text)} characters')

# The unique characters in the file
vocab = sorted(set(text))
print(f'{len(vocab)} unique characters')

ids_from_chars = tf.keras.layers.StringLookup(
    vocabulary=list(vocab), mask_token=None)
chars_from_ids = tf.keras.layers.StringLookup(
    vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)
def text_from_ids(ids):
  return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)

all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)

def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text

# Length of the vocabulary in chars
vocab_size = len(vocab)

class MyModel(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, rnn_units):
    super().__init__(self)
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(rnn_units,
                                   return_sequences=True,
                                   return_state=True)
    self.dense = tf.keras.layers.Dense(vocab_size)

  def call(self, inputs, states=None, return_state=False, training=False):
    x = inputs
    x = self.embedding(x, training=training)
    if states is None:
      states = self.gru.get_initial_state(x)
    x, states = self.gru(x, initial_state=states, training=training)
    x = self.dense(x, training=training)

    if return_state:
      return x, states
    else:
      return x

if args.checkpoint_path:
    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(args.checkpoint_path, args.checkpoint_prefix + "_{epoch}")

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)

class CustomTraining(MyModel):
  @tf.function
  def train_step(self, inputs, model):
      inputs, labels = inputs
      with tf.GradientTape() as tape:
          predictions = self(inputs, training=True)
          loss = self.loss(labels, predictions)
      grads = tape.gradient(loss, model.trainable_variables)
      self.optimizer.apply_gradients(zip(grads, model.trainable_variables))

      return {'loss': loss}

class OneStep(tf.keras.Model):
  def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
    super().__init__()
    self.temperature = temperature
    self.model = model
    self.chars_from_ids = chars_from_ids
    self.ids_from_chars = ids_from_chars

    # Create a mask to prevent "[UNK]" from being generated.
    skip_ids = self.ids_from_chars(['[UNK]'])[:, None]
    sparse_mask = tf.SparseTensor(
        # Put a -inf at each bad index.
        values=[-float('inf')]*len(skip_ids),
        indices=skip_ids,
        # Match the shape to the vocabulary
        dense_shape=[len(ids_from_chars.get_vocabulary())])
    self.prediction_mask = tf.sparse.to_dense(sparse_mask)

  @tf.function
  def generate_one_step(self, inputs, states=None):
    # Convert strings to token IDs.
    input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
    input_ids = self.ids_from_chars(input_chars).to_tensor()

    # Run the model
    # predicted_logits.shape is [batch, char, next_char_logits]
    predicted_logits, states = self.model(inputs=input_ids, states=states,
                                          return_state=True)
    # Only use the last prediction.
    predicted_logits = predicted_logits[:, -1, :]
    predicted_logits = predicted_logits/self.temperature
    # Apply the prediction mask: prevent "[UNK]" from being generated.
    predicted_logits = predicted_logits + self.prediction_mask

    # Sample the output logits to generate token IDs.
    predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
    predicted_ids = tf.squeeze(predicted_ids, axis=-1)

    # Convert from token ids to characters
    predicted_chars = self.chars_from_ids(predicted_ids)

    # Return the characters and model state.
    return predicted_chars, states

def gen_text(model, prompt, length):
    one_step_model = OneStep(model, chars_from_ids, ids_from_chars)    
    next_char = tf.constant([prompt])
    states = None
    result = [next_char]

    for n in range(length):
      next_char, states = one_step_model.generate_one_step(next_char, states=states)
      result.append(next_char)
    return tf.strings.join(result)[0].numpy().decode('utf-8')

from tqdm import tqdm

def create_dataset(seq_length, BATCH_SIZE):
    global args
    examples_per_epoch = len(text)//(seq_length+1)
    sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)

    dataset = sequences.map(split_input_target)

    # Buffer size to shuffle the dataset
    # (TF data is designed to work with possibly infinite sequences,
    # so it doesn't attempt to shuffle the entire sequence in memory. Instead,
    # it maintains a buffer in which it shuffles elements).
    BUFFER_SIZE = args.buffer_size

    dataset = (
        dataset
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE))
    return dataset

def create_model(embedding_dim, rnn_units):
    model = CustomTraining(
    vocab_size=len(ids_from_chars.get_vocabulary()),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units)
    model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    return model

def train(EPOCHS, BATCH_SIZE, seq_length, rnn_units, embedding_dim):
    global args, checkpoint_prefix
    
    dataset = create_dataset(seq_length, BATCH_SIZE)
    model = create_model(rnn_units, embedding_dim)
    
    mean = tf.metrics.Mean()
    dataset = list(dataset)
    ma = []

    if args.use_wandb:
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity)

        wandb.config.epochs = EPOCHS
        wandb.config.batch_size = BATCH_SIZE
        wandb.config.seq_size = seq_length
        wandb.config.rnn_units = rnn_units
        wandb.config.embedding_dim = embedding_dim

    srt = time.time()
    text = ""
    losses = []
    text_logs = []
    for epoch in range(0, EPOCHS):
        print(f'Epoch {epoch+1}/{EPOCHS}')

        start = time.time()

        mean.reset_states()
        for (batch_n, (inp, target)) in tqdm(enumerate(dataset), total=len(dataset)):
            logs = model.train_step([inp, target], model)
            mean.update_state(logs['loss'])
            losses.append(mean.result().numpy())
 
        if args.checkpoint_path:
            if (epoch + 1) % args.epochs_per_checkpoint == 0:
                model.save_weights(checkpoint_prefix.format(epoch=epoch))
        
        if args.epochs_per_text_log != 0:
            if (epoch + 1) % args.epochs_per_text_log == 0:
                text = gen_text(model, args.text_log_prompt, args.text_log_prompt_len)
                text_logs.append(text)
                print(f"Sample for epoch {epoch+1}:\\n{text}")
                    
        t = time.time() - start
        print(f"Loss: {mean.result().numpy():.4f}")
        
        if args.use_wandb or args.in_sweep:
            wandb.log({"loss": mean.result().numpy(), "time_per_epoch": t, "gen_text": wandb.Html(text)})
            
        print(f'Time taken for 1 epoch {t:.2f} sec')
        if len(ma) < 3:
          ma.append(t)
        else:
          ma.append(t)
          ma.pop(0)
        eta = (sum(ma)/len(ma))*EPOCHS-(time.time()-srt)
        print(f"ETA: {float(eta)}s ({(eta/60):.2f}m, {(eta/3600):.2f}h)")
    if args.checkpoint_path:
        model.save_weights(checkpoint_prefix.format(epoch=epoch))
    one_step_model = OneStep(model, chars_from_ids, ids_from_chars)
    tf.saved_model.save(one_step_model, args.export_model_filepath)
    if args.use_wandb or args.in_sweep:
        #wandb.save('/model/*', base_path="/model")
        wandb.finish()
        
    if args.export_loss_file:
        with open(args.export_loss_file, 'w') as f:
            for loss in losses:
                f.write(f'{loss}\\n')
    
    if args.export_text_logs:
        with open(args.export_text_logs, 'w') as f:
            for text in text_logs:
                f.write(f'{text}\\n\\n')
    
    return model

# EPOCHS, BATCH_SIZE, seq_length, rnn_units, embedding_dim
model = train(args.epochs, args.batch_size, args.seq_size, args.rnn_units, args.embedding_dim)

if args.text_out_file:
    with open(args.text_out_file, 'w+') as f:
        f.write(gen_text(model, args.text_out_prompt, args.text_out_prompt_len))
