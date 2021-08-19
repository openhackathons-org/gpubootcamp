#!/usr/bin/env python
# coding: utf-8
# %%

# ## Introduction
# 
# This example demonstrates how to implement an autoregressive language model
# using a miniature version of the GPT model.
# The model consists of a single Transformer block with causal masking
# in its attention layer.
# 
# 
# **References:**
# 
# - [GPT](https://www.semanticscholar.org/paper/Improving-Language-Understanding-by-Generative-Radford/cd18800a0fe0b668a1cc19f2ec95b5003d0a5035)
# - [GPT-2](https://www.semanticscholar.org/paper/Language-Models-are-Unsupervised-Multitask-Learners-Radford-Wu/9405cc0d6169988371b2755e573cc28650d14dfe)
# - [GPT-3](https://arxiv.org/abs/2005.14165)

# ## Setup
import argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import numpy as np
import re
import string
import random
import os
import sys
import time
# Disable warning , info etc.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['CUDA_VISIBLE_DEVICES']='0'
### Prepare the data for word-level language modelling



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    args = parser.parse_args()

    return args

# ## Implement a Transformer block as a layer
def causal_attention_mask(batch_size, n_dest, n_src, dtype):
    """
    Mask the upper half of the dot product matrix in self attention.
    This prevents flow of information from future tokens to current token.
    1's in the lower triangle, counting from the lower right corner.
    """
    i = tf.range(n_dest)[:, None]
    j = tf.range(n_src)
    m = i >= j - n_src + n_dest
    mask = tf.cast(m, dtype)
    mask = tf.reshape(mask, [1, n_dest, n_src])
    mult = tf.concat(
        [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], 0
    )
    return tf.tile(mask, mult)


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads, embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        causal_mask = causal_attention_mask(batch_size, seq_len, seq_len, tf.bool)
        attention_output = self.att(inputs, inputs, attention_mask=causal_mask)
        attention_output = self.dropout1(attention_output)
        out1 = self.layernorm1(inputs + attention_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)


# ## Implement an embedding layer
# 
# Create two seperate embedding layers: one for tokens and one for token index
# (positions).

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions





def create_model():
    inputs = layers.Input(shape=(maxlen,), dtype=tf.int32)
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, feed_forward_dim)
    x = transformer_block(x)
    outputs = layers.Dense(vocab_size)(x)
    
    model = keras.Model(inputs=inputs, outputs=[outputs, x])
    
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    model.compile(
        "adam", loss=[loss_fn, None],
    )  # No loss and optimization based on word embeddings from transformer block
    return model


def main():
    args = parse_args()
    global g_args
    g_args = args
    batch_size = args.batch_size
    print("Batch size: "+str(batch_size))
    ### Implement the miniature GPT model

    global vocab_size
    vocab_size = 20000  # Only consider the top 20k words
    global maxlen
    maxlen = 80  # Max sequence size
    global embed_dim
    embed_dim = 256  # Embedding size for each token
    global num_heads
    num_heads = 2  # Number of attention heads
    global feed_forward_dim
    feed_forward_dim = 256  # Hidden layer size in feed forward network inside transformer

    # The dataset contains each review in a separate text file
    # The text files are present in four different folders
    # Create a list all files
    filenames = []
    directories = [
        "/workspace/python/source_code/Data/wikitext-2"
    ]
    for dir in directories:
        for f in os.listdir(dir):
            filenames.append(os.path.join(dir, f))

    # print(f"{len(filenames)} files")

    # Create a dataset from text files
    random.shuffle(filenames)
    text_ds = tf.data.TextLineDataset(filenames)
    text_ds = text_ds.shuffle(buffer_size=256)
    text_ds = text_ds.batch(batch_size)


    def custom_standardization(input_string):
        """ Remove html line-break tags and handle punctuation """
        lowercased = tf.strings.lower(input_string)
        stripped_html = tf.strings.regex_replace(lowercased, "<br />", " ")
        return tf.strings.regex_replace(stripped_html, f"([{string.punctuation}])", r" \1")


    # Create a vectorization layer and adapt it to the text
    vectorize_layer = TextVectorization(
        standardize=custom_standardization,
        max_tokens=vocab_size - 1,
        output_mode="int",
        output_sequence_length=maxlen + 1,
    )
    vectorize_layer.adapt(text_ds)
    vocab = vectorize_layer.get_vocabulary()  # To get words back from token indices


    def prepare_lm_inputs_labels(text):
        """
        Shift word sequences by 1 position so that the target for position (i) is
        word at position (i+1). The model will use all words up till position (i)
        to predict the next word.
        """
        text = tf.expand_dims(text, -1)
        tokenized_sentences = vectorize_layer(text)
        x = tokenized_sentences[:, :-1]
        y = tokenized_sentences[:, 1:]
        return x, y


    text_ds = text_ds.map(prepare_lm_inputs_labels)
    text_ds = text_ds.prefetch(tf.data.experimental.AUTOTUNE)


    # ## Implement a Keras callback for generating text

    # %%
    class PrintLR(tf.keras.callbacks.Callback):
        def __init__(self, total_images=0):
            self.total_images = total_images
        def on_epoch_begin(self, epoch, logs=None):
            self.epoch_start_time = time.time()
        def on_epoch_end(self, epoch, logs=None):
            epoch_time = time.time() - self.epoch_start_time
            print('Epoch time : {}'.format(epoch_time))
            images_per_sec = round(self.total_images / epoch_time, 2)
            print('Units/sec: {}'.format(images_per_sec))


    model = create_model()
    model.fit(text_ds, verbose=1, epochs=3, callbacks=[PrintLR(total_images=44880)])


main()


