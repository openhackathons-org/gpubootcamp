# Copyright (c) 2020 NVIDIA Corporation.  All rights reserved.
# -*- coding: utf-8 -*-
import os , sys
from tokenizers import Tokenizer
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer
from tokenizers.normalizers import NFKC, Sequence
from tokenizers.pre_tokenizers import Whitespace
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', default=None, type=str,  help='path to the txt files')
    parser.add_argument('--bpe_path', default=None, type=str, help='output GPTBPE path')
    parser.add_argument('--load_pretrained', action='store_true', help='load pretrained GPTBPE model')
    parser.add_argument('--pretrained_gpt_dir', default=None, type=str, help='path to pretrained GPT vocab and merge files, default None')
    parser.add_argument('--vocab_size', default=None, type=int, 
                        help='specify the vocab_size when training HF GPTBPE for own language usually 16k/32k/48k/64k')
    args = parser.parse_args()  
    tokenizer = Tokenizer(models.BPE())
    if args.load_pretrained :
        if args.pretrained_gpt_dir is not None :
            print("loading gpt2bpe english vocab and merge \n")
            vocab_file=args.pretrained_gpt_dir+'/gpt2-vocab.json'
            merge_file=args.pretrained_gpt_dir+'/gpt2-merges.txt'
            tokenizer.model = models.BPE.from_file(vocab_file, merge_file) 
        else:
            print("please provide path to the pretrained gpt vocab and merge file !")
    
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    tokenizer.decoder = ByteLevelDecoder()

    print("include minimal special token end of text ")
    special_tokens= ["<|endoftext|>"]
        
    # Set the training hyperparameters
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    # Train it with either files or an iterator:
    tokenizer.train([args.infile], trainer=trainer)
    print("Trained vocab size: {}".format(tokenizer.get_vocab_size()))
    # You will see the generated files in the output.
    print("saving trained BPE model to : ", args.bpe_path)
    tokenizer.model.save(args.bpe_path)
    print("model saved ! \n\n\n")
    print("testing ...\n\n\n")
    test_txt="Har någon funderat på varför man inte får inomhusteperaturens kurva synlig i grafen? Är det någon som frågat Thermia? Skulle det inte vara väsentligt att kunna kolla historiken på den då man skall ställa in kurvan?"
    output = tokenizer.encode(test_txt)
    print(output.tokens)
