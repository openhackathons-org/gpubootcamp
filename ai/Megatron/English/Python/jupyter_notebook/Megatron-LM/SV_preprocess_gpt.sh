INPUT_JSON_FILE=/workspace/SVdata/raw/json/79803/SV_CC100Sprakbank.json
#OUTPUT_PATH=./SVdata/gpt2bpe/SV_GPT3_56kvocab_CC100Sprakbank
OUTPUT_PATH=./SVdata/gpt2bpe/SV_GPT3_56kvocab_CC100Sprakbank
VOCAB_FILE=./SVdata/gpt2bpe/56k/vocab.json
MERGE_FILE=./SVdata/gpt2bpe/56k/merges.txt
NUM_CPUS=1

python tools/preprocess_data.py \
       --input $INPUT_JSON_FILE \
       --output-prefix $OUTPUT_PATH \
       --json-keys text \
       --vocab-file $VOCAB_FILE \
       --merge-file $MERGE_FILE \
       --dataset-impl mmap \
       --tokenizer-type GPT2BPETokenizer \
       --workers $NUM_CPUS \
       --append-eod

