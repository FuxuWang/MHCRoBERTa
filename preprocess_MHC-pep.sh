# Split data into pep sequence, MHC sequence, and label
for f in preprocessed_data/split_tokenized/full/tokenized_data*; do
        cut -f1 -d',' "$f" > preprocessed_data/split_tokenized/pep/$(basename "$f").pep
        cut -f2 -d',' "$f" > preprocessed_data/split_tokenized/MHC/$(basename "$f").MHC
	cut -f3 -d',' "$f" > preprocessed_data/split_tokenized/label/$(basename "$f").label
done
# Binarize sequences
fairseq-preprocess \
        --only-source \
        --trainpref preprocessed_data/split_tokenized/pep/tokenized_data.split.train.80.pep \
        --validpref preprocessed_data/split_tokenized/pep/tokenized_data.split.valid.10.pep \
        --testpref preprocessed_data/split_tokenized/pep/tokenized_data.split.test.10.pep \
        --destdir preprocessed_data/split_binarized/input0 \
        --workers 60 \
        --srcdict pretraining_data/split_binarized/dict.txt

fairseq-preprocess \
        --only-source \
        --trainpref preprocessed_data/split_tokenized/MHC/tokenized_data.split.train.80.MHC \
        --validpref preprocessed_data/split_tokenized/MHC/tokenized_data.split.valid.10.MHC \
        --testpref preprocessed_data/split_tokenized/MHC/tokenized_data.split.test.10.MHC \
        --destdir preprocessed_data/split_binarized/input1 \
        --workers 60 \
        --srcdict pretraining_data/split_binarized/dict.txt

# Binarize labels
fairseq-preprocess \
	--only-source \
	--trainpref preprocessed_data/split_tokenized/label/tokenized_data.split.train.80.label \
        --validpref preprocessed_data/split_tokenized/label/tokenized_data.split.valid.10.label \
        --testpref preprocessed_data/split_tokenized/label/tokenized_data.split.test.10.label \
	--destdir preprocessed_data/split_binarized/label \
	--workers 60
