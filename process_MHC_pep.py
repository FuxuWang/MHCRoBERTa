import pandas as pd
import sentencepiece as spm
import re
import math


model_path = 'm_reviewed.model'
seq_dict = dict()
MHC_II_dict = dict()
allelelist_map = dict()


def get_uniprot_protein_seq():
    f = open("pretraining_data/uniprot_sprot_56k_seq.txt", "r")
    all_protein_seqs = []
    each_protein = ""
    for line in f:
        line = line.replace("\n", "")
        attr = line[0:2]
        if attr == "SQ":
            each_protein = ""
        elif attr == "  ":
            each_protein = each_protein + line.replace(" ", "")
        elif attr == "//":
            all_protein_seqs.append(each_protein)
    all_protein_seqs_df = pd.DataFrame(all_protein_seqs)
    all_protein_seqs_df.to_csv("uniprot_pretraining_data.txt", index=False)


def get_map_HLA():
    '''
    Read the sequece of each allele. If there are multiple subtypes, choose the first one.
    Args:
        1. path: path to the data file binding_data_train.txt
    Return values:
        1. A dictionary whose keys are the name of MHC alleles and the corresponding dict
        values are amino acid sequences of those alleles.
    '''
    f = open("hla_prot.fasta", "r")
    allele = None
    for line in f:
        if line[0] == ">":  # A new allele
            match = re.search("(\w+\*\d+:\d+)", line)  # The standard allele id are like "A*01:01:..."
            # For alleles with the same two digits, like A*01:01:01 and A*01:01:02, we take the first one as the representative
            allele = None  # While reading the sequence of the same alleles from different
            # lines of the file, allele is not None, so that the sequences of each line
            # will be added to the end of the correspondong sequence
            # Some times we run into alleles with incorrect names, so that allele is set to None
            # and match == None so allele will not be reset, then the following lines of sequences
            # will not be recorded
            if match != None:  # If the current allele has a name with the correct format
                match_str = match.groups()[0]
                match_str = match_str.replace("*", "")
                if "HLA-" + match_str not in seq_dict.keys():
                    allele = "HLA-" + match_str  # A new allele
                    seq_dict[allele] = ""  # And its sequence
        elif allele != None:
            seq_dict[allele] = seq_dict[allele] + line[:-1]
    for allele in list(seq_dict.keys()):
        if len(seq_dict[allele]) < len(seq_dict['HLA-B07:02']):
            # Some sequences lack certain parts like the leader peptide, and cannot
            # be aligned to other sequences well. The ones longer than B*07:02 can
            # be aligned well with the majority of HLA A and B alleles, (see this in uniprot)
            seq_dict.pop(allele)


def tokenize_pep_MHC_I_BA_data(model, path, num):
    out = []
    pep_MHC_bindings = open(path, "r", encoding='utf-8')
    index = 0
    for line in pep_MHC_bindings:
        index = index + 1
        line = line.replace("\n", "")
        line = line.split(" ")
        if "HLA-" not in line[2] or line[2] not in seq_dict:
            continue
        out.append([" ".join(model.encode_as_pieces(line[0])), " ".join(model.encode_as_pieces(seq_dict[line[2]])), line[1]])
    out_df = pd.DataFrame(out)
    if(num == 0):
        out_df.to_csv('preprocessed_data/tokenized_fine_tuning_data.csv', index=False, mode='w', header=False)
    else:
        out_df.to_csv('preprocessed_data/tokenized_fine_tuning_data.csv', index=False, mode='a', header=False)


def get_MHC_I_tokenize(model):
    tokenize_pep_MHC_I_BA_data(model, "NetMHCpan_train/c000_ba", 0)
    tokenize_pep_MHC_I_BA_data(model, "NetMHCpan_train/c001_ba", 1)
    tokenize_pep_MHC_I_BA_data(model, "NetMHCpan_train/c002_ba", 1)
    tokenize_pep_MHC_I_BA_data(model, "NetMHCpan_train/c003_ba", 1)
    tokenize_pep_MHC_I_BA_data(model, "NetMHCpan_train/c004_ba", 1)


def tokenize_protein_data(model, path):
    out = []
    uniq_protein = dict()
    protein_seqs = open(path, "r", encoding='utf-8')
    for line in protein_seqs:
        line = line.replace("\n", "")
        out.append([" ".join(model.encode_as_pieces(line))])

    tokenized_vec = []
    for one in out:
        if one[0] not in uniq_protein:
            tokenized_vec.append(one[0])
            uniq_protein[one[0]] = 1

    out_df = pd.DataFrame(tokenized_vec)
    out_df.to_csv("preprocessed_data/uniprot_pretraining_data.csv", index=False, mode='w', header=False)


if __name__ == "__main__":
    get_map_HLA()
    get_uniprot_protein_seq()
    spm.SentencePieceTrainer.Train('--input=uniprot_pretraining_data.txt --model_prefix=m_reviewed_MHC --vocab_size=10000 --character_coverage=1.0 --model_type=bpe --max_sentence_length=1024')
    model = spm.SentencePieceProcessor()
    model.load(model_path)
    get_MHC_I_tokenize(model)
    tokenize_protein_data(model, "uniprot_pretraining_data.txt")
