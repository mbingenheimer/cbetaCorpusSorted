from transformers import AutoTokenizer, AutoModel
#from transformers.pipelines import pipeline
import pandas as pd 
import numpy as np
#from bertopic import BERTopic
import glob, os, sys
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
import jieba
import re
import argparse
import time, datetime
import pickle
from tqdm import tqdm
import socket
import logging

"""
TODO: 1) batch processing currently only works for segmented sentences; extend this to other cases.

"""

_sentence_delim_regex = re.compile(r'[:。！？;、）（：；〈〉]')
_sentence_kill_regex = re.compile(r'[)(『』》《「」〃【】〔〕〖〗〘〙〚〛〝〞〟〰〾〿…‧﹏.]')

_apocrypha = ['T24n1484.txt', 'T85n2865.txt', 'T85n2866.txt',
              "T85n2867.txt", "T85n2868.txt", "T85n2869.txt",
              "T85n2870.txt", "T85n2871.txt", "T85n2872.txt",
              "T85n2873.txt", "T85n2874.txt", "T85n2875.txt",
              "T85n2876.txt", "T85n2877.txt", "T85n2878.txt",
              "T85n2879.txt", "T85n2880.txt", "T85n2881.txt",
              "T85n2882.txt", "T85n2883.txt", "T85n2884.txt",
              "T85n2885.txt", "T85n2886.txt", "T85n2887.txt",
              "T85n2888.txt", "T85n2889.txt", "T85n2890.txt",
              "T85n2891.txt", "T85n2892.txt", "T85n2893.txt",
              "T85n2894.txt", "T85n2895.txt", "T85n2896.txt",
              "T85n2897.txt", "T85n2898.txt", "T85n2899.txt",
              "T85n2900.txt", "T85n2901.txt", "T85n2902.txt",
              "T85n2903.txt", "T85n2904.txt", "T85n2905.txt",
              "T85n2906.txt", "T85n2907.txt", "T85n2908.txt",
              "T85n2909.txt", "T85n2910.txt", "T85n2911.txt",
              "T85n2912.txt", "T85n2913.txt", "T85n2914.txt",
              "T85n2915.txt", "T85n2916.txt", "T85n2917.txt",
              "T85n2918.txt", "T85n2919.txt", "T85n2920.txt"]



if socket.gethostname() == 'alaya':
    code_folder = "/home/justin/projects/buddhist_nlp/early_chinese_buddhism_topic_modelling/"
else:
    code_folder = "/data/brody/early_chinese_buddhism_topic_modelling/"



def get_chinese_corpus(pre_aligned=False):
    """ Get Chinese corpus from files """
    print('Getting Chinese corpus...')
    if not pre_aligned:
        corpus = []
        for file in glob.glob("chinese_chinese/*.txt"):
            with open(file, 'r') as f:
                corpus.append(f.read())
    else:
        corpus = []
        for filename in glob.glob("chinese_chinese_aligned/*.txt"):
            taisho_id = filename.split('/')[-1].split('.')[0].split('_')[-1]
            taisho_filename = get_taisho_filename(taisho_id)
            with open(taisho_filename, 'r') as f:
                taisho_text = f.read()
                corpus.append(taisho_text)
    return corpus

def load_corpus(corpus_sign = 'chinese', use_segmented=False, keep_doc_ids=False):
    folder = 'chinese-chinese' if corpus_sign == 'chinese' else 'indian-chinese'
    full_folder = os.path.join(code_folder, folder)

    if use_segmented:
        file_dict = parse_filenames(full_folder)
        docs = load_docs_segmented(file_dict)
        # Figure out vectorizer
    else:
        docs = load_docs(full_folder, use_sentences=use_sentences)


    if keep_doc_ids:
        return [open(doc, 'r').read() for doc in docs], [x.split('/')[-1] for x in docs]
    else:
        return [open(doc, 'r').read() for doc in docs]

def load_docs_segmented(fdict):
    base_folder = os.path.join(code_folder, "word-segment-main/word-segmented-cbeta/seged-txt")
    docs = []
    for filename in fdict:
        prefix = fdict[filename][0]
        folder = re.split('\d+', prefix)[0]
        path = os.path.join(base_folder, folder, prefix, filename)
        docs.append(path)
    return docs


def parse_filenames(folder):
    filenames  = glob.glob(os.path.join(folder, "*.txt"))
    base_names = [f.split('/')[-1] for f in filenames]
    years = [f.split('_')[0] for f in base_names]
    names = [ f.split('_')[-1] for f in base_names]
    prefix = [ f.split('n')[0] for f in names]
    names_with_years = dict(zip(names, zip(prefix, years)))
    return names_with_years

def get_word_embeddings(corpus, device="cuda", keep_doc_ids=False):
    # Returns a torch tensor of embeddings of each word in the corpus
    # Corpus is assumed to be a list of strings
    # We will assume that the corpus is already segmented
    words = set()
    word_counts = {}
    word_embeddings = {}
    embeddings = []
    if keep_doc_ids:
        corpus, doc_ids = corpus
        docs_list = []
        doc_idx=0
    for doc in tqdm(corpus):
        if keep_doc_ids:
            doc_id = doc_ids[doc_idx]
        for word in get_words(doc):
            word = word.replace('\n', '')
            if 0 < len(word)  < 512:
                print("processing word: ", word, end='\r')
                if word not in words:
                    words.add(word)
                    word_counts[word] = 1
                    #emb = hf_model(word)
                    # Switch to cc_model
                    emb = cc_model.encode(word)
                    word_embeddings[word] = emb
                    embeddings.append(emb)
                    if keep_doc_ids:
                        docs_list.append(doc_id)
                else:
                    word_counts[word] += 1
            if len(word) > 512:
                print("\n\nWord too long: ", word)
        if keep_doc_ids:
            doc_idx += 1
    if keep_doc_ids:
        return embeddings, word_counts, word_embeddings, docs_list
    else:
        return embeddings, word_counts, word_embeddings


def get_words(doc):
    # Return the words from a document.
    # Assume the docuemnt is a segmented string, with words separated by '/'
    return doc.split('/')

def embedding_save(embeddings, filename):
    np.save(filename, embeddings)

def get_sentence_embeddings(corpus, device="cuda", keep_doc_ids=False, segmented=False, skip_embeddings=False, batch_size=4):
    # Returns a torch tensor of embeddings of each sentence in the corpus
    # Corpus is assumed to be a list of strings
    # We will assume that the corpus is already segmented
    # skip_embeddings is a hack to fix an earlier mistake where we didn't save sentence_doc_ids as a dictionary
    
    sentences = set()
    sentence_counts = {}
    sentence_embeddings = {}
    if keep_doc_ids:
        corpus, doc_ids = corpus
        doc_idx=0
        sentence_doc_ids = {}

    total_sentences = 0
    sentences_processed = 0
    batch_count = 0
    unsegmented_batch = []
    segmented_batch = []

    for doc in tqdm(corpus):
        if keep_doc_ids:
            doc_id = doc_ids[doc_idx]

        all_sentences = get_sentences(doc)
        for sentence in all_sentences:
            sentence = sentence.replace('\n', '')
            if 0 < len(sentence):
                total_sentences += 1
                if sentence not in sentences:
                    sentences.add(sentence)
                    sentence_counts[sentence] = 1
                    if keep_doc_ids:
                        sentence_doc_ids[sentence] = [doc_id]
                    if skip_embeddings:
                        continue
                    if segmented:
                        unsegmented = sentence.replace('/', '')   # TODO:  strip whitespace as well?  Let's say no; could be a word delimeter.  Although it shouldn't be in the current context.
                        unsegmented_batch.append(unsegmented)
                        segmented_batch.append(sentence)
                        batch_count += 1
                        if batch_count % batch_size == 0:
                            batch_embeddings = cc_model.encode(unsegmented_batch, batch_size = batch_size, show_progress_bar=False)
                            for i, sentence in enumerate(segmented_batch):
                                sentence_embeddings[sentence] = batch_embeddings[i]
                            unsegmented_batch = []
                            segmented_batch = []
                            batch_count = 0
                            sentences_processed += batch_size
                    else:
                        emb = cc_model.encode(sentence)
                    #sentence_embeddings[sentence] = emb
                else:
                    sentence_counts[sentence] += 1
                    if keep_doc_ids:
                        sentence_doc_ids[sentence].append(doc_id)
            if len(sentence) > 512:
                logging.warning(f"Long sentence: {sentence}")

        if keep_doc_ids:
            doc_idx += 1
                
    if batch_count % batch_size != 0:   # Final odd-sized batch
        batch_embeddings = cc_model.encode(unsegmented_batch, batch_size = batch_size)

        for i, sentence in enumerate(segmented_batch):
            sentence_embeddings[sentence] = batch_embeddings[i]
        unsegmented_batch = []
        segmented_batch = []
        batch_count = 0                
        sentences_processed += len(segmented_batch)

    print(f"Processed a total of {sentences_processed} sentences out of {total_sentences} sentences in the corpus.")
    if keep_doc_ids:
        return sentence_embeddings, sentence_counts, sentence_doc_ids
    else:
        return sentence_counts, sentence_embeddings

def get_sentences(doc, segmented=True):
    # Return the sentences from a document.
    # Want to keep the segmentation
    #if segmented:
    #    doc = doc.replace('/', '')
    doc = re.sub(_sentence_kill_regex, '', doc)
    return re.split(_sentence_delim_regex, doc)

def get_doc_embeddings(corpus, device="cuda",keep_doc_ids=False):
    # Returns a torch tensor of embeddings of each document in the corpus
    # Corpus is assumed to be a list of strings
    # We will assume that the corpus is already segmented
    if keep_doc_ids:
        corpus, doc_ids = corpus
    if keep_doc_ids:
        return [hf_model(doc) for doc in corpus], doc_ids
    return [hf_model(doc) for doc in corpus]

def main():
    print("Run with args: ", " ".join(sys.argv))
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu', help='device to use for computations')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for embedding computation')
    parser.add_argument('--not_prealigned', action='store_true', help='do not use pre-aligned corpus')
    parser.add_argument('--skip_sentences', action='store_true', help="don't compute sentence embeddings")
    parser.add_argument('--skip_words', action='store_true', help="don't compute word embeddings")
    parser.add_argument('--skip_docs', action='store_true', help="don't compute document embeddings")
    parser.add_argument('--keep_doc_ids', action='store_true', help="retain source documents with embeddings")
    parser.add_argument('--checkpoint', type=str, default="KoichiYasuoka/roberta-classical-chinese-large-char", help='checkpoint directory')
    parser.add_argument('--output_folder', type=str, default='bert_embeddings', help='checkpoint directory')
    parser.add_argument('--include_apocrypha', action='store_true', help='include texts from apocrypha list')
    args = parser.parse_args()
    global tokenizer, hf_model, guwenb_model, cc_model
    #tokenizer = AutoTokenizer.from_pretrained("ethanyt/guwenbert-base")
    logging.basicConfig(level=logging.INFO,
                        filename=os.path.join(args.output_folder, 'create_embeddings.log'))

    cc_model = SentenceTransformer(args.checkpoint)
    keep_docs = args.keep_doc_ids
    apocrypha = args.include_apocrypha
    segmented = not args.not_prealigned

    start_time = time.time()
    chinese_corpus = load_corpus('chinese', use_segmented=segmented, keep_doc_ids=keep_docs)
    indian_corpus = load_corpus('indian', use_segmented=segmented, keep_doc_ids=keep_docs)
    if apocrypha:
        apochrypha_corpus = load_corpus('apocrypha', use_segmented=segmented, keep_doc_ids=keep_docs)

    if not args.skip_words:
        chinese_word_output = os.path.join(args.output_folder, 'chinese_word_embeddings.pkl')
        indian_word_output = os.path.join(args.output_folder, 'indian_word_embeddings.pkl')
        print("Computing Chinese word embeddings...")
        chinese_word_embeddings = get_word_embeddings(chinese_corpus, args.device, keep_docs)
        print("Saving Chinese-Chinese word embeddings...")
        pickle.dump(chinese_word_embeddings, open(chinese_word_output, "wb"))
        print("Computing Indian word embeddings...")
        indian_word_embeddings = get_word_embeddings(indian_corpus, args.device, keep_docs)
        print("Saving Chinese-Indian word embeddings...")
        pickle.dump(indian_word_embeddings, open(indian_word_output, "wb"))

    if not args.skip_sentences:
        chinese_sentence_output = os.path.join(args.output_folder, 'chinese_sentence_embeddings-docids.pkl')
        indian_sentence_output = os.path.join(args.output_folder, 'indian_sentence_embeddings-docids.pkl')
        print("Computing Chinese sentence embeddings...")
        chinese_sentence_embeddings = get_sentence_embeddings(chinese_corpus, args.device, keep_docs, segmented, False, batch_size=args.batch_size)
        print("Saving Chinese-Chinese sentence embeddings...")
        #pickle.dump(chinese_sentence_embeddings, open("chinese_sentence_embeddings.pkl", "wb"))
        pickle.dump(chinese_sentence_embeddings, open(chinese_sentence_output, "wb"))
        print("Computing Indian sentence embeddings...")
        indian_sentence_embeddings = get_sentence_embeddings(indian_corpus, args.device, keep_docs, segmented, False)
        print("Saving Chinese-Indian sentence embeddings...")
        #pickle.dump(indian_sentence_embeddings, open("indian_sentence_embeddings.pkl", "wb"))
        pickle.dump(indian_sentence_embeddings, open(indian_sentence_output, "wb"))

    if not args.skip_docs:
        chinese_doc_output = os.path.join(args.output_folder, 'chinese_doc_embeddings.pkl')
        indian_doc_output = os.path.join(args.output_folder, 'indian_doc_embeddings.pkl')
        print("Computing Chinese document embeddings...")
        chinese_doc_embeddings = get_doc_embeddings(chinese_corpus, args.device, keep_docs)
        print("Saving Chinese-Chinese document embeddings...")
        pickle.dump(chinese_doc_embeddings, open(chinese_doc_output, "wb"))
        print("Computing Indian document embeddings...")
        indian_doc_embeddings = get_doc_embeddings(indian_corpus, args.device, keep_docs)
        print("Saving Chinese-Indian document embeddings...")
        pickle.dump(indian_doc_embeddings, open(indian_doc_embeddings, "wb"))
    
    # print("Saving embeddings...")
    # embedding_names = ['chinese_word_embeddings', 'indian_word_embeddings', 'chinese_sentence_embeddings', 'indian_sentence_embeddings', 'chinese_doc_embeddings', 'indian_doc_embeddings']
    # embedding_list = [chinese_word_embeddings, indian_word_embeddings, chinese_sentence_embeddings, indian_sentence_embeddings, chinese_doc_embeddings, indian_doc_embeddings]
    # for name, embedding in zip(embedding_names, embedding_list):
    #     print("Saving {}...".format(name))
    #     pickle.dump(embedding, open(name + '.pkl', 'wb'))
    
    print('Done.')

    
if __name__ == "__main__":
    main()
