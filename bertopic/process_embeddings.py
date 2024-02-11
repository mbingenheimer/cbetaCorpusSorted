# Note:  this should be run in buddhist_nlp virtual environment if using translations. 
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import umap
import hdbscan
from create_embeddings import _sentence_delim_regex, _sentence_kill_regex
import re
import os

_cuda = False
if _cuda:
    import cudf, cuml
    import cupy as cp
    from cuml.manifold import UMAP as cumlUMAP
    from cuml.cluster import HDBSCAN as cumlHDBSCAN
#from create_report import create_report

from process_embeddings_utils import get_all_points

from sklearn import decomposition

def load_embeddings(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def get_meaningful_clusters_bak_deprecated(hdbfile, metric='sentential', k=5, handpicked=False, display_scores_for_handpicking=False):
    """
    Find the clusters which most disintguish the chinese and indian corpora.
    Extract the exemplar words from each cluster.
    Metric determines how we measure what's meaningful:
       'bertopic':  Find all the words in each cluster, and then use the countmetric as in bertopic to determine the most meaningful clusters.
       'sentential':  Use the sentence embeddings to find the most meaningful clusters, and then extract the relevant words afterward.
    """
    clusters = sentences_in_clusters(hdbfile)
    #clusters = words_in_clusters(clusters)
    #word_frequncies = get_word_frequencies(clusters)
    if display_scores_for_handpicking:
        scores_for_handpicking(clusters)
    elif metric == 'bertopic':
        return top_bertopic_clusters(word_clusters, word_frequncies, k)
    elif metric == 'sentential':
        if handpicked:
            result = handpicked_clusters(clusters)
            pickle.dump(result, open('topk_words_hp.pkl', 'wb'))
        else:
            result = top_sentential_clusters(clusters, k) 
            pickle.dump(result, open('topk_words.pkl', 'wb'))

def plot_clusters():
    hdb_file = pickle.load(open("hdbscan_3_sentence_100minCluster_100minSamples.pkl", "rb"))
    umap_data = np.load("umap_3_sentence_all.npy")
    num_chinese = 1887900
    pca = decomposition.PCA(n_components=2)
    T = pca.fit_transform(umap_data)
    #chinese_pca_embeds = T[:num_chinese]
    #indian_pca_embeds = T[num_chinese:]

    l = hdb_file.labels_
    chinese_clusters = [(172, 0.93), (398, 0.91), (455, 0.995),  (509, 0.89), (775, 0.87)]

    indian_clusters = [(507, 0.10), (459, 0.17)]

    # fig = plt.figure(1, figsize=(4, 3))
    # plt.clf()
    # ax = fig.add_subplot(111, projection="3d", elev=48, azim=134)
    # ax.set_position([0, 0, 0.95, 1])
    plt.cla()
    plt.scatter(T[:, 0], T[:, 1], c = (0.1, 0.1, 0.1, 0.01), label="Other")
    # Color conventions:
    # Amount of blue indicates how "chinese" the cluster is
    # Red clusters are chinese
    # Green clusters are indian
    for idx, ccl in enumerate(chinese_clusters):
        cnum, cprob = ccl
        red = float(idx + 1) / len(chinese_clusters)
        indices = np.where(l == cnum)[0]
        plt.scatter(T[indices, 0], T[indices, 1], c=(red, 0, cprob), label="Cluster {}".format(cnum))
    for idx, icl in enumerate(indian_clusters):
        inum, iprob = icl
        green = float(idx + 1) / len(indian_clusters)
        indices = np.where(l == inum)[0]
        plt.scatter(T[indices, 0], T[indices, 1], c=(0, green, iprob), label="Cluster {}".format(inum))
    plt.legend(loc='lower right')
    plt.show()



def handpicked_clusters(clusters):
    """ Based on a hand analysis:
      Cluster 141 is large(ish) (11170) and 93% chinese
      Cluster 473 is large  (11432)   and 92% chinese
      Cluster 509 is large (11048) and >99% chinese
      Cluster 668 is large (21311) and 95% chinese	

      Cluster 566 is large (24272) and 8.7% chinese (mostly indian)
      Cluster 553 is large (26210) and 13.7% chinese (mostly indian)

    Base on new embeddings:
    Cluster 172 is large (12594.0) and 93% chinese
    Cluster 398 is large (12385.0) and 91% chinese
    Cluster 455 is large (11555.0) and 99.5% chinese
    Cluster 509 is large (14067.0) and 89% chinese
    Cluster 775 is large (53800.0) and 87% chinese

    Cluster 507 is large (39367.0) and 10% chinese
    Cluster 459 is large (32048.0) and 17% chinese
    """
    topk_words = {}
    topk_words[0] = {}
    topk_words[0]['chinese'] = {172: get_individual_cluster(172, clusters), 398: get_individual_cluster(398, clusters), 455: get_individual_cluster(455, clusters), 509: get_individual_cluster(509, clusters), 775: get_individual_cluster(775, clusters)}
    topk_words[0]['indian'] = {507: get_individual_cluster(507, clusters), 459: get_individual_cluster(459, clusters)}
    return topk_words

def top_sentential_clusters(clusters, k=5, mi_based=False):
    """Use the sentence embeddings to find the most meaningful clusters,
    and then extract the relevant words afterward.

    The most meaningful clusters will be those clusters K which
    maximize I(K; C) where C is the category (chinese or indian) and K
    is the cluster.

    An alternate algorithm will find the top k clusters for P(indian|K) and the top k clusters for P(chinese|K).

    Or, imitating tf-idf, we can give each cluster K a score, for each category c, as follows:
               Pr(K|c)*IC(K)
where Pr(K|c) is the probability that a sentence in category c is in cluster K, and IC(K) is the information content of cluster K, given by log(1/Pr(K))

    The above probably doesn't actually make sense, although it does
    correspond to tf-idf.  It basically counts the size of a cluster,
    weighted by how 'interesting' it is.  I think it would make more
    sense to count the size of a cluster, weighted by the proportion
    of the cluster in the given category.  Thus, for cluster K and
    category cat, the score is Pr(K) * Pr(cat|K)

    """
    cluster_scores = {}
    topk_indices = {}
    topk_words = {}
    for alpha in [1, 2, 4, 8, 10, 20, 32, 50, 75, 100]:
        topk_words[alpha] = {}
        for cat in ['indian', 'chinese']:
            cluster_scores[cat] = calculate_cluster_scores(clusters, cat, alpha=alpha)
            topk_indices[cat] = np.argpartition(cluster_scores[cat][0], -k)[-k:]
            print(f"Top {k} clusters for {cat} (alpha={alpha}):", topk_indices[cat])
            topk_words[alpha][cat] = cluster_words(topk_indices[cat], clusters)
 
    return topk_words

    
def cluster_words(indices, clusters):
    result = {}
    for i in indices:
        result[i] = get_individual_cluster(i, clusters)
    return result



    

def find_exemplars(hdbfile, umap_file, output_filename="correlated_exemplars.pkl"):
    """ The pickle file will be a list of exemplar.  
    Each exemplar is a list of (100?)  (embedding, sentence, source) triples, 
    (where source is a list of source documents)   """
    clusters = pickle.load(open(hdbfile, "rb"))
    raw_exemplars_ = clusters.exemplars_
    correlated_exemplars = correlate_exemplars(raw_exemplars_, umap_file)
    pickle.dump(correlated_exemplars, open(output_filename, 'wb'))
    return correlated_exemplars

def print_cluster_exemplars(dictionary_filename, latex=True, handpicked_dict=None):
    D = pickle.load(open(dictionary_filename, 'rb'))
    if handpicked_dict is None:
        handpicked = False
    else:
        handpicked = True
        H = pickle.load(open(handpicked_dict, 'rb'))

    def print_dictionary(D, omit_alpha=False):

        for alpha in D:
            if latex:
                if omit_alpha:
                    #print("\\section{Top Topics in Each Corpus for Handpicked Clusters}")
                    print("")
                else:
                    print("\\section{Top 5 Topics in Each Corpus for" + f" $\\alpha={alpha}$" + "}")
            for corpus in ['indian', 'chinese']:
                if latex:
                    print(f"\\subsection{{{corpus.capitalize()} Corpus}}")
                else:
                    print(f"Alpha={alpha}, corpus={corpus.capitalize()}")
                for cnum, cluster in enumerate(D[alpha][corpus]):
                    if latex:
                        print("\\begin{table}[H]")

                        print("\\begin{tabular}[h]{l|l|l|l}")
                        print(f"Word & Score & Freq in Cluster & IC Across Clusters \\\\ \\hline")
                    for i, (word, translation, score, tf, ic) in enumerate(D[alpha][corpus][cluster]):
                        if latex:
                            print(f"{word} ({translation}) & {score:.2f} & {tf} & {ic:.2f} \\\\")
                        else:
                            print(f"{word} ({translation}): {score}")
                        if i > 19:
                            break
                    if latex:
                        print("\\end{tabular}")
                        if omit_alpha:
                            print("\\caption*{" +
                              f" Handpicked Clusters, corpus={corpus.capitalize()}, cluster number {cnum+1} \\\\" +
                              "Abs cluster " + str(cluster) +  ": top 20 words and scores}")
                        else:
                            print("\\caption*{" +
                              f" $\\alpha={alpha}$, corpus={corpus.capitalize()}, cluster number {cnum+1} \\\\" +
                              "Abs cluster " + str(cluster) +  ": top 20 words and scores}")
                        print("\\end{table}")
                    else:
                        print()
    if latex:
        print("\\documentclass[UTF8]{ctexart}")
        print("\\usepackage{CJKutf8}")
        print("\setCJKmainfont{BabelStone Han}")
        print("\\usepackage{float}")
        print("\\usepackage{caption}")
        print("\\usepackage{hyperref}")
        print("\\begin{document}")
        print("\\tableofcontents")
        print("\\AddToHook{cmd/section/before}{\\clearpage}")
        if handpicked:
            print("\\section{Handpicked Topics}")
            print("Based on a hand analysis of the clusters, we have selected the following topics.")
            print("")
            print("\\begin{tabular}{l}")
            print("Cluster 172 is large (12594.0) and 93\\% chinese. \\\\")
            print("Cluster 398 is large (12385.0) and 91\\% chinese. \\\\")
            print("Cluster 455 is large (11555.0) and 99.5\\% chinese. \\\\")
            print("Cluster 509 is large (14067.0) and 89\\% chinese. \\\\")
            print("Cluster 775 is large (53800.0) and 87\\% chinese. \\\\")
            print("\\hline\\  \\")
            print("Cluster 507 is large (39367.0) and 10\\% chinese. \\\\")
            print("Cluster 459 is large (32048.0) and 17\\% chinese")
            print("\\end{tabular}{l}")
            print_dictionary(H, omit_alpha=True)
            print("\n\n")
    print_dictionary(D)
    if latex:
        print("\\end{document}")

def get_umap_correlations(umap_file, sentence_embedding=True):
    """
    umap_to_key:  The key (word/sentence) associated with a point in UMAP-space.
    key_to_umap:  The UMAP-space point associated with a key (word/sentence).
    emb_to_key:   The key (word/sentence) associated with a point in the original embedding space.
    key_to_emb:   The original embedding space point associated with a key (word/sentence).
    """
    X, num_chinese, num_indian, total, X_key_list, doc_ids = get_all_points(repeats=False, sentences=sentence_embedding, get_key_list=True, get_doc_ids=True)
    umap_embeddings = np.load(umap_file)
    key_to_umap = {}
    umap_to_key = {}
    key_to_emb = {}
    emb_to_key = {}
    for i in range(len(X_key_list)):
        strXi = ' '.join(["{:.6f}".format(x) for x in X[i]])
        strUMi = ' '.join(["{:.6f}".format(x) for x in umap_embeddings[i]])
        key_to_emb[X_key_list[i]] = strXi
        emb_to_key[strXi] = X_key_list[i]
        key_to_umap[X_key_list[i]] = strUMi
        umap_to_key[strUMi] = X_key_list[i]

    return umap_to_key, key_to_umap, emb_to_key, key_to_emb, doc_ids


def correlate_exemplars(raw_exemplars_, umap_file):
    global _embedding_folder
    indian_embeddings = load_embeddings(os.path.join(_embedding_folder,
                                                     'indian_sentence_embeddings.pkl'))
    chinese_embeddings = load_embeddings(os.path.join(_embedding_folder,
                                                      'chinese_sentence_embeddings.pkl'))
    i_dict = indian_embeddings[2]
    c_dict = chinese_embeddings[2]
    doc_dict = {}
    for ikey in i_dict:
        doc_dict[ikey] = i_dict[ikey]
        if ikey in c_dict:
            doc_dict[ikey] += c_dict[ikey]
    for ckey in c_dict:
        if ckey not in doc_dict:
            doc_dict[ckey] = c_dict[ckey]
    correlated_exemplars = []
    umap_keys, key_to_umap, emb_to_key, key_to_emb, doc_ids = get_umap_correlations(umap_file)
    for exemplar in raw_exemplars_:
        correlated_exemplars.append( find_match(exemplar, umap_keys, doc_dict))
    return correlated_exemplars



def find_match(exemplar, umap_keys, doc_ids):
    """ Find the closest exemplar. """
    mexemplar = np.mean(exemplar, axis=1).flatten()
    result = []
    for ex in exemplar:
        exstr = ' '.join(['{:.6f}'.format(x) for x in ex])
        key = umap_keys[exstr]
        result.append( (exemplar, key, doc_ids[key]) )
    return result

def collate_embeddings(embeddings, sentences=False):
    """ Collate embeddings into a single dictionary.
        If sentences is True, then value will be ????
        Otherwise, we are working with word embeddings and the value will be (num_occurrences, embedding, list_of_docs

        The raw data is of the form:
           sentence_embeddings, sentence_counts, doc_ids                   for sentence embeddings
           embeddings, word_counts, word_embeddings, docs_list               for word embeddings
    """
    print('Collating embeddings...')
    collated_embeddings = {}
    for key in embeddings[1]:
        if sentences:
            collated_embeddings[key] = (embeddings[0][key], embeddings[1][key], embeddings[2][key])
        else:
            collated_embeddings[key] = (embeddings[1][key], np.mean(np.array(embeddings[2][key]), axis=1).flatten())
    return collated_embeddings


def plot_pca(chinese_pca_embeds, indian_pca_embeds):
    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()

    ax = fig.add_subplot(111, projection="3d", elev=48, azim=134)
    ax.set_position([0, 0, 0.95, 1])
    plt.cla()
    ax.scatter(chinese_pca_embeds[:,0], chinese_pca_embeds[:, 1], chinese_pca_embeds[:,2], c='r', label='Chinese')
    ax.scatter(indian_pca_embeds[:,0], indian_pca_embeds[:, 1], indian_pca_embeds[:, 2] , c='b', label='Indian')
    plt.show()


def save_collated_embeddings(sentences=False, embedding_folder="./"):
    if sentences:
        file_list =  ['chinese_sentence_embeddings-docids.pkl', 'indian_sentence_embeddings-docids.pkl']
    else:
        file_list =  ['chinese_word_embeddings.pkl', 'indian_word_embeddings.pkl']
    for emb_file in file_list:
        # Load embeddings
        emb_file_with_path = os.path.join(embedding_folder, emb_file)
        embeddings = load_embeddings(emb_file_with_path)
        # Collate embeddings
        collated_embeddings = collate_embeddings(embeddings, sentences=sentences)
        # Save embeddings
        emb_file_name = os.path.join(embedding_folder, 'collated_' + emb_file)
        with open(emb_file_name, 'wb') as f:
            pickle.dump(collated_embeddings, f)
    
def pca(X, num_chinese, num_indian, gpu=False):
    if gpu:
        import pycuda.autoinit
        import pycuda.gpuarray as gpuarray
        import skcuda.linalg as linalg
        from skcuda.linalg import PCA as cuPCA
        pca = cuPCA(n_components=3)  # map the data to 43dimensions
    else:
        pca = decomposition.PCA(n_components=3)


    if gpu:
        X_gpu = gpuarray.GPUArray(X.shape, np.float64, order="F") # note that order="F" or a transpose is necessary. fit_transform requires row-major matrices, and column-major is the default
        X_gpu.set(X) # copy data to gpu
        T_gpu = pca.fit_transform(X_gpu) # calculate the principal components
        T = T_gpu.get() # copy the result back to the host
    else:
        T = pca.fit_transform(X)

    np.save('pca_all.npy', T)

    chinese_pca_embeds = T[:num_chinese]
    indian_pca_embeds = T[num_chinese:]
    return chinese_pca_embeds, indian_pca_embeds


def plot_tsne(X, num_chinese, num_indian, total, gpu=False):
    if gpu:
        from tsnecuda import TSNE
        tsne = TSNE(n_components=2)
        T = tsne.fit_transform(X)
    else:
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2)
        T = tsne.fit_transform(X)
    save_dict = {'chinese': T[:num_chinese], 'indian': T[num_chinese:]}
    with open('tsne_all.pkl', 'wb') as f:
        pickle.dump(save_dict, f)
    plt.scatter(T[:num_chinese, 0], T[:num_chinese, 1], 'r.')
    plt.scatter(T[num_chinese:, 0], T[num_chinese:, 1], 'b.')
    plt.show()

def plot_2d(points, num_chinese):
    plt.clf()
    plt.scatter(points[:num_chinese, 0], points[:num_chinese, 1], 'r.')
    plt.scatter(points[num_chinese:, 0], points[num_chinese:, 1], 'b.')
    plt.show()

def plot_3d(points, num_chinese):
    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()

    ax = fig.add_subplot(111, projection="3d", elev=48, azim=134)
    ax.set_position([0, 0, 0.95, 1])
    plt.cla()
    ax.scatter(points[:,0], points[:, 1], points[:,2], c='r', label='Chinese')
    ax.scatter(points[:,0], points[:, 1], points[:, 2] , c='b', label='Indian')
    plt.show()
    
def main():
    global _embedding_folder
    global _output_folder
    import argparse

    argparse = argparse.ArgumentParser(epilog="""
    Once the embeddings are created, the following steps form the basic pipeline: \n\n
    \t--save collated:   collate the embeddings with the sentences and documents
                      (collated_*_*_embeddings.pkl)
    \n\t--umap:  dimensionality reduction  (umap_*_*.npy)
    \n\t--hdbscan:  clustering (hdscan_*_*.pkl)
    \n\t--exemplars:  for each of the clusters, find examplar senteces and correlate with umap data
                  (correlated_exemplars_*.pkl)
    \n\t--topics:  find the words that correspond to each topic.
    
    """)
    argparse.add_argument('--save_collated', action='store_true', help='Save collated embeddings')
    argparse.add_argument('--save_pca', action='store_true', help='Save PCA embeddings')
    argparse.add_argument('--display_pca', action='store_true', help='Display PCA')
    argparse.add_argument('--display_umap', action='store_true', help='Display UMAP')
    argparse.add_argument('--word_embeddings', action='store_true', default=True, help='Use word embeddings')
    argparse.add_argument('--sentence_embedding', action='store_true', default=False, help='Use sentence embeddings')
    argparse.add_argument('--gpu', action='store_true', default=False, help='Use GPU')
    argparse.add_argument('--tsne', action='store_true', default=False, help='Use t-SNE')
    argparse.add_argument('--umap', action='store_true', default=False, help='Save and display UMAP')
    argparse.add_argument('--hdbscan', action='store_true', default=False, help='Perform HDBSCAN on UMAP output')
    argparse.add_argument('--hdbscan_min_cluster_size', type=int, default=100, help='HDBSCAN min cluster size')
    argparse.add_argument('--hdbscan_min_samples', type=int, default=100, help='HDBSCAN min samples')
    argparse.add_argument('--exemplars', action='store_true', default=False, help='Get exemplar sentences from topics created by HDBSCAN')
    argparse.add_argument('--topics', action='store_true', default=False, help='Get topics from HDBSCAN')
    argparse.add_argument('--include_handpicked', action='store_true', default=False, help='Include handpicked exemplars')
    argparse.add_argument('--create_cluster_report', action='store_true', default=False, help='Extract words, their c-tf-idf, and their translations from HDBSCAN file')
    argparse.add_argument('--input_folder', type=str, default='bert_embeddings', help='Input folder')
    argparse.add_argument('--output_folder', type=str, default='outputs', help='Output folder')
    argparse.add_argument('--num_jobs', type=int, default=32, help='Number of jobs to run in parallel')
    args = argparse.parse_args()

    if not args.create_cluster_report:
        print(f"Running with argumnets {args}")

    _embedding_folder = args.input_folder
    _output_folder = args.output_folder

    if args.save_collated:
        save_collated_embeddings(sentences=args.sentence_embedding,
                                 embedding_folder=args.output_folder)

    chinese_pca_embeds = None
    indian_pca_embeds = None
    if args.save_pca:
        X, num_chinese, num_indian, total = get_all_points()

        print("Performing PCA")
        chinese_pca_embeds, indian_pca_embeds = pca(X, num_chinese, num_indian, args.gpu)

        np.save(os.path.join(_output_folder, 'chinese_pca_embeds.npy'), chinese_pca_embeds)
        np.save(os.path.join(_output_folder, 'indian_pca_embeds.npy'), indian_pca_embeds)

    if args.umap:
        chinese_sentences_filename = os.path.join(args.output_folder,
                                                  'collated_chinese_sentence_embeddings-docids.pkl')
        indian_sentences_filename = os.path.join(args.output_folder,
                                                 'collated_indian_sentence_embeddings-docids.pkl')
        X, num_chinese, num_indian, total = get_all_points(repeats=False,
                                                           sentences=args.sentence_embedding,
                                                           chinese_sentences_filename=chinese_sentences_filename,
                                                           indian_sentences_filename=indian_sentences_filename)
        
        for dims in [2, 3, 10]:
            print("Performing UMAP with {} dimensions".format(dims))
            if args.gpu:
                print(f"Using gpu on data of shape {X.shape}")
                reducer = cumlUMAP(n_components=dims)
                embedding = reducer.fit_transform(X)
                sentence_label = "sentence" if args.sentence_embedding else "word"
            else:
                reducer = umap.UMAP(n_components=dims)
                embedding = reducer.fit_transform(X)
                sentence_label = "sentence" if args.sentence_embedding else "word"
            np.save(os.path.join(_output_folder, 'umap_{}_{}_all.npy'.format(dims,sentence_label)),
                    embedding)

    if args.hdbscan:
        min_cluster_size=args.hdbscan_min_cluster_size
        min_samples=args.hdbscan_min_samples

        
        #for dims in [2, 3, 10]:
        for dims in [3]:
            if args.gpu:
                print("Using GPU.")
                hdb = cumlHDBSCAN( gen_min_span_tree=True,
                                       min_cluster_size=min_cluster_size,
                                       min_samples=min_samples
                                      )
            else:
                hdb = hdbscan.HDBSCAN( gen_min_span_tree=True,
                                       min_cluster_size=min_cluster_size,
                                       min_samples=min_samples
                                      )
            print("Performing HDBSCAN with {} dimensions".format(dims))
            sentence_label = "sentence" if args.sentence_embedding else "word"
            embeddings = np.load(os.path.join(_output_folder, 
                                              'umap_{}_{}_all.npy'.format(dims,sentence_label)))
            hdb.fit(embeddings)

            hdbfilename = os.path.join(_output_folder,
                                       'hdbscan_{}_{}_{}minCluster_{}minSamples.pkl'.format(dims,
                                                                                            sentence_label,
                                                                                            min_cluster_size,
                                                                                            min_samples))
            with open(hdbfilename , "wb") as hf:
                pickle.dump(hdb, hf)
                
    if args.exemplars:
        min_cluster_size=args.hdbscan_min_cluster_size
        min_samples=args.hdbscan_min_samples
        for dims in [3]:
            sentence_label = "sentence" if args.sentence_embedding else "word"
            clusters_file = os.path.join(args.output_folder,  'hdbscan_{}_{}_{}minCluster_{}minSamples.pkl'.format(dims,sentence_label, min_cluster_size, min_samples))
            umap_file = os.path.join(args.output_folder, 'umap_{}_{}_all.npy'.format(dims,sentence_label))
            print(f"Correlating {dims} dimensional {sentence_label} exemplars")
            output_filename = os.path.join(args.output_folder,
                                           "correlated_exemplars_{}.pkl".format(dims))
            find_exemplars(clusters_file, umap_file,output_filename )

    if args.display_pca:
        if chinese_pca_embeds is None:
            chinese_pca_embeds = np.load('chinese_pca_embeds.npy')
            indian_pca_embeds = np.load('indian_pca_embeds.npy')
        plot_pca(chinese_pca_embeds, indian_pca_embeds)
                                       
    if args.tsne:
        X, num_chinese, num_indian, total = get_all_points()
        print("Running t-SNE")
        plot_tsne(X, num_chinese, num_indian, total, args.gpu)
                                       
    if args.display_umap:
        plot_umap()
                                       
    if args.topics:
        get_topics()

    if args.create_cluster_report:
        sentence_label = "sentence" if args.sentence_embedding else "word"
        clusters_file = 'hdbscan_{}_{}_{}minCluster_{}minSamples.pkl'.format(3,sentence_label, args.hdbscan_min_cluster_size, args.hdbscan_min_samples)
        create_report(clusters_file, args.output_folder, latex=True, all_clusters=True, num_jobs=args.num_jobs)
                                       
if __name__ == '__main__':
    #display_clusters('hdbscan_{}_{}_{}minCluster_{}minSamples.pkl'.format(3,"sentence",100,100),
    #                 'umap_{}_{}_all.npy'.format(3,"sentence"))
    #get_meaningful_clusters('hdbscan_{}_{}_{}minCluster_{}minSamples.pkl'.format(3,"sentence",100,100), display_scores_for_handpicking=True)
    #get_meaningful_clusters('hdbscan_{}_{}_{}minCluster_{}minSamples.pkl'.format(3,"sentence",100,100))
    #get_meaningful_clusters('hdbscan_{}_{}_{}minCluster_{}minSamples.pkl'.format(3,"sentence",100,100), handpicked = True)
    #print_cluster_exemplars("topk_words.pkl", latex=True, handpicked_dict="topk_words_hp.pkl")
    main()
                                       
                                       
                                       
                                       
