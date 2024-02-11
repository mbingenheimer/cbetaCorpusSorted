from process_embeddings_utils import get_all_points, cluster_analysis
from process_embeddings import cluster_words
from ddb_trans import ddb_trans
import os, pickle
import numpy as np
import re
from joblib import Parallel, delayed


from create_embeddings import _sentence_delim_regex, _sentence_kill_regex

def create_report(clusters_file,
                  folder=None,
                  latex=True,
                  all_clusters = True,
                  use_handpicked=False,
                  num_jobs=32,
                  precomputed_tops=False,
                  num_clusters_per_cat=5
                  ):

    """
    Create a report on all the clusters (each of which represents a topic)
    """
    if all_clusters:
        handpicked = 'all'
    elif use_handpicked:
        handpicked = 'handpicked'
    else:
        handpicked = False

    if precomputed_tops:
        _, cluster_anal, _, _, _ = get_meaningful_clusters(clusters_file, folder=folder, metric='sentential', k=num_clusters_per_cat, handpicked=handpicked, display_scores_for_handpicking=False, num_jobs=num_jobs, cluster_analysis_only=True)
        top1_file  = os.path.join(folder, 'topk_words_single.pkl')
        top3_file = os.path.join(folder, 'topk_words_triple.pkl')
    else:
        top_file, cluster_anal, triple_file, context1_file, context3_file = get_meaningful_clusters(clusters_file, folder=folder, metric='sentential', k=num_clusters_per_cat, handpicked=handpicked, display_scores_for_handpicking=False, num_jobs=num_jobs)

    # Top-file contains the most meaningful clusters broken down
    # into indidual words and triples of words.  cluster_anal
    # gives descriptive statistics on the clusters.
        
    # Parse the file and create the report.
    print_cluster_exemplars(top_file, cluster_anal, latex=latex, secondary_dictionary_file=triple_file, words1_in_context = context1_file )

    

_all_cluster_words = {}
def get_cluster_words(cluster,n=3):
    # This should be memoized since it is called multiple times
    # with same i.
    #
    # Recall that each cluster is a tuple of the form
    # (cluster_number, sentences, docs, num_csent, num_isent).
    # We want to extract the words from sentences
    if n not in _all_cluster_words:
        _all_cluster_words[n] = {}
    
    cluster_number, sentences, _, _, _ = cluster
    if cluster_number in _all_cluster_words[n]:
        return _all_cluster_words[n][cluster_number]
    else:
        word_counts = {}
        for sentence in sentences:
            sentence = re.sub(_sentence_kill_regex, '', sentence)
            words = sentence.split('/')
            words = [ "".join(words[i:i+n]) for i in range(len(words) -n+1)]
            for word in words:
                if word in word_counts:
                    word_counts[word] += 1
                else:
                    word_counts[word] = 1
        _all_cluster_words[n][cluster_number] = word_counts
        return word_counts


def get_topics():
    """This was in main at some point; probably doesn't need to live for
    too long.  Mostly calling two other with different parameters.
    """
    get_meaningful_clusters('hdbscan_{}_{}_{}minCluster_{}minSamples.pkl'.format(3,"sentence",100,100), display_scores_for_handpicking=True)
    #get_meaningful_clusters('hdbscan_{}_{}_{}minCluster_{}minSamples.pkl'.format(3,"sentence",100,100))
    #get_meaningful_clusters('hdbscan_{}_{}_{}minCluster_{}minSamples.pkl'.format(3,"sentence",100,100), handpicked = True)
    print_cluster_exemplars("topk_words.pkl", latex=True)    
    #print_cluster_exemplars("topk_words.pkl", latex=True, handpicked_dict="topk_words_hp.pkl")


def get_cluster_words_in_context(cluster_words_file, full_clusters, n=20, k=5):
    """For each of the top (n) exemplary words in each cluster, give (k)
    sentences in the cluster in which that word appears.


    """
    all_cluster_words = pickle.load(open(cluster_words_file, "rb"))
    words_in_context = {}
    for cluster_num in all_cluster_words:
        cluster_words = all_cluster_words[cluster_num][0]
        _, sentences, _, _, _ = full_clusters[cluster_num]
        # cluster_words is a list of 5-tuples
        # each tuple is of the form (Word, Translation, tfidf, tf, ic)
        word_list = [ tup[0] for tup in cluster_words[:n] ]
        words_in_context[cluster_num] = get_word_sentences(word_list, sentences, k)
    return words_in_context

def get_word_sentences(list_of_words, cluster_sentences, k=10):
    return { word: find_sentences_with_word(word, cluster_sentences, k) for word in list_of_words }

def find_sentences_with_word(word, cluster_sentences, k=10):
    return [sentence for sentence in cluster_sentences if word in sentence][:k]

def get_meaningful_clusters(hdbfile, folder=None, metric='sentential', k=5, handpicked=False, display_scores_for_handpicking=False, num_jobs=32, cluster_analysis_only=False):
    """

Find the clusters which most disintguish the chinese and indian
corpora, using top_sentential_clusters or top_bertopic_clusters.


    Extract the exemplar words from each cluster.
    Metric determines how we measure what's meaningful:
       'bertopic':  Find all the words in each cluster, and then use the countmetric as in bertopic to determine the most meaningful clusters.
       'sentential':  Use the sentence embeddings to find the most meaningful clusters, and then extract the relevant words afterward.

    Store results in "topk_words.pkl" and "topk_words_hp.pkl" (for handpicked clusters)

    """
    clusters, cluster_anal = sentences_in_clusters(hdbfile, folder, get_cluster_analysis=True)

    if cluster_analysis_only:
        return None, cluster_anal, None
    
    if display_scores_for_handpicking:
        scores_for_handpicking(clusters)
    elif metric == 'bertopic':
        return top_bertopic_clusters(word_clusters, word_frequncies, k)
    elif metric == 'sentential':
        if handpicked == 'all':
            result1 = all_clusters(clusters, 1, num_jobs=num_jobs)
            result3 = all_clusters(clusters, 3, num_jobs=num_jobs)
            filename1 = os.path.join(folder, 'topk_words_single.pkl')
            filename3 = os.path.join(folder, 'topk_words_triple.pkl')
            pickle.dump(result1, open(filename1, 'wb'))
            pickle.dump(result3, open(filename3, 'wb'))
            context1 = get_cluster_words_in_context(filename1, clusters)
            context3 = get_cluster_words_in_context(filename3, clusters)
            context_filename1 = os.path.join(folder, 'topk_words_single_in_context.pkl')
            context_filename3 = os.path.join(folder, 'topk_words_triple_in_context.pkl')
            pickle.dump(context1, open(context_filename1, 'wb'))
            pickle.dump(context3, open(context_filename3, 'wb'))
        elif handpicked != False:
            result = handpicked_clusters(clusters)
            filename = os.path.join(folder, 'topk_words_hp.pkl')
            pickle.dump(result, open(filename, 'wb'))
        else:
            result = top_sentential_clusters(clusters, k)
            filename = os.path.join(folder, 'topk_words.pkl')
            pickle.dump(result, open(filename, 'wb'))

        return filename1, cluster_anal, filename3, context_filename1, context_filename3



def sentences_in_clusters(hdbfile, folder=None, get_cluster_analysis=False):
    """For each of the clusters defined in the HDBFILE, extract the sentences in the cluster.
    
    If get_cluster_analysis is True, also return descirptitve
    statistics (like proportions from each corpus in cluster) on each
    cluster.

    """
    from create_embeddings import load_corpus

    chinese_sentences_filename="collated_chinese_sentence_embeddings.pkl"
    indian_sentences_filename="collated_indian_sentence_embeddings.pkl"
    if folder is not None:
        hdbfile = os.path.join(folder, hdbfile)
        chinese_sentences_filename = os.path.join(folder, chinese_sentences_filename)
        indian_sentences_filename = os.path.join(folder, indian_sentences_filename)


    chinese_docs = set(load_corpus('chinese', True, True)[1])
    indian_docs = set(load_corpus('indian', True, True)[1])

    clusters = pickle.load(open(hdbfile, "rb"))
    cluster_labels = np.array(clusters.labels_)
    num_clusters = np.max(cluster_labels) + 1
    #data = load_collated_embeddings()
    cluster_words = []
    clusters = []
    
    X, num_chinese, num_indian, total, X_key_list, doc_ids = get_all_points(repeats=False, sentences=True, get_key_list=True, get_doc_ids=True, chinese_sentences_filename=chinese_sentences_filename, indian_sentences_filename=indian_sentences_filename)
    if get_cluster_analysis:
        cluster_anal = cluster_analysis(cluster_labels, num_chinese, num_indian, total)  # This is redundant, should be part of computation below.  
    for cluster in range(num_clusters):
        num_csent, num_isent = 0, 0
        indices = np.where(cluster_labels == cluster)[0]
        sentences = [X_key_list[i] for i in indices]
        docs = set()
        for i in indices:
            for doc in doc_ids[i]:
                if doc in chinese_docs:
                    num_csent += 1
                    docs.add((doc, 'C'))
                else:
                    num_isent += 1
                    docs.add((doc, 'I'))
        clusters.append((cluster, sentences, docs, num_csent, num_isent))

    if get_cluster_analysis:
        return clusters, cluster_anal
    else:
        return clusters


#
#  
#


def top_sentential_clusters(clusters, k=5, mi_based=False):
    """Use the sentence embeddings to find the most meaningful clusters,
    and then extract the relevant words afterward.  Here k is the
    number of clusters we should return.

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
            
def get_individual_cluster(idx, clusters, n=1):
    cluster = clusters[idx]
    # Enumerate through the words in the cluster
    words = get_cluster_words(cluster,n)
    other_class_words = {}
    for i in range(len(clusters)):
        if i != idx:
            other_class_words[i] = get_cluster_words(clusters[i], n)
    
    tfidf = {}
    A = (sum([ len(other_class_words[i]) for i in range(len(clusters)) if i != idx]) + len(words)) / len(clusters)
    f = get_total_frequencies(clusters, n)
    for word in words:
        tfidf[word] = calculate_tfidf(word, words, other_class_words, f, A)

    final = sorted( [ (k, translate(k,True), tfidf[k]) for k in tfidf ], key=lambda x: x[2][0], reverse=True)
    final = [ (t[0], translate(t[0], i > 20), t[2][0], t[2][1], t[2][2]) for i, t in enumerate(final) ]
    return final


def all_clusters(clusters, n=1, num_jobs=32):
    """ Return all the clusters. """
    def get_individual_cluster_wrapper(i):
        return get_individual_cluster(i, clusters, n)
    all_clusters_list = Parallel(n_jobs=num_jobs,timeout=99999)( delayed(get_individual_cluster_wrapper)(i) for i in range(len(clusters)) )

    
    return { i: (all_clusters_list[i], clusters[i][2]) for i in range(len(clusters)) }
                                                                            
                                                                            
def handpicked_clusters(clusters):
    """ Based on a hand analysis:
      Cluster 141 is large(ish) (11170) and 93% chinese
      Cluster 473 is large  (11432)   and 92% chinese
      Cluster 509 is large (11048) and >99% chines
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

def top_bertopic_clusters(word_clusters, word_frequncies, k=5):
    pass

def calculate_mutual_information(word_cluster, cluster_num):
    """
    Recall that I(X; Y) = sum_{x,y} p(x,y) log(p(x,y)/(p(x)p(y)))

    Here we are trying to calculate I(K; C) where K is the cluster and C is the category (chinese or indian).
    """
    pass

def calculate_cluster_scores(word_clusters,  category, alpha=1.0):
    """Calculate the cluster scores for the given category. 
    For cluster K and category cat, the score is 

               Pr(K|cat)*IC(K)
    
    where Pr(K|cat) is the probability that a sentence in category cat is
    in cluster K, and IC(K) is the information content of cluster K,
    given by log(1/Pr(K))

    The above probably doesn't actually make sense, although it does
    correspond to tf-idf.  It basically counts the size of a cluster,
    weighted by how 'interesting' it is.  I think it would make more
    sense to count the size of a cluster, weighted by the proportion
    of the cluster in the given category.  Thus, for cluster K and
    category cat, the score is Pr(K) * Pr(cat|K)

    Returns an array scores, where scores[i] is the score for cluster i.
    """
    def compute_pr_k(word_clusters, frequency_only=False):
        pr_k = np.zeros(len(word_clusters))
        total_words = 0
        for i, cluster in enumerate(word_clusters):
            cluster, sentences, docs, num_csent, num_isent = cluster
            pr_k[i] = len(sentences)
        if not frequency_only:
            pr_k /= np.sum(pr_k)
        return pr_k

    def compute_pr_k_given_cat(word_clusters, category):
        """
        This is wrong -- it calculates Pr(cat|K)
        """
        pr_k_given_cat = np.zeros(len(word_clusters))
        for i, cluster in enumerate(word_clusters):
            cluster, sentences, docs, num_csent, num_isent = cluster
            total = num_csent + num_isent
            pr_k_given_cat[i] = num_csent / total if category == 'chinese' else num_isent / total
        return pr_k_given_cat

    def compute_pr_cat_given_k(word_clusters, category):
        pr_cat_given_k = np.zeros(len(word_clusters))
        for i, cluster in enumerate(word_clusters):
            cluster, sentences, docs, num_csent, num_isent = cluster
            total = num_csent + num_isent
            pr_cat_given_k[i] = num_csent / total if category == 'chinese' else num_isent / total
        return pr_cat_given_k

    scores = []
    pr_k = compute_pr_k(word_clusters, frequency_only=True)
    pr_cat_given_k= compute_pr_cat_given_k(word_clusters, category)
    scores = pr_k * np.exp(alpha*pr_cat_given_k)
    return scores, pr_k, pr_cat_given_k


def scores_for_handpicking(word_clusters):
    # only show clusters with weight > 10000:
    def compute_pr_k(word_clusters, frequency_only=False):
        pr_k = np.zeros(len(word_clusters))
        total_words = 0
        for i, cluster in enumerate(word_clusters):
            cluster, sentences, docs, num_csent, num_isent = cluster
            pr_k[i] = len(sentences)
        if not frequency_only:
            pr_k /= np.sum(pr_k)
        return pr_k


    def compute_pr_cat_given_k(word_clusters, category):
        pr_cat_given_k = np.zeros(len(word_clusters))
        for i, cluster in enumerate(word_clusters):
            cluster, sentences, docs, num_csent, num_isent = cluster
            total = num_csent + num_isent
            pr_cat_given_k[i] = num_csent / total if category == 'chinese' else num_isent / total
        return pr_cat_given_k

    scores = []
    pr_k = compute_pr_k(word_clusters, frequency_only=True)
    pr_cat_given_k= compute_pr_cat_given_k(word_clusters, 'chinese')
    big_clusters = np.where(pr_k > 10000)[0]
    print("Clusters with weight > 10000: ", big_clusters)
    print("Big cluster weights: ", pr_k[big_clusters])
    print("Chinese-only weights: ", pr_cat_given_k[big_clusters])
    print("\n\n")
    print("Reformatting for handpicking:")
    for z in zip( list(big_clusters), list(pr_k[big_clusters]), list(pr_cat_given_k[big_clusters]) ):
        print(z)


def get_total_frequencies(clusters,n):
    f = {}
    for i in range(len(clusters)):
        words = get_cluster_words(clusters[i], n)
        for word in words:
            if word in f:
                f[word] += words[word]
            else:
                f[word] = words[word]
    return f


def get_individual_cluster_bak(idx, clusters):
    cluster = clusters[idx]
    # Enumerate through the words in the cluster
    words = get_cluster_words(cluster)
    other_class_words = {}
    for i in range(len(clusters)):
        if i != idx:
            other_class_words[i] = get_cluster_words(clusters[i])

    tfidf = {}
    A = (sum([len(other_class_words[i]) for i in range(len(clusters)) if i != idx]) + len(words)) / len(clusters)
    f = get_total_frequencies(clusters)
    for word in words:
        tfidf[word] = calculate_tfidf(word, words, other_class_words, f, A)

    final = sorted([(k, translate(k, True), tfidf[k]) for k in tfidf], key=lambda x: x[2][0], reverse=True)
    final = [(t[0], translate(t[0], i > 20), t[2][0], t[2][1], t[2][2]) for i, t in enumerate(final)]
    return final
    return final


_translator = None
def translate(chinese_word, dummy=True):
    if dummy:
        return "<untranslated>"
    else:
        global _translator
        if _translator is None:
            _translator = ddb_trans()
        # Return the english translation of the chinese word
        return _translator.translate(chinese_word, texify=True)


def calculate_tfidf(word, words, other_class_words, f, A):
    # The frequency of the word (in class) times the information
    # content of the word (across classes).
    tf = words[word]
    ic =  np.log(1 + (A/f[word]))
    return tf * ic, tf, ic
    
    

def find_exemplars(hdbfile, umap_file, output_filename="correlated_exemplars.pkl"):
    """ The pickle file will be a list of exemplar.  
    Each exemplar is a list of (100?)  (embedding, sentence, source) triples, 
    (where source is a list of source documents)   """
    clusters = pickle.load(open(hdbfile, "rb"))
    raw_exemplars_ = clusters.exemplars_
    correlated_exemplars = correlate_exemplars(raw_exemplars_, umap_file)
    pickle.dump(correlated_exemplars, open(output_filename, 'wb'))
    return correlated_exemplars

def print_cluster_exemplars(dictionary_filename, cluster_anal, latex=True, handpicked_dict=None, hack_print=True, secondary_dictionary_file=None, words1_in_context = None):
    """
    Dictionary gives list of words in each cluster.
    Cluster anal gives descriptive statistics on each cluster, including the documents the setences in the cluster are from.
    Handpicked dict indicates whether we're using a pre-given set of clusters.
    Hack_print is as the name implies.
    Secondary dictionary file is for the triple-segment clusters (I think).
    Words1_in_context is a file containing the sentences in which the words in the single-segment clusters appear.
    """
    D = pickle.load(open(dictionary_filename, 'rb'))
    # The keys of D are cluster numbers.  The values are:
    #   0) a list of tuples of the form
    #      (chinese_word, translation, score, tf, ic)
    #   1) the set of documents from which the sentences in the
    #      cluster are drawn

    if secondary_dictionary_file is not None:
        secondary_dictionary = pickle.load(open(secondary_dictionary_file, 'rb'))

    if words1_in_context is None:
        context1 = None
    else:
        context1 = pickle.load(open(words1_in_context, "rb"))
        
        # context1 will be a list dictionaries; the ith list element
        # is for the ith topic; each dictionary has words as keys and
        # a list of occuring sentences as values.
            
    if hack_print:
        hack_clusters = []
        secondary_hack_clusters = []
        for cluster_num in D:
            if cluster_num == -1:
                continue
            chinese_prob = cluster_anal[cluster_num]['chinese_proportion']
            indian_prob = cluster_anal[cluster_num]['indian_proportion']
            monochromaticity = 2*np.abs(chinese_prob - 0.5)
            hack_clusters.append((D[cluster_num], monochromaticity, cluster_num))
            secondary_hack_clusters.append((secondary_dictionary[cluster_num], monochromaticity, cluster_num))
        hack_clusters = sorted(hack_clusters, key=lambda x: x[1], reverse=True)
        secondary_hack_clusters = sorted(secondary_hack_clusters, key=lambda x: x[1], reverse=True)
    if handpicked_dict is None:
        handpicked = False
    else:
        handpicked = True
        H = pickle.load(open(handpicked_dict, 'rb'))

    def print_dictionary(D, omit_alpha=False, hack_print=True, secondary_clusters=None, context=None):
            if hack_print:
                cdict_list = [D] if secondary_clusters is None else [D, secondary_clusters]
                cluster_numbers = range(len(D))
                for cluster_num in cluster_numbers:

                    for j in range(2):
                        cdict = cdict_list[j]
                        cluster = cdict[cluster_num][0][0]
                        docs = cdict[cluster_num][0][1]
                        cnum = cdict[cluster_num][2]

                        csize = cluster_anal[cnum]['total']
                        pchin = cluster_anal[cnum]['chinese_proportion']

                        if j == 0:
                            mc = 2*np.abs(0.5 - pchin)
                            print(f"\\section{{Cluster {cnum}; Size {csize}; Monochromaticity {mc}  }}")

                        if latex:
                            print("\\begin{table}[H]")

                            print("\\begin{tabular}[h]{l|l|l|l}")
                            print(f"Word & Score & Freq in Cluster & IC Across Clusters \\\\ \\hline")
                        for i, (word, translation, score, tf, ic) in enumerate(cluster):
                            if latex:
                                print(f"{word} ({translation}) & {score:.2f} & {tf} & {ic:.2f} \\\\")
                            else:
                                print(f"{word} ({translation}): {score}")
                            if i > 19:
                                break
                        if latex:
                            print("\\end{tabular}")
                            arity_str = "(single segment)" if j==0 else "(triple segment)"
                            print("\\caption*{" +
                                  f"cluster number {cnum} {arity_str} \\\\" +
                                  f"Size {csize}, P(chinese): {pchin}" + "}")
                            print("\\end{table}")
                            print(f"Documents: {docs}")
                            print("\\vfill\\eject")

                        if latex and context is not None and j==0 and False:  # Context
                            context_dict = context[cnum]
                            #print("\\begin{table}[H]")
                            #print("\\begin{tabular}[h]{l|l|} \\\\ \\hline ")
                            print("\\begin{longtable}{| p{.20 \\textwidth} | p{.80\\textwidth} |}")
                            print(f"Word & Sentence \\\\ \\hline")
                            for word in context_dict:
                                print(f"{word} & \\\\")
                                for sentence in context_dict[word]:
                                    print("& [ " + f"{sentence[:100]}" + " ] \\\\")
                                print("\\hline")

                            arity_str = "(single segment)" if j==0 else "(triple segment)"
                            print("\\caption*{" +
                                  f"cluster number {cnum} {arity_str} \\\\" +
                                  f"Sentential context for each word" + "}")
                            print("\\end{longtable}")
                            #print("\\end{table}")
                            print("\\vfill\\eject")

                return
                            

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
        print("\\usepackage{longtable}")
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
    if hack_print:
        print_dictionary(hack_clusters, secondary_clusters=secondary_hack_clusters, context=context1)
    else:
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

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Create report')
    parser.add_argument('--clusters_file', type=str, help='clusters file')
    parser.add_argument('--folder', type=str, help='output folder')
    parser.add_argument('--latex', action='store_true', help='Output in latex format')
    parser.add_argument('--use_handpicked', action='store_true', help='Use handpicked clusters')
    parser.add_argument('--num_jobs', type=int, default=32, help='Number of jobs to run in parallel')
    parser.add_argument('--precomputed_tops', action='store_true', help='Use precomputed tops')
    parser.add_argument('--all_clusters', action='store_true', help='Use all clusters.')
    parser.add_argument('--num_clusters', type=int, default=5, help='Number of clusters to use')
    args = parser.parse_args()
    create_report(args.clusters_file, args.folder, latex=args.latex, all_clusters=args.all_clusters, use_handpicked=args.use_handpicked, num_jobs=args.num_jobs, precomputed_tops=args.precomputed_tops)
    
if __name__ == '__main__':
    main()
