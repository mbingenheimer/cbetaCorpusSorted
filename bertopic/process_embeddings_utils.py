import numpy as np
import pickle


def cluster_analysis(cluster_labels, num_chinese, num_indian, total):
    """
    Descriptive statistic on the cluters.
      The number of sentences which are not clustered, and how many in each sub-corpus.
      For each cluster, the number and proportion in each sub-corpus..
    
    """
    sums = np.zeros(cluster_labels.max()+1)
    num_chinese_per_cluster = np.zeros(cluster_labels.max()+1)
    num_indian_per_cluster = np.zeros(cluster_labels.max()+1)

    num_clusterless = 0
    num_chinese_clusterless = 0
    num_indian_clusterless = 0

    for i in range(len(cluster_labels)):
        if cluster_labels[i] != -1:
            sums[cluster_labels[i]] += 1
            if i < num_chinese:
                num_chinese_per_cluster[cluster_labels[i]] += 1
            else:
                num_indian_per_cluster[cluster_labels[i]] += 1
        else:
            num_clusterless += 1
            if i < num_chinese:
                num_chinese_clusterless += 1
            else:
                num_indian_clusterless += 1

    results = { -1: {"total": num_clusterless, "chinese": num_chinese_clusterless, "indian": num_indian_clusterless, "chinese_proportion": num_chinese_clusterless / num_clusterless, "indian_proportion": num_indian_clusterless / num_clusterless }}
    for i in range(cluster_labels.max()+1):
        results[i] = { "total": sums[i], "chinese": num_chinese_per_cluster[i], "indian": num_indian_per_cluster[i], "chinese_proportion": num_chinese_per_cluster[i] / sums[i], "indian_proportion": num_indian_per_cluster[i] / sums[i] }
    return results


def get_all_points(repeats=True, sentences=False, get_key_list=False, get_doc_ids=False,
                   chinese_sentences_filename='collated_chinese_sentence_embeddings.pkl',
                   indian_sentences_filename='collated_indian_sentence_embeddings.pkl'):
    key_list = []
    if sentences:
        chinese_embeddings, indian_embeddings = load_collated_embeddings(chinese_filename=chinese_sentences_filename, indian_filename=indian_sentences_filename)
    else:
        chinese_embeddings, indian_embeddings = load_collated_embeddings()
    chinese_points = []
    num_chinese = 0
    num_indian = 0
    total = 0
    all_points = []
    doc_ids = []

    # Get the raw embeddings
    #print("Getting Chinese embeddings")
    for key in chinese_embeddings:
        if repeats:
            raise Exception("Not implemented")
            all_points.extend([*(chinese_embeddings[key][0] * [chinese_embeddings[key][1]])])
            if get_key_list:
                key_list.extend([key] * len(chinese_embeddings[key][0]))
        else:
            all_points.append(chinese_embeddings[key][0])
            if get_key_list:
                key_list.append(key)
            if get_doc_ids:
                doc_ids.append(chinese_embeddings[key][2])

    num_chinese = len(all_points)
    #chinese_points = np.array(chinese_points)
    #indian_points = []
    #print("Getting Indian embeddings")
    for key in indian_embeddings:
        if repeats:
            raise Exception("Not implemented")
            all_points.extend([*(indian_embeddings[key][0] * [indian_embeddings[key][1]])])
            if get_key_list:
                key_list.extend([key] * len(indian_embeddings[key][0]))
        else:
            all_points.append(indian_embeddings[key][0])
            if get_key_list:
                key_list.append(key)
            if get_doc_ids:
                doc_ids.append(indian_embeddings[key][2])
    #indian_points = np.array(indian_points)
    total = len(all_points)
    num_indian = total - num_chinese
    if get_key_list:
        if get_doc_ids:
            return np.array(all_points), num_chinese, num_indian, total, key_list, doc_ids
        else:
            return np.array(all_points), num_chinese, num_indian, total, key_list
    else:
        return np.array(all_points), num_chinese, num_indian, total

def load_collated_embeddings(chinese_filename='collated_chinese_word_embeddings.pkl', indian_filename='collated_indian_word_embeddings.pkl'):
    """ Load collated embeddings """
    with open(chinese_filename, 'rb') as f:
        chinese_embeddings = pickle.load(f)
    with open(indian_filename, 'rb') as f:
        indian_embeddings = pickle.load(f)
    return chinese_embeddings, indian_embeddings
