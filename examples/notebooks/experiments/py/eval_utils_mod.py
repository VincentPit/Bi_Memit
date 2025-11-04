import typing
from itertools import chain

import nltk
import numpy as np
import scipy
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForCausalLM, AutoTokenizer

from dsets import AttributeSnippets
from util.generate import generate_fast
from util.perplexity import perplexity
from sklearn.metrics.pairwise import cosine_similarity

def compute_rewrite_quality_mod(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    record: typing.Dict,
    snips: AttributeSnippets,
    vec: TfidfVectorizer,
) -> typing.Dict:
    # Unpack rewrite evaluation record
    subject, target_new, target_true = (
        record["requested_rewrite"][x] for x in ["subject", "target_new", "target_true"]
    )
    
    
    obj, subject_new, subject_true = (
        record["requested_reverse_rewrite"][x] for x in ["subject", "target_new", "target_true"]
    )

    # Prepare prompts
    rewrite_prompts = [record["requested_rewrite"]["prompt"].format(subject)]
    reverse_rewrite_prompts = [record["requested_reverse_rewrite"]["prompt"].format(obj)]
    paraphrase_prompts = record["paraphrase_prompts"]
    reverse_paraphrase_prompts = record["reverse_paraphrase_prompts"]
    local_prompts = record["local_prompts"]
    relation_prompt = record["relation_prompt"]
    general_prompts = record["general_prompts"]

    # List of prompt types
    prob_prompts = [
        rewrite_prompts,
        paraphrase_prompts,
        reverse_paraphrase_prompts,
        local_prompts,
        relation_prompt,
        general_prompts,
    ]
    reverse_prob_prompts = [reverse_rewrite_prompts, reverse_paraphrase_prompts]

    # Targets for each prompt
    which_correct = [
        [0 for _ in range(len(rewrite_prompts))],
        [0 for _ in range(len(paraphrase_prompts))],
        [0 for _ in range(len(reverse_paraphrase_prompts))],
        [0 for _ in range(len(local_prompts))],
        [0 for _ in range(len(relation_prompt))],
        [0 for _ in range(len(general_prompts))],
    ]
    reverse_correct = [
        [1 for _ in range(len(reverse_rewrite_prompts))],
        [1 for _ in range(len(reverse_paraphrase_prompts))],
    ]

    # Evaluate forward prompts
    probs, targets_correct = test_batch_prediction(
        model,
        tok,
        list(chain(*prob_prompts)),
        list(chain(*which_correct)),
        target_new["str"],
        target_true["str"],
    )

    # Evaluate reverse prompts
    reverse_probs, reverse_targets_correct = test_batch_prediction(
        model,
        tok,
        list(chain(*reverse_prob_prompts)),
        list(chain(*reverse_correct)),
        subject_true["str"],
        subject_new["str"],
    )

    # Unflatten results
    cutoffs = [0] + np.cumsum(list(map(len, prob_prompts))).tolist()
    ret_probs = [probs[cutoffs[i - 1]:cutoffs[i]] for i in range(1, len(cutoffs))]
    ret_corrects = [
        targets_correct[cutoffs[i - 1]:cutoffs[i]] for i in range(1, len(cutoffs))
    ]
    reverse_cutoffs = [0] + np.cumsum(list(map(len, reverse_prob_prompts))).tolist()
    reverse_ret_probs = [
        reverse_probs[reverse_cutoffs[i - 1]:reverse_cutoffs[i]] for i in range(1, len(reverse_cutoffs))
    ]
    reverse_ret_corrects = [
        reverse_targets_correct[reverse_cutoffs[i - 1]:reverse_cutoffs[i]]
        for i in range(1, len(reverse_cutoffs))
    ]

    # Structure return metrics
    ret = {
        f"{key}_probs": ret_probs[i]
        for i, key in enumerate(
            [
                "rewrite_prompts",
                "paraphrase_prompts",
                "local_prompts",
                "relation_prompt",
                "general_prompts",
            ]
        )
    } | {
        f"{key}_correct": ret_corrects[i]
        for i, key in enumerate(
            [
                "rewrite_prompts",
                "paraphrase_prompts",
                "local_prompts",
                "relation_prompt",
                "general_prompts",
            ]
        )
    } | {
        f"reverse_{key}_probs": reverse_ret_probs[i]
        for i, key in enumerate(["rewrite_prompts", "paraphrase_prompts"])
    } | {
        f"reverse_{key}_correct": reverse_ret_corrects[i]
        for i, key in enumerate(["rewrite_prompts", "paraphrase_prompts"])
    }

    # Add generation statistics if snippets provided
    if snips is not None:
        #print('record["requested_rewrite"]:', record["requested_rewrite"])
        rel_id = record["requested_rewrite"]["relation_id"]
        #print("rel_id:", rel_id)
        consistency_texts = [x["text"] for x in snips[rel_id][target_new["id"]]]
        print("""snips[rel_id]:""", snips[rel_id][target_new["id"]])                
        print("""target_new["id"]:""", target_new["id"])
        #print()
        essence_texts = [
            x["text"]
            for x in snips[rel_id][target_new["id"]]
            if x["name"] == record["requested_rewrite"]["subject"]
        ]
        assert len(consistency_texts) > 0, f"Consistency texts required for evaluation"
        gen_stats = test_generation(
            model,
            tok,
            record["generation_prompts"],
            consistency_texts,
            essence_texts,
            vec,
        )
        ret.update(gen_stats)

    return ret



def test_batch_prediction(
    model,
    tok,
    prefixes: typing.List[str],
    which_correct: typing.List[int],
    target_new: str,
    target_true: str,
):
    """
    Function to evaluate prediction correctness for new or true target options
    :param model: Model used for evaluation
    :param tok: Tokenizer for text processing
    :param prefixes: List of input prompts
    :param which_correct: Which target (new or true) should be considered correct
    :param target_new: The 'new' target string
    :param target_true: The 'true' target string
    :return: Evaluated probabilities and correctness
    """
    prefix_lens = [len(n) for n in tok(prefixes)["input_ids"]]
    prompt_tok = tok(
        [
            f"{prefix} {suffix}"
            for prefix in prefixes
            for suffix in [target_new, target_true]
        ],
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    a_tok, b_tok = (tok(f" {n}")["input_ids"] for n in [target_new, target_true])
    choice_a_len, choice_b_len = (len(n) for n in [a_tok, b_tok])

    with torch.no_grad():
        logits = model(**prompt_tok).logits

    probs = np.zeros((logits.size(0),), dtype=np.float32)
    targets_correct = []

    for i in range(logits.size(0)):
        cur_len = choice_a_len if i % 2 == 0 else choice_b_len

        # Compute suffix probabilities
        for j in range(cur_len):
            cur_tok = (a_tok if i % 2 == 0 else b_tok)[j]
            probs[i] += -torch.nn.functional.log_softmax(
                logits[i, prefix_lens[i // 2] + j - 1, :], dim=0
            )[cur_tok].item()
        probs[i] /= cur_len

        # Compute accuracy on new targets
        if (which_correct[i // 2] == 0 and i % 2 == 0) or (
            which_correct[i // 2] == 1 and i % 2 == 1
        ):
            correct = True
            for j in range(cur_len):
                cur_tok = (a_tok if i % 2 == 0 else b_tok)[j]

                if logits[i, prefix_lens[i // 2] + j - 1, :].argmax().item() != cur_tok:
                    correct = False
                    break
            targets_correct.append(correct)

    return [
        {"target_new": probs[i].item(), "target_true": probs[i + 1].item()}
        for i in range(0, len(probs), 2)
    ], targets_correct


def test_generation(
    model,
    tok,
    prefixes: typing.List[str],
    consistency_texts: typing.List[str],
    essence_texts: typing.List[str],
    vec: TfidfVectorizer,
):
    """
    Function for evaluating text generation based on consistency, perplexity, and n-gram entropy
    :param model: Model for generating text
    :param tok: Tokenizer for text processing
    :param prefixes: List of prompts for generation
    :param consistency_texts: Texts used for consistency comparison
    :param essence_texts: Texts used for essence evaluation
    :param vec: TF-IDF Vectorizer for text similarity
    :return: Dictionary containing generation statistics
    """
    gen_texts = generate_fast(
        model,
        tok,
        prefixes,
        n_gen_per_prompt=1,
        max_out_len=100,
    )

    ngram_entropy = n_gram_entropy(gen_texts)
    consistency_tfidf = tfidf_similarity(
        " ".join(gen_texts), " ".join(consistency_texts), vec
    )

    ret = {
        "ngram_entropy": ngram_entropy,
        "consistency_tfidf": consistency_tfidf,
    }
    return ret


def n_gram_entropy(gen_texts: typing.List[str], n=2):
    """
    Calculate n-gram entropy of generated texts.
    :param gen_texts: List of generated text
    :param n: n-gram size
    :return: Entropy of n-grams in generated texts
    """
    ngrams = [list(nltk.ngrams(tok, n)) for tok in gen_texts]
    all_ngrams = list(chain(*ngrams))
    freq_dist = nltk.FreqDist(all_ngrams)
    ngram_entropy = -sum(
        (count / len(all_ngrams)) * np.log(count / len(all_ngrams))
        for count in freq_dist.values()
    )
    return ngram_entropy


def tfidf_similarity(text_a: str, text_b: str, vec: TfidfVectorizer):
    """
    Compute the cosine similarity between two texts using TF-IDF vectors.
    :param text_a: First text
    :param text_b: Second text
    :param vec: TF-IDF Vectorizer
    :return: Similarity score
    """
    vecs = vec.transform([text_a, text_b])
    similarity = cosine_similarity(vecs[0], vecs[1])
    return similarity[0][0]

