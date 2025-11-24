import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
def load_corpus_20news(max_docs=2000):
    data = fetch_20newsgroups(
        subset="train",
        remove=("headers", "footers", "quotes")
    )
    texts = data.data[:max_docs]
    return texts

def preprocess_corpus(texts, min_df=10):
    vectorizer = CountVectorizer(
        lowercase=True,
        stop_words="english",
        min_df=min_df
    )
    X = vectorizer.fit_transform(texts)  # shape: (D, V)
    vocab = np.array(vectorizer.get_feature_names_out())
    D, V = X.shape

    # Convert sparse doc-word counts to per-token word-id lists
    docs_word_ids = []
    for d in range(D):
        row = X.getrow(d)
        word_indices = row.indices
        counts = row.data
        tokens = []
        for w, c in zip(word_indices, counts):
            tokens.extend([w] * c)
        docs_word_ids.append(tokens)

    # For UMass coherence: docs that contain each word
    docs_with_word = [set() for _ in range(V)]
    for d in range(D):
        row = X.getrow(d)
        for w in row.indices:
            docs_with_word[w].add(d)

    return docs_word_ids, vocab, docs_with_word
class LDAGibbsSampler:
    def __init__(self, docs_word_ids, V, K=10, alpha=0.1, beta=0.1,
                 iterations=150, random_state=0):
        rng = np.random.RandomState(random_state)

        self.docs = docs_word_ids
        self.V = V
        self.D = len(docs_word_ids)
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        self.rng = rng

        # Topic assignment for each token: list of list
        self.z = []

        # Counts
        self.ndk = np.zeros((self.D, self.K), dtype=np.int32)   # n_d(k)
        self.mkv = np.zeros((self.K, self.V), dtype=np.int32)   # m_k(v)
        self.mk = np.zeros(self.K, dtype=np.int32)              # m_k

        self._initialize()

    def _initialize(self):
        # Randomly assign topics to each token
        for d, doc in enumerate(self.docs):
            doc_topics = []
            for w in doc:
                k = self.rng.randint(self.K)
                doc_topics.append(k)
                self.ndk[d, k] += 1
                self.mkv[k, w] += 1
                self.mk[k] += 1
            self.z.append(doc_topics)

    def run(self, verbose=True):
        Vbeta = self.V * self.beta
        for it in range(self.iterations):
            for d, doc in enumerate(self.docs):
                for i, w in enumerate(doc):
                    k_old = self.z[d][i]

                    # remove old assignment
                    self.ndk[d, k_old] -= 1
                    self.mkv[k_old, w] -= 1
                    self.mk[k_old] -= 1

                    # compute q_k ∝ (α + n_dk^-dj) * (β + m_kv^-dj) / (Vβ + m_k^-dj)
                    left = self.alpha + self.ndk[d]          # shape (K,)
                    right_num = self.beta + self.mkv[:, w]   # shape (K,)
                    right_den = Vbeta + self.mk
                    q = left * right_num / right_den

                    # normalize to probabilities
                    q_sum = q.sum()
                    if q_sum == 0:
                        q = np.ones(self.K) / self.K
                    else:
                        q = q / q_sum

                    # sample new topic
                    k_new = self.rng.choice(self.K, p=q)
                    self.z[d][i] = k_new

                    # add new assignment
                    self.ndk[d, k_new] += 1
                    self.mkv[k_new, w] += 1
                    self.mk[k_new] += 1

            if verbose and (it + 1) % 10 == 0:
                print(f"Iteration {it + 1}/{self.iterations} finished")

    def estimate_phi_theta(self):
        # φ_kv = (m_kv + β) / (m_k + Vβ)
        phi = (self.mkv + self.beta) / (self.mk[:, None] + self.V * self.beta)

        # θ_dk = (n_dk + α) / (N_d + Kα)
        theta = np.zeros_like(self.ndk, dtype=float)
        for d, doc in enumerate(self.docs):
            Nd = len(doc)
            theta[d] = (self.ndk[d] + self.alpha) / (Nd + self.K * self.alpha)

        return phi, theta

    def top_words(self, phi, vocab, top_n=20):
        topics = []
        for k in range(self.K):
            top_indices = np.argsort(phi[k])[::-1][:top_n]
            topics.append([(vocab[i], float(phi[k, i])) for i in top_indices])
        return topics
def umass_coherence_for_topic(word_indices, docs_with_word):
    M = len(word_indices)
    if M < 2:
        return 0.0

    score = 0.0
    count_pairs = 0

    for m in range(1, M):
        w_m = word_indices[m]
        docs_m = docs_with_word[w_m]
        for l in range(m):
            w_l = word_indices[l]
            docs_l = docs_with_word[w_l]
            D_wl = len(docs_l)
            if D_wl == 0:
                continue
            D_wmwl = len(docs_m & docs_l)
            score += np.log((D_wmwl + 1.0) / D_wl)
            count_pairs += 1

    if count_pairs == 0:
        return 0.0
    return score / count_pairs

def compute_topic_coherences(topics, vocab, docs_with_word):
    coherences = []
    for topic in topics:
        word_indices = [np.where(vocab == w)[0][0] for w, _ in topic]
        c = umass_coherence_for_topic(word_indices, docs_with_word)
        coherences.append(c)
    return np.array(coherences)
def run_experiments():
    texts = load_corpus_20news(max_docs=2000)
    docs_word_ids, vocab, docs_with_word = preprocess_corpus(texts, min_df=10)
    V = len(vocab)
    print(f"Documents: {len(docs_word_ids)}, Vocabulary size: {V}")

    settings = [
        (10, 0.1, 0.1),
        (50, 0.1, 0.1),
        (10, 0.01, 0.01),
        (50, 0.01, 0.01),
    ]

    for K, alpha, beta in settings:
        print("\n" + "=" * 60)
        print(f"Running LDA with K={K}, alpha={alpha}, beta={beta}")
        lda = LDAGibbsSampler(
            docs_word_ids,
            V,
            K=K,
            alpha=alpha,
            beta=beta,
            iterations=150,
            random_state=0
        )
        lda.run(verbose=True)
        phi, theta = lda.estimate_phi_theta()
        topics = lda.top_words(phi, vocab, top_n=20)
        coherences = compute_topic_coherences(topics, vocab, docs_with_word)

        # Sort topics by coherence (descending)
        order = np.argsort(coherences)[::-1]

        print("\nTop 5 topics by UMass coherence:")
        for idx in order[:5]:
            print(f"\nTopic {idx} (coherence={coherences[idx]:.3f}):")
            words = [w for (w, p) in topics[idx]]
            print(", ".join(words))

        # Simple visual evaluation: coherence bar plot
        plt.figure()
        plt.bar(range(K), coherences)
        plt.xlabel("Topic index")
        plt.ylabel("UMass coherence")
        plt.title(f"K={K}, alpha=beta={alpha}")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    run_experiments()
