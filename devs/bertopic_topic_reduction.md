# BERTopic Topic Reduction via HDBSCAN on c-TF-IDF

How BERTopic reduces the number of topics by clustering topic-level c-TF-IDF vectors, based on the source code in `BERTopic - CLONE/`.

---

## Overview

After the initial HDBSCAN clustering of documents, BERTopic often produces more topics than desired. The library offers two strategies to reduce them — both operate on **topic vectors**, not on individual documents. The default representation used is the **c-TF-IDF matrix**.

---

## 1. How c-TF-IDF Topic Vectors Are Built

Source: [`bertopic/vectorizers/_ctfidf.py`](../BERTopic%20-%20CLONE/bertopic/vectorizers/_ctfidf.py), [`bertopic/_bertopic.py`](../BERTopic%20-%20CLONE/bertopic/_bertopic.py) (methods `_c_tf_idf()`, `_extract_topics()`)

### Step 1: Group documents by topic

All documents assigned to the same topic are concatenated into a single mega-document:

```python
documents_per_topic = documents.groupby("Topic").agg({"Document": " ".join})
```

### Step 2: Bag-of-words via CountVectorizer

Each mega-document is tokenized into unigrams (default). Output: sparse matrix of shape `(n_topics, n_vocabulary)` with raw word counts.

### Step 3: c-TF-IDF weighting

The `ClassTfidfTransformer` applies:

```
c-TF-IDF[i, j] = TF[i, j]  x  log( avg_words_per_topic / df[j]  +  1 )
```

Where:
- **TF[i, j]** = L1-normalized frequency of word `j` in topic `i` (each topic's word frequencies sum to 1)
- **df[j]** = total frequency of word `j` across all topics
- **avg_words_per_topic** = mean total words per mega-document

The result is a **sparse matrix of shape `(n_topics, n_vocabulary)`** — one row per topic, each dimension representing a word's importance to that topic.

---

## 2. Topic Reduction Strategies

Source: [`bertopic/_bertopic.py`](../BERTopic%20-%20CLONE/bertopic/_bertopic.py) (methods `_reduce_to_n_topics()`, `_auto_reduce_topics()`)

### Strategy A: Reduce to N topics (Agglomerative Clustering)

Called when the user specifies a target number of topics.

1. Extract topic representations (c-TF-IDF rows by default)
2. Compute cosine distance matrix: `D = 1 - cosine_similarity(c_tf_idf)`
3. Run `AgglomerativeClustering(n_clusters=target, metric="precomputed", linkage="average")`
4. Map original topic IDs to new cluster labels

```python
# From _reduce_to_n_topics(), approx line 4435
topic_embeddings = select_topic_representation(
    self.c_tf_idf_, self.topic_embeddings_, use_ctfidf, output_ndarray=True
)[0][self._outliers:]

distance_matrix = 1 - cosine_similarity(topic_embeddings)
np.fill_diagonal(distance_matrix, 0)

cluster = AgglomerativeClustering(
    self.nr_topics - self._outliers,
    metric="precomputed",
    linkage="average"
)
cluster.fit(distance_matrix)
```

### Strategy B: Automatic reduction (HDBSCAN on topic vectors)

Called when the user passes `nr_topics="auto"`. This runs HDBSCAN on the **topic-level** c-TF-IDF vectors (not the original document embeddings).

1. Extract topic representations (c-TF-IDF rows by default)
2. L2-normalize the vectors
3. Run HDBSCAN with `min_cluster_size=2` on the normalized topic vectors
4. Topics that HDBSCAN clusters together get merged; noise topics (label -1) remain separate

```python
# From _auto_reduce_topics(), approx line 4490
embeddings = select_topic_representation(
    self.c_tf_idf_, self.topic_embeddings_, use_ctfidf, output_ndarray=True
)[0]

norm_data = normalize(embeddings, norm="l2")

predictions = HDBSCAN(
    min_cluster_size=2,
    metric="euclidean",
    cluster_selection_method="eom",
    prediction_data=True,
).fit_predict(norm_data[self._outliers:])
```

The intuition: if two topics have similar word distributions (close c-TF-IDF vectors), HDBSCAN will cluster them together, and they get merged.

---

## 3. Post-Merge Recalculation

Source: [`bertopic/_bertopic.py`](../BERTopic%20-%20CLONE/bertopic/_bertopic.py) (methods `_extract_topics()`, `_create_topic_vectors()`, `_map_probabilities()`, `_sort_mappings_by_frequency()`)

After topics are merged, BERTopic performs a full recalculation:

### 3a. Reorder by frequency

Topics are renumbered so that the largest topic becomes topic 0, second largest becomes topic 1, etc. Outliers stay at -1.

### 3b. Recalculate c-TF-IDF from scratch

Documents are re-grouped by the new merged topic assignments, and c-TF-IDF is computed fresh — not simply averaged from the old vectors:

```python
# Re-group and re-compute
documents_per_topic = documents.groupby("Topic").agg({"Document": " ".join})
self.c_tf_idf_, words = self._c_tf_idf(documents_per_topic)
```

This ensures the merged topic's word distribution accurately reflects the combined document set.

### 3c. Update topic embeddings (weighted average)

Semantic topic embeddings (the dense vector type) are updated via weighted average, where weights are original topic sizes:

```python
embds = np.array(self.topic_embeddings_)[np.array(topic_ids) + self._outliers]
topic_embedding = np.average(embds, axis=0, weights=topic_sizes)
```

### 3d. Aggregate HDBSCAN probabilities

If soft clustering probabilities exist, they are summed across merged topics:

```python
# If T0 and T1 merge into T0_new:
# prob_new[doc, T0_new] = prob[doc, T0] + prob[doc, T1]
for from_topic, to_topic in mappings.items():
    if to_topic != -1 and from_topic != -1:
        mapped_probabilities[:, to_topic] += probabilities[:, from_topic]
```

---

## 4. Hierarchical Topic View

Source: [`bertopic/_bertopic.py`](../BERTopic%20-%20CLONE/bertopic/_bertopic.py) (method `hierarchical_topics()`)

For visualization purposes, BERTopic also supports building a full hierarchy using scipy's linkage:

```python
distance_function = lambda x: 1 - cosine_similarity(x)
linkage_function = lambda x: sch.linkage(x, "ward", optimal_ordering=True)

embeddings = select_topic_representation(
    self.c_tf_idf_, self.topic_embeddings_, use_ctfidf
)[0][self._outliers:]

X = distance_function(embeddings)
Z = linkage_function(X)
```

This produces a dendrogram showing which topics would merge first (most similar c-TF-IDF vectors) through to last.

---

## 5. Representation Selection

Source: [`bertopic/_utils.py`](../BERTopic%20-%20CLONE/bertopic/_utils.py) (function `select_topic_representation()`)

All reduction methods accept a `use_ctfidf` parameter (default `True`) to choose between:

| `use_ctfidf` | Representation | Shape | Similarity metric |
|---|---|---|---|
| `True` (default) | c-TF-IDF sparse vectors | `(n_topics, n_vocabulary)` | Cosine similarity |
| `False` | Semantic embeddings (mean of document embeddings) | `(n_topics, embedding_dim)` | Cosine similarity (strategy A) / Euclidean on L2-normed vectors (strategy B) |

---

## 6. Key Design Decisions

- **Outliers (topic -1) are never merged** — always excluded from reduction via `[self._outliers:]` slicing
- **c-TF-IDF is fully recomputed** after merging (not averaged), so merged topics get accurate word distributions
- **Single-pass clustering**, not iterative pairwise merging — Agglomerative Clustering or HDBSCAN determines all merges at once
- **No stop word removal by default** — the CountVectorizer keeps all tokens of 2+ characters; topic-specificity comes from the IDF weighting instead

---

## Source File Reference

| File | Key contents |
|---|---|
| [`bertopic/_bertopic.py`](../BERTopic%20-%20CLONE/bertopic/_bertopic.py) | Main BERTopic class: `reduce_topics()`, `_reduce_to_n_topics()`, `_auto_reduce_topics()`, `_extract_topics()`, `_c_tf_idf()`, `_create_topic_vectors()`, `_map_probabilities()` |
| [`bertopic/vectorizers/_ctfidf.py`](../BERTopic%20-%20CLONE/bertopic/vectorizers/_ctfidf.py) | `ClassTfidfTransformer`: the c-TF-IDF formula (fit/transform) |
| [`bertopic/cluster/_utils.py`](../BERTopic%20-%20CLONE/bertopic/cluster/_utils.py) | HDBSCAN delegation utilities: `hdbscan_delegator()`, soft clustering helpers |
| [`bertopic/_utils.py`](../BERTopic%20-%20CLONE/bertopic/_utils.py) | `select_topic_representation()`: chooses between c-TF-IDF and semantic embeddings |