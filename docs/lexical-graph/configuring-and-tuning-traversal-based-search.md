## Configuring and Tuning Traversal-Based Search

### Overview

You can customize traversal-based search operations to better suit your specific application, dataset, and query types. The following configuration options are available to help you optimize search performance:

  - [**Search results configuration**](#search-results-configuration) Adjust the number of search results and statements returned and set scoring thresholds to filter out low-quality statements and results
  - [**Retriever selection**](#retriever-selection) Specify which retrievers to use when fetching information
  - [**Reranking strategy**](#reranking-strategy) Modify how statements and results are reranked and sorted
  - **Graph and vector search parameters** Customize parameters that control graph queries and vector searches
  - **Entity network context selection** Configure parameters used to select entity network contexts

These options allow you to fine-tune your search behavior based on your specific requirements and improve the relevance of returned results.

### Search results configuration

When configuring search functionality, you can use the following parameters to control the number and quality of returned results:

#####  `max_search_results`

Defines the maximum number of search results to return. Each search result contains one or more statements that belong to the same topic (and source). If you set this to `None`, all matching search results will be returned. The default value is `5`.

#####  `max_statements_per_topic`

Controls how many statements can be included with a single topic, effectively limiting the size of each search result. If set to `None`, all statements belonging to the topic that match the search will be included in the result. The default value is `10`.

#####  `max_statements`

Limits the total number of statements across the entire resultset. If you set this to `None`, all statements from all results will be returned. The default value is `100`.

#####  `statement_pruning_factor`

This parameter helps filter out lower-quality statements based on a percentage of the highest statement score in the entire set of results. Any statement with a score less than `<maximum_statement_score> * statement_pruning_factor` will be removed from the results. The default value is `0.1` (10% of the maximum score).

##### `statement_pruning_threshold`

Sets an absolute minimum score threshold for statements. Any statement with a score lower than this threshold will be removed from the results. The default value is `None`.

#### Example

```python
query_engine = LexicalGraphQueryEngine.for_traversal_based_search(
    graph_store, 
    vector_store,
    statement_pruning_threshold=0.2
)
```

#### When to use search results configuration

The `max_search_results`, `max_statements_per_topic` and `max_statements` parameters allow you to control the overall size of the results. 

Each search result comprises one or more statements belonging to a single topic from a single source. Statements from the same source but different topics appear as separate search results. Increasing `max_search_results` increases the variety of sources in your results. Increasing `max_statements_per_topic` adds more detail to each individual search result.

When increasing the number of statements (either overall or per topic), you should consider increasing the statement pruning parameters as well. This helps ensure that even with larger result sets, you're still getting highly relevant statements rather than less relevant information.

### Retriever selection

You can configure the traversal-based search with up to three different retrievers:

##### `ChunkBasedSearch` 

This retriever uses a vector similarity search to find information that is similar to the original query. The retriever first finds relevant chunks using vector similarity search. From these chunks, the retriever traverses topics, statements, and facts. Chunk-based search tends to return a narrowly-scoped set of results based on the statement and fact neighbourhoods of chunks that match the original query.

##### `EntityBasedSearch`

This retriever uses as its starting points the entities in an entity network context. From these entities, the retriever traverses facts, statements and topics. Entity-based search tends to return a broadly-scoped set of results, based on the neighbourhoods of individual entities and the facts that connect entities.

##### `EntityNetworkSearch` 

This retriever uses textual transcriptions of an entity network context to drive vector searches for information that is dissimilar to the original query but nonetheless structurally relevant for creating an accurate and full response. These vector searches return chunks that are similar to 'something different from the question being asked'. From these chunks, the retriever traverses topics, statements, and facts to explore the structurally relevant space of dissimilar content.
	
#### Example

```python
from graphrag_toolkit.lexical_graph.retrieval.retrievers import *

query_engine = LexicalGraphQueryEngine.for_traversal_based_search(
    graph_store, 
    vector_store,
    retrievers=[ChunkBasedSearch, EntityBasedSearch]
)
```

#### When to use different retrievers

By default, traversal-based search is configured to use a combination of `ChunkBasedSearch` and `EntityNetworkSearch`. This combination provides access to content that is both directly similar to the question and content that may be relevant but not explicitly mentioned in the query.

Consider using the `ChunkBasedSearch` retriever by itself if:

  - Your queries need primarily similarity-based search
  - You want to focus on individual relevant statements rather than entire chunks
  - You need broader search scope than traditional vector search

This retriever uses local connectivity to find relevant statements in other chunks from the same source, expanding beyond basic vector similarity.

The `EntityBasedSearch` and `EntityNetworkSearch` retrievers provide different ways of utilising entity networks in a search:

  - The `EntityBasedSearch` uses global connectivity to find statements from different sources connected by the same facts. It often produces more diverse results than other retrievers. 
   - The `EntityNetworkSearch` retriever converts an entity network (retrieved through graph traversal) into a set of similarity searches. This approach balances global and local connectivity.

### Reranking strategy

Traversal-based search incorporates reranking at two key points during the retrieval process:

  - When generating entity network contexts, both entities and entity networks are reranked
  - Before finalizing search results, the complete set of statements undergoes reranking

Reranking is managed through a single parameter:

#####  `reranker`

Parameters options:

  - `model`: Uses a LlamaIndex-based `SentenceReranker` to rerank all statements in the result set.
  - `tfidf` (default): Applies a term frequency-inverse document frequency measure to rank statements. This option is significantly faster than the model-based approach.
  - `None`: Disables the reranking feature completely.

To use the model reranker, you must install the following additional dependencies:

```
pip install torch sentence_transformers
```

#### Example

```python
query_engine = LexicalGraphQueryEngine.for_traversal_based_search(
    graph_store, 
    vector_store,
    reranker='model'
)
```


### Graph and vector search parameters

### Entity network context selection