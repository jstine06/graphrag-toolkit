## Configuring and Tuning Traversal-Based Search

### Overview

You can customize traversal-based search operations to better suit your specific application, dataset, and query types. The following configuration options are available to help you optimize search performance:

  - *Search results configuration* Adjust the number of search results and statements returned and set scoring thresholds to filter out low-quality statements and results
  - *Retriever selection* Specify which retrievers to use when fetching information
  - *Reranking strategy* Modify how statements and results are reranked and sorted
  - *Graph and vector search parameters* Customize parameters that control graph queries and vector searches
  - *Entity network context selection* Configure parameters used to select entity network contexts

These options allow you to fine-tune your search behavior based on your specific requirements and improve the relevance of returned results.

### Search results configuration

When configuring search functionality, you can use the following parameters to control the number and quality of returned results:

####  `max_search_results`

Defines the maximum number of search results to return. Each search result contains one or more statements that belong to the same topic (and source). If you set this to `None`, all matching search results will be returned. The default value is `5`.

####  `max_statements_per_topic`

Controls how many statements can be included with a single topic, effectively limiting the size of each search result. If set to `None`, all statements belonging to the topic that match the search will be included in the result. The default value is `10`.

####  `max_statements`

Limits the total number of statements across the entire resultset. If you set this to `None`, all statements from all results will be returned. The default value is `100`.

####  `statement_pruning_factor`

This parameter helps filter out lower-quality statements based on a percentage of the highest statement score in the entire set of results. Any statement with a score less than `<maximum_statement_score> * statement_pruning_factor` will be removed from the results. The default value is `0.1` (10% of the maximum score).

#### `statement_pruning_threshold`

Sets an absolute minimum score threshold for statements. Any statement with a score lower than this threshold will be removed from the results. The default value is `None`.

### Retriever selection

### Reranking strategy

### Graph and vector search parameters

### Entity network context selection