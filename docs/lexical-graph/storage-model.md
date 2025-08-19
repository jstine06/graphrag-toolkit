[[Home](./)]

## Storage Model

### Topics

- [Overview](#overview)
- [Graph store](#graph-store)
- [Vector store](#vector-store)

### Overview

The lexical-graph uses two separate stores: a `GraphStore` and a `VectorStore`. A `VectorStore` acts as a container for a collection of `VectorIndex`. When constructing or querying a graph, you must provide instances of both a graph store and vector store.

The toolkit provides graph store implementations for both [Amazon Neptune Analytics](https://docs.aws.amazon.com/neptune-analytics/latest/userguide/what-is-neptune-analytics.html) and [Amazon Neptune Database](https://docs.aws.amazon.com/neptune/latest/userguide/intro.html) (engine version 1.4.1.0 or later), and now [FalkorDB](https://docs.falkordb.com/)**,** along with vector store implementations for Neptune Analytics, [Amazon OpenSearch Serverless](https://docs.aws.amazon.com/opensearch-service/latest/developerguide/serverless.html) and Postgres with the pgvector extension. The lexical-graph provides several convenient factory methods for creating instances of these stores.

> This early release of the toolkit provides support for Amazon Neptune and Amazon OpenSearch Serverless, but we welcome alternative store implementations. The store APIs and the ways in which the stores are used have been designed to anticipate alternative implementations. However, the proof is in the development: if you experience issues developing an alternative store, [let us know](https://github.com/awslabs/graphrag-toolkit/issues).

Graph stores and vector stores provide connectivity to an *existing* storage instance, which you will need to have provisioned beforehand.

#### Code examples

The code examples here are formatted to run in a Jupyter notebook. If you’re building an application with a main entry point, put your application logic inside a method, and add an [`if __name__ == '__main__'` block](./faq.md#runtimeerror-please-use-nest_asyncioapply-to-allow-nested-event-loops).

### Graph store

Graph stores must support the [openCypher](https://opencypher.org/) property graph query language. Graph construction queries typically use an `UNWIND ... MERGE` idiom to create or update the graph for a [batch of inputs](https://docs.aws.amazon.com/neptune-analytics/latest/userguide/best-practices-content.html#best-practices-content-14). The Neptune graph store implementations override the `GraphStore.node_id()` method to ensure that node ids in the code (e.g. `chunkId`) are mapped to Neptune's `~id` reserved property. Alternative graph store implementations can leave the base implementation of `node_id()` as-is. This will result in node ids being mapped to a property of the same name (i.e. a reference to `chunkId` in the code will be mapped to a `chunkId` property of a node).

You use the `GraphStoreFactory.for_graph_store()` static factory method to create a graph store.

The lexical-graph supports the following graph databases:

  - [Amazon Neptune](./graph-store-neptune-db.md)
  - [Amazon Neptune Analytics](./graph-store-neptune-analytics.md)
  - [Neo4j](./graph-store-neo4j.md)
  - [FalkorDB](./graph-store-falkor-db.md)

### Vector store

A vector store is a collection of vector indexes. The lexical-graph uses up to two vector indexes: a chunk index and a statement index. The chunk index is typically much smaller than the statement index. If you want to use the [SemanticGuidedRetriever](./querying.md#semanticguidedretriever), you will need to enable the statement index. If you want to use the [TraversalBasedRetriever](./querying.md#traversalbasedretriever), you will need to enable the chunk index. If you want to use both retrievers, you will need to enable both indexes. (The `VectorStoreFactory` described below enables both indexes by default.)

You use the `VectorStoreFactory.for_vector_store()` static factory method to create a vector store.

The lexical-graph supports the following vector-stores:

  - [Amazon OpenSearch Serverless](./vector-store-opensearch-serverless.md)
  - [Amazon Neptune Analytics](./vector-store-neptune-analytics.md)
  - [Postgres with the pgvector extension](./vector-store-postgres.md)

By default, the `VectorStoreFactory` will enable both the statement index and the chunk index. If you want to enable just one of the indexes, pass an `index_names` argument to the factory method:

```python
vector_store = VectorStoreFactory.for_vector_store(opensearch_connection_info, index_names=['chunk'])
```

