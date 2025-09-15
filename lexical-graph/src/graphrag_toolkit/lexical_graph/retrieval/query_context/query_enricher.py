# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
from typing import List

from graphrag_toolkit.lexical_graph.storage.graph import GraphStore
from graphrag_toolkit.lexical_graph.storage.vector import VectorStore
from graphrag_toolkit.lexical_graph.retrieval.processors import ProcessorArgs
from graphrag_toolkit.lexical_graph.utils.reranker_utils import score_values_with_tfidf

from llama_index.core.schema import QueryBundle

logger = logging.getLogger(__name__)

class QueryEnricher():
    
    def __init__(self, graph_store:GraphStore, vector_store:VectorStore, args:ProcessorArgs):
        self.graph_store = graph_store
        self.vector_store = vector_store
        self.args = args


    def _get_top_statement_id(self, query_bundle:QueryBundle) -> str:
        
        logger.debug(f"Query: {query_bundle.query_str}")
        
        top_k_results = self.vector_store.get_index('chunk').top_k(query_bundle, 1)
        chunk_ids = [r['chunk']['chunkId'] for r in top_k_results]
        
        logger.debug(f"Chunk ids: {chunk_ids}")
        
        cypher = f'''
        // Get statements for top chunk
        MATCH (c:`__Chunk__`)<-[:`__MENTIONED_IN__`]-(s:`__Statement__`)
        WHERE {self.graph_store.node_id("c.chunkId")} in $chunkIds 
        RETURN {{
            statement: s.value,
            statementId: id(s)
        }} AS result
        '''
        
        properties = {
            'chunkIds': chunk_ids
        }
    
    
        results = self.graph_store.execute_query(cypher, properties)
        
        statements = {
            r['result']['statement']:r['result']['statementId'] for r in results
        }

        logger.debug(f"Top statements: {statements}")
        
        scored_statements = score_values_with_tfidf(list(statements.keys()), [query_bundle.query_str], 1)

        if scored_statements:
            return statements[list(scored_statements.keys())[0]]
        else:
            return None
    
    def _get_entities_for_statement(self, statement_id:str) -> List[str]:
        
        cypher = f'''
        // Get entities for statement
        MATCH (s)<-[:`__SUPPORTS__`]-(f)<-[:`__SUBJECT__`|`__OBJECT__`]-(e)
        WHERE {self.graph_store.node_id("s.statementId")} in $statementIds
        RETURN distinct e.value AS entity'''

        properties = {
            'statementIds': [statement_id]
        }
    
        results = self.graph_store.execute_query(cypher, properties)

        entities = [
            r['entity']
            for r in results
        ]

        return entities


    def enrich_query(self, query_bundle:QueryBundle) -> QueryBundle:
        
        if not self.args.enrich_query:
            logger.debug(f'Returning original query')
            return query_bundle

        top_statement_id = self._get_top_statement_id(query_bundle)

        if not top_statement_id:
            logger.debug(f'No statements found, so returning original query')
            return query_bundle
        
        entities = self._get_entities_for_statement(top_statement_id)

        query_str_lower = query_bundle.query_str.lower()

        entities = [
            entity
            for entity in entities
            if entity.lower() not in query_str_lower
        ]

        if entities:
            query_str=f'{query_bundle.query_str} [{" ".join(entities)}]'
            logger.debug(f'Enriched query: {query_str}')
            return QueryBundle(query_str=query_str)
        else:
            logger.debug(f'No entities found, so returning original query')
            return query_bundle

       