# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
from typing import List, Optional, Dict

from graphrag_toolkit.lexical_graph import GraphRAGConfig
from graphrag_toolkit.lexical_graph.metadata import FilterConfig
from graphrag_toolkit.lexical_graph.storage.graph import GraphStore
from graphrag_toolkit.lexical_graph.storage.vector import VectorStore
from graphrag_toolkit.lexical_graph.storage.graph.graph_utils import node_result
from graphrag_toolkit.lexical_graph.retrieval.model import ScoredEntity
from graphrag_toolkit.lexical_graph.utils.tfidf_utils import score_values
from graphrag_toolkit.lexical_graph.retrieval.query_context.entity_provider_base import EntityProviderBase
from graphrag_toolkit.lexical_graph.retrieval.query_context.entity_provider import EntityProvider
from graphrag_toolkit.lexical_graph.retrieval.processors import ProcessorArgs
from graphrag_toolkit.lexical_graph.retrieval.post_processors import SentenceReranker

from llama_index.core.schema import QueryBundle, NodeWithScore, TextNode


logger = logging.getLogger(__name__)

class EntityVSSProvider(EntityProviderBase):
    
    def __init__(self, graph_store:GraphStore, vector_store:VectorStore, args:ProcessorArgs, filter_config:Optional[FilterConfig]=None):
        super().__init__(graph_store=graph_store, args=args, filter_config=filter_config)
        self.vector_store = vector_store

        
    def _get_chunk_ids(self, keywords:List[str]) -> List[str]:
        
        query_bundle =  QueryBundle(query_str=', '.join(keywords))
        vss_results = self.vector_store.get_index('chunk').top_k(query_bundle, 3, filter_config=self.filter_config)

        chunk_ids = [result['chunk']['chunkId'] for result in vss_results]

        logger.debug(f'chunk_ids: {chunk_ids}')

        return chunk_ids

    def _get_entities_for_chunks(self, chunk_ids:List[str]) -> List[ScoredEntity]:

        cypher = f"""
        // get entities for chunk ids
        MATCH (c:`__Chunk__`)<-[:`__MENTIONED_IN__`]-(:`__Statement__`)
        <-[:`__SUPPORTS__`]-()<-[:`__SUBJECT__`|`__OBJECT__`]-(entity)
        WHERE {self.graph_store.node_id("c.chunkId")} in $chunkIds
        WITH DISTINCT entity
        MATCH (entity)-[r:`__SUBJECT__`|`__OBJECT__`]->()
        WITH entity, count(r) AS score ORDER BY score DESC LIMIT $limit
        RETURN {{
            {node_result('entity', self.graph_store.node_id('entity.entityId'), properties=['value', 'class'])},
            score: score
        }} AS result
        """

        parameters = {
            'chunkIds': chunk_ids,
            'limit': self.args.intermediate_limit
        }

        results = self.graph_store.execute_query(cypher, parameters)

        scored_entities = [
            ScoredEntity.model_validate(result['result'])
            for result in results
            if result['result']['score'] != 0
        ]

        return scored_entities
    
    def _update_reranked_entity_name_scores(self, reranked_entity_names:Dict[str, float], keywords:List[str]):

        num_keywords = len(keywords)

        for i, keyword in enumerate(keywords):
            multiplier = num_keywords - i
            entity_reranking_score = reranked_entity_names.get(keyword, None)
            if entity_reranking_score:
                reranked_entity_names[keyword] = entity_reranking_score * multiplier

        return reranked_entity_names

    
    def _get_reranked_entities(self, entities:List[ScoredEntity], reranked_entity_names:Dict[str, float]) -> List[ScoredEntity]:

        entity_id_map = {}

        for reranked_entity_name, reranking_score in reranked_entity_names.items():
            for entity in entities:
                if entity.entity.value.lower() == reranked_entity_name and entity.entity.entityId not in entity_id_map:
                    entity.reranking_score = reranking_score
                    entity_id_map[entity.entity.entityId] = None
                    

        entities.sort(key=lambda e: (-e.reranking_score, -e.score))
        
        return entities
    
    def _get_reranked_entity_names_model(self, entities:List[ScoredEntity], keywords:List[str]) -> List[ScoredEntity]:

        reranker = SentenceReranker(model=GraphRAGConfig.reranking_model, top_n=len(entities))
        rank_query = QueryBundle(query_str=' '.join(keywords))

        reranked_values = reranker.postprocess_nodes(
            [
                NodeWithScore(node=TextNode(text=entity.entity.value.lower()), score=0.0)
                for entity in entities
            ],
            rank_query
        )

        reranked_entity_names =  {
            reranked_value.text : reranked_value.score
            for reranked_value in reranked_values
        }

        return reranked_entity_names
    
    def _get_reranked_entity_names_tfidf(self, entities:List[ScoredEntity], keywords:List[str]) -> List[ScoredEntity]:
        
        entity_names = [entity.entity.value.lower() for entity in entities]
        reranked_entity_names = score_values(entity_names, keywords, ngram_length=2)

        return reranked_entity_names

                        
    def _get_entities(self, keywords:List[str]) -> List[ScoredEntity]:

        initial_entity_provider = EntityProvider(self.graph_store, self.args, self.filter_config)
        initial_entities = initial_entity_provider.get_entities(keywords)

        num_chunk_entities = max(self.args.ec_num_entities - len(initial_entities), 0)

        reranked_entities = []
        chunk_entities = []

        if num_chunk_entities > 0:
        
            chunk_ids = self._get_chunk_ids(keywords)
            chunk_entities = self._get_entities_for_chunks(chunk_ids)

        logger.debug(f'initial_entities: {initial_entities}')
        logger.debug(f'chunk_entities: {chunk_entities}')

        reranked_entity_names = {}
        all_entities = initial_entities + chunk_entities
        
        if self.args.reranker == 'model':
            reranked_entity_names = self._get_reranked_entity_names_model(all_entities, keywords) 
        else:
            reranked_entity_names = self._get_reranked_entity_names_tfidf(all_entities, keywords)

        logger.debug(f'reranked_entity_names: {reranked_entity_names}')

        updated_reranked_entity_names = self._update_reranked_entity_name_scores(reranked_entity_names, keywords)

        logger.debug(f'updated_reranked_entity_names: {updated_reranked_entity_names}')

        reranked_entities = self._get_reranked_entities(all_entities, updated_reranked_entity_names) 
       
        logger.debug(f'reranked_entities: {reranked_entities}')
        
        return reranked_entities

        