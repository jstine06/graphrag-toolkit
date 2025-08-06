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

        
    def _get_chunk_ids(self, values:List[str]) -> List[str]:
        
        query_bundle =  QueryBundle(query_str=', '.join(values))
        vss_results = self.vector_store.get_index('chunk').top_k(query_bundle, 3, filter_config=self.filter_config)

        chunk_ids = [result['chunk']['chunkId'] for result in vss_results]

        logger.debug(f'chunk_ids: {chunk_ids}')

        return chunk_ids

    def _get_entities_for_chunks(self, chunk_ids:List[str]) -> List[ScoredEntity]:

        cypher = f"""
        // get entities for chunk ids
        MATCH (c:`__Chunk__`)
            <-[:`__MENTIONED_IN__`]-()
            <-[:`__BELONGS_TO__`]-()
            <-[:`__SUPPORTS__`]-()
            <-[:`__SUBJECT__`|`__OBJECT__`]-(entity)
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
    
    def _get_reranked_entity_names_model(self, entities:List[ScoredEntity], keywords:List[str]) -> Dict[str, float]:

        reranker = SentenceReranker(model=GraphRAGConfig.reranking_model, top_n=3)
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
    
    def _get_reranked_entity_names_tfidf(self, entities:List[ScoredEntity], keywords:List[str]) -> Dict[str, float]:
        
        entity_names = [entity.entity.value.lower() for entity in entities]
        reranked_entity_names = score_values(entity_names, keywords, 3)

        return reranked_entity_names
    
    def _get_reranked_entity_names(self, entities:List[ScoredEntity], keywords:List[str]) -> Dict[str, float]:
        
        if self.args.reranker == 'model':
            return self._get_reranked_entity_names_model(entities, keywords) 
        else:
            return self._get_reranked_entity_names_tfidf(entities, keywords)
        
    def _get_entities_by_keyword_match(self, keywords:List[str], query_bundle:QueryBundle) -> List[ScoredEntity]:
        initial_entity_provider = EntityProvider(self.graph_store, self.args, self.filter_config)
        return initial_entity_provider.get_entities(keywords, query_bundle)
    
    def _get_entities_for_value(self, keyword:str) -> List[ScoredEntity]:
        chunk_ids = self._get_chunk_ids([keyword])
        chunk_entities = self._get_entities_for_chunks(chunk_ids)

        logger.debug(f'chunk entities: [keyword: {keyword}, chunk_ids: {chunk_ids}, entities: {chunk_entities}]')

        reranked_entity_names = self._get_reranked_entity_names(chunk_entities, [keyword])
        return self._get_reranked_entities(chunk_entities, reranked_entity_names)

                        
    def _get_entities(self, keywords:List[str], query_bundle:QueryBundle) -> List[ScoredEntity]:

        all_entities_map = {}

        def add_to_entities_map(entities):
            all_entities_map.update({e.entity.entityId:e for e in entities})

        add_to_entities_map(self._get_entities_by_keyword_match(keywords, query_bundle))
        add_to_entities_map(self._get_entities_for_value(query_bundle.query_str)[:3])
        
        for keyword in keywords:
            add_to_entities_map(self._get_entities_for_value(keyword)[:3])

        all_reranked_entity_names = self._get_reranked_entity_names(list(all_entities_map.values()), [query_bundle.query_str] + keywords)
        all_reranked_entities = self._get_reranked_entities(list(all_entities_map.values()), all_reranked_entity_names)
             
        logger.debug(f'reranked_entities: {all_reranked_entities}')
        
        return all_reranked_entities

        