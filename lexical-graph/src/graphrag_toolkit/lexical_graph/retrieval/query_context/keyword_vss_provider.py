# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
from typing import List, Optional

from graphrag_toolkit.lexical_graph.config import GraphRAGConfig
from graphrag_toolkit.lexical_graph.utils import LLMCache, LLMCacheType
from graphrag_toolkit.lexical_graph.metadata import FilterConfig
from graphrag_toolkit.lexical_graph.storage.graph import GraphStore
from graphrag_toolkit.lexical_graph.storage.vector import VectorStore
from graphrag_toolkit.lexical_graph.storage.graph.graph_utils import node_result
from graphrag_toolkit.lexical_graph.retrieval.model import ScoredEntity
from graphrag_toolkit.lexical_graph.retrieval.utils.vector_utils import get_diverse_vss_elements
from graphrag_toolkit.lexical_graph.retrieval.query_context.keyword_provider_base import KeywordProviderBase
from graphrag_toolkit.lexical_graph.retrieval.processors import ProcessorArgs

from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import QueryBundle

logger = logging.getLogger(__name__)

IDENTIFY_RELEVANT_ENTITIES_PROMPT = '''
You are an expert AI assistant specialising in knowledge graphs. Below is a user-supplied question a list of entities, and the context in which those entities appear. Given the question and the context, your task is to identify up to {num_entities} of the most relevant entities from the list. Return them, most relevant first. You do not have to return the maximum number of entities; you can return fewer. 

<question>
{question}
</question>

<entities>
{entities}
</entities>

<context>
{context}
</context>

Put the relevant entities on separate lines. Do not provide any other explanatory text. Do not surround the output with tags. Do not exceed {num_entities} entities in your response.
'''

class KeywordVSSProvider(KeywordProviderBase):
    
    def __init__(self,
                 graph_store:GraphStore,
                 vector_store:VectorStore,
                 args:ProcessorArgs,
                 filter_config:Optional[FilterConfig]=None,
                 llm:LLMCacheType=None
                ):
        
        super().__init__(args)

        self.graph_store = graph_store
        self.vector_store = vector_store
        self.filter_config = filter_config
       
        self.llm = llm if llm and isinstance(llm, LLMCache) else LLMCache(
            llm=llm or GraphRAGConfig.extraction_llm,
            enable_cache=GraphRAGConfig.enable_cache
        )

    def _get_chunk_ids(self, query_bundle:QueryBundle) -> List[str]:

        vss_results = get_diverse_vss_elements('chunk', query_bundle, self.vector_store, 5, 3, self.filter_config)
        
        chunk_ids = [result['chunk']['chunkId'] for result in vss_results]

        logger.debug(f'chunk_ids: {chunk_ids}')

        return chunk_ids
    
    def _get_chunk_content(self, chunk_ids:List[str]) -> List[str]:
        cypher = f"""
        // get chunk content
        MATCH (c:`__Chunk__`)
        WHERE {self.graph_store.node_id("c.chunkId")} in $chunkIds
        RETURN c.value AS content
        """

        parameters = {
            'chunkIds': chunk_ids
        }

        results = self.graph_store.execute_query(cypher, parameters)

        chunk_content = [result['content'] for result in results]

        return chunk_content
    
    def _get_entities_for_chunks(self, chunk_ids:List[str]) -> List[ScoredEntity]:

        cypher = f"""
        // get entities for chunk ids
        MATCH (c:`__Chunk__`)<-[:`__MENTIONED_IN__`]-(:`__Statement__`)
        <-[:`__SUPPORTS__`]-()<-[:`__SUBJECT__`|`__OBJECT__`]-(entity)
        -[r:`__RELATION`]-()
        WHERE {self.graph_store.node_id("c.chunkId")} in $chunkIds
        WITH DISTINCT entity, count(DISTINCT r) AS score ORDER BY score DESC LIMIT $limit
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

        logger.debug(f'entities: {scored_entities}')

        return scored_entities
        
 
    def _get_keywords_for_entities(self, query:str, chunk_content:List[str], entities:List[ScoredEntity]) -> List[str]:

        entity_names = list(set([entity.entity.value for entity in entities]))
        num_entities = self.args.ec_num_entities

        response = self.llm.predict(
            PromptTemplate(template=IDENTIFY_RELEVANT_ENTITIES_PROMPT),
            question=query,
            context='\n\n'.join(chunk_content),
            entities='\n'.join(entity_names),
            num_entities=num_entities
        )

        logger.debug(f'response: {response}')

        keywords = [k for k in response.split('\n') if k]

        return keywords

    def _get_keywords(self, query_bundle:QueryBundle) -> List[str]:
        
        chunk_ids =self._get_chunk_ids(query_bundle)
        chunk_content = self._get_chunk_content(chunk_ids)
        entities = self._get_entities_for_chunks(chunk_ids)
        keywords = self._get_keywords_for_entities(query_bundle.query_str, chunk_content, entities)
        
        return keywords