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
You are an expert AI assistant specialising in knowledge graphs. Given a user-supplied question and a piece of context, your task is to identify up to {num_keywords} of the most relevant keywords from the context. Return them, most relevant first. You do not have to return the maximum number of keywords; you can return fewer. 

<question>
{question}
</question>

<context>
{context}
</context>

Put the relevant keywords on separate lines. Do not provide any other explanatory text. Do not surround the output with tags. Do not exceed {num_keywords} keywords in your response.
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
        
 
    def _get_keywords_from_content(self, query:str, chunk_content:List[str]) -> List[str]:

        response = self.llm.predict(
            PromptTemplate(template=IDENTIFY_RELEVANT_ENTITIES_PROMPT),
            question=query,
            context='\n\n'.join(chunk_content),
            num_keywords=self.args.max_keywords
        )

        logger.debug(f'response: {response}')

        keywords = [k for k in response.split('\n') if k]

        return keywords

    def _get_keywords(self, query_bundle:QueryBundle) -> List[str]:
        
        chunk_ids =self._get_chunk_ids(query_bundle)
        chunk_content = self._get_chunk_content(chunk_ids)
        keywords = self._get_keywords_from_content(query_bundle.query_str, chunk_content)
        
        return keywords