# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import concurrent.futures
import logging
from itertools import repeat
from typing import List, Iterator, cast, Optional

from graphrag_toolkit.lexical_graph.metadata import FilterConfig
from graphrag_toolkit.lexical_graph.config import GraphRAGConfig
from graphrag_toolkit.lexical_graph.storage.graph import GraphStore
from graphrag_toolkit.lexical_graph.storage.graph.graph_utils import node_result, search_string_from
from graphrag_toolkit.lexical_graph.utils import LLMCache, LLMCacheType
from graphrag_toolkit.lexical_graph.retrieval.model import ScoredEntity
from graphrag_toolkit.lexical_graph.retrieval.prompts import SIMPLE_EXTRACT_KEYWORDS_PROMPT, EXTENDED_EXTRACT_KEYWORDS_PROMPT

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

logger = logging.getLogger(__name__)

class KeywordEntitySearch(BaseRetriever):
    """Performs keyword-based entity search using a graph database.

    This class is designed to perform retrieval tasks by identifying keywords in a
    query, mapping those keywords to entities in a graph database, and scoring
    entities based on their relevance. It supports keyword enrichment, expansion of
    retrieved entities, and can interact with an LLM cache for keyword extraction.
    The class is intended for use in systems that leverage graph stores for
    retrieval tasks, such as knowledge-based question answering or entity-centric
    information retrieval.

    Attributes:
        graph_store (GraphStore): The graph database used for storing and querying
            entities and their relationships.
        llm (LLMCache): Cache-enabled or raw large language model implementation
            used for keyword extraction.
        simple_extract_keywords_template (str): Prompt template used for extracting
            simpler keywords from the query.
        extended_extract_keywords_template (str): Prompt template used for
            extracting enriched keywords from the query.
        max_keywords (int): Maximum number of keywords to extract and process for
            entity retrieval.
        expand_entities (bool): Flag indicating whether to perform entity expansion
            after initial keyword-based retrieval.
        filter_config (Optional[FilterConfig]): Optional configuration for applying
            filters during entity retrieval.
    """
    def __init__(self,
                 graph_store:GraphStore,
                 llm:LLMCacheType=None, 
                 simple_extract_keywords_template=SIMPLE_EXTRACT_KEYWORDS_PROMPT,
                 extended_extract_keywords_template=EXTENDED_EXTRACT_KEYWORDS_PROMPT,
                 max_keywords=10,
                 expand_entities=False,
                 filter_config:Optional[FilterConfig]=None):
        """
        Initializes the class with parameters for managing a graph store and large language
        model (LLM), along with configurations for keyword extraction and entity expansion.

        The initializer sets up the graph store, language model instance, keyword extraction
        template, maximum number of allowed keywords, and optional filter configuration.

        Args:
            graph_store (GraphStore): A graph store instance used for managing and querying
                the graph.
            llm (LLMCacheType): A large language model instance, either provided or created
                from the default configuration. Defaults to `None`.
            simple_extract_keywords_template: Template for extracting simple keywords.
                Defaults to SIMPLE_EXTRACT_KEYWORDS_PROMPT.
            extended_extract_keywords_template: Template for extracting extended keywords.
                Defaults to EXTENDED_EXTRACT_KEYWORDS_PROMPT.
            max_keywords (int): The limit on the number of keywords extracted. Defaults
                to 10.
            expand_entities (bool): Flag indicating whether to expand extracted entities.
                Defaults to False.
            filter_config (Optional[FilterConfig]): Configuration to filter extracted data.
                Defaults to `None`.
        """
        self.graph_store = graph_store
        self.llm = llm if llm and isinstance(llm, LLMCache) else LLMCache(
            llm=llm or GraphRAGConfig.response_llm,
            enable_cache=GraphRAGConfig.enable_cache
        )
        self.simple_extract_keywords_template=simple_extract_keywords_template
        self.extended_extract_keywords_template=extended_extract_keywords_template
        self.max_keywords = max_keywords
        self.expand_entities = expand_entities
        self.filter_config = filter_config

    
    def _expand_entities(self, scored_entities:List[ScoredEntity]):
        """
        Expands a list of scored entities by fetching related entities from a graph store, scoring them,
        and appending the most relevant ones to the original list. This function performs several
        steps, including applying constraints on the maximum keywords, determining related entities
        through graph queries, scoring these entities based on their relationships, and appending
        additional entities based on scoring thresholds.

        Constraints are applied using a maximum number of keywords (`max_keywords`)
        and a scoring threshold (`upper_score_threshold`) that ensures irrelevant or less-relevant
        entities are filtered out. The process involves interacting with a graph store, where Cypher
        queries are executed to determine entity relationships and scores.

        The function preserves the order of the entities based on their scores in descending order.
        Debug information is logged to trace the expanded entity list.

        Args:
            scored_entities (List[ScoredEntity]): A list of entities already scored that will be
                used as the seed for expanding and scoring related entities from the graph store.

        Returns:
            List[ScoredEntity]: The modified input list with additional scored and expanded
                entities appended, sorted by score in descending order.
        """
        if not scored_entities or len(scored_entities) >= self.max_keywords:
            return scored_entities
        
        upper_score_threshold = max(sc.score for sc in scored_entities) * 2
        
        original_entity_ids = [entity.entity.entityId for entity in scored_entities if entity.score > 0]  
        neighbour_entity_ids = set()
        
        start_entity_ids = set(original_entity_ids) 
        exclude_entity_ids = set(start_entity_ids)

        for limit in range (3, 1, -1):
        
            cypher = f"""
            // expand entities
            MATCH (entity:`__Entity__`)
            -[:`__SUBJECT__`|`__OBJECT__`]->()<-[:`__SUBJECT__`|`__OBJECT__`]-
            (other:`__Entity__`)
            WHERE  {self.graph_store.node_id('entity.entityId')} IN $entityIds
            AND NOT {self.graph_store.node_id('other.entityId')} IN $excludeEntityIds
            WITH entity, other
            MATCH (other)-[r:`__SUBJECT__`|`__OBJECT__`]->()
            WITH entity, other, count(r) AS score ORDER BY score DESC
            RETURN {{
                {node_result('entity', self.graph_store.node_id('entity.entityId'), properties=['value', 'class'])},
                others: collect(DISTINCT {self.graph_store.node_id('other.entityId')})[0..$limit]
            }} AS result    
            """

            params = {
                'entityIds': list(start_entity_ids),
                'excludeEntityIds': list(exclude_entity_ids),
                'limit': limit
            }
        
            results = self.graph_store.execute_query(cypher, params)

            other_entity_ids = set([
                other_id
                for result in results
                for other_id in result['result']['others'] 
            ])
            
            neighbour_entity_ids.update(other_entity_ids)

            exclude_entity_ids.update(other_entity_ids)
            start_entity_ids = other_entity_ids
            
      
        cypher = f"""
        // expand entities: score entities by number of facts
        MATCH (entity:`__Entity__`)-[r:`__SUBJECT__`]->()
        WHERE {self.graph_store.node_id('entity.entityId')} IN $entityIds
        WITH entity, count(r) AS score
        RETURN {{
            {node_result('entity', self.graph_store.node_id('entity.entityId'), properties=['value', 'class'])},
            score: score
        }} AS result
        """

        params = {
            'entityIds': list(neighbour_entity_ids)
        }

        results = self.graph_store.execute_query(cypher, params)
        
        neighbour_entities = [
            ScoredEntity.model_validate(result['result'])
            for result in results 
            if result['result']['entity']['entityId'] not in original_entity_ids and result['result']['score'] <= upper_score_threshold and result['result']['score'] > 0.0
        ]
        
        neighbour_entities.sort(key=lambda e:e.score, reverse=True)

        num_addition_entities = self.max_keywords - len(scored_entities)

        scored_entities.extend(neighbour_entities[:num_addition_entities])        
        scored_entities.sort(key=lambda e:e.score, reverse=True)

        logger.debug('Expanded entities:\n' + '\n'.join(
            entity.model_dump_json(exclude_unset=True, exclude_defaults=True, exclude_none=True, warnings=False) 
            for entity in scored_entities)
        )

        return scored_entities
        
    def _get_entities_for_keyword(self, keyword:str) -> List[ScoredEntity]:
        """
        Generates and executes a graph database query to retrieve entities related to the given keyword
        and their associated scores. The method interprets the keyword to dynamically adjust query
        behavior, supports optional filtering by classification, and ensures the filtering of results
        with a score of zero.

        Args:
            keyword (str): A string representing the keyword for searching entities. It may include an
                optional classification separated by a vertical bar "|" (e.g., "keyword|classification").

        Returns:
            List[ScoredEntity]: A list of `ScoredEntity` instances, each representing an entity and
            its associated score, filtered to only include entities with a nonzero score.
        """
        parts = keyword.split('|')

        if len(parts) > 1:

            cypher = f"""
            // get entities for keywords
            MATCH (entity:`__Entity__`)-[r:`__SUBJECT__`|`__OBJECT__`]->()
            WHERE entity.search_str = $keyword and entity.class STARTS WITH $classification
            WITH entity, count(r) AS score ORDER BY score DESC
            RETURN {{
                {node_result('entity', self.graph_store.node_id('entity.entityId'), properties=['value', 'class'])},
                score: score
            }} AS result"""

            params = {
                'keyword': search_string_from(parts[0]),
                'classification': parts[1]
            }
        else:
            cypher = f"""
            // get entities for keywords
            MATCH (entity:`__Entity__`)-[r:`__SUBJECT__`|`__OBJECT__`]->()
            WHERE entity.search_str = $keyword
            WITH entity, count(r) AS score ORDER BY score DESC
            RETURN {{
                {node_result('entity', self.graph_store.node_id('entity.entityId'), properties=['value', 'class'])},
                score: score
            }} AS result"""

            params = {
                'keyword': search_string_from(parts[0])
            }

        results = self.graph_store.execute_query(cypher, params)

        return [
            ScoredEntity.model_validate(result['result'])
            for result in results
            if result['result']['score'] != 0
        ]
                        
    def _get_entities_for_keywords(self, keywords:List[str])  -> List[ScoredEntity]:
        """
        Retrieves and processes entities for a list of keywords asynchronously, aggregating
        scores for identical entities and sorting the results by score in descending order.

        Args:
            keywords (List[str]): A list of keywords to analyze and retrieve entities for.

        Returns:
            List[ScoredEntity]: A sorted list of scored entities based on the provided keywords.

        """

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(keywords)) as p:
            scored_entity_batches:Iterator[List[ScoredEntity]] = p.map(self._get_entities_for_keyword, keywords)
            scored_entities = sum(scored_entity_batches, start=cast(List[ScoredEntity], []))

        scored_entity_mappings = {}
        for scored_entity in scored_entities:
            entity_id = scored_entity.entity.entityId
            if entity_id not in scored_entity_mappings:
                scored_entity_mappings[entity_id] = scored_entity
            else:
                scored_entity_mappings[entity_id].score += scored_entity.score

        scored_entities = list(scored_entity_mappings.values())

        scored_entities.sort(key=lambda e:e.score, reverse=True)

        logger.debug('Initial entities:\n' + '\n'.join(
            entity.model_dump_json(exclude_unset=True, exclude_defaults=True, exclude_none=True, warnings=False) 
            for entity in scored_entities)
        )

        return scored_entities

        
    def _extract_keywords(self, s:str, num_keywords:int, prompt_template:str):
        """
        Extracts keywords from a given input text using a specified prompt template and
        the number of keywords to be extracted.

        This method uses a language model (LLM) to perform keyword extraction based on
        the provided input string, number of desired keywords, and a template for the
        prompt. The resulting keywords are split from the model's output using the '^'
        delimiter.

        Args:
            s: The input text from which keywords will be extracted.
            num_keywords: The number of keywords to extract from the input text.
            prompt_template: The prompt template to be used by the LLM to guide the
                extraction process.

        Returns:
            A list of keywords extracted from the input text.
        """
        results = self.llm.predict(
            PromptTemplate(template=prompt_template),
            text=s,
            max_keywords=num_keywords
        )

        keywords = results.split('^')
        return keywords

    def _get_simple_keywords(self, query, num_keywords):
        """
        Extracts and returns a list of simple keywords from the given query using the specified
        template for simple keyword extraction.

        Args:
            query: The input string from which keywords need to be extracted.
            num_keywords: The number of keywords to extract from the input query.

        Returns:
            A list containing extracted simple keywords.
        """
        simple_keywords = self._extract_keywords(query, num_keywords, self.simple_extract_keywords_template)
        logger.debug(f'Simple keywords: {simple_keywords}')
        return simple_keywords
    
    def _get_enriched_keywords(self, query, num_keywords):
        """
        Generates a list of enriched keywords based on the given query and specified
        number of keywords. It utilizes an extended keyword extraction template to
        enhance the keyword generation process.

        Args:
            query: Specifies the input query to extract keywords from.
            num_keywords: Determines the desired number of keywords to be retrieved.

        Returns:
            The enriched keywords generated as a result of the extraction
            process.
        """
        enriched_keywords = self._extract_keywords(query, num_keywords, self.extended_extract_keywords_template)
        logger.debug(f'Enriched keywords: {enriched_keywords}')
        return enriched_keywords

    def _get_keywords(self, query, max_keywords):
        """Extracts and retrieves unique keywords from a given query by using parallel keyword
        generation methods.

        The function runs two asynchronous keyword generation methods: `_get_simple_keywords`
        and `_get_enriched_keywords`, each producing a batch of keywords. It then combines
        the resulting keywords, removes duplicates, and returns the set of unique keywords.

        Args:
            query (str): The input query string from which keywords are extracted.
            max_keywords (int): The maximum number of keywords to be generated.

        Returns:
            list: A list containing unique keywords extracted from the query.
        """
        num_keywords = max(int(max_keywords/2), 1)

        with concurrent.futures.ThreadPoolExecutor() as p:
            keyword_batches: Iterator[List[str]] = p.map(
                lambda f, *args: f(*args),
                (self._get_simple_keywords, self._get_enriched_keywords),
                repeat(query),
                repeat(num_keywords)
            )
            keywords = sum(keyword_batches, start=cast(List[str], []))
            unique_keywords = list(set(keywords))

        logger.debug(f'Keywords: {unique_keywords}')
        
        return unique_keywords

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Retrieves relevant nodes based on the query contained in the given
        query bundle. The process involves extracting keywords from the query,
        retrieving entities related to these keywords, optionally expanding
        these entities, and mapping them to nodes with their respective scores.

        Args:
            query_bundle (QueryBundle): A bundle containing the query string
                and additional metadata if available.

        Returns:
            List[NodeWithScore]: A list of nodes with their associated scores,
            representing the relevance of each node based on the processed query.
        """
        query = query_bundle.query_str
        
        keywords = self._get_keywords(query, self.max_keywords)
        scored_entities:List[ScoredEntity] = self._get_entities_for_keywords(keywords)

        if self.expand_entities:
            scored_entities = self._expand_entities(scored_entities)

        return [
            NodeWithScore(
                node=TextNode(text=scored_entity.entity.model_dump_json(exclude_none=True, exclude_defaults=True, indent=2)),
                score=scored_entity.score
            ) 
            for scored_entity in scored_entities
        ]