# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any

from graphrag_toolkit.lexical_graph.indexing.model import Fact
from graphrag_toolkit.lexical_graph.storage.graph import GraphStore
from graphrag_toolkit.lexical_graph.storage.graph.graph_utils import search_string_from, label_from
from graphrag_toolkit.lexical_graph.indexing.build.graph_builder import GraphBuilder
from graphrag_toolkit.lexical_graph.indexing.constants import DEFAULT_CLASSIFICATION
from graphrag_toolkit.lexical_graph.indexing.utils.fact_utils import string_complement_to_entity

from llama_index.core.schema import BaseNode

logger = logging.getLogger(__name__)

class EntityGraphBuilder(GraphBuilder):
    """
    Handles the process of building and interacting with a graph database for entity and fact data
    representation. Supports operations to insert and manage entities and their relationships in the
    graph structure. Provides mechanisms for integrating metadata into the graph storage system.

    This class is designed to work with a specific graph storage client and encapsulates the logic
    necessary for mapping entities and facts from node data to a graph database, considering the domain
    classification when required. It assumes an ontology structure with subject and object entities
    linked by facts.

    Attributes:
        DEFAULT_CLASSIFICATION (str): The default classification for entities when not explicitly provided.
    """
    @classmethod
    def index_key(cls) -> str:
        """
        Provides a method to retrieve the index key associated with the class.

        This method is a class-level function that returns the index
        key string associated with the class. It can be used to
        uniquely identify or categorize instances of the class in
        various contexts.

        Returns:
            str: The index key associated with the class.
        """
        return 'fact'
    
    def build(self, node:BaseNode, graph_client: GraphStore, **kwargs:Any):
        """
        Processes a given node and builds the corresponding entities in the graph database.

        This method extracts fact metadata from the provided node to construct nodes and
        relationships in a graph database using Cypher queries. It validates the fact
        metadata and leverages the graph_client to execute the queries. Properties for
        the subject and object are set or updated based on whether they already exist in
        the graph. Optionally, domain-specific labels can be included.

        The function handles missing fact metadata by logging a warning message. It also
        ensures reliability through query retries with controlled attempts and wait times.

        Args:
            node (BaseNode): The node from which fact metadata is to be extracted.
            graph_client (GraphStore): The graph database client to execute queries.
            **kwargs (Any): Additional options, such as `include_domain_labels`, which
                determines whether domain-specific labels are added to the entities.
        """
        fact_metadata = node.metadata.get('fact', {})
        include_domain_labels = kwargs['include_domain_labels']

        if fact_metadata:

            fact = Fact.model_validate(fact_metadata)

            fact = string_complement_to_entity(fact)
        
            logger.debug(f'Inserting entities for fact [fact_id: {fact.factId}]')

            statements = [
                '// insert entities',
                'UNWIND $params AS params'
            ]

            if include_domain_labels:
                statements.append(f'MERGE (subject:`__Entity__`:`{label_from(fact.subject.classification or DEFAULT_CLASSIFICATION)}`{{{graph_client.node_id("entityId")}: params.s_id}})')
            else:
                statements.append(f'MERGE (subject:`__Entity__`{{{graph_client.node_id("entityId")}: params.s_id}})')

            statements.extend([
                'ON CREATE SET subject.value = params.s, subject.search_str = params.s_search_str, subject.class = params.sc',
                'ON MATCH SET subject.value = params.s, subject.search_str = params.s_search_str, subject.class = params.sc',
            ])

            properties = {
                's_id': fact.subject.entityId,
                's': fact.subject.value,
                's_search_str': search_string_from(fact.subject.value),
                'sc': fact.subject.classification or DEFAULT_CLASSIFICATION
            }

            if fact.object and fact.object.entityId != fact.subject.entityId:

                if include_domain_labels:
                    statements.append(f'MERGE (object:`__Entity__`:`{label_from(fact.object.classification or DEFAULT_CLASSIFICATION)}`{{{graph_client.node_id("entityId")}: params.o_id}})')
                else:
                    statements.append(f'MERGE (object:`__Entity__`{{{graph_client.node_id("entityId")}: params.o_id}})')

                statements.extend([
                    'ON CREATE SET object.value = params.o, object.search_str = params.o_search_str, object.class = params.oc',
                    'ON MATCH SET object.value = params.o, object.search_str = params.o_search_str, object.class = params.oc', 
                ])

                properties.update({                
                    'o_id': fact.object.entityId,
                    'o': fact.object.value,
                    'o_search_str': search_string_from(fact.object.value),
                    'oc': fact.object.classification or DEFAULT_CLASSIFICATION
                })

            elif fact.complement and fact.complement.entityId != fact.subject.entityId:

                if include_domain_labels:
                    statements.append(f'MERGE (object:`__Entity__`:`{label_from(fact.complement.classification or DEFAULT_CLASSIFICATION)}`{{{graph_client.node_id("entityId")}: params.o_id}})')
                else:
                    statements.append(f'MERGE (object:`__Entity__`{{{graph_client.node_id("entityId")}: params.o_id}})')

                statements.extend([
                    'ON CREATE SET object.value = params.o, object.search_str = params.o_search_str, object.class = params.oc',
                    'ON MATCH SET object.value = params.o, object.search_str = params.o_search_str, object.class = params.oc', 
                ])

                properties.update({                
                    'o_id': fact.complement.entityId,
                    'o': fact.complement.value,
                    'o_search_str': search_string_from(fact.complement.value),
                    'oc': fact.complement.classification or DEFAULT_CLASSIFICATION
                })
        
            query = '\n'.join(statements)
                
            graph_client.execute_query_with_retry(query, self._to_params(properties), max_attempts=5, max_wait=7)
           

        else:
            logger.warning(f'fact_id missing from fact node [node_id: {node.node_id}]')