# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any

from graphrag_toolkit.lexical_graph.indexing.model import Fact
from graphrag_toolkit.lexical_graph.storage.graph import GraphStore
from graphrag_toolkit.lexical_graph.storage.graph.graph_utils import relationship_name_from, new_query_var
from graphrag_toolkit.lexical_graph.indexing.build.graph_builder import GraphBuilder
from graphrag_toolkit.lexical_graph.indexing.utils.fact_utils import string_complement_to_entity
from graphrag_toolkit.lexical_graph.indexing.constants import LOCAL_ENTITY_CLASSIFICATION

from llama_index.core.schema import BaseNode

logger = logging.getLogger(__name__)

class LocalEntityRewritesGraphBuilder(GraphBuilder):
    
    @classmethod
    def index_key(cls) -> str:
        return 'fact'
    
    def build(self, node:BaseNode, graph_client: GraphStore, **kwargs:Any):
        
        fact_metadata = node.metadata.get('fact', {})
        include_domain_labels = kwargs['include_domain_labels']
        include_local_entities = kwargs['include_local_entities']

        if fact_metadata:

            fact = Fact.model_validate(fact_metadata)
            fact = string_complement_to_entity(fact)

            if fact.subject.classification and fact.subject.classification == LOCAL_ENTITY_CLASSIFICATION:
                if not include_local_entities:
                    logger.debug(f'Ignoring local entity rewrites for fact [fact_id: {fact.factId}]')
                    return
                
            if fact.subject:

                # Subject may map to complememnts (e.g. it may be an email address value that elsewhere acts as a complement)

                copy_complement_rels_statements = [
                    '// copy complement relationships to subject entity',
                    'UNWIND $params AS params',
                    f"MATCH (n:`__Entity__`{{{graph_client.node_id('entityId')}: params.nId}}),",   
                    f"(s)-[r:`__RELATION__`]->(c:`__Entity__`{{search_str: n.search_str, class: '{LOCAL_ENTITY_CLASSIFICATION}'}})-[:`__OBJECT__`]->(f)",
                    "WHERE c <> n",
                    "WITH n, s, r, f",
                    "MERGE (s)-[:`__RELATION__`{value:r.value}]->(n)",
                    "MERGE (n)-[:`__OBJECT__`]->(f)"
                ]

                copy_complement_rels_params = {
                    'nId': fact.subject.entityId
                }

                copy_complement_rels_query = '\n'.join(copy_complement_rels_statements)

                graph_client.execute_query_with_retry(copy_complement_rels_query, self._to_params(copy_complement_rels_params), max_attempts=5, max_wait=7)


                delete_complement_statements = [
                    '// delete complement and rels if real subject entity',
                    'UNWIND $params AS params',
                    f"MATCH (n:`__Entity__`{{{graph_client.node_id('entityId')}: params.nId}}),",   
                    f"()-[r1:`__RELATION__`]->(c:`__Entity__`{{search_str: n.search_str, '{LOCAL_ENTITY_CLASSIFICATION}'}})-[r2:`__OBJECT__`]->()",
                    "WHERE c <> n",
                    "WITH n, r1, r2, c",
                    "DELETE r1",
                    "DELETE r2",
                    "DETACH DELETE c"
                ]

                delete_complement_params = {
                    'nId': fact.subject.entityId
                }

                delete_complement_query = '\n'.join(delete_complement_statements)

                graph_client.execute_query_with_retry(delete_complement_query, self._to_params(delete_complement_params), max_attempts=5, max_wait=7)


            if fact.subject and fact.complement:

                # Complement may map to a real entity (e.g. it may be an email address value that elsewhere acts as a subject entity)

                copy_complement_rels_statements = [
                    '// copy complement relationships to real entity',
                    'UNWIND $params AS params',
                    f"MATCH (n:`__Entity__`{{{graph_client.node_id('entityId')}: params.nId}}),",   
                    f"(s)-[r:`__RELATION__`]->(c:`__Entity__`{{{graph_client.node_id('entityId')}: params.cId}})-[:`__OBJECT__`]->(f)",
                    "WITH n, s, r, f",
                    "MERGE (s)-[:`__RELATION__`{value:r.value}]->(n)",
                    "MERGE (n)-[:`__OBJECT__`]->(f)"
                ]

                copy_complement_rels_params = {
                    'nId': fact.complement.altEntityId,
                    'cId': fact.complement.entityId
                }

                copy_complement_rels_query = '\n'.join(copy_complement_rels_statements)

                graph_client.execute_query_with_retry(copy_complement_rels_query, self._to_params(copy_complement_rels_params), max_attempts=5, max_wait=7)

                delete_complement_statements = [
                    '// delete complement and rels if real entity',
                    'UNWIND $params AS params',
                    f"MATCH (n:`__Entity__`{{{graph_client.node_id('entityId')}: params.nId}}),",   
                    f"()-[r1:`__RELATION__`]->(c:`__Entity__`{{{graph_client.node_id('entityId')}: params.cId}})-[r2:`__OBJECT__`]->()",
                    "WITH n, r1, r2, c",
                    "DELETE r1",
                    "DELETE r2",
                    "DETACH DELETE c"
                ]

                delete_complement_params = {
                    'nId': fact.complement.altEntityId,
                    'cId': fact.complement.entityId
                }

                delete_complement_query = '\n'.join(delete_complement_statements)

                graph_client.execute_query_with_retry(delete_complement_query, self._to_params(delete_complement_params), max_attempts=5, max_wait=7)

        else:
            logger.warning(f'fact_id missing from fact node [node_id: {node.node_id}]')