# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any

from graphrag_toolkit.lexical_graph.storage.graph import GraphStore
from graphrag_toolkit.lexical_graph.indexing.build.graph_builder import GraphBuilder

from llama_index.core.schema import BaseNode
from llama_index.core.schema import NodeRelationship

logger = logging.getLogger(__name__)

class ChunkGraphBuilder(GraphBuilder):
    """Class responsible for building and managing a graph representation
    for chunks.

    The ChunkGraphBuilder class specializes in generating graph structures
    that represent "chunks" from input nodes. It retrieves metadata,
    relationships, and properties from the given node and translates
    them into database queries for storing this structure in a graph
    database. This class also handles linking chunks with their sources,
    parent nodes, child nodes, and their sequential order based on defined
    relationships.

    Attributes:
        _some_attribute (type): Description of an attribute (if any). Replace
        or expand this list with specific attributes used by the class.

    """
    @classmethod
    def index_key(cls) -> str:
        """
        Returns a string key that identifies the index used within a given context.

        Returns:
            str: A string representing the index key 'chunk'.
        """
        return 'chunk'
    
    def build(self, node:BaseNode, graph_client: GraphStore, **kwargs:Any):
        """
        Builds and inserts a chunk node along with its relationships into a graph database. Handles the
        insertion of child, parent, previous, next, and source relationships. If a chunk ID or required
        relationship information is missing, the function logs warnings.

        Args:
            node: The node object containing chunk data and its relationships.
            graph_client: The graph client interface to interact with the graph database.
            **kwargs: Additional optional parameters for configuring the operation.
        """
        chunk_metadata = node.metadata.get('chunk', {})
        chunk_id = chunk_metadata.get('chunkId', None)

        if chunk_id:

            logger.debug(f'Inserting chunk [chunk_id: {chunk_id}]')

            statements = [
                '// insert chunks',
                'UNWIND $params AS params'
            ]

            statements.extend([
                f'MERGE (chunk:`__Chunk__`{{{graph_client.node_id("chunkId")}: params.chunk_id}})',
                'ON CREATE SET chunk.value = params.text ON MATCH SET chunk.value = params.text'
            ])
            
            source_info = node.relationships.get(NodeRelationship.SOURCE, None)

            if source_info:
                
                source_id = source_info.node_id

                statements.extend([
                    f'MERGE (source:`__Source__`{{{graph_client.node_id("sourceId")}: params.source_id}})',
                    'MERGE (chunk)-[:`__EXTRACTED_FROM__`]->(source)'
                ])

                properties = {
                    'chunk_id': chunk_id,
                    'source_id': source_id,
                    'text': node.text
                }
            else:
                logger.warning(f'source_id missing from chunk node [node_id: {chunk_id}]')
            
            key_index = 0
            
            for node_relationship,relationship_info in node.relationships.items():
                
                key_index += 1
                key = f'node_relationship_{key_index}'
                node_id = relationship_info.node_id

                if node_relationship == NodeRelationship.PARENT:
                    statements.append(f'MERGE (parent:`__Chunk__`{{{graph_client.node_id("chunkId")}: params.{key}}})')
                    statements.append('MERGE (chunk)-[:`__PARENT__`]->(parent)')
                    properties[key] = node_id
                if node_relationship == NodeRelationship.CHILD:
                    statements.append(f'MERGE (child:`__Chunk__`{{{graph_client.node_id("chunkId")}: params.{key}}})')
                    statements.append('MERGE (chunk)-[:`__CHILD__`]->(child)')
                    properties[key] = node_id
                elif node_relationship == NodeRelationship.PREVIOUS:
                    statements.append(f'MERGE (previous:`__Chunk__`{{{graph_client.node_id("chunkId")}: params.{key}}})')
                    statements.append('MERGE (chunk)-[:`__PREVIOUS__`]->(previous)')
                    properties[key] = node_id
                elif node_relationship == NodeRelationship.NEXT:
                    statements.append(f'MERGE (next:`__Chunk__`{{{graph_client.node_id("chunkId")}: params.{key}}})')
                    statements.append('MERGE (chunk)-[:`__NEXT__`]->(next)')
                    properties[key] = node_id
                            
            query = '\n'.join(statements)

            graph_client.execute_query_with_retry(query, self._to_params(properties), max_attempts=5, max_wait=7)

        else:
            logger.warning(f'chunk_id missing from chunk node [node_id: {node.node_id}]')