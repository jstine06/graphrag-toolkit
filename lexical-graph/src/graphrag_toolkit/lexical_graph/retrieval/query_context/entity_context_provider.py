# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
import time
from typing import List, Dict

from graphrag_toolkit.lexical_graph.storage.graph import GraphStore
from graphrag_toolkit.lexical_graph.storage.graph.graph_utils import node_result
from graphrag_toolkit.lexical_graph.retrieval.model import ScoredEntity
from graphrag_toolkit.lexical_graph.retrieval.processors import ProcessorArgs
from graphrag_toolkit.lexical_graph.utils.tfidf_utils import score_values

from llama_index.core.schema import QueryBundle


logger = logging.getLogger(__name__)

class EntityContextProvider():
    
    def __init__(self, graph_store:GraphStore, args:ProcessorArgs):
        self.graph_store = graph_store
        self.args = args
        
    def _get_entity_id_context_tree(self, entities:List[ScoredEntity]) -> Dict[str, Dict]:
        
        start = time.time()
        
        entity_ids = [entity.entity.entityId for entity in entities if entity.score > 0] 
        exclude_entity_ids = set(entity_ids)
        neighbour_entity_ids = set()
        
        entity_id_context_tree = { entity_id:{} for entity_id in entity_ids }
        
        for entity_id, entity_id_context in entity_id_context_tree.items():
            
            start_entity_ids = set([entity_id])
            
            current_entity_id_contexts = { entity_id: entity_id_context  }

            for num_neighbours in range (3, 1, -1):

                cypher = f"""
                // get next level in tree
                MATCH (entity:`__Entity__`)-[:`__RELATION__`]->(other)
                      -[r:`__SUBJECT__`|`__OBJECT__`]->()
                WHERE  {self.graph_store.node_id('entity.entityId')} IN $entityIds
                AND NOT {self.graph_store.node_id('other.entityId')} IN $excludeEntityIds
                WITH entity, other, count(DISTINCT r) AS score ORDER BY score DESC
                RETURN {{
                    {node_result('entity', self.graph_store.node_id('entity.entityId'), properties=['value', 'class'])},
                    others: collect({self.graph_store.node_id('other.entityId')})[0..$numNeighbours]
                }} AS result    
                """

                params = {
                    'entityIds': list(start_entity_ids),
                    'excludeEntityIds': list(exclude_entity_ids),
                    'numNeighbours': num_neighbours
                }

                results = self.graph_store.execute_query(cypher, params)

                new_entity_id_contexts = {}

                for result in results:
                    
                    start_entity_id = result['result']['entity']['entityId']
                    other_entity_ids = result['result']['others']

                    for other_entity_id in other_entity_ids:
                        child_context = { }
                        current_entity_id_contexts[start_entity_id][other_entity_id] = child_context
                        new_entity_id_contexts[other_entity_id] = child_context


                other_entity_ids = set([
                    other_id
                    for result in results
                    for other_id in result['result']['others'] 
                ])

                neighbour_entity_ids.update(other_entity_ids)
                start_entity_ids = other_entity_ids

                current_entity_id_contexts = new_entity_id_contexts

        end = time.time()
        duration_ms = (end-start) * 1000

        logger.debug(f'entity_id_context_tree: {entity_id_context_tree} ({duration_ms:.2f} ms)')
                
        return entity_id_context_tree
    
    def _get_neighbour_entities(self, entity_id_context_tree:Dict[str, Dict], baseline_score:float) -> List[ScoredEntity]:

        start = time.time()

        neighbour_entity_ids = set()

        def walk_tree(d):
            for entity_id, children in d.items():
                neighbour_entity_ids.add(entity_id)
                walk_tree(children)
            
        for _, d in entity_id_context_tree.items():
            walk_tree(d)
        
        logger.debug(f'neighbour_entity_ids: {list(neighbour_entity_ids)}')

        cypher = f"""
        // expand entities: score entities by number of relations
        MATCH (entity:`__Entity__`)-[r:`__SUBJECT__`|`__OBJECT__`]->()
        WHERE {self.graph_store.node_id('entity.entityId')} IN $entityIds
        WITH entity, count(DISTINCT r) AS score
        RETURN {{
            {node_result('entity', self.graph_store.node_id('entity.entityId'), properties=['value', 'class'])},
            score: score
        }} AS result
        """

        params = {
            'entityIds': list(neighbour_entity_ids)
        }

        results = self.graph_store.execute_query(cypher, params)

        upper_score_threshold = baseline_score * self.args.ec_max_score_factor
        lower_score_threshhold = baseline_score * self.args.ec_min_score_factor

        
        all_neighbour_entities = [
            ScoredEntity.model_validate(result['result'])
            for result in results 
            if result['result']['score'] <= upper_score_threshold and result['result']['score'] >= lower_score_threshhold
        ]

        logger.debug(f'all_neighbour_entities: {all_neighbour_entities}')

        neighbour_entities = [
            e 
            for e in all_neighbour_entities 
            if e.score <= upper_score_threshold and e.score >= lower_score_threshhold
        ]

        neighbour_entities.sort(key=lambda e:e.score, reverse=True)

        end = time.time()
        duration_ms = (end-start) * 1000
        
        logger.debug(f'neighbour_entities: {neighbour_entities} ({duration_ms:.2f} ms)')

        return neighbour_entities

        
    def _get_entity_contexts(self, entities:List[ScoredEntity], entity_id_context_tree:Dict[str, Dict], query_bundle:QueryBundle) -> List[List[ScoredEntity]]:

        start = time.time()
       
        all_entities = {
            entity.entity.entityId:entity for entity in entities
        }

        logger.debug(f'all_entities: {all_entities}')

        all_contexts_map = {}

        def context_id(context):
            return ':'.join([se.entity.entityId for se in context])

        def walk_tree_ex(current_context, d):
            if not d:
               all_contexts_map[context_id(current_context)] = current_context
            
            for entity_id, children in d.items():
                context = [c for c in current_context]
                if entity_id in all_entities:
                    context.append(all_entities[entity_id])
                walk_tree_ex(context, children)
                

        walk_tree_ex([], entity_id_context_tree)

        logger.debug(f'all_contexts_map: {all_contexts_map}')

        partial_path_keys = []
        
        for key in all_contexts_map.keys():
            for other_key in all_contexts_map.keys():
                if key != other_key and other_key.startswith(key):
                    partial_path_keys.append(key)

        for key in partial_path_keys:
            all_contexts_map.pop(key, None)

        all_contexts = [context for _, context in all_contexts_map.items()]

        logger.debug(f'all_contexts: {all_contexts}')

        contexts = all_contexts[:self.args.ec_max_contexts]

        end = time.time()
        duration_ms = (end-start) * 1000

        logger.debug(f'contexts: {contexts} ({duration_ms:.2f} ms)')

        return contexts

                        
    def get_entity_contexts(self, entities:List[ScoredEntity], query_bundle:QueryBundle)  -> List[List[ScoredEntity]]:

        start = time.time()

        if entities:
        
            entity_id_context_tree = self._get_entity_id_context_tree(entities)
            
            neighbour_entities = self._get_neighbour_entities(
                entity_id_context_tree=entity_id_context_tree,
                baseline_score=entities[0].score
            )

            entities.extend(neighbour_entities)     
        
            entity_contexts = self._get_entity_contexts(
                entities=entities,
                entity_id_context_tree=entity_id_context_tree,
                query_bundle=query_bundle
            )

        else:
            entity_contexts = []

        end = time.time()
        duration_ms = (end-start) * 1000

        logger.debug(f"""Retrieved {len(entity_contexts)} entity contexts for '{query_bundle.query_str}' ({duration_ms:.2f} ms): {[
            str([e.entity.value for e in context])
            for context in entity_contexts
        ]}""")
    
       
        return entity_contexts