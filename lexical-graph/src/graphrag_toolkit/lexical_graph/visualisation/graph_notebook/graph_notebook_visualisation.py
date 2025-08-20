# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import pandas as pd
import json

from json import JSONDecodeError
from typing import Optional

from graphrag_toolkit.lexical_graph.retrieval.model import SearchResult
from graphrag_toolkit.lexical_graph.tenant_id import to_tenant_id

LABELS_TO_REFORMAT = ['Source', 'Chunk', 'Topic', 'Statement', 'Fact', 'Entity']

def format_params(params):
    return str(params).replace("'startId'", 'startId').replace("'endId'", 'endId').replace("'endIds'", 'endIds')

def for_each_disjoint_unique(values):
    params = []
    for idx, value in enumerate(values[:-1]):
        other_values = values[idx+1:]
        params.append({'startId': value, 'endIds': other_values})
    return params

def get_query_params(nodes):
    
    source_topic_ids = []
    topic_statement_ids = []
    statement_ids = []
    
    for n in nodes:
        
        search_result = SearchResult.model_validate(n.metadata['result'])
        source_id = search_result.source.sourceId
        
        for topic in search_result.topics:
        
            source_topic_ids.append({'startId': source_id, 'endId': topic.topicId})
            
            for statement in topic.statements:
            
                topic_statement_ids.append({'startId': topic.topicId, 'endId': statement.statementId})
                statement_ids.append(statement.statementId)
                             
    
    query_parameters = { 
        'source_topic_ids': source_topic_ids,
        'topic_statement_ids': topic_statement_ids,
        'statement_id_sets': for_each_disjoint_unique(statement_ids)
    }
    
    return query_parameters

def get_query(query_parameters, include_sources=True):
    
    sources_cypher = '' if not include_sources else f'''
    UNWIND {format_params(query_parameters["source_topic_ids"])} AS source_topic_ids
    MATCH p=(s)<-[:`__EXTRACTED_FROM__`]-()<-[:`__MENTIONED_IN__`]-(t)
    WHERE id(s) = source_topic_ids.startId AND id(t) = source_topic_ids.endId
    RETURN p
    UNION
    '''
    
    cypher = f'''{sources_cypher}
    UNWIND {format_params(query_parameters["topic_statement_ids"])} AS topic_statement_ids
    MATCH p=(t)<-[:`__BELONGS_TO__`]-(l)<-[:`__SUPPORTS__`]-()
    WHERE id(t) = topic_statement_ids.startId AND id(l) = topic_statement_ids.endId
    RETURN p
    UNION
    UNWIND {format_params(query_parameters["statement_id_sets"])} AS statement_id_sets
    MATCH p=(l)<-[:`__SUPPORTS__`]-()<-[:`__SUBJECT__`|`__OBJECT__`]-()
            -[:`__SUBJECT__`|`__OBJECT__`]->()-[:`__SUPPORTS__`]->(ll)
    WHERE id(l) = statement_id_sets.startId AND id(ll) IN statement_id_sets.endIds
    RETURN p LIMIT 10
    '''
    
    return cypher

def get_schema_query(tenant_id):

    label = tenant_id.format_label('__SYS_Class__')

    cypher = f'''MATCH (n:{label})
    WITH n, n.count AS score ORDER BY score DESC LIMIT 100
    CALL {{ 
        WITH n 
        MATCH p=(n)-[r]->(x) //WHERE n <> x
        WITH p, r.count AS score ORDER BY score DESC LIMIT 10
        RETURN p
    }}
    RETURN DISTINCT p'''

    return cypher

class GraphNotebookVisualisation():

    def __init__(self, display_edge_labels=False, formatting_config=None, nb_classic=False):
        self.display_edge_labels = display_edge_labels
        self.formatting_config = formatting_config
        self.nb_classic = nb_classic

    def _get_graph(self, formatting_config):

        try:
            import graph_notebook.magics.graph_magic
            from graph_notebook.magics import Graph
            from graph_notebook.options import OPTIONS_DEFAULT_DIRECTED, vis_options_merge
            from graph_notebook.magics.graph_magic import encode_html_chars
            from graph_notebook.visualization.rows_and_columns import opencypher_get_rows_and_columns
        except ImportError as e:
            raise ImportError(
                "graph_notebook package not found, install with 'pip install graph_notebook'"
            ) from e
        
        def oc_results_df(oc_res, oc_res_format: str = None):

            def reformat_label(l):
                for label in LABELS_TO_REFORMAT:
                    if l.startswith(f'__{label}'):
                        return label
                return l
                
            
            results = []
            
            for result in oc_res['results']:
                for e in result['p']:
                    if e['~entityType'] == 'node':
                        e['~labels'] = [reformat_label(l) for l in e['~labels']]
                    else:
                        e['~type'] = e['~type'].lower().replace('__', '')
                results.append(result)
                
            oc_res['results'] = results
            
            
            rows_and_columns = opencypher_get_rows_and_columns(oc_res, oc_res_format)
            if rows_and_columns:
                results_df = pd.DataFrame(rows_and_columns['rows']).convert_dtypes()
                results_df = results_df.astype(str)
                results_df = results_df.map(lambda x: encode_html_chars(x))
                col_0_value = range(1, len(results_df) + 1)
                results_df.insert(0, "#", col_0_value)
                for col_index, col_name in enumerate(rows_and_columns['columns']):
                    results_df.rename({results_df.columns[col_index + 1]: col_name},
                                      axis='columns',
                                      inplace=True)
                has_results = True
            else:
                results_df = None
                has_results = False
            return results_df, has_results
        
        def _graph_notebook_vis_options(self, line='', cell='', local_ns: dict = None):
            parser = argparse.ArgumentParser()
            parser.add_argument('--silent', action='store_true', default=False, help="Display no output.")
            parser.add_argument('--store-to', type=str, default='', help='store visualization settings to this variable')
            parser.add_argument('--load-from', type=str, default='', help='load visualization settings from this variable')
            line_args = line.split()
            if line_args:
                if line_args[0] == 'reset':
                    line = 'reset'
                    if len(line_args) > 1:
                        line_args = line_args[1:]
                    else:
                        line_args = []
            args = parser.parse_args(line_args)
    
            if line == 'reset':
                self.graph_notebook_vis_options = OPTIONS_DEFAULT_DIRECTED
    
            if cell == '' and not args.load_from:
                if not args.silent:
                    print(json.dumps(self.graph_notebook_vis_options, indent=2))
            else:
                try:
                    if args.load_from:
                        try:
                            options_raw = local_ns[args.load_from]
                            if isinstance(options_raw, dict):
                                options_raw = json.dumps(options_raw)
                            options_dict = json.loads(options_raw)
                        except KeyError:
                            print(f"Unable to load visualization settings, variable [{args.load_from}] does not exist in "
                                  f"the local namespace.")
                            return
                    else:
                        options_dict = json.loads(cell)
                except (JSONDecodeError, TypeError) as e:
                    print(f"Unable to load visualization settings, variable [{args.load_from}] is not in valid JSON "
                          f"format:\n")
                    print(e)
                    return
                self.graph_notebook_vis_options = vis_options_merge(self.graph_notebook_vis_options, options_dict)
                print("Visualization settings successfully changed to:\n")
                print(json.dumps(self.graph_notebook_vis_options, indent=2))
    
        
        
        graph_notebook.magics.graph_magic.oc_results_df = oc_results_df
        setattr(Graph, '_graph_notebook_vis_options', _graph_notebook_vis_options)

        g = Graph(None) 
        g._graph_notebook_vis_options('reset', cell=formatting_config, local_ns={})

        return g

    def display_schema(self, tenant_id:Optional[str]=None):
        
        formatting_config = '''
        {
            "edges": {
                "color": {
                "inherit": false
                },
                "smooth": {
                "enabled": true,
                "type": "dynamic"
                },
                "arrows": {
                "to": {
                    "enabled": true,
                    "type": "arrow"
                }
                },
                "font": {
                "face": "courier new"
                }
            }
        }
        '''

        g = self._get_graph(formatting_config)

        line = f'query  -d value --edge-display-property value -rel 25 -l 25'

        cypher = get_schema_query(to_tenant_id(tenant_id))
        
        g.oc(line, cell=cypher, local_ns={}) 


    def display_results(self, response, include_sources=True):

        face = 'FontAwesome' if self.nb_classic else "'Font Awesome 5 Free'"
        
        formatting_config = self.formatting_config or f'''
        {{
          "groups": {{
            "Source": {{
              "shape": "icon",
              "icon": {{
                "face": "{face}",
                "weight": "bold",
                "code": "\uf15b",
                "color": "#336699",
                "size": 100
              }}
            }},
            "Chunk": {{
              "shape": "icon",
              "icon": {{
                "face": "{face}",
                "weight": "bold",
                "code": "\uf249",
                "color": "#336699",
                "size": 60
              }}
            }},
            "Topic": {{
              "shape": "icon",
              "icon": {{
                "face": "{face}",
                "weight": "bold",
                "code": "\uf07b",
                "color": "#669900",
                "size": 100
              }}
            }},
            "Statement": {{
              "shape": "icon",
              "icon": {{
                "face": "{face}",
                "weight": "bold",
                "code": "\uf0ce",
                "color": "#99cc00",
                "size": 60
              }}
            }},
            "Fact": {{
              "shape": "icon",
              "icon": {{
                "face": "{face}",
                "weight": "bold",
                "code": "\uf1b3",
                "color": "#99cc00",
                "size": 50
              }}
            }},
            "Entity": {{
              "shape": "icon",
              "icon": {{
                "face": "{face}",
                "weight": "bold",
                "code": "\uf1b2",
                "color": "#ff9900",
                "size": 80
              }}
            }}
          }}
        }}
        '''

        g = self._get_graph(formatting_config)

        edge_label_length = 25 if self.display_edge_labels else 0
        line = f'query -d value -l 25 -rel {edge_label_length}'

        nodes = response if isinstance(response, list) else response.source_nodes
        query_parameters = get_query_params(nodes)
        cypher = get_query(query_parameters, include_sources)
        
        g.oc(line, cell=cypher, local_ns={}) 