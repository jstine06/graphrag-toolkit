# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import argparse
import json
import logging
import sys

from graphrag_toolkit.lexical_graph.storage.vector.repair_opensearch_vector_store import repair_opensearch_vector_store as repair

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def do_repair():

    parser = argparse.ArgumentParser()
    parser.add_argument('--graph-store', help = 'Graph store connection string')
    parser.add_argument('--vector-store', help = 'Vector store connection string')
    parser.add_argument('--tenant-ids', nargs='*', help = 'Space-separated list of tenant ids (optional)')
    parser.add_argument('--batch-size', nargs='?', help = 'Batch size (optional, default 100)')
    parser.add_argument('--dry-run', action='store_true', help = 'Dry run (optional)')
    args, _ = parser.parse_known_args()
    args_dict = { k:v for k,v in vars(args).items() if v}

    graph_store_info = args_dict['graph_store']
    vector_store_info = args_dict['vector_store']
    tenant_ids = args_dict.get('tenant_ids', [])
    batch_size = int(args_dict.get('batch_size', 1000))
    dry_run = bool(args_dict.get('dry_run', False))

    results = repair(graph_store_info, vector_store_info, tenant_ids, batch_size, dry_run)
    print(json.dumps(results, indent=2))
    
            
if __name__ == '__main__':
    do_repair()