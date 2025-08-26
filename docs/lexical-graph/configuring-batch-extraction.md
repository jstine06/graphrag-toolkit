[[Home](./)]

## Configuring Batch Extraction

### Topics

  - [Overview](#overview)
  - [Optimizing batch extraction performance](#optimizing-batch-extraction-performance)

### Overview

### Optimizing batch extraction performance

The most important setting for controlling batch extraction performance are:

  - `GraphRAGConfig.extraction_batch_size`: Sets how many source documents go to the extraction pipeline. Ensure (source documents × average chunks per document) is enough to fill your planned simultaneous batch jobs
  - `GraphRAGConfig.extraction_num_workers`: Sets how many CPUs run batch jobs simultaneously
  - `BatchConfig.max_num_concurrent_batches`: Sets how many concurrent batch jobs each worker runs
  - `BatchConfig.max_batch_size`: Sets the maximum number of chunks per batch job

To maximize the efficiency of batch extraction, follow these three key principles:

  - **Maximize file capacity** Each batch job file can hold up to 50,000 records. However, Amazon Bedrock enforces input file size limits, typically between 1-5 GB. Check the specific limits for your model in the Amazon Bedrock service quotas section (see the **Batch inference job size** quotas in the [Amazon Bedrock service quotas section](https://docs.aws.amazon.com/general/latest/gr/bedrock.html#limits_bedrock ) for the limits particuar to the model you are using). Note that the toolkit doesn't automatically verify file sizes, so jobs may fail if they exceed these quotas. You may need to use fewer records than the maximum limit to stay within file size boundaries. Configure the `BatchConfig.max_batch_size` to set the maximum number of records per batch job.
  - **Use larger, fewer files** Focus on using a minimal number of large files rather than splitting the work across many smaller ones. For example, it's more efficient to process 40,000 records in a single job than to divide them into four parallel jobs of 10,000 records each.
  - **Leverage parallel processing** Take advantage of parallel job execution using `GraphRAGConfig.extraction_num_workers` and `BatchConfig.max_num_concurrent_batches`. The total number of jobs (number of workers × number of concurrent batches) must stay within Bedrock's quota of 20 combined in-progress and submitted batch inference jobs per region. If you exceed this limit, additional jobs will wait in the queue until capacity becomes available.
