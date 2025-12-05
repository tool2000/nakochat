from collections import deque

import torch
import pyarrow.parquet as pq
import pyarrow.ipc as ipc

from nanochat.common import get_dist_info
from nanochat.dataset import list_parquet_files
from nanochat.tokenizer import get_tokenizer

def tokenizing_distributed_data_loader_with_state(B, T, split, tokenizer_threads=4, tokenizer_batch_size=128, device="cuda", resume_state_dict=None):
    """
    Stream pretraining text from parquet files, tokenize, yield training batches.

    This implementation became a bit more complex because we wish to support approximate resume training.
    Instead of turning this into a Class, we opt to return the state_dict with every batch,
    and then the caller can pass in a state_dict to resume training from a desired point.
    Note that this resumption is atm only *approximate* for simplicity.
    We won't repeat the same documents but we might skip a few.
    The state_dict that is returned can be later passed into this function via `resume_state_dict` to approximately resume.

    Perfect state resumption is possible but would be a lot more bloated, probably not worth it atm.
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"

    # infinite iterator over document batches (list of text strings)
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    def document_batches():
        parquet_paths = list_parquet_files(split=split)
        resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict is not None else 0
        resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict is not None else None
        pq_idx = resume_pq_idx # we kick off parquet files at the resume index (or by default just 0)
        while True: # iterate infinitely (multi-epoch)
            while pq_idx < len(parquet_paths): # iterate over all parquet files
                filepath = parquet_paths[pq_idx]
                is_parquet = filepath.endswith(".parquet")
                if is_parquet:
                    pf = pq.ParquetFile(filepath)
                    num_chunks = pf.num_row_groups
                    read_chunk = lambda idx: pf.read_row_group(idx)
                    to_list = lambda chunk: chunk.column('text').to_pylist()
                    chunk_iterable = None
                else:
                    # Try Arrow file first, fall back to stream
                    chunk_iterable = None
                    try:
                        reader = ipc.open_file(filepath)
                        num_chunks = reader.num_record_batches
                        read_chunk = lambda idx: reader.get_batch(idx)
                        to_list = lambda batch: batch.column('text').to_pylist()
                    except Exception:
                        reader = ipc.open_stream(filepath)
                        num_chunks = None  # unknown
                        chunk_iterable = enumerate(reader)
                        to_list = lambda batch: batch.column('text').to_pylist()

                # Start from resume point if resuming on same file, otherwise from DDP rank
                if resume_rg_idx is not None:
                    base_idx = resume_rg_idx // ddp_world_size
                    base_idx += 1
                    rg_idx = base_idx * ddp_world_size + ddp_rank
                    resume_rg_idx = None
                else:
                    rg_idx = ddp_rank

                if chunk_iterable is not None:
                    # Stream reader path (no random access)
                    for idx, batch in chunk_iterable:
                        if idx < rg_idx or (idx - ddp_rank) % ddp_world_size != 0:
                            continue
                        chunk = batch
                        batch_list = to_list(chunk)
                        for i in range(0, len(batch_list), tokenizer_batch_size):
                            yield batch_list[i:i+tokenizer_batch_size], (pq_idx, idx)
                    # cannot resume precisely in stream mode, but streams are rare
                else:
                    while rg_idx < num_chunks:
                        chunk = read_chunk(rg_idx)
                        batch = to_list(chunk)
                        for i in range(0, len(batch), tokenizer_batch_size):
                            yield batch[i:i+tokenizer_batch_size], (pq_idx, rg_idx)
                        rg_idx += ddp_world_size
                pq_idx += 1 # advance to the next parquet file
    batches = document_batches()

    # Now emit batches of tokens.
    needed_tokens = B * T + 1 # +1 is because we also need the target at the last token
    # get the tokenizer and the bos token
    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_bos_token_id()
    # scratch buffer holds the tokens for one iteration
    token_buffer = deque() # we stream tokens on the right and pop from the left
    while True:
        # Accumulate enough tokens for one iteration before yielding.
        while len(token_buffer) < needed_tokens:
            doc_batch, (pq_idx, rg_idx) = next(batches)
            token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
            for tokens in token_lists:
                token_buffer.extend(tokens)
        # Move tokens from the deque into the scratch buffer
        tokens = [token_buffer.popleft() for _ in range(needed_tokens)]
        # CUDA supports memory pinning for asynchronous transfers between CPU and GPU
        use_cuda_optimizations = device == "cuda"
        scratch = torch.tensor(tokens, dtype=torch.long, pin_memory=use_cuda_optimizations) # in PyTorch, long=int64
        # Create the inputs/targets as 1D tensors
        inputs_cpu = scratch[:-1]
        targets_cpu = scratch[1:]
        # Reshape to 2D and move to GPU async
        inputs = inputs_cpu.view(B, T).to(device=device, non_blocking=use_cuda_optimizations)
        targets = targets_cpu.view(B, T).to(device=device, non_blocking=use_cuda_optimizations)
        state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx} # we need this in case we wish to approximately resume training
        yield inputs, targets, state_dict

def tokenizing_distributed_data_loader(*args, **kwargs):
    # helper function that only emits the inputs/targets and not the state_dict
    for inputs, targets, state_dict in tokenizing_distributed_data_loader_with_state(*args, **kwargs):
        yield inputs, targets
