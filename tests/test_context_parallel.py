"""CPU tests for the context-parallel launcher configuration.

Real-collective coverage of ``all_to_all_with_grad`` (the Ulysses exchange
forward and its transposed backward on NCCL) lives with the other torchrun
tests in tests/test_cp_attention.py, launched by
tests/run_cp_attention_tests.sh.  The mp.spawn+Queue harness that used to run
it from this file crashed on GPU CI nodes: the spawn-context mp.Queue's named
POSIX semaphores vanished from /dev/shm before the worker processes could
unpickle the queue (SemLock._rebuild -> FileNotFoundError), so the workers
died at startup and the parent timed out in queue.get.
"""

from types import SimpleNamespace

import pytest

from model_training.common.context_parallel import configure_context_parallel_shard_size


def test_context_parallel_defaults_shard_size_when_unset():
    args = SimpleNamespace(context_parallel_size=8, shard_size=None)

    configure_context_parallel_shard_size(args)

    assert args.shard_size == 8


def test_context_parallel_requires_matching_shard_size():
    args = SimpleNamespace(context_parallel_size=8, shard_size=16)

    with pytest.raises(AssertionError, match="shard_size must equal context_parallel_size"):
        configure_context_parallel_shard_size(args)


def test_context_parallel_size_one_preserves_explicit_shard_size():
    args = SimpleNamespace(context_parallel_size=1, shard_size=4)

    configure_context_parallel_shard_size(args)

    assert args.shard_size == 4
