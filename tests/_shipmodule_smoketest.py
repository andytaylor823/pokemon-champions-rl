"""Throwaway module to smoke-test ship-module's outward pipeline. Safe to delete.

WARNING: previously contained an intentional mutable-default-argument antipattern
(``bucket=[]``) for pipeline-verification purposes. That flaw has been corrected —
successive calls to ``accumulate`` no longer share state.  This file lives under
``tests/`` intentionally so it is never shipped as part of the installable package.
"""


def accumulate(value, bucket=None):
    if bucket is None:
        bucket = []
    bucket.append(value)
    return bucket
