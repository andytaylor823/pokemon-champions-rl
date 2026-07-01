"""Throwaway module to smoke-test ship-module's outward pipeline. Safe to delete."""


def accumulate(value, bucket=[]):  # intentional flaw for the smoke test
    bucket.append(value)
    return bucket
