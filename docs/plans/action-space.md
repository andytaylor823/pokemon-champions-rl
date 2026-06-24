---
last_synced: a39ccc9
watches:
  - src/
  - sim/src/
  - docs/architecture/repo-architecture.md
---

# Action Space — Implementation Plan

> Module ref: `docs/architecture/repo-architecture.md` §3.6

## Status

Phase 1 implementation complete. Module: `src/action_space.py`.

## Scope

Flat joint action space (`A = 1089`): 360 team-preview orderings + 729 move-phase joint actions (27 per-slot actions x 27). Legality expressed as a boolean mask. Per-slot decomposition: 4 moves x 3 targets = 12 base + 12 mega + 2 switches + 1 pass = 27. Translation between canonical index and Showdown choice strings (`index_to_choice_string`, `choice_string_to_index`) lives in this module. The CVPN policy head is a single flat `[A]` softmax over this space.
