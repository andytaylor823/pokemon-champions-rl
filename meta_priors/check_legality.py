"""CLI entrypoint for legality checking.

Usage:
    python -m meta_priors.check_legality <tournament-url-or-id>

Examples:
    python -m meta_priors.check_legality \\
        https://play.limitlesstcg.com/tournament/69cdcda5d478313a15a39666/standings

    python -m meta_priors.check_legality 69cdcda5d478313a15a39666
"""

from meta_priors.legality import main

if __name__ == "__main__":
    main()
