# Metadata-only baseline (sanity check)

Features: keyword + country_code + word_count (no text).

- threshold=0.50: f1=0.2541 p=0.1550 r=0.7035
- best-on-dev threshold=0.505 (optimistic upper bound): f1=0.2571 p=0.1573 r=0.7035

Purpose: quantify how far topic priors + metadata can go, and show that strong performance requires modelling the text.