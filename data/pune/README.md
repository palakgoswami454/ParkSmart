# Pune data

Hand-curated, sourced facts about parking in Pune. They were checked on **29 Oct 2026**. The files are loaded and validated by `src/data_loader.py`, and `tests/test_pune.py` checks them on every test run.

| File | Contents |
|---|---|
| `facilities.json` | 19 parking facilities and two published tariffs |
| `destinations.json` | 16 destinations for *Find parking*, each a cited landmark/station point (Camp is matched by name only) |
| `policy.json` | dated policy/news items, PMC zone-rate tables, and the list of things searched for but not found |
| `sources.json` | 40 sources: official documents, news reports, Wikipedia articles, third-party directory listings |

## Rules

1. **Only what a source says.** `null` means the sources found don't say. No capacity, price, hour, coordinate or occupancy is estimated.
2. **Every facility has a status reflecting the evidence on the source date.**

| Status | Meaning | Ranking weight |
|---|---|---|
| `reported` | An official statement or report says it operates (e.g. Cantonment Board rates Jul 2026, airport car park opening 2022) | 0.90 |
| `private` | Parking at a private venue, documented on Wikipedia | 0.70 |
| `lapsed` | In an official register, but the listed contract period has ended (Pune station) | 0.50 |
| `announced` | Announced (Pune Metro, Feb 2024) with no later confirmation found | 0.40 |
| `listing` | Only in a third-party directory | 0.35 |
| `conflicting` | Announced, then contradicted by a later report (PCMC, Swargate metro) | 0.20 |
| `restricted` | Exists but not for the public (Swargate bus stand: bus travellers only) | excluded |
| `closed` | Reported non-operational (Chhatrapati Sambhaji Maharaj multi-level parking) | excluded |

3. **Coordinates mark the cited station, landmark or listing, not a surveyed entrance.** `coord_precision` says which. Wikipedia locality coordinates were cross-checked against nearby stations. Kothrud, Aundh and Pune Camp were wrong by several km and were **not** used. The Mandai lot's position was decoded from the plus code `GV64+P85` given in its listing.
4. **No Pune occupancy.** No public occupancy history or feed was found. The facility schema deliberately has no occupancy or availability field, and a test enforces this.
5. The PMC open-data portal, the data.gov.in "Parking lots: Pune" catalog and the Pune Smart City planning PDF could not be read from the build environment (robots/permission blocks). Nothing here depends on them. If you can download the data.gov.in resource, it would be the first place to look for an official lot list.

## Updating

Edit the JSON, add any new source to `sources.json` with its `accessed` date, update `last_checked`, then run `pytest -q tests/test_pune.py`. The loader rejects a fact without a source, a coordinate outside the Pune area, an unknown status, or a malformed tariff.
