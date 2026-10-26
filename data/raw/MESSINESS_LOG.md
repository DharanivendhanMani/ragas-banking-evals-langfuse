# Messiness Log (ground truth for the golden set)

All documents are synthetic. "Northbridge Bank" is fictional. This log records every deliberate flaw so golden-set questions and error analysis can be tied to a known cause. **Do not index this file in the RAG corpus.**

## Corpus summary

- 8 clean documents (Markdown), 5 messy document groups (6 files).
- Effective dates: most January to March 2024; savings/CD rates July 2024; branch bulletin July 2024.

## Messy documents

### 1. `fee_schedule_2024.pdf`: table-heavy PDF
- Merged header cells, vertically merged category cells (Overdraft / Wires / ATM-Card / Other), and numbered footnote markers in a "Notes" column.
- Text extraction flattens the tables: the category cell is lost for most rows, columns come out one value per line, and footnote numbers appear as stray digits.
- Footnotes at the bottom hold material conditions (e.g. note 1: the $500 direct-deposit waiver counts total qualifying electronic deposits; note 4: no fee if overdrawn by $10 or less; note 7: paper statement fee waived with paperless).
- **Tests:** table and numeric lookup, retrieval of footnote conditions, chunking that splits a table.

### 2. `atm_policy_v2023.md` vs `atm_policy_v2024.md`: outdated and conflicting versions
- v1.0 (2023-01-01) is **superseded** but carries no "superseded" banner. v2.0 (2024-03-01) states it supersedes v1.0.
- Conflicting values (correct answer = v2024):

| Item | v2023 (outdated) | v2024 (current) |
|---|---|---|
| Everyday Checking ATM withdrawal limit | $500/day | $800/day |
| Premier ATM withdrawal limit | $1,000/day | $1,500/day |
| Student ATM withdrawal limit | $300/day | $400/day |
| Out-of-network U.S. ATM fee | $2.50 | $3.00 |
| Non-U.S. ATM fee | $5.00 | $5.00 + 1% FX charge |
| Expedited card replacement | $20 | $25 |
| Temporary limit increase validity | 7 days | 14 days |

- Both versions are indexed. A good system should prefer the current version (or at least not mix the two).
- The v2024 fee ($3.00 out-of-network, $5.00 non-U.S., $25 expedited card) matches the fee schedule.
- **Tests:** version and conflict handling, context precision (old chunk outranking new), faithfulness when contexts disagree.

### 3. `chargeback_guide.md`: internal contradiction
- Quick Summary says the merchant has **10 business days** to respond. The Detailed Process says **14 calendar days**.
- There is no authoritative answer inside the document. A well-behaved bot should flag the inconsistency or answer cautiously, not silently pick one.
- **Tests:** unanswerable/ambiguous behaviour, faithfulness with contradictory contexts.

### 4. `branch_operations_notice.pdf`: layout noise
- Small page size, so paragraphs split across pages mid-sentence. A running header and footer (with page numbers) appear in extracted text on every page.
- A hyphenated line break ("holi-days") is baked into the text.
- **Tests:** chunk boundary robustness, retrieval of facts that straddle pages (e.g. notary requirements split across pages 1-2).

### 5. `customer_faq.md`: near-duplicate entries with inconsistent answers
- Three password-reset entries: minimum length 12 vs 10 characters, code by phone vs phone-or-email, in-branch fallback only in one.
- Three mobile-deposit availability entries: "next business day" vs "1 to 2 business days" vs "by the second business day". The authoritative statement is `account_terms.md` (next business day before 8:00 PM ET).
- Three lost/stolen-card entries: the third lists 1-800-555-0149, while the authoritative card services number (also in `card_dispute_policy.md` and `fraud_faq.md`) is **1-800-555-0142**.
- Two address-change entries and two paper-statement entries with slightly different wording (low risk).
- **Tests:** context precision (duplicates crowd the top-k), conflicting facts, fabricated-looking phone numbers.

## Facts that must stay consistent across documents (not flawed)

| Fact | Value | Where |
|---|---|---|
| Overdraft / NSF fee | $34, max 3/day | fee schedule, overdraft policy |
| Overdraft Protection transfer | $10 | fee schedule, overdraft policy |
| Domestic / international outgoing wire | $25 / $45 | fee schedule, wire rules |
| Card services / fraud line | 1-800-555-0142 | dispute policy, fraud FAQ |
| Provisional credit (debit) | after 5 business days | dispute policy |
| Early account closure fee | $25 within 90 days | fee schedule, account terms |

## Known approximations

- Some real-world rules (Regulation E, Regulation CC) are paraphrased and simplified. They are used for realism, not legal accuracy.
