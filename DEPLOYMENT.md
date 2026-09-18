# Coolify deployment

The production image is `docker/prospective.Dockerfile`. It serves the public
dashboard and runs the frozen 365-day, four-region prospective protocol. Keep
the scheduled task disabled until every state-verification command below passes.

## Environment

Configure these values in Coolify. Never commit their real values.

```text
DATABASE_URL=postgresql://...
S3_ENDPOINT_URL=https://...
S3_BUCKET=...
S3_REGION=...
S3_ACCESS_KEY_ID=...
S3_SECRET_ACCESS_KEY=...
S3_PREFIX=prospective/v1
PROSPECTIVE_PROTOCOL_PATH=configs/prospective/multi-region-spatial-etas-prospective-v1.json
AUTO_MIGRATE=1
VERIFY_OBJECT_STORAGE=1
REQUIRE_OBJECT_STORAGE=1
PUBLIC_BASE_URL=https://etas2.bboga.com
NEWSLETTER_PUBLIC_BASE_URL=https://etas2.bboga.com
NEWSLETTER_FROM="Multi-Region ETAS Test <hello@bboga.com>"
```

Expose port `8080`; health is `/health`. If a database password or S3 key has
ever appeared in a terminal transcript or local deployment note, rotate it
before activation.

## One-time four-region state preparation

The commands below use the already verified California state and build the
three new regional states. `CUTOFF` must remain identical in every bootstrap
command. `AS_OF` must be a completed UTC-midnight boundary no later than it.
The historical catalog imports are resumable and are the long-running steps.

After deploying the new image, first seed and verify the frozen protocol:

```bash
python scripts/verify_prospective_protocol.py
python scripts/verify_object_storage.py
python scripts/seed_prospective_database.py
```

Reuse the byte-identical, already verified California causal state:

```bash
python scripts/transfer_california_state_to_multiregion.py \
  --as-of "2026-09-18T00:00:00+00:00"
```

Bootstrap New Zealand, Chile, and Japan C. Japan target scoring is M≥5, while
its frozen neural feature context correctly retains M≥3 events.

```bash
python scripts/bootstrap_prospective_catalogs.py \
  --cutoff "2026-09-18T11:01:00+00:00" \
  --region new-zealand-csep \
  --region chile-subduction \
  --region japan-c

python scripts/verify_prospective_bootstrap.py \
  --cutoff "2026-09-18T11:01:00+00:00" \
  --region new-zealand-csep \
  --region chile-subduction \
  --region japan-c
```

Build the three external states directly at the same causal boundary:

```bash
python scripts/build_prospective_initial_states.py \
  --as-of "2026-09-18T00:00:00+00:00" \
  --catalog-cutoff "2026-09-18T11:01:00+00:00" \
  --region new-zealand-csep \
  --region chile-subduction \
  --region japan-c
```

Verify all four states together:

```bash
python scripts/verify_prospective_initial_states.py \
  --as-of "2026-09-18T00:00:00+00:00" \
  --region california-relm \
  --region new-zealand-csep \
  --region chile-subduction \
  --region japan-c
```

The evidence gate is exactly zero at prospective start in every region.
Retrospective outcomes are never imported into its online evidence state.

## Start the 365-day clock

Only after the four-region verification succeeds, schedule this at `00:05 UTC`
every day:

```bash
python scripts/run_prospective_daily.py
```

The first run collects rolling catalogs, advances causal states, publishes all
four forecast pairs before `00:15 UTC`, and then activates the protocol only if
all four regions published the same target day. A partial first issue does not
start the clock. Missed forecasts are recorded and never backfilled.

Completed days receive a provisional IGPE and CSEP N/L/R score immediately.
The same target day receives its catalog-settled final score seven days later;
forecasting continues normally during that interval.

The optional double-opt-in status email runs independently at `03:00 UTC`:

```bash
python scripts/send_daily_newsletter.py
```
