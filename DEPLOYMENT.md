# Coolify deployment

The production image is `docker/prospective.Dockerfile`. The first deployment
serves the dashboard and seeds the locked 14-day operational dry-run protocol;
it does not automatically start a claim-bearing scientific test.

## Environment

Configure these values in Coolify:

```text
DATABASE_URL=postgresql://...
S3_ENDPOINT_URL=https://...
S3_BUCKET=...
S3_REGION=...
S3_ACCESS_KEY_ID=...
S3_SECRET_ACCESS_KEY=...
S3_PREFIX=prospective/v1
PROSPECTIVE_PROTOCOL_PATH=configs/prospective/evidence-gate-california-dry-run-v1.json
AUTO_MIGRATE=1
VERIFY_OBJECT_STORAGE=1
REQUIRE_OBJECT_STORAGE=1
PUBLIC_BASE_URL=https://YOUR-DOMAIN
NEWSLETTER_PUBLIC_BASE_URL=https://YOUR-DOMAIN
```

Expose container port `8080`. The health endpoint is `/health`.

## One-time causal state bootstrap

Run the following in the deployed container. Replace `CUTOFF` with one fixed
UTC timestamp after the intended `TO_AS_OF` boundary. Keep exactly the same
cutoff in every command; interrupted catalog imports are resumable. These are
the long-running initialization commands and should be run manually.

```bash
python scripts/verify_prospective_protocol.py
python scripts/verify_object_storage.py

python scripts/bootstrap_prospective_catalogs.py \
  --cutoff "CUTOFF" --region california-relm
python scripts/verify_prospective_bootstrap.py \
  --cutoff "CUTOFF" --region california-relm

python scripts/build_prospective_initial_states.py \
  --as-of "2026-08-19T00:00:00+00:00" \
  --catalog-cutoff "CUTOFF" --region california-relm
python scripts/verify_prospective_initial_states.py \
  --as-of "2026-08-19T00:00:00+00:00" --region california-relm

python scripts/advance_prospective_bootstrap_states.py \
  --from-as-of "2026-08-19T00:00:00+00:00" \
  --to-as-of "TO_AS_OF" --catalog-cutoff "CUTOFF" \
  --region california-relm
python scripts/verify_prospective_initial_states.py \
  --as-of "TO_AS_OF" --region california-relm
```

`TO_AS_OF` must be a UTC-midnight boundary no later than `CUTOFF`. Use the
latest completed UTC-midnight boundary before the first scheduled issuance.
The evidence gate remains at log Bayes factor zero throughout bootstrap;
retrospective outcomes are never imported into its online evidence state.

## Daily schedule

After bootstrap, schedule this command at `00:05 UTC` every day:

```bash
python scripts/run_prospective_daily.py
```

It collects a revision-aware catalog snapshot, advances causal state, publishes
the next UTC day's immutable ETAS/safe/fixed/gated layers before `00:15 UTC`,
and scores completed targets. A missed forecast is recorded and never
backfilled.

The optional double-opt-in status email runs independently at `03:00 UTC`:

```bash
python scripts/send_daily_newsletter.py
```

After the 14-day dry run and seven-day catalog settlement pass operational
review, create and commit a separate 365-day protocol before starting the
scientific prospective clock.
