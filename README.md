# X Liked Tweets Exporter

This repo contains a single script that downloads your liked tweets and saves
one Markdown file and one HTML file per tweet. It uses Playwright to read the
likes page and open each tweet, and it does not use the X API.

Note: X does not provide an official export for bookmarks. This script exports
likes (favorites), which many people use as a lightweight bookmark workflow.

This project is extracted from the original docflow project:
https://github.com/domingogallardo/docflow

## What it does

- Loads your X likes page
- Collects the newest liked tweet URLs (up to a limit you set)
- Saves each tweet as:
  - `Tweet - handle-id.md`
  - `Tweet - handle-id.html`

## Requirements

- Python 3.10+
- Playwright + Chromium
- markdown (Python package)

## Quick start

1) Clone this repo and enter it:

```bash
git clone <REPO_URL>
cd x-liked-tweets-exporter
```

2) Install dependencies:

```bash
pip install playwright markdown
playwright install chromium
```

3) Create a logged in storage state (one time):

```bash
python create_x_state.py
```

Log in when the browser opens, then press Enter in the terminal. The file
`x_state.json` will be created in the repo.

If Chrome is not installed, run:

```bash
python create_x_state.py --channel ""
```

4) Run the downloader:

```bash
python download_liked_tweets.py \
  --likes-url https://x.com/YOUR_USER/likes \
  --max-tweets 50 \
  --dest-dir output
```

The `output/` folder will contain both `.md` and `.html` files.

## Re-running without duplicates

By default the script keeps a `tweets_processed.txt` file in the destination
folder. On the next run it stops when it reaches the most recent URL in that
file and skips URLs that were already processed.

You can disable this behavior with:

```bash
python download_liked_tweets.py --no-history ...
```

You can also pass your own history file:

```bash
python download_liked_tweets.py --processed-file /path/to/processed.txt ...
```

## Options

- `--likes-url`        X likes URL (example: `https://x.com/USER/likes`)
- `--state-path`       Path to your Playwright storage state (default: `x_state.json`)
- `--dest-dir`         Output folder (default: `liked_tweets`)
- `--max-tweets`       Number of likes to capture in this run (default: 50)
- `--stop-at-url`      Stop when this URL appears in the likes feed
- `--processed-file`   Path to the processed URLs file
- `--no-history`       Disable processed URLs tracking
- `--wait-ms`          Extra wait time after loading each tweet (default: 5000)
- `--no-headless`      Run Chromium with UI (useful if login walls appear)

## Environment variables

These are optional alternatives to CLI flags:

- `TWEET_LIKES_URL`
- `TWEET_LIKES_STATE`
- `TWEET_LIKES_DEST`
- `TWEET_LIKES_MAX`
- `TWEET_LIKES_WAIT_MS`

## Notes and troubleshooting

- The `x_state.json` file contains your login cookies. Keep it private.
- If you see empty results, run with `--no-headless` and confirm you are logged in.
- X changes its UI frequently. If the script breaks, update Playwright and re-run.
