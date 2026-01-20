# Docflow sync notes

Purpose: track the docflow files and commits that map to this repo, so future
syncs are quick and consistent.

Last sync: 2026-01-20
Current x-liked-tweets-exporter HEAD: b8d7815
Docflow commit with equivalent tweet extraction logic: ba8e872

Relevant docflow files to compare
- ../docflow/utils/tweet_to_markdown.py -> download_liked_tweets.py (tweet parsing + markdown)
- ../docflow/utils/x_likes_fetcher.py -> download_liked_tweets.py (likes collection)
- ../docflow/utils/create_x_state.py -> create_x_state.py

Common commands
- git -C ../docflow log --oneline -- utils/tweet_to_markdown.py utils/x_likes_fetcher.py utils/create_x_state.py
- diff -u download_liked_tweets.py ../docflow/utils/tweet_to_markdown.py
- diff -u create_x_state.py ../docflow/utils/create_x_state.py

Notes
- Docflow changes in pipeline_manager, config, serve_docs, and web UI are
  usually not relevant to this exporter.
- This repo keeps HTML conversion + likes history helpers that do not exist in
  docflow; avoid deleting them during syncs.
- Latest tweet-utils change pulled from docflow: login-wall/unavailable detection (ba8e872).
