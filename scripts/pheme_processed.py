"""
PHEME Dataset Builder
Binary Veracity Version (FULL PHEME)

Labels:
1 = TRUE rumor
0 = FALSE rumor
"""

import json
from pathlib import Path


############################################################
# LOAD RETWEETS (JSONL)
############################################################

def load_retweets(retweet_file):

    retweets = []

    if not retweet_file.exists():
        return retweets

    with open(retweet_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                retweets.append(json.loads(line.strip()))
            except:
                continue

    return retweets


############################################################
# LOAD THREAD
############################################################

def load_pheme_thread(thread_dir, label, event_name):

    thread_id = thread_dir.name

    source_file = thread_dir / "source-tweet" / f"{thread_id}.json"

    if not source_file.exists():
        return None

    with open(source_file, "r", encoding="utf-8") as f:
        source_tweet = json.load(f)

    # Replies
    replies = []
    replies_dir = thread_dir / "reactions"

    if replies_dir.exists():
        for reply_file in sorted(replies_dir.glob("*.json")):
            try:
                with open(reply_file, "r", encoding="utf-8") as f:
                    replies.append(json.load(f))
            except:
                pass

    # Retweets
    retweets = load_retweets(thread_dir / "retweets.json")

    # Structure
    structure = {}
    structure_file = thread_dir / "structure.json"

    if structure_file.exists():
        with open(structure_file, "r", encoding="utf-8") as f:
            structure = json.load(f)

    tweets = [source_tweet] + replies
    tweets.sort(key=lambda t: t.get("created_at", ""))

    return {
        "thread_id": thread_id,
        "event_type": event_name,
        "source": source_tweet,
        "replies": replies,
        "retweets": retweets,
        "structure": structure,
        "tweets": tweets,
        "texts": [t.get("text", "") for t in tweets],
        "num_replies": len(replies),
        "num_retweets": len(retweets),
        "label": label,
    }


############################################################
# BUILD DATASET
############################################################

def build_pheme_dataset(raw_data_dir, output_dir):

    raw_data_dir = Path(raw_data_dir)
    output_dir = Path(output_dir)

    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    all_threads = []
    event_veracity = {}

    for event_dir in sorted(raw_data_dir.iterdir()):

        if not event_dir.is_dir():
            continue

        event_name = event_dir.name
        print(f"\nProcessing event: {event_name}")

        event_threads = []
        true_count = 0
        false_count = 0

        source_total = 0
        reply_total = 0
        retweet_total = 0

        ####################################################
        # ⭐ NEW: iterate rumours / non-rumours folders
        ####################################################
        for label_folder in ["rumours", "non-rumours"]:

            label_path = event_dir / label_folder

            if not label_path.exists():
                continue

            label = 1 if label_folder == "rumours" else 0

            for thread_dir in sorted(label_path.iterdir()):

                if not thread_dir.is_dir():
                    continue

                thread = load_pheme_thread(
                    thread_dir,
                    label,
                    event_name
                )

                if thread:
                    event_threads.append(thread)

                    source_total += 1
                    reply_total += thread["num_replies"]
                    retweet_total += thread["num_retweets"]

                    if label == 1:
                        true_count += 1
                    else:
                        false_count += 1

        ####################################################

        if event_threads:

            total_event = len(event_threads)

            print(f"  Loaded {total_event} threads")

            print("\n  Tweet Statistics:")
            print(f"      Source tweets : {source_total}")
            print(f"      Reply tweets  : {reply_total}")
            print(f"      Retweets      : {retweet_total}")

            true_ratio = (true_count / total_event) * 100
            false_ratio = (false_count / total_event) * 100

            print("\n  Veracity:")
            print(f"      True rumors : {true_count} ({true_ratio:.1f}%)")
            print(f"      False rumors: {false_count} ({false_ratio:.1f}%)")

            event_veracity[event_name] = {
                "true": true_count,
                "false": false_count,
                "sources": source_total,
                "replies": reply_total,
                "retweets": retweet_total,
            }

            event_output = output_dir / event_name
            event_output.mkdir(parents=True, exist_ok=True)

            for thread in event_threads:
                out_file = event_output / f"{thread['thread_id']}.json"
                with open(out_file, "w", encoding="utf-8") as f:
                    json.dump(thread, f, indent=2, ensure_ascii=False)

            all_threads.extend(event_threads)

    print("\n✅ Dataset built successfully")
    print(f"Total threads: {len(all_threads)}")

    create_dataset_summary(all_threads, event_veracity, output_dir)

    return all_threads


############################################################
# SUMMARY
############################################################

def create_dataset_summary(threads, event_veracity, output_dir):

    import collections

    label_counts = collections.Counter(t["label"] for t in threads)

    print("\n📊 Dataset Summary")
    print(f"True rumors : {label_counts[1]}")
    print(f"False rumors: {label_counts[0]}")

    print("\n📋 Event Breakdown:")
    for event, counts in event_veracity.items():
        total = counts["true"] + counts["false"]
        tr = counts["true"] / total * 100
        fr = counts["false"] / total * 100

        print(
            f"  {event}: {total} threads | "
            f"Sources={counts['sources']}, "
            f"Replies={counts['replies']}, "
            f"Retweets={counts['retweets']} | "
            f"True={counts['true']} ({tr:.1f}%), "
            f"False={counts['false']} ({fr:.1f}%)"
        )
        
    ########################################################
    # PAPER TABLE II STATISTICS (PER EVENT)
    ########################################################
    import numpy as np
    from collections import defaultdict, Counter

    print("\n===================================================")
    print("PAPER TABLE II VERIFICATION (PER EVENT)")
    print("===================================================")

    # store statistics per event
    event_stats = defaultdict(lambda: {
        "posts": 0,
        "users": set(),
        "events": 0,
        "true": 0,
        "false": 0,
        "event_sizes": []
    })

    # ----------------------------------------------------
    # Collect statistics
    # ----------------------------------------------------
    for thread in threads:

        event = thread["event_type"]
        tweets = thread["tweets"]
        label = thread["label"]

        event_stats[event]["events"] += 1
        event_stats[event]["posts"] += len(tweets)
        event_stats[event]["event_sizes"].append(len(tweets))

        if label == 1:
            event_stats[event]["true"] += 1
        else:
            event_stats[event]["false"] += 1

        for tw in tweets:

            user = tw.get("user")

            # Case 1: normal tweet with user metadata
            if user and "id" in user and user["id"] is not None:
                uid = user["id"]

            # Case 2: missing user info (VERY COMMON in PHEME)
            else:
                # create stable pseudo-user from tweet id
                uid = f"unknown_user_{tw['id']}"

            event_stats[event]["users"].add(uid)

    # ----------------------------------------------------
    # Print Table (Paper Style)
    # ----------------------------------------------------
    header = (
        f"{'Event':<15}"
        f"{'#Posts':>10}"
        f"{'#Users':>10}"
        f"{'#Events':>10}"
        f"{'True':>8}"
        f"{'False':>8}"
        f"{'AvgPosts':>12}"
        f"{'MaxPosts':>12}"
        f"{'MinPosts':>12}"
    )

    print("\n" + header)
    print("-" * len(header))

    for event in sorted(event_stats.keys()):

        stats = event_stats[event]
        sizes = stats["event_sizes"]

        print(
            f"{event:<15}"
            f"{stats['posts']:>10}"
            f"{len(stats['users']):>10}"
            f"{stats['events']:>10}"
            f"{stats['true']:>8}"
            f"{stats['false']:>8}"
            f"{np.mean(sizes):>12.2f}"
            f"{np.max(sizes):>12}"
            f"{np.min(sizes):>12}"
        )


############################################################
# MAIN(google colab)
############################################################

if __name__ == "__main__":

    print("=" * 70)
    print("PHEME DATASET BUILDER — BINARY VERACITY")
    print("=" * 70)

    # Project root (works on Windows, Linux, Colab)
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    # Raw dataset location (relative path)
    raw_data_dir = PROJECT_ROOT / "pheme-rnr-dataset"

    # Output location
    output_dir = PROJECT_ROOT / "data" / "processed" / "pheme_dataset"

    print(f"\nRaw dataset path: {raw_data_dir}")
    print(f"Output path: {output_dir}")

    threads = build_pheme_dataset(raw_data_dir, output_dir)

    print("\n✅ DATASET READY FOR TRAINING")