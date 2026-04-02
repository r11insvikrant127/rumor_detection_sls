import random
from datetime import datetime
import numpy as np


class PPCPreprocessor:
    def __init__(self, max_len=40):
        self.max_len = max_len  # paper uses ~30–40

    # --------------------------------------------
    # TIME PARSER (FIXED FOR TWITTER FORMAT)
    # --------------------------------------------
    def _parse_time(self, t):
        if t is None:
            return None
        try:
            return datetime.strptime(t, "%a %b %d %H:%M:%S %z %Y")
        except Exception:
            return None

    # --------------------------------------------
    # USER FEATURE EXTRACTION (8 FEATURES)
    # --------------------------------------------
    def _extract_user_features(self, tweet, source_time):
        user = tweet.get("user", {})

        tweet_time = self._parse_time(tweet.get("created_at"))
        user_time = self._parse_time(user.get("created_at"))

        # registration age (in days)
        if tweet_time and user_time:
            reg_age = (tweet_time - user_time).total_seconds() / 86400.0
        else:
            reg_age = 0.0

        return [
            len(user.get("description") or ""),          # 1
            len(user.get("screen_name") or ""),          # 2
            user.get("followers_count", 0),              # 3
            user.get("friends_count", 0),                # 4
            user.get("statuses_count", 0),               # 5
            reg_age,                                     # 6
            int(user.get("verified", False)),            # 7
            int(user.get("geo_enabled", False)),         # 8
        ]

    # --------------------------------------------
    # MAIN FUNCTION
    # --------------------------------------------
    def process_thread(self, data):
        tweets = data.get("tweets", [])

        # -------- Step 1: parse + filter --------
        valid_tweets = []
        for t in tweets:
            t_time = self._parse_time(t.get("created_at"))
            if t_time is not None:
                valid_tweets.append((t, t_time))
         # 🔥 DEBUG PRINT HERE
        thread_id = data.get("thread_id", "unknown")
        if len(valid_tweets) != len(tweets):
            print(f"Mismatch in {thread_id}: {len(tweets)} -> {len(valid_tweets)}")
        if len(valid_tweets) == 0:
            print("All timestamps failed:", tweets[0].get("created_at"))
            return None

        # -------- Step 2: sort by time --------
        valid_tweets.sort(key=lambda x: x[1])

        # -------- Step 3: get source time --------
        source_time = valid_tweets[0][1]

        # -------- Step 4: build sequence --------
        sequence = []
        for tweet, _ in valid_tweets:
            features = self._extract_user_features(tweet, source_time)
            sequence.append(features)

        # -------- Step 5: fix length --------
        if len(sequence) > self.max_len:
            sequence = sequence[:self.max_len]
        else:
            while len(sequence) < self.max_len:
                sequence.append(random.choice(sequence))

        return np.array(sequence, dtype=np.float32)