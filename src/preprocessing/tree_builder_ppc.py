import networkx as nx
from typing import List, Dict
from datetime import datetime


class TreeBuilderPPC:
    """
    PPC-SPECIFIC Propagation Builder

    Paper requirement:
    ------------------
    Each node must contain:
    - time (relative propagation time)
    - features (user feature vector)

    Output:
    -------
    Graph with:
        node["time"]
        node["features"]
    """

    def __init__(self):
        self.graph = nx.DiGraph()

    # --------------------------------------------------
    # UTILITIES
    # --------------------------------------------------
    def _normalize_id(self, tid) -> str:
        if tid is None:
            return ""
        return str(tid).strip()

    def _parse_time(self, t):
        if t is None:
            return None
        if isinstance(t, datetime):
            return t
        try:
            return datetime.fromisoformat(str(t).replace("Z", "+00:00"))
        except Exception:
            return None

    # --------------------------------------------------
    # USER FEATURE EXTRACTION (PPC CORE)
    # --------------------------------------------------
    def _extract_user_features(self, tweet):

        user = tweet.get("user", {})

        # -------- REGISTRATION AGE --------
        created_at = self._parse_time(tweet.get("created_at"))
        user_created = self._parse_time(user.get("created_at"))

        if created_at and user_created:
            reg_age = (created_at - user_created).total_seconds() / 86400.0
        else:
            reg_age = 0.0

        return [
            user.get("followers_count", 0),
            user.get("friends_count", 0),
            user.get("statuses_count", 0),
            reg_age,  
            int(user.get("verified", False)),
            len(user.get("description", "") or ""),
            len(user.get("screen_name", "") or ""),
            int(user.get("geo_enabled", False)),
        ]

    # --------------------------------------------------
    # MAIN BUILDER
    # --------------------------------------------------
    def build_from_tweets(
        self,
        tweets: List[Dict],
        source_id: str,
    ) -> nx.DiGraph:

        self.graph.clear()
        seen_ids = set()

        source_id = self._normalize_id(source_id)

        # ---------- FIND SOURCE TIME ----------
        source_time = None
        for tweet in tweets:
            tid = self._normalize_id(tweet.get("id", tweet.get("tweet_id")))
            if tid == source_id:
                source_time = self._parse_time(tweet.get("created_at"))
                break

        # fallback if missing
        if source_time is None:
            source_time = datetime.utcnow()

        # ---------- PASS 1: ADD NODES ----------
        for tweet in tweets:

            tweet_id = self._normalize_id(
                tweet.get("id", tweet.get("tweet_id"))
            )

            if not tweet_id or tweet_id in seen_ids:
                continue

            seen_ids.add(tweet_id)

            created_at = self._parse_time(tweet.get("created_at"))

            # -------- RELATIVE TIME --------
            if created_at:
                time_delta = (created_at - source_time).total_seconds()
            else:
                time_delta = 0.0

            # -------- FEATURES --------
            features = self._extract_user_features(tweet)

            self.graph.add_node(
                tweet_id,
                time=float(time_delta),     # 🔥 REQUIRED
                features=features,          # 🔥 REQUIRED
            )

        # ---------- PASS 2: ADD EDGES ----------
        for tweet in tweets:

            child_id = self._normalize_id(
                tweet.get("id", tweet.get("tweet_id"))
            )

            if child_id not in self.graph.nodes():
                continue

            parent_id = self._normalize_id(
                tweet.get("in_reply_to_status_id")
            )

            if (
                parent_id
                and parent_id in self.graph.nodes()
                and parent_id != child_id
            ):
                self.graph.add_edge(parent_id, child_id)

        return self.graph