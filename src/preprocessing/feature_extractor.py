import numpy as np
from typing import Dict
from textblob import TextBlob
from datetime import datetime
import re

from .tree_builder import TreeBuilder
from .kernel_subtree import KernelSubtreeExtractor


class FeatureExtractor:
    """
    PAPER-FAITHFUL SLS Feature Extractor
    Implements EXACT 31 features from Table I.
    """

    FEATURE_NAMES = [
        "total_tweets",
        "kernel_ratio",
        "leaf_to_total_ratio",
        "max_depth",
        "depth_to_kernel_ratio",
        "leaf_to_responsive_ratio",

        "influential_account_age",
        "influential_followers",
        "influential_posts",
        "influential_reposts_per_follower",
        "influential_favorites_per_follower",

        "kernel_profile_pic_ratio",
        "kernel_verified_ratio",
        "kernel_avg_account_age",
        "kernel_avg_friends",
        "kernel_avg_followers",
        "kernel_avg_posts",
        "kernel_avg_reposts",
        "kernel_avg_favorites",

        "influential_mentions_per_kernel",
        "influential_sentiment",

        "kernel_avg_text_length",
        "kernel_avg_sentiment",
        "kernel_enquiry_ratio",
        "kernel_hashtag_ratio",
        "kernel_question_ratio",
        "kernel_exclamation_ratio",
        "kernel_multiple_punct_ratio",
        "kernel_media_ratio",
        "kernel_url_ratio",
        "kernel_mention_ratio",
    ]

    PAPER_MODE = False

    # ---------------- REGEX ----------------
    @staticmethod
    def build_patterns(paper_mode):
        return [
            r"\bis\s+(that|this|it)\s+true\b",
            r"\bwh[a]*t[?!][?1]*" if paper_mode else r"\bwh[a]*t[?!]+",
            r"\b(real\?|really\s?\?|unconfirmed)\b",
            r"\b(rumor|debunk)\b",
            r"\b(that|this|it)\s+is\s+not\s+true\b",
        ]

    # ---------------- INIT ----------------
    def __init__(self):
        self.tree_builder = TreeBuilder()
        self.kernel_extractor = KernelSubtreeExtractor()

        self.multi_punct = re.compile(r"[?!]{2,}")

        self.compiled_patterns = [
            re.compile(p, re.IGNORECASE)
            for p in self.build_patterns(self.PAPER_MODE)
        ]

    # ---------------- ENQUIRY ----------------
    def _is_enquiry(self, text: str) -> int:
        text = text.lower()
        return int(any(p.search(text) for p in self.compiled_patterns))

    # ---------------- ACCOUNT AGE ----------------
    def _account_age_days(self, tweet, ref_time):
        user = tweet.get("user", {})
        user_time = user.get("created_at")

        if not user_time:
            return 0.0

        try:
            user_dt = datetime.strptime(user_time, "%a %b %d %H:%M:%S %z %Y")
            return float((ref_time - user_dt).days)
        except Exception:
            return 0.0

    def get_feature_names(self):
        return self.FEATURE_NAMES

    # ---------------- MAIN ----------------
    def extract_features(self, event_data: Dict):

        tweets = event_data["tweets"]
        source_id = event_data.get("source_id")

        # event time
        times = [
            datetime.strptime(t["created_at"], "%a %b %d %H:%M:%S %z %Y")
            for t in tweets if t.get("created_at")
        ]
        event_time = max(times) if times else datetime.now()

        graph = self.tree_builder.build_from_tweets(
            tweets, source_id=source_id
        )

        # ✅ VALID TREE ONLY
        valid_nodes = [
            n for n in graph.nodes()
            if graph.nodes[n].get("depth", -1) >= 0
        ]
        valid_nodes = [
            n for n in graph.nodes()
            if graph.nodes[n].get("depth", -1) >= 0
        ]

        valid_graph = graph.subgraph(valid_nodes).copy()

        max_node, kernel_nodes = \
            self.kernel_extractor.extract_kernel_subtree(valid_graph)

        features = []
        features += self._propagation_features(valid_graph, kernel_nodes)
        features += self._user_features(tweets, kernel_nodes, max_node, event_time)
        features += self._content_features(tweets, kernel_nodes, max_node)

        assert len(features) == 31
        return np.array(features, dtype=np.float32)

    # ---------------- PROPAGATION ----------------
    def _propagation_features(self, graph, kernel_nodes):

        total = graph.number_of_nodes()

        leaf_nodes = sum(
            1 for n in graph if graph.out_degree(n) == 0
        )

        responsive = sum(
            1 for n in graph if graph.out_degree(n) > 0
        )

        depths = [
            graph.nodes[n]["depth"]
            for n in graph
            if graph.nodes[n]["depth"] >= 0
        ]

        max_depth = max(depths) if depths else 0

        kernel_size = len(kernel_nodes)

        return [
            float(total),
            kernel_size / total if total else 0.0,
            leaf_nodes / total if total else 0.0,
            float(max_depth),
            max_depth / kernel_size if kernel_size else 0.0,
            leaf_nodes / responsive if responsive else 0.0,
        ]

    # ---------------- USER ----------------
    def _user_features(self, tweets, kernel_nodes, max_node, event_time):

        kernel_ids = set(map(str, kernel_nodes))
        kernel_tweets = [
            t for t in tweets if str(t["id"]) in kernel_ids
        ]

        influential = next(
            (t for t in tweets if str(t["id"]) == str(max_node)),
            tweets[0]
        )

        user = influential.get("user", {})

        followers = float(user.get("followers_count", 0))
        reposts = float(influential.get("retweet_count", 0))
        favorites = float(influential.get("favorite_count", 0))

        def mean(lst):
            return float(np.mean(lst)) if lst else 0.0

        return [
            self._account_age_days(influential, event_time),
            followers,
            float(user.get("statuses_count", 0)),
            reposts / max(followers, 1),
            favorites / max(followers, 1),

            mean([
                0 if t.get("user", {}).get("default_profile_image") else 1
                for t in kernel_tweets
            ]),
            mean([1 if t.get("user", {}).get("verified") else 0 for t in kernel_tweets]),
            mean([self._account_age_days(t, event_time) for t in kernel_tweets]),
            mean([t.get("user", {}).get("friends_count", 0) for t in kernel_tweets]),
            mean([t.get("user", {}).get("followers_count", 0) for t in kernel_tweets]),
            mean([t.get("user", {}).get("statuses_count", 0) for t in kernel_tweets]),
            mean([t.get("retweet_count", 0) for t in kernel_tweets]),
            mean([t.get("favorite_count", 0) for t in kernel_tweets]),
        ]

    # ---------------- CONTENT ----------------
    def _content_features(self, tweets, kernel_nodes, max_node):

        kernel_ids = set(map(str, kernel_nodes))
        kernel_tweets = [
            t for t in tweets if str(t["id"]) in kernel_ids
        ]

        influential = next(
            (t for t in tweets if str(t["id"]) == str(max_node)),
            tweets[0]
        )

        def sentiment(text):
            s = TextBlob(text).sentiment.polarity
            return max(min(s, 1.0), -1.0)

        text = influential.get("text", "")
        ents = influential.get("entities", {})

        kernel_size = max(len(kernel_nodes), 1)

        def mean(lst):
            return float(np.mean(lst)) if lst else 0.0

        texts = [t.get("text", "").lower() for t in kernel_tweets]

        return [
            len(ents.get("user_mentions", [])) / kernel_size,
            sentiment(text),

            mean([len(t) for t in texts]),
            mean([sentiment(t) for t in texts]),
            mean([self._is_enquiry(t) for t in texts]),
            mean([bool(t.get("entities", {}).get("hashtags")) for t in kernel_tweets]),
            mean(["?" in t for t in texts]),
            mean(["!" in t for t in texts]),
            mean([bool(self.multi_punct.search(t)) for t in texts]),
            mean([bool(t.get("entities", {}).get("media")) for t in kernel_tweets]),
            mean([bool(t.get("entities", {}).get("urls")) for t in kernel_tweets]),
            mean([bool(t.get("entities", {}).get("user_mentions")) for t in kernel_tweets]),
        ]