import numpy as np
from typing import List, Dict
from textblob import TextBlob
import networkx as nx

from .tree_builder import TreeBuilder
from .kernel_subtree import KernelSubtreeExtractor


class FeatureExtractor:
    """
    Paper-faithful feature extractor for SLS model.
    Implements EXACT 31 features from Table I.
    """

    FEATURE_NAMES = [
        # Propagation (1–6)
        "total_tweets",
        "kernel_ratio",
        "leaf_to_total_ratio",
        "max_depth",
        "depth_to_kernel_ratio",
        "leaf_to_responsive_ratio",

        # Influential user (7,16–19)
        "influential_account_age",
        "influential_followers",
        "influential_posts",
        "influential_reposts_per_follower",
        "influential_favorites_per_follower",

        # Kernel user aggregation (8–15)
        "kernel_profile_pic_ratio",
        "kernel_verified_ratio",
        "kernel_avg_account_age",
        "kernel_avg_friends",
        "kernel_avg_followers",
        "kernel_avg_posts",
        "kernel_avg_reposts",
        "kernel_avg_favorites",

        # Influential content (20–21)
        "influential_mentions_per_kernel",
        "influential_sentiment",

        # Kernel content aggregation (22–31)
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

    # Paper-faithful enquiry indicators (NO '?')
    ENQUIRY_PHRASES = [
        "fake", "hoax", "rumor", "debunk",
        "fact check", "is this", "really", "actually"
    ]

    def __init__(self):
        self.tree_builder = TreeBuilder()
        self.kernel_extractor = KernelSubtreeExtractor()
    
    def get_feature_names(self):
        """Return ordered feature names (paper exact)."""
        return self.FEATURE_NAMES

    # --------------------------------------------------
    # MAIN ENTRY
    # --------------------------------------------------
    def extract_features(self, event_data: Dict) -> np.ndarray:

        tweets = event_data["tweets"]
        source_id = event_data.get("source_id")

        graph = self.tree_builder.build_from_tweets(
            tweets, source_id=source_id
        )

        max_node, kernel_nodes = \
            self.kernel_extractor.extract_kernel_subtree(graph)

        features = []
        features += self._propagation_features(graph, kernel_nodes)
        features += self._user_features(tweets, graph, kernel_nodes, max_node)
        features += self._content_features(tweets, graph, kernel_nodes, max_node)

        assert len(features) == 31
        return np.array(features, dtype=np.float32)

    # --------------------------------------------------
    # PROPAGATION FEATURES (1–6)
    # --------------------------------------------------
    def _propagation_features(self, graph, kernel_nodes):

        total = graph.number_of_nodes()
        leaf_nodes = sum(1 for n in graph if graph.out_degree(n) == 0)
        responsive = sum(1 for n in graph if graph.out_degree(n) > 0)

        max_depth = max(
            (graph.nodes[n].get("depth", 0) for n in graph),
            default=0,
        )

        kernel_size = len([n for n in kernel_nodes if n in graph])

        return [
            float(total),
            kernel_size / total if total else 0.0,
            leaf_nodes / total if total else 0.0,
            float(max_depth),
            max_depth / kernel_size if kernel_size else 0.0,
            leaf_nodes / responsive if responsive else 0.0,
        ]

    # --------------------------------------------------
    # USER FEATURES (7–19)
    # --------------------------------------------------
    def _user_features(self, tweets, graph, kernel_nodes, max_node):

        kernel_ids = {str(n) for n in kernel_nodes}
        kernel_tweets = [
            t for t in tweets if str(t.get("id")) in kernel_ids
        ]

        influential = next(
            t for t in tweets if str(t.get("id")) == str(max_node)
        )

        user = influential.get("user", {})

        followers = float(user.get("followers_count", 0))
        reposts = float(influential.get("retweet_count", 0))
        favorites = float(influential.get("favorite_count", 0))

        features = [
            float(user.get("account_age_days", 365)),  # dataset supplied
            followers,
            float(user.get("statuses_count", 0)),
            reposts / max(followers, 1),
            favorites / max(followers, 1),
        ]

        # kernel aggregation
        def mean(lst):
            return float(np.mean(lst)) if lst else 0.0

        features += [
            mean([0 if t["user"].get("default_profile_image") else 1 for t in kernel_tweets]),
            mean([1 if t["user"].get("verified") else 0 for t in kernel_tweets]),
            mean([t["user"].get("account_age_days", 365) for t in kernel_tweets]),
            mean([t["user"].get("friends_count", 0) for t in kernel_tweets]),
            mean([t["user"].get("followers_count", 0) for t in kernel_tweets]),
            mean([t["user"].get("statuses_count", 0) for t in kernel_tweets]),
            mean([t.get("retweet_count", 0) for t in kernel_tweets]),
            mean([t.get("favorite_count", 0) for t in kernel_tweets]),
        ]

        return features

    # --------------------------------------------------
    # CONTENT FEATURES (20–31)
    # --------------------------------------------------
    def _content_features(self, tweets, graph, kernel_nodes, max_node):

        kernel_ids = {str(n) for n in kernel_nodes}
        kernel_tweets = [
            t for t in tweets if str(t.get("id")) in kernel_ids
        ]

        influential = next(
            t for t in tweets if str(t.get("id")) == str(max_node)
        )

        text = influential.get("text", "").lower()
        ents = influential.get("entities", {})

        kernel_size = max(len(kernel_tweets), 1)

        features = [
            len(ents.get("user_mentions", [])) / kernel_size,
            TextBlob(text).sentiment.polarity,
        ]

        def mean(lst):
            return float(np.mean(lst)) if lst else 0.0

        texts = [t.get("text", "").lower() for t in kernel_tweets]

        features += [
            mean([len(t) for t in texts]),
            mean([TextBlob(t).sentiment.polarity for t in texts]),
            mean([any(p in t for p in self.ENQUIRY_PHRASES) for t in texts]),
            mean([bool(t.get("entities", {}).get("hashtags")) for t in kernel_tweets]),
            mean(["?" in t for t in texts]),
            mean(["!" in t for t in texts]),
            mean([t.count("?") > 1 or t.count("!") > 1 for t in texts]),
            mean([bool(t.get("entities", {}).get("media")) for t in kernel_tweets]),
            mean([bool(t.get("entities", {}).get("urls")) for t in kernel_tweets]),
            mean([bool(t.get("entities", {}).get("user_mentions")) for t in kernel_tweets]),
        ]

        return features