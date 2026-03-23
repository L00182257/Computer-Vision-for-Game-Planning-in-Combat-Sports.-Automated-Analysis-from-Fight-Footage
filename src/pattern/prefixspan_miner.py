from collections import defaultdict


class PrefixSpanMiner:

    def __init__(self):
        pass

    def _build_frequent_items(self, sequences, min_support):
        counts = defaultdict(int)

        for seq in sequences:
            unique_items = set(seq)
            for item in unique_items:
                counts[item] += 1

        return {item: sup for item, sup in counts.items() if sup >= min_support}

    def _project(self, sequences, prefix):
        projected = []

        for seq in sequences:
            if not prefix:
                projected.append(seq)
                continue

            if len(seq) < len(prefix):
                continue

            if tuple(seq[: len(prefix)]) == tuple(prefix):
                projected.append(seq[len(prefix):])

        return projected

    def _prefixspan(self, prefix, sequences, min_support, max_len, patterns):

        if len(prefix) > 0:
            patterns[tuple(prefix)] = len(sequences)

        if max_len is not None and len(prefix) >= max_len:
            return

        frequent_items = self._build_frequent_items(sequences, min_support)

        for item, support in sorted(frequent_items.items(), key=lambda x: -x[1]):
            new_prefix = prefix + [item]
            projected = self._project(sequences, new_prefix)

            if len(projected) >= min_support:
                self._prefixspan(new_prefix, projected, min_support, max_len, patterns)

    def mine(self, sequences, min_support=2, max_pattern_len=5):
        # sequences: list of event sequences (list of strings) for each fighter or timeline
        patterns = {}

        if not sequences:
            return []

        self._prefixspan([], sequences, min_support, max_pattern_len, patterns)

        list_patterns = [
            {
                "pattern": list(pattern),
                "support": support
            }
            for pattern, support in patterns.items()
        ]

        list_patterns.sort(key=lambda x: x["support"], reverse=True)

        return list_patterns
