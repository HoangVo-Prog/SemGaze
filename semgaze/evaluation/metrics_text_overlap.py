"""LLada-compatible text overlap metrics for one-reference semantic units.

BLEU-4 is computed corpus-wide with the COCO scorer's tiny/small smoothing.
ROUGE in LLada means ROUGE-L (LCS, beta=1.2). METEOR uses Java METEOR
1.5's SCORE/EVAL protocol; NLTK is deliberately not substituted.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import math
from pathlib import Path
import shutil
import subprocess


def _words(value):
    return (value or "").split()


def _lcs_length(a, b):
    if len(a) < len(b):
        a, b = b, a
    previous = [0] * (len(b) + 1)
    for word in a:
        current = [0]
        for j, other in enumerate(b, 1):
            current.append(previous[j - 1] + 1 if word == other
                           else max(previous[j], current[-1]))
        previous = current
    return previous[-1]


def score_rouge_l_corpus(units):
    """Mean of LLada/COCO ROUGE-L per-reference F scores, beta=1.2."""
    if not units:
        return None
    beta_sq = 1.2 ** 2
    scores = []
    for unit in units:
        if not unit.get("candidate"):
            scores.append(0.0)
            continue
        reference = unit["reference"].split(" ")
        candidate = unit["candidate"].split(" ")
        overlap = _lcs_length(reference, candidate)
        precision = overlap / len(candidate)
        recall = overlap / len(reference)
        scores.append(((1 + beta_sq) * precision * recall /
                       (recall + beta_sq * precision)) if overlap else 0.0)
    return sum(scores) / len(scores)


def _ngrams(words, n):
    return Counter(tuple(words[i:i + n]) for i in range(max(0, len(words) - n + 1)))


def score_bleu4_corpus(units):
    """LLada's COCO-style corpus BLEU-4 for a single reference per semantic unit.

    Missing predictions contribute zero candidate ngrams while the reference
    remains in the denominator and brevity penalty. This is not sentence BLEU.
    """
    if not units:
        return None
    correct, guess = [0] * 4, [0] * 4
    hypothesis_length = reference_length = 0
    for unit in units:
        candidate = _words(unit["candidate"])
        reference = _words(unit["reference"])
        hypothesis_length += len(candidate)
        reference_length += len(reference)
        for n in range(1, 5):
            hypothesis_counts = _ngrams(candidate, n)
            reference_counts = _ngrams(reference, n)
            guess[n - 1] += sum(hypothesis_counts.values())
            correct[n - 1] += sum(min(count, reference_counts[ngram])
                                  for ngram, count in hypothesis_counts.items())
    if hypothesis_length == 0:
        return 0.0
    bleu_product = 1.0
    for matches, attempts in zip(correct, guess):
        bleu_product *= (matches + 1e-15) / (attempts + 1e-9)
    brevity_penalty = math.exp(min(0.0, 1.0 - reference_length / hypothesis_length))
    return brevity_penalty * bleu_product ** 0.25


class Meteor15:
    """Reusable METEOR 1.5 process with LLada English normalization/options."""

    def __init__(self, *, jar_path=None, java_bin="java"):
        root = Path(__file__).resolve().parents[2]
        jar = Path(jar_path).expanduser() if jar_path is not None else (
            root / "third_party" / "llada_meteor" / "meteor-1.5.jar")
        if not jar.is_absolute():
            jar = root / jar
        jar = jar.expanduser().resolve()
        if not jar.is_file():
            raise FileNotFoundError(
                f"METEOR 1.5 JAR missing: {jar}; see README 'METEOR 1.5 setup'")
        binary = shutil.which(java_bin)
        if binary is None:
            raise RuntimeError(f"METEOR 1.5 requires Java on PATH: {java_bin}")
        self.jar_path = jar
        self.process = subprocess.Popen(
            [binary, "-Xmx2G", "-jar", str(jar), "-", "-", "-stdio",
             "-l", "en", "-norm", "-a", "data/paraphrase-en.gz"],
            cwd=str(jar.parent), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, encoding="utf-8", bufsize=1)
        self.provenance = {
            "implementation": "LLada_Meteor_1.5",
            "jar_sha256": hashlib.sha256(jar.read_bytes()).hexdigest(),
            "language": "en", "normalization": True,
            "paraphrase": "data/paraphrase-en.gz",
            "aggregation": "METEOR_1.5_corpus_final_score",
        }

    def _read_float(self):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(
                "METEOR 1.5 Java process exited before returning all scores "
                f"(returncode={self.process.poll()}); check JAR and Java version")
        try:
            value = float(line.strip())
        except ValueError as exc:
            raise RuntimeError(f"Invalid METEOR 1.5 output: {line[:200]!r}") from exc
        if not math.isfinite(value):
            raise RuntimeError("Non-finite METEOR 1.5 result")
        return value

    @staticmethod
    def _sanitize(value):
        return " ".join(str(value or "").replace("|||", "").split())

    def score_units(self, units):
        if not units:
            return None
        if self.process.poll() is not None:
            raise RuntimeError("METEOR 1.5 Java process is not running")
        stats = []
        for unit in units:
            reference = self._sanitize(unit["reference"])
            candidate = self._sanitize(unit["candidate"])
            self.process.stdin.write(f"SCORE ||| {reference} ||| {candidate}\n")
            self.process.stdin.flush()
            stat = self.process.stdout.readline().strip()
            if not stat:
                raise RuntimeError("METEOR 1.5 failed to return sentence statistics")
            stats.append(stat)
        self.process.stdin.write("EVAL" + "".join(f" ||| {stat}" for stat in stats) + "\n")
        self.process.stdin.flush()
        for _ in units:
            self._read_float()
        return self._read_float()

    def close(self):
        process = getattr(self, "process", None)
        if process is None:
            return
        self.process = None
        try:
            if process.stdin:
                process.stdin.close()
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        finally:
            if process.stdout:
                process.stdout.close()

    def __del__(self):
        try:
            self.close()
        except (OSError, ValueError, AttributeError):
            pass


def build_meteor_scorer(*, jar_path=None, java_bin="java"):
    scorer = Meteor15(jar_path=jar_path, java_bin=java_bin)
    return scorer, scorer.provenance
