"""轻量 RAG：文档切分 + 纯 Python TF-IDF 向量检索 + 引用回填。

生产版可把 SimpleVectorStore 直接替换为 FAISS / Chroma + BGE embedding，
对外接口 search(query, top_k, threshold) 保持不变。
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from . import config

_WORD_RE = re.compile(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]")


@dataclass
class Document:
    doc_id: str
    title: str
    text: str
    source: str


def tokenize(text: str) -> list[str]:
    """中文按单字 + 相邻二元切分，英文按单词切分，无需第三方分词器。"""
    tokens: list[str] = []
    for match in _WORD_RE.finditer(text):
        word = match.group()
        if re.match(r"[a-zA-Z0-9]", word):
            tokens.append(word.lower())
        else:
            tokens.append(word)
    bigrams = [tokens[i] + tokens[i + 1] for i in range(len(tokens) - 1)
               if not tokens[i][0].isascii() and not tokens[i + 1][0].isascii()]
    return tokens + bigrams


def load_documents(docs_dir: Path | None = None) -> list[Document]:
    docs_dir = docs_dir or config.DOCS_DIR
    docs: list[Document] = []

    for path in sorted(docs_dir.glob("*.txt")):
        title = path.stem
        chunks = [c.strip() for c in path.read_text(encoding="utf-8").split("\n\n") if c.strip()]
        for i, chunk in enumerate(chunks):
            docs.append(Document(
                doc_id=f"{path.stem}-{i:03d}",
                title=_friendly_title(title),
                text=chunk,
                source=str(path.relative_to(docs_dir.parent.parent)),
            ))

    faq_path = docs_dir.parent / "faq.json"
    if faq_path.exists():
        for i, item in enumerate(json.loads(faq_path.read_text(encoding="utf-8"))):
            text = f"问题：{item['question']}\n答案：{item['answer']}"
            docs.append(Document(
                doc_id=f"faq-{i:03d}",
                title="教务与校园常见问答",
                text=text,
                source="data/faq.json",
            ))
    return docs


def _friendly_title(stem: str) -> str:
    mapping = {
        "python-course": "Python 程序设计",
        "ml-course": "机器学习与模式识别",
        "nlp-course": "自然语言处理",
        "db-course": "数据库原理及应用",
        "data-structure": "数据结构与算法",
        "bigdata-course": "大数据技术与云计算",
    }
    return mapping.get(stem, stem)


class SimpleVectorStore:
    """内存 TF-IDF 向量库：构建倒排索引，用余弦相似度检索。"""

    def __init__(self, documents: list[Document] | None = None):
        self.documents = documents if documents is not None else load_documents()
        self._df: Counter[str] = Counter()
        self._tf: list[Counter[str]] = []
        self._norms: list[float] = []
        self._build()

    def _build(self) -> None:
        for doc in self.documents:
            counts = Counter(tokenize(doc.title + " " + doc.text))
            self._tf.append(counts)
            self._df.update(counts.keys())

        n = len(self.documents)
        for counts in self._tf:
            vec = {}
            for term, tf in counts.items():
                idf = math.log((n + 1) / (self._df[term] + 1)) + 1
                vec[term] = (1 + math.log(tf)) * idf
            self._norms.append(math.sqrt(sum(v * v for v in vec.values())) or 1.0)
            # 把带 idf 的权重存回 counts，供检索复用
            counts.clear()
            counts.update(vec)

    def search(self, query: str, top_k: int | None = None,
               threshold: float | None = None) -> list[tuple[Document, float]]:
        top_k = config.TOP_K if top_k is None else top_k
        threshold = config.SIMILARITY_THRESHOLD if threshold is None else threshold

        q_counts = Counter(tokenize(query))
        q_vec = {}
        n = len(self.documents)
        for term, tf in q_counts.items():
            idf = math.log((n + 1) / (self._df[term] + 1)) + 1
            q_vec[term] = (1 + math.log(tf)) * idf
        q_norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0

        scored: list[tuple[Document, float]] = []
        for i, weights in enumerate(self._tf):
            score = 0.0
            for term, q_weight in q_vec.items():
                weight = weights.get(term)
                if weight:
                    score += q_weight * weight
            score /= (q_norm * self._norms[i])
            if score >= threshold:
                scored.append((self.documents[i], score))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    def format_context(self, hits: list[tuple[Document, float]]) -> tuple[str, list[dict]]:
        parts, citations = [], []
        for i, (doc, score) in enumerate(hits, 1):
            parts.append(f"[{i}] 《{doc.title}》：{doc.text}")
            citations.append({"index": i, "title": doc.title, "doc_id": doc.doc_id,
                              "score": round(score, 4), "source": doc.source})
        return "\n\n".join(parts), citations
