"""Main orchestrator for Laboratory Work #4 (Retrieval-Augmented Generation system).

Usage:
    python main_lab4.py          full run: corpus -> preprocessing -> vector DB -> search -> RAG ->
                                 dialogue with history -> ranking -> quality evaluation -> plots
    python main_lab4.py chat     interactive chat with the RAG system (history is kept on disk)
    python main_lab4.py setup    download (install) the local models and check that they work

Console output is mirrored to artifacts/lab4/run_output.txt (full) and artifacts/lab4/setup_output.txt (setup).
"""

import logging
import os
import sys
import time
from typing import Sequence

# Must be set before the libraries are imported: silences tokenizer fork warnings and progress bars
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import pandas as pd

from src.config.lab4_config import EvaluatedSetup, Lab4Config, RetrievalConfig
from src.rag.corpus import WikipediaCorpusLoader
from src.rag.evaluation import LlmJudge, load_questions
from src.rag.experiments import AnswerQualityExperiment, RetrievalGridExperiment, summarize_setups
from src.rag.factory import RagSystemFactory
from src.rag.memory import ConversationMemory
from src.rag.model_setup import ModelInstaller, describe_hardware, smoke_test_models
from src.rag.pipeline import RagPipeline
from src.rag.plots import RagVisualizer
from src.rag.preprocessing import TextCleaner
from src.rag.reporting import (SummaryWriter, answers_table, cleaning_stats, corpus_table, format_answer,
                               index_table, wrap)
from src.rag.vector_store import RetrievedChunk
from src.utils.console import mirrored_stdout, print_banner, setup_logging

def base_setup(config: Lab4Config) -> EvaluatedSetup:
    """The reference configuration of the study (used for the demos and the chat)."""
    return next(setup for setup in config.setups if setup.is_base)


def print_ranked(results: Sequence[RetrievedChunk], score_label: str = "score") -> None:
    """Prints retrieved chunks with their relevance scores, best first."""
    for rank, item in enumerate(results, start=1):
        print(f"  [{rank}] {score_label}={item.score:.3f}  {item.chunk.chunk_id:<28} {item.chunk.text[:88]}...")


def run_full(config: Lab4Config) -> None:
    """All steps of the laboratory work in order; every table and plot is saved to artifacts/lab4/."""
    logger = logging.getLogger("Application")
    artifacts = config.paths.artifacts_dir
    demo = config.demo
    logger.info("Initializing Laboratory Work #4: Retrieval-Augmented Generation system")

    print_banner("Step 2: Corpus of documents (Ukrainian Wikipedia, AI topics)")
    documents = WikipediaCorpusLoader(config.corpus, config.corpus_dir).load()
    corpus_md = corpus_table(documents)
    print(corpus_md)

    print_banner("Step 3: Preprocessing (noise removal, chunking with overlap)")
    factory = RagSystemFactory(config, documents)
    cleaner = TextCleaner(config.corpus.stop_sections)
    cleaning = cleaning_stats(documents, cleaner)
    logger.info("Words before cleaning: %d; after cleaning: %d (%.1f%% removed as noise)", cleaning.raw_words,
                cleaning.clean_words, 100 * cleaning.removed_share)
    sample_raw = documents[0].text[:230].replace("\n", " ")
    print(f"\nRaw text sample:\n{wrap(sample_raw)}\nCleaned text sample:\n{wrap(cleaner.clean_text(sample_raw))}")
    chunks = factory.chunks(config.chunking)
    logger.info("Base chunking: %d words per chunk, %d words overlap -> %d chunks", config.chunking.chunk_size_words,
                config.chunking.overlap_words, len(chunks))
    print(f"\nChunk {chunks[1].chunk_id} | section: {chunks[1].section} | {chunks[1].word_count} words:\n"
          + wrap(chunks[1].text[:420] + " ..."))
    print(f"\nChunk {chunks[1].chunk_id} ends with:\n" + wrap("... " + chunks[1].text[-200:]))
    print(f"Overlap: chunk {chunks[2].chunk_id} starts with:\n" + wrap(chunks[2].text[:200] + " ..."))

    print_banner("Step 4 & 5: Embeddings and the vector database (ChromaDB)")
    for chunking in config.retrieval_grid_chunkings:
        factory.store(chunking)
    index_md = index_table(list(factory.index_stats.values()))
    print("\n" + index_md)

    print_banner("Step 6: Search (query -> top-k nearest chunks)")
    retriever = factory.retriever(config.chunking, RetrievalConfig(top_k=demo.shown_chunks))
    print(f"Query: {demo.search_question}")
    print_ranked(retriever.retrieve(demo.search_question), score_label="cosine")

    print_banner("Step 7 & 8: LLM and the RAG pipeline (query -> retrieval -> context -> answer)")
    llm = factory.llm()
    setup = base_setup(config)
    answer = factory.pipeline(setup, ConversationMemory(max_turns=0)).ask(demo.search_question)
    context = RagPipeline.format_context(answer.retrieved)
    logger.info("Context for the LLM: %d chunks, %d words, %d tokens", len(answer.retrieved), len(context.split()),
                llm.count_tokens(context))
    print(format_answer(answer))
    no_rag_setup = next(s for s in config.setups if not s.use_retrieval)
    without_rag = factory.pipeline(no_rag_setup, ConversationMemory(max_turns=0)).ask(demo.search_question)
    print("\nThe same question WITHOUT retrieval (LLM memory only):")
    print(format_answer(without_rag, show_chunks=False))

    print_banner("Step 9: Dialogue with stored history")
    history_path = artifacts / "dialogue_history.json"
    history_path.unlink(missing_ok=True)
    memory = ConversationMemory(config.memory.max_turns_in_prompt, storage_path=history_path)
    dialogue = factory.pipeline(setup, memory)
    for question in demo.dialogue:
        print(format_answer(dialogue.ask(question)))
    logger.info("Dialogue history: %d turns saved to %s", len(memory.turns), history_path)
    restored = ConversationMemory(config.memory.max_turns_in_prompt, storage_path=history_path)
    logger.info("History restored from disk: %d turns; the last question was: '%s'", len(restored.turns),
                restored.turns[-1].question)

    print_banner("Step 10: Ranking of results by relevance (cosine vs cross-encoder)")
    print(f"Query: {demo.ranking_question}")
    for label, use_reranker in (("Cosine similarity (bi-encoder)", False), ("Cross-encoder re-ranking", True)):
        ranked = factory.retriever(config.chunking, RetrievalConfig(top_k=demo.shown_chunks,
                                                                    use_reranker=use_reranker))
        print(f"\n  {label}:")
        print_ranked(ranked.retrieve(demo.ranking_question))

    print_banner("Step 11a: Evaluation - retrieval quality (no LLM)")
    questions = load_questions(config.questions_path)
    logger.info("Control questions: %d (%d answerable, %d without an answer in the corpus)", len(questions),
                sum(q.answerable for q in questions), sum(not q.answerable for q in questions))
    grid = RetrievalGridExperiment(config, factory).run(questions)
    grid_md = grid.round(3).to_markdown(index=False)
    print("\n" + grid_md)

    print_banner("Step 11b: Evaluation - answer quality for different system parameters")
    started = time.perf_counter()
    judge = LlmJudge(llm, config.models.judge_max_tokens)
    records = AnswerQualityExperiment(factory, judge).run(config.setups, questions)
    logger.info("Evaluated %d answers in %.1f min", len(records), (time.perf_counter() - started) / 60)
    summary = summarize_setups(records)
    summary_md = summary.round(2).to_markdown(index=False)
    print("\nSummary per setup (LLM-judge grades 1-5; key facts and evidence: share 0-1):\n" + summary_md)

    print_banner("Step 12: Plots and saved results")
    visualizer = RagVisualizer(config.paths.plots_dir)
    tokens = {f"{c.chunk_size_words} words": factory.chunk_tokens(c) for c in config.retrieval_grid_chunkings}
    plot_paths = [
        ("Chunk lengths vs embedder limit", visualizer.plot_chunk_lengths(tokens, factory.embedder.max_tokens)),
        ("Retrieval quality grid", visualizer.plot_retrieval_grid(grid)),
        ("Answer quality per setup", visualizer.plot_answer_quality(summary)),
        ("Key-fact coverage per question", visualizer.plot_question_heatmap(records)),
        ("Latency per setup", visualizer.plot_latency(summary)),
    ]
    for title, path in plot_paths:
        logger.info("Saved plot '%s' to %s", title, path)
    summary_path = SummaryWriter(config).write(records, [
        ("Corpus", corpus_md), ("Indexes", index_md), ("Retrieval Quality (answerable questions)", grid_md),
        ("Answer Quality per Setup", summary_md),
        (f"Answers of Setup '{setup.name}'", answers_table(records, questions, setup.name))], plot_paths)
    logger.info("Summary saved to %s", summary_path)
    logger.info("All tasks completed successfully!")


def run_setup(config: Lab4Config) -> None:
    """Installs every model used by the system and runs a smoke test."""
    logger = logging.getLogger("Setup")
    print_banner("Hardware")
    for line in describe_hardware():
        print(line)
    print_banner("Downloading models from Hugging Face Hub")
    models = config.models
    installer = ModelInstaller()
    installed = [installer.install(role, model_id) for role, model_id in (
        ("embeddings", models.embedding_model), ("re-ranker", models.reranker_model),
        ("LLM (main)", models.llm_model), ("LLM (alt.)", models.alternative_llm_model))]
    print("\n" + pd.DataFrame([{"Role": m.role, "Model": m.model_id, "Size, GB": round(m.size_gb, 2),
                                "Weight files": m.weight_files, "Seconds": round(m.seconds, 1)}
                               for m in installed]).to_markdown(index=False))
    logger.info("Total size on disk: %.1f GB", sum(m.size_gb for m in installed))
    print_banner("Smoke test")
    for line in smoke_test_models(models.embedding_model, models.llm_model,
                                  ("Що таке штучний інтелект?", "Одним реченням: що таке нейронна мережа?")):
        print(line)
    logger.info("Setup completed successfully!")


def run_chat(config: Lab4Config) -> None:
    """Interactive console chat. Commands: /history, /clear, /exit."""
    documents = WikipediaCorpusLoader(config.corpus, config.corpus_dir).load()
    factory = RagSystemFactory(config, documents)
    memory = ConversationMemory(config.memory.max_turns_in_prompt,
                                storage_path=config.paths.artifacts_dir / "chat_history.json")
    pipeline = factory.pipeline(base_setup(config), memory)
    print(f"\nRAG chat is ready ({len(memory.turns)} turns restored). Commands: /history, /clear, /exit\n")
    while True:
        try:
            question = input("Ви> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not sys.stdin.isatty():  # input piped from a file: echo it so the transcript is readable
            print(question)
        if question == "/exit":
            break
        if question == "/clear":
            memory.clear()
            print("History cleared.")
        elif question == "/history":
            for number, turn in enumerate(memory.turns, start=1):
                print(f"{number}. Q: {turn.question}\n   A: {turn.answer}\n   sources: {', '.join(turn.sources)}")
        elif question:
            print(format_answer(pipeline.ask(question)))


NOISY_LOGGERS = ("httpx", "httpx2", "httpcore", "urllib3", "chromadb", "sentence_transformers", "huggingface_hub")


def silence_library_loggers() -> None:
    """Hides per-request HTTP and library messages so the run log shows only the lab steps."""
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.ERROR)


def main() -> None:
    """Entry point: python main_lab4.py [full|chat|setup]."""
    config = Lab4Config()
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    if mode == "chat":
        logging.basicConfig(level=logging.WARNING)
        run_chat(config)
    elif mode == "setup":
        with mirrored_stdout(config.paths.artifacts_dir / "setup_output.txt"):
            setup_logging()
            silence_library_loggers()
            run_setup(config)
    elif mode == "full":
        with mirrored_stdout(config.paths.artifacts_dir / "run_output.txt"):
            setup_logging()
            silence_library_loggers()
            run_full(config)
    else:
        sys.exit(f"Unknown mode '{mode}', expected: full | chat | setup")


if __name__ == "__main__":
    main()
