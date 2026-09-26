"""Enterprise Command-Line Interface (CLI) for indexing and querying."""

from pathlib import Path
import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.core.config import get_settings
from src.core.logging import setup_logger
from src.embeddings.local_provider import LocalSentenceTransformerEmbeddings
from src.llm.sarvam_client import MockLLMClient, SarvamGLMClient
from src.models.query import QueryRequest
from src.retrieval.bm25_index import BM25Index
from src.retrieval.hybrid_retriever import HybridRetriever
from src.services.indexing_service import IndexingService
from src.services.rag_service import RAGService
from src.vectorstore.chroma_store import ChromaVectorStore

console = Console()
logger = setup_logger("rag_cli")


def _init_components():
    """Factory helper to wire dependencies."""
    settings = get_settings()
    vector_store = ChromaVectorStore(
        persist_dir=settings.chroma_persist_directory,
        collection_name=settings.chroma_collection_name,
    )
    embedding_provider = LocalSentenceTransformerEmbeddings(
        model_name=settings.embedding_model_name,
    )
    bm25_index = BM25Index()

    # If Chroma already has documents, populate the in-memory BM25 index from it
    if vector_store.count() > 0:
        # Reconstruct chunks from vector store for BM25
        # (By querying with a dummy vector to retrieve all chunks)
        dummy_vec = [0.0] * embedding_provider.dimension
        stored = vector_store.similarity_search_by_vector(dummy_vec, top_k=vector_store.count())
        bm25_index.index_chunks([chunk for chunk, _ in stored])

    retriever = HybridRetriever(
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        bm25_index=bm25_index,
        dense_weight=settings.dense_weight,
        sparse_weight=settings.sparse_weight,
        rrf_k=settings.rrf_k,
    )

    if settings.sarvam_api_key and not settings.sarvam_api_key.startswith("your_sarvam"):
        llm_client = SarvamGLMClient(
            api_key=settings.sarvam_api_key,
            base_url=settings.sarvam_base_url,
            model=settings.sarvam_model,
            temperature=settings.sarvam_temperature,
            max_tokens=settings.sarvam_max_tokens,
            timeout=settings.sarvam_timeout_seconds,
        )
    else:
        console.print(
            "[yellow]Note: SARVAM_API_KEY not set in .env. Running in Mock/Preview mode.[/yellow]"
        )
        llm_client = MockLLMClient()

    indexing_service = IndexingService(
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        bm25_index=bm25_index,
    )
    rag_service = RAGService(retriever=retriever, llm_client=llm_client)

    return indexing_service, rag_service, vector_store, settings


@click.group()
def cli():
    """Future-Ready Children - Enterprise RAG Chatbot CLI."""
    pass


@cli.command()
@click.option(
    "--file",
    "-f",
    default="data/raw/Future-Ready Children_ FAQ.docx",
    help="Path to source FAQ document (DOCX or PDF)",
)
@click.option(
    "--clear",
    is_flag=True,
    default=True,
    help="Clear existing index before indexing",
)
def index(file: str, clear: bool):
    """Index or re-index the campaign FAQ document."""
    file_path = Path(file)
    if not file_path.exists():
        console.print(f"[red]Error: File not found at '{file_path}'[/red]")
        return

    console.print(f"[bold cyan]Indexing document:[/bold cyan] {file_path}")
    indexing_service, _, vector_store, _ = _init_components()

    chunks = indexing_service.index_document(file_path=file_path, clear_existing=clear)
    console.print(
        Panel.fit(
            f"[green]✓ Indexing complete![/green]\n"
            f"Total Chunks: [bold]{len(chunks)}[/bold]\n"
            f"Vector Store Count: [bold]{vector_store.count()}[/bold]",
            title="Ingestion Status",
        )
    )


@cli.command()
@click.argument("question")
@click.option("--top-k", default=4, help="Number of retrieved chunks")
@click.option("--section", default=None, help="Filter by section title")
def ask(question: str, top_k: int, section: str | None):
    """Ask a single question and display grounded answer with citations."""
    _, rag_service, vector_store, _ = _init_components()

    if vector_store.count() == 0:
        console.print("[red]Vector store is empty! Please run `index` first.[/red]")
        return

    console.print(f"\n[bold]Question:[/bold] {question}\n")
    request = QueryRequest(query=question, top_k=top_k, filter_section=section)
    response = rag_service.answer_query(request)

    # Output Direct Answer
    console.print(
        Panel(
            response.answer,
            title="[bold green]Direct Answer[/bold green]",
            border_style="green",
        )
    )

    # Output Champion Talking Point
    if response.champion_talking_point:
        console.print(
            Panel(
                response.champion_talking_point,
                title="[bold blue]Champion Talking Point[/bold blue]",
                border_style="blue",
            )
        )

    # Output Citations Table
    if response.citations:
        table = Table(title="Sources & Citations", show_header=True)
        table.add_column("Chunk ID", style="cyan")
        table.add_column("Section", style="magenta")
        table.add_column("Question", style="yellow")
        table.add_column("Excerpt", style="white")

        for c in response.citations:
            table.add_row(
                c.chunk_id,
                c.section,
                f"Q{c.question_number}: {c.question_text}" if c.question_number else "Overview",
                c.excerpt[:90] + "...",
            )
        console.print(table)


@cli.command()
def chat():
    """Start an interactive chat session in the terminal."""
    _, rag_service, vector_store, _ = _init_components()

    if vector_store.count() == 0:
        console.print("[red]Vector store is empty! Please run `index` first.[/red]")
        return

    console.print(
        Panel(
            "[bold green]Future-Ready Children Campaign Advisor[/bold green]\n"
            "Ask any question about the campaign, pedagogy, costs, or champion roles.\n"
            "Type 'exit' or 'quit' to end.",
            title="Interactive Session",
        )
    )

    while True:
        try:
            query = console.input("\n[bold cyan]You > [/bold cyan]").strip()
            if not query:
                continue
            if query.lower() in {"exit", "quit", "q"}:
                console.print("[yellow]Exiting chat session. Goodbye![/yellow]")
                break

            response = rag_service.answer_query(QueryRequest(query=query))
            console.print(f"\n[bold green]Assistant ({response.model_used}):[/bold green]")
            console.print(response.answer)

            if response.champion_talking_point:
                console.print(f"\n[italic cyan]Talking Point for Champions:[/italic cyan] {response.champion_talking_point}")

            if response.citations:
                cite_str = ", ".join(
                    [
                        f"{c.section} (Q{c.question_number})"
                        for c in response.citations
                        if c.question_number
                    ]
                )
                if cite_str:
                    console.print(f"[dim]Sources: {cite_str}[/dim]")

        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Session ended.[/yellow]")
            break


if __name__ == "__main__":
    cli()
