#!/usr/bin/env python3
"""
Command-line interface for claude-session-mcp.

Provides commands to archive and restore Claude Code sessions.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path
from typing import Optional

import typer

from src.cli_logger import CLILogger
from src.services.archive import SessionArchiveService
from src.services.discovery import SessionDiscoveryService
from src.services.parser import SessionParserService
from src.services.restore import SessionRestoreService
from src.storage.local import LocalFileSystemStorage

app = typer.Typer(
    name='claude-session',
    help='Archive and restore Claude Code sessions',
    add_completion=False,
)


@app.command()
def archive(
    session_id: str = typer.Argument(..., help='Session ID to archive'),
    output: Path = typer.Argument(..., help='Output file path (e.g., session.json or session.json.zst)'),
    format: Optional[str] = typer.Option(None, '--format', '-f', help='Archive format: json or zst'),
    verbose: bool = typer.Option(False, '--verbose', '-v', help='Verbose output'),
) -> None:
    """Archive a Claude Code session by ID."""
    asyncio.run(_archive_async(session_id, output, format, verbose))


@app.command()
def restore(
    archive_path: Path = typer.Argument(..., help='Path to archive file'),
    project: Optional[Path] = typer.Option(None, '--project', '-p', help='Target project directory (default: current)'),
    no_translate: bool = typer.Option(False, '--no-translate', help="Don't translate file paths"),
    verbose: bool = typer.Option(False, '--verbose', '-v', help='Verbose output'),
) -> None:
    """Restore a Claude Code session from an archive."""
    asyncio.run(_restore_async(archive_path, project, not no_translate, verbose))


async def _archive_async(session_id: str, output: Path, format: Optional[str], verbose: bool) -> None:
    """Async implementation of archive command."""
    logger = CLILogger(verbose=verbose)

    try:
        # Find the session
        discovery = SessionDiscoveryService()
        session_info = await discovery.find_session_by_id(session_id)

        if not session_info:
            typer.secho(f'Error: Session not found: {session_id}', fg=typer.colors.RED, err=True)
            typer.echo(f'Searched in: {discovery.claude_sessions_dir}', err=True)
            raise typer.Exit(1)

        await logger.info(f'Found session in project: {session_info.project_path}')

        # Initialize services
        with tempfile.TemporaryDirectory(prefix='claude-session-') as temp_dir:
            parser_service = SessionParserService()
            archive_service = SessionArchiveService(
                session_id=session_id,
                project_path=session_info.project_path,
                temp_dir=Path(temp_dir),
                parser_service=parser_service,
            )

            # Create storage backend
            storage = LocalFileSystemStorage(output.parent.resolve())

            # Create archive
            await logger.info(f'Creating archive: {output}')
            metadata = await archive_service.create_archive(
                storage=storage, output_path=str(output), format_param=format, logger=logger
            )

            # Print success
            typer.secho(f'✓ Archive created successfully!', fg=typer.colors.GREEN)
            typer.echo(f'  Path: {metadata.file_path}')
            typer.echo(f'  Format: {metadata.format}')
            typer.echo(f'  Size: {metadata.size_mb} MB')
            typer.echo(f'  Records: {metadata.record_count:,}')
            typer.echo(f'  Files: {metadata.file_count}')
            if verbose:
                typer.echo('\n  File breakdown:')
                for file_meta in metadata.files:
                    typer.echo(f'    - {file_meta.filename}: {file_meta.record_count} records')

    except FileExistsError as e:
        typer.secho(f'Error: {e}', fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    except Exception as e:
        await logger.error(f'Failed to create archive: {e}')
        if verbose:
            import traceback

            traceback.print_exc()
        raise typer.Exit(1)


async def _restore_async(
    archive_path: Path, project: Optional[Path], translate_paths: bool, verbose: bool
) -> None:
    """Async implementation of restore command."""
    logger = CLILogger(verbose=verbose)

    try:
        # Validate archive exists
        if not archive_path.exists():
            typer.secho(f'Error: Archive not found: {archive_path}', fg=typer.colors.RED, err=True)
            raise typer.Exit(1)

        # Determine project path
        if project:
            project_path = project.resolve()
            if not project_path.exists():
                typer.secho(f'Error: Project directory does not exist: {project_path}', fg=typer.colors.RED, err=True)
                raise typer.Exit(1)
        else:
            project_path = Path.cwd()

        await logger.info(f'Restoring to project: {project_path}')

        # Initialize restore service
        restore_service = SessionRestoreService(project_path)

        # Restore the archive
        await logger.info(f'Loading archive: {archive_path}')
        result = await restore_service.restore_archive(
            archive_path=str(archive_path), translate_paths=translate_paths, logger=logger
        )

        # Print success
        typer.secho(f'✓ Session restored successfully!', fg=typer.colors.GREEN)
        typer.echo(f'  New session ID: {result.new_session_id}')
        typer.echo(f'  Original session ID: {result.original_session_id}')
        typer.echo(f'  Project: {result.project_path}')
        typer.echo(f'  Files restored: {result.files_restored}')
        typer.echo(f'  Records restored: {result.records_restored:,}')
        typer.echo(f'  Paths translated: {result.paths_translated}')
        typer.echo()
        typer.echo(f'To continue this session, run:')
        typer.secho(f'  claude --resume {result.new_session_id}', fg=typer.colors.CYAN)

    except Exception as e:
        await logger.error(f'Failed to restore session: {e}')
        if verbose:
            import traceback

            traceback.print_exc()
        raise typer.Exit(1)


def main() -> None:
    """Entry point for CLI."""
    app()


if __name__ == '__main__':
    main()