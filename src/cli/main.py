#!/usr/bin/env python3
"""
Command-line interface for claude-session-mcp.

Provides commands to archive and restore Claude Code sessions.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import traceback
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, TypeGuard

import httpx
import typer

from src.cli.logger import CLILogger
from src.exceptions import ClaudeSessionError
from src.launcher import launch_claude_with_session
from src.services.archive import SessionArchiveService
from src.services.claude_process import auto_detect_session_id
from src.services.clone import SessionCloneService
from src.services.delete import SessionDeleteService
from src.services.discovery import SessionDiscoveryService
from src.services.info import SessionInfoService
from src.services.lineage import LineageService
from src.services.parser import SessionParserService
from src.services.restore import SessionRestoreService
from src.storage.gist import GistStorage
from src.storage.local import LocalFileSystemStorage

app = typer.Typer(
    name='claude-session',
    help='Archive and restore Claude Code sessions',
    add_completion=False,
)

# Type aliases and validators
ArchiveFormat = Literal['json', 'zst']


def _is_archive_format(value: str) -> TypeGuard[ArchiveFormat]:
    """Type guard for valid archive formats."""
    return value in ('json', 'zst')


def _validate_archive_format(value: str | None) -> ArchiveFormat | None:
    """Validate and narrow archive format for typer callback."""
    if value is None:
        return None
    if _is_archive_format(value):
        return value
    raise typer.BadParameter("Must be 'json' or 'zst'")


def _resolve_session_id(session_id: str | None) -> str:
    """Resolve session_id, auto-detecting from Claude Code if not provided."""
    if session_id is not None:
        return session_id
    detected = auto_detect_session_id()
    if detected is None:
        typer.secho('Error: Not running inside Claude Code.', fg=typer.colors.RED, err=True)
        typer.echo('Provide a session ID: claude-session <command> <session-id>', err=True)
        raise typer.Exit(1)
    return detected


@app.command()
def archive(
    session_id: str = typer.Argument(..., help='Session ID to archive'),
    output: str = typer.Argument(..., help='Output path or Gist URL (gist://<gist-id> or gist:// or file path)'),
    format: ArchiveFormat | None = typer.Option(
        None, '--format', '-f', help='Archive format: json or zst', callback=_validate_archive_format
    ),
    gist_token: str | None = typer.Option(None, '--gist-token', help='GitHub token (or use GITHUB_TOKEN env)'),
    gist_visibility: Literal['public', 'secret'] = typer.Option(
        'secret', '--gist-visibility', help='Gist visibility (public or secret)'
    ),
    gist_description: str = typer.Option('Claude Code Session Archive', '--gist-description', help='Gist description'),
    verbose: bool = typer.Option(False, '--verbose', '-v', help='Verbose output'),
) -> None:
    """Archive a Claude Code session to local file or GitHub Gist."""
    asyncio.run(_archive_async(session_id, output, format, gist_token, gist_visibility, gist_description, verbose))


@app.command(context_settings={'allow_extra_args': True, 'ignore_unknown_options': True})
def restore(
    ctx: typer.Context,
    archive: str = typer.Argument(..., help='Archive path or Gist URL (gist://<gist-id> or file path)'),
    project: Path | None = typer.Option(None, '--project', '-p', help='Target project directory (default: current)'),
    no_translate: bool = typer.Option(False, '--no-translate', help="Don't translate file paths"),
    in_place: bool = typer.Option(False, '--in-place', help='Restore with original session ID (verbatim restore)'),
    launch: bool = typer.Option(False, '--launch', '-l', help='Launch Claude Code after restore'),
    gist_token: str | None = typer.Option(None, '--gist-token', help='GitHub token for private gists'),
    verbose: bool = typer.Option(False, '--verbose', '-v', help='Verbose output'),
) -> None:
    """Restore a Claude Code session from local file or GitHub Gist.

    By default, restore creates a new session ID (UUIDv7) for the restored session.
    Use --in-place to restore with the original session ID for verbatim restoration.

    Extra arguments after -- are passed to claude CLI:

        claude-session restore ARCHIVE --launch -- --chrome
    """
    asyncio.run(_restore_async(archive, project, not no_translate, in_place, launch, gist_token, verbose, ctx.args))


async def _archive_async(
    session_id: str,
    output: str,
    format: Literal['json', 'zst'] | None,
    gist_token: str | None,
    gist_visibility: Literal['public', 'secret'],
    gist_description: str,
    verbose: bool,
) -> None:
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

        await logger.info(f'Found session in folder: {session_info.session_folder}')

        # Parse output parameter - check if it's a Gist URL
        use_gist = output.startswith('gist://')
        gist_id = None

        if use_gist:
            # Extract gist ID if provided: gist://abc123 or just gist://
            gist_id = output[7:] if len(output) > 7 else None

            # Get token from CLI or environment
            token = gist_token or os.environ.get('GITHUB_TOKEN')
            if not token:
                typer.secho('Error: GitHub token required for Gist storage.', fg=typer.colors.RED, err=True)
                typer.echo('Provide via --gist-token or set GITHUB_TOKEN environment variable.')
                typer.echo()
                typer.echo('To create a token:')
                typer.echo('  1. Go to https://github.com/settings/tokens')
                typer.echo('  2. Generate new token (classic)')
                typer.echo("  3. Select 'gist' scope")
                typer.echo('  4. Copy the token')
                raise typer.Exit(1)

        # Use the resolved full session ID (supports prefix matching)
        resolved_session_id = session_info.session_id

        # Initialize services
        with tempfile.TemporaryDirectory(prefix='claude-session-') as temp_dir:
            parser_service = SessionParserService()
            archive_service = SessionArchiveService(
                session_id=resolved_session_id,
                temp_dir=Path(temp_dir),
                parser_service=parser_service,
                session_folder=session_info.session_folder,  # Use folder directly from discovery
            )

            # Create storage backend
            storage: GistStorage | LocalFileSystemStorage
            if use_gist:
                # Generate filename for Gist with correct extension
                timestamp = datetime.now(UTC).strftime('%Y%m%d_%H%M%S')
                ext = '.json.zst' if format == 'zst' else '.json'
                filename = f'session-{resolved_session_id[:8]}-{timestamp}{ext}'

                if token is None:
                    raise typer.BadParameter('GitHub token required for Gist storage')
                storage = GistStorage(
                    token=token,
                    gist_id=gist_id,
                    visibility=gist_visibility,
                    description=gist_description,
                )
                output_path = None  # Don't pass to archive service
                await logger.info(f'Creating Gist archive: {filename}')
            else:
                # Local filesystem
                output_file = Path(output)
                storage = LocalFileSystemStorage(output_file.parent.resolve())
                output_path = str(output)
                filename = output_file.name
                await logger.info(f'Creating archive: {output}')

            # Create archive
            metadata = await archive_service.create_archive(
                storage=storage, output_path=output_path, format_param=format, logger=logger
            )

            # Print success
            if use_gist and isinstance(storage, GistStorage):
                typer.secho('✓ Archive uploaded to GitHub Gist!', fg=typer.colors.GREEN)
                typer.echo(f'  URL: {metadata.file_path}')
                typer.echo(f'  Gist ID: {storage.gist_id}')
                typer.echo(f'  Format: {metadata.format}')
                typer.echo(f'  Size: {metadata.size_mb} MB')
                typer.echo(f'  Records: {metadata.session_records:,} session, {metadata.agent_records:,} agent')
                typer.echo()
                typer.echo('To restore, use:')
                typer.secho(f'  claude-session restore gist://{storage.gist_id}', fg=typer.colors.CYAN)
            else:
                typer.secho('✓ Archive created successfully!', fg=typer.colors.GREEN)
                typer.echo(f'  Path: {metadata.file_path}')
                typer.echo(f'  Format: {metadata.format}')
                typer.echo(f'  Size: {metadata.size_mb} MB')
                typer.echo(f'  Records: {metadata.session_records:,} session, {metadata.agent_records:,} agent')
                typer.echo(f'  Files: {metadata.file_count}')
                if verbose:
                    typer.echo('\n  File breakdown:')
                    for file_meta in metadata.files:
                        typer.echo(f'    - {file_meta.filename}: {file_meta.record_count} records')

    except (ClaudeSessionError, FileNotFoundError, FileExistsError) as e:
        typer.secho(f'Error: {e}', fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    except Exception as e:
        await logger.error(f'Failed to create archive: {e}')
        if verbose:
            traceback.print_exc()
        raise typer.Exit(1)


async def _restore_async(
    archive: str,
    project: Path | None,
    translate_paths: bool,
    in_place: bool,
    launch: bool,
    gist_token: str | None,
    verbose: bool,
    extra_args: Sequence[str],
) -> None:
    """Async implementation of restore command."""
    logger = CLILogger(verbose=verbose)

    try:
        # Check if it's a Gist URL
        if archive.startswith('gist://'):
            gist_id = archive[7:]

            if not gist_id:
                typer.secho('Error: Gist ID required in format gist://<gist-id>', fg=typer.colors.RED, err=True)
                raise typer.Exit(1)

            # Get token (public gists don't need auth for reading, but use it if provided)
            token = gist_token or os.environ.get('GITHUB_TOKEN') or ''

            await logger.info(f'Downloading from Gist: {gist_id}')

            # Create Gist storage
            storage = GistStorage(token=token, gist_id=gist_id)

            # Find the archive file in the gist (look for .json or .json.zst)
            async with httpx.AsyncClient() as client:
                headers = {'Accept': 'application/vnd.github.v3+json', 'X-GitHub-Api-Version': '2022-11-28'}
                if token:
                    headers['Authorization'] = f'Bearer {token}'

                response = await client.get(f'https://api.github.com/gists/{gist_id}', headers=headers)

                if response.status_code == 404:
                    typer.secho(f'Error: Gist not found: {gist_id}', fg=typer.colors.RED, err=True)
                    typer.echo('Check the gist ID and ensure it exists.')
                    raise typer.Exit(1)

                response.raise_for_status()
                gist_data = response.json()

                # Find archive file (including base64-encoded variants)
                files = gist_data['files']
                archive_file = None
                for filename in files:
                    # Strip .b64 suffix for format detection
                    base_name = filename[:-4] if filename.endswith('.b64') else filename
                    if base_name.endswith('.json') or base_name.endswith('.json.zst'):
                        archive_file = filename
                        break

                if not archive_file:
                    typer.secho(f'Error: No archive file found in gist {gist_id}', fg=typer.colors.RED, err=True)
                    typer.echo(f'Available files: {", ".join(files.keys())}')
                    raise typer.Exit(1)

            # Download to temp file
            await logger.info(f'Downloading {archive_file}...')
            data = await storage.load(archive_file)

            # Use logical filename (without .b64) so restore service detects format correctly
            logical_filename = archive_file[:-4] if archive_file.endswith('.b64') else archive_file
            with tempfile.NamedTemporaryFile(suffix=f'-{logical_filename}', delete=False) as temp_file:
                temp_file.write(data)
                archive_path = Path(temp_file.name)

            await logger.info(f'Downloaded {len(data):,} bytes')

        else:
            # Local file
            archive_path = Path(archive)

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
        if in_place:
            await logger.info('In-place mode: restoring with original session ID')
        result = await restore_service.restore_archive(
            archive_path=str(archive_path),
            translate_paths=translate_paths,
            in_place=in_place,
            logger=logger,
        )

        # Clean up temp file if we downloaded from Gist
        if archive.startswith('gist://'):
            archive_path.unlink()

        # Print success
        typer.secho('✓ Session restored successfully!', fg=typer.colors.GREEN)
        typer.echo(f'  New session ID: {result.new_session_id}')
        typer.echo(f'  Original session ID: {result.original_session_id}')
        typer.echo(f'  Project: {result.project_path}')
        typer.echo(f'  Mode: {"in-place" if result.was_in_place else "fork"}')
        typer.echo()
        typer.echo('  Files restored:')
        typer.echo(f'    - Session: 1 main + {len(result.agent_files)} agents')
        if result.plan_files_restored:
            typer.echo(f'    - Plans: {result.plan_files_restored}')
        if result.tool_results_restored:
            typer.echo(f'    - Tool results: {result.tool_results_restored}')
        if result.todos_restored:
            typer.echo(f'    - Todos: {result.todos_restored}')
        if result.tasks_restored:
            typer.echo(f'    - Tasks: {result.tasks_restored}')
        typer.echo()
        typer.echo(f'  Records: {result.main_records_restored:,} main + {result.agent_records_restored:,} agent')
        typer.echo(f'  Paths translated: {result.paths_translated}')

        if launch:
            typer.echo()
            typer.echo('Launching Claude Code...')
            launch_claude_with_session(result.new_session_id, extra_args=extra_args)
            # Note: launch_claude_with_session uses execvp, so we never reach here
        else:
            typer.echo()
            typer.echo('To continue this session, run:')
            typer.secho(f'  claude --resume {result.new_session_id}', fg=typer.colors.CYAN)

    except Exception as e:
        await logger.error(f'Failed to restore session: {e}')
        if verbose:
            traceback.print_exc()
        raise typer.Exit(1)


@app.command(context_settings={'allow_extra_args': True, 'ignore_unknown_options': True})
def clone(
    ctx: typer.Context,
    session_id: str = typer.Argument(..., help='Session ID to clone (full UUID or prefix)'),
    project: Path | None = typer.Option(None, '--project', '-p', help='Target project directory (default: current)'),
    no_translate: bool = typer.Option(False, '--no-translate', help="Don't translate file paths"),
    launch: bool = typer.Option(False, '--launch', '-l', help='Launch Claude Code after clone'),
    verbose: bool = typer.Option(False, '--verbose', '-v', help='Verbose output'),
) -> None:
    """Clone a session directly (no archive file needed).

    Extra arguments after -- are passed to claude CLI:

        claude-session clone SESSION_ID --launch -- --chrome
    """
    asyncio.run(_clone_async(session_id, project, not no_translate, launch, verbose, ctx.args))


async def _clone_async(
    session_id: str,
    project: Path | None,
    translate_paths: bool,
    launch: bool,
    verbose: bool,
    extra_args: Sequence[str],
) -> None:
    """Async implementation of clone command."""
    logger = CLILogger(verbose=verbose)

    try:
        # Determine project path
        if project:
            project_path = project.resolve()
            if not project_path.exists():
                typer.secho(f'Error: Project directory does not exist: {project_path}', fg=typer.colors.RED, err=True)
                raise typer.Exit(1)
        else:
            project_path = Path.cwd()

        await logger.info(f'Cloning to project: {project_path}')

        # Initialize clone service
        clone_service = SessionCloneService(project_path)

        # Clone the session
        result = await clone_service.clone(
            source_session_id=session_id,
            translate_paths=translate_paths,
            logger=logger,
        )

        # Print success
        typer.secho('✓ Session cloned successfully!', fg=typer.colors.GREEN)
        typer.echo(f'  New session ID: {result.new_session_id}')
        typer.echo(f'  Original session ID: {result.original_session_id}')
        typer.echo(f'  Project: {result.project_path}')
        typer.echo()
        typer.echo('  Files cloned:')
        typer.echo(f'    - Session: 1 main + {len(result.agent_files)} agents')
        if result.plan_files_restored:
            typer.echo(f'    - Plans: {result.plan_files_restored}')
        if result.tool_results_restored:
            typer.echo(f'    - Tool results: {result.tool_results_restored}')
        if result.todos_restored:
            typer.echo(f'    - Todos: {result.todos_restored}')
        typer.echo()
        typer.echo(f'  Records: {result.main_records_restored:,} main + {result.agent_records_restored:,} agent')
        typer.echo(f'  Paths translated: {result.paths_translated}')

        if launch:
            typer.echo()
            typer.echo('Launching Claude Code...')
            launch_claude_with_session(result.new_session_id, extra_args=extra_args)
            # Note: launch_claude_with_session uses execvp, so we never reach here
        else:
            typer.echo()
            typer.echo('To continue this session, run:')
            typer.secho(f'  claude --resume {result.new_session_id}', fg=typer.colors.CYAN)

    except (ClaudeSessionError, FileNotFoundError, FileExistsError) as e:
        typer.secho(f'Error: {e}', fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    except Exception as e:
        await logger.error(f'Failed to clone session: {e}')
        if verbose:
            traceback.print_exc()
        raise typer.Exit(1)


@app.command()
def delete(
    session_id: str | None = typer.Argument(
        None, help='Session ID (full or prefix). Auto-detected inside Claude Code.'
    ),
    force: bool = typer.Option(False, '--force', '-f', help='Required to delete native (UUIDv4) sessions'),
    terminate: bool = typer.Option(False, '--terminate', '-t', help='Terminate running Claude process before deletion'),
    no_backup: bool = typer.Option(False, '--no-backup', help="Don't keep a backup file for undo"),
    dry_run: bool = typer.Option(False, '--dry-run', help='Preview what would be deleted'),
    project: Path | None = typer.Option(None, '--project', '-p', help='Project directory (default: current)'),
    verbose: bool = typer.Option(False, '--verbose', '-v', help='Verbose output'),
) -> None:
    """Delete session artifacts with auto-backup.

    By default, only cloned/restored sessions (UUIDv7) can be deleted.
    Native Claude sessions (UUIDv4) require --force.

    If the session is currently running, use --terminate to kill the Claude
    process before deletion. This prevents the session file from being
    recreated when the process exits.

    A backup is saved to ~/.claude-session-mcp/deleted/ for undo capability.
    Use 'restore --in-place' on the backup to undo.
    """
    asyncio.run(_delete_async(_resolve_session_id(session_id), force, terminate, no_backup, dry_run, project, verbose))


async def _delete_async(
    session_id: str,
    force: bool,
    terminate: bool,
    no_backup: bool,
    dry_run: bool,
    project: Path | None,
    verbose: bool,
) -> None:
    """Async implementation of delete command."""
    logger = CLILogger(verbose=verbose)

    try:
        # Determine project path
        if project:
            project_path = project.resolve()
            if not project_path.exists():
                typer.secho(f'Error: Project directory does not exist: {project_path}', fg=typer.colors.RED, err=True)
                raise typer.Exit(1)
        else:
            project_path = Path.cwd()

        # Resolve session ID prefix to full ID
        info_service = SessionInfoService()
        session_info = await info_service.resolve_session(session_id)
        full_session_id = session_info.session_id

        # Check if session is running
        is_running, running_pid = info_service.is_session_running(full_session_id)

        if is_running:
            assert running_pid is not None  # is_running=True guarantees this
            if dry_run:
                # Dry-run: show warning but continue
                typer.secho(f'Warning: Session is currently running (PID {running_pid})', fg=typer.colors.YELLOW)
            elif not terminate:
                # Running without --terminate: error
                typer.secho(
                    f'Error: Session {full_session_id[:12]}... is currently running (PID {running_pid}).',
                    fg=typer.colors.RED,
                    err=True,
                )
                typer.echo('Use --terminate to kill the process before deletion.', err=True)
                raise typer.Exit(1)
            else:
                # Running with --terminate: will terminate
                await logger.info(f'Session is running (PID {running_pid}), will terminate before deletion')

        # Initialize delete service
        delete_service = SessionDeleteService(project_path)

        # Pass PID to terminate if running and --terminate was specified
        terminate_pid = running_pid if is_running and terminate and not dry_run else None

        # Delete the session
        result = await delete_service.delete_session(
            session_id=full_session_id,
            force=force,
            no_backup=no_backup,
            dry_run=dry_run,
            logger=logger,
            terminate_pid_before_delete=terminate_pid,
        )

        if not result.success:
            typer.secho(f'Error: {result.error_message}', fg=typer.colors.RED, err=True)
            raise typer.Exit(1)

        if dry_run:
            typer.secho('Dry run - would delete:', fg=typer.colors.YELLOW)
            typer.echo(f'  Session ID: {result.session_id}')
            typer.echo(f'  Files: {result.files_deleted}')
            typer.echo(f'  Directories: {len(result.directories_removed)}')
            typer.echo(f'  Size: {result.size_freed_bytes:,} bytes')
            if verbose:
                typer.echo('\n  Files to delete:')
                for path in result.deleted_files:
                    typer.echo(f'    - {path}')
                if result.directories_removed:
                    typer.echo('\n  Directories to clean up:')
                    for path in result.directories_removed:
                        typer.echo(f'    - {path}')
        else:
            typer.secho('✓ Session deleted successfully!', fg=typer.colors.GREEN)
            typer.echo(f'  Session ID: {result.session_id}')
            typer.echo()
            typer.echo('  Files deleted:')
            typer.echo(f'    - Session: {result.session_files_deleted}')
            if result.plan_files_deleted:
                typer.echo(f'    - Plans: {result.plan_files_deleted}')
            if result.tool_results_deleted:
                typer.echo(f'    - Tool results: {result.tool_results_deleted}')
            if result.todos_deleted:
                typer.echo(f'    - Todos: {result.todos_deleted}')
            if result.tasks_deleted:
                typer.echo(f'    - Tasks: {result.tasks_deleted}')
            typer.echo()
            typer.echo(f'  Directories removed: {len(result.directories_removed)}')
            typer.echo(f'  Size freed: {result.size_freed_bytes:,} bytes')
            typer.echo(f'  Duration: {result.duration_ms:.0f}ms')
            if result.backup_path:
                typer.echo(f'  Backup: {result.backup_path}')
                typer.echo()
                typer.echo('To undo, run:')
                typer.secho(f'  claude-session restore --in-place {result.backup_path}', fg=typer.colors.CYAN)

    except (ClaudeSessionError, FileNotFoundError, FileExistsError) as e:
        typer.secho(f'Error: {e}', fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    except Exception as e:
        await logger.error(f'Failed to delete session: {e}')
        if verbose:
            traceback.print_exc()
        raise typer.Exit(1)


@app.command()
def lineage(
    session_id: str | None = typer.Argument(
        None, help='Session ID (full or prefix). Auto-detected inside Claude Code.'
    ),
    format: Literal['text', 'tree', 'json'] = typer.Option(
        'text', '--format', '-f', help='Output format: text, tree, or json'
    ),
) -> None:
    """Show the lineage (parent-child relationships) for a session.

    When run inside Claude Code without a session ID, auto-detects the current session.

    Examples:
        claude-session lineage
        claude-session lineage 019b53ff
        claude-session lineage c3bac5a6 --format tree
    """
    session_id = _resolve_session_id(session_id)
    try:
        lineage_service = LineageService()

        if format == 'text':
            entry = lineage_service.get_entry(session_id)
            if entry:
                typer.echo(f'Session: {entry.child_session_id}')
                typer.echo(f'Parent:  {entry.parent_session_id}')
                typer.echo(f'Cloned:  {entry.cloned_at}')
                typer.echo(f'Method:  {entry.method}')
                typer.echo(f'Source:  {entry.parent_project_path}')
                typer.echo(f'Target:  {entry.target_project_path}')
                typer.echo(f'Machine: {entry.target_machine_id}')
                if entry.parent_machine_id:
                    if entry.parent_machine_id != entry.target_machine_id:
                        typer.secho(
                            f'Source Machine: {entry.parent_machine_id} (cross-machine)', fg=typer.colors.YELLOW
                        )
                    else:
                        typer.echo(f'Source Machine: {entry.parent_machine_id} (same machine)')
                if entry.archive_path:
                    typer.echo(f'Archive: {entry.archive_path}')
                if entry.paths_translated:
                    typer.secho('Paths translated: yes', fg=typer.colors.CYAN)
            else:
                typer.secho(f'No lineage entry found for {session_id}', fg=typer.colors.YELLOW)
                typer.echo('(This is either a native session or lineage tracking was not enabled)')

        elif format == 'tree':
            ancestry = lineage_service.get_ancestry(session_id)
            if not ancestry:
                typer.secho(f'No lineage found for {session_id}', fg=typer.colors.YELLOW)
            else:
                for i, ancestor_id in enumerate(ancestry):
                    indent = '  ' * i
                    prefix = '└─ ' if i > 0 else ''
                    # Highlight the requested session
                    if ancestor_id == session_id or ancestor_id.startswith(session_id):
                        typer.secho(f'{indent}{prefix}{ancestor_id}', fg=typer.colors.GREEN)
                    else:
                        typer.echo(f'{indent}{prefix}{ancestor_id}')

        elif format == 'json':
            entry = lineage_service.get_entry(session_id)
            if entry:
                typer.echo(entry.model_dump_json(indent=2))
            else:
                typer.echo('null')

    except ClaudeSessionError as e:
        typer.secho(f'Error: {e}', fg=typer.colors.RED, err=True)
        raise typer.Exit(1)


@app.command()
def info(
    session_id: str | None = typer.Argument(
        None, help='Session ID (full or prefix). Auto-detected inside Claude Code.'
    ),
    format: Literal['text', 'json'] = typer.Option('text', '--format', '-f', help='Output format: text or json'),
) -> None:
    """Display comprehensive information about a session.

    Shows session context including ID, project path, file locations,
    origin (how it was created), state, and characteristics.

    When run inside Claude Code without a session ID, auto-detects the current session.

    Examples:
        claude-session info
        claude-session info 019b53ff
        claude-session info c3bac5a6 --format json
    """
    asyncio.run(_info_async(_resolve_session_id(session_id), format))


async def _info_async(
    session_id: str,
    format: Literal['text', 'json'],
) -> None:
    """Async implementation of info command."""
    info_service = SessionInfoService()
    try:
        context = await info_service.get_info(session_id)
    except (ClaudeSessionError, FileNotFoundError) as e:
        typer.secho(f'Error: {e}', fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    if format == 'json':
        typer.echo(context.model_dump_json(indent=2))
        return

    # Text format
    typer.echo(f'Session: {context.session_id}')
    if context.custom_title:
        typer.echo(f'Title: {context.custom_title}')
    typer.echo(f'Project: {context.project_path}')
    typer.echo()

    # Origin section
    typer.secho('Origin:', bold=True)
    source_display: str = context.source
    if context.parent_id:
        source_display += f' (parent: {context.parent_id[:12]}...)'
    typer.echo(f'  Source: {source_display}')
    typer.echo(f'  State: {context.state}')
    typer.echo(f'  Native: {"yes" if context.is_native else "no (cloned/restored)"}')
    if context.has_lineage:
        typer.secho('  Lineage: tracked', fg=typer.colors.CYAN)
    typer.echo()

    # Temporal section - Authoritative timestamps
    typer.secho('Timestamps:', bold=True)
    if context.first_message_at:
        typer.echo(f'  First Message: {context.first_message_at}')
    if context.process_created_at:
        typer.echo(f'  Process Created: {context.process_created_at}')
    if context.session_ended_at:
        reason_suffix = f' ({context.session_end_reason})' if context.session_end_reason else ''
        typer.echo(f'  Session Ended: {context.session_ended_at}{reason_suffix}')
    if context.crash_detected_at:
        typer.secho(f'  Crash Detected: {context.crash_detected_at}', fg=typer.colors.RED)
    if context.cloned_at:
        typer.secho(f'  Cloned: {context.cloned_at}', fg=typer.colors.CYAN)
    typer.echo()

    # Files section
    typer.secho('Files:', bold=True)
    typer.echo(f'  Session: {context.session_file}')
    typer.echo(f'  Debug: {context.debug_file}')

    # Environment section (if available)
    if context.machine_id or context.claude_pid or context.claude_version:
        typer.echo()
        typer.secho('Environment:', bold=True)
        if context.machine_id:
            typer.echo(f'  Machine: {context.machine_id}')
        if context.claude_pid:
            typer.echo(f'  Claude PID: {context.claude_pid}')
        if context.claude_version:
            typer.echo(f'  Claude Version: {context.claude_version}')
        if context.temp_dir:
            typer.echo(f'  Temp dir: {context.temp_dir}')


def main() -> None:
    """Entry point for CLI."""
    app()


if __name__ == '__main__':
    main()
