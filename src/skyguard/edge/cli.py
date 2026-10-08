"""Edge deployment CLI commands.

Adds `skyguard edge` subcommands for:
  * hardware  - probe and print Jetson hardware info
  * power     - query/set power mode
  * build     - build TensorRT engine from ONNX
  * deploy    - full deploy sequence
  * health    - one-shot health snapshot
  * watch     - continuous health monitoring
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from skyguard.core.logger import get_logger, setup_logging
from skyguard.edge.hardware import detect_jetson, print_hardware_info
from skyguard.edge.monitor import EdgeMonitor, print_health
from skyguard.edge.power import PowerManager, PowerMode, print_power_state
from skyguard.edge.trt_engine import (
    EngineBuildConfig,
    Precision,
    TRTEngineBuilder,
    print_builder_info,
)
from skyguard.edge.deploy import DeploymentConfig, EdgeDeploymentManager

edge_cli = typer.Typer(name="edge", help="Jetson edge deployment commands")
console = Console()
log = get_logger("skyguard.edge.cli")


@edge_cli.command(name="hardware")
def hardware_cmd() -> None:
    """Probe and print Jetson hardware information."""
    setup_logging()
    hw = detect_jetson()
    console.print(Panel.fit(
        "\n".join(_format_hw_lines(hw)),
        title=f"[bold cyan]SkyGuard Edge Hardware[/bold cyan]",
        border_style="cyan",
    ))


def _format_hw_lines(hw) -> list:
    lines = [f"Platform         : {hw.arch}"]
    lines.append(f"Model            : {hw.model}")
    lines.append(f"Is Jetson        : {hw.is_jetson}")
    if hw.is_jetson:
        lines.append(f"Family           : {hw.family}")
        lines.append(f"CUDA Cores       : {hw.cuda_cores}")
        lines.append(f"Tensor Cores     : {hw.tensor_cores}")
        lines.append(f"Memory           : {hw.memory_mb} MB")
        lines.append(f"Max Power        : {hw.max_power_w} W")
        lines.append(f"JetPack          : {hw.jetpack_version or 'unknown'}")
        lines.append(f"Supports FP16    : {hw.supports_fp16}")
        lines.append(f"Supports INT8    : {hw.supports_int8}")
    lines.append(f"CUDA Available   : {hw.cuda_available}")
    lines.append(f"CUDA Version     : {hw.cuda_version or 'N/A'}")
    lines.append(f"TensorRT         : {hw.tensorrt_available}")
    lines.append(f"TensorRT Version : {hw.tensorrt_version or 'N/A'}")
    lines.append(f"GPU Name         : {hw.gpu_name or 'N/A'}")
    return lines


@edge_cli.command(name="power")
def power_cmd(
    set_mode: Optional[int] = typer.Option(None, "--set", "-s", help="Set power mode (0=MAXN, 1=10W, ...)"),
    max_perf: bool = typer.Option(False, "--max-perf", help="Enable MAXN + jetson_clocks"),
) -> None:
    """Query or set Jetson power mode."""
    setup_logging()
    pm = PowerManager()

    if max_perf:
        if pm.enable_max_performance():
            console.print("[green]Max performance enabled (MAXN + jetson_clocks)[/green]")
        else:
            console.print("[red]Failed to enable max performance[/red]")
        raise typer.Exit()

    if set_mode is not None:
        try:
            mode = PowerMode(set_mode)
        except ValueError:
            console.print(f"[red]Invalid power mode: {set_mode}[/red]")
            raise typer.Exit(code=1)
        if pm.set_mode(mode):
            console.print(f"[green]Power mode set to {mode.name} ({int(mode)})[/green]")
        else:
            console.print(f"[red]Failed to set power mode {mode.name}[/red]")
        raise typer.Exit()

    # Default: query and print
    state = pm.get_state()
    table = Table(title="Power State", show_header=False, border_style="cyan")
    table.add_column("Key", style="dim")
    table.add_column("Value")
    table.add_row("Available", str(state.available))
    if state.available:
        table.add_row("Mode", f"{state.mode} ({state.mode_name})")
        table.add_row("Jetson Clocks", "ON" if state.jetson_clocks_active else "OFF")
        rec = pm.recommend_mode()
        table.add_row("Recommended", f"{int(rec)} ({rec.name})")
    console.print(table)


@edge_cli.command(name="build")
def build_cmd(
    onnx: Path = typer.Argument(..., help="Source ONNX model path"),
    output: Path = typer.Argument(..., help="Output .engine file path"),
    precision: str = typer.Option("fp16", "--precision", "-p", help="fp32 | fp16 | int8"),
    imgsz: int = typer.Option(640, "--imgsz", help="Input image size"),
    batch: int = typer.Option(1, "--batch", "-b", help="Max batch size"),
    workspace: int = typer.Option(4, "--workspace", "-w", help="Workspace size in GB"),
    calib_dir: Optional[Path] = typer.Option(None, "--calib-dir", help="INT8 calibration images dir"),
    force: bool = typer.Option(False, "--force", help="Rebuild even if engine exists"),
) -> None:
    """Build a TensorRT engine from an ONNX model."""
    setup_logging()

    builder = TRTEngineBuilder()
    if not builder.available:
        console.print("[red]TensorRT is not available on this host.[/red]")
        console.print("[dim]Run this command on a Jetson device or NVIDIA GPU host with TensorRT installed.[/dim]")
        raise typer.Exit(code=1)

    try:
        prec = Precision(precision.lower())
    except ValueError:
        console.print(f"[red]Invalid precision: {precision}. Use fp32, fp16, or int8.[/red]")
        raise typer.Exit(code=1)

    if output.exists() and not force:
        console.print(f"[yellow]Engine already exists: {output}[/yellow]")
        console.print("[dim]Use --force to rebuild.[/dim]")
        raise typer.Exit()

    cfg = EngineBuildConfig(
        precision=prec,
        imgsz=imgsz,
        max_batch_size=batch,
        workspace_gb=workspace,
        calib_data_dir=calib_dir,
    )

    console.print(f"[cyan]Building TensorRT engine[/cyan]")
    console.print(f"  ONNX      : {onnx}")
    console.print(f"  Output    : {output}")
    console.print(f"  Precision : {prec.value}")
    console.print(f"  Imgsz     : {imgsz}")
    console.print(f"  Batch     : {batch}")
    console.print(f"  Workspace : {workspace} GB")
    if calib_dir:
        console.print(f"  Calib Dir : {calib_dir}")
    console.print()

    def progress(msg: str) -> None:
        console.print(f"[dim]{msg}[/dim]")

    try:
        result = builder.build(onnx, output, cfg, progress_callback=progress)
    except Exception as e:
        console.print(f"[red]Build failed: {e}[/red]")
        raise typer.Exit(code=1)

    console.print()
    console.print(Panel.fit(
        f"Engine Path   : {result.engine_path}\n"
        f"Precision     : {result.precision.value}\n"
        f"Input Shape   : {result.input_shape}\n"
        f"Build Time    : {result.build_time_sec:.1f}s\n"
        f"Engine Size   : {result.engine_size_mb:.1f} MB",
        title="[bold green]Build Successful[/bold green]",
        border_style="green",
    ))


@edge_cli.command(name="deploy")
def deploy_cmd(
    onnx: Path = typer.Argument(..., help="ONNX model path"),
    engine: Path = typer.Argument(..., help="Output engine path"),
    precision: str = typer.Option("fp16", "--precision", "-p"),
    imgsz: int = typer.Option(640, "--imgsz"),
    batch: int = typer.Option(1, "--batch", "-b"),
    calib_dir: Optional[Path] = typer.Option(None, "--calib-dir"),
    no_power: bool = typer.Option(False, "--no-power", help="Skip power mode setup"),
    no_clocks: bool = typer.Option(False, "--no-clocks", help="Skip jetson_clocks"),
    force: bool = typer.Option(False, "--force", help="Force engine rebuild"),
    start_api: bool = typer.Option(False, "--start-api", help="Start API after deploy"),
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8000, "--port"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Plan only, no changes"),
) -> None:
    """Full deployment: power + engine build + optional API start."""
    setup_logging()

    try:
        prec = Precision(precision.lower())
    except ValueError:
        console.print(f"[red]Invalid precision: {precision}[/red]")
        raise typer.Exit(code=1)

    config = DeploymentConfig(
        model_onnx_path=onnx,
        model_engine_path=engine,
        precision=prec,
        imgsz=imgsz,
        max_batch_size=batch,
        auto_set_power_mode=not no_power,
        auto_enable_jetson_clocks=not no_clocks,
        calib_data_dir=calib_dir,
        api_host=host,
        api_port=port,
    )

    manager = EdgeDeploymentManager(config, dry_run=dry_run)

    # Pre-flight
    warnings = manager.preflight()
    if warnings:
        console.print("[yellow]Pre-flight warnings:[/yellow]")
        for w in warnings:
            console.print(f"  [yellow]![/yellow] {w}")
        console.print()

    console.print("[cyan]Starting deployment...[/cyan]")
    report = manager.deploy(force_rebuild=force)

    # Print report
    table = Table(title="Deployment Report", border_style="cyan", show_header=False)
    table.add_column("Key", style="dim")
    table.add_column("Value")
    table.add_row("Hardware", report.hardware.model)
    table.add_row("Is Jetson", str(report.hardware.is_jetson))
    table.add_row("Power Mode Set", str(report.power_mode_set.name if report.power_mode_set else "N/A"))
    table.add_row("Jetson Clocks", "ON" if report.jetson_clocks_enabled else "OFF")
    table.add_row("Engine Reused", str(report.engine_reused))
    if report.engine_build:
        table.add_row("Engine Built", str(report.engine_build.engine_path))
        table.add_row("Build Time", f"{report.engine_build.build_time_sec:.1f}s")
        table.add_row("Engine Size", f"{report.engine_build.engine_size_mb:.1f} MB")
    table.add_row("Deploy Time", f"{report.deploy_time_sec:.1f}s")
    table.add_row("Success", "[green]Yes[/green]" if report.success else "[red]No[/red]")
    console.print(table)

    if report.warnings:
        console.print()
        console.print("[yellow]Warnings:[/yellow]")
        for w in report.warnings:
            console.print(f"  [yellow]![/yellow] {w}")

    if start_api and report.success:
        console.print()
        console.print(f"[cyan]Starting SkyGuard API on {host}:{port}...[/cyan]")
        manager.start_api(block=True)


@edge_cli.command(name="health")
def health_cmd(
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Take a one-shot health snapshot."""
    setup_logging()
    monitor = EdgeMonitor()
    health = monitor.snapshot()

    if json_output:
        console.print_json(json.dumps(health.to_dict()))
    else:
        table = Table(title="Edge Health", border_style="cyan", show_header=False)
        table.add_column("Metric", style="dim")
        table.add_column("Value")
        table.add_row("CPU Usage", f"{health.cpu_percent:.1f}%")
        table.add_row("Memory", f"{health.memory_percent:.1f}% ({health.memory_used_mb:.0f} / {health.memory_total_mb:.0f} MB)")
        table.add_row("Disk", f"{health.disk_percent:.1f}%")
        if health.gpu_percent is not None:
            table.add_row("GPU Usage", f"{health.gpu_percent:.1f}%")
        if health.gpu_temp is not None:
            table.add_row("GPU Temp", f"{health.gpu_temp:.1f}°C")
        if health.cpu_temp is not None:
            table.add_row("CPU Temp", f"{health.cpu_temp:.1f}°C")
        if health.thermal is not None:
            table.add_row("Board Temp", f"{health.thermal:.1f}°C")
        if health.power_draw_mw is not None:
            table.add_row("Power Draw", f"{health.power_draw_mw:.0f} mW")
        if health.jetson_clocks_active is not None:
            table.add_row("Jetson Clocks", "ON" if health.jetson_clocks_active else "OFF")
        if health.power_mode is not None:
            table.add_row("Power Mode", str(health.power_mode))
        console.print(table)

        if health.warnings:
            console.print()
            for w in health.warnings:
                if w.startswith("CRITICAL"):
                    console.print(f"[red bold]{w}[/red bold]")
                else:
                    console.print(f"[yellow]{w}[/yellow]")


@edge_cli.command(name="watch")
def watch_cmd(
    interval: float = typer.Option(2.0, "--interval", "-i", help="Seconds between snapshots"),
    duration: float = typer.Option(0, "--duration", "-d", help="Stop after N seconds (0 = forever)"),
) -> None:
    """Continuously monitor edge device health."""
    setup_logging()
    monitor = EdgeMonitor()

    console.print(f"[cyan]Monitoring edge health every {interval}s...[/cyan]")
    console.print("[dim]Press Ctrl+C to stop[/dim]")
    console.print()

    async def _run():
        count = 0
        async for health in monitor.stream(interval=interval):
            count += 1
            temps = []
            if health.gpu_temp is not None:
                temps.append(f"GPU={health.gpu_temp:.0f}°C")
            if health.cpu_temp is not None:
                temps.append(f"CPU={health.cpu_temp:.0f}°C")
            temp_str = " ".join(temps) if temps else ""
            power_str = f"PWR={health.power_draw_mw:.0f}mW" if health.power_draw_mw else ""

            console.print(
                f"[{count:4d}] CPU={health.cpu_percent:5.1f}% "
                f"MEM={health.memory_percent:5.1f}% "
                f"GPU={health.gpu_percent if health.gpu_percent is not None else '--':>5} "
                f"{temp_str} {power_str}"
            )

            if health.warnings:
                for w in health.warnings:
                    console.print(f"      [yellow]![/yellow] {w}")

            if duration > 0 and count * interval >= duration:
                break

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped[/yellow]")


@edge_cli.command(name="info")
def info_cmd() -> None:
    """Print combined hardware + power + TensorRT info."""
    setup_logging()
    console.print(Panel.fit("[bold cyan]SkyGuard Edge Info[/bold cyan]", border_style="cyan"))
    console.print("\n[bold]Hardware[/bold]")
    print_hardware_info()
    console.print("\n[bold]Power[/bold]")
    print_power_state()
    console.print("\n[bold]TensorRT[/bold]")
    print_builder_info()
