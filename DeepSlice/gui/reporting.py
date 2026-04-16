from datetime import datetime
import io


def generate_pdf_report(output_path: str, summary: dict, options: dict):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.lib.utils import ImageReader
        from reportlab.pdfgen import canvas
    except ImportError as exc:
        raise RuntimeError(
            "reportlab is required for PDF report generation. Install with 'pip install reportlab'."
        ) from exc

    pdf = canvas.Canvas(output_path, pagesize=A4)
    width, height = A4

    x_pos = 20 * mm
    y = height - 20 * mm

    def line(text: str, step: float = 7 * mm):
        nonlocal y
        pdf.drawString(x_pos, y, text)
        y -= step

    pdf.setTitle("DeepSlice Alignment Report")

    pdf.setFont("Helvetica-Bold", 16)
    line("DeepSlice Alignment Report", 10 * mm)

    pdf.setFont("Helvetica", 10)
    line(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    line("")

    if options.get("include_stats", True):
        pdf.setFont("Helvetica-Bold", 12)
        line("Summary", 8 * mm)
        pdf.setFont("Helvetica", 10)
        line(f"Total slices: {summary.get('slice_count', 0)}")
        line(f"Processed slices: {summary.get('processed', 0)}")
        line(f"Excluded slices: {summary.get('excluded', 0)}")
        line(
            "Mean angular deviation (deg): "
            + f"{summary.get('mean_angular_deviation', 0.0):.3f}"
        )
        line("")

        pdf.setFont("Helvetica-Bold", 12)
        line("Run Options", 8 * mm)
        pdf.setFont("Helvetica", 10)
        line(f"Species: {options.get('species', 'unknown')}")
        line(f"Section number detection: {options.get('section_numbers', True)}")
        line(f"Legacy section parsing: {options.get('legacy_section_numbers', False)}")
        line(f"Ensemble mode: {options.get('ensemble', False)}")
        line(f"Use secondary model only: {options.get('use_secondary_model', False)}")
        line(f"Direction override: {options.get('direction', 'Auto')}")

        thickness = options.get("thickness_um")
        thickness_str = f"{thickness:.2f} um" if thickness is not None else "Auto"
        line(f"Thickness: {thickness_str}")
        line("")

    if options.get("include_plot", True):
        pdf.setFont("Helvetica-Bold", 12)
        line("Linearity Plot", 8 * mm)

        payload = options.get("linearity_payload", {})
        x_vals = payload.get("x", []) if isinstance(payload, dict) else []
        y_vals = payload.get("y", []) if isinstance(payload, dict) else []
        trend_vals = payload.get("trend", []) if isinstance(payload, dict) else []
        conf_vals = payload.get("confidence", []) if isinstance(payload, dict) else []

        if len(x_vals) > 0 and len(y_vals) == len(x_vals):
            try:
                import numpy as np
                from matplotlib.figure import Figure

                x_data = np.asarray(x_vals, dtype=float)
                y_data = np.asarray(y_vals, dtype=float)
                trend = np.asarray(trend_vals, dtype=float)
                if trend.shape[0] != x_data.shape[0]:
                    trend = np.linspace(float(np.min(y_data)), float(np.max(y_data)), x_data.shape[0])

                conf = np.asarray(conf_vals, dtype=float)
                if conf.shape[0] != x_data.shape[0]:
                    conf = np.ones_like(x_data, dtype=float) * 0.5

                figure = Figure(figsize=(6.2, 2.8), dpi=150)
                axis = figure.add_subplot(111)
                axis.set_facecolor("#F8FAFC")
                scatter = axis.scatter(
                    x_data,
                    y_data,
                    c=conf,
                    cmap="RdYlGn",
                    edgecolors="#334155",
                    linewidths=0.4,
                    s=16,
                )
                axis.plot(x_data, trend, color="#2563EB", linewidth=1.5, label="Linear fit")
                axis.set_xlabel("Section Index")
                axis.set_ylabel("Predicted AP Position")
                axis.grid(alpha=0.25)
                axis.legend(loc="best", fontsize=7)
                figure.colorbar(scatter, ax=axis, fraction=0.045, pad=0.02, label="Confidence")
                figure.tight_layout()

                buffer = io.BytesIO()
                figure.savefig(buffer, format="png", dpi=150)
                buffer.seek(0)

                available_width = 170 * mm
                max_height = 80 * mm
                image_width = available_width
                image_height = max_height

                if y - image_height < 20 * mm:
                    pdf.showPage()
                    y = height - 20 * mm
                    pdf.setFont("Helvetica-Bold", 12)
                    line("Linearity Plot", 8 * mm)

                pdf.drawImage(
                    ImageReader(buffer),
                    x_pos,
                    y - image_height,
                    width=image_width,
                    height=image_height,
                    preserveAspectRatio=True,
                    mask="auto",
                )
                y -= image_height + 6 * mm
            except Exception:
                pdf.setFont("Helvetica", 10)
                line("Could not render plot image; data is available in CSV export.")
                line("")
        else:
            pdf.setFont("Helvetica", 10)
            line("Plot data unavailable for this session.")
            line("")

    if options.get("include_images", True):
        pdf.setFont("Helvetica-Bold", 12)
        line("Sample Images (Placeholder)", 8 * mm)
        pdf.setFont("Helvetica", 10)
        line("Representative section alignments will be included here in future versions.")
        line("")

    if options.get("include_angles", True):
        pdf.setFont("Helvetica-Bold", 12)
        line("Angle Metrics", 8 * mm)
        pdf.setFont("Helvetica", 10)
        line(
            "Angle metrics are derived from DV/ML distributions and summarized in the GUI export panel."
        )
        line("")

    pdf.save()
