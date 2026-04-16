from datetime import datetime


def generate_pdf_report(output_path: str, summary: dict, options: dict):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.pdfgen import canvas
    except ImportError as exc:
        raise RuntimeError(
            "reportlab is required for PDF report generation. Install with 'pip install reportlab'."
        ) from exc

    pdf = canvas.Canvas(output_path, pagesize=A4)
    width, height = A4

    x = 20 * mm
    y = height - 20 * mm

    def line(text: str, step: float = 7 * mm):
        nonlocal y
        pdf.drawString(x, y, text)
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
        line("Linearity Plot (Placeholder)", 8 * mm)
        pdf.setFont("Helvetica", 10)
        line("A linearity plot will be included here in future versions.")
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
