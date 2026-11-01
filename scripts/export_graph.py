"""Export the LangGraph workflow graph to Mermaid PNG and Markdown."""

import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.graph import get_compiled_graph

def export_graph():
    print("Loading compiled LangGraph workflow...")
    app = get_compiled_graph()
    graph = app.get_graph()

    output_dir = root_dir / "docs" / "img"
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "workflow_graph.png"

    print("Generating Mermaid PNG via LangGraph...")
    try:
        png_bytes = graph.draw_mermaid_png()
        png_path.write_bytes(png_bytes)
        print(f"✅ Successfully exported graph image to: {png_path} ({len(png_bytes)} bytes)")
    except Exception as e:
        print(f"⚠️ Could not generate PNG directly: {e}")

    mermaid_syntax = graph.draw_mermaid()
    mermaid_path = output_dir / "workflow_graph.mmd"
    mermaid_path.write_text(mermaid_syntax, encoding="utf-8")
    print(f"✅ Saved Mermaid definition to: {mermaid_path}")

    return mermaid_syntax

if __name__ == "__main__":
    export_graph()
