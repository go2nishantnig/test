#!/bin/bash
# Script to generate all PlantUML diagrams as PNG images
# This helps verify that all diagrams are valid and can be rendered

echo "======================================"
echo "PlantUML Diagram Generator"
echo "======================================"
echo ""

# Check if PlantUML is installed
if ! command -v plantuml &> /dev/null; then
    echo "Error: PlantUML is not installed."
    echo ""
    echo "To install PlantUML:"
    echo "  Ubuntu/Debian: sudo apt-get install plantuml"
    echo "  macOS: brew install plantuml"
    echo "  Or download from: https://plantuml.com/download"
    echo ""
    exit 1
fi

echo "PlantUML found: $(plantuml -version 2>&1 | head -1)"
echo ""

# Change to diagrams directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Count .puml files
TOTAL_FILES=$(ls -1 *.puml 2>/dev/null | wc -l)

if [ "$TOTAL_FILES" -eq 0 ]; then
    echo "Error: No .puml files found in $SCRIPT_DIR"
    exit 1
fi

echo "Found $TOTAL_FILES PlantUML diagram files"
echo ""

# Generate PNG images
echo "Generating PNG images..."
plantuml -tpng *.puml 2>&1 | grep -v "^(" | grep -v "^$"

# Count generated images
PNG_COUNT=$(ls -1 *.png 2>/dev/null | wc -l)

echo ""
echo "======================================"
echo "Generation complete!"
echo "======================================"
echo "Generated $PNG_COUNT PNG images"
echo ""

if [ "$PNG_COUNT" -gt 0 ]; then
    echo "Generated files:"
    ls -lh *.png
    echo ""
    echo "Note: PNG files are ignored by git (see .gitignore)"
    echo "These are for local viewing only."
fi

echo ""
echo "To generate SVG instead (better quality):"
echo "  plantuml -tsvg *.puml"
echo ""
