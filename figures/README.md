# Architecture Figures

LaTeX/TikZ figures for the Enhanced 3D U-Net architecture.

## Files

- `enhanced_unet_architecture.tex` - Main architecture diagram
- `enhanced_unet_with_inset.tex` - Architecture with attention gate detail inset (recommended)

## Compilation

```bash
# Compile to PDF
pdflatex enhanced_unet_with_inset.tex

# Or use latexmk for automatic rebuilds
latexmk -pdf enhanced_unet_with_inset.tex
```

## Requirements

LaTeX packages:
- `tikz` (with libraries: shapes.geometric, arrows.meta, positioning, calc, backgrounds, fit, decorations.pathreplacing)
- `xcolor`
- `amsmath`
- `standalone` (document class)

## Converting to Other Formats

```bash
# PDF to PNG (high resolution)
pdftoppm -png -r 600 enhanced_unet_with_inset.pdf enhanced_unet

# PDF to SVG (for editing in Inkscape/Illustrator)
pdf2svg enhanced_unet_with_inset.pdf enhanced_unet.svg

# Or use Inkscape directly
inkscape enhanced_unet_with_inset.pdf --export-filename=enhanced_unet.svg
```

## Customization

### Colors
Edit the `\definecolor` commands at the top to change the color scheme:
```latex
\definecolor{encoderblue}{RGB}{55, 126, 184}
\definecolor{decodergreen}{RGB}{77, 175, 74}
\definecolor{bottleneckorange}{RGB}{255, 127, 0}
\definecolor{attentionpurple}{RGB}{152, 78, 163}
```

### Block sizes
Adjust `minimum height` values in the node definitions to change block proportions.

### Channel labels
Update the channel numbers (32, 64, 128, 256) in the `\node[chanlabel]` lines.

## Including in Papers

### LaTeX
```latex
\begin{figure}[t]
    \centering
    \includegraphics[width=\textwidth]{figures/enhanced_unet_with_inset.pdf}
    \caption{Architecture of the Enhanced 3D Residual U-Net with attention gates
    and deep supervision. Numbers above blocks indicate channel dimensions.
    AG: Attention Gate.}
    \label{fig:architecture}
\end{figure}
```

### Suggested caption
> **Figure X.** Enhanced 3D Residual U-Net architecture. The encoder path (blue)
> extracts hierarchical features using residual blocks with squeeze-and-excitation
> (SE) attention. Skip connections pass through attention gates (AG, purple) that
> learn to suppress irrelevant regions. The decoder path (green) reconstructs the
> segmentation with deep supervision branches (red) providing auxiliary losses at
> intermediate scales. Inset shows the attention gate mechanism.
