#!/usr/bin/env python3
"""
DC Microgrid MATLAB Visualization Documentation
"""

from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Preformatted, Image
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import registerFontFamily

# Register fonts
pdfmetrics.registerFont(TTFont('Times New Roman', '/usr/share/fonts/truetype/english/Times-New-Roman.ttf'))
pdfmetrics.registerFont(TTFont('DejaVuSans', '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'))
registerFontFamily('Times New Roman', normal='Times New Roman', bold='Times New Roman')

# Create document
output_path = '/home/z/my-project/download/MATLAB_Circuit_Visualization_Guide.pdf'
doc = SimpleDocTemplate(
    output_path,
    pagesize=letter,
    title='MATLAB Circuit Visualization Guide',
    author='Z.ai',
    creator='Z.ai',
    subject='Guide for MATLAB circuit schematic visualization with fault highlighting'
)

# Styles
styles = getSampleStyleSheet()

title_style = ParagraphStyle('CustomTitle', parent=styles['Title'], fontName='Times New Roman',
    fontSize=24, textColor=colors.HexColor('#1a1a2e'), spaceAfter=10)

subtitle_style = ParagraphStyle('Subtitle', parent=styles['Normal'], fontName='Times New Roman',
    fontSize=14, textColor=colors.HexColor('#666666'), alignment=TA_CENTER, spaceAfter=20)

h1_style = ParagraphStyle('H1', parent=styles['Heading1'], fontName='Times New Roman',
    fontSize=16, textColor=colors.HexColor('#e94560'), spaceBefore=20, spaceAfter=10)

h2_style = ParagraphStyle('H2', parent=styles['Heading2'], fontName='Times New Roman',
    fontSize=14, textColor=colors.HexColor('#16213e'), spaceBefore=15, spaceAfter=8)

body_style = ParagraphStyle('Body', parent=styles['Normal'], fontName='Times New Roman',
    fontSize=11, leading=16, alignment=TA_JUSTIFY, spaceAfter=10)

code_style = ParagraphStyle('Code', fontName='DejaVuSans', fontSize=8, leading=11,
    backColor=colors.HexColor('#f5f5f5'), borderColor=colors.HexColor('#ddd'),
    borderWidth=1, borderPadding=8, spaceAfter=10)

header_style = ParagraphStyle('TableHeader', fontName='Times New Roman', fontSize=10,
    textColor=colors.white, alignment=TA_CENTER)

cell_style = ParagraphStyle('TableCell', fontName='Times New Roman', fontSize=10,
    textColor=colors.black, alignment=TA_LEFT)

# Build content
story = []

# Title
story.append(Paragraph('MATLAB Circuit Visualization Guide', title_style))
story.append(Paragraph('Digital Twin with Fault Location Highlighting', subtitle_style))
story.append(Spacer(1, 10))

# Overview
story.append(Paragraph('<b>Overview</b>', h1_style))
story.append(Paragraph(
    'This document describes the MATLAB-based circuit visualization system for the DC Microgrid '
    'Fault Detection Platform. Unlike the previous node-graph approach, this system renders a '
    'proper electrical circuit schematic with bus bars, cables, generators, and loads. When a fault '
    'is injected, the exact location is visually highlighted on the schematic with animated indicators.',
    body_style
))
story.append(Spacer(1, 10))

# Features
story.append(Paragraph('<b>Key Features</b>', h2_style))

features_data = [
    [Paragraph('<b>Feature</b>', header_style), Paragraph('<b>Description</b>', header_style)],
    [Paragraph('Proper Schematic', cell_style),
     Paragraph('Renders electrical circuit with bus bars, cables, generators, loads using standard symbols', cell_style)],
    [Paragraph('Fault Highlighting', cell_style),
     Paragraph('Animated pulsing circles and lightning bolt symbols indicate exact fault location', cell_style)],
    [Paragraph('Real-time Updates', cell_style),
     Paragraph('Circuit state updates in real-time via MATLAB Engine API or file-based mode', cell_style)],
    [Paragraph('Voltage Display', cell_style),
     Paragraph('Each bus shows current voltage level, updating as conditions change', cell_style)],
    [Paragraph('Component Status', cell_style),
     Paragraph('Generators, loads, and buses show operational status (normal/fault)', cell_style)],
    [Paragraph('Interactive Legend', cell_style),
     Paragraph('Color-coded legend explains all visual elements', cell_style)],
]

features_table = Table(features_data, colWidths=[1.8*inch, 4.5*inch])
features_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1F4E79')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
    ('BACKGROUND', (0, 1), (-1, -1), colors.white),
    ('BACKGROUND', (0, 2), (-1, 2), colors.HexColor('#F5F5F5')),
    ('BACKGROUND', (0, 4), (-1, 4), colors.HexColor('#F5F5F5')),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('LEFTPADDING', (0, 0), (-1, -1), 6),
    ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ('TOPPADDING', (0, 0), (-1, -1), 4),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
]))
story.append(features_table)
story.append(Spacer(1, 15))

# Page break
story.append(PageBreak())

# File Structure
story.append(Paragraph('<b>Files Provided</b>', h1_style))

files_data = [
    [Paragraph('<b>File</b>', header_style), 
     Paragraph('<b>Purpose</b>', header_style),
     Paragraph('<b>Location</b>', header_style)],
    [Paragraph('DC_Microgrid_Visualize.m', cell_style),
     Paragraph('MATLAB script for rendering circuit schematic with fault highlighting', cell_style),
     Paragraph('matlab/', cell_style)],
    [Paragraph('matlab_visualizer.py', cell_style),
     Paragraph('Python bridge to MATLAB visualization', cell_style),
     Paragraph('src/adapters/', cell_style)],
    [Paragraph('digital_twin_enhanced.py', cell_style),
     Paragraph('Streamlit page with embedded circuit schematic', cell_style),
     Paragraph('src/ui/pages/', cell_style)],
]

files_table = Table(files_data, colWidths=[2.0*inch, 2.8*inch, 1.5*inch])
files_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1F4E79')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
    ('BACKGROUND', (0, 1), (-1, -1), colors.white),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('LEFTPADDING', (0, 0), (-1, -1), 6),
    ('RIGHTPADDING', (0, 0), (-1, -1), 6),
]))
story.append(files_table)
story.append(Spacer(1, 15))

# MATLAB Usage
story.append(Paragraph('<b>MATLAB Script Usage</b>', h1_style))

story.append(Paragraph('<b>Basic Usage (No Fault):</b>', body_style))
story.append(Preformatted('''
% Load circuit data
circuit_data = load('circuit_state.mat');

% Visualize circuit
result = DC_Microgrid_Visualize(circuit_data.circuit);

% Result contains figure handle for updates
disp(result.figure);
''', code_style))

story.append(Paragraph('<b>With Fault Highlighting:</b>', body_style))
story.append(Preformatted('''
% Define fault
fault_data.active = true;
fault_data.location = '4';        % Bus ID or Line ID (e.g., 'L2')
fault_data.type = 'LINE_TO_LINE';
fault_data.severity = 0.8;

% Visualize with fault
result = DC_Microgrid_Visualize(circuit_data.circuit, fault_data);
''', code_style))

story.append(Paragraph('<b>Real-time Updates:</b>', body_style))
story.append(Preformatted('''
% Initial visualization
result = DC_Microgrid_Visualize(circuit_data.circuit);

% Later, when fault occurs:
new_fault.active = true;
new_fault.location = '2';
new_fault.type = 'ARC_FAULT';
new_fault.severity = 0.6;

% Update the display
result.update(new_fault);
''', code_style))
story.append(Spacer(1, 10))

# Python Integration
story.append(Paragraph('<b>Python Integration</b>', h1_style))

story.append(Paragraph('<b>Initialization:</b>', body_style))
story.append(Preformatted('''
from src.adapters.matlab_visualizer import MatlabCircuitVisualizer

# Create visualizer (auto-detects MATLAB availability)
viz = MatlabCircuitVisualizer()
''', code_style))

story.append(Paragraph('<b>Update Circuit:</b>', body_style))
story.append(Preformatted('''
# Update with current circuit state
viz.update_circuit(circuit_model, emulator)

# Highlight fault
viz.highlight_fault(
    location='4',              # Bus ID
    fault_type='LINE_TO_LINE',
    severity=0.8,
    distance=150.0             # meters from sensor
)

# Clear fault
viz.clear_fault()
''', code_style))

story.append(Paragraph('<b>Event Handler Integration:</b>', body_style))
story.append(Preformatted('''
from src.adapters.matlab_visualizer import VisualizationEventHandler

# Create handler
handler = VisualizationEventHandler(viz)

# Subscribe to events
bus.subscribe(FaultLocationEvent, handler.on_fault_location)
bus.subscribe(SystemTripEvent, handler.on_system_trip)
''', code_style))
story.append(Spacer(1, 10))

# Page break
story.append(PageBreak())

# Circuit Data Format
story.append(Paragraph('<b>Circuit Data Format</b>', h1_style))

story.append(Paragraph('<b>MATLAB Input Structure:</b>', body_style))
story.append(Preformatted('''
circuit.buses(i).id       = 1;
circuit.buses(i).name     = 'Main PCC';
circuit.buses(i).type     = 'Slack';
circuit.buses(i).x        = 0.5;      % Layout coordinate
circuit.buses(i).y        = 0.85;
circuit.buses(i).voltage  = 400.0;    % Real-time voltage (V)

circuit.lines(i).id       = 1;
circuit.lines(i).from_bus = 6;
circuit.lines(i).to_bus   = 1;
circuit.lines(i).r_ohm    = 0.01;
circuit.lines(i).length_km = 0.05;

circuit.generators(i).id    = 1;
circuit.generators(i).bus_id = 6;
circuit.generators(i).p_mw  = 0.5;

circuit.loads(i).id      = 1;
circuit.loads(i).bus_id  = 4;
circuit.loads(i).p_mw    = 0.15;
''', code_style))
story.append(Spacer(1, 10))

# Fault Types
story.append(Paragraph('<b>Supported Fault Types</b>', h1_style))

fault_data = [
    [Paragraph('<b>Fault Type</b>', header_style), 
     Paragraph('<b>Symbol</b>', header_style),
     Paragraph('<b>Visual Effect</b>', header_style)],
    [Paragraph('LINE_TO_LINE', cell_style),
     Paragraph('Lightning Bolt', cell_style),
     Paragraph('Red bus/cable, pulsing circles, flash symbol', cell_style)],
    [Paragraph('LINE_TO_GROUND', cell_style),
     Paragraph('Ground Arc', cell_style),
     Paragraph('Ground symbol with arc, voltage drop display', cell_style)],
    [Paragraph('ARC_FAULT', cell_style),
     Paragraph('Zigzag Arc', cell_style),
     Paragraph('Intermittent flashing, zigzag pattern', cell_style)],
    [Paragraph('NOISE', cell_style),
     Paragraph('Noise Wave', cell_style),
     Paragraph('Wavy pattern overlay', cell_style)],
    [Paragraph('DRIFT', cell_style),
     Paragraph('Down Arrow', cell_style),
     Paragraph('Gradual voltage decline indicator', cell_style)],
]

fault_table = Table(fault_data, colWidths=[1.6*inch, 1.4*inch, 3.0*inch])
fault_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1F4E79')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
    ('BACKGROUND', (0, 1), (-1, -1), colors.white),
    ('BACKGROUND', (0, 2), (-1, 2), colors.HexColor('#FFF3E0')),
    ('BACKGROUND', (0, 4), (-1, 4), colors.HexColor('#FFF3E0')),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('LEFTPADDING', (0, 0), (-1, -1), 6),
    ('RIGHTPADDING', (0, 0), (-1, -1), 6),
]))
story.append(fault_table)
story.append(Spacer(1, 15))

# Layout Recommendations
story.append(Paragraph('<b>Layout Recommendations</b>', h1_style))
story.append(Paragraph(
    'For optimal schematic visualization, use the following layout conventions:',
    body_style
))
story.append(Preformatted('''
RECOMMENDED LAYOUT:
====================

        [Grid Infeed] - Bus 6 (y=0.95)
              |
        [Main PCC] ---- Bus 1 (y=0.85)
           /    \\
    [Solar]    [Battery]    (y=0.85, left/right)
      |            |
   [Load A]    [Load B]     (y=0.35)

COORDINATE GUIDELINES:
- y=0.9-1.0: Main DC bus / Grid connection
- y=0.7-0.9: Generation sources
- y=0.3-0.5: Load centers
- x spread: 0.1 to 0.9 for visual separation
''', code_style))
story.append(Spacer(1, 10))

# Standalone Mode
story.append(Paragraph('<b>Standalone Mode (No MATLAB Engine)</b>', h1_style))
story.append(Paragraph(
    'If the MATLAB Engine API is not available, the visualizer operates in standalone mode, '
    'exporting circuit state to files that can be manually loaded in MATLAB:',
    body_style
))
story.append(Preformatted('''
# Files generated in standalone mode:
/tmp/dc_microgrid_viz/circuit_state.mat   # MATLAB format
/tmp/dc_microgrid_viz/circuit_state.json  # JSON format

# In MATLAB, load and visualize:
cd /tmp/dc_microgrid_viz
data = load('circuit_state.mat');
DC_Microgrid_Visualize(data.circuit);
''', code_style))
story.append(Spacer(1, 10))

# Installation
story.append(Paragraph('<b>Installation</b>', h1_style))

story.append(Paragraph('<b>Requirements:</b>', body_style))
story.append(Preformatted('''
# Python dependencies (already in project)
scipy>=1.0.0
numpy>=1.20.0

# Optional: MATLAB Engine for Python
pip install matlabengine
''', code_style))

story.append(Paragraph('<b>MATLAB Setup:</b>', body_style))
story.append(Preformatted('''
1. Copy DC_Microgrid_Visualize.m to your MATLAB path or project's matlab/ directory

2. Ensure MATLAB can find the script:
   addpath('/path/to/project/matlab');

3. Test the visualization:
   circuit = load('test_circuit.mat');
   DC_Microgrid_Visualize(circuit);
''', code_style))
story.append(Spacer(1, 10))

# Conclusion
story.append(Paragraph('<b>Summary</b>', h1_style))
story.append(Paragraph(
    'The MATLAB circuit visualization system provides a professional electrical schematic '
    'display with real-time fault location highlighting. Unlike the previous node-graph approach, '
    'this renders a proper circuit diagram with standard electrical symbols. The system supports '
    'both direct MATLAB Engine integration for real-time updates and file-based mode for '
    'standalone operation. When a fault is injected, the exact component (bus or line) is '
    'visually highlighted with animated indicators, making it immediately clear where the fault '
    'occurred in the physical circuit.',
    body_style
))

# Build PDF
doc.build(story)
print(f"PDF generated: {output_path}")
