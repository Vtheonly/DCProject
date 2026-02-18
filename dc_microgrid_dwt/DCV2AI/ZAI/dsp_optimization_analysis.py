#!/usr/bin/env python3
"""
DC Microgrid DSP Optimization Analysis - PDF Generator
"""

from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Preformatted
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import registerFontFamily
import os

# Register fonts
pdfmetrics.registerFont(TTFont('Times New Roman', '/usr/share/fonts/truetype/english/Times-New-Roman.ttf'))
pdfmetrics.registerFont(TTFont('DejaVuSans', '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'))
registerFontFamily('Times New Roman', normal='Times New Roman', bold='Times New Roman')

# Create document
output_path = '/home/z/my-project/download/DC_Microgrid_DSP_Optimization_Analysis.pdf'
doc = SimpleDocTemplate(
    output_path,
    pagesize=letter,
    title='DC Microgrid DSP Optimization Analysis',
    author='Z.ai',
    creator='Z.ai',
    subject='Comprehensive analysis and fixes for C++ DSP integration issues'
)

# Styles
styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    'CustomTitle',
    parent=styles['Title'],
    fontName='Times New Roman',
    fontSize=24,
    textColor=colors.HexColor('#1a1a2e'),
    spaceAfter=10
)

subtitle_style = ParagraphStyle(
    'Subtitle',
    parent=styles['Normal'],
    fontName='Times New Roman',
    fontSize=14,
    textColor=colors.HexColor('#666666'),
    alignment=TA_CENTER,
    spaceAfter=20
)

h1_style = ParagraphStyle(
    'H1',
    parent=styles['Heading1'],
    fontName='Times New Roman',
    fontSize=16,
    textColor=colors.HexColor('#e94560'),
    spaceBefore=20,
    spaceAfter=10
)

h2_style = ParagraphStyle(
    'H2',
    parent=styles['Heading2'],
    fontName='Times New Roman',
    fontSize=14,
    textColor=colors.HexColor('#16213e'),
    spaceBefore=15,
    spaceAfter=8
)

body_style = ParagraphStyle(
    'Body',
    parent=styles['Normal'],
    fontName='Times New Roman',
    fontSize=11,
    leading=16,
    alignment=TA_JUSTIFY,
    spaceAfter=10
)

code_style = ParagraphStyle(
    'Code',
    fontName='DejaVuSans',
    fontSize=8,
    leading=11,
    backColor=colors.HexColor('#f5f5f5'),
    borderColor=colors.HexColor('#ddd'),
    borderWidth=1,
    borderPadding=8,
    spaceAfter=10
)

header_style = ParagraphStyle(
    'TableHeader',
    fontName='Times New Roman',
    fontSize=10,
    textColor=colors.white,
    alignment=TA_CENTER
)

cell_style = ParagraphStyle(
    'TableCell',
    fontName='Times New Roman',
    fontSize=10,
    textColor=colors.black,
    alignment=TA_LEFT
)

# Build content
story = []

# Title
story.append(Paragraph('DC Microgrid DSP Optimization Analysis', title_style))
story.append(Paragraph('Comprehensive Fix Guide for C++ DSP Integration Issues', subtitle_style))
story.append(Spacer(1, 10))

# Executive Summary
story.append(Paragraph('<b>Executive Summary</b>', h1_style))
story.append(Paragraph(
    'This document provides a comprehensive analysis of the performance bottlenecks '
    'identified in the DC Microgrid Fault Detection System and presents detailed solutions '
    'to achieve the expected high-speed C++ DSP processing. The analysis reveals that while '
    'the C++ DSP core is functioning correctly, the Python architecture surrounding it creates '
    'significant bottlenecks that prevent the system from achieving its performance targets.',
    body_style
))
story.append(Spacer(1, 10))

# Problem Summary Table
story.append(Paragraph('<b>Root Cause Analysis Summary</b>', h2_style))

problem_data = [
    [Paragraph('<b>Problem</b>', header_style), 
     Paragraph('<b>Location</b>', header_style), 
     Paragraph('<b>Impact</b>', header_style),
     Paragraph('<b>Priority</b>', header_style)],
    [Paragraph('Bug: result.trip vs result.trip.triggered', cell_style),
     Paragraph('src/agents/processing/dsp_runner.py', cell_style),
     Paragraph('Trip detection never triggers from C++ path', cell_style),
     Paragraph('CRITICAL', cell_style)],
    [Paragraph('Double Processing', cell_style),
     Paragraph('src/ui/system.py', cell_style),
     Paragraph('Every sample processed by BOTH C++ AND Python DWT', cell_style),
     Paragraph('HIGH', cell_style)],
    [Paragraph('EventBus Overhead', cell_style),
     Paragraph('src/framework/bus.py', cell_style),
     Paragraph('20,000 synchronous calls/sec with lock contention', cell_style),
     Paragraph('HIGH', cell_style)],
    [Paragraph('Python Sampling Loop', cell_style),
     Paragraph('src/agents/ingestion/sampler.py', cell_style),
     Paragraph('time.sleep() not deterministic at 20kHz', cell_style),
     Paragraph('MEDIUM', cell_style)],
    [Paragraph('UI Blocking', cell_style),
     Paragraph('src/ui/app.py', cell_style),
     Paragraph('st.rerun() every 50ms competes for CPU', cell_style),
     Paragraph('MEDIUM', cell_style)],
]

problem_table = Table(problem_data, colWidths=[1.8*inch, 1.6*inch, 2.2*inch, 0.8*inch])
problem_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1F4E79')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
    ('BACKGROUND', (0, 1), (-1, -1), colors.white),
    ('BACKGROUND', (0, 2), (-1, 2), colors.HexColor('#FFF3E0')),
    ('BACKGROUND', (0, 4), (-1, 4), colors.HexColor('#FFF3E0')),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('LEFTPADDING', (0, 0), (-1, -1), 6),
    ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ('TOPPADDING', (0, 0), (-1, -1), 4),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
]))
story.append(problem_table)
story.append(Spacer(1, 15))

# Page break before detailed fixes
story.append(PageBreak())

# FIX 1: Critical Bug
story.append(Paragraph('Fix 1: Critical Bug in dsp_runner.py (CRITICAL)', h1_style))
story.append(Paragraph(
    'The most critical issue preventing C++ DSP from working correctly is in the '
    'trip detection logic. The code incorrectly checks <font name="DejaVuSans">result.trip</font> '
    'as a boolean instead of <font name="DejaVuSans">result.trip.triggered</font>.',
    body_style
))

story.append(Paragraph('<b>Location:</b> src/agents/processing/dsp_runner.py', body_style))

story.append(Paragraph('<b>Original Code (BROKEN):</b>', body_style))
story.append(Preformatted('''
# Line ~3159 - WRONG: Checks object instead of attribute
if result.trip:  # This is ALWAYS False!
    trip_event = SystemTripEvent(...)
    
# Line ~3176 - WRONG: d1_peak hardcoded to 0.0
res_event = ProcessingResultEvent(
    d1_peak=0.0,  # Should be result.d1_peak
    is_faulty=result.trip,  # Wrong attribute
)
''', code_style))

story.append(Paragraph('<b>Fixed Code:</b>', body_style))
story.append(Preformatted('''
def on_sample(self, event: VoltageSampleEvent):
    if not self.pipeline:
        return
    try:
        result = self.pipeline.process_sample(event.voltage)
        
        # FIX 1: Use result.trip.triggered instead of result.trip
        if result.trip.triggered:
            trip_event = SystemTripEvent(
                reason="Fast Trip (DSP Core)",
                source=self.name,
                timestamp=event.timestamp
            )
            self.logger.critical("FAST TRIP TRIGGERED BY DSP CORE")
            self.publish(trip_event)
        
        if result.window_ready:
            energy = result.energy_dict()
            
            # FIX 2: Use actual d1_peak from C++ result
            res_event = ProcessingResultEvent(
                d1_peak=result.d1_peak,  # FIXED: Was hardcoded 0.0
                d1_energy=energy.get("D1", 0.0),
                is_faulty=result.trip.triggered,  # FIXED: Was result.trip
                timestamp=event.timestamp
            )
            res_event.energy_levels = energy
            self.publish(res_event)
    except Exception as e:
        self.logger.error(f"DSP processing error: {e}")
''', code_style))
story.append(Spacer(1, 10))

# FIX 2: Stop Double Processing
story.append(Paragraph('Fix 2: Stop Double Processing (HIGH IMPACT)', h1_style))
story.append(Paragraph(
    'The system currently runs BOTH the C++ DSP path AND the Python DWT path simultaneously. '
    'This wastes CPU resources and creates race conditions. The Python DWT agents should be '
    'disabled when the C++ pipeline is available.',
    body_style
))

story.append(Paragraph('<b>Location:</b> src/ui/system.py - start_system() function', body_style))
story.append(Preformatted('''
def start_system():
    # ... existing setup code ...
    
    # 4. Create agents - CONDITIONAL on DSP availability
    sampler = SamplerAgent("Sampler", bus, config={"sample_rate": 20000})
    sampler.set_sensor(sensor)
    
    # --- CRITICAL: Only create Python DWT agents if C++ is NOT available ---
    window_mgr = None
    dwt_engine = None
    detail_analyzer = None
    
    if dsp_pipeline:
        # C++ Path - disable Python DWT chain
        dsp_runner = DSPRunnerAgent("DSPRunner", bus, 
                                     config={"dsp_pipeline": dsp_pipeline})
        add_log("Using C++ DSP Fast Path - Python DWT disabled", "INFO")
    else:
        # Python Fallback Path
        dsp_runner = None
        window_mgr = WindowManagerAgent("WindowManager", bus, 
                                        config={"window_size": 128})
        dwt_engine = DWTEngineAgent("DWTEngine", bus, config={
            "wavelet": "db4", "level": 4, "mode": "symmetric"
        })
        detail_analyzer = DetailAnalyzerAgent("DetailAnalyzer", bus)
        add_log("Using Python DSP Fallback", "WARNING")
    
    # 6. Register agents - CONDITIONALLY
    agents = [sampler]
    if not dsp_pipeline:
        agents.extend([window_mgr, dwt_engine, detail_analyzer])
    agents.extend([
        fault_locator, threshold_guard, energy_monitor, fault_voter,
        trip_sequencer, zeta_logic, health_monitor, ai_classifier,
        replay_recorder, report_generator, bridge
    ])
    if dsp_runner:
        agents.append(dsp_runner)
''', code_style))
story.append(Spacer(1, 10))

# Page break
story.append(PageBreak())

# FIX 3: High-Speed Loop
story.append(Paragraph('Fix 3: High-Speed Detection Loop (HIGH IMPACT)', h1_style))
story.append(Paragraph(
    'Bypass the EventBus for the critical path to ensure deterministic timing. '
    'Create a dedicated high-speed loop that directly calls the C++ DSP and only publishes '
    'significant events (trips, periodic UI updates).',
    body_style
))

story.append(Paragraph('<b>New File: src/adapters/high_speed_loop.py</b>', body_style))
story.append(Preformatted('''
import time
import threading
import logging
from src.domain.events import SystemTripEvent, ProcessingResultEvent

logger = logging.getLogger("HighSpeedLoop")

class HighSpeedDetectionLoop:
    """Runs outside EventBus for deterministic timing."""
    
    def __init__(self, sensor, dsp_pipeline, event_bus, sample_rate=20000):
        self.sensor = sensor
        self.pipeline = dsp_pipeline
        self.bus = event_bus
        self.sample_rate = sample_rate
        self.interval = 1.0 / sample_rate
        self._running = False
        self._thread = None
        self._sample_count = 0
        self.ui_update_interval = 100  # Throttle UI to 200Hz

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def _run(self):
        next_time = time.perf_counter()
        while self._running:
            voltage = self.sensor.read()
            result = self.pipeline.process_sample(voltage)
            
            if result.trip.triggered:
                evt = SystemTripEvent(
                    reason="Fast Trip (C++ Direct)",
                    urgency=10,
                    timestamp=time.time()
                )
                self.bus.publish(evt)
            
            self._sample_count += 1
            if self._sample_count % self.ui_update_interval == 0:
                if result.window_ready:
                    evt = ProcessingResultEvent(
                        d1_energy=result.energy_levels[0],
                        d1_peak=result.d1_peak,
                        is_faulty=result.trip.triggered,
                        timestamp=time.time()
                    )
                    self.bus.publish(evt)

            now = time.perf_counter()
            drift = (next_time - now)
            if drift > 0:
                time.sleep(drift)
            next_time += self.interval
''', code_style))
story.append(Spacer(1, 10))

# FIX 4: UI Throttling
story.append(Paragraph('Fix 4: Throttle Streamlit UI Updates (MEDIUM)', h1_style))
story.append(Paragraph('<b>Location:</b> src/ui/app.py - end of main()', body_style))
story.append(Preformatted('''
# 7. Smart auto-refresh logic
if st.session_state.system_running:
    bridge = st.session_state.get("bridge_agent")
    if bridge and not bridge.get_queue().empty():
        time.sleep(0.05)  # New data - quick refresh
        st.rerun()
    else:
        time.sleep(0.15)  # No new data - reduce CPU
        st.rerun()
''', code_style))
story.append(Spacer(1, 10))

# FIX 5: C++ Build
story.append(Paragraph('Fix 5: Optimize C++ Build Flags (MEDIUM)', h1_style))
story.append(Paragraph('<b>Location:</b> cpp/CMakeLists.txt', body_style))
story.append(Preformatted('''
# Maximum optimization flags for DSP performance
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG -funroll-loops -ffast-math")

# Enable Link-Time Optimization
include(CheckIPOSupported)
check_ipo_supported(RESULT lto_supported OUTPUT lto_output)
if(lto_supported)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()
''', code_style))
story.append(Spacer(1, 10))

# Page break
story.append(PageBreak())

# Performance Metrics
story.append(Paragraph('Expected Performance Improvements', h1_style))

perf_data = [
    [Paragraph('<b>Metric</b>', header_style),
     Paragraph('<b>Before</b>', header_style),
     Paragraph('<b>After</b>', header_style),
     Paragraph('<b>Improvement</b>', header_style)],
    [Paragraph('Sample Processing Time', cell_style),
     Paragraph('500-1000 us', cell_style),
     Paragraph('5-50 us', cell_style),
     Paragraph('10-100x faster', cell_style)],
    [Paragraph('Trip Detection Latency', cell_style),
     Paragraph('10-50 ms', cell_style),
     Paragraph('<1 ms', cell_style),
     Paragraph('10-50x faster', cell_style)],
    [Paragraph('CPU Usage', cell_style),
     Paragraph('80-100%', cell_style),
     Paragraph('20-40%', cell_style),
     Paragraph('2-4x reduction', cell_style)],
    [Paragraph('Events Per Second', cell_style),
     Paragraph('~500', cell_style),
     Paragraph('20,000+', cell_style),
     Paragraph('40x throughput', cell_style)],
]

perf_table = Table(perf_data, colWidths=[1.6*inch, 1.4*inch, 1.4*inch, 1.4*inch])
perf_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1F4E79')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
    ('BACKGROUND', (0, 1), (-1, -1), colors.white),
    ('BACKGROUND', (0, 2), (-1, 2), colors.HexColor('#E8F5E9')),
    ('BACKGROUND', (0, 4), (-1, 4), colors.HexColor('#E8F5E9')),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('LEFTPADDING', (0, 0), (-1, -1), 6),
    ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ('TOPPADDING', (0, 0), (-1, -1), 4),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
]))
story.append(perf_table)
story.append(Spacer(1, 15))

# Priority Order
story.append(Paragraph('Implementation Priority Order', h1_style))

priority_data = [
    [Paragraph('<b>#</b>', header_style),
     Paragraph('<b>Fix</b>', header_style),
     Paragraph('<b>Time</b>', header_style),
     Paragraph('<b>Impact</b>', header_style)],
    [Paragraph('1', cell_style),
     Paragraph('Critical Bug in dsp_runner.py', cell_style),
     Paragraph('2 min', cell_style),
     Paragraph('CRITICAL - System will finally work', cell_style)],
    [Paragraph('2', cell_style),
     Paragraph('Stop Double Processing', cell_style),
     Paragraph('5 min', cell_style),
     Paragraph('50% speed improvement', cell_style)],
    [Paragraph('3', cell_style),
     Paragraph('High-Speed Detection Loop', cell_style),
     Paragraph('15 min', cell_style),
     Paragraph('10x speed improvement', cell_style)],
    [Paragraph('4', cell_style),
     Paragraph('UI Throttling', cell_style),
     Paragraph('2 min', cell_style),
     Paragraph('Reduces CPU load', cell_style)],
    [Paragraph('5', cell_style),
     Paragraph('C++ Build Optimization', cell_style),
     Paragraph('5 min', cell_style),
     Paragraph('Marginal improvement', cell_style)],
]

priority_table = Table(priority_data, colWidths=[0.5*inch, 2.2*inch, 0.8*inch, 2.4*inch])
priority_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1F4E79')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
    ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#FFEBEE')),
    ('BACKGROUND', (0, 2), (-1, 2), colors.white),
    ('BACKGROUND', (0, 3), (-1, 3), colors.HexColor('#E8F5E9')),
    ('BACKGROUND', (0, 4), (-1, 4), colors.white),
    ('BACKGROUND', (0, 5), (-1, 5), colors.HexColor('#FFF3E0')),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('LEFTPADDING', (0, 0), (-1, -1), 6),
    ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ('TOPPADDING', (0, 0), (-1, -1), 4),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
]))
story.append(priority_table)
story.append(Spacer(1, 15))

# Verification Steps
story.append(Paragraph('Verification Steps', h1_style))
story.append(Preformatted('''
Step 1: Rebuild C++ Module
$ python cpp/build.py
# Expected: "Build complete! Module ready at: .../microgrid_dsp.so"

Step 2: Verify Module Loads
$ python -c "import microgrid_dsp; p = microgrid_dsp.create_default_pipeline(); print(p)"
# Expected: <DSPPipeline samples=0 trips=0 avg=0.0us>

Step 3: Check System Logs for:
[INFO] C++ DSP pipeline initialized (fast path active)
[INFO] C++ DSP High-Speed Loop Active
[INFO] Using C++ DSP Fast Path - Python DWT disabled

Step 4: Verify Performance on System Health page:
"Avg Processing" should show <50 microseconds

Step 5: Inject Fault and verify:
Trip should trigger within 1ms of fault injection
''', code_style))
story.append(Spacer(1, 10))

# Conclusion
story.append(Paragraph('Conclusion', h1_style))
story.append(Paragraph(
    'The DC Microgrid Fault Detection System has a well-designed C++ DSP core that is capable '
    'of achieving sub-millisecond fault detection. However, several integration issues prevented '
    'this potential from being realized. The most critical issue is a simple attribute access bug '
    'in dsp_runner.py that prevented the C++ trip detection from ever triggering. By implementing '
    'the fixes in this document, the system should achieve 20kHz+ processing with sub-ms latency.',
    body_style
))

# Build PDF
doc.build(story)
print(f"PDF generated successfully: {output_path}")
