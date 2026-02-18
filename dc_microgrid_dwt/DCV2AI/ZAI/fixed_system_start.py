"""
System Control Module - FIXED start_system() FUNCTION

This file contains the corrected start_system() function.
Apply these changes to: src/ui/system.py

KEY FIXES:
1. Conditional agent creation - only create Python DWT agents if C++ is NOT available
2. Integration of HighSpeedDetectionLoop
3. Proper cleanup in stop_system()
"""

# Add this import at the top of system.py:
# from src.adapters.high_speed_loop import HighSpeedDetectionLoop


def start_system():
    """Start the complete fault detection system.
    
    FIXED VERSION - Only creates Python DWT chain if C++ is not available.
    """
    if st.session_state.system_running:
        add_log("System already running", "WARNING")
        return

    add_log("Starting DC Microgrid Fault Detection System...", "INFO")

    try:
        # 1. Create EventBus and infrastructure
        bus = EventBus()
        obs = Observability()
        registry = AgentRegistry()

        # 2. Create GridEmulator and load circuit
        emulator = GridEmulator(sample_rate=20000, base_voltage=400.0)

        circuit = st.session_state.circuit_model
        if circuit is None:
            circuit = create_reference_circuit()
            st.session_state.circuit_model = circuit
            add_log("Loaded reference 6-bus microgrid circuit", "INFO")

        emulator.load_circuit(circuit)
        add_log(f"Circuit loaded: {len(circuit.buses)} buses, {len(circuit.lines)} lines", "INFO")

        # 3. Create sensor (reads from emulator)
        sensor = SimulatedADCSensor(emulator)
        relay = SimulatedRelayDriver()

        # 3a. Initialize C++ DSP pipeline EARLY
        dsp_pipeline = None
        high_speed_loop = None
        
        if DSP_AVAILABLE:
            try:
                dsp_pipeline = microgrid_dsp.create_pipeline(
                    window_size=128, levels=4,
                    sample_rate=20000.0, cutoff=8000.0,
                    trip_threshold=100.0
                )
                st.session_state.dsp_pipeline = dsp_pipeline
                st.session_state.dsp_available = True
                add_log("C++ DSP pipeline initialized", "INFO")
                
                # Log module location and performance
                add_log(f"C++ DSP Module: {microgrid_dsp.__file__}", "INFO")
                
            except Exception as e:
                add_log(f"C++ DSP init failed: {e}", "WARNING")
                st.session_state.dsp_available = False

        # 4. Create agents - CONDITIONAL on DSP availability
        sampler = SamplerAgent("Sampler", bus, config={"sample_rate": 20000})
        sampler.set_sensor(sensor)

        # --- CRITICAL FIX: Only create Python DWT agents if C++ is NOT available ---
        window_mgr = None
        dwt_engine = None
        detail_analyzer = None
        dsp_runner = None
        
        if dsp_pipeline:
            # C++ Path - create DSP runner, disable Python DWT chain
            dsp_runner = DSPRunnerAgent("DSPRunner", bus, config={"dsp_pipeline": dsp_pipeline})
            add_log("Using C++ DSP Fast Path - Python DWT disabled", "INFO")
            
            # Create High-Speed Loop for deterministic timing
            try:
                high_speed_loop = HighSpeedDetectionLoop(
                    sensor, dsp_pipeline, bus, sample_rate=20000
                )
                high_speed_loop.start()
                add_log("C++ DSP High-Speed Loop Active", "INFO")
            except Exception as e:
                add_log(f"High-Speed Loop failed: {e}", "WARNING")
                
        else:
            # Python Fallback Path - create full Python DWT chain
            window_mgr = WindowManagerAgent("WindowManager", bus, config={"window_size": 128})
            dwt_engine = DWTEngineAgent("DWTEngine", bus, config={
                "wavelet": "db4", "level": 4, "mode": "symmetric"
            })
            detail_analyzer = DetailAnalyzerAgent("DetailAnalyzer", bus)
            add_log("Using Python DSP Fallback", "WARNING")

        # Create remaining agents
        fault_locator = PreciseFaultLocatorAgent("FaultLocator", bus)
        fault_locator.emulator = emulator

        threshold_guard = ThresholdGuardAgent("ThresholdGuard", bus, config={
            "d1_peak_threshold": 50.0
        })
        energy_monitor = EnergyMonitorAgent("EnergyMonitor", bus)
        fault_voter = FaultVoterAgent("FaultVoter", bus)

        trip_sequencer = TripSequencerAgent("TripSequencer", bus, config={"relay_driver": relay})
        zeta_logic = ZetaLogicAgent("ZetaLogic", bus)

        health_monitor = HealthMonitorAgent("HealthMonitor", bus, config={"check_interval": 2.0})
        ai_classifier = AIClassifierAgent("AIClassifier", bus)
        replay_recorder = ReplayRecorderAgent("ReplayRecorder", bus)
        report_generator = ReportGeneratorAgent("ReportGenerator", bus)

        # UI bridge agent (downsample 50x to 400Hz)
        bridge = BridgeAgent("UIBridge", bus, config={"downsample_factor": 50})

        # 6. Register agents - CONDITIONALLY
        agents = [sampler]
        
        # Only add Python DWT chain if C++ is NOT used
        if not dsp_pipeline:
            agents.extend([window_mgr, dwt_engine, detail_analyzer])
        
        agents.extend([
            fault_locator, threshold_guard, energy_monitor, fault_voter,
            trip_sequencer, zeta_logic,
            health_monitor, ai_classifier, replay_recorder, report_generator,
            bridge
        ])
        
        # Add DSP runner if C++ is available
        if dsp_runner:
            agents.append(dsp_runner)
        
        for agent in agents:
            registry.register(agent)

        # 7. Start all agents
        registry.start_all()
        add_log(f"Started {len(agents)} agents", "INFO")

        # 8. Subscribe UI bridge to additional events
        bus.subscribe(DWTResultEvent, lambda e: _on_dwt_result(e, bridge.get_queue()))
        bus.subscribe(FaultLocationEvent, lambda e: bridge.get_queue().put(e))
        bus.subscribe(HealthStatusEvent, lambda e: bridge.get_queue().put(e))
        bus.subscribe(AIAnalysisEvent, lambda e: bridge.get_queue().put(e))

        # 9. Start emulator simulation thread
        emulator.start()
        add_log("Grid emulator started", "INFO")

        # 10. Store references in session state
        st.session_state.event_bus = bus
        st.session_state.registry = registry
        st.session_state.emulator = emulator
        st.session_state.bridge_agent = bridge
        st.session_state.system_running = True
        st.session_state.high_speed_loop = high_speed_loop

        add_log("System started successfully!", "INFO")

    except Exception as e:
        add_log(f"Failed to start system: {e}", "ERROR")
        logger.exception("System start failed")


def stop_system():
    """Stop the fault detection system - FIXED with HighSpeedLoop cleanup."""
    if not st.session_state.system_running:
        return

    add_log("Stopping system...", "INFO")

    try:
        # Stop high-speed loop FIRST
        if st.session_state.get("high_speed_loop"):
            st.session_state.high_speed_loop.stop()
            st.session_state.high_speed_loop = None
            
        if st.session_state.registry:
            st.session_state.registry.stop_all()
        if st.session_state.emulator:
            st.session_state.emulator.stop()

        st.session_state.system_running = False
        add_log("System stopped", "INFO")
    except Exception as e:
        add_log(f"Error during shutdown: {e}", "ERROR")
