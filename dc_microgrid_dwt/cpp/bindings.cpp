/**
 * PyBind11 Bindings for DC Microgrid DSP Core
 *
 * Exposes the C++ DSP pipeline to Python as the 'microgrid_dsp' module.
 * This allows Streamlit and other Python code to call the fast DSP
 * engine while keeping the UI layer in Python.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "dsp_core.h"

namespace py = pybind11;

PYBIND11_MODULE(microgrid_dsp, m) {
    m.doc() = R"doc(
        DC Microgrid DSP Core — High-Performance Wavelet Fault Detection

        This module provides a C++ implementation of the critical DSP path:
        anti-aliasing filter → circular buffer → DWT → threshold trip.

        Usage:
            import microgrid_dsp as dsp

            pipeline = dsp.DSPPipeline(
                dsp.DSPConfig(window_size=128, sample_rate_hz=20000),
                dsp.ThresholdConfig(d1_energy_trip=100.0)
            )

            result = pipeline.process_sample(380.5)
            if result.trip.triggered:
                print("FAULT DETECTED!")
    )doc";

    // --- DSPConfig ---
    py::class_<microgrid::DSPConfig>(m, "DSPConfig",
        "Configuration for the DSP pipeline.")
        .def(py::init<>())
        .def(py::init([](size_t ws, size_t dl, double sr, double fc, size_t hc) {
            microgrid::DSPConfig cfg;
            cfg.window_size = ws;
            cfg.dwt_levels = dl;
            cfg.sample_rate_hz = sr;
            cfg.filter_cutoff_hz = fc;
            cfg.history_capacity = hc;
            return cfg;
        }),
            py::arg("window_size") = 128,
            py::arg("dwt_levels") = 4,
            py::arg("sample_rate_hz") = 20000.0,
            py::arg("filter_cutoff_hz") = 8000.0,
            py::arg("history_capacity") = 4096
        )
        .def_readwrite("window_size", &microgrid::DSPConfig::window_size)
        .def_readwrite("dwt_levels", &microgrid::DSPConfig::dwt_levels)
        .def_readwrite("sample_rate_hz", &microgrid::DSPConfig::sample_rate_hz)
        .def_readwrite("filter_cutoff_hz", &microgrid::DSPConfig::filter_cutoff_hz)
        .def_readwrite("history_capacity", &microgrid::DSPConfig::history_capacity)
        .def("__repr__", [](const microgrid::DSPConfig& cfg) {
            return "<DSPConfig window=" + std::to_string(cfg.window_size)
                + " levels=" + std::to_string(cfg.dwt_levels)
                + " rate=" + std::to_string(static_cast<int>(cfg.sample_rate_hz)) + "Hz>";
        });

    // --- ThresholdConfig ---
    py::class_<microgrid::ThresholdConfig>(m, "ThresholdConfig",
        "Configuration for fast trip thresholds.")
        .def(py::init<>())
        .def(py::init([](double d1e, double d1p, double d2e, bool adp, double nf) {
            microgrid::ThresholdConfig cfg;
            cfg.d1_energy_trip = d1e;
            cfg.d1_peak_trip = d1p;
            cfg.d2_energy_warn = d2e;
            cfg.adaptive = adp;
            cfg.noise_floor = nf;
            return cfg;
        }),
            py::arg("d1_energy_trip") = 100.0,
            py::arg("d1_peak_trip") = 50.0,
            py::arg("d2_energy_warn") = 50.0,
            py::arg("adaptive") = false,
            py::arg("noise_floor") = 0.5
        )
        .def_readwrite("d1_energy_trip", &microgrid::ThresholdConfig::d1_energy_trip)
        .def_readwrite("d1_peak_trip", &microgrid::ThresholdConfig::d1_peak_trip)
        .def_readwrite("d2_energy_warn", &microgrid::ThresholdConfig::d2_energy_warn)
        .def_readwrite("adaptive", &microgrid::ThresholdConfig::adaptive)
        .def_readwrite("noise_floor", &microgrid::ThresholdConfig::noise_floor);

    // --- TripResult ---
    py::class_<microgrid::TripResult>(m, "TripResult",
        "Result from fast trip threshold check.")
        .def(py::init<>())
        .def_readonly("triggered", &microgrid::TripResult::triggered)
        .def_readonly("d1_energy", &microgrid::TripResult::d1_energy)
        .def_readonly("d1_peak", &microgrid::TripResult::d1_peak)
        .def_readonly("threshold_used", &microgrid::TripResult::threshold_used)
        .def_readonly("timestamp_us", &microgrid::TripResult::timestamp_us)
        .def("__repr__", [](const microgrid::TripResult& r) {
            return "<TripResult triggered=" + std::string(r.triggered ? "True" : "False")
                + " d1_energy=" + std::to_string(r.d1_energy) + ">";
        });

    // --- DSPResult ---
    py::class_<microgrid::DSPResult>(m, "DSPResult",
        "Complete result from processing one sample through the DSP pipeline.")
        .def(py::init<>())
        .def_property_readonly("energy_levels", [](const microgrid::DSPResult& r) {
            py::list result;
            for (auto v : r.energy_levels) result.append(v);
            return result;
        })
        .def_readonly("d1_peak", &microgrid::DSPResult::d1_peak)
        .def_readonly("total_energy", &microgrid::DSPResult::total_energy)
        .def_readonly("high_freq_ratio", &microgrid::DSPResult::high_freq_ratio)
        .def_property_readonly("trip", [](const microgrid::DSPResult& r) {
            return r.trip;
        }, py::return_value_policy::copy)
        .def_readonly("filtered_value", &microgrid::DSPResult::filtered_value)
        .def_readonly("processing_time_us", &microgrid::DSPResult::processing_time_us)
        .def_readonly("sample_count", &microgrid::DSPResult::sample_count)
        .def_readonly("window_ready", &microgrid::DSPResult::window_ready)
        .def("energy_dict", [](const microgrid::DSPResult& r) {
            py::dict d;
            d["D1"] = r.energy_levels[0];
            d["D2"] = r.energy_levels[1];
            d["D3"] = r.energy_levels[2];
            d["D4"] = r.energy_levels[3];
            d["A4"] = r.energy_levels[4];
            return d;
        }, "Get energy levels as a Python dict with keys D1, D2, D3, D4, A4.")
        .def("__repr__", [](const microgrid::DSPResult& r) {
            return "<DSPResult sample=" + std::to_string(r.sample_count)
                + " d1_energy=" + std::to_string(r.energy_levels[0])
                + " trip=" + std::string(r.trip.triggered ? "YES" : "no")
                + " time=" + std::to_string(r.processing_time_us) + "μs>";
        });

    // --- DSPPipeline (non-copyable due to std::mutex — use unique_ptr holder) ---
    py::class_<microgrid::DSPPipeline, std::unique_ptr<microgrid::DSPPipeline>>(m, "DSPPipeline",
        R"doc(
        High-performance DSP pipeline for wavelet-based fault detection.

        Orchestrates: Anti-aliasing filter → Circular buffer → Rolling DWT
        → Energy computation → Fast trip check.

        All computations run in C++ at native speed.
        )doc")
        .def(py::init<>(), "Create pipeline with default configuration.")
        .def(py::init<const microgrid::DSPConfig&, const microgrid::ThresholdConfig&>(),
            py::arg("dsp_config"), py::arg("threshold_config"),
            "Create pipeline with custom configuration.")

        .def("process_sample", &microgrid::DSPPipeline::process_sample,
            py::arg("raw_voltage"),
            R"doc(
            Process one ADC voltage sample through the full pipeline.

            Returns DSPResult with energy levels, trip decision, and timing.
            Call this at your sample rate (e.g., 20kHz).
            )doc")

        .def("get_coefficients", &microgrid::DSPPipeline::get_coefficients,
            "Get DWT coefficient arrays from last transform [A4, D4, D3, D2, D1].")

        .def("get_voltage_history", &microgrid::DSPPipeline::get_voltage_history,
            py::arg("max_count") = 0,
            "Get filtered voltage history for plotting.")

        .def("get_energy_history", [](const microgrid::DSPPipeline& p, size_t max_count) {
            auto hist = p.get_energy_history(max_count);
            // Convert to list of dicts for easy Python consumption
            py::list result;
            for (const auto& entry : hist) {
                py::dict d;
                d["D1"] = entry[0];
                d["D2"] = entry[1];
                d["D3"] = entry[2];
                d["D4"] = entry[3];
                d["A4"] = entry[4];
                result.append(d);
            }
            return result;
        },
            py::arg("max_count") = 0,
            "Get energy level history as list of dicts.")

        .def("update_dsp_config", &microgrid::DSPPipeline::update_dsp_config,
            py::arg("config"), "Update DSP configuration (window size, levels, etc).")

        .def("update_threshold_config", &microgrid::DSPPipeline::update_threshold_config,
            py::arg("config"), "Update trip thresholds.")

        .def("reset", &microgrid::DSPPipeline::reset,
            "Reset all buffers, coefficients, and counters.")

        .def_property_readonly("total_samples", &microgrid::DSPPipeline::total_samples)
        .def_property_readonly("total_trips", &microgrid::DSPPipeline::total_trips)
        .def_property_readonly("avg_processing_us", &microgrid::DSPPipeline::avg_processing_us)

        .def("__repr__", [](const microgrid::DSPPipeline& p) {
            return "<DSPPipeline samples=" + std::to_string(p.total_samples())
                + " trips=" + std::to_string(p.total_trips())
                + " avg=" + std::to_string(p.avg_processing_us()) + "μs>";
        });

    // --- Module-level convenience ---
    m.def("create_default_pipeline", []() {
        return std::make_unique<microgrid::DSPPipeline>();
    }, "Create a DSP pipeline with default settings.");

    m.def("create_pipeline", [](
        size_t window_size, size_t levels, double sample_rate,
        double cutoff, double trip_threshold
    ) {
        microgrid::DSPConfig dsp_cfg;
        dsp_cfg.window_size = window_size;
        dsp_cfg.dwt_levels = levels;
        dsp_cfg.sample_rate_hz = sample_rate;
        dsp_cfg.filter_cutoff_hz = cutoff;

        microgrid::ThresholdConfig thresh_cfg;
        thresh_cfg.d1_energy_trip = trip_threshold;

        return std::make_unique<microgrid::DSPPipeline>(dsp_cfg, thresh_cfg);
    },
        py::arg("window_size") = 128,
        py::arg("levels") = 4,
        py::arg("sample_rate") = 20000.0,
        py::arg("cutoff") = 8000.0,
        py::arg("trip_threshold") = 100.0,
        "Create a DSP pipeline with common parameters."
    );
}
