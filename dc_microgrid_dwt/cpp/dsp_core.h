/**
 * DC Microgrid DSP Core — High-Performance Wavelet Fault Detection
 *
 * This module implements the time-critical DSP pipeline in C++ for
 * integration with the Python-based monitoring platform via PyBind11.
 *
 * Components:
 *   - CircularBuffer: Lock-free ring buffer for ADC samples
 *   - ButterworthLPF: 4th-order IIR anti-aliasing filter
 *   - LiftingDWT: Daubechies-4 wavelet via lifting scheme
 *   - FastTripCheck: Sub-microsecond threshold comparator
 *   - DSPPipeline: Orchestrates the full sample→trip path
 *
 * Performance targets:
 *   - Process one sample in <50μs
 *   - Trip decision in <1μs after DWT completion
 *   - Sustain 20kHz+ sample rates
 */

#pragma once

#include <vector>
#include <array>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <numeric>
#include <string>
#include <mutex>
#include <atomic>
#include <chrono>

namespace microgrid {

// ============================================================================
// Configuration
// ============================================================================

struct DSPConfig {
    size_t window_size = 128;       // DWT window size (power of 2)
    size_t dwt_levels = 4;         // Decomposition levels
    double sample_rate_hz = 20000; // ADC sample rate
    double filter_cutoff_hz = 8000; // Anti-aliasing cutoff (< Nyquist/2)
    size_t history_capacity = 4096; // Max samples to retain for viz
};

struct ThresholdConfig {
    double d1_energy_trip = 100.0;  // D1 energy threshold for fast trip
    double d1_peak_trip = 50.0;     // D1 peak threshold for fast trip
    double d2_energy_warn = 50.0;   // D2 energy warning level
    bool adaptive = false;          // Enable adaptive thresholds (future)
    double noise_floor = 0.5;       // Below this, ignore as noise
};

// ============================================================================
// Results
// ============================================================================

struct TripResult {
    bool triggered = false;
    double d1_energy = 0.0;
    double d1_peak = 0.0;
    double threshold_used = 0.0;
    int64_t timestamp_us = 0;   // Microsecond timestamp
};

struct DSPResult {
    // DWT energy per level: D1, D2, D3, D4, A4
    std::array<double, 5> energy_levels = {0, 0, 0, 0, 0};
    double d1_peak = 0.0;
    double total_energy = 0.0;
    double high_freq_ratio = 0.0; // D1 / total_detail

    // Trip decision
    TripResult trip;

    // Filtered sample value
    double filtered_value = 0.0;

    // Processing time in microseconds
    int64_t processing_time_us = 0;

    // Number of samples processed so far
    int64_t sample_count = 0;

    // Whether we have a full window yet
    bool window_ready = false;
};

// ============================================================================
// CircularBuffer — Lock-free ring buffer for sample storage
// ============================================================================

class CircularBuffer {
public:
    explicit CircularBuffer(size_t capacity = 4096);

    void push(double value);
    double get(size_t index_from_newest) const; // 0 = newest
    size_t size() const;
    bool full() const;
    void clear();

    // Get contiguous window of N most recent samples
    std::vector<double> get_window(size_t count) const;

    // Get entire history for visualization
    std::vector<double> get_history(size_t max_count = 0) const;

private:
    std::vector<double> buffer_;
    size_t capacity_;
    size_t head_ = 0; // Next write position
    size_t count_ = 0;
};

// ============================================================================
// ButterworthLPF — 4th-order IIR anti-aliasing filter
// ============================================================================

class ButterworthLPF {
public:
    ButterworthLPF() = default;

    /**
     * Design a 4th-order Butterworth low-pass filter.
     * @param cutoff_hz  Cutoff frequency in Hz
     * @param sample_rate_hz  Sample rate in Hz
     */
    void design(double cutoff_hz, double sample_rate_hz);

    /**
     * Filter one sample (direct-form II transposed).
     * Call this once per ADC sample.
     */
    double process(double input);

    void reset();

private:
    // 4th-order = two cascaded biquad sections
    struct Biquad {
        double b0 = 1, b1 = 0, b2 = 0;
        double a1 = 0, a2 = 0;
        double z1 = 0, z2 = 0; // State variables

        double process(double x) {
            double y = b0 * x + z1;
            z1 = b1 * x - a1 * y + z2;
            z2 = b2 * x - a2 * y;
            return y;
        }

        void reset() { z1 = z2 = 0; }
    };

    Biquad sections_[2]; // Two biquads = 4th order
    bool designed_ = false;
};

// ============================================================================
// LiftingDWT — Daubechies-4 Wavelet via Lifting Scheme
// ============================================================================

class LiftingDWT {
public:
    explicit LiftingDWT(size_t levels = 4);

    /**
     * Perform DWT on input signal using Daubechies-4 lifting.
     * @param signal  Input signal (length must be >= 2^levels)
     * @return Coefficient arrays: [A_n, D_n, D_{n-1}, ..., D_1]
     *         i.e. result[0] = approximation, result[levels] = D1 (finest detail)
     */
    std::vector<std::vector<double>> transform(const std::vector<double>& signal);

    /**
     * Compute energy for each decomposition level.
     * @return Array of energies: [D1, D2, D3, D4, A4]
     */
    std::array<double, 5> compute_energy(
        const std::vector<std::vector<double>>& coefficients) const;

    /**
     * Find peak absolute value in D1 coefficients.
     */
    double find_d1_peak(const std::vector<std::vector<double>>& coefficients) const;

    size_t levels() const { return levels_; }

private:
    size_t levels_;

    // One level of Daubechies-4 lifting
    void lifting_step(std::vector<double>& data, size_t length);
};

// ============================================================================
// FastTripCheck — Sub-microsecond threshold comparator
// ============================================================================

class FastTripCheck {
public:
    explicit FastTripCheck(const ThresholdConfig& config = {});

    void update_config(const ThresholdConfig& config);

    /**
     * Check if fault threshold is exceeded.
     * Must complete in <1μs.
     */
    TripResult check(double d1_energy, double d1_peak);

    const ThresholdConfig& config() const { return config_; }

private:
    ThresholdConfig config_;
};

// ============================================================================
// DSPPipeline — Full sample→trip orchestration
// ============================================================================

class DSPPipeline {
public:
    DSPPipeline();
    explicit DSPPipeline(const DSPConfig& dsp_cfg, const ThresholdConfig& thresh_cfg);

    /**
     * Process one ADC sample through the full pipeline:
     *   1. Anti-aliasing filter
     *   2. Store in circular buffer
     *   3. If window is full, run DWT
     *   4. Compute energy levels
     *   5. Check trip thresholds
     *
     * @param raw_voltage  Raw ADC voltage sample
     * @return DSPResult with all computed values and trip decision
     */
    DSPResult process_sample(double raw_voltage);

    // --- Getters for Python visualization ---

    /** Get D1-D4 and A4 coefficient arrays from last DWT. */
    std::vector<std::vector<double>> get_coefficients() const;

    /** Get filtered voltage history for plotting. */
    std::vector<double> get_voltage_history(size_t max_count = 0) const;

    /** Get energy level history (each entry is [D1,D2,D3,D4,A4]). */
    std::vector<std::array<double, 5>> get_energy_history(size_t max_count = 0) const;

    // --- Configuration ---

    void update_dsp_config(const DSPConfig& cfg);
    void update_threshold_config(const ThresholdConfig& cfg);
    void reset();

    // --- Stats ---
    int64_t total_samples() const { return sample_count_; }
    int64_t total_trips() const { return trip_count_; }
    double avg_processing_us() const;

private:
    DSPConfig dsp_cfg_;
    ThresholdConfig thresh_cfg_;

    ButterworthLPF filter_;
    CircularBuffer sample_buffer_;
    LiftingDWT dwt_;
    FastTripCheck trip_check_;

    // Cached results
    std::vector<std::vector<double>> last_coefficients_;
    std::vector<std::array<double, 5>> energy_history_;
    size_t energy_history_max_ = 500;

    // Counters
    int64_t sample_count_ = 0;
    int64_t trip_count_ = 0;
    double total_processing_us_ = 0;

    // Sliding window counter (process DWT every N samples for rolling)
    size_t dwt_stride_ = 1; // Process every sample for true rolling
    size_t stride_counter_ = 0;

    mutable std::mutex mutex_;
};

} // namespace microgrid
