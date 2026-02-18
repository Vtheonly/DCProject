/**
 * DC Microgrid DSP Core — Implementation
 *
 * High-performance wavelet-based fault detection for DC microgrids.
 * All functions are designed for deterministic, low-latency execution.
 */

#include "dsp_core.h"
#include <stdexcept>
#include <cstring>

namespace microgrid {

// ============================================================================
// CircularBuffer
// ============================================================================

CircularBuffer::CircularBuffer(size_t capacity)
    : buffer_(capacity, 0.0), capacity_(capacity), head_(0), count_(0) {}

void CircularBuffer::push(double value) {
    buffer_[head_] = value;
    head_ = (head_ + 1) % capacity_;
    if (count_ < capacity_) ++count_;
}

double CircularBuffer::get(size_t index_from_newest) const {
    if (index_from_newest >= count_) return 0.0;
    size_t idx = (head_ + capacity_ - 1 - index_from_newest) % capacity_;
    return buffer_[idx];
}

size_t CircularBuffer::size() const { return count_; }
bool CircularBuffer::full() const { return count_ >= capacity_; }

void CircularBuffer::clear() {
    head_ = 0;
    count_ = 0;
    std::fill(buffer_.begin(), buffer_.end(), 0.0);
}

std::vector<double> CircularBuffer::get_window(size_t count) const {
    size_t n = std::min(count, count_);
    std::vector<double> result(n);
    for (size_t i = 0; i < n; ++i) {
        // Oldest first: index n-1-i from newest
        result[n - 1 - i] = get(i);
    }
    return result;
}

std::vector<double> CircularBuffer::get_history(size_t max_count) const {
    size_t n = max_count > 0 ? std::min(max_count, count_) : count_;
    return get_window(n);
}

// ============================================================================
// ButterworthLPF — 4th-order IIR via cascaded biquads
// ============================================================================

void ButterworthLPF::design(double cutoff_hz, double sample_rate_hz) {
    // Pre-warp the cutoff frequency for bilinear transform
    double wc = 2.0 * sample_rate_hz * std::tan(M_PI * cutoff_hz / sample_rate_hz);

    // Normalized frequency
    double K = wc / (2.0 * sample_rate_hz);
    double K2 = K * K;

    // 4th-order Butterworth has 2 complex conjugate pole pairs
    // s-domain poles at angles: pi*(2k+1)/(2*4) for k=0..3
    // Section 1: poles at 22.5° and 157.5°
    // Section 2: poles at 67.5° and 112.5°

    // Section 1: Q = 1/(2*cos(pi/8)) = 1/(2*0.92388) ≈ 0.5412
    double Q1 = 1.0 / (2.0 * std::cos(M_PI / 8.0));
    // Section 2: Q = 1/(2*cos(3*pi/8)) = 1/(2*0.38268) ≈ 1.3066
    double Q2 = 1.0 / (2.0 * std::cos(3.0 * M_PI / 8.0));

    auto design_biquad = [&](Biquad& bq, double Q) {
        double norm = 1.0 + K / Q + K2;
        bq.b0 = K2 / norm;
        bq.b1 = 2.0 * K2 / norm;
        bq.b2 = K2 / norm;
        bq.a1 = 2.0 * (K2 - 1.0) / norm;
        bq.a2 = (1.0 - K / Q + K2) / norm;
        bq.z1 = bq.z2 = 0.0;
    };

    design_biquad(sections_[0], Q1);
    design_biquad(sections_[1], Q2);
    designed_ = true;
}

double ButterworthLPF::process(double input) {
    if (!designed_) return input; // Passthrough if not designed
    double y = sections_[0].process(input);
    return sections_[1].process(y);
}

void ButterworthLPF::reset() {
    sections_[0].reset();
    sections_[1].reset();
}

// ============================================================================
// LiftingDWT — Daubechies-4 via Lifting Scheme
// ============================================================================

LiftingDWT::LiftingDWT(size_t levels) : levels_(levels) {
    if (levels == 0 || levels > 10) {
        throw std::invalid_argument("DWT levels must be 1-10");
    }
}

void LiftingDWT::lifting_step(std::vector<double>& data, size_t length) {
    // Daubechies-4 lifting scheme coefficients
    // Based on factorization from Sweldens/Daubechies
    static const double sqrt3 = std::sqrt(3.0);
    static const double c1 = sqrt3;
    static const double c2 = sqrt3 / 4.0;
    static const double c3 = (sqrt3 - 2.0) / 4.0;
    static const double scale = (sqrt3 + 1.0) / (2.0 * std::sqrt(2.0));
    static const double inv_scale = (sqrt3 - 1.0) / (2.0 * std::sqrt(2.0));

    size_t half = length / 2;
    if (half == 0) return;

    // Split into even/odd (lazy wavelet)
    std::vector<double> even(half), odd(half);
    for (size_t i = 0; i < half; ++i) {
        even[i] = data[2 * i];
        odd[i] = data[2 * i + 1];
    }

    // Predict step 1: Update odd using even
    for (size_t i = 0; i < half; ++i) {
        odd[i] += c1 * even[i];
    }

    // Update step 1: Update even using odd
    for (size_t i = 0; i < half; ++i) {
        size_t prev = (i == 0) ? half - 1 : i - 1; // Periodic boundary
        even[i] += c2 * odd[i] + c3 * odd[prev];
    }

    // Predict step 2: Update odd using even
    for (size_t i = 0; i < half; ++i) {
        size_t next = (i + 1) % half; // Periodic boundary
        odd[i] -= even[next];
    }

    // Scale
    for (size_t i = 0; i < half; ++i) {
        even[i] *= scale;
        odd[i] *= inv_scale;
    }

    // Write back: approximation first, then detail
    for (size_t i = 0; i < half; ++i) {
        data[i] = even[i];          // Approximation coefficients
        data[half + i] = odd[i];    // Detail coefficients
    }
}

std::vector<std::vector<double>> LiftingDWT::transform(
    const std::vector<double>& signal
) {
    size_t min_len = static_cast<size_t>(1) << levels_;
    if (signal.size() < min_len) {
        throw std::invalid_argument(
            "Signal length must be >= 2^levels (" + std::to_string(min_len) + ")"
        );
    }

    // Work on a copy, padded to next power of 2 if needed
    size_t n = signal.size();
    size_t padded = 1;
    while (padded < n) padded <<= 1;

    std::vector<double> work(padded, 0.0);
    std::copy(signal.begin(), signal.end(), work.begin());
    // Symmetric boundary extension for padding
    for (size_t i = n; i < padded; ++i) {
        work[i] = signal[2 * n - 2 - i]; // Mirror
    }

    // Result: [A_n, D_n, D_{n-1}, ..., D_1]
    // Index 0 = approximation at deepest level
    // Index levels_ = D1 (finest detail)
    std::vector<std::vector<double>> result(levels_ + 1);

    size_t current_len = padded;
    for (size_t level = 0; level < levels_; ++level) {
        lifting_step(work, current_len);

        size_t half = current_len / 2;

        // Extract detail coefficients (D_{level+1})
        // Store as D1..D4: result[levels_ - level] = finest first mapping
        // Actually store in order: result[levels_] = D1
        result[levels_ - level].assign(work.begin() + half, work.begin() + current_len);

        current_len = half;
    }

    // Remaining is the approximation at deepest level
    result[0].assign(work.begin(), work.begin() + current_len);

    return result;
}

std::array<double, 5> LiftingDWT::compute_energy(
    const std::vector<std::vector<double>>& coefficients
) const {
    std::array<double, 5> energy = {0, 0, 0, 0, 0};

    // coefficients layout: [A4, D4, D3, D2, D1]
    // energy layout: [D1, D2, D3, D4, A4]

    auto calc_energy = [](const std::vector<double>& v) -> double {
        double sum = 0.0;
        for (double x : v) sum += x * x;
        return v.empty() ? 0.0 : sum / v.size();
    };

    // D1 = coefficients[levels_] (finest)
    if (coefficients.size() > levels_) {
        energy[0] = calc_energy(coefficients[levels_]);     // D1
    }
    if (coefficients.size() > levels_ - 1 && levels_ >= 2) {
        energy[1] = calc_energy(coefficients[levels_ - 1]); // D2
    }
    if (coefficients.size() > levels_ - 2 && levels_ >= 3) {
        energy[2] = calc_energy(coefficients[levels_ - 2]); // D3
    }
    if (coefficients.size() > levels_ - 3 && levels_ >= 4) {
        energy[3] = calc_energy(coefficients[levels_ - 3]); // D4
    }
    if (!coefficients.empty()) {
        energy[4] = calc_energy(coefficients[0]);            // A4
    }

    return energy;
}

double LiftingDWT::find_d1_peak(
    const std::vector<std::vector<double>>& coefficients
) const {
    if (coefficients.size() <= levels_) return 0.0;

    const auto& d1 = coefficients[levels_];
    double peak = 0.0;
    for (double v : d1) {
        double abs_v = std::abs(v);
        if (abs_v > peak) peak = abs_v;
    }
    return peak;
}

// ============================================================================
// FastTripCheck
// ============================================================================

FastTripCheck::FastTripCheck(const ThresholdConfig& config) : config_(config) {}

void FastTripCheck::update_config(const ThresholdConfig& config) {
    config_ = config;
}

TripResult FastTripCheck::check(double d1_energy, double d1_peak) {
    TripResult result;
    auto now = std::chrono::high_resolution_clock::now();
    result.timestamp_us = std::chrono::duration_cast<std::chrono::microseconds>(
        now.time_since_epoch()
    ).count();

    result.d1_energy = d1_energy;
    result.d1_peak = d1_peak;
    result.threshold_used = config_.d1_energy_trip;

    // Skip noise floor
    if (d1_energy < config_.noise_floor && d1_peak < config_.noise_floor) {
        return result;
    }

    // Fast trip: OR logic — either energy or peak exceeds threshold
    if (d1_energy > config_.d1_energy_trip || d1_peak > config_.d1_peak_trip) {
        result.triggered = true;
    }

    return result;
}

// ============================================================================
// DSPPipeline
// ============================================================================

DSPPipeline::DSPPipeline()
    : DSPPipeline(DSPConfig{}, ThresholdConfig{}) {}

DSPPipeline::DSPPipeline(const DSPConfig& dsp_cfg, const ThresholdConfig& thresh_cfg)
    : dsp_cfg_(dsp_cfg),
      thresh_cfg_(thresh_cfg),
      sample_buffer_(dsp_cfg.history_capacity),
      dwt_(dsp_cfg.dwt_levels),
      trip_check_(thresh_cfg)
{
    filter_.design(dsp_cfg_.filter_cutoff_hz, dsp_cfg_.sample_rate_hz);
    energy_history_.reserve(energy_history_max_);

    // For a rolling DWT, stride=1 means DWT every sample (most responsive).
    // stride=window_size/4 is a good balance for performance vs latency.
    dwt_stride_ = std::max<size_t>(1, dsp_cfg_.window_size / 4);
}

DSPResult DSPPipeline::process_sample(double raw_voltage) {
    auto start = std::chrono::high_resolution_clock::now();

    DSPResult result;
    result.sample_count = ++sample_count_;

    // Step 1: Anti-aliasing filter
    double filtered = filter_.process(raw_voltage);
    result.filtered_value = filtered;

    // Step 2: Store in circular buffer
    sample_buffer_.push(filtered);

    // Step 3: Check if we have enough samples for DWT
    if (sample_buffer_.size() < dsp_cfg_.window_size) {
        result.window_ready = false;
        auto end = std::chrono::high_resolution_clock::now();
        result.processing_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
            end - start).count();
        return result;
    }

    result.window_ready = true;

    // Step 4: Rolling DWT — run every 'stride' samples
    ++stride_counter_;
    if (stride_counter_ >= dwt_stride_) {
        stride_counter_ = 0;

        // Get the current window
        auto window = sample_buffer_.get_window(dsp_cfg_.window_size);

        // Run DWT
        {
            std::lock_guard<std::mutex> lock(mutex_);
            last_coefficients_ = dwt_.transform(window);
        }

        // Step 5: Compute energy
        result.energy_levels = dwt_.compute_energy(last_coefficients_);
        result.d1_peak = dwt_.find_d1_peak(last_coefficients_);

        result.total_energy = 0;
        for (int i = 0; i < 4; ++i) result.total_energy += result.energy_levels[i];
        result.high_freq_ratio = (result.total_energy > 0.001)
            ? result.energy_levels[0] / result.total_energy : 0.0;

        // Store energy history for visualization
        {
            std::lock_guard<std::mutex> lock(mutex_);
            energy_history_.push_back(result.energy_levels);
            if (energy_history_.size() > energy_history_max_) {
                energy_history_.erase(energy_history_.begin());
            }
        }

        // Step 6: Fast trip check
        result.trip = trip_check_.check(result.energy_levels[0], result.d1_peak);
        if (result.trip.triggered) {
            ++trip_count_;
        }
    } else {
        // Between DWT runs, carry forward last energy
        std::lock_guard<std::mutex> lock(mutex_);
        if (!energy_history_.empty()) {
            result.energy_levels = energy_history_.back();
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.processing_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
        end - start).count();
    total_processing_us_ += result.processing_time_us;

    return result;
}

std::vector<std::vector<double>> DSPPipeline::get_coefficients() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_coefficients_;
}

std::vector<double> DSPPipeline::get_voltage_history(size_t max_count) const {
    return sample_buffer_.get_history(max_count);
}

std::vector<std::array<double, 5>> DSPPipeline::get_energy_history(size_t max_count) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (max_count == 0 || max_count >= energy_history_.size()) {
        return energy_history_;
    }
    return std::vector<std::array<double, 5>>(
        energy_history_.end() - max_count, energy_history_.end()
    );
}

void DSPPipeline::update_dsp_config(const DSPConfig& cfg) {
    dsp_cfg_ = cfg;
    filter_.design(cfg.filter_cutoff_hz, cfg.sample_rate_hz);
    dwt_ = LiftingDWT(cfg.dwt_levels);
    dwt_stride_ = std::max<size_t>(1, cfg.window_size / 4);
}

void DSPPipeline::update_threshold_config(const ThresholdConfig& cfg) {
    thresh_cfg_ = cfg;
    trip_check_.update_config(cfg);
}

void DSPPipeline::reset() {
    filter_.reset();
    sample_buffer_.clear();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        last_coefficients_.clear();
        energy_history_.clear();
    }
    sample_count_ = 0;
    trip_count_ = 0;
    total_processing_us_ = 0;
    stride_counter_ = 0;
}

double DSPPipeline::avg_processing_us() const {
    return sample_count_ > 0 ? total_processing_us_ / sample_count_ : 0.0;
}

} // namespace microgrid
