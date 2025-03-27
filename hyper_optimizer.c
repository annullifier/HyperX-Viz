/**
 * @file hyperx_optimizer.c
 * @brief Finds optimal HyperX network configurations using a parallel, pruned exhaustive search.
 *
 * This program searches for the HyperX topology with the minimum number of switches
 * that satisfies user-defined constraints on:
 * - Target number of hosts (N)
 * - Switch radix (R)
 * - Minimum required bisection bandwidth fraction (B)
 * - Minimum dimension size (min_dim_size)
 * - Optional constraint for dimension sizes to be powers of 2.
 *
 * The search explores different numbers of dimensions (L) and, for each L,
 * explores possible dimension shapes (S = [S1, S2, ..., SL]) and
 * trunking factors (K = [K1, K2, ..., KL]).
 *
 * It uses a branch-and-bound approach (pruning) to eliminate suboptimal
 * parts of the search space early, making the search feasible.
 * The search up to L=5 is performed in parallel using pthreads.
 *
 * Based on concepts from "HyperX: Topology, Routing, and Packaging of Efficient
 * Large-Scale Networks" by Ahn et al., SC09[cite: 1].
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>
#include <limits.h>
#include <unistd.h>
#include <signal.h>
#include <pthread.h>
#include <sys/sysctl.h> // For macOS core count (with Linux fallback)
#include <stdarg.h>
#include <stdint.h>     // For uintptr_t if needed, good practice

// --- Configuration Constants ---

/** @brief Maximum number of dimensions (L) to consider in the topology. */
#define MAX_DIMENSIONS 6
/** @brief Maximum size allowed for any single dimension (Sk). */
#define MAX_DIM_SIZE 64
/** @brief Maximum trunking factor (Kk) to consider for any dimension. */
#define MAX_TRUNKING 16
/** @brief Maximum number of worker threads allowed. */
#define MAX_THREADS 32
/** @brief Log file name. */
#define LOG_FILE "hyperx_parallel.log"
/** @brief Interval (in seconds) for updating progress on the console. */
#define PROGRESS_INTERVAL 1
/** @brief Macro for minimum of two values. */
#define min(a,b) ((a) < (b) ? (a) : (b))
/** @brief Macro for maximum of two values. */
#define max(a,b) ((a) > (b) ? (a) : (b))

// --- Data Structures ---

/**
 * @brief Represents a specific HyperX network configuration and its properties.
 */
typedef struct {
    int dimensions;                 /**< Number of dimensions (L). */
    int shape[MAX_DIMENSIONS];      /**< Array of sizes for each dimension (S = [S1, ..., SL]). */
    int trunking[MAX_DIMENSIONS];   /**< Array of trunking factors for each dimension (K = [K1, ..., KL]). */
    int terminals;                  /**< Number of terminals connected per switch (T). */
    int available_terminal_ports;   /**< Number of switch ports available for terminals (radix - inter-switch ports). */
    int total_switches;             /**< Total number of switches (P = product of shape S). */
    int total_terminals;            /**< Total terminals in the network (N_actual = T * P). */
    double actual_bisection;        /**< Achieved relative bisection bandwidth (beta = min(Kk*Sk) / 2T). */
    int network_diameter;           /**< Max shortest path hops between any two switches (sum of (Sk > 1 ? 1 : 0)). */
    double avg_path_length;         /**< Average path length between switches (approx). */
    double port_utilization;        /**< Percentage of used ports on a switch ((inter-switch + T) / radix). */
} HyperXConfig;

/**
 * @brief Holds data required by each worker thread.
 */
typedef struct {
    int thread_id;              /**< Unique ID for the worker thread (0 to num_threads-1). */
    int start_idx;              /**< Start index in dim_sizes for the first dimension (S1) this thread explores. */
    int end_idx;                /**< End index (exclusive) in dim_sizes for the first dimension (S1). */
    const int *dim_sizes;       /**< Pointer to the array of possible dimension sizes for the current L. */
    int num_dim_sizes;          /**< Number of elements in dim_sizes array. */
    int L;                      /**< The dimension count (L) this thread is working on. */
    int min_dim_size;           /**< Minimum allowed size for any dimension Sk. */
    int hosts;                  /**< Target minimum number of hosts (N) required. */
    int radix;                  /**< Radix (R) of the switches. */
    double bisection;           /**< Target minimum relative bisection bandwidth (B). */
    HyperXConfig *best_config;  /**< Pointer to the shared global best configuration found so far. */
    pthread_mutex_t *best_config_mutex; /**< Mutex to protect access to the shared best_config. */
    volatile long configs_processed;    /**< Counter for configurations fully evaluated by this thread. `volatile` because accessed by monitor thread. */
    volatile long configs_pruned;       /**< Counter for configuration branches pruned by this thread. `volatile` because accessed by monitor thread. */
    volatile bool is_running;           /**< Flag indicating if the thread is actively working. `volatile` because accessed by monitor thread. */
} ThreadData;

// --- Global Variables ---

/** @brief Mutex to protect access to the log file. */
pthread_mutex_t log_mutex = PTHREAD_MUTEX_INITIALIZER;
/** @brief File pointer for the log file. */
FILE* log_file = NULL;

/** @brief Mutex to protect access to the shared global_best_config. */
pthread_mutex_t best_config_mutex = PTHREAD_MUTEX_INITIALIZER;
/** @brief Stores the best valid HyperX configuration found across all threads and dimensions. */
HyperXConfig global_best_config;

/** @brief Array to hold data for each worker thread. */
ThreadData thread_data[MAX_THREADS];
/** @brief Tracks the number of worker threads successfully created for the current search dimension (L). */
int num_active_threads = 0;
/** @brief Global flag to signal interruption (e.g., via Ctrl+C). `volatile` as it's modified by signal handler and read by workers/monitor. */
volatile bool keep_running = true;
/** @brief Global flag to control the monitor thread's execution loop. `volatile` as it's modified by main thread and read by monitor. */
volatile bool monitor_active = false;
/** @brief Stores the time when the overall search for a given set of parameters started. */
time_t search_start_time;

// --- Function Prototypes ---

// Logging
void log_message(const char* format, ...);

// System Info
int get_num_cores();

// Signal Handling
void signal_handler(int sig);

// Utility Functions
void format_with_commas(long num, char *str);
int product(const int *array, int size);

// Core Logic: Evaluation & Search
void evaluate_configuration(const int *shape, int dim_count, int hosts, int radix, double bisection,
                           HyperXConfig *best_config, pthread_mutex_t *best_config_mutex,
                           bool *found_better_in_thread, int thread_id,
                           volatile long *configs_processed, volatile long *configs_pruned);
void generate_shapes_worker(const int *dim_sizes, int num_dim_sizes, int L, int min_dim_size,
                          int hosts, int radix, double bisection, HyperXConfig *best_config,
                          pthread_mutex_t *best_config_mutex, int *current_shape, int level,
                          int start_idx_for_level, int thread_id,
                          volatile long *configs_processed, volatile long *configs_pruned);

// Thread Functions
void *worker_thread_function(void *arg);
void *monitor_thread(void *arg);

// Main Orchestration
HyperXConfig find_optimal_hyperx(int hosts, int radix, double bisection,
                              bool use_power_of_2, int min_dim_size, bool force_exhaustive, int num_threads);

// Output & Input
void print_hyperx_config(const HyperXConfig *config);
int get_int_input(const char *prompt, int min_value, int max_value);
double get_double_input(const char *prompt, double min_value, double max_value);
bool get_yes_no_input(const char *prompt, bool default_value);

// --- Logging Implementation ---

/**
 * @brief Writes a formatted message to the log file with a timestamp. Thread-safe.
 *
 * @param format The format string (printf-style).
 * @param ... Variable arguments for the format string.
 */
void log_message(const char* format, ...) {
    // Lock mutex to ensure exclusive access to the log file
    pthread_mutex_lock(&log_mutex);
    // Check if file is open (might fail during startup)
    if (!log_file) {
        pthread_mutex_unlock(&log_mutex);
        return;
    }

    // Use variadic arguments (va_list) to handle printf-style formatting
    va_list args;
    va_start(args, format);

    // Get current time and format it
    time_t now = time(NULL);
    char time_str[26];
    ctime_r(&now, time_str); // ctime_r is the thread-safe version of ctime
    time_str[24] = '\0'; // Remove the newline character added by ctime_r

    // Print timestamp and formatted message
    fprintf(log_file, "[%s] ", time_str);
    vfprintf(log_file, format, args); // vfprintf handles va_list
    fprintf(log_file, "\n");

    // Ensure the message is written immediately (important for monitoring progress)
    fflush(log_file);

    va_end(args);
    pthread_mutex_unlock(&log_mutex); // Release the lock
}

// --- System Info Implementation ---

/**
 * @brief Gets the number of logical CPU cores available on the system.
 * Uses sysctlbyname on macOS and sysconf as a fallback (for Linux/other POSIX).
 * @return The number of detected cores (minimum 1).
 */
int get_num_cores() {
    int count = 0;
    size_t size = sizeof(count);

    // Try macOS specific method first
    if (sysctlbyname("hw.ncpu", &count, &size, NULL, 0) == 0) {
         // Successfully got count via sysctl
         if (count < 1) count = 1; // Safety check
    } else {
        // Fallback to POSIX standard method (for Linux etc.)
        count = sysconf(_SC_NPROCESSORS_ONLN);
         if (count < 1) count = 1; // Ensure at least 1
    }
    log_message("Detected %d logical CPU core(s)", count);
    printf("Detected %d logical CPU core(s)\n", count);
    return count;
}


// --- Signal Handling Implementation ---

/**
 * @brief Signal handler for SIGINT (Ctrl+C).
 * Sets the global keep_running flag to false to request graceful shutdown.
 * @param sig The signal number (expected to be SIGINT).
 */
void signal_handler(int sig) {
    // Check if the signal received is SIGINT
    if (sig == SIGINT) {
        log_message("SIGINT received. Requesting search termination...");
        printf("\n\nInterrupt signal received! Attempting graceful shutdown...\n");
        // Set the volatile flag. Worker threads and the monitor thread check this flag.
        keep_running = false;
    }
}

// --- Utility Function Implementations ---

/**
 * @brief Formats a long integer with commas as thousands separators.
 *
 * @param num The number to format.
 * @param str Output buffer to store the formatted string. Must be large enough.
 *
 * Example: format_with_commas(1234567, str) -> str = "1,234,567"
 *
 * Note: This implementation is basic and assumes sufficient buffer size.
 * It handles the insertion logic based on string length and modulo 3.
 */
void format_with_commas(long num, char *str) {
    char temp[50]; // Temporary buffer for the number as a string
    sprintf(temp, "%ld", num);
    int len = strlen(temp);
    int j = 0; // Index for the output string `str`
    int lead = len % 3; // Number of digits in the first group (1, 2, or 3)
    if (lead == 0) lead = 3; // If length is multiple of 3, first group has 3 digits

    for (int i = 0; i < len; i++) {
        // Add comma before starting a new group of 3, except at the very beginning
        if (i == lead && i != len) { // Check if we are at the position to insert a comma
             str[j++] = ',';
             lead += 3; // Move the next comma position
        }
        str[j++] = temp[i]; // Copy the digit
    }
    str[j] = '\0'; // Null-terminate the output string
}

/**
 * @brief Calculates the product of elements in an integer array.
 * Uses `long long` internally to mitigate intermediate overflow.
 *
 * @param array Pointer to the integer array.
 * @param size Number of elements in the array.
 * @return The product of the elements, or INT_MAX if overflow occurs.
 */
int product(const int *array, int size) {
    long long p = 1; // Use 64-bit integer for intermediate product
    for (int i = 0; i < size; i++) {
        // Check for potential overflow *before* multiplication
        // If p is already large, multiplying by array[i] might exceed LLONG_MAX
        // A simpler check: if p > INT_MAX / array[i] (handle array[i]==0 case)
        // For now, just check if intermediate p exceeds INT_MAX
        if (p > INT_MAX) return INT_MAX; // If p already too big

        p *= array[i];

        // Check *after* multiplication if product exceeds INT_MAX
        if (p > INT_MAX) return INT_MAX;
    }
    // Final check before casting back to int (redundant if check after mult works)
    if (p > INT_MAX) return INT_MAX;
    return (int)p;
}


// --- Core Logic: Configuration Evaluation ---

/**
 * @brief Evaluates a single, fully specified HyperX shape (S) to find the optimal
 * trunking (K) and terminals per switch (T) that satisfy all constraints.
 * Updates the global best configuration if this shape yields a better result.
 *
 * @param shape The dimension sizes array S = [S1, ..., SL] for the configuration.
 * @param dim_count The number of dimensions (L).
 * @param hosts The target minimum number of hosts (N).
 * @param radix The switch radix (R).
 * @param bisection The target minimum relative bisection bandwidth (B).
 * @param best_config Pointer to the shared global best configuration.
 * @param best_config_mutex Mutex protecting access to best_config.
 * @param found_better_in_thread Output flag, set to true if this call updated the global best.
 * @param thread_id ID of the calling thread (for logging).
 * @param configs_processed Pointer to the thread's processed counter (volatile long).
 * @param configs_pruned Pointer to the thread's pruned counter (volatile long).
 */
void evaluate_configuration(const int *shape, int dim_count, int hosts, int radix, double bisection,
                           HyperXConfig *best_config, pthread_mutex_t *best_config_mutex,
                           bool *found_better_in_thread, int thread_id,
                           volatile long *configs_processed, volatile long *configs_pruned)
{
    // --- Constraint Checks and Pruning ---

    // 1. Check global stop flag (e.g., from Ctrl+C)
    if (!keep_running) return;

    // 2. Calculate total switches P = Product(Si)
    int total_switches = product(shape, dim_count);

    // 3. Pruning based on Objective Function (Switch Count)
    //    Compare P with the switch count of the best configuration found so far (O).
    //    (Ref: Paper Eq 8: Product(Sk) <= O)
    pthread_mutex_lock(best_config_mutex);
    int current_best_switches = best_config->total_switches;
    pthread_mutex_unlock(best_config_mutex);

    // If P >= O, this shape cannot be better than the current best. Prune.
    if (total_switches == INT_MAX || total_switches >= current_best_switches) {
        // Note: Pruning count is incremented in generate_shapes_worker where this condition is usually caught earlier.
        return;
    }

    // --- Determine Maximum Trunking Factor (max_k) ---
    // Find the max Kk possible given the radix R and dimension count L.
    // Based roughly on radix constraint (Eq 2): T + Sum(Kk*(Sk-1)) <= R
    // Assuming T>=1 and Kk=K, R >= 1 + K * Sum(Sk-1). K <= (R-1) / Sum(Sk-1)
    // Paper uses simpler bound K <= R / (2*L) as approximation.
    int max_k = 0;
    if (dim_count > 0) {
         int min_switch_ports_for_K1 = 0; // Ports needed if Kk=1 for all dimensions Sk > 1
         for(int i=0; i<dim_count; ++i) {
             min_switch_ports_for_K1 += (shape[i] > 1) ? (shape[i] - 1) : 0;
         }

         // Need at least 1 port for terminals (T>=1 assumed for meaningful network)
         if (radix > min_switch_ports_for_K1) {
             // Use the simple approximation K_max = floor(R / (2*L)) but ensure K>=1 if possible
             max_k = max(1, radix / (2 * dim_count));
         } else {
             // Not enough ports even for K=1 connection. This shape is impossible.
             return; // Effectively pruned
         }
    } else { // L=0 case (single switch)
        max_k = 0; // No inter-switch links needed
    }

    // Apply user-defined maximum K and ensure K>=1 if needed
    if (max_k > MAX_TRUNKING) max_k = MAX_TRUNKING;
    // If dim_count > 0 and max_k is calculated as 0 (e.g. R=1, L=1), force K=1 if connectivity is needed.
    // The loop logic handles K=0 for Si=1 dimensions.
    if (max_k < 1 && dim_count > 0 && total_switches > 1) max_k = 1;

    // --- Iterate Through Trunking (K) Vector Combinations ---
    // Goal: Find the combination K = [K1, ..., KL] and the corresponding T
    //       that satisfy all constraints and minimize port waste for *this specific shape S*.

    int best_local_trunking[MAX_DIMENSIONS] = {0}; // Best K found *for this S*
    int best_local_T = 0;                          // Best T found *for this S*
    int min_port_waste = INT_MAX; // Tracks minimum (Available T ports - Required T ports)
    bool valid_trunking_found_for_shape = false;   // Flag if *any* valid (K, T) is found for this S

    int current_trunking[MAX_DIMENSIONS];
    // Initialize K vector: Kk = 1 if Sk > 1, else Kk = 0
    for (int i = 0; i < dim_count; i++) {
        current_trunking[i] = (shape[i] > 1) ? 1 : 0;
    }

    // Iterating through K combinations:
    // Treat the `current_trunking` array as digits of a number in base (max_k + 1).
    // Start from K = [1/0, 1/0, ..., 1] and increment the last "digit" (K_L).
    // When K_L exceeds max_k, reset it to 1/0 and "carry over" to K_{L-1}, etc.
    int dim_idx = dim_count - 1; // Index for the current "digit" being incremented
    while (dim_idx >= 0 && keep_running) { // Loop until all K combinations checked or interrupted

        // --- Check Validity of current_trunking K = [K1, ..., KL] ---

        // 1. Calculate Inter-Switch Ports Used based on K and S
        //    Ports = Sum( Kk * (Sk - 1) ) for k=1 to L
        int switch_ports_used = 0;
        bool possible_k = true; // Check basic validity (Kk=0 if Sk=1, Kk>=1 if Sk>1)
        for (int i = 0; i < dim_count; i++) {
             if (shape[i] <= 1 && current_trunking[i] > 0) { possible_k = false; break; }
             if (shape[i] > 1 && current_trunking[i] == 0) { possible_k = false; break; }
            switch_ports_used += current_trunking[i] * (shape[i] - 1);
        }

        // 2. Check Radix Constraint (Eq 2: T + Sum(Kk*(Sk-1)) <= R)
        //    Rearranged: Sum(Kk*(Sk-1)) <= R - T. Since T >= 1 (implicitly), Sum <= R-1.
        //    If switch_ports_used already exceeds R, it's impossible.
        if (!possible_k || switch_ports_used >= radix) {
            // Skip to the next K combination immediately
            goto next_trunking;
        }

        // 3. Calculate Available Ports for Terminals (T)
        int available_T_ports = radix - switch_ports_used;

        if (available_T_ports > 0) { // Only proceed if ports remain for terminals

            // 4. Calculate Minimum T for Host Count Constraint (Eq 3: T*P >= N)
            //    T >= N / P. Since T must be integer, T >= ceil(N / P).
            //    Using integer arithmetic for ceil: (N + P - 1) / P
            int T_min_hosts = (hosts + total_switches - 1) / total_switches;

            if (T_min_hosts <= available_T_ports) { // Check if enough ports for min hosts T

                // 5. Calculate Minimum T for Bisection Bandwidth Constraint (Eq 4: beta = min(Kk*Sk)/(2T) >= B)
                //    Rearranged: T >= min(Kk*Sk) / (2*B). Since T is integer, T >= ceil(min(Kk*Sk) / (2*B)).
                int min_S_K = INT_MAX; // Find the minimum product Kk * Sk across dimensions k where Sk > 1
                for (int i = 0; i < dim_count; i++) {
                     if (shape[i] > 1) { // Dimensions of size 1 don't contribute to these bisections
                        int s_k_product = current_trunking[i] * shape[i];
                        if (s_k_product < min_S_K) {
                            min_S_K = s_k_product;
                        }
                     }
                }

                 // Calculate T_min_bisection if required bisection B > 0 and min_S_K was found
                 int T_min_bisection = 0;
                 if (min_S_K != INT_MAX && bisection > 1e-9) { // Avoid division by zero/negative B
                     // Calculate ceil(min_S_K / (2.0 * B)) using floating point
                     T_min_bisection = ceil((double)min_S_K / (2.0 * bisection));
                 }
                 // If min_S_K == INT_MAX (e.g., L=1, S1=1), bisection constraint is trivially met or not applicable.


                // 6. Determine the Required T for this K
                //    T must be at least T_min_hosts AND T_min_bisection.
                int T_required = max(T_min_hosts, T_min_bisection);

                // 7. Check if Required T Fits Available Ports
                if (T_required > 0 && T_required <= available_T_ports) {

                    // 8. Check Actual Bisection Bandwidth Achieved
                    //    Calculate beta_actual = min(Kk*Sk) / (2 * T_required)
                    double actual_bisection_bw = 0.0;
                    if (T_required > 0 && min_S_K != INT_MAX) {
                        actual_bisection_bw = (double)min_S_K / (2.0 * T_required);
                    } else if (min_S_K == INT_MAX && dim_count <= 1) { // Single switch (L=0 or L=1, S1=1)
                        actual_bisection_bw = 1.0; // Effectively full bisection
                    }

                    // 9. Verify Actual Bisection >= Target Bisection (allowing for float tolerance)
                    if (actual_bisection_bw >= (bisection - 1e-6)) {
                        // *** This (S, K, T_required) is a VALID configuration ***

                        // Calculate port waste (metric to choose best K for this S)
                        int port_waste = available_T_ports - T_required;

                        // Check if this K is better (less waste) than previous K's for this S
                        if (!valid_trunking_found_for_shape || port_waste < min_port_waste) {
                            min_port_waste = port_waste;
                            valid_trunking_found_for_shape = true;
                            best_local_T = T_required; // Store the T for this best K
                            memcpy(best_local_trunking, current_trunking, dim_count * sizeof(int)); // Store this K

                            // This shape S, with this K and T, is a candidate for the global optimum.
                            // The check against global_best_config happens *after* the K loop finishes,
                            // ensuring we use the *most port-efficient* K for this S in the comparison.
                        }
                    } // End bisection check
                } // End T required fits check
            } // End T min hosts fits check
        } // End available T ports check


        // --- Go to Next Trunking Combination ---
        next_trunking: // Label for goto jump from invalid K checks

        if (!keep_running) break; // Exit K loop if interrupted

        // Increment K vector like multi-digit counter
        dim_idx = dim_count - 1; // Start from the rightmost "digit" (K_{L-1})
        while (dim_idx >= 0) {
            current_trunking[dim_idx]++; // Increment current dimension's K

            // Check if the new Kk is valid (<= max_k and consistent with Sk)
            bool k_valid_for_s = (shape[dim_idx] > 1 || current_trunking[dim_idx] == 0);
            if (current_trunking[dim_idx] <= max_k && k_valid_for_s) {
                 // Valid increment, stop carrying over and proceed to evaluate this new K
                 break;
            }

            // Kk exceeded max_k or became invalid for Sk=1. Reset and carry over.
            current_trunking[dim_idx] = (shape[dim_idx] > 1) ? 1 : 0; // Reset to min valid (1 or 0)
            dim_idx--; // Move to the next digit to the left
        }
        // If dim_idx < 0, all K combinations have been tried. The outer while loop terminates.

    } // End while (iterating through K combinations)

    // --- Update Global Best Configuration (if applicable) ---
    // This check happens *after* iterating through all K for the given shape S.
    if (valid_trunking_found_for_shape && keep_running) {
        // We found at least one valid (K, T) pair for this shape S.
        // The best one (minimum port waste) is stored in best_local_trunking/best_local_T.
        // Now, compare this shape S (with its best K/T) to the global best.

        // Lock the mutex for thread-safe access to global_best_config
        pthread_mutex_lock(best_config_mutex);

        // Re-check the global best switch count, as another thread might have updated it
        // while we were evaluating K combinations.
        if (total_switches < best_config->total_switches) {
            // *** This shape S IS better than the current global best! ***
            // Update the global best configuration.

            *found_better_in_thread = true; // Set flag for the calling worker thread

            best_config->dimensions = dim_count;
            memcpy(best_config->shape, shape, dim_count * sizeof(int));
            memcpy(best_config->trunking, best_local_trunking, dim_count * sizeof(int));
            best_config->terminals = best_local_T;
            best_config->total_switches = total_switches;
            best_config->total_terminals = (long long)best_local_T * total_switches; // Use long long for safety

            // Recalculate derived metrics for the *new* global best config
            int final_switch_ports = 0;
            int final_min_S_K = INT_MAX;
             for (int i = 0; i < dim_count; i++) {
                final_switch_ports += best_local_trunking[i] * (shape[i] - 1);
                 if (shape[i] > 1) {
                     int s_k_prod = best_local_trunking[i] * shape[i];
                    if (s_k_prod < final_min_S_K) final_min_S_K = s_k_prod;
                 }
            }
            best_config->available_terminal_ports = radix - final_switch_ports;

            // Actual Bisection
            if (best_local_T > 0 && final_min_S_K != INT_MAX) {
                best_config->actual_bisection = (double)final_min_S_K / (2.0 * best_local_T);
            } else {
                 best_config->actual_bisection = (dim_count <= 1 && total_switches <=1) ? 1.0 : 0.0;
            }

            // Diameter and Avg Path Length
            best_config->network_diameter = 0;
            best_config->avg_path_length = 0.0;
            for (int i = 0; i < dim_count; i++) {
                if (shape[i] > 1) {
                    best_config->network_diameter++;
                    // Avg path length approximation from Dragonfly paper [Kim et al., ISCA 2008]
                    // Assumes uniform random traffic. Sum over dimensions i of (1 - 1/Si)
                    best_config->avg_path_length += (1.0 - 1.0/shape[i]);
                }
            }

             // Port Utilization
             best_config->port_utilization = (double)100.0 * (final_switch_ports + best_local_T) / radix;

        } // End if (total_switches < best_config->total_switches)
        pthread_mutex_unlock(best_config_mutex); // Release the lock
    } // End if (valid_trunking_found_for_shape)

    // Increment the processed counter for this thread, as one full shape evaluation is complete.
    (*configs_processed)++;
}


// --- Core Logic: Recursive Shape Generation (Worker Thread Task) ---

/**
 * @brief Recursively generates candidate HyperX shapes S = [S1, ..., SL] and
 * calls evaluate_configuration() for complete shapes. Implements pruning.
 *
 * @param dim_sizes Array of possible sizes for each dimension.
 * @param num_dim_sizes Number of possible dimension sizes.
 * @param L The target number of dimensions for shapes being generated.
 * @param min_dim_size Minimum allowed size Sk.
 * @param hosts Target number of hosts N.
 * @param radix Switch radix R.
 * @param bisection Target bisection bandwidth fraction B.
 * @param best_config Pointer to the shared global best configuration.
 * @param best_config_mutex Mutex for best_config access.
 * @param current_shape Array holding the partially or fully built shape S.
 * @param level Current dimension index being decided (0 to L-1).
 * @param start_idx_for_level Optimization: Index in dim_sizes to start searching for S[level].
 * Ensures S1 <= S2 <= ... <= SL, avoiding permutations[cite: 142].
 * @param thread_id ID of the calling thread.
 * @param configs_processed Pointer to the thread's processed counter.
 * @param configs_pruned Pointer to the thread's pruned counter.
 */
void generate_shapes_worker(const int *dim_sizes, int num_dim_sizes, int L, int min_dim_size,
                          int hosts, int radix, double bisection, HyperXConfig *best_config,
                          pthread_mutex_t *best_config_mutex, int *current_shape, int level,
                          int start_idx_for_level,
                          int thread_id,
                          volatile long *configs_processed, volatile long *configs_pruned)
{
    // Check global stop flag
    if (!keep_running) return;

    // --- Pruning Check (Objective Bound - Paper Eq 8) ---
    // This check occurs *before* recursing deeper for the remaining dimensions.
    if (level > 0) { // No need to prune before the first dimension is chosen
        // Calculate the product of the dimensions chosen so far (S1 * ... * S_{level-1})
        long long partial_product = 1;
        for (int i = 0; i < level; i++) {
            partial_product *= current_shape[i];
            // Check for potential overflow during partial product calculation
            if (partial_product >= INT_MAX) { partial_product = INT_MAX; break; }
        }

        // Calculate the minimum possible total switches if we complete this partial shape
        // using the smallest allowed dimension size (min_dim_size) for all remaining dimensions.
        // min_total = partial_product * (min_dim_size ^ (L - level))
        long long min_possible_switches_ll = partial_product;
        if (min_possible_switches_ll < INT_MAX) {
            for (int i = level; i < L; i++) {
                // Check for overflow *before* multiplying
                if (min_dim_size == 0) { min_possible_switches_ll = 0; break; } // Avoid division by zero check if needed elsewhere
                if (min_possible_switches_ll > INT_MAX / min_dim_size) {
                     min_possible_switches_ll = INT_MAX; // Mark as overflow
                     break;
                 }
                 min_possible_switches_ll *= min_dim_size;
            }
        }
        // Ensure result fits in int, saturate at INT_MAX if overflow occurred
        int min_possible_switches = (min_possible_switches_ll >= INT_MAX) ? INT_MAX : (int)min_possible_switches_ll;

        // Get the current best switch count found globally (thread-safe)
        pthread_mutex_lock(best_config_mutex);
        int current_best = best_config->total_switches;
        pthread_mutex_unlock(best_config_mutex);

        // ** THE PRUNING DECISION **
        // If the minimum possible switch count for this branch is already not better
        // than the best solution found so far, prune this entire branch.
        if (min_possible_switches >= current_best) {
            (*configs_pruned)++; // Increment prune counter for this thread
            return; // Do not recurse further down this path
        }
    }

    // --- Base Case: Complete Shape ---
    if (level == L) {
        // We have selected sizes for all L dimensions. current_shape is complete.
        // Evaluate this complete shape S to find its best K and T.
        bool found_better = false; // Local flag for whether this specific call updates global best
        evaluate_configuration(current_shape, L, hosts, radix, bisection,
                              best_config, best_config_mutex, &found_better, thread_id,
                              configs_processed, configs_pruned);

        // If evaluate_configuration updated the global best, log and print the finding.
        if (found_better) {
            // Format the shape S = [S1, ..., SL] into a string for output.
            char shape_str[150] = "";
             char temp[20];
             for(int i=0; i<L; ++i) {
                 sprintf(temp, "%d%s", current_shape[i], (i < L - 1) ? ", " : "");
                 // Simple bounds check for strcat
                 if (strlen(shape_str) + strlen(temp) < sizeof(shape_str) - 1) {
                    strcat(shape_str, temp);
                 } else {
                    // Avoid buffer overflow if shape string gets too long
                    strcat(shape_str, "...");
                    break;
                 }
             }

            // Read the new best switch count (thread-safe) for reporting
            pthread_mutex_lock(best_config_mutex);
            int new_best_switches = best_config->total_switches;
            pthread_mutex_unlock(best_config_mutex);

            // Log and print the improvement
            log_message("Thread %d found new best: %d switches, Shape [%s]", thread_id, new_best_switches, shape_str);
            // Print to console, trying not to overwrite the monitor line excessively.
            printf("\n[Thread %d] New best: %d switches, Shape [%s]\n", thread_id, new_best_switches, shape_str);
            // Monitor thread will refresh the progress line on its next cycle.
        }
        return; // End recursion for this complete shape
    }

    // --- Recursive Step: Choose size for current level ---
    // Iterate through possible sizes for the current dimension `level`.
    // Optimization (Symmetry Constraint - Paper Eq 6):
    // Start iterating from `start_idx_for_level`. This index comes from the dimension
    // size choice made at the *previous* level (level-1). By ensuring that
    // S[level] >= S[level-1], we generate shapes in non-decreasing order
    // (e.g., [4, 8, 8] is generated, but [8, 4, 8] is not), avoiding redundant permutations.
    for (int i = start_idx_for_level; i < num_dim_sizes && keep_running; i++) {
        // Assign the chosen size to the current dimension level
        current_shape[level] = dim_sizes[i];

        // Recurse to choose the size for the next dimension (level + 1)
        generate_shapes_worker(dim_sizes, num_dim_sizes, L, min_dim_size,
                             hosts, radix, bisection, best_config,
                             best_config_mutex, current_shape, level + 1, // Move to next level
                             i, // Pass current index 'i' as the start index for the *next* level (S_{L+1} >= S_L)
                             thread_id, configs_processed, configs_pruned);
    }
}


// --- Thread Function Implementations ---

/**
 * @brief Entry point for worker threads.
 * Manages the exploration of a subset of the search space for a given dimension L.
 *
 * @param arg Pointer to the ThreadData structure for this thread.
 * @return NULL.
 */
void *worker_thread_function(void *arg) {
    ThreadData *data = (ThreadData *)arg; // Cast argument back to ThreadData pointer
    data->is_running = true;              // Signal that the thread has started work
    data->configs_processed = 0;          // Initialize counters for this run
    data->configs_pruned = 0;

    log_message("Thread %d started: L=%d, DimSizes=%d, Idx Range=[%d, %d)",
               data->thread_id, data->L, data->num_dim_sizes, data->start_idx, data->end_idx);

    // --- Work Distribution ---
    // Each thread is responsible for a specific range of possible sizes for the *first* dimension (S0).
    // It iterates through its assigned `dim_sizes` indices (`data->start_idx` to `data->end_idx`).
    for (int i = data->start_idx; i < data->end_idx && keep_running; i++) {
        int shape[MAX_DIMENSIONS] = {0}; // Allocate space for the shape being built in this branch
        shape[0] = data->dim_sizes[i];   // Set the first dimension size S0 based on the thread's assigned range

        // --- Initiate Recursive Search for Remaining Dimensions ---
        // Call generate_shapes_worker to explore dimensions S1, S2, ..., S(L-1).
        // Start recursion at level 1 (we've already set level 0).
        // Pass 'i' as start_idx_for_level: Ensures S1 >= S0, S2 >= S1, etc. (Paper Eq 6)
        generate_shapes_worker(data->dim_sizes, data->num_dim_sizes, data->L, data->min_dim_size,
                             data->hosts, data->radix, data->bisection, data->best_config,
                             data->best_config_mutex, shape, 1, // Start recursion at level 1
                             i, // start_idx_for_level for S1 is index used for S0
                             data->thread_id,
                             &data->configs_processed, // Pass pointers to thread's counters
                             &data->configs_pruned);
    }

    // Log thread completion and statistics
    log_message("Thread %d finished: Processed=%ld, Pruned=%ld",
               data->thread_id, data->configs_processed, data->configs_pruned);

    data->is_running = false; // Signal that the thread has finished work
    return NULL; // Standard pthread return
}

/**
 * @brief Entry point for the monitor thread.
 * Periodically collects statistics from worker threads and prints progress.
 *
 * @param arg Not used.
 * @return NULL.
 */
void* monitor_thread(void* arg) {
    log_message("Monitor thread started.");
    monitor_active = true; // Signal that the monitor is running
    long last_processed = 0; // Track total processed count from previous interval
    time_t last_time = search_start_time; // Track time of previous interval

    // Loop while the search is active (keep_running) and the monitor hasn't been told to stop (monitor_active)
    while (keep_running && monitor_active) {
        // Sleep for the defined interval
        sleep(PROGRESS_INTERVAL);

        // Re-check flags after waking up, in case they changed during sleep
         if (!keep_running || !monitor_active) break;

        // --- Collect Statistics ---
        time_t now = time(NULL);
        double elapsed_total = difftime(now, search_start_time);
        if (elapsed_total < 1) elapsed_total = 1; // Avoid division by zero

        // Sum up counts from all worker threads.
        // Reading volatile variables without a lock is generally acceptable for approximate progress monitoring,
        // assuming reads are atomic enough (true for `long` on most modern platforms).
        // A lock (e.g., progress_mutex) could be added for strict accuracy if needed.
        long total_processed = 0;
        long total_pruned = 0;
        int running_threads = 0;
        // Iterate up to num_active_threads (set by main thread for the current L)
        for (int i = 0; i < num_active_threads; i++) {
            total_processed += thread_data[i].configs_processed;
            total_pruned += thread_data[i].configs_pruned;
            if (thread_data[i].is_running) { // Check the volatile flag
                running_threads++;
            }
        }

         // --- Calculate Processing Rate ---
         // Calculate rate based on the change *during the last interval* for smoother reporting.
         double interval = difftime(now, last_time);
         long processed_interval = total_processed - last_processed;
         // Calculate rate, avoiding division by small interval and handling potential decrease if threads finish exactly at interval boundary.
         double rate = (interval > 0.1 && processed_interval >= 0) ? (double)processed_interval / interval : 0.0;

         // Update state for the next interval
         last_processed = total_processed;
         last_time = now;


        // --- Format and Print Progress ---
        char processed_str[32], pruned_str[32], rate_str[32];
        format_with_commas(total_processed, processed_str);
        format_with_commas(total_pruned, pruned_str);
        format_with_commas((long)rate, rate_str); // Display rate as integer/sec

        // Use '\r' to move cursor to beginning of line, overwriting previous progress update.
        // Add spaces at the end to clear any leftover characters from longer previous lines.
        printf("\r[%ds] Active: %d/%d | Eval: %s | Pruned: %s | Rate: %s/s   ",
               (int)elapsed_total, running_threads, num_active_threads,
               processed_str, pruned_str, rate_str);
        fflush(stdout); // Ensure the output appears immediately

        // --- Log Progress Less Frequently ---
        static time_t last_log_time = 0; // Static variable to track last log time
        if (now - last_log_time >= 10) { // Log every 10 seconds
              log_message("Progress: Active=%d/%d, Eval=%s, Pruned=%s, Rate=%s/s",
                     running_threads, num_active_threads, processed_str, pruned_str, rate_str);
             last_log_time = now;
         }

        // --- Check for Worker Completion (Optional Exit Condition) ---
        // If all worker threads (`num_active_threads` launched for this L) are no longer running.
        if (running_threads == 0 && num_active_threads > 0) {
             // Workers might have just finished. Wait briefly to see if main thread signals stop.
             sleep(1);
             if (!monitor_active) break; // Exit if main thread set flag during sleep
             // Optional: Could break here if main thread doesn't manage monitor lifecycle precisely.
             // log_message("Monitor detected all workers finished for L=%d.", ???); // Need L info if breaking here
             // break; // Currently letting main thread control exit via monitor_active=false
        }
    } // End while loop

    printf("\nMonitor thread finished.\n"); // Print a newline after the final progress update
    log_message("Monitor thread finished.");
    return NULL;
}


// --- Main Search Orchestration ---

/**
 * @brief Main function to find the optimal HyperX configuration.
 * Orchestrates the search across dimensions (L) and manages worker threads.
 *
 * @param hosts Target minimum number of hosts (N).
 * @param radix Switch radix (R).
 * @param bisection Target minimum relative bisection bandwidth (B).
 * @param use_power_of_2 If true, restrict dimension sizes Sk to powers of 2.
 * @param min_dim_size Minimum allowed size for any dimension Sk.
 * @param force_exhaustive If true, disable the heuristic early exit between L levels.
 * @param num_threads Number of worker threads to use.
 * @return The best HyperXConfig found, or an empty/invalid config if none found.
 */
HyperXConfig find_optimal_hyperx(int hosts, int radix, double bisection,
                               bool use_power_of_2, int min_dim_size, bool force_exhaustive, int num_threads)
{
    // Initialize global best config for this search run
    memset(&global_best_config, 0, sizeof(HyperXConfig));
    global_best_config.total_switches = INT_MAX; // Initialize objective to maximize (minimize switches)

    // Reset global state for a new search
    keep_running = true;    // Allow search to run
    monitor_active = false; // Monitor thread starts inactive
    search_start_time = time(NULL); // Record search start time

    // --- Start Monitor Thread ---
    pthread_t monitor_thread_id = 0; // Use 0 to indicate potential creation failure
    if (pthread_create(&monitor_thread_id, NULL, monitor_thread, NULL) != 0) {
        log_message("FATAL: Failed to create monitor thread!");
        perror("pthread_create (monitor)");
        // Decide how to handle: exit or continue without monitor? Logged, continue for now.
    }

    // --- Iterate through Dimension Counts (L) ---
    // Search dimensions L=1 up to a practical limit (e.g., 5), respecting keep_running flag.
    const int MAX_L_TO_SEARCH = 5; // Set maximum dimension based on practical considerations
    for (int L = 1; L <= MAX_L_TO_SEARCH && keep_running; L++) {

        log_message("Starting search for %d-dimensional topologies...", L);
        printf("\n===== Evaluating %d-Dimensional Topologies =====\n", L);

        // --- Generate candidate dimension sizes (Sk) for this L ---
        // Creates an array `dim_sizes` containing possible values for Sk,
        // respecting `min_dim_size` and `use_power_of_2` constraints.
        int dim_sizes[MAX_DIM_SIZE];
        int num_dim_sizes = 0;
        int max_dim_size_for_L = MAX_DIM_SIZE; // Could potentially cap based on L, R, N

        if (use_power_of_2) {
            int current_power = 1;
            while(current_power < min_dim_size) current_power *= 2; // Start >= min_dim_size
            while(current_power <= max_dim_size_for_L) {
                 if (num_dim_sizes < MAX_DIM_SIZE) {
                     dim_sizes[num_dim_sizes++] = current_power;
                 } else {
                     log_message("Warning: Exceeded dim_sizes buffer for L=%d (PowerOf2)", L);
                     break;
                 }
                 if (max_dim_size_for_L / 2 < current_power) break; // Avoid overflow in next multiplication
                 current_power *= 2;
            }
        } else { // Any integer size >= min_dim_size
            for (int size = min_dim_size; size <= max_dim_size_for_L; size++) {
                 if (num_dim_sizes < MAX_DIM_SIZE) {
                     dim_sizes[num_dim_sizes++] = size;
                 } else {
                      log_message("Warning: Exceeded dim_sizes buffer for L=%d (AnySize)", L);
                      break;
                 }
            }
        }

        // If no valid sizes possible (e.g., min_dim_size > MAX_DIM_SIZE), skip this L
        if (num_dim_sizes == 0) {
            log_message("No valid dimension sizes found for L=%d (min_dim=%d, max_dim=%d, pow2=%d)",
                       L, min_dim_size, max_dim_size_for_L, use_power_of_2);
            printf("  Skipping L=%d: No valid dimension sizes found.\n", L);
            continue;
        }

        // Log the candidate sizes
        char sizes_str[200] = "";
        int count = 0;
        for(int i=0; i<num_dim_sizes && count < 10; ++i, ++count) {
            char temp[10]; sprintf(temp, "%d ", dim_sizes[i]);
             if (strlen(sizes_str) + strlen(temp) < sizeof(sizes_str)-1) strcat(sizes_str, temp); else break;
        }
        if (num_dim_sizes > 10) strcat(sizes_str, "...");
        log_message("L=%d: Considering %d dimension sizes: %s", L, num_dim_sizes, sizes_str);
        printf("  Considering %d possible sizes per dimension (min %d, %s): %s\n",
               num_dim_sizes, min_dim_size, use_power_of_2 ? "pow2" : "any", sizes_str);


        // --- Optional: Evaluate a Balanced Cube Configuration First ---
        // Quickly check a configuration where all Sk are equal (or close) to N^(1/L).
        // This often provides a good initial solution, improving pruning effectiveness early on.
        int balanced_shape[MAX_DIMENSIONS];
        double target_size_per_dim = pow((double)hosts, 1.0/L);
        int closest_size = -1;
        double min_diff = 1e18; // Initialize with a large difference
        // Find the size in dim_sizes closest to the ideal target
         for (int i=0; i < num_dim_sizes; ++i) {
             double diff = fabs(dim_sizes[i] - target_size_per_dim);
             if (diff < min_diff) {
                 min_diff = diff;
                 closest_size = dim_sizes[i];
             }
              // Optimization: if we find a size very close, use it immediately
              if (diff < 0.1) break;
         }
         // If a closest size was found, create and evaluate the balanced shape
         if (closest_size != -1) {
             for(int i=0; i<L; ++i) balanced_shape[i] = closest_size;
             log_message("Evaluating initial balanced shape [%d,...(%dx)] for L=%d", closest_size, L, L);
             bool found_better_balanced = false;
             // Use temporary counters as this is outside the main thread loop work division
             volatile long temp_processed = 0, temp_pruned = 0;
             // Call evaluate_configuration directly from the main thread (thread_id = -1)
             evaluate_configuration(balanced_shape, L, hosts, radix, bisection,
                                   &global_best_config, &best_config_mutex, &found_better_balanced,
                                   -1, &temp_processed, &temp_pruned);
             if (found_better_balanced) {
                 printf("  Initial balanced config is promising (Switches: %d).\n", global_best_config.total_switches);
                 log_message("Initial balanced config improved best (Switches: %d)", global_best_config.total_switches);
             }
         } else {
             log_message("Could not determine a balanced configuration for L=%d", L);
         }


        // --- Start Worker Threads for Parallel Search of Dimension L ---
        pthread_t worker_threads[MAX_THREADS];
        num_active_threads = 0; // Reset count for this L

        // --- Work Division Strategy ---
        // Divide the iteration over the possible sizes for the *first* dimension (S0)
        // among the available threads. Each thread handles a slice of the `dim_sizes` array.
        int items_total = num_dim_sizes; // Total number of choices for S0
        int items_per_thread = (items_total + num_threads - 1) / num_threads; // Ceil division

        printf("  Launching %d worker threads for L=%d search...\n", num_threads, L);
        log_message("L=%d: Launching %d threads, %d items total, ~%d items/thread", L, num_threads, items_total, items_per_thread);

        // Create and launch worker threads
        for (int t = 0; t < num_threads; t++) {
            // Populate the ThreadData struct for this thread
            thread_data[t].thread_id = t;
            thread_data[t].start_idx = t * items_per_thread; // Start index for S0 choices
            thread_data[t].end_idx = min((t + 1) * items_per_thread, items_total); // End index (exclusive)
            // Ensure end index doesn't go past start if workload is small
             if (thread_data[t].end_idx < thread_data[t].start_idx) {
                 thread_data[t].end_idx = thread_data[t].start_idx; // Assign no work
             }
            // Copy or point to necessary parameters
            thread_data[t].dim_sizes = dim_sizes;
            thread_data[t].num_dim_sizes = num_dim_sizes;
            thread_data[t].L = L;
            thread_data[t].min_dim_size = min_dim_size;
            thread_data[t].hosts = hosts;
            thread_data[t].radix = radix;
            thread_data[t].bisection = bisection;
            thread_data[t].best_config = &global_best_config; // Pointer to shared best
            thread_data[t].best_config_mutex = &best_config_mutex; // Pointer to shared mutex
            thread_data[t].configs_processed = 0; // Reset thread counters
            thread_data[t].configs_pruned = 0;
            thread_data[t].is_running = false; // Thread sets this to true when it starts

            // Create the thread only if it has work to do
            if (thread_data[t].start_idx < thread_data[t].end_idx) {
                if (pthread_create(&worker_threads[t], NULL, worker_thread_function, &thread_data[t]) != 0) {
                    // Handle thread creation error
                    log_message("ERROR: Failed to create worker thread %d for L=%d", t, L);
                    perror("pthread_create (worker)");
                    // Don't increment num_active_threads if creation failed
                } else {
                    num_active_threads++; // Count successfully launched threads
                }
            } else {
                 // Log if a thread slot is assigned no work
                 log_message("Thread %d has no work assigned (range %d to %d), not starting.", t, thread_data[t].start_idx, thread_data[t].end_idx);
            }
        }
        log_message("L=%d: Successfully launched %d worker threads.", L, num_active_threads);

        // --- Wait for Worker Threads for this L to Complete ---
        int joined_threads = 0;
        for (int t = 0; t < num_threads; t++) {
             // Only join threads that were actually created and assigned work
             // Need to match the condition used for pthread_create
             bool was_active = (thread_data[t].start_idx < thread_data[t].end_idx) && (t < num_active_threads); // Rough check if it should have been created
             // A more robust way might require storing thread handles only for created threads.
             // Assuming for now that indices match created threads up to num_active_threads.
             if (t < num_active_threads) { // Only join threads we attempted to create and counted
                 pthread_join(worker_threads[t], NULL); // Blocks until thread 't' completes
                 joined_threads++;
             }
        }
        // All active threads for dimension L have finished.
        log_message("L=%d: All %d active worker threads joined.", L, joined_threads);


        // --- Post-Dimension Analysis & Optional Early Exit ---
        // Heuristic: If NOT forcing exhaustive search, and we have found a good solution at L,
        // consider skipping higher dimensions (L+1, ...) as they are less likely to yield
        // significantly fewer switches (often requiring more switches for same N).
        if (!force_exhaustive && global_best_config.total_switches != INT_MAX && L >= 2 && L < MAX_L_TO_SEARCH) {
             // Compare current best switch count to a rough estimate for the next dimension level.
             // This heuristic is somewhat arbitrary.
             double ideal_switches_for_L_plus_1 = pow(hosts, (L+1.0)/L); // Ideal switches if shape was (N^(1/L))^(L+1) ? Maybe just N?
             // Paper's heuristic might be more involved. Using a simple placeholder condition.
             // Example: Stop if current switch count is "close" to N. (e.g., < 2*N ?) Need better heuristic.
             // Let's reuse the previous heuristic for now, comparing to ideal switches.
             if (global_best_config.total_switches < 2.0 * ideal_switches_for_L_plus_1 * ideal_switches_for_L_plus_1 ) { // Condition from previous code version
                  log_message("Current solution (L=%d, Switches=%d) deemed good enough. Skipping higher dimensions.",
                             global_best_config.dimensions, global_best_config.total_switches);
                  printf("  Current best solution is good. Skipping search for L > %d.\n", L);
                  // Setting keep_running = false is one way, but break is cleaner here as loop condition checks L.
                  break; // Exit the loop over L immediately
             }
        }

        // Check if Ctrl+C was pressed while waiting for threads
        if (!keep_running) {
             log_message("Interruption detected after L=%d search.", L);
             break; // Exit the loop over L
        }

    } // End for loop over L (L=1 to MAX_L_TO_SEARCH)

    // --- Cleanup and Finalization ---
    log_message("Search phase finished or interrupted (up to L=%d).", MAX_L_TO_SEARCH);
    printf("\nSearch phase finished (up to L=%d).\n", MAX_L_TO_SEARCH);

    // Signal monitor thread to stop and wait for it to finish
    monitor_active = false; // Set flag to false
     if (monitor_thread_id != 0) { // Check if monitor thread was successfully created
         log_message("Waiting for monitor thread to exit...");
         pthread_join(monitor_thread_id, NULL); // Wait for monitor thread to terminate
         log_message("Monitor thread joined.");
     }

    // Return the best configuration found during the entire search
    return global_best_config;
}


// --- Output Formatting ---

/**
 * @brief Prints the details of a HyperX configuration in a readable format.
 * Handles the case where no valid configuration was found.
 *
 * @param config Pointer to the HyperXConfig struct to print.
 */
void print_hyperx_config(const HyperXConfig *config) {
    // Check if a valid configuration was found (total_switches initialized to INT_MAX)
    if (config->total_switches <= 0 || config->total_switches == INT_MAX) {
        printf("\n===================================\n");
        printf("No valid HyperX configuration found.\n");
        printf("Consider relaxing constraints (e.g., bisection, min dimension size) or increasing host count/radix.\n");
        printf("===================================\n\n");
        log_message("Final Result: No valid configuration found.");
        return;
    }

    // Format large numbers with commas for readability
    char sw_str[32], term_str[32];
    format_with_commas(config->total_switches, sw_str);
    format_with_commas(config->total_terminals, term_str);

    // Print the optimal configuration details
    printf("\n=========== Optimal HyperX Found ===========\n");
    printf("Dimensions (L):          %d\n", config->dimensions);

    printf("Shape (S):               [");
    for (int i = 0; i < config->dimensions; i++) {
        printf("%d%s", config->shape[i], i < config->dimensions - 1 ? ", " : "");
    }
    printf("]\n");

    printf("Trunking (K):            [");
    for (int i = 0; i < config->dimensions; i++) {
        printf("%d%s", config->trunking[i], i < config->dimensions - 1 ? ", " : "");
    }
    printf("]\n");

    printf("Total Switches:          %s\n", sw_str);
    printf("Terminals per Switch (T):%d (of %d ports available)\n",
           config->terminals, config->available_terminal_ports);
    printf("Total Terminals:         %s\n", term_str); // Target hosts N is implicit requirement >= N
    printf("Network Diameter:        %d hops\n", config->network_diameter);
    // Avg Path Length might be less critical for system design, commented out by default
    // printf("Avg Path Length:         %.2f hops\n", config->avg_path_length);
    printf("Actual Bisection BW:     %.4f\n", config->actual_bisection); // Target B provided by user
    printf("Switch Port Utilization: %.1f%%\n", config->port_utilization);
    printf("==========================================\n\n");

    // Log a summary of the best configuration
    char config_log[512];
    // Create a compact representation for the log (shape/trunking omitted for brevity)
     sprintf(config_log, "Best Found: L=%d, Switches=%d, T=%d, TotalTerms=%d, Bisection=%.4f, Util=%.1f%%",
             config->dimensions, config->total_switches, config->terminals, config->total_terminals,
             config->actual_bisection, config->port_utilization);
    log_message(config_log);
}


// --- User Input Handling ---

/**
 * @brief Prompts the user for integer input and validates it within a range.
 *
 * @param prompt The message to display to the user.
 * @param min_value The minimum acceptable value.
 * @param max_value The maximum acceptable value (or <=0 for no upper limit).
 * @return The validated integer input from the user.
 */
int get_int_input(const char *prompt, int min_value, int max_value) {
    int value;
    char buffer[256]; // Input buffer
    while (1) {
        printf("%s", prompt);
        // Read line from standard input
        if (fgets(buffer, sizeof(buffer), stdin) == NULL) {
             // Handle EOF (Ctrl+D) or read error
             if (feof(stdin)) {
                 printf("\nEOF detected, exiting.\n");
                 log_message("EOF detected during input.");
                 if(log_file) fclose(log_file); // Attempt to close log file
                 exit(1); // Exit program
             }
            perror("fgets"); // Log other read errors
            continue; // Retry input
        }
        // Remove trailing newline character
        buffer[strcspn(buffer, "\n")] = 0;

        // Attempt to parse integer using strtol for better error checking
        char *endptr; // Will point to first non-digit character
        long val_l = strtol(buffer, &endptr, 10); // Base 10 conversion

        // Check for parsing errors:
        // - endptr == buffer: No digits were read.
        // - *endptr != '\0': Extra non-digit characters found after the number.
        // - val_l out of int range: Handled implicitly by later checks if needed, strtol checks LONG range.
        if (endptr == buffer || *endptr != '\0' || val_l < INT_MIN || val_l > INT_MAX) {
            printf("  Invalid input. Please enter an integer.\n");
            continue;
        }
        value = (int)val_l; // Convert valid long to int

        // Check if value is within the specified range
        if (value < min_value || (max_value > 0 && value > max_value)) {
            if (max_value > 0) printf("  Value must be between %d and %d.\n", min_value, max_value);
            else printf("  Value must be at least %d.\n", min_value);
            continue;
        }
        // Input is valid
        return value;
    }
}

/**
 * @brief Prompts the user for double-precision floating-point input and validates it.
 *
 * @param prompt The message to display to the user.
 * @param min_value The minimum acceptable value.
 * @param max_value The maximum acceptable value.
 * @return The validated double input from the user.
 */
double get_double_input(const char *prompt, double min_value, double max_value) {
     double value;
     char buffer[256];
     while(1) {
         printf("%s", prompt);
         if (fgets(buffer, sizeof(buffer), stdin) == NULL) {
             if (feof(stdin)) { printf("\nEOF detected, exiting.\n"); log_message("EOF detected during input."); if(log_file) fclose(log_file); exit(1); }
             perror("fgets"); continue;
         }
         buffer[strcspn(buffer, "\n")] = 0;

         // Use strtod for robust parsing
         char *endptr;
         value = strtod(buffer, &endptr);

         // Check for parsing errors
         if (endptr == buffer || *endptr != '\0') {
             printf("  Invalid input. Please enter a number.\n");
             continue;
         }
         // Check if value is within the specified range
         if (value < min_value || value > max_value) {
             printf("  Value must be between %.2f and %.2f.\n", min_value, max_value);
             continue;
         }
         // Input is valid
         return value;
     }
}

/**
 * @brief Prompts the user for a yes/no answer.
 *
 * @param prompt The question to ask the user.
 * @param default_value The value to return if the user just presses Enter.
 * @return true for 'yes'/'y', false for 'no'/'n'.
 */
bool get_yes_no_input(const char *prompt, bool default_value) {
    char buffer[256];
    while (1) {
        // Display prompt with default value indicator [Y/n] or [y/N]
        printf("%s [%s]: ", prompt, default_value ? "Y/n" : "y/N");
        if (fgets(buffer, sizeof(buffer), stdin) == NULL) {
             // Handle EOF or error - maybe return default or exit? Exit for safety.
             if (feof(stdin)) { printf("\nEOF detected, exiting.\n"); log_message("EOF detected during input."); if(log_file) fclose(log_file); exit(1); }
             perror("fgets"); continue;
        }
        buffer[strcspn(buffer, "\n")] = 0; // Remove newline

        // If user just pressed Enter, return the default value
        if (buffer[0] == '\0') return default_value;

        // Check for variations of yes (case-insensitive)
        if (strcasecmp(buffer, "y") == 0 || strcasecmp(buffer, "yes") == 0) return true;
        // Check for variations of no (case-insensitive)
        if (strcasecmp(buffer, "n") == 0 || strcasecmp(buffer, "no") == 0) return false;

        // If input is not recognized, prompt again
        printf("  Please enter 'y' or 'n'.\n");
    }
}


// --- Main Program Entry Point ---

/**
 * @brief Main function. Sets up logging, signal handling, gets user input,
 * calls the optimization function, and prints the result.
 *
 * @param argc Argument count (not used).
 * @param argv Argument vector (not used).
 * @return 0 on successful completion, 1 on error (e.g., log file creation).
 */
int main(int argc, char *argv[]) {
    // --- Setup ---

    // Set up signal handler for SIGINT (Ctrl+C) using sigaction for reliability
    struct sigaction sa;
    sa.sa_handler = signal_handler; // Function to call on signal
    sigemptyset(&sa.sa_mask);       // Do not block any other signals during handler execution
    sa.sa_flags = 0; // No special flags (like SA_RESTART)
    sigaction(SIGINT, &sa, NULL);   // Register handler for SIGINT

    // Open log file
    log_file = fopen(LOG_FILE, "w"); // Open in write mode (creates/truncates)
    if (!log_file) {
        perror("Error opening log file"); // Print system error message
        fprintf(stderr, "Warning: Could not open log file '%s'. Continuing without logging.\n", LOG_FILE);
        // Program can continue, but logging will be disabled.
    } else {
         // Disable buffering on the log file so messages appear immediately.
         setbuf(log_file, NULL);
    }

    log_message("HyperX Optimizer (Parallel Version) Started.");
    printf("\n===== HyperX Network Optimizer (Multi-Core) =====\n");
    if(log_file) printf("Logging progress and details to: %s\n", LOG_FILE);

    // Get system information (number of cores)
    int num_cores = get_num_cores();

    // --- Main Input & Search Loop ---
    // Loop allows user to perform multiple searches without restarting.
    while (1) {
        printf("\nEnter network requirements (or Ctrl+C to exit):\n");

        // Get required network parameters from user with validation
        int radix = get_int_input("Switch Radix (ports, e.g., 64-256): ", 4, 2048);
        int min_hosts = get_int_input("Target Host Count (e.g., 1024-262144): ", 4, 0); // 0 max = no upper limit
        double bisection = get_double_input("Target Bisection BW Fraction (0.1-1.0): ", 0.01, 1.0);

        // Get constraints on the shape S
        printf("\nShape Constraints:\n");
        bool use_power_of_2 = get_yes_no_input("Restrict dimension sizes (S) to powers of 2?", false);
        int min_dim_size = use_power_of_2 ? 2 : get_int_input("Minimum size (S) for any dimension (>=2): ", 2, MAX_DIM_SIZE);
        // If user wants power-of-2, ensure min_dim_size itself is a power of 2 >= requested value
        if (use_power_of_2) {
             int power = 1;
             while (power < min_dim_size) power *= 2;
             if (power != min_dim_size) {
                min_dim_size = power;
                printf("  (Adjusted minimum power-of-2 size to %d)\n", min_dim_size);
             }
        }

        // Get search options
        printf("\nSearch Options:\n");
        bool force_exhaustive = get_yes_no_input("Force exhaustive search (ignore early exit heuristic)?", false);
        int default_threads = max(1, num_cores); // Default to available cores, minimum 1
        // Prompt for number of threads, using default if user enters 0 or invalid
        int threads_to_use = get_int_input("Number of worker threads (1-MAX_THREADS, default = num cores): ", 1, MAX_THREADS);
         if (threads_to_use <= 0) threads_to_use = default_threads;
         threads_to_use = min(threads_to_use, MAX_THREADS); // Cap at MAX_THREADS


        // Log the parameters for this search run
        log_message("--- New Search Parameters ---");
        log_message("Radix: %d, TargetHosts: %d, TargetBisection: %.3f", radix, min_hosts, bisection);
        log_message("Constraints: Pow2Sizes=%s, MinDimSize=%d", use_power_of_2 ? "Yes" : "No", min_dim_size);
        log_message("Search: ForceExhaustive=%s, Threads=%d", force_exhaustive ? "Yes" : "No", threads_to_use);

        // --- Execute the Optimization Search ---
        printf("\nStarting search with %d threads...\n", threads_to_use);
        clock_t start_clock = clock(); // Record CPU time start

        // Call the main optimization function
        HyperXConfig result_config = find_optimal_hyperx(min_hosts, radix, bisection,
                                                        use_power_of_2, min_dim_size, force_exhaustive, threads_to_use);

        double elapsed_time = (double)(clock() - start_clock) / CLOCKS_PER_SEC; // Calculate elapsed CPU time

        // --- Report Results ---
        // Collect final aggregate counts from threads *after* search completes
        long final_processed = 0;
        long final_pruned = 0;
        // Note: num_active_threads holds the count from the *last* L iteration.
        // This might slightly undercount if threads from earlier L were still running? Unlikely.
        // Assumes thread_data holds the final counts for threads used in the last iteration.
        for(int i=0; i<num_active_threads; ++i) {
             final_processed += thread_data[i].configs_processed;
             final_pruned += thread_data[i].configs_pruned;
        }
         char proc_str[32], prune_str[32];
         format_with_commas(final_processed, proc_str);
         format_with_commas(final_pruned, prune_str);

        // Log and print summary statistics
        log_message("Search %s. Total Time: %.2f sec. Total Evaluated: %s, Total Pruned: %s",
                    keep_running ? "completed" : "interrupted", elapsed_time, proc_str, prune_str);
        printf("\nSearch %s in %.2f seconds.\n",
               keep_running ? "completed" : "interrupted", elapsed_time);
         printf("Total Configurations Evaluated: %s\n", proc_str);
         printf("Total Configurations Pruned:    %s\n", prune_str);

        // Print the details of the best configuration found
        print_hyperx_config(&result_config);

        // --- Ask User to Continue or Exit ---
         if (keep_running) { // Only ask if the search wasn't interrupted
            if (!get_yes_no_input("Perform another search?", true)) {
                break; // Exit the while(1) loop
            }
         } else {
             // If interrupted, print message and exit loop
             printf("Search was interrupted. Exiting.\n");
             break;
         }

    } // End main input loop (while(1))

    // --- Program Exit ---
    printf("\nExiting HyperX Optimizer.\n");
    log_message("HyperX Optimizer Finished Normally.");
    // Close the log file if it was successfully opened
    if (log_file) fclose(log_file);
    log_file = NULL;

    return 0; // Indicate successful execution
}
