/* Enable POSIX.1-2008 + GNU extensions for clock_gettime and O_CLOEXEC */
#define _GNU_SOURCE

/*
 * neopixel_cdev_write.c
 *
 * WS2812B bit-bang driver using Linux GPIO character device v2 ABI.
 * Target: Jetson AGX Orin, gpiochip0 line 106 (PQ.06 / board.D4 / GP66 / Pin 7)
 *
 * This library is loaded by neopixel_write.py (local shadow of the Adafruit
 * neopixel_write module) to provide a working neopixel_write backend on Jetson,
 * which the upstream Adafruit Blinka library does not support.
 *
 * Build: bash build_neopixel.sh
 * Run:   sudo chrt -f 99 python3 debug_led.py
 */

#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <time.h>
#include <stdio.h>
#include <linux/gpio.h>

/* ------------------------------------------------------------------ */
/* Hardware configuration                                               */
/* ------------------------------------------------------------------ */

#define CHIP_PATH    "/dev/gpiochip0"
#define LINE_OFFSET  106u    /* PQ.06 = board.D4 = GP66 on Jetson AGX Orin */
#define CONSUMER     "neopixel"

/* ------------------------------------------------------------------ */
/* WS2812B timing targets (nanoseconds)                                 */
/* Centred within ±150ns tolerance windows from the WS2812B datasheet. */
/* ------------------------------------------------------------------ */

#define T0H_NS   350    /* 0-bit high:  400ns ±150 */
#define T0L_NS   900    /* 0-bit low:   850ns ±150 */
#define T1H_NS   800    /* 1-bit high:  800ns ±150 */
#define T1L_NS   450    /* 1-bit low:   450ns ±150 */
#define RESET_NS 60000  /* reset: >50µs, use 60µs  */

/* ------------------------------------------------------------------ */
/* Module-level state                                                   */
/* ------------------------------------------------------------------ */

static int     s_chip_fd          = -1;
static int     s_line_fd          = -1;
static int64_t s_ioctl_overhead_ns = 200;  /* updated by calibrate() */

/* ------------------------------------------------------------------ */
/* Helper: current time in nanoseconds (CLOCK_MONOTONIC)               */
/* ------------------------------------------------------------------ */

static inline int64_t ns_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + (int64_t)ts.tv_nsec;
}

/* ------------------------------------------------------------------ */
/* Helper: busy-spin until absolute deadline                            */
/* ------------------------------------------------------------------ */

static inline void wait_until(int64_t deadline_ns)
{
    while (ns_now() < deadline_ns)
        ;
}

/* ------------------------------------------------------------------ */
/* Helper: set GPIO output value (0 or 1) via single ioctl             */
/* Declared inline to minimise call overhead on the hot path.          */
/* ------------------------------------------------------------------ */

static inline void gpio_set(int val)
{
    struct gpio_v2_line_values v;
    v.bits = (uint64_t)(val & 1);
    v.mask = 1ULL;
    ioctl(s_line_fd, GPIO_V2_LINE_SET_VALUES_IOCTL, &v);
}

/* ------------------------------------------------------------------ */
/* Calibration: measure median ioctl round-trip time                   */
/*                                                                      */
/* The ioctl call itself consumes ~150–400 ns on a Cortex-A78 under    */
/* SCHED_FIFO. We measure this at startup and subtract it from the     */
/* busy-wait deadline before the falling edge, so the GPIO transition  */
/* lands at the intended time despite syscall latency.                 */
/* ------------------------------------------------------------------ */

#define CAL_ROUNDS 64

static void calibrate(void)
{
    int64_t samples[CAL_ROUNDS];
    struct gpio_v2_line_values v = { .bits = 0, .mask = 1ULL };

    for (int i = 0; i < CAL_ROUNDS; i++) {
        int64_t t0 = ns_now();
        ioctl(s_line_fd, GPIO_V2_LINE_SET_VALUES_IOCTL, &v);
        samples[i] = ns_now() - t0;
    }

    /* Insertion sort (64 elements — negligible cost) */
    for (int i = 1; i < CAL_ROUNDS; i++) {
        int64_t key = samples[i];
        int j = i - 1;
        while (j >= 0 && samples[j] > key) {
            samples[j + 1] = samples[j];
            j--;
        }
        samples[j + 1] = key;
    }

    /* Trimmed mean: discard bottom and top 25% */
    int lo = CAL_ROUNDS / 4;
    int hi = CAL_ROUNDS * 3 / 4;
    int64_t sum = 0;
    for (int i = lo; i < hi; i++) sum += samples[i];
    s_ioctl_overhead_ns = sum / (hi - lo);

    /* Clamp to a sane range */
    if (s_ioctl_overhead_ns < 50)  s_ioctl_overhead_ns = 50;
    if (s_ioctl_overhead_ns > 600) s_ioctl_overhead_ns = 600;
}

/* ------------------------------------------------------------------ */
/* GPIO line setup                                                      */
/* ------------------------------------------------------------------ */

static int open_gpio_line(void)
{
    s_chip_fd = open(CHIP_PATH, O_RDWR | O_CLOEXEC);
    if (s_chip_fd < 0) {
        perror("neopixel: open " CHIP_PATH);
        return -1;
    }

    struct gpio_v2_line_request req;
    memset(&req, 0, sizeof(req));

    req.offsets[0] = LINE_OFFSET;
    req.num_lines  = 1;
    strncpy(req.consumer, CONSUMER, GPIO_MAX_NAME_SIZE - 1);

    /* Output, initially low */
    req.config.flags    = GPIO_V2_LINE_FLAG_OUTPUT;
    req.config.num_attrs = 1;
    req.config.attrs[0].attr.id     = GPIO_V2_LINE_ATTR_ID_OUTPUT_VALUES;
    req.config.attrs[0].attr.values = 0;
    req.config.attrs[0].mask        = 1ULL;

    if (ioctl(s_chip_fd, GPIO_V2_GET_LINE_IOCTL, &req) < 0) {
        perror("neopixel: GPIO_V2_GET_LINE_IOCTL");
        close(s_chip_fd);
        s_chip_fd = -1;
        return -1;
    }

    s_line_fd = req.fd;
    return 0;
}

/* ------------------------------------------------------------------ */
/* Library constructor / destructor                                     */
/* ------------------------------------------------------------------ */

__attribute__((constructor))
static void lib_init(void)
{
    if (open_gpio_line() != 0) return;
    calibrate();
}

__attribute__((destructor))
static void lib_fini(void)
{
    if (s_line_fd >= 0) { close(s_line_fd); s_line_fd = -1; }
    if (s_chip_fd >= 0) { close(s_chip_fd); s_chip_fd = -1; }
}

/* ------------------------------------------------------------------ */
/* Core: write one WS2812B bit                                          */
/*                                                                      */
/* Timing model (example: 1-bit, T1H=800ns, overhead=200ns):           */
/*                                                                      */
/*  t0   gpio_set(1) [ioctl ~200ns]                                     */
/*       spin until t0 + 800 - 200 = t0+600ns                          */
/*       gpio_set(0) [ioctl ~200ns → falling edge at t0+800ns]         */
/*       spin until t0 + 800 + 450 - 200                               */
/*                                                                      */
/* Subtracting overhead before each gpio_set(0/1) call ensures the     */
/* GPIO transition lands at the target time.                            */
/* ------------------------------------------------------------------ */

static inline void write_bit(int bit, int64_t *t_base)
{
    int64_t th   = bit ? T1H_NS : T0H_NS;
    int64_t tl   = bit ? T1L_NS : T0L_NS;
    int64_t t0   = *t_base;
    int64_t ovhd = s_ioctl_overhead_ns;

    gpio_set(1);
    wait_until(t0 + th - ovhd);
    gpio_set(0);
    wait_until(t0 + th + tl - ovhd);

    *t_base = t0 + th + tl;
}

/* ------------------------------------------------------------------ */
/* Public API                                                           */
/*                                                                      */
/* buf : GRB-ordered pixel bytes (as produced by adafruit_pixelbuf     */
/*       after the pixel_order transform)                               */
/* len : byte count (NUM_PIXELS * 3 for GRB, * 4 for GRBW)            */
/*                                                                      */
/* Bits are sent MSB-first within each byte, per WS2812B spec.         */
/* ------------------------------------------------------------------ */

void neopixel_write_c(const uint8_t *buf, int len)
{
    if (s_line_fd < 0 || !buf || len <= 0) return;

    int64_t t = ns_now();

    for (int i = 0; i < len; i++) {
        uint8_t byte = buf[i];
        for (int b = 7; b >= 0; b--) {
            write_bit((byte >> b) & 1, &t);
        }
    }

    /* Reset: hold LOW for at least 50µs */
    gpio_set(0);
    wait_until(t + RESET_NS);
}
