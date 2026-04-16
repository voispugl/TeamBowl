#include <stdio.h>
#include <stdlib.h>
#include "pico/stdlib.h"
#include "hardware/pio.h"
#include "hardware/clocks.h"
#include "hardware/sync.h"
#include "hardware/i2c.h"
#include "hardware/timer.h"
#include "hardware/structs/ioqspi.h"
#include "hardware/structs/sio.h"
#include "ws2812.pio.h"

// ── Hardware config ────────────────────────────────────────────────────────────
#define PIN_TX       28
#define NUM_PIXELS   10
#define BRIGHTNESS   0.3f

#define I2C_SDA_PIN  4
#define I2C_SCL_PIN  5
#define I2C_ADDR     0x2A   // slave address seen by the I2C master

// ── I2C command bytes ──────────────────────────────────────────────────────────
// 1-byte commands:
//   0x00–0x03   set predefined color index (Red/Green/Yellow/Purple)
//   0x20        wave right (moving dot with fade tail)
//   0x21        wave left
//   0x22        stop animation → static
//   0x30        sequential fill right (Corvette-style)
//   0x31        sequential fill left
// 4-byte command:
//   0x10 R G B  set custom RGB color (static)
#define CMD_CUSTOM_RGB  0x10
#define CMD_WAVE_RIGHT  0x20
#define CMD_WAVE_LEFT   0x21
#define CMD_STOP        0x22
#define CMD_SEQ_RIGHT   0x30
#define CMD_SEQ_LEFT    0x31

#define WAVE_TAIL_LEN   6   // pixels in the fade tail behind the wave head

// ── LED animation mode ─────────────────────────────────────────────────────────
typedef enum {
    MODE_STATIC,
    MODE_WAVE_RIGHT,
    MODE_WAVE_LEFT,
    MODE_SEQ_RIGHT,   // Corvette sequential fill, left→right
    MODE_SEQ_LEFT,    // Corvette sequential fill, right→left
} led_mode_t;

// ── BOOTSEL button (RP2350 — must run from RAM) ────────────────────────────────
bool __no_inline_not_in_flash_func(get_bootsel_button)() {
    const uint CS_PIN_INDEX = 1;
    uint32_t flags = save_and_disable_interrupts();

    uint32_t original_ctrl = ioqspi_hw->io[CS_PIN_INDEX].ctrl;
    hw_write_masked(&ioqspi_hw->io[CS_PIN_INDEX].ctrl,
                    IO_QSPI_GPIO_QSPI_SS_CTRL_OEOVER_VALUE_DISABLE << IO_QSPI_GPIO_QSPI_SS_CTRL_OEOVER_LSB,
                    IO_QSPI_GPIO_QSPI_SS_CTRL_OEOVER_BITS);
    busy_wait_at_least_cycles(1000);
    bool button_state = !(sio_hw->gpio_hi_in & SIO_GPIO_HI_IN_QSPI_CSN_BITS);
    ioqspi_hw->io[CS_PIN_INDEX].ctrl = original_ctrl;

    restore_interrupts(flags);
    return button_state;
}

// ── Pixel helpers ──────────────────────────────────────────────────────────────

// Build a GRB word with global brightness × per-pixel scale applied.
static inline uint32_t pixel_color(uint8_t r, uint8_t g, uint8_t b, float scale) {
    float s = BRIGHTNESS * scale;
    return ((uint32_t)((uint8_t)((float)g * s)) << 16) |
           ((uint32_t)((uint8_t)((float)r * s)) << 8)  |
           (uint32_t) ((uint8_t)((float)b * s));
}

static inline void put_pixel(uint32_t pixel_grb) {
    pio_sm_put_blocking(pio0, 0, pixel_grb << 8u);
}

// ── Predefined colors (raw RGB) — used by I2C 0x00–0x03 commands ──────────────
static const uint8_t base_colors[4][3] = {
    {255,   0,   0},   // 0 Red
    {  0, 255,   0},   // 1 Green
    {255, 255,   0},   // 2 Yellow
    {128,   0, 128},   // 3 Purple
};

// ── BOOTSEL step sequence ──────────────────────────────────────────────────────
typedef struct { led_mode_t mode; uint8_t r, g, b; } led_step_t;

static const led_step_t steps[] = {
    {MODE_STATIC,       0, 255,   0},   // Green
    {MODE_STATIC,     255,   0,   0},   // Red

    {MODE_STATIC,     255, 255,   0},   // Yellow
    {MODE_STATIC,     128,   0, 128},   // Purple
    {MODE_WAVE_RIGHT, 255, 120,   0},   // Wave right  (orange)
    {MODE_WAVE_LEFT,  255, 120,   0},   // Wave left   (orange)
    {MODE_SEQ_RIGHT,  255, 120,   0},   // Seq right   (orange)
    {MODE_SEQ_LEFT,   255, 120,   0},   // Seq left    (orange)
};
#define NUM_STEPS ((int)(sizeof(steps) / sizeof(steps[0])))

// ── Built-in LED heartbeat ─────────────────────────────────────────────────────
static bool led_blink_cb(repeating_timer_t *rt) {
    gpio_xor_mask(1u << PICO_DEFAULT_LED_PIN);
    return true;
}

// Wave tail brightness ramp (index 0 = head)
static const float tail_scales[WAVE_TAIL_LEN] = {1.0f, 0.7f, 0.5f, 0.3f, 0.2f, 0.1f};

// ── Main ───────────────────────────────────────────────────────────────────────
int main() {
    stdio_init_all();

    // Built-in green LED — blinks at 2 Hz always
    gpio_init(PICO_DEFAULT_LED_PIN);
    gpio_set_dir(PICO_DEFAULT_LED_PIN, GPIO_OUT);
    repeating_timer_t blink_timer;
    add_repeating_timer_ms(-250, led_blink_cb, NULL, &blink_timer);

    // PIO — WS2812
    PIO pio = pio0;
    int sm = 0;
    uint offset = pio_add_program(pio, &ws2812_program);
    ws2812_program_init(pio, sm, offset, PIN_TX, 800000, false);

    // I2C slave
    i2c_init(i2c0, 100000);
    gpio_set_function(I2C_SDA_PIN, GPIO_FUNC_I2C);
    gpio_set_function(I2C_SCL_PIN, GPIO_FUNC_I2C);
    gpio_pull_up(I2C_SDA_PIN);
    gpio_pull_up(I2C_SCL_PIN);
    i2c_set_slave_mode(i2c0, true, I2C_ADDR);

    // State
    led_mode_t mode  = MODE_STATIC;
    int  step_idx    = 0;
    uint8_t cur_r    = steps[0].r;
    uint8_t cur_g    = steps[0].g;
    uint8_t cur_b    = steps[0].b;

    int  wave_pos  = 0;
    int  seq_pos   = 0;
    int  seq_hold  = 0;

    bool last_button_state = false;

    // I2C receive buffer (max 4 bytes per command)
    uint8_t cmd_buf[4];
    int     cmd_len = 0;

    while (true) {

        // ── BOOTSEL button ────────────────────────────────────────────────────
        bool btn = get_bootsel_button();
        if (btn && !last_button_state) {
            step_idx = (step_idx + 1) % NUM_STEPS;
            const led_step_t *s = &steps[step_idx];
            cur_r = s->r;  cur_g = s->g;  cur_b = s->b;
            mode  = s->mode;
            // reset animation counters when entering a new mode
            wave_pos = (mode == MODE_WAVE_LEFT) ? NUM_PIXELS - 1 : 0;
            seq_pos  = 0;
            seq_hold = 0;
            printf("Button: step %d  mode %d\n", step_idx, mode);
        }
        last_button_state = btn;

        // ── I2C command polling ───────────────────────────────────────────────
        while (i2c_get_read_available(i2c0)) {
            uint8_t b;
            i2c_read_raw_blocking(i2c0, &b, 1);
            cmd_buf[cmd_len++] = b;

            bool done = false;

            if (cmd_buf[0] <= 0x03 && cmd_len == 1) {
                // Predefined color index
                uint8_t idx = cmd_buf[0];
                cur_r = base_colors[idx][0];
                cur_g = base_colors[idx][1];
                cur_b = base_colors[idx][2];
                mode = MODE_STATIC;
                done = true;

            } else if (cmd_buf[0] == CMD_CUSTOM_RGB && cmd_len == 4) {
                cur_r = cmd_buf[1];
                cur_g = cmd_buf[2];
                cur_b = cmd_buf[3];
                mode = MODE_STATIC;
                done = true;

            } else if (cmd_buf[0] == CMD_WAVE_RIGHT && cmd_len == 1) {
                mode = MODE_WAVE_RIGHT;
                wave_pos = 0;
                done = true;

            } else if (cmd_buf[0] == CMD_WAVE_LEFT && cmd_len == 1) {
                mode = MODE_WAVE_LEFT;
                wave_pos = NUM_PIXELS - 1;
                done = true;

            } else if (cmd_buf[0] == CMD_STOP && cmd_len == 1) {
                mode = MODE_STATIC;
                done = true;

            } else if (cmd_buf[0] == CMD_SEQ_RIGHT && cmd_len == 1) {
                mode = MODE_SEQ_RIGHT;
                seq_pos = 0;
                seq_hold = 0;
                done = true;

            } else if (cmd_buf[0] == CMD_SEQ_LEFT && cmd_len == 1) {
                mode = MODE_SEQ_LEFT;
                seq_pos = 0;
                seq_hold = 0;
                done = true;
            }

            if (cmd_len >= 4) done = true;   // garbage or completed 4-byte cmd
            if (done) cmd_len = 0;
        }

        // ── Render ────────────────────────────────────────────────────────────
        switch (mode) {

            case MODE_STATIC: {
                for (int i = 0; i < NUM_PIXELS; i++)
                    put_pixel(pixel_color(cur_r, cur_g, cur_b, 1.0f));
                sleep_ms(20);
                break;
            }

            case MODE_WAVE_RIGHT:
            case MODE_WAVE_LEFT: {
                for (int i = 0; i < NUM_PIXELS; i++) {
                    int dist = (mode == MODE_WAVE_RIGHT)
                        ? (wave_pos - i + NUM_PIXELS) % NUM_PIXELS
                        : (i - wave_pos + NUM_PIXELS) % NUM_PIXELS;
                    put_pixel(dist < WAVE_TAIL_LEN
                        ? pixel_color(cur_r, cur_g, cur_b, tail_scales[dist])
                        : 0);
                }
                wave_pos = (mode == MODE_WAVE_RIGHT)
                    ? (wave_pos + 1) % NUM_PIXELS
                    : (wave_pos - 1 + NUM_PIXELS) % NUM_PIXELS;
                sleep_ms(30);
                break;
            }

            case MODE_SEQ_RIGHT:
            case MODE_SEQ_LEFT: {
                // Render current fill level
                for (int i = 0; i < NUM_PIXELS; i++) {
                    bool lit = (mode == MODE_SEQ_RIGHT)
                        ? (i < seq_pos)
                        : (i >= NUM_PIXELS - seq_pos);
                    put_pixel(lit ? pixel_color(cur_r, cur_g, cur_b, 1.0f) : 0);
                }
                // Advance: fill one LED per frame; hold briefly when full; reset
                if (seq_pos < NUM_PIXELS) {
                    seq_pos++;
                } else if (seq_hold < 5) {
                    seq_hold++;
                } else {
                    seq_pos  = 0;
                    seq_hold = 0;
                }
                sleep_ms(40);
                break;
            }
        }
    }

    return 0;
}
