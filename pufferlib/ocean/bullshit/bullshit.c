#include "Bullshit.h"
#include <time.h>

#define NOOP -1

/* Simple interactive driver and perf test modeled after tripletriad.c */

void interactive() {
    CBullshit env = {
        .width = 900,
        .height = 500,
        .game_over = 0,
        .tick = 0
    };

    /* allocate env for standalone mode (bindings overwrite these pointers when used from Python) */
    allocate_cbullshit(&env);
    c_reset(&env);

    env.client = make_client_bs(env.width, env.height);

    int tick = 0;
    int action;
    srand(time(NULL));

    while (!WindowShouldClose()) {
        action = NOOP;

        /* Manual control: if left shift held and B pressed => call BS */
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            if (IsKeyPressed(KEY_B)) {
                action = CALL_BS_ACTION;
            }
            /* manual play selection is intentionally simple here; for fine-grained manual play you'd need
               a UI to select rank and count — omitted to keep parity with the automated flow. */
        } else if (tick % 45 == 0) {
            /* choose a random legal action (simple agent) */
            update_action_masks(&env);
            /* find legal actions */
            int legal[ACTIONS_SIZE];
            int n_legal = 0;
            for (int i = 0; i < ACTIONS_SIZE; i++) {
                if (env.action_masks[i] == 0) legal[n_legal++] = i;
            }
            if (n_legal > 0) action = legal[rand() % n_legal];
            else action = NOOP;
        }

        tick = (tick + 1) % 45;

        if (action != NOOP) {
            env.actions[0] = action;
            c_step(&env);
        }

        c_render(&env);
    }

    c_close(env.client ? &env : NULL); // ensure window closed
    free_allocated_cbullshit(&env);
}

/* performance test similar to tripletriad.c */
void performance_test() {
    long test_time = 10;
    CBullshit env = {
        .width = 900,
        .height = 500,
        .game_over = 0
    };
    allocate_cbullshit(&env);
    c_reset(&env);

    long start = time(NULL);
    int i = 0;
    while (time(NULL) - start < test_time) {
        /* random legal action each step */
        update_action_masks(&env);
        int legal[ACTIONS_SIZE];
        int n_legal = 0;
        for (int a = 0; a < ACTIONS_SIZE; a++) if (env.action_masks[a] == 0) legal[n_legal++] = a;
        if (n_legal > 0) env.actions[0] = legal[rand() % n_legal];
        else env.actions[0] = NOOP;
        c_step(&env);
        i++;
    }
    long end = time(NULL);
    printf("SPS: %ld\n", i / (end - start));
    free_allocated_cbullshit(&env);
}

int main() {
    /* Uncomment to run perf test instead */
    // performance_test();
    interactive();
    return 0;
}
