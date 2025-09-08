#ifndef BULLSHIT_H
#define BULLSHIT_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "raylib.h"

/*
  Bullshit environment
  - Action encoding: actions 0..(MAX_RANK*MAX_PLAY_COUNT - 1) represent (declared_rank, declared_count)
    rank = action / MAX_PLAY_COUNT + 1
    count = action % MAX_PLAY_COUNT + 1
  - CALL_BS_ACTION is the final action
  - OBS layout: player0 per-rank(13) + player1 per-rank(13) + player0_total + player1_total +
                pile_size + last_declared_rank + last_declared_count + action masks (ACTIONS_SIZE)
*/

#define MAX_RANK 13
#define MAX_PLAY_COUNT 4
#define CALL_BS_ACTION (MAX_RANK * MAX_PLAY_COUNT)
#define ACTIONS_SIZE (MAX_RANK * MAX_PLAY_COUNT + 1)
#define OBS_SIZE (31 + ACTIONS_SIZE)   // 31 = 13+13+2+1+1+1

#define MAX_PILE_CARDS 52
#define MAX_HAND_CARDS 52

#define TICK_RATE (1.0f/60.0f)
#define MAX_EPISODE_LENGTH 1000

const Color BULL_RED = (Color){187, 0, 0, 255};
const Color BULL_BLUE = (Color){0, 102, 187, 255};
const Color BULL_WHITE = (Color){241, 241, 241, 241};
const Color BULL_BG = (Color){12, 20, 30, 255};

typedef struct Log Log;
struct Log {
    float perf;
    float episode_return;
    float episode_length;
    float n;
};

typedef struct Client Client;
typedef struct CBullshit CBullshit;
struct CBullshit {
    /* RL interface pointers (these will be set by bindings, but allocated for standalone run) */
    float* observations;      // length OBS_SIZE
    int* actions;             // length 1 (action index)
    float* rewards;           // length 1
    unsigned char* terminals; // length 1

    /* Game state */
    int deck[52];
    int deck_index;

    int hands[2][MAX_RANK+1];   // counts of ranks per player (1..13), index 0 unused for convenience
    int hand_totals[2];

    int pile[MAX_PILE_CARDS];
    int pile_size;

    int last_declared_rank;
    int last_declared_count;
    int can_call; /* whether CALL_BS is allowed right now */

    int action_masks[ACTIONS_SIZE];

    /* bookkeeping & logging */
    Log log;
    int game_over;
    int tick;
    float perf;
    float episode_return;
    float episode_length;

    /* rendering / client */
    int width;
    int height;
    Client* client;
};

/* ------------------------------ Prototypes -------------------------------- */
void add_log(CBullshit* env);

void init_cbullshit(CBullshit* env);
void allocate_cbullshit(CBullshit* env);
void c_reset(CBullshit* env);
void compute_observations(CBullshit* env);
void update_action_masks(CBullshit* env);

void place_play(CBullshit* env, int declared_rank, int declared_count, int player_idx);
int get_bot_play(CBullshit* env);
int get_bot_call(CBullshit* env);
void check_call_resolution(CBullshit* env, int caller_idx);

void c_step(CBullshit* env);

Client* make_client_bs(int width, int height);
void c_render(CBullshit* env);

void c_close(CBullshit* env);
void free_allocated_cbullshit(CBullshit* env);

/* --------------------------- Implementations ------------------------------ */

void add_log(CBullshit* env) {
    env->log.perf += env->perf;
    env->log.episode_return += env->episode_return;
    env->log.episode_length += env->episode_length;
    env->log.n += 1.0f;
}

/* Fisher–Yates shuffle */
static void shuffle_deck(int *deck, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = deck[i];
        deck[i] = deck[j];
        deck[j] = tmp;
    }
}

void init_cbullshit(CBullshit* env) {
    /* Initialize deck: 4 of each rank 1..13 */
    int idx = 0;
    for (int r = 1; r <= MAX_RANK; r++) {
        for (int c = 0; c < 4; c++) {
            env->deck[idx++] = r;
        }
    }
    env->deck_index = 0;
    env->pile_size = 0;
    env->last_declared_rank = 0;
    env->last_declared_count = 0;
    env->can_call = 0;

    env->tick = 0;
    env->game_over = 0;
    env->perf = 0.0f;
    env->episode_return = 0.0f;
    env->episode_length = 0.0f;
    memset(&env->log, 0, sizeof(Log));
    env->client = NULL;

    for (int p = 0; p < 2; p++) {
        env->hand_totals[p] = 0;
        for (int r = 0; r <= MAX_RANK; r++) env->hands[p][r] = 0;
    }
    for (int i = 0; i < ACTIONS_SIZE; i++) env->action_masks[i] = 0;
}

/* allocate buffers for standalone runs; binding layer will overwrite these pointers */
void allocate_cbullshit(CBullshit* env) {
    /* allocate observations/actions/rewards/terminals for standalone C */
    env->observations = (float*)calloc(OBS_SIZE, sizeof(float));
    env->actions = (int*)calloc(1, sizeof(int));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->terminals = (unsigned char*)calloc(1, sizeof(unsigned char));

    init_cbullshit(env);
}

/* Free internal allocated buffers; binding may free the struct itself */
void c_close(CBullshit* env) {
    if (env->client != NULL) {
        CloseWindow();
        env->client = NULL;
    }
    /* free any arrays allocated in allocate_cbullshit */
    if (env->observations) { free(env->observations); env->observations = NULL; }
    if (env->actions) { free(env->actions); env->actions = NULL; }
    if (env->rewards) { free(env->rewards); env->rewards = NULL; }
    if (env->terminals) { free(env->terminals); env->terminals = NULL; }
}

void free_allocated_cbullshit(CBullshit* env) {
    c_close(env);
    /* Note: if binding allocated the struct (VecEnv), binding is responsible for free(env) */
}

/* Reset: shuffle deck, deal evenly to two players, reset pile and flags */
void c_reset(CBullshit* env) {
    init_cbullshit(env);
    shuffle_deck(env->deck, 52);
    env->deck_index = 0;

    /* deal alternately to player 0 and 1 */
    int p = 0;
    for (int i = 0; i < 52; i++) {
        int r = env->deck[i];
        env->hands[p][r] += 1;
        env->hand_totals[p] += 1;
        p = 1 - p;
    }
    env->deck_index = 52; /* deck fully dealt */
    env->pile_size = 0;
    env->last_declared_rank = 0;
    env->last_declared_count = 0;
    env->can_call = 0;
    env->terminals[0] = 0;
    env->rewards[0] = 0.0f;
    env->episode_length = 0;
    env->episode_return = 0.0f;

    compute_observations(env);
}

/* Observation layout:
   0..12: player0 per-rank (1..13)
   13..25: player1 per-rank
   26: player0 total
   27: player1 total
   28: pile size
   29: last_declared_rank
   30: last_declared_count
   31.. : action masks (ACTIONS_SIZE floats)
*/
void compute_observations(CBullshit* env) {
    int idx = 0;
    for (int r = 1; r <= MAX_RANK; r++) env->observations[idx++] = (float)env->hands[0][r];
    for (int r = 1; r <= MAX_RANK; r++) env->observations[idx++] = (float)env->hands[1][r];
    env->observations[idx++] = (float)env->hand_totals[0];
    env->observations[idx++] = (float)env->hand_totals[1];
    env->observations[idx++] = (float)env->pile_size;
    env->observations[idx++] = (float)env->last_declared_rank;
    env->observations[idx++] = (float)env->last_declared_count;
    for (int i = 0; i < ACTIONS_SIZE; i++) env->observations[idx++] = (float)env->action_masks[i];
}

/* Update masks: prevent playing more cards than you have; CALL_BS only if can_call */
void update_action_masks(CBullshit* env) {
    int my_total = env->hand_totals[0];
    for (int a = 0; a < MAX_RANK * MAX_PLAY_COUNT; a++) {
        int declared_count = (a % MAX_PLAY_COUNT) + 1;
        if (my_total < declared_count || my_total == 0) env->action_masks[a] = 1;
        else env->action_masks[a] = 0;
    }
    env->action_masks[CALL_BS_ACTION] = env->can_call ? 0 : 1;
}

/* Add cards (array of ranks) to player's hand */
static void add_cards_to_hand(CBullshit* env, int player_idx, int *cards, int n) {
    for (int i = 0; i < n; i++) {
        int r = cards[i];
        if (r >= 1 && r <= MAX_RANK) {
            env->hands[player_idx][r] += 1;
            env->hand_totals[player_idx] += 1;
        }
    }
}

/* Place a play: deterministically remove declared_count cards from player and append their real ranks to pile.
   Removal rule: take from lowest rank upward until declared_count removed. */
void place_play(CBullshit* env, int declared_rank, int declared_count, int player_idx) {
    int removed = 0;
    for (int r = 1; r <= MAX_RANK && removed < declared_count; r++) {
        while (env->hands[player_idx][r] > 0 && removed < declared_count) {
            env->hands[player_idx][r] -= 1;
            env->hand_totals[player_idx] -= 1;
            if (env->pile_size < MAX_PILE_CARDS) {
                env->pile[env->pile_size++] = r;
            }
            removed++;
        }
    }
    /* record declared info */
    env->last_declared_rank = declared_rank;
    env->last_declared_count = declared_count;
    env->can_call = 1;
}

/* Simple bot heuristics */
int get_bot_play(CBullshit* env) {
    int bot = 1;
    int ranks_with_cards[MAX_RANK];
    int n = 0;
    for (int r = 1; r <= MAX_RANK; r++) {
        if (env->hands[bot][r] > 0) ranks_with_cards[n++] = r;
    }
    int chosen_rank = (n > 0) ? ranks_with_cards[rand() % n] : ((rand() % MAX_RANK) + 1);
    int available = env->hand_totals[bot];
    int max_allowed = (available < MAX_PLAY_COUNT) ? available : MAX_PLAY_COUNT;
    int chosen_count = (max_allowed > 1) ? (rand() % max_allowed) + 1 : 1;
    int action = (chosen_rank - 1) * MAX_PLAY_COUNT + (chosen_count - 1);
    return action;
}

int get_bot_call(CBullshit* env) {
    float p = 0.12f;
    if (env->last_declared_count >= 3) p = 0.35f;
    float r = (float)rand() / (float)RAND_MAX;
    return (r < p) ? 1 : 0;
}

/* Caller resolves a BS call. caller_idx is the calling player (0 human, 1 bot).
   If the last declared segment is entirely equal to declared_rank -> caller was wrong -> caller picks up pile.
   Otherwise the play_player picks up the pile. */
void check_call_resolution(CBullshit* env, int caller_idx) {
    if (env->pile_size == 0 || env->last_declared_count == 0) {
        env->can_call = 0;
        return;
    }

    int play_player = 1 - caller_idx;
    int declared = env->last_declared_rank;
    int declared_count = env->last_declared_count;
    int start_idx = env->pile_size - declared_count;
    if (start_idx < 0) start_idx = 0;

    int all_match = 1;
    for (int i = start_idx; i < env->pile_size; i++) {
        if (env->pile[i] != declared) { all_match = 0; break; }
    }

    /* copy pile */
    int temp[MAX_PILE_CARDS];
    int ntemp = env->pile_size;
    for (int i = 0; i < env->pile_size; i++) temp[i] = env->pile[i];

    /* clear pile and flags */
    env->pile_size = 0;
    env->last_declared_rank = 0;
    env->last_declared_count = 0;
    env->can_call = 0;

    if (all_match) {
        /* caller wrong -> caller picks up pile */
        add_cards_to_hand(env, caller_idx, temp, ntemp);
        if (caller_idx == 0) {
            env->rewards[0] -= 0.5f; env->episode_return -= 0.5f;
        }
    } else {
        /* liar picks up pile */
        add_cards_to_hand(env, play_player, temp, ntemp);
        if (play_player == 0) {
            env->rewards[0] -= 0.5f; env->episode_return -= 0.5f;
        } else {
            env->rewards[0] += 0.5f; env->episode_return += 0.5f;
        }
    }

    /* detect terminal */
    if (env->hand_totals[0] == 0 || env->hand_totals[1] == 0) {
        env->terminals[0] = 1;
        env->game_over = 1;
        if (env->hand_totals[0] == 0 && env->hand_totals[1] != 0) { env->rewards[0] += 1.0f; env->episode_return += 1.0f; env->perf = 1.0f; }
        else if (env->hand_totals[1] == 0 && env->hand_totals[0] != 0) { env->rewards[0] -= 1.0f; env->episode_return -= 1.0f; env->perf = 0.0f; }
        else { env->perf = 0.0f; }
    }
}

/* One-step environment update
   - reads env->actions[0]
   - apply play or CALL_BS
   - resolves bot responses within the same step
   - updates rewards, observations, masks
*/
void c_step(CBullshit* env) {
    env->episode_length += 1;
    env->rewards[0] = 0.0f;

    if (env->episode_length >= MAX_EPISODE_LENGTH) {
        env->game_over = 1;
        env->episode_return -= 1.0f;
        env->rewards[0] -= 1.0f;
    }

    if (env->game_over == 1) {
        env->perf = (env->hand_totals[0] == 0) ? 1.0f : 0.0f;
        add_log(env);
        c_reset(env);
        return;
    }

    int action = env->actions[0];

    /* invalid action guard */
    if (action < 0 || action >= ACTIONS_SIZE) {
        env->episode_return -= 0.1f; env->rewards[0] -= 0.1f;
    } else if (action == CALL_BS_ACTION) {
        if (env->can_call) {
            check_call_resolution(env, 0);
        } else {
            env->episode_return -= 0.05f; env->rewards[0] -= 0.05f;
        }
    } else {
        /* play action */
        int declared_rank = (action / MAX_PLAY_COUNT) + 1;
        int declared_count = (action % MAX_PLAY_COUNT) + 1;
        if (declared_count > env->hand_totals[0] || env->hand_totals[0] == 0) {
            env->episode_return -= 0.1f; env->rewards[0] -= 0.1f;
        } else {
            place_play(env, declared_rank, declared_count, 0);

            /* Bot may call or play */
            if (env->can_call) {
                if (get_bot_call(env)) {
                    check_call_resolution(env, 1);
                } else {
                    int bot_action = get_bot_play(env);
                    int bot_rank = (bot_action / MAX_PLAY_COUNT) + 1;
                    int bot_count = (bot_action % MAX_PLAY_COUNT) + 1;
                    if (bot_count > env->hand_totals[1]) bot_count = env->hand_totals[1] > 0 ? 1 : 0;
                    if (bot_count > 0) {
                        place_play(env, bot_rank, bot_count, 1);
                        /* after bot play, human can CALL_BS on next step */
                    }
                }
            }
        }
    }

    /* check terminal after step */
    if (env->hand_totals[0] == 0 || env->hand_totals[1] == 0) {
        env->game_over = 1;
        env->terminals[0] = 1;
        if (env->hand_totals[0] == 0 && env->hand_totals[1] != 0) {
            env->rewards[0] += 1.0f; env->episode_return += 1.0f; env->perf = 1.0f;
        } else if (env->hand_totals[1] == 0 && env->hand_totals[0] != 0) {
            env->rewards[0] -= 1.0f; env->episode_return -= 1.0f; env->perf = 0.0f;
        } else {
            /* draw */
            env->perf = 0.0f;
        }
    }

    update_action_masks(env);
    compute_observations(env);
}

/* Minimal client for rendering */
struct Client { float width; float height; };

Client* make_client_bs(int width, int height) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->width = width;
    client->height = height;
    InitWindow(width, height, "PufferLib Ray Bullshit");
    SetTargetFPS(60);
    return client;
}

void c_render(CBullshit* env) {
    if (IsKeyDown(KEY_ESCAPE)) exit(0);
    if (env->client == NULL) env->client = make_client_bs(env->width, env->height);

    BeginDrawing();
    ClearBackground(BULL_BG);

    DrawText(TextFormat("Pile size: %d", env->pile_size), 20, 20, 20, BULL_WHITE);
    DrawText(TextFormat("Last declared: %d x %d", env->last_declared_rank, env->last_declared_count), 20, 40, 20, BULL_WHITE);

    DrawText(TextFormat("You: %d cards", env->hand_totals[0]), 20, env->height - 80, 30, BULL_BLUE);
    DrawText(TextFormat("Bot: %d cards", env->hand_totals[1]), env->width - 200, env->height - 80, 30, BULL_RED);

    /* per-rank counts for both players (debug view) */
    for (int r = 1; r <= MAX_RANK; r++) {
        int x = 20 + (r - 1) * 36;
        DrawText(TextFormat("%d", env->hands[0][r]), x, env->height - 50, 12, BULL_WHITE);
        DrawText(TextFormat("%d", env->hands[1][r]), x, env->height - 70, 12, BULL_WHITE);
    }

    DrawText("Press B to call BS (manual). Otherwise agent acts automatically.", 20, env->height - 120, 12, BULL_WHITE);

    EndDrawing();
}

#endif /* BULLSHIT_H */
