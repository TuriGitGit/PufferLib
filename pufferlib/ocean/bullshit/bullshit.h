#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "raylib.h"

/*
  Bullshit environment
  - Action encoding: actions 0..(RANKS*SUITS - 1) represent (declared_rank, declared_count)
    rank = action / SUITS + 1
    count = action % SUITS + 1
  - CALL_BS_ACTION is the final action
  - OBS layout: player0 per-rank(13) + player1 per-rank(13) + player0_total + player1_total +
                pile_size + last_declared_rank + last_declared_count + action masks (ACTIONS_SIZE)
*/

#define RANKS 13
#define SUITS 4
#define CALL_BS_ACTION (RANKS * SUITS)
#define ACTIONS_SIZE (RANKS * SUITS + 1)
#define OBS_SIZE (31 + ACTIONS_SIZE)   // 31 = 13+13+2+1+1+1

#define DECK_SIZE 52
#define PLAYERS 4

static const int DECK[DECK_SIZE] = {
    1,1,1,1, 2,2,2,2, 3,3,3,3,
    4,4,4,4, 5,5,5,5, 6,6,6,6,
    7,7,7,7, 8,8,8,8, 9,9,9,9,
    10,10,10,10,  11,11,11,11,
    12,12,12,12,  13,13,13,13
};

#define MAX_EPISODE_LENGTH 1000

const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 255};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};

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
    float* observations;
    int* actions;
    float* rewards;
    unsigned char* terminals;
    Log log;
    int game_over;
    int tick;
    float perf;
    float episode_return;
    float episode_length;

    int deck[DECK_SIZE];
    int deck_index;

    int hands[PLAYERS][RANKS];
    int hand_totals[PLAYERS];
    int claimed[RANKS];
    int known[RANKS];

    int pile[DECK_SIZE];
    int pile_size;

    int last_declared_rank;
    int last_declared_count;
    int can_call;

    int action_masks[ACTIONS_SIZE];

    int width;
    int height;
    Client* client;
};

void addLog(CBullshit* env) {
    env->log.perf += env->perf;
    env->log.episode_return += env->episode_return;
    env->log.episode_length += env->episode_length;
    env->log.n += 1.0f;
}

/* Fisher–Yates shuffle */
static inline void shuffle(int *deck) {
    for (int i = DECK_SIZE - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = deck[i];
        deck[i] = deck[j];
        deck[j] = tmp;
    }
}

void initCBullShit(CBullshit* env) {
    /* Initialize deck: 4 of each rank 1..13 */
    memcpy(env->deck, DECK, sizeof(DECK));

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

    for (int p = 0; p < PLAYERS; p++) {
        env->hand_totals[p] = 0;
        for (int r = 0; r < RANKS; r++) env->hands[p][r] = 0;
    }
    for (int i = 0; i < ACTIONS_SIZE; i++) env->action_masks[i] = 0;
}

static inline void allocateCBullShit(CBullshit* env) {
    env->actions = (int*)calloc(1, sizeof(int));
    env->observations = (float*)calloc(OBS_SIZE, sizeof(float));
    env->terminals = (unsigned char*)calloc(1, sizeof(unsigned char));
    env->rewards = (float*)calloc(1, sizeof(float));
    initCBullShit(env);
}

static inline void c_close(CBullshit* env) {
    if (env->client != NULL)closeClient(env->client); env->client = NULL;
    free(env->observations); env->observations = NULL;
    free(env->actions); env->actions = NULL;
    free(env->rewards); env->rewards = NULL;
    free(env->terminals); env->terminals = NULL;
}

void freeCBullshit(CBullshit* env) {
    c_close(env);
}

void c_reset(CBullshit* env) {
    initCBullShit(env);
    shuffle(env->deck);
    env->deck_index = 0;

    for (int i = 0; i < DECK_SIZE; i+=PLAYERS) {
        int r0 = env->deck[i];
        int r1 = env->deck[i+1];
        int r2 = env->deck[i+2];
        int r3 = env->deck[i+3];
        env->hands[0][r0-1] += 1;
        env->hands[1][r1-1] += 1;
        env->hands[2][r2-1] += 1;
        env->hands[3][r3-1] += 1;
    }
    env->hand_totals[0] += DECK_SIZE/PLAYERS;
    env->hand_totals[1] += DECK_SIZE/PLAYERS;
    env->hand_totals[2] += DECK_SIZE/PLAYERS;
    env->hand_totals[3] += DECK_SIZE/PLAYERS;

    env->deck_index = DECK_SIZE;
    env->pile_size = 0;
    env->last_declared_rank = 0;
    env->last_declared_count = 0;
    computeObs(env); //  i think it goes here but my brain is dying
    env->can_call = 0;
    env->terminals[0] = 0;
    env->rewards[0] = 0.0f;
    env->episode_length = 0;
    env->episode_return = 0.0f;
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

/* Observation supposed to be layout:
   0..12: player0 hand per-rank (1..13)
   13..25: claims history (1..13)
   26..38: guarnteed history (1..13) // this is what the nn placed and what has been revealed
   39: player1 total
   40: player2 total
   41: player3 total
   42: pile size
   43..55: last_declared_rank
   56: last_declared_count
   57..109 : action masks (53 floats)
*/

static inline void computeObs(CBullshit* env) {
    int idx = 0;
    for (int r = 0; r < RANKS; r++) env->observations[idx++] = (float)env->hands[0][r];
    for (int r = 0; r < RANKS; r++) env->observations[idx++] = (float)env->hands[1][r];
    env->observations[idx++] = (float)env->hand_totals[0];
    env->observations[idx++] = (float)env->hand_totals[1];
    env->observations[idx++] = (float)env->pile_size;
    env->observations[idx++] = (float)env->last_declared_rank;
    env->observations[idx++] = (float)env->last_declared_count;
    for (int i = 0; i < ACTIONS_SIZE; i++) env->observations[idx++] = (float)env->action_masks[i];
}

static inline void updateActionMasks(CBullshit* env) {
    int agent_total = env->hand_totals[0];
    for (int a = 0; a < RANKS * SUITS; a++) {
        int declared_count = 1 + (a % SUITS);
        env->action_masks[a] = (agent_total < declared_count);
    }
    env->action_masks[CALL_BS_ACTION] = !env->can_call;
}

static void addCardsToHand(CBullshit* env, int player, int *cards, int n) {
    for (int i = 0; i < n; i++) {
        int r = cards[i] -1;
        env->hands[player][r] += 1;
    }
    env->hand_totals[player] += n;
}

static inline void placeCards(CBullshit* env, int declared_rank, int declared_count, int player) { 
    env->hands[player][declared_rank] -= declared_count;
    env->hand_totals[player] -= declared_count;
   for (int i = 0; i < declared_count; i++) {
        env->pile[env->pile_size++] = declared_rank;
    }
    env->last_declared_rank = declared_rank;
    env->last_declared_count = declared_count;
    env->can_call = 1;
}

static inline int botAct(CBullshit* env, int bot) {
    int rank = -1;
    int count = 0;
    int total_ranks = 0;
    for (int r = 0; r < RANKS; r++) {
        if (env->hands[bot][r] > 0) {
            total_ranks++;
            if (rand() % total_ranks == 0) {
                rank = r;
                count = env->hands[bot][r];
            }
        }
    }
    int chosen_count = (count > 1) ? rand() % count + 1 : 1;
    return rank * SUITS + (chosen_count - 1);
}



static inline int botCallBS(CBullshit* env) { 
    float p = 0.12f;
    p *= env->last_declared_count;
    float r = (float)rand() / (float)RAND_MAX;
    return (r < p);
}

void resolveCall(CBullshit* env, int caller_idx) {
    if (env->pile_size == 0 || env->last_declared_count == 0) {
        env->can_call = 0;
        return;
    }

    int play_player = (caller_idx + PLAYERS - 1) % PLAYERS;
    int declared = env->last_declared_rank;
    int start_idx = env->pile_size - env->last_declared_count;

    int all_match = 1;
    int* p = &env->pile[start_idx];
    int* e = &env->pile[env->pile_size];
    for (; p < e; p++) {
        if (*p != declared) { all_match = 0; break; }
    }

    if (all_match) {
        addCardsToHand(env, caller_idx, env->pile, env->pile_size);
        if (caller_idx == 0) {
            env->rewards[0] -= 0.5f; env->episode_return -= 0.5f;
        }
    } else {
        addCardsToHand(env, play_player, env->pile, env->pile_size);
        if (play_player == 0) {
            env->rewards[0] -= 0.5f; env->episode_return -= 0.5f;
        } else {
            env->rewards[0] += 0.5f; env->episode_return += 0.5f;
        }
    }
    env->pile_size = 0;
    env->last_declared_rank = 0;
    env->last_declared_count = 0;
    env->can_call = 0;
}

/* One-step environment update
   - reads env->actions[0]
   - apply play or CALL_BS
   - resolves bot responses within the same step // use a for loop for simplicity and then use #pragma unroll <BOTS>
   - updates rewards, observations, masks
*/
void c_step(CBullshit* env) {
    env->episode_length += 1;
    env->rewards[0] = 0.0f;
    int action = env->actions[0];

    if (env->episode_length >= MAX_EPISODE_LENGTH) {
        env->game_over = 1;
        env->episode_return -= 1.0f;
        env->rewards[0] -= 1.0f;
    }

    if (env->game_over) {
        env->perf = (env->hand_totals[0] == 0) ? 1.0f : 0.0f;
        addLog(env);
        c_reset(env);
        return;
    }


    /* invalid action guard */
    if (action < 0 || action >= ACTIONS_SIZE) {
        env->episode_return -= 0.1f; env->rewards[0] -= 0.1f;
    } else if (action == CALL_BS_ACTION) {
        if (env->can_call) {
            resolveCall(env, 0);
        } else {
            env->episode_return -= 0.05f; env->rewards[0] -= 0.05f;
        }
    } else {
        /* play action */
        int declared_rank = action / SUITS;
        int declared_count = 1+ (action % SUITS);
        if (declared_count > env->hand_totals[0] || env->hand_totals[0] == 0) {
            env->episode_return -= 0.1f; env->rewards[0] -= 0.1f;
        } else {
            placeCards(env, declared_rank, declared_count, 0);

            /* Bot may call or play */ // TODO: make multiple different bots
            if (env->can_call) {
                if (botCallBS(env)) {
                    resolveCall(env, 1);
                } else {
                    int bot = 1; // TODO: make turn based selection
                    int bot_action = botAct(env, bot);
                    int bot_rank = bot_action / SUITS;
                    int bot_count = 1+ (bot_action % SUITS);
                    if (bot_count > env->hand_totals[1]) bot_count = env->hand_totals[1] > 0 ? 1 : 0;
                    if (bot_count > 0) {
                        placeCards(env, bot_rank, bot_count, 1);
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

    updateActionMasks(env);
    computeObs(env);
}

/* Minimal client for rendering */
struct Client { float width; float height; };

Client* createBullshitClient(int width, int height) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->width = width;
    client->height = height;
    InitWindow(width, height, "PufferLib Ray Bullshit");
    SetTargetFPS(60);
    return client;
}

void c_render(CBullshit* env) {
    if (IsKeyDown(KEY_ESCAPE)) exit(0);
    if (env->client == NULL) env->client = createBullshitClient(env->width, env->height);

    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);

    DrawText(TextFormat("Pile size: %d", env->pile_size), 20, 20, 20, PUFF_WHITE);
    DrawText(TextFormat("Last declared: %d x %d", 1+ env->last_declared_rank, env->last_declared_count), 20, 40, 20, PUFF_WHITE);

    DrawText(TextFormat("Agent: %d cards", env->hand_totals[0]), 20, env->height - 80, 30, PUFF_CYAN);
    DrawText(TextFormat("Bot: %d cards", env->hand_totals[1]), env->width - 200, env->height - 80, 30, PUFF_RED);

    /* per-rank counts for both players (debug view) */
    for (int r = 0; r < RANKS; r++) {
        int x = 20 + r * 36;
        DrawText(TextFormat("%d", env->hands[0][r]), x, env->height - 50, 12, PUFF_WHITE);
        DrawText(TextFormat("%d", env->hands[1][r]), x, env->height - 70, 12, PUFF_WHITE);
    }

    DrawText("Press B to call BS (manual). Otherwise agent acts automatically.", 20, env->height - 120, 12, PUFF_WHITE); // TODO: DIE

    EndDrawing();
}

/* stolen from TT*/
void closeClient(Client* client) {
    CloseWindow();
    free(client);
}
