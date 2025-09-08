#pragma once

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

#define DECK_SIZE 52

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
const Color PUFF_WHITE = (Color){241, 241, 241, 241};
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

    int hands[2][MAX_RANK];
    int hand_totals[2];
    int claimed[MAX_RANK];
    int known[MAX_RANK];

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
static void shuffle(int *deck) {
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

    for (int p = 0; p < 2; p++) {
        env->hand_totals[p] = 0;
        for (int r = 0; r <= MAX_RANK; r++) env->hands[p][r] = 0;
    }
    for (int i = 0; i < ACTIONS_SIZE; i++) env->action_masks[i] = 0;
}

/* Stolen from TT */
void allocateCBullShit(CBullshit* env) {
    env->actions = (int*)calloc(1, sizeof(int));
    env->observations = (float*)calloc(OBS_SIZE, sizeof(float));
    env->terminals = (unsigned char*)calloc(1, sizeof(unsigned char));
    env->rewards = (float*)calloc(1, sizeof(float));
    initCBullShit(env);
}

/* this should be safer then TT's implementation but who knows ¯\_(ツ)_/¯ */
void c_close(CBullshit* env) {
    if (env->client != NULL)closeClient(env->client); env->client = NULL;
    free(env->observations); env->observations = NULL;
    free(env->actions); env->actions = NULL;
    free(env->rewards); env->rewards = NULL;
    free(env->terminals); env->terminals = NULL;
}

/* random bullshit go! */
void free_allocated_cbullshit(CBullshit* env) {
    c_close(env);
}

void c_reset(CBullshit* env) {
    initCBullShit(env);
    shuffle(env->deck);
    env->deck_index = 0;

    for (int i = 0; i < DECK_SIZE; i+=2) {
        int r0 = env->deck[i];
        int r1 = env->deck[i+1];
        env->hands[0][r0] += 1;
        env->hand_totals[0] += 1;
        env->hands[1][r1] += 1;
        env->hand_totals[1] += 1;
    }

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

void computeObs(CBullshit* env) {
    int idx = 0;
    for (int r = 0; r < MAX_RANK; r++) env->observations[idx++] = (float)env->hands[0][r];
    for (int r = 0; r < MAX_RANK; r++) env->observations[idx++] = (float)env->hands[1][r];
    env->observations[idx++] = (float)env->hand_totals[0];
    env->observations[idx++] = (float)env->hand_totals[1];
    env->observations[idx++] = (float)env->pile_size;
    env->observations[idx++] = (float)env->last_declared_rank;
    env->observations[idx++] = (float)env->last_declared_count;
    for (int i = 0; i < ACTIONS_SIZE; i++) env->observations[idx++] = (float)env->action_masks[i];
}

void updateActionMasks(CBullshit* env) {
    int agent_total = env->hand_totals[0];
    for (int a = 0; a < MAX_RANK * MAX_PLAY_COUNT; a++) {
        int declared_count = 1 + (a % MAX_PLAY_COUNT);
        env->action_masks[a] = (agent_total < declared_count);
    }
    env->action_masks[CALL_BS_ACTION] = !env->can_call;
}

static void addCardsToHand(CBullshit* env, int player, int *cards, int n) {
    for (int i = 0; i < n; i++) {
        int r = cards[i];
        env->hands[player][r] += 1;
    }
    env->hand_totals[player] += n;
}

void placeCards(CBullshit* env, int declared_rank, int declared_count, int player) { 
    env->hands[player][declared_rank] -= declared_count;
    env->hand_totals[player] -= declared_count;
   for (int i = 0; i < declared_count; i++) {
        env->pile[env->pile_size++] = declared_rank;
    }
    env->last_declared_rank = declared_rank;
    env->last_declared_count = declared_count;
    env->can_call = 1;
}

/* Simple bot heuristics */
int botAct(CBullshit* env) {
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

int botCallBS(CBullshit* env) {
    float p = 0.12f;
    if (env->last_declared_count >= 3) p = 0.35f;
    float r = (float)rand() / (float)RAND_MAX;
    return (r < p) ? 1 : 0;
}

/* Caller resolves a BS call. caller_idx is the calling player (0 human, 1 bot).
   If the last declared segment is entirely equal to declared_rank -> caller was wrong -> caller picks up pile.
   Otherwise the play_player picks up the pile. */
void resolveCall(CBullshit* env, int caller_idx) {
    if (env->pile_size == 0 || env->last_declared_count == 0) {
        env->can_call = 0;
        return;
    }

    int play_player = 1 - caller_idx;
    int declared = env->last_declared_rank;
    int start_idx = env->pile_size - env->last_declared_count;
    if (start_idx < 0) start_idx = 0;

    int all_match = 1;
    int* p = env->pile[start_idx];
    int* e = env->pile[env->pile_size];
    for (; p < e; p++) {
        if (*p != declared) { all_match = 0; break; }
    }

    /* copy pile */
    int temp[DECK_SIZE];
    int ntemp = env->pile_size;
    for (int i = 0; i < env->pile_size; i++) temp[i] = env->pile[i];

    /* clear pile and flags */
    env->pile_size = 0;
    env->last_declared_rank = 0;
    env->last_declared_count = 0;
    env->can_call = 0;

    if (all_match) {
        /* caller wrong -> caller picks up pile */
        addCardsToHand(env, caller_idx, temp, ntemp);
        if (caller_idx == 0) {
            env->rewards[0] -= 0.5f; env->episode_return -= 0.5f;
        }
    } else {
        /* liar picks up pile */
        addCardsToHand(env, play_player, temp, ntemp);
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
    // START copy from TT ========================================
    env->episode_length += 1;
    env->rewards[0] = 0.0f;
    int action = env->actions[0];

    if (env->episode_length >= MAX_EPISODE_LENGTH) {
        env->game_over = 1;
        env->episode_return -= 1.0f;
        env->rewards[0] -= 1.0f;
    }

    if (env->game_over == 1) {
        env->perf = (env->hand_totals[0] == 0) ? 1.0f : 0.0f;
        addLog(env);
        c_reset(env);
        return;
    }
    // END copy from TT ===========================================


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
        int declared_rank = (action / MAX_PLAY_COUNT) + 1;
        int declared_count = (action % MAX_PLAY_COUNT) + 1;
        if (declared_count > env->hand_totals[0] || env->hand_totals[0] == 0) {
            env->episode_return -= 0.1f; env->rewards[0] -= 0.1f;
        } else {
            placeCards(env, declared_rank, declared_count, 0);

            /* Bot may call or play */ // TODO: make multiple different bots
            if (env->can_call) {
                if (botCallBS(env)) {
                    resolveCall(env, 1);
                } else {
                    int bot_action = botAct(env);
                    int bot_rank = (bot_action / MAX_PLAY_COUNT) + 1;
                    int bot_count = (bot_action % MAX_PLAY_COUNT) + 1;
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
    DrawText(TextFormat("Last declared: %d x %d", env->last_declared_rank, env->last_declared_count), 20, 40, 20, PUFF_WHITE);

    DrawText(TextFormat("Agent: %d cards", env->hand_totals[0]), 20, env->height - 80, 30, PUFF_CYAN);
    DrawText(TextFormat("Bot: %d cards", env->hand_totals[1]), env->width - 200, env->height - 80, 30, PUFF_RED);

    /* per-rank counts for both players (debug view) */
    for (int r = 1; r <= MAX_RANK; r++) {
        int x = 20 + (r - 1) * 36;
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
