#pragma once

#include "core/definitions.h"

typedef enum LRSchedulerType {
  STEP_LR,
  MULTI_STEP_LR,
  EXPONENTIAL_LR,
  COSINE_ANNEALING_LR
} LRSchedulerType;

typedef struct LRScheduler {
  LRSchedulerType type;
  double *base_lrs;
  u64 num_groups;
  i64 last_epoch;
  
  union {
    struct {
      u64 step_size;
      double gamma;
    } step_lr;
    
    struct {
      u64 *milestones;
      u64 num_milestones;
      double gamma;
    } multi_step_lr;
    
    struct {
      double gamma;
    } exponential_lr;
    
    struct {
      u64 T_max;
      double eta_min;
    } cosine_annealing_lr;
  };
} LRScheduler;

typedef enum ReduceLROnPlateauMode {
  MIN_MODE,
  MAX_MODE
} ReduceLROnPlateauMode;

typedef struct ReduceLROnPlateau {
  double factor;
  u64 patience;
  double threshold;
  u64 cooldown;
  double *min_lrs;
  double eps;
  ReduceLROnPlateauMode mode;
  
  u64 num_groups;
  double best;
  u64 num_bad_epochs;
  i64 last_epoch;
  u64 cooldown_counter;
} ReduceLROnPlateau;

#ifdef __cplusplus
extern "C" {
#endif

LRScheduler *create_step_lr(double *base_lrs, u64 num_groups, u64 step_size, double gamma, i64 last_epoch);
LRScheduler *create_multi_step_lr(double *base_lrs, u64 num_groups, u64 *milestones, u64 num_milestones, double gamma, i64 last_epoch);
LRScheduler *create_exponential_lr(double *base_lrs, u64 num_groups, double gamma, i64 last_epoch);
LRScheduler *create_cosine_annealing_lr(double *base_lrs, u64 num_groups, u64 T_max, double eta_min, i64 last_epoch);
void free_lr_scheduler(LRScheduler *scheduler);
void lr_scheduler_step(LRScheduler *scheduler, double *current_lrs, i64 epoch);

ReduceLROnPlateau *create_reduce_lr_on_plateau(u64 num_groups, double factor, u64 patience, double threshold, u64 cooldown, double *min_lrs, double eps, ReduceLROnPlateauMode mode);
void free_reduce_lr_on_plateau(ReduceLROnPlateau *scheduler);
bool reduce_lr_on_plateau_step(ReduceLROnPlateau *scheduler, double metrics, double *current_lrs, i64 epoch);

#ifdef __cplusplus
}
#endif
