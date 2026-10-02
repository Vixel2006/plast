#include "lr_scheduler/lr_scheduler.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static LRScheduler *alloc_scheduler(LRSchedulerType type, double *base_lrs, u64 num_groups, i64 last_epoch) {
  LRScheduler *sched = malloc(sizeof(LRScheduler));
  sched->type = type;
  sched->num_groups = num_groups;
  sched->last_epoch = last_epoch;
  sched->base_lrs = malloc(sizeof(double) * num_groups);
  memcpy(sched->base_lrs, base_lrs, sizeof(double) * num_groups);
  return sched;
}

LRScheduler *create_step_lr(double *base_lrs, u64 num_groups, u64 step_size, double gamma, i64 last_epoch) {
  LRScheduler *sched = alloc_scheduler(STEP_LR, base_lrs, num_groups, last_epoch);
  sched->step_lr.step_size = step_size;
  sched->step_lr.gamma = gamma;
  return sched;
}

LRScheduler *create_multi_step_lr(double *base_lrs, u64 num_groups, u64 *milestones, u64 num_milestones, double gamma, i64 last_epoch) {
  LRScheduler *sched = alloc_scheduler(MULTI_STEP_LR, base_lrs, num_groups, last_epoch);
  sched->multi_step_lr.num_milestones = num_milestones;
  sched->multi_step_lr.gamma = gamma;
  sched->multi_step_lr.milestones = malloc(sizeof(u64) * num_milestones);
  memcpy(sched->multi_step_lr.milestones, milestones, sizeof(u64) * num_milestones);
  return sched;
}

LRScheduler *create_exponential_lr(double *base_lrs, u64 num_groups, double gamma, i64 last_epoch) {
  LRScheduler *sched = alloc_scheduler(EXPONENTIAL_LR, base_lrs, num_groups, last_epoch);
  sched->exponential_lr.gamma = gamma;
  return sched;
}

LRScheduler *create_cosine_annealing_lr(double *base_lrs, u64 num_groups, u64 T_max, double eta_min, i64 last_epoch) {
  LRScheduler *sched = alloc_scheduler(COSINE_ANNEALING_LR, base_lrs, num_groups, last_epoch);
  sched->cosine_annealing_lr.T_max = T_max;
  sched->cosine_annealing_lr.eta_min = eta_min;
  return sched;
}

void free_lr_scheduler(LRScheduler *scheduler) {
  if (scheduler) {
    if (scheduler->base_lrs) {
      free(scheduler->base_lrs);
    }
    if (scheduler->type == MULTI_STEP_LR && scheduler->multi_step_lr.milestones) {
      free(scheduler->multi_step_lr.milestones);
    }
    free(scheduler);
  }
}

void lr_scheduler_step(LRScheduler *scheduler, double *current_lrs, i64 epoch) {
  if (epoch < 0) {
    scheduler->last_epoch += 1;
  } else {
    scheduler->last_epoch = epoch;
  }

  i64 cur_epoch = scheduler->last_epoch;

  switch (scheduler->type) {
    case STEP_LR: {
      u64 step_size = scheduler->step_lr.step_size;
      double gamma = scheduler->step_lr.gamma;
      if (cur_epoch == 0 || cur_epoch % step_size != 0) {
        return;
      }
      for (u64 i = 0; i < scheduler->num_groups; i++) {
        current_lrs[i] = current_lrs[i] * gamma;
      }
      break;
    }
    case MULTI_STEP_LR: {
      bool is_milestone = false;
      for (u64 i = 0; i < scheduler->multi_step_lr.num_milestones; i++) {
        if (scheduler->multi_step_lr.milestones[i] == cur_epoch) {
          is_milestone = true;
          break;
        }
      }
      if (!is_milestone) {
        return;
      }
      double gamma = scheduler->multi_step_lr.gamma;
      for (u64 i = 0; i < scheduler->num_groups; i++) {
        current_lrs[i] = current_lrs[i] * gamma;
      }
      break;
    }
    case EXPONENTIAL_LR: {
      double gamma = scheduler->exponential_lr.gamma;
      if (cur_epoch == 0) {
        for (u64 i = 0; i < scheduler->num_groups; i++) {
          current_lrs[i] = scheduler->base_lrs[i];
        }
      } else {
        for (u64 i = 0; i < scheduler->num_groups; i++) {
          current_lrs[i] = current_lrs[i] * gamma;
        }
      }
      break;
    }
    case COSINE_ANNEALING_LR: {
      u64 T_max = scheduler->cosine_annealing_lr.T_max;
      double eta_min = scheduler->cosine_annealing_lr.eta_min;
      if (cur_epoch == 0) {
        for (u64 i = 0; i < scheduler->num_groups; i++) {
          current_lrs[i] = scheduler->base_lrs[i];
        }
      } else if ((cur_epoch - 1 - T_max) % (2 * T_max) == 0) {
        for (u64 i = 0; i < scheduler->num_groups; i++) {
          current_lrs[i] = current_lrs[i] + (scheduler->base_lrs[i] - eta_min) * (1.0 - cos(M_PI / T_max)) / 2.0;
        }
      } else {
        double cos_curr = cos(M_PI * cur_epoch / T_max);
        double cos_prev = cos(M_PI * (cur_epoch - 1) / T_max);
        for (u64 i = 0; i < scheduler->num_groups; i++) {
          current_lrs[i] = eta_min + (current_lrs[i] - eta_min) * (1.0 + cos_curr) / (1.0 + cos_prev);
        }
      }
      break;
    }
  }
}

ReduceLROnPlateau *create_reduce_lr_on_plateau(u64 num_groups, double factor, u64 patience, double threshold, u64 cooldown, double *min_lrs, double eps, ReduceLROnPlateauMode mode) {
  ReduceLROnPlateau *sched = malloc(sizeof(ReduceLROnPlateau));
  sched->factor = factor;
  sched->patience = patience;
  sched->threshold = threshold;
  sched->cooldown = cooldown;
  sched->eps = eps;
  sched->mode = mode;
  sched->num_groups = num_groups;
  
  sched->min_lrs = malloc(sizeof(double) * num_groups);
  memcpy(sched->min_lrs, min_lrs, sizeof(double) * num_groups);
  
  sched->best = (mode == MIN_MODE) ? INFINITY : -INFINITY;
  sched->num_bad_epochs = 0;
  sched->last_epoch = 0;
  sched->cooldown_counter = 0;
  
  return sched;
}

void free_reduce_lr_on_plateau(ReduceLROnPlateau *scheduler) {
  if (scheduler) {
    if (scheduler->min_lrs) {
      free(scheduler->min_lrs);
    }
    free(scheduler);
  }
}

bool reduce_lr_on_plateau_step(ReduceLROnPlateau *scheduler, double metrics, double *current_lrs, i64 epoch) {
  if (epoch < 0) {
    scheduler->last_epoch += 1;
  } else {
    scheduler->last_epoch = epoch;
  }
  
  bool is_better = false;
  if (scheduler->mode == MIN_MODE) {
    is_better = (metrics < scheduler->best - scheduler->threshold);
  } else {
    is_better = (metrics > scheduler->best + scheduler->threshold);
  }
  
  if (is_better) {
    scheduler->best = metrics;
    scheduler->num_bad_epochs = 0;
  } else {
    scheduler->num_bad_epochs += 1;
  }
  
  if (scheduler->cooldown_counter > 0) {
    scheduler->cooldown_counter -= 1;
    scheduler->num_bad_epochs = 0;
  }
  
  bool reduced = false;
  if (scheduler->num_bad_epochs > scheduler->patience) {
    for (u64 i = 0; i < scheduler->num_groups; i++) {
      double old_lr = current_lrs[i];
      double new_lr = old_lr * scheduler->factor;
      if (new_lr < scheduler->min_lrs[i]) {
        new_lr = scheduler->min_lrs[i];
      }
      if (old_lr - new_lr > scheduler->eps) {
        current_lrs[i] = new_lr;
        reduced = true;
      }
    }
    scheduler->cooldown_counter = scheduler->cooldown;
    scheduler->num_bad_epochs = 0;
  }
  return reduced;
}
