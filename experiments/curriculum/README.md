# Shared 370M trainer (not a paper experiment)

This directory no longer holds a reported experiment. The curriculum paper
reports runs from the `curriculum-new` branch of the OLMo-core fork, not from
code in this repository.

What remains is here because the domain-weighting runs depend on it:

- `train_curriculum_regmix_370m.py` began as the August curriculum trainer.
  The MixLaw and Skill-It 370M trainers
  (`../skill-dag/mixlaw/train_mixlaw_validation_370m.py` and
  `../skill-dag/skillit/train_skillit_370m.py`) import its model builder,
  checkpoint save/load, bookkeeping and recipe constants.
- `curriculum_pacing.py` is imported by that trainer at module load.
- `tests/test_trainer_hardening.py` checks the checkpoint, task-loss eval and
  W&B publish ordering those trainers rely on.
