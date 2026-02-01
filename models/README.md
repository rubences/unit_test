# Models Directory

This directory contains trained model checkpoints.

## File Naming Convention

Models should be named following this pattern:
```
{algorithm}_{track}_{timestamp}.pt
```

Example: `ppo_silverstone_20260117.pt`

## Best Practices

- Keep the 5 best checkpoints based on validation performance
- Save full model state including optimizer state for resuming training
- Include metadata file alongside each checkpoint with training metrics
