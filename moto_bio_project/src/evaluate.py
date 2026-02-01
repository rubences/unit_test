"""
Evaluación del modelo entrenado.

Funciones:
  - evaluate_trained_model: Ejecutar modelo en entorno de test
  - get_evaluation_metrics: Calcular métricas de rendimiento
"""

import numpy as np
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def evaluate_trained_model(
    model,
    num_episodes: int = 5,
    render: bool = False,
    max_steps: int = 5000
) -> Dict[str, float]:
    """
    Evaluar modelo PPO entrenado.
    
    Args:
        model: Modelo PPO de stable-baselines3
        num_episodes: Número de episodios para evaluar
        render: Mostrar visualización
        max_steps: Máximo de pasos por episodio
    
    Returns:
        Diccionario con métricas de evaluación
    """
    try:
        if model is None:
            logger.warning("Modelo es None, retornando métricas vacías")
            return {
                'mean_reward': 0.0,
                'std_reward': 0.0,
                'mean_length': 0.0,
                'episodes': 0
            }
        
        # Obtener el entorno del modelo
        env = model.get_env()
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            obs, info = env.reset()
            episode_reward = 0
            episode_length = 0
            
            for step in range(max_steps):
                # Predecir acción usando política entrenada
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                
                episode_reward += reward
                episode_length += 1
                
                if terminated or truncated:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            logger.info(f"Episodio {episode+1}/{num_episodes}: "
                       f"Reward={episode_reward:.2f}, Length={episode_length}")
        
        # Calcular estadísticas
        metrics = {
            'mean_reward': float(np.mean(episode_rewards)),
            'std_reward': float(np.std(episode_rewards)),
            'min_reward': float(np.min(episode_rewards)),
            'max_reward': float(np.max(episode_rewards)),
            'mean_length': float(np.mean(episode_lengths)),
            'episodes': num_episodes
        }
        
        logger.info(f"Evaluación completada:")
        logger.info(f"  Mean Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
        logger.info(f"  Mean Length: {metrics['mean_length']:.0f} pasos")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error en evaluación: {e}")
        return {
            'mean_reward': 0.0,
            'std_reward': 0.0,
            'mean_length': 0.0,
            'episodes': 0,
            'error': str(e)
        }


def get_evaluation_metrics(
    rewards: np.ndarray,
    lengths: np.ndarray
) -> Dict[str, float]:
    """
    Calcular métricas de evaluación desde arrays.
    
    Args:
        rewards: Array de rewards por episodio
        lengths: Array de lengths por episodio
    
    Returns:
        Diccionario con métricas
    """
    return {
        'mean_reward': float(np.mean(rewards)),
        'std_reward': float(np.std(rewards)),
        'min_reward': float(np.min(rewards)),
        'max_reward': float(np.max(rewards)),
        'mean_length': float(np.mean(lengths)),
        'std_length': float(np.std(lengths)),
    }
