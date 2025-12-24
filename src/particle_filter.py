"""
Particle Filter for drone tracking.
Particles represent possible drone positions and are weighted by likelihood.
"""
import numpy as np


class ParticleFilter:
    """
    Particle Filter for smooth position tracking.
    
    Helps reduce jitter and improves robustness to missed detections.
    """
    
    def __init__(self, initial_state, num_particles=100, process_noise=10.0, measurement_noise=20.0):
        """
        Initialize particle filter.
        
        Args:
            initial_state: Initial [x, y, vx, vy] position and velocity
            num_particles: Number of particles for tracking
            process_noise: Process noise (motion uncertainty)
            measurement_noise: Measurement noise (detection uncertainty)
        """
        self.num_particles = num_particles
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        
        # Initialize particles around initial state
        self.particles = np.zeros((num_particles, 4))  # [x, y, vx, vy]
        self.particles[:, :] = initial_state + np.random.randn(num_particles, 4) * process_noise
        self.weights = np.ones(num_particles) / num_particles
        
        # Keep track of weighted state
        self.state = initial_state.copy()
        self.cov = np.eye(4) * (process_noise ** 2)
    
    def predict(self, dt=1.0):
        """
        Predict next position.
        
        Args:
            dt: Time delta (frame time)
        
        Returns:
            Predicted state [x, y, vx, vy]
        """
        # Add process noise to particles
        noise = np.random.randn(self.num_particles, 4) * self.process_noise
        self.particles = self.particles + noise
        
        # Simple motion model: position += velocity * dt
        self.particles[:, 0] += self.particles[:, 2] * dt
        self.particles[:, 1] += self.particles[:, 3] * dt
        
        # Compute weighted state estimate
        self.state = np.average(self.particles, axis=0, weights=self.weights)
        
        return self.state.copy()
    
    def update(self, measurement):
        """
        Update particles based on measurement (detection).
        
        Args:
            measurement: [x, y] detection position
        """
        # Compute weights based on measurement likelihood
        # Particles close to measurement get higher weight
        distances = np.linalg.norm(self.particles[:, :2] - measurement, axis=1)
        likelihoods = np.exp(-distances ** 2 / (2 * self.measurement_noise ** 2))

        # Safely normalize weights (avoid zeros/nans)
        w = np.nan_to_num(likelihoods, nan=0.0)
        w = np.maximum(w, 0.0)
        s = float(np.sum(w))
        if not np.isfinite(s) or s <= 0.0:
            # fallback to uniform weights
            self.weights = np.ones(self.num_particles) / float(self.num_particles)
        else:
            self.weights = w / s
        
        # Resample particles if effective sample size is low
        n_eff = 1.0 / np.sum(self.weights ** 2)
        if n_eff < self.num_particles / 2:
            self._resample()
        
        # Update state estimate
        self.state = np.average(self.particles, axis=0, weights=self.weights)
        
        # Update velocity estimate from detection
        # Smooth transition: 80% from KF, 20% from detection
        self.state[2:4] = 0.8 * self.state[2:4] + 0.2 * (measurement - self.state[:2])
    
    def _resample(self):
        """Resample particles from high-weight particles."""
        # Ensure weights are a valid probability distribution (non-negative, sums to 1)
        w = np.nan_to_num(self.weights, nan=0.0)
        w = np.maximum(w, 0.0)
        s = float(np.sum(w))
        if not np.isfinite(s) or s <= 0.0:
            # fallback to uniform weights
            w = np.ones(self.num_particles) / float(self.num_particles)
        else:
            w = w / s

        indices = np.random.choice(
            self.num_particles,
            size=self.num_particles,
            p=w,
            replace=True
        )
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles
    
    def get_state(self):
        """Get current state estimate [x, y, vx, vy]."""
        return self.state.copy()
    
    def get_position(self):
        """Get current position [x, y]."""
        return self.state[:2].copy()
