import numpy as np
from scipy.stats import entropy
from sklearn.neighbors import KernelDensity

def estimate_pdf(samples, bandwidth=0.5):
    # Estimate probability density function (PDF) using kernel density estimation (KDE)
    kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde.fit(samples)
    return kde

def jensen_shannon_divergence_from_samples(samples_p, samples_q, bandwidth=0.5, num_points=100):
    # Estimate PDFs from samples_p and samples_q using kernel density estimation (KDE)
    kde_p = estimate_pdf(samples_p, bandwidth=bandwidth)
    kde_q = estimate_pdf(samples_q, bandwidth=bandwidth)
    
    # Determine the dimensions of the samples
    num_dims = samples_p.shape[1]
    
    # Generate grid of points for PDF evaluation
    ranges = []
    for d in range(num_dims):
        min_val = min(np.min(samples_p[:, d]), np.min(samples_q[:, d]))
        max_val = max(np.max(samples_p[:, d]), np.max(samples_q[:, d]))
        ranges.append((min_val, max_val))
    
    # Create meshgrid for PDF evaluation
    grid_points = [np.linspace(r[0], r[1], num_points) for r in ranges]
    grid_points_mesh = np.meshgrid(*grid_points)
    grid_points_stack = np.column_stack([mesh.ravel() for mesh in grid_points_mesh])
    
    # Evaluate PDFs at the grid points
    pdf_p = np.exp(kde_p.score_samples(grid_points_stack)).reshape(grid_points_mesh[0].shape)
    pdf_q = np.exp(kde_q.score_samples(grid_points_stack)).reshape(grid_points_mesh[0].shape)
    
    # Compute the average distribution M
    m_distribution = 0.5 * (pdf_p + pdf_q)
    
    # Compute KL divergences
    kl_p_m = entropy(pdf_p.ravel(), m_distribution.ravel())
    kl_q_m = entropy(pdf_q.ravel(), m_distribution.ravel())
    
    # Compute Jensen-Shannon Divergence (JSD)
    jsd = 0.5 * (kl_p_m + kl_q_m)
    
    return jsd


def jsd(samples_p, samples_q):
    result_jsd = 0
    for i, j  in zip(samples_p, samples_q):
        result_jsd += jensen_shannon_divergence_from_samples(i, j)
    return result_jsd

# Example usage with arbitrary-dimensional datasets
samples_p = np.random.randn(96, 1000, 1)  # Sample from distribution P (3 dimensions)
samples_q = np.random.randn(96, 1000, 1) * 5 + 1  # Sample from distribution Q (3 dimensions)