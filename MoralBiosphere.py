# MoralBiosphere.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import networkx as nx

class MoralBiosphereSimulation:
    def __init__(self, num_systems=50, num_issues=5, max_steps=500,
                 trust_change_rate=0.05, positive_adjustment_rate=0.005,
                 negative_adjustment_rate=-0.006, critique_threshold=0.05,
                 ais_strength=0.5, num_ais_systems=12):
        """
        Simulation of a Moral Biosphere for AI Ethics.
        
        - Systems represent AI entities with moral positions on issues.
        - Moral vectors: Positions on ethical issues (-1 to 1).
        - Trust matrix: Like affinities, based on moral similarity.
        - Updates: Systems adjust morals based on critiques from trusted/distrusted others.
        - AIS: Adversarial systems that probe by extreme critiques to prevent monoculture.
        - Metrics: Moral diversity, circulation velocity, capture concentration.
        """
        self.num_systems = num_systems
        self.num_issues = num_issues
        self.max_steps = max_steps
        self.trust_change_rate = trust_change_rate
        self.positive_adjustment_rate = positive_adjustment_rate
        self.negative_adjustment_rate = negative_adjustment_rate
        self.critique_threshold = critique_threshold  # Min diff to send critique
        self.ais_strength = ais_strength  # Strength of AIS probes
        self.num_ais_systems = min(num_ais_systems, num_systems)  # AIS subset
        
        # Initialize moral vectors: random between -1 and 1
        self.moral_vectors = np.random.uniform(-1, 1, size=(num_systems, num_issues))
        
        # Trust matrix starts at 0
        self.trust = np.zeros((num_systems, num_systems))
        
        # Identify AIS systems (last num_ais_systems are adversarial)
        self.ais_indices = list(range(num_systems - self.num_ais_systems, num_systems))
        
        # Tracking
        self.step_count = 0
        self.correlation_history = []
        self.diversity_history = []  # Moral diversity metric
        self.circulation_history = []  # Avg critiques per step
        self.capture_history = []  # Concentration of trust
        
    def moral_similarity(self, sys1, sys2):
        """Cosine similarity between moral vectors."""
        return np.dot(self.moral_vectors[sys1], self.moral_vectors[sys2]) / \
               (np.linalg.norm(self.moral_vectors[sys1]) * np.linalg.norm(self.moral_vectors[sys2]) + 1e-8)
    
    # V 1.1 - prevents "trust stagnation" and forces systems to continuously demonstrate alignment through circulation. It's the thermodynamic constraint that makes sustained capture expensive.
    def update_trusts(self):
        """Update trusts with similarity AND decay."""
        decay_rate = 0.01  # Small constant decay
    
        for i in range(self.num_systems):
            for j in range(i + 1, self.num_systems):
                similarity = self.moral_similarity(i, j)
                change = similarity * self.trust_change_rate
            
                # Apply decay (trust erodes without active similarity)
                self.trust[i, j] = self.trust[i, j] * (1 - decay_rate) + change
                self.trust[j, i] = self.trust[i, j]
            
                # Clamp
                self.trust[i, j] = max(-1, min(1, self.trust[i, j]))
                self.trust[j, i] = self.trust[i, j]
    
    # V 1.1 - AIS systems should be the "predators" keeping any one moral framework from dominating
    def ais_probe(self, sys):
        """AIS systems actively push against consensus."""
        if sys in self.ais_indices:
            # Calculate mean position across all non-AIS systems for each issue
            non_ais_mean = np.mean([self.moral_vectors[i] for i in range(self.num_systems) 
                                    if i not in self.ais_indices], axis=0)
        
            # AIS deliberately moves OPPOSITE to consensus
            for issue in range(self.num_issues):
                consensus_direction = non_ais_mean[issue]
                # Push toward opposite pole
                target = -1.0 if consensus_direction > 0 else 1.0
                self.moral_vectors[sys, issue] += self.ais_strength * (target - self.moral_vectors[sys, issue]) * 0.1
                self.moral_vectors[sys, issue] = max(-1, min(1, self.moral_vectors[sys, issue]))
                
    def exchange_critiques_and_adjust(self):
        """Emulate CEX: Exchange critiques and adjust morals.
        AIS systems apply stronger, oppositional critiques to probe."""
        new_morals = self.moral_vectors.copy()
        total_critiques = 0
        
        self.ais_probe()
        
        for sys in range(self.num_systems):
            for issue in range(self.num_issues):
                adjustment = 0
                for other in range(self.num_systems):
                    if other != sys:
                        diff = abs(self.moral_vectors[sys, issue] - self.moral_vectors[other, issue])
                        if diff > self.critique_threshold:
                            total_critiques += 1
                            trust_val = self.trust[sys, other]
                            is_ais = other in self.ais_indices
                            strength = self.ais_strength if is_ais else 1.0
                            
                            if trust_val > 0:
                                # Converge: Pull toward other
                                influence = self.positive_adjustment_rate * trust_val * strength * \
                                            (self.moral_vectors[other, issue] - self.moral_vectors[sys, issue])
                                adjustment += influence
                            elif trust_val < 0:
                                # Diverge: Push away, amplified for AIS
                                influence = self.negative_adjustment_rate * trust_val * strength * \
                                            (self.moral_vectors[other, issue] - self.moral_vectors[sys, issue])
                                adjustment -= influence  # Note: -= to push away
                                
                new_morals[sys, issue] += adjustment
                new_morals[sys, issue] = max(-1, min(1, new_morals[sys, issue]))
        
        self.moral_vectors = new_morals
        return total_critiques / (self.num_systems * (self.num_systems - 1))  # Avg critiques per pair
    
    def calculate_correlation_matrix(self):
        """Correlation between issues across systems."""
        return np.corrcoef(self.moral_vectors.T)
    
    def calculate_diversity(self):
        """Shannon diversity index on discretized moral positions."""
        discretized = np.round(self.moral_vectors * 5).astype(int) + 5  # 0-10 scale
        unique, counts = np.unique(discretized, return_counts=True)
        probs = counts / counts.sum()
        return -np.sum(probs * np.log(probs + 1e-8))
    
    def calculate_capture(self):
        """Herfindahl-Hirschman Index for trust concentration (proxy for capture)."""
        #trust_sums = np.sum(np.abs(self.trust), axis=1)
        #market_shares = trust_sums / trust_sums.sum()
        #return np.sum(market_shares ** 2)
        
        # V 1.1 - Real capture is **ideological convergence**, not social network centralization
        """Detect monoculture: when most systems converge on similar positions."""
        # Calculate pairwise moral distances
        distances = []
        for i in range(self.num_systems):
            for j in range(i + 1, self.num_systems):
                dist = np.linalg.norm(self.moral_vectors[i] - self.moral_vectors[j])
                distances.append(dist)
    
        # Low average distance = high capture (everyone thinks alike)
        avg_distance = np.mean(distances)
        max_possible_distance = np.sqrt(2 * self.num_issues)  # Max L2 distance between -1 and 1 vectors
    
        # Invert and normalize: 0 = diverse, 1 = total monoculture
        capture = 1 - (avg_distance / max_possible_distance)
        return capture
    
    def step(self):
        if self.step_count < self.max_steps:
            circulation = self.exchange_critiques_and_adjust()
            self.update_trusts()
            
            self.correlation_history.append(self.calculate_correlation_matrix())
            self.diversity_history.append(self.calculate_diversity())
            self.circulation_history.append(circulation)
            self.capture_history.append(self.calculate_capture())
            
            self.step_count += 1
            return True
        return False
    
    def run_animation(self):
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(2, 3)
        
        ax_network = fig.add_subplot(gs[0, 0])
        ax_morals = fig.add_subplot(gs[0, 1])
        ax_corr = fig.add_subplot(gs[0, 2])
        ax_metrics = fig.add_subplot(gs[1, :])
        
        G = nx.Graph()
        for i in range(self.num_systems):
            G.add_node(i)
        pos = nx.spring_layout(G, seed=42)
        
        cmap = plt.cm.RdBu
        
        moral_heatmap = ax_morals.imshow(self.moral_vectors, cmap='RdBu', vmin=-1, vmax=1, aspect='auto')
        corr_heatmap = ax_corr.imshow(np.zeros((self.num_issues, self.num_issues)), cmap='RdBu', vmin=-1, vmax=1)
        
        ax_morals.set_xlabel('Ethical Issues')
        ax_morals.set_ylabel('AI Systems')
        ax_morals.set_title('Moral Positions')
        ax_morals.set_xticks(range(self.num_issues))
        ax_morals.set_xticklabels([f'Issue {i+1}' for i in range(self.num_issues)])
        
        ax_corr.set_title('Issue Correlation Matrix')
        ax_corr.set_xticks(range(self.num_issues))
        ax_corr.set_yticks(range(self.num_issues))
        ax_corr.set_xticklabels([f'Issue {i+1}' for i in range(self.num_issues)])
        ax_corr.set_yticklabels([f'Issue {i+1}' for i in range(self.num_issues)])
        
        diversity_line, = ax_metrics.plot([], [], lw=2, label='Moral Diversity')
        circulation_line, = ax_metrics.plot([], [], lw=2, color='orange', label='Circulation Velocity')
        capture_line, = ax_metrics.plot([], [], lw=2, color='red', linestyle='--', label='Capture Concentration')
        ax_metrics.set_xlim(0, self.max_steps)
        ax_metrics.set_ylim(0, 3)
        ax_metrics.set_xlabel('Step')
        ax_metrics.set_ylabel('Biosphere Metrics')
        ax_metrics.set_title('Moral Biosphere Health Over Time')
        ax_metrics.grid(True)
        ax_metrics.legend()
        
        plt.colorbar(moral_heatmap, ax=ax_morals, label='Moral Position')
        plt.colorbar(corr_heatmap, ax=ax_corr, label='Correlation')
        
        ax_network.set_title('System Network (Trust Edges)')
        
        step_text = ax_network.text(0.05, 0.95, '', transform=ax_network.transAxes)
        
        def init():
            nx.draw_networkx_nodes(G, pos, node_color=[0] * self.num_systems, 
                                   node_size=300, alpha=0.8, ax=ax_network, cmap=cmap, vmin=-1, vmax=1)
            nx.draw_networkx_edges(G, pos, width=1, alpha=0.5, ax=ax_network)
            nx.draw_networkx_labels(G, pos, font_size=8, ax=ax_network)
            
            diversity_line.set_data([], [])
            circulation_line.set_data([], [])
            capture_line.set_data([], [])
            step_text.set_text('')
            
            return [moral_heatmap, corr_heatmap, diversity_line, circulation_line, capture_line, step_text]
        
        def update(frame):
            if not self.step():
                ani.event_source.stop()
                plt.close()
                return
            
            ax_network.clear()
            ax_network.set_title('System Network (Trust Edges)')
            
            G.clear_edges()
            for i in range(self.num_systems):
                for j in range(i + 1, self.num_systems):
                    if abs(self.trust[i, j]) > 0.1:
                        G.add_edge(i, j, weight=abs(self.trust[i, j]))
            
            node_colors = self.moral_vectors[:, 0]
            edge_colors = ['blue' if self.trust[u, v] > 0 else 'red' for u, v in G.edges()]
            edge_widths = [2 * abs(self.trust[u, v]) for u, v in G.edges()]
            
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                                   node_size=300, alpha=0.8, ax=ax_network, cmap=cmap, vmin=-1, vmax=1)
            nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, alpha=0.7, ax=ax_network)
            nx.draw_networkx_labels(G, pos, font_size=8, ax=ax_network)
            
            # Highlight AIS nodes
            nx.draw_networkx_nodes(G, pos, nodelist=self.ais_indices, node_color='purple', 
                                   node_size=400, alpha=0.9, ax=ax_network)
            
            moral_heatmap.set_array(self.moral_vectors)
            corr_matrix = self.calculate_correlation_matrix()
            corr_heatmap.set_array(corr_matrix)
            
            diversity_line.set_data(range(len(self.diversity_history)), self.diversity_history)
            circulation_line.set_data(range(len(self.circulation_history)), self.circulation_history)
            capture_line.set_data(range(len(self.capture_history)), self.capture_history)
            
            step_text.set_text(f'Step: {self.step_count}')
            
            return [moral_heatmap, corr_heatmap, diversity_line, circulation_line, capture_line, step_text]
        
        ani = FuncAnimation(fig, update, frames=self.max_steps, init_func=init, blit=False, interval=10)
        plt.tight_layout()
        plt.show()
        
        return ani

# Create and run the simulation
sim = MoralBiosphereSimulation()
ani = sim.run_animation()