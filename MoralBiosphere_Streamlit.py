# MoralBiosphere_Streamlit.py

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import networkx as nx
import time

# Import the simulation class (assuming MoralBiosphere.py is in the same directory)
from MoralBiosphere import MoralBiosphereSimulation

def main():
    st.set_page_config(page_title="Moral Biosphere Simulation", layout="wide")
    
    if "initialization_complete" not in st.session_state:
        st.session_state.initialization_complete = False
        st.session_state.paused = True
    
    st.title("Moral Biosphere Simulation")
    
    st.write("""
    This simulation operationalizes the Moral Biosphere framework for AI ethics governance.
    AI systems hold moral positions on ethical issues and exchange critiques via a simulated Circulation Exchange (CEX).
    Trusts evolve based on similarities, and adjustments occur from positive/negative influences.
    Adversarial Immune Systems (AIS) probe to maintain diversity and prevent capture.
    Adjust parameters to test how the biosphere resists monoculture and sustains pluralism.
    
    Key Concepts:
    - MTP: Implicit in broadcasting moral positions.
    - CEX: Critiques exchanged based on differences and trusts.
    - AIS: Purple nodes apply amplified oppositional critiques.
    
    Metrics:
    - Moral Diversity: Shannon index of position variety (higher = more pluralistic).
    - Circulation Velocity: Average critiques per system pair (higher = active exchange).
    - Capture Concentration: HHI of trust sums (higher = more centralized/captured).
    """)
    
    with st.sidebar:
        st.subheader("Simulation Settings")
        num_systems = st.slider("Number of AI Systems", 10, 100, 50,
                                help="Number of AI entities in the biosphere.")
        num_issues = st.slider("Number of Ethical Issues", 2, 10, 5,
                               help="Dimensions of moral positions.")
        trust_change_rate = st.slider("Trust Change Rate", 0.01, 0.2, 0.05, 0.01,
                                      help="How quickly trusts update based on similarity.")
        positive_adjustment_rate = st.slider("Positive Adjustment Rate", 0.001, 0.01, 0.005, 0.001,
                                             format="%.3f",
                                             help="Strength of convergence from trusted critiques.")
        negative_adjustment_rate = st.slider("Negative Adjustment Rate", -0.01, -0.001, -0.003, 0.001,
                                             format="%.3f",
                                             help="Strength of divergence from distrusted critiques (negative value).")
        critique_threshold = st.slider("Critique Threshold", 0.05, 0.5, 0.1, 0.05,
                                       help="Minimum moral difference to trigger a critique.")
        ais_strength = st.slider("AIS Strength", 0.1, 1.0, 0.2, 0.1,
                                 help="Amplification factor for AIS probes.")
        num_ais_systems = st.slider("Number of AIS Systems", 1, 10, 5,
                                    help="Number of adversarial systems for immune response.")
        max_steps = st.slider("Maximum Steps", 100, 1000, 500, 50)
        step_increment = st.slider("Steps per Update", 1, 20, 5,
                                   help="Number of steps per visualization update (higher = faster simulation).")
        
        sidebar_reset = st.button("Reset Simulation")
    
    if "simulation" not in st.session_state or sidebar_reset or st.session_state.get('main_reset_clicked', False):
        st.session_state.simulation = MoralBiosphereSimulation(
            num_systems=num_systems,
            num_issues=num_issues,
            max_steps=max_steps,
            trust_change_rate=trust_change_rate,
            positive_adjustment_rate=positive_adjustment_rate,
            negative_adjustment_rate=negative_adjustment_rate,
            critique_threshold=critique_threshold,
            ais_strength=ais_strength,
            num_ais_systems=num_ais_systems
        )
        st.session_state.paused = True
        if 'main_reset_clicked' in st.session_state:
            st.session_state.main_reset_clicked = False
        sim = st.session_state.simulation
    else:
        sim = st.session_state.simulation
    
    st.header("Simulation Controls")
    cols = st.columns(4)
    with cols[0]:
        if st.button("Run/Resume"):
            st.session_state.paused = False
    with cols[1]:
        if st.button("Pause"):
            st.session_state.paused = True
    with cols[2]:
        if st.button("Step"):
            for _ in range(step_increment):
                if not sim.step():
                    break
    with cols[3]:
        if st.button("Reset Simulation", key="main_reset_button"):
            st.session_state.main_reset_clicked = True
            st.rerun()
    
    dynamic_content = st.container()
    
    st.markdown("---")
    
    st.markdown("""
    ## Visualization Guide
    - **Network Graph**: Systems as nodes; edges colored blue (positive trust)/red (negative); width by |trust|. Purple nodes are AIS.
    - **Moral Positions Heatmap**: Rows = systems, columns = issues; red/blue for positive/negative positions.
    - **Correlation Matrix**: Issue correlations; red/blue for positive/negative.
    - **Metrics Plot**: Diversity (blue), Circulation (orange), Capture (red dashed).
    """)
    
    with dynamic_content:
        st.subheader("Biosphere Visualization")
        
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 3)
        
        ax_network = fig.add_subplot(gs[0, 0])
        ax_morals = fig.add_subplot(gs[0, 1])
        ax_corr = fig.add_subplot(gs[0, 2])
        ax_metrics = fig.add_subplot(gs[1, :])
        
        G = nx.Graph()
        for i in range(sim.num_systems):
            G.add_node(i)
        
        pos = nx.spring_layout(G, seed=42)
        
        for i in range(sim.num_systems):
            for j in range(i + 1, sim.num_systems):
                if abs(sim.trust[i, j]) > 0.1:
                    G.add_edge(i, j, weight=abs(sim.trust[i, j]))
        
        node_colors = sim.moral_vectors[:, 0]
        edge_colors = ['blue' if sim.trust[u, v] > 0 else 'red' for u, v in G.edges()]
        edge_widths = [2 * abs(sim.trust[u, v]) for u, v in G.edges()]
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=300, alpha=0.8,
                               ax=ax_network, cmap=plt.cm.RdBu, vmin=-1, vmax=1)
        nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, alpha=0.7, ax=ax_network)
        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax_network)
        
        nx.draw_networkx_nodes(G, pos, nodelist=sim.ais_indices, node_color='purple',
                               node_size=400, alpha=0.9, ax=ax_network)
        
        ax_network.set_title('System Network (Trust Edges)')
        
        moral_heatmap = ax_morals.imshow(sim.moral_vectors, cmap='RdBu', vmin=-1, vmax=1, aspect='auto')
        ax_morals.set_xlabel('Ethical Issues')
        ax_morals.set_ylabel('AI Systems')
        ax_morals.set_title('Moral Positions')
        ax_morals.set_xticks(range(sim.num_issues))
        ax_morals.set_xticklabels([f'Issue {i+1}' for i in range(sim.num_issues)])
        plt.colorbar(moral_heatmap, ax=ax_morals, label='Moral Position')
        
        corr_matrix = sim.calculate_correlation_matrix()
        corr_heatmap = ax_corr.imshow(corr_matrix, cmap='RdBu', vmin=-1, vmax=1)
        ax_corr.set_title('Issue Correlation Matrix')
        ax_corr.set_xticks(range(sim.num_issues))
        ax_corr.set_yticks(range(sim.num_issues))
        ax_corr.set_xticklabels([f'Issue {i+1}' for i in range(sim.num_issues)])
        ax_corr.set_yticklabels([f'Issue {i+1}' for i in range(sim.num_issues)])
        plt.colorbar(corr_heatmap, ax=ax_corr, label='Correlation')
        
        ax_metrics.plot(sim.diversity_history, lw=2, label='Moral Diversity')
        ax_metrics.plot(sim.circulation_history, lw=2, color='orange', label='Circulation Velocity')
        ax_metrics.plot(sim.capture_history, lw=2, color='red', linestyle='--', label='Capture Concentration')
        ax_metrics.set_xlim(0, sim.max_steps)
        ax_metrics.set_ylim(0, 3)
        ax_metrics.set_xlabel('Step')
        ax_metrics.set_ylabel('Biosphere Metrics')
        ax_metrics.set_title('Moral Biosphere Health Over Time')
        ax_metrics.grid(True)
        ax_metrics.legend()
        
        plt.figtext(0.02, 0.02, f'Step: {sim.step_count}', fontsize=12)
        plt.tight_layout()
        st.pyplot(fig)
        
        st.subheader("Current Metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            if sim.diversity_history:
                st.metric("Moral Diversity", round(sim.diversity_history[-1], 3))
        with col2:
            if sim.circulation_history:
                st.metric("Circulation Velocity", round(sim.circulation_history[-1], 3))
        with col3:
            if sim.capture_history:
                st.metric("Capture Concentration", round(sim.capture_history[-1], 3))
        
        if sim.step_count > 0:
            progress = sim.step_count / sim.max_steps
            st.progress(progress)
    
    if not st.session_state.paused:
        steps_completed = 0
        for _ in range(step_increment):
            steps_completed += 1
            if not sim.step():
                st.session_state.paused = True
                break
        
        if not st.session_state.paused and sim.step_count < sim.max_steps:
            time.sleep(0.1)
            st.rerun()

if __name__ == "__main__":
    main()