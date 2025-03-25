# HyperX-Viz
Visualization of 3D and 4D HyperX topologies

## Overview
The HyperX network is a flexible, high-performance datacenter network topology that generalizes the hypercube and flattened butterfly architectures. It provides high bisection bandwidth, low diameter, and good scalability across varying network sizes. A HyperX network consists of L dimensions, with S switches per dimension, K links per connection (trunking factor), and T terminals (servers/hosts) per switch.
This code implements a complete HyperX network generator and visualizer, supporting both 3D and 4D topologies. It creates a graph representation of the network, calculates key network metrics (links, radix, bisection bandwidth), and provides advanced interactive 3D visualization capabilities. The visualization techniques include dimension-specific color coding, interactive highlighting, 4D projections, and curved spline connections that make complex topologies visually interpretable.

## Key Inputs and Configuration
S: Switches per dimension (uniform across all dimensions)
K: Link bandwidth/trunking factor
T: Terminals (compute nodes) per switch
L: Number of dimensions (typically 3 or 4)

The code automatically calculates derived metrics including:

Total switches (P): Product of switches in each dimension
Total terminals (N): T × P
Switch radix (R): Sum of inter-switch connections and terminal connections
Total network links: Sum of all switch-to-switch and switch-to-terminal links
Network diameter: Typically equal to L when using dimension-ordered routing

## Main Algorithmic Components
Network Generation: Creates a graph structure with switches and terminals as nodes
Dimension-based Connectivity: Organizes switch connections in dimensional groups
3D Position Calculation: Maps multidimensional coordinates to 3D visualization space
4D Projection Techniques: Uses spiral offsets and color gradients to represent the 4th dimension
Curved Spline Generation: Creates visually distinct, dimension-specific curved connections
Interactive Visualization Controls: Provides tools to explore and highlight network features

![Screenshot 2025-03-25 at 6 52 13 PM](https://github.com/user-attachments/assets/5237e10d-9070-4055-b5c2-6b9cca8a7eba)
