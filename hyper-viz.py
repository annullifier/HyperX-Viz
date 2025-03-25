import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
from itertools import product
import matplotlib.animation as animation
from scipy import interpolate

class HyperX:
    """
    HyperX Network Generator and Visualizer.
    
    A HyperX is a flexible, high-performance datacenter network topology that expands 
    on hypercube architectures to allow arbitrary numbers of switches in each dimension.
    This class supports both 3D and 4D HyperX topologies, with full graph generation, 
    metrics calculation, and advanced visualization capabilities.
    
    The class can visualize network topologies with customizable options including
    dimension highlighting, 4D slicing, terminal visibility, and interactive controls.
    """
    
    def __init__(self, S=3, K=1, T=1, L=4):
        """
        Initialize a HyperX network with configurable dimensions and parameters.
        
        Args:
            S (int): Switches per dimension (same in all dimensions)
                     Controls the size of each dimension
            K (int): Link bandwidth (trunking factor)
                     Number of parallel links between switches
            T (int): Terminals per switch
                     Compute nodes attached to each switch
            L (int): Number of dimensions (3 or 4)
                     Determines the topology complexity
                     
        The initialization process:
        1. Sets the basic parameters and expands them to arrays if needed
        2. Calculates derived metrics (switches, terminals, bisection bandwidth, radix)
        3. Counts total network links
        4. Creates the network graph
        5. Calculates 3D visualization positions for all nodes
        """
        # Store the basic network parameters
        self.L = L  # Number of dimensions
        self.S = [S] * self.L  # Switches per dimension (expanded to array)
        self.K = [K] * self.L  # Link bandwidth per dimension (expanded to array)
        self.T = T  # Terminals per switch
        
        # Calculate total number of switches (product of switches across all dimensions)
        self.P = np.prod(self.S)
        
        # Calculate total number of terminals in the network
        self.N = self.T * self.P
        
        # Calculate minimum relative bisection bandwidth
        # This represents the network's ability to handle bisection traffic patterns
        self.beta = min([k*s/(2*T) for k, s in zip(self.K, self.S)]) if T > 0 else float('inf')
        
        # Calculate switch radix (total ports per switch)
        # Sum of terminal connections and inter-switch connections per dimension
        self.R = self.T + sum([k*(s-1) for k, s in zip(self.K, self.S)])
        
        # Calculate total links in the network
        self.total_links = self._calculate_total_links()
        
        # Create the network graph structure using NetworkX
        self.G = self._create_graph()
        
        # Generate 3D positions for visualization
        self.positions_3d = self._generate_3d_positions()
        
    def _calculate_total_links(self):
        """
        Calculate the total number of links in the network.
        
        This method computes both switch-to-terminal and switch-to-switch links.
        For switch-to-switch links, it accounts for the fully connected groups
        within each dimension.
        
        Returns:
            int: Total number of links in the network
        
        Algorithm:
        1. Calculate switch-to-terminal links (trivial: P * T)
        2. For each dimension:
           a. Calculate number of independent fully connected groups
           b. For each group, count the links in the complete subgraph
           c. Multiply by the trunking factor K
        3. Sum all links
        """
        # Switch-to-terminal links: each switch connects to T terminals
        term_links = self.P * self.T
        
        # Switch-to-switch links (counting each link only once)
        switch_links = 0
        for dim in range(self.L):
            # For each dimension, we have groups of fully connected switches
            # The number of such groups is the product of switches in other dimensions
            groups = np.prod([self.S[i] for i in range(self.L) if i != dim])
            
            # Each group forms a complete graph with S[dim] switches
            # In a complete graph with n nodes, there are n*(n-1)/2 edges
            links_per_group = self.S[dim] * (self.S[dim] - 1) // 2
            
            # Multiply by the trunking factor for this dimension
            switch_links += groups * links_per_group * self.K[dim]
        
        # Return total links
        return term_links + switch_links
    
    def _create_graph(self):
        """
        Create the NetworkX graph representing the HyperX topology.
        
        This method:
        1. Creates switch nodes at each coordinate position
        2. Adds terminal nodes connected to each switch
        3. Adds switch-to-switch connections by dimension
        
        Returns:
            NetworkX.Graph: Complete graph representation of the HyperX network
            
        The key insight is organizing switches by dimension groups - switches that share
        the same coordinates in all dimensions except one form a fully connected subgraph.
        """
        # Initialize an undirected graph
        G = nx.Graph()
        
        # Generate all possible coordinate combinations using Cartesian product
        # This creates all switch positions in the multidimensional space
        all_indices = list(product(*[range(s) for s in self.S]))
        
        # Add switch nodes and their terminals
        for idx in all_indices:
            # Create switch node ID using coordinates (e.g., "s-0_1_2_3")
            node_id = f"s-{'_'.join(map(str, idx))}"
            # Add the switch node with its type and coordinates as attributes
            G.add_node(node_id, type='switch', coords=idx)
            
            # Add terminal nodes for each switch (if T > 0)
            for t in range(self.T):
                # Create terminal ID (e.g., "t-0_1_2_3-0")
                terminal_id = f"t-{'_'.join(map(str, idx))}-{t}"
                # Add the terminal node with attributes
                G.add_node(terminal_id, type='terminal', coords=idx, terminal_idx=t)
                # Connect terminal to its parent switch
                G.add_edge(node_id, terminal_id, type='switch-terminal')
        
        # Add switch-to-switch connections organized by dimension
        for dim in range(self.L):
            # Group switches by identical coordinates in all dimensions except 'dim'
            # These switches will form a fully connected subgraph
            dimension_groups = {}
            
            for idx in all_indices:
                # Create a key from all coordinates except the current dimension
                # This effectively groups switches that only differ in one dimension
                key = tuple(c for i, c in enumerate(idx) if i != dim)
                
                # Initialize group if needed
                if key not in dimension_groups:
                    dimension_groups[key] = []
                
                # Add this switch to its group
                node_id = f"s-{'_'.join(map(str, idx))}"
                dimension_groups[key].append(node_id)
            
            # For each group, create a fully connected subgraph (complete graph)
            for group in dimension_groups.values():
                # Connect each node to all other nodes in the group
                for i, node1 in enumerate(group):
                    for node2 in group[i+1:]:
                        # Add edge with dimension and bandwidth attributes
                        G.add_edge(node1, node2, type='switch-switch', 
                                  dimension=dim, bandwidth=self.K[dim])
        
        return G
    
    def _generate_3d_positions(self):
        """
        Generate 3D positions for all nodes for visualization purposes.
        
        This method maps the multidimensional coordinates to 3D space:
        - For 3D HyperX: Direct mapping of coordinates to x, y, z
        - For 4D HyperX: The 4th dimension is visualized using spiral offsets
        
        Returns:
            dict: Mapping of node IDs to their 3D positions (x, y, z)
            
        This is a critical visualization component that determines how the
        network will be displayed. For 4D networks, it employs a clever
        spiral projection technique to make the 4th dimension visible.
        """
        positions = {}
        
        # Spacing factor for better visual separation between nodes
        spacing = 2
        
        # Process each node in the graph
        for node in self.G.nodes():
            # Handle switch nodes
            if self.G.nodes[node]['type'] == 'switch':
                coords = self.G.nodes[node]['coords']
                
                # Basic position: First 3 dimensions map directly to x, y, z
                x = coords[0] * spacing
                y = coords[1] * spacing
                z = coords[2] * spacing if self.L > 2 else 0  # If only 2D, use z=0
                
                # For 4D topology, create visualization of the 4th dimension
                if self.L > 3:
                    # Extract 4th dimension coordinate
                    w = coords[3]
                    
                    # Create a spiral offset pattern based on the 4th coordinate
                    # This visually separates nodes with different 4D coordinates
                    theta = w * np.pi / (2 * self.S[3])  # Angular offset
                    
                    # Apply offsets in all three dimensions to create a spiral effect
                    x_offset = 0.4 * spacing * w * np.cos(theta)
                    y_offset = 0.4 * spacing * w * np.sin(theta)
                    z_offset = 0.2 * spacing * w
                    
                    # Final position includes the 4D-based offset
                    positions[node] = (x + x_offset, y + y_offset, z + z_offset)
                else:
                    # For 3D topology, just use the direct mapping
                    positions[node] = (x, y, z)
                
            # Handle terminal nodes
            elif self.G.nodes[node]['type'] == 'terminal':
                # Get terminal coordinates and index
                coords = self.G.nodes[node]['coords']
                t_idx = self.G.nodes[node]['terminal_idx']
                
                # Find parent switch position
                parent_id = f"s-{'_'.join(map(str, coords))}"
                if parent_id in positions:
                    # Get parent switch coordinates
                    px, py, pz = positions[parent_id]
                    
                    # Position terminal in a circular pattern around its parent switch
                    # This creates a visually pleasing arrangement of terminals
                    angle = 2 * np.pi * t_idx / max(1, self.T)  # Distribute evenly in a circle
                    radius = 0.3  # Radius of the circle
                    
                    # Calculate terminal position
                    positions[node] = (
                        px + radius * np.cos(angle),  # X offset
                        py + radius * np.sin(angle),  # Y offset
                        pz - 0.2  # Slightly below the switch in Z
                    )
        
        return positions
    
    def _hide_axes(self, ax):
        """
        Hide all axes elements for cleaner visualization.
        
        This utility method removes all axis-related visual elements from a
        matplotlib 3D axes object, creating a clean, minimalist view that
        focuses entirely on the network structure.
        
        Args:
            ax: The matplotlib 3D axes object to modify
        """
        # Hide panes (the "walls" of the 3D box)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        
        # Hide pane edges
        ax.xaxis.pane.set_edgecolor('none')
        ax.yaxis.pane.set_edgecolor('none')
        ax.zaxis.pane.set_edgecolor('none')
        
        # Hide grid
        ax.grid(False)
        
        # Hide axes ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        
        # Set background to white
        ax.set_facecolor('white')
        
        # Remove axis labels
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_zlabel('')
    
    def _calculate_curve_control_point(self, x1, y1, z1, x2, y2, z2, dim, dist):
        """
        Calculate control point for creating aesthetically pleasing curved connections.
        
        This method is a key visual enhancement that makes the network visualization
        more interpretable by:
        1. Creating distinct curve patterns for each dimension
        2. Ensuring curves don't extend beyond reasonable bounds
        3. Making long-distance connections visually distinguishable
        
        Args:
            x1, y1, z1: Coordinates of the first point
            x2, y2, z2: Coordinates of the second point
            dim: Dimension index (0=X, 1=Y, 2=Z, 3=W)
            dist: Distance between points in their dimension
            
        Returns:
            tuple: (x, y, z) coordinates of the control point
            
        This method employs a dimension-specific approach to curve generation
        that creates visually distinct patterns for each dimension while keeping
        the curves contained within reasonable bounds.
        """
        # Calculate midpoint between the two endpoints
        midx = (x1 + x2) / 2
        midy = (y1 + y2) / 2
        midz = (z1 + z2) / 2
        
        # Calculate vector from point 1 to point 2
        vx = x2 - x1
        vy = y2 - y1
        vz = z2 - z1
        
        # Calculate vector length (straight-line distance)
        length = np.sqrt(vx**2 + vy**2 + vz**2)
        
        # Use a moderate scale factor to keep arcs within reasonable bounds
        base_scale = 0.2
        
        # Create dimension-specific curve patterns for visual distinction
        # Each dimension gets a unique curve style for easier identification
        if dim % 3 == 0:  # X-dimension connections: vertical arcs (up in Z)
            return (midx, midy, midz + base_scale * length)
            
        elif dim % 3 == 1:  # Y-dimension connections: vertical with X offset
            return (midx + 0.05 * length, midy, midz + base_scale * length)
            
        else:  # Z-dimension connections: arcs in X-Y plane
            # For Z-dimension, create arc in X-Y plane to avoid skewed appearance
            return (midx + base_scale * length, midy + base_scale * length, midz)
    
    def visualize(self, show_terminals=True, highlight_dim=None, w_slice=None,
                  alpha_links=0.3, rotate_view=False, show_axes=False):
        """
        Visualize the HyperX network in 3D with customizable display options.
        
        This comprehensive visualization method renders the network with color-coded
        dimensions, terminal nodes, and curved connections. It supports highlighting
        specific dimensions and showing slices of 4D networks.
        
        Args:
            show_terminals (bool): Whether to show terminal nodes
            highlight_dim (int): Dimension to highlight (0-L or None)
            w_slice (int): Show only nodes with specific 4th coordinate (4D only)
            alpha_links (float): Transparency of links (0.0-1.0)
            rotate_view (bool): Enable rotation animation
            show_axes (bool): Whether to show the coordinate axes
            
        Returns:
            Animation object if rotate_view=True, otherwise None
            
        This method handles both the structure generation and styling, making
        complex network structures visually interpretable through color coding,
        transparency control, and selective highlighting.
        """
        # Create the figure and 3D axes
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Hide axes if requested (for cleaner visualization)
        if not show_axes:
            self._hide_axes(ax)
        
        # Prepare collections for switch and terminal nodes
        switch_positions = []  # Coordinates
        switch_colors = []     # Color values
        switch_sizes = []      # Node sizes
        terminal_positions = [] # Terminal coordinates
        
        # Collect and filter nodes to visualize
        for node in self.G.nodes():
            # Get node type
            node_type = self.G.nodes[node]['type']
            
            # Skip if node not in position dictionary (shouldn't happen)
            if node not in self.positions_3d:
                continue
                
            # Filter by w_slice if specified (for 4D visualization)
            if w_slice is not None and self.L > 3:
                coords = self.G.nodes[node]['coords']
                # Skip nodes not in the requested 4D slice
                if coords[3] != w_slice:
                    continue
            
            # Process switch nodes
            if node_type == 'switch':
                # Add to switch collections
                switch_positions.append(self.positions_3d[node])
                
                # Color switches based on their highest dimension coordinate
                # This provides visual differentiation based on position
                coords = self.G.nodes[node]['coords']
                last_dim = min(self.L-1, 3)  # Use last dimension up to 3
                color_value = coords[last_dim] / max(1, self.S[last_dim] - 1)
                switch_colors.append(color_value)
                switch_sizes.append(100)  # Fixed size for switches
                
            # Process terminal nodes if enabled
            elif node_type == 'terminal' and show_terminals:
                terminal_positions.append(self.positions_3d[node])
        
        # Plot switch nodes with a colormap
        if switch_positions:
            xs, ys, zs = zip(*switch_positions)
            sc = ax.scatter(xs, ys, zs, c=switch_colors, s=switch_sizes, 
                           cmap='viridis', alpha=0.8, edgecolors='black')
        
        # Plot terminal nodes with green color
        if terminal_positions:
            t_xs, t_ys, t_zs = zip(*terminal_positions)
            ax.scatter(t_xs, t_ys, t_zs, c='green', s=30, alpha=0.8, edgecolors='black')
        
        # Draw all edges (connections between nodes)
        for u, v, data in self.G.edges(data=True):
            # Skip if either node is filtered out or not in positions
            if (u not in self.positions_3d or v not in self.positions_3d):
                continue
                
            # Skip if w_slice is specified and nodes don't match (4D only)
            if w_slice is not None and self.L > 3:
                u_coords = self.G.nodes[u]['coords']
                v_coords = self.G.nodes[v]['coords']
                if u_coords[3] != w_slice or v_coords[3] != w_slice:
                    continue
            
            # Skip terminal nodes if terminals are disabled
            if not show_terminals and (self.G.nodes[u]['type'] == 'terminal' or 
                                       self.G.nodes[v]['type'] == 'terminal'):
                continue
            
            # Get coordinates for the edge endpoints
            x1, y1, z1 = self.positions_3d[u]
            x2, y2, z2 = self.positions_3d[v]
            
            # Get edge type (switch-terminal or switch-switch)
            edge_type = data.get('type', '')
            
            # Handle switch-to-terminal connections (simple straight lines)
            if edge_type == 'switch-terminal':
                # Draw thin black lines for switch-terminal connections
                ax.plot([x1, x2], [y1, y2], [z1, z2], 'k-', linewidth=0.5, alpha=0.5)
            
            # Handle switch-to-switch connections (color-coded by dimension)
            elif edge_type == 'switch-switch':
                # Get the dimension this connection belongs to
                dim = data.get('dimension', 0)
                
                # Color coding by dimension - each dimension gets its own color
                colors = ['red', 'blue', 'green', 'purple'][:self.L]
                color = colors[dim]
                
                # Adjust opacity and width based on highlighting settings
                if highlight_dim is not None:
                    # If highlighting is enabled, emphasize the selected dimension
                    if dim == highlight_dim:
                        alpha = 0.8  # More opaque for highlighted dimension
                        linewidth = 2.0  # Thicker lines for highlighted dimension
                    else:
                        alpha = 0.1  # Very transparent for non-highlighted dimensions
                        linewidth = 0.5  # Thinner lines for non-highlighted dimensions
                else:
                    # Normal mode - use consistent transparency
                    alpha = alpha_links
                    # Slightly emphasize the last dimension
                    linewidth = 1.5 if dim == self.L-1 else 1.0
                
                # Get node coordinates for determining connection type
                u_coords = self.G.nodes[u]['coords']
                v_coords = self.G.nodes[v]['coords']
                
                # Calculate distance in this dimension
                # This helps determine if we need a curved or straight line
                if u_coords[dim] == v_coords[dim]:
                    # Nodes are in the same position in this dimension (shouldn't happen)
                    dist = 0
                else:
                    # Calculate direct distance (not considering wraparound)
                    dist = abs(u_coords[dim] - v_coords[dim])
                
                # Visualization technique: straight lines for adjacent nodes,
                # curved splines for distant connections
                if dist <= 1:
                    # Direct connection - simple straight line
                    ax.plot([x1, x2], [y1, y2], [z1, z2], '-', color=color, 
                          linewidth=linewidth, alpha=alpha)
                else:
                    # Distant connection - create a curved spline for better visualization
                    # This is a key visualization enhancement that improves readability
                    
                    # Calculate control point for a nice parabolic arc
                    cx, cy, cz = self._calculate_curve_control_point(
                        x1, y1, z1, x2, y2, z2, dim, dist
                    )
                    
                    # Create a curved path using the control point
                    curve_x = np.array([x1, cx, x2])
                    curve_y = np.array([y1, cy, y2])
                    curve_z = np.array([z1, cz, z2])
                    
                    # Create a smooth spline curve through the points
                    # We use k=2 (quadratic) spline because we only have 3 points
                    # and need to satisfy the m > k constraint
                    t = np.linspace(0, 1, 20)  # 20 interpolation points for smooth curve
                    
                    # Create spline representations for each dimension
                    x_spline = interpolate.splrep(np.array([0, 0.5, 1]), curve_x, k=2)
                    y_spline = interpolate.splrep(np.array([0, 0.5, 1]), curve_y, k=2)
                    z_spline = interpolate.splrep(np.array([0, 0.5, 1]), curve_z, k=2)
                    
                    # Evaluate the splines to get smooth curve points
                    x_points = interpolate.splev(t, x_spline)
                    y_points = interpolate.splev(t, y_spline)
                    z_points = interpolate.splev(t, z_spline)
                    
                    # Draw the curved connection
                    ax.plot(x_points, y_points, z_points, '-', color=color, 
                            linewidth=linewidth, alpha=alpha)
        
        # Create an informative title showing network parameters and metrics
        dimensions = "3D" if self.L == 3 else "4D"
        title = f"{dimensions} HyperX: S={self.S[0]}, K={self.K[0]}, T={self.T}\n"
        title += f"Switches={self.P}, Links={self.total_links}, R={self.R}"
        
        # Add information about visualization settings
        if w_slice is not None and self.L > 3:
            title += f", Showing 4D slice w={w_slice}"
        
        if highlight_dim is not None:
            dim_names = ['X', 'Y', 'Z', 'W'][:self.L]
            title += f", Highlighting {dim_names[highlight_dim]} dimension"
            
        ax.set_title(title)
        
        # Set equal aspect ratio for better 3D visualization
        ax.set_box_aspect([1,1,1])
        
        # Set initial view angle
        ax.view_init(elev=20, azim=30)
        
        # Add dimension color legend
        dim_colors = ['red', 'blue', 'green', 'purple'][:self.L]
        dim_names = ['X dimension', 'Y dimension', 'Z dimension', 'W dimension'][:self.L]
        legend_elements = [plt.Line2D([0], [0], color=c, lw=2, label=l) 
                          for c, l in zip(dim_colors, dim_names)]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Create rotation animation if requested
        if rotate_view:
            def rotate(angle):
                ax.view_init(elev=20, azim=angle)
                return []
                
            ani = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, 2), 
                                         interval=50, blit=True)
            plt.tight_layout()
            plt.show()
            return ani
        else:
            plt.tight_layout()
            plt.show()

    def visualize_interactive(self):
        """
        Create an interactive visualization with user controls for exploring the network.
        
        This method provides a rich interactive interface including:
        - Radio buttons for selecting dimensions to highlight
        - Checkbox for showing/hiding terminals
        - Checkbox for showing/hiding axes
        - Slider for 4D slice selection (4D networks only)
        - Slider for controlling link transparency
        
        These interactive controls allow users to explore complex networks
        from different perspectives and focus on specific structural elements.
        """
        # Import required widgets
        import matplotlib.pyplot as plt
        from matplotlib.widgets import RadioButtons, CheckButtons, Slider
        
        # Create a larger figure for the interactive visualization
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Add dimension selection radio buttons
        rax = plt.axes([0.02, 0.7, 0.12, 0.2], facecolor='lightgoldenrodyellow')
        radio_labels = ['All'] + [f'Dimension {i}' for i in range(self.L)]
        radio = RadioButtons(rax, radio_labels)
        
        # Add checkbox for showing/hiding terminals
        cax = plt.axes([0.02, 0.5, 0.12, 0.1], facecolor='lightgoldenrodyellow')
        check = CheckButtons(cax, ['Show Terminals'], [True])
        
        # Add checkbox for showing/hiding axes
        acax = plt.axes([0.02, 0.4, 0.12, 0.1], facecolor='lightgoldenrodyellow')
        axes_check = CheckButtons(acax, ['Show Axes'], [False])
        
        # Add slider for 4D slice selection (only for 4D)
        slice_slider = None
        if self.L > 3:
            sax = plt.axes([0.02, 0.3, 0.12, 0.1], facecolor='lightgoldenrodyellow')
            slice_slider = Slider(sax, '4D Slice', -1, self.S[3]-1, valinit=-1, valstep=1)
        
        # Add slider for link transparency
        tax = plt.axes([0.02, 0.2, 0.12, 0.05], facecolor='lightgoldenrodyellow')
        transparency_slider = Slider(tax, 'Transparency', 0.1, 1.0, valinit=0.3, valstep=0.1)
        
        # Define update function that redraws the visualization based on control settings
        def update(val=None):
            ax.clear()
            
            # Get current control values
            dim_label = radio.value_selected
            highlight_dim = None if dim_label == 'All' else int(dim_label.split()[-1])
            
            show_terminals = check.get_status()[0]
            show_axes = axes_check.get_status()[0]
            
            # For 4D, get w_slice value
            w_slice = None
            if self.L > 3 and slice_slider is not None:
                w_slice = None if slice_slider.val == -1 else int(slice_slider.val)
                
            link_alpha = transparency_slider.val
            
            # Hide axes if requested
            if not show_axes:
                self._hide_axes(ax)
            
            # Prepare separate collections for switch and terminal nodes
            switch_positions = []
            switch_colors = []
            switch_sizes = []
            
            terminal_positions = []
            
            # Collect nodes to visualize (same logic as in visualize method)
            for node in self.G.nodes():
                # Filter by node type
                node_type = self.G.nodes[node]['type']
                
                # Skip if not in the positions dictionary
                if node not in self.positions_3d:
                    continue
                    
                # Filter by w_slice if specified (4D only)
                if w_slice is not None and self.L > 3:
                    coords = self.G.nodes[node]['coords']
                    if coords[3] != w_slice:
                        continue
                
                # Handle switches
                if node_type == 'switch':
                    switch_positions.append(self.positions_3d[node])
                    
                    # Color based on last coordinate
                    coords = self.G.nodes[node]['coords']
                    last_dim = min(self.L-1, 3)  # Last dimension up to 3
                    color_value = coords[last_dim] / max(1, self.S[last_dim] - 1)
                    switch_colors.append(color_value)
                    switch_sizes.append(100)
                    
                # Handle terminals if enabled
                elif node_type == 'terminal' and show_terminals:
                    terminal_positions.append(self.positions_3d[node])
            
            # Plot switches with a colormap
            if switch_positions:
                xs, ys, zs = zip(*switch_positions)
                sc = ax.scatter(xs, ys, zs, c=switch_colors, s=switch_sizes, 
                               cmap='viridis', alpha=0.8, edgecolors='black')
            
            # Plot terminals separately (green)
            if terminal_positions:
                t_xs, t_ys, t_zs = zip(*terminal_positions)
                ax.scatter(t_xs, t_ys, t_zs, c='green', s=30, alpha=0.8, edgecolors='black')
            
            # Draw edges (same logic as in visualize method)
            for u, v, data in self.G.edges(data=True):
                # Skip if either node is filtered out or not in positions
                if (u not in self.positions_3d or v not in self.positions_3d):
                    continue
                    
                # Skip if w_slice is specified and nodes don't match (4D only)
                if w_slice is not None and self.L > 3:
                    u_coords = self.G.nodes[u]['coords']
                    v_coords = self.G.nodes[v]['coords']
                    if u_coords[3] != w_slice or v_coords[3] != w_slice:
                        continue
                
                # Skip terminal nodes if terminals are disabled
                if not show_terminals and (self.G.nodes[u]['type'] == 'terminal' or 
                                           self.G.nodes[v]['type'] == 'terminal'):
                    continue
                
                x1, y1, z1 = self.positions_3d[u]
                x2, y2, z2 = self.positions_3d[v]
                
                edge_type = data.get('type', '')
                
                if edge_type == 'switch-terminal':
                    # Thin black lines for switch-terminal connections
                    ax.plot([x1, x2], [y1, y2], [z1, z2], 'k-', linewidth=0.5, alpha=0.5)
                
                elif edge_type == 'switch-switch':
                    dim = data.get('dimension', 0)
                    
                    # Set color and width based on dimension
                    # Use a color palette suitable for dimension count
                    colors = ['red', 'blue', 'green', 'purple'][:self.L]
                    color = colors[dim]
                    
                    # Highlight specific dimension if requested
                    if highlight_dim is not None:
                        if dim == highlight_dim:
                            alpha = 0.8
                            linewidth = 2.0
                        else:
                            alpha = 0.1
                            linewidth = 0.5
                    else:
                        alpha = link_alpha
                        linewidth = 1.5 if dim == self.L-1 else 1.0  # Emphasize last dimension
                    
                    # Determine if this is a wraparound connection by checking coordinates
                    u_coords = self.G.nodes[u]['coords']
                    v_coords = self.G.nodes[v]['coords']
                    
                    # Calculate distance in this dimension
                    if u_coords[dim] == v_coords[dim]:
                        # Nodes are in the same position in this dimension (shouldn't happen)
                        dist = 0
                    else:
                        # Get distance (direct distance, not considering wraparound)
                        dist = abs(u_coords[dim] - v_coords[dim])
                    
                    # If nodes are adjacent (dist=1), draw a straight line
                    # Otherwise, draw a curved spline for better visualization
                    if dist <= 1:
                        # Direct connection - straight line
                        ax.plot([x1, x2], [y1, y2], [z1, z2], '-', color=color, 
                              linewidth=linewidth, alpha=alpha)
                    else:
                        # This is a wraparound or distant connection - draw a curved spline
                        # Calculate a better control point for more parabolic arcs
                        cx, cy, cz = self._calculate_curve_control_point(
                            x1, y1, z1, x2, y2, z2, dim, dist
                        )
                        
                        # Create a curved path using the control point
                        curve_x = np.array([x1, cx, x2])
                        curve_y = np.array([y1, cy, y2])
                        curve_z = np.array([z1, cz, z2])
                        
                        # Create a smoother curve 
                        # Use k=2 (quadratic) spline instead of default k=3 (cubic)
                        # because we only have 3 points and need m > k
                        t = np.linspace(0, 1, 20)
                        x_spline = interpolate.splrep(np.array([0, 0.5, 1]), curve_x, k=2)
                        y_spline = interpolate.splrep(np.array([0, 0.5, 1]), curve_y, k=2)
                        z_spline = interpolate.splrep(np.array([0, 0.5, 1]), curve_z, k=2)
                        
                        x_points = interpolate.splev(t, x_spline)
                        y_points = interpolate.splev(t, y_spline)
                        z_points = interpolate.splev(t, z_spline)
                        
                        # Draw the curved path
                        ax.plot(x_points, y_points, z_points, '-', color=color, 
                                linewidth=linewidth, alpha=alpha)
            
            # Set axis labels if axes are shown
            if show_axes:
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
            
            # Create descriptive title with total links information
            dimensions = "3D" if self.L == 3 else "4D"
            title = f"{dimensions} HyperX: S={self.S[0]}, K={self.K[0]}, T={self.T}\n"
            title += f"Switches={self.P}, Links={self.total_links}, R={self.R}"
            
            if w_slice is not None and self.L > 3:
                title += f", Showing 4D slice w={w_slice}"
            
            if highlight_dim is not None:
                dim_names = ['X', 'Y', 'Z', 'W'][:self.L]
                title += f", Highlighting {dim_names[highlight_dim]} dimension"
                
            ax.set_title(title)
            
            # Set equal aspect ratio
            ax.set_box_aspect([1,1,1])
            
            # Add dimension legend
            dim_colors = ['red', 'blue', 'green', 'purple'][:self.L]
            dim_names = ['X dimension', 'Y dimension', 'Z dimension', 'W dimension'][:self.L]
            legend_elements = [plt.Line2D([0], [0], color=c, lw=2, label=l) 
                              for c, l in zip(dim_colors, dim_names)]
            ax.legend(handles=legend_elements, loc='upper right')
            
            # Refresh the plot
            fig.canvas.draw_idle()
        
        # Connect the callback functions to the widgets
        radio.on_clicked(lambda label: update())
        check.on_clicked(lambda label: update())
        axes_check.on_clicked(lambda label: update())
        if slice_slider:
            slice_slider.on_changed(update)
        transparency_slider.on_changed(update)
        
        # Initial plot
        update()
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(left=0.15)  # Make room for the control panel
        plt.show()
    
    def print_network_info(self):
        """
        Print detailed information about the HyperX network configuration and metrics.
        
        This method outputs a comprehensive summary of the network, including:
        - Basic parameters (dimensions, switches, bandwidth, terminals)
        - Derived metrics (total switches, terminals, links, radix)
        - Connectivity statistics by dimension
        - Network diameter
        - Visualization tips
        
        This is useful for verifying network properties and understanding
        the topology's scale and characteristics.
        """
        # Determine network dimensionality for display
        dimensions = "3D" if self.L == 3 else "4D"
        
        # Print basic network parameters
        print(f"\n=== {dimensions} HyperX Network Information ===")
        print(f"Dimensions: {self.L}")
        print(f"Switches per dimension (S): {self.S[0]}")
        print(f"Link bandwidth (K): {self.K[0]}")
        print(f"Terminals per switch (T): {self.T}")
        
        # Print derived network metrics
        print(f"\nNetwork Size:")
        print(f"  Total switches: {self.P}")
        print(f"  Total terminals: {self.N}")
        print(f"  Total links: {self.total_links}")
        print(f"  Switch radix (R): {self.R}")
        
        # Calculate and print connectivity statistics by dimension
        dim_connect = [(self.S[i]-1) for i in range(self.L)]
        total_connect = sum(dim_connect)
        
        print(f"\nConnectivity per switch:")
        dim_names = ['X', 'Y', 'Z', 'W']
        for i in range(self.L):
            print(f"  {dim_names[i]} dimension: {dim_connect[i]} connections")
        print(f"  Total interswitch connections: {total_connect}")
        
        # Print network diameter (based on dimension-ordered routing)
        diameter = self.L
        print(f"\nNetwork diameter: {diameter} hops")
        
        # Print visualization tips
        print(f"\n{dimensions} Visualization Tips:")
        print("  - Use interactive mode to explore different dimensions")
        if self.L > 3:
            print("  - View 3D slices of the 4D structure using the slider")
            print("  - The 4th dimension (W) is represented by color and spatial offset")
        print("\n=================================")


if __name__ == "__main__":
    # Create a 3D HyperX
    hyperx_3d = HyperX(S=4, K=1, T=1, L=3)
    
    # Print network information
    hyperx_3d.print_network_info()
    
    # Standard visualization with no axes (default)
    hyperx_3d.visualize(show_terminals=True, show_axes=False)
    
    # Highlight just one dimension
    hyperx_3d.visualize(highlight_dim=2, show_axes=False)  # Highlight the Z dimension
    
    # Interactive visualization 
    hyperx_3d.visualize_interactive()
    
    # Create a 4D HyperX with 3 switches per dimension
    hyperx_4d = HyperX(S=3, K=1, T=1, L=4)
    
    # Print network information
    hyperx_4d.print_network_info()
    
    # Standard visualization with no axes (default)
    hyperx_4d.visualize(show_terminals=True, show_axes=False)
    
    # Highlight just one dimension
    hyperx_4d.visualize(highlight_dim=3, show_axes=False)  # Highlight the 4th dimension
    
    # Show just one 4D slice
    hyperx_4d.visualize(w_slice=1, show_axes=False)  # Show only nodes with w=1
    
    # Interactive visualization (best way to explore)
    hyperx_4d.visualize_interactive()
